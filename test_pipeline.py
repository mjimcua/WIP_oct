"""test_pipeline.py — system-level checks of SFF v3 on the synthetic dataset.

    python test_pipeline.py   →  exit 0 if healthy, 1 with the failures

No reference database, no numeric regression: the synthetic raw was DESIGNED with one
scenario per feature (see synthetic_v3.py), so every scenario has an expected
behaviour that can be asserted as logic. When the code changes and a check fails, the
question is "is the new behaviour right?", not "does it match the old number?".

Blocks:
  A  run_analysis end to end on taxonomy 1 (mandatory = [region]) · 32 tables written ·
     every scenario lands where it should · hold-out 2026 · extended 2027 · validation PASS
  B  run_pipeline (decisions read from SQL) reproduces the analysis' forecast
  C  taxonomy 2 (mandatory = [region, product]) changes the ladder, not the totals' order of magnitude
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import PHYSICAL_TABLE_NAMES, Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from pipeline import run_analysis, run_pipeline
from synthetic_v3 import build_raw
from test_fixtures import quiet

RECORDER = CheckRecorder()
TIMEVARYING = {"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"}


class SyntheticConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        return build_raw(7)


def synthetic_config(folder: str, mandatory: list, extras: list) -> Config:
    engine = create_engine(f"sqlite:///{os.path.join(folder, 'v3.db')}")
    return SyntheticConfig(sql_engine=engine, sql_schema=None, outdir=folder, business_mandatory_dims=mandatory,
                           structural_timevarying_dims=TIMEVARYING, extra_renovacion=extras,
                           extra_revalorizacion=["discount", "newcust"], backtest_test_start="2026-01",
                           extended_horizon_end="2027-12", uplift_parent_keep_columns=["newcust"],
                           benchmark_group_dims=["region", "product"], benchmark_min_support=100,
                           signed_ladder_max_loss=0.0, challenger_technique="T2_mean",
                           backtest_max_targets=24, backtest_screen_horizons=[1, 2, 3, 6, 12], backtest_horizon_cap=12,
                           backtest_horizons=[1, 2, 3, 4, 6, 9, 12],
                           backtest_horizon_bands={"h1": [1, 1], "corto": [2, 3], "medio": [4, 6], "largo": [7, 12]},
                           challenger_margin_by_band={"h1": 0.10, "corto": 0.10, "medio": 0.05, "largo": 0.0})


def test_analysis_taxonomy_1() -> dict:
    RECORDER.start_block("A · run_analysis · taxonomy 1 (mandatory = region)")
    folder = tempfile.mkdtemp()
    configuration = synthetic_config(folder, ["region"], ["product", "channel"])
    with quiet():
        results = run_analysis(configuration)
    written = set(inspect(configuration.engine).get_table_names())
    expected = {configuration._resolve_physical_table_name(name) for name in PHYSICAL_TABLE_NAMES}
    RECORDER.check(expected <= written, f"all {len(PHYSICAL_TABLE_NAMES)} registered tables are written ({len(expected - written)} missing)")
    card = results["series_card"].set_index("fs_id")
    # scenarios
    RECORDER.check(card.loc["EU|0|0|0|0|A|tele", "peldano"] == 2 and card.loc["EU|0|0|0|0|A|tele", "n_efectivo"] > 600,
                   "mute channel: EU|A tele (n=12) annuls channel and pools with EU|A web")
    negatives = card.loc[["EU|1|0|0|0|A|web", "EU|0|1|0|0|A|web", "EU|0|0|1|0|A|web"]]
    RECORDER.check((negatives["id_estimacion"] == "EU|SIG=negativo|A|web").all() and (negatives["n_efectivo"] > 30).all(),
                   "sign grouping: the three small negatives (15/12/10) pool as 'EU|SIG=negativo|A|web' ≥ 30")
    RECORDER.check(card.loc["EU|0|0|0|1|A|web", "nivel_riesgo"] == "S_signo_bajo_suelo", "the minority positive stays in its sign under the floor (S)")
    RECORDER.check(card.loc["EU|1|0|0|1|B|web", "nivel_riesgo"] == "M_signo_mixto", "the mixed-sign series is alone (M)")
    RECORDER.check(card.loc["NA|0|0|0|0|A|tele", "peldano"] == 2 and card.loc["NA|0|0|0|0|A|tele", "id_estimacion"] == "NA|SIG=neutro|A|*",
                   "a neutral small series borrows from its cell's neutrals")
    RECORDER.check(card.loc["NA|0|0|0|0|B|tele", "huecos"] in (10, 11), "gaps: NA|B tele has ~11 synthetic months with undefined rate (2026-08 is pending, not history)")
    RECORDER.check(card.loc["NA|0|0|0|0|A|kiosk", "nivel_riesgo"] == "N_sin_impacto" and card.loc["EU|0|0|0|0|B|tienda", "nivel_riesgo"] == "D_sin_historia"
                   and card.loc["EU|0|0|0|0|A|kiosk", "nivel_riesgo"] == "T_universo_ts",
                   "routes and universes: no_impact → N, projection-only → D, time_series → T")
    benchmark = results["decisions"]["decision_estacionalidad"].set_index("fs_id")
    RECORDER.check(benchmark.loc["EU|0|0|0|0|A|web", "veredicto_estacional"] == 1 and benchmark.loc["EU|0|0|0|0|A|web", "phi"] > 3,
                   "the benchmark declares EU|A seasonal (φ > 3, shape beats the level)")
    reference = results["decisions"]["pool_reference"].set_index("id_estimacion")
    RECORDER.check(reference.loc["EU|0|0|0|0|A|web", "estacional"] == 1, "…and the pool reference carries it")
    technique = results["decisions"]["decision_technique"]
    technique = technique[technique["tramo_h"] == "h1"].set_index("id_estimacion")
    RECORDER.check(technique.loc["EU|0|0|0|0|A|web", "tecnica_origen"] == "campeon" and technique.loc["NA|0|0|0|0|A|web", "tecnica"] == "T2_mean",
                   "a champion on the seasonal series; the challenger on a flat one")
    decomposition = results["composition"]["mix_shift_decomposition"]
    na = decomposition[decomposition["celda_id"] == "NA"]
    RECORDER.check(na["delta_composicion_pp"].abs().mean() > 0.05, "the NA cell (product weights shifting) shows a composition term in the Kitagawa decomposition")
    holdout = results["backtest"]["backtest_holdout"]
    RECORDER.check(holdout["mes_objetivo"].min() == "2026-01" and holdout["mes_objetivo"].max() == "2026-07",
                   "hold-out = the closed months from backtest_test_start (2026-01) to the last closed one (2026-07; 2026-08 is pending)")
    h1 = holdout[holdout["h"] == 1]
    RECORDER.check(h1["err_pp"].abs().mean() < 8 and 0.8 <= h1["dentro_banda"].mean() <= 1.0,
                   f"hold-out h=1: mean |error| {h1['err_pp'].abs().mean():.1f} pp, {h1['dentro_banda'].mean():.0%} inside the band")
    horizon = results["forecast"]["horizon_report"]
    RECORDER.check(horizon["period"].min() == "2026-08" and horizon["period"].max() == "2027-12" and len(horizon) == 17,
                   "the forecast covers the pending month (2026-08), the current month and 2027 (17 months)")
    RECORDER.check((horizon.loc[horizon["period"] >= "2027-01", "pct_simulado"] == 100).all(), "2027 is built entirely on simulated pipeline")
    # the sheet resolves any key to a series and returns its tables and summary
    from sheet import sheet
    from config import hash_key
    with quiet():
        by_id = sheet("EU|0|0|0|0|A|tele", results=results, figure=False)
        by_estimation = sheet(hash_key("EU|SIG=negativo|A|web"), results=results, figure=False)
    RECORDER.check(by_id["keys"]["kind"] == "fs_key" and "SERIES EU|0|0|0|0|A|tele" in by_id["summary"] and len(by_id["tables"]["parent_ladder"]) >= 3,
                   "sheet(fs_id): the series' tables and summary")
    RECORDER.check(by_estimation["keys"]["kind"] == "estimacion_key" and len(by_estimation["keys"]["members"]) == 4,
                   "sheet(estimacion_key): resolves the pool and lists its 4 member series")
    movers = results["top_movers"]
    RECORDER.check(set(movers["tipo"]) >= {"deterioro", "mejora", "senal_negativa", "banda_ancha", "precio"} and (movers.groupby("tipo")["rank"].min() == 1).all(),
                   "top_movers has every kind, ranked by money inside each kind")
    negative = movers[movers["tipo"] == "senal_negativa"]
    RECORDER.check((negative["delta_pp"] > 0).all(), "negative-signal series renew below the neutral rate of their cell (recoverable money is positive)")
    validation = results["validation"]
    RECORDER.check(not ((validation["estado"] == "FAIL") & validation["familia"].isin(["INTEGRITY", "DOCTRINE"])).any(),
                   "the validation panel has no INTEGRITY / DOCTRINE failure")
    return dict(folder=folder, configuration=configuration, results=results)


def test_monthly_run(context: dict) -> None:
    RECORDER.start_block("B · run_pipeline reads the decisions and reproduces the forecast")
    with quiet():
        monthly = run_pipeline(context["configuration"])
    analysis_total = context["results"]["forecast"]["forecast_detail"]["esperado_usd"].sum()
    monthly_total = monthly["forecast"]["forecast_detail"]["esperado_usd"].sum()
    RECORDER.check(abs(analysis_total - monthly_total) < 0.01, f"same total: ${analysis_total:,.0f} (analysis) vs ${monthly_total:,.0f} (monthly run)")
    left = context["results"]["forecast"]["forecast_detail"].set_index("fu_comb_key")["tasa"]
    right = monthly["forecast"]["forecast_detail"].set_index("fu_comb_key")["tasa"]
    RECORDER.check(np.allclose(left.sort_index(), right.reindex(left.sort_index().index)), "same rate on every row")
    # the monthly run refuses to start without decisions
    empty = synthetic_config(tempfile.mkdtemp(), ["region"], ["product", "channel"])
    refused = False
    try:
        with quiet():
            run_pipeline(empty)
    except RuntimeError as error:
        refused = "missing" in str(error)
    RECORDER.check(refused, "without decision tables the monthly run refuses to run and says which is missing")


def test_analysis_taxonomy_2() -> None:
    RECORDER.start_block("C · taxonomy 2 (mandatory = region, product)")
    folder = tempfile.mkdtemp()
    configuration = synthetic_config(folder, ["region", "product"], ["channel"])
    with quiet():
        results = run_analysis(configuration)
    card = results["series_card"].set_index("fs_id")
    RECORDER.check(card.loc["EU|A|0|0|0|0|tele", "id_estimacion"] == "EU|A|SIG=neutro|*", "with product mandatory the tele series pools inside the EU|A cell")
    RECORDER.check(card.loc["EU|A|0|1|0|0|web", "id_estimacion"] == "EU|A|SIG=negativo|web", "sign grouping still happens inside the finer cell")
    total_1 = 548_000
    total_2 = results["forecast"]["forecast_detail"]["esperado_usd"].sum()
    RECORDER.check(0.8 * total_1 < total_2 < 1.2 * total_1, f"the total stays in the same range (${total_2:,.0f})")
    RECORDER.check(not ((results["validation"]["estado"] == "FAIL")).any(), "validation passes on the second taxonomy")


def test_signal_maturation() -> None:
    """A raw whose future months carry FEWER signals than its closed months: the maturation
    step must price the pending migration downwards and raise the level alert where the
    photo is already above the final share."""
    RECORDER.start_block("D · signal maturation (future photo below the final composition)")
    base = build_raw(7)
    future = base["period"] >= "2026-09"
    immature = base.copy()
    immature.loc[future & (immature["region"] == "EU"), "softcancel"] = 0          # EU: nobody has announced yet
    immature.loc[future & (immature["region"] == "NA") & (immature["product"] == "A"), "dormant"] = 1   # NA·A: already everyone dormant
    measures = [c for c in immature.columns if c.startswith("total_")]
    keys = [c for c in immature.columns if c not in measures]
    immature = immature.groupby(keys, as_index=False, dropna=False)[measures].sum()     # rows that became identical merge

    class Immature(Config):
        def read_raw(self):
            return immature
    with tempfile.TemporaryDirectory() as folder:
        configuration = Immature(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"],
                                 structural_timevarying_dims=TIMEVARYING, extra_renovacion=["product", "channel"],
                                 extra_revalorizacion=["discount", "newcust"], extended_horizon_end="2027-12",
                                 benchmark_group_dims=["region", "product"], benchmark_min_support=100, sheets_top_series=0,
                                 console_explanations=False)
        with quiet():
            results = run_analysis(configuration)
    adjustment = results["signal_maturation"]["signal_adjustment"]
    eu_soft = adjustment[(adjustment["celda_comp"].str.startswith("EU")) & (adjustment["senal"] == "softcancel") & (adjustment["h"] > 1)
                         & (adjustment["share_final_esperado"] > 0)]                       # cells where softcancel existed in the closed months
    RECORDER.check(len(eu_soft) > 0 and (eu_soft["pendiente"] > 0).all() and (eu_soft["ajuste_usd"].fillna(0) <= 0).all(),
                   "EU cells that used to have softcancel, now with none announced: pending maturation > 0 and a negative adjustment beyond h=1")
    RECORDER.check((adjustment[adjustment["h"] <= 1]["pendiente"] == 0).all(), "at h ≤ 1 nothing is pending (what was going to appear has appeared)")
    alerts = results["signal_maturation"]["signal_alerts"]
    na_dormant = alerts[(alerts["celda_comp"].str.startswith("NA")) & (alerts["senal"] == "dormant") & (alerts["tipo"] == "nivel_anomalo")]
    RECORDER.check(len(na_dormant) > 0, "NA·A already fully dormant: the level alert fires (share today far above the final share)")
    answers = results["forecast"]["business_summary"]
    RECORDER.check((answers["forecast_ajustado_usd"] <= answers["forecast_usd"] + 1e-6).all() and (answers["ajuste_senales_usd"] < 0).any(),
                   "business_summary carries forecast_ajustado_usd below the photo forecast")


def test_contract_uplift() -> None:
    """The synthetic raw carries `discount_pct` (0.0 / 0.4, unknown in the kiosk channel).
    With the contract path on: rows with a known discount get uplift = increase / (1 − d) ×
    realization ratio, rows with an unknown discount keep the statistical uplift, projected
    re-entries renew at list price (discount 0), and a declared price increase lifts the
    contract uplift from its month on."""
    RECORDER.start_block("E · the contract path of the uplift")
    with tempfile.TemporaryDirectory() as folder:
        class Synthetic(Config):
            def read_raw(self):
                return build_raw(7, with_discount_pct=True)
        configuration = Synthetic(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"],
                                  structural_timevarying_dims=TIMEVARYING, extra_renovacion=["product", "channel"],
                                  extra_revalorizacion=["discount", "newcust"], extended_horizon_end="2027-12",
                                  benchmark_group_dims=["region", "product"], benchmark_min_support=100, sheets_top_series=0,
                                  console_explanations=False, discount_value_column="discount_pct",
                                  price_increase_by_period={"2027-06": 1.10}, contract_apply_realization_ratio=True)
        with quiet():
            results = run_analysis(configuration)
    detail = results["forecast"]["forecast_detail"]
    decision = results["decisions"]["decision_uplift"]
    RECORDER.check(set(detail["uplift_via"]) == {"contrato", "estadistico"}, "both paths are used: contract where the discount is known, statistical where it is not")
    kiosk = detail[detail["fs_id"].str.contains("kiosk") & (detail["origen_pipeline"] == "real")]
    RECORDER.check((kiosk["uplift_via"] == "estadistico").all(), "real rows with an unknown discount (kiosk) stay on the statistical path (their projected re-entries renew at list price and become contract)")
    contract = detail[(detail["uplift_via"] == "contrato") & (detail["origen_pipeline"] == "real")]
    ratio_of = dict(zip(decision["uplift_cell_id"], decision["ratio_realizacion"]))
    fine = results["fine_table"]
    sample = contract.merge(fine[["fu_comb_key", "discount_pct"]], on="fu_comb_key", how="left").dropna(subset=["discount_pct"]).head(200)
    expected = np.where(sample["period"].astype(str) >= "2027-06", 1.10, 1.0) / (1 - sample["discount_pct"]) * sample["uplift_cell_id"].map(ratio_of).fillna(1.0)
    RECORDER.check(np.allclose(sample["uplift"], np.minimum(expected, 3.0), atol=1e-6), "contract uplift = increase(period) / (1 − discount) × realization ratio of the cell, capped")
    projected = detail[(detail["origen_pipeline"] == "proyectada") & (detail["uplift_via"] == "contrato")]
    RECORDER.check(len(projected) > 0 and (projected["uplift"] <= 1.10 * projected["uplift_cell_id"].map(ratio_of).fillna(1.0).max() + 1e-6).all(),
                   "projected re-entries renew at list price: their contract uplift is only the price increase (× ratio), never 1/(1 − d) again")
    check = results["decisions"].get("decision_uplift")
    RECORDER.check("ratio_realizacion" in decision.columns and (decision["ratio_realizacion"] > 0).all(), "the realization ratio is persisted per cell in decision_uplift")


def main() -> int:
    print("═" * 74 + "\nTEST pipeline · SFF v3 on the synthetic dataset\n" + "═" * 74)
    context = test_analysis_taxonomy_1()
    test_monthly_run(context)
    test_analysis_taxonomy_2()
    test_signal_maturation()
    test_contract_uplift()
    return RECORDER.print_panel("PIPELINE TEST")


if __name__ == "__main__":
    sys.exit(main())
