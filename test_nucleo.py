"""test_nucleo.py — the core table tells the whole story and matches the raw.

    python test_nucleo.py   →  exit 0 if healthy, 1 with the failures
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
import nucleo
from pipeline import run_analysis
from synthetic_v3 import build_raw
from test_fixtures import quiet

RECORDER = CheckRecorder()
TIMEVARYING = {"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"}


def run_synthetic() -> tuple:
    folder = tempfile.mkdtemp()
    raw = build_raw(7, with_discount_pct=True)

    class Synthetic(Config):
        def read_raw(self):
            return raw
    configuration = Synthetic(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"],
                              structural_timevarying_dims=TIMEVARYING, extra_renovacion=["product", "channel"],
                              extra_revalorizacion=["discount", "newcust"], extended_horizon_end="2027-12",
                              benchmark_group_dims=["region", "product"], benchmark_min_support=100, sheets_top_series=0,
                              console_explanations=False, discount_value_column="discount_pct")
    with quiet():
        results = run_analysis(configuration)
    return raw, configuration, results


def test_reconciliation(raw, configuration, results) -> None:
    RECORDER.start_block("N1 · the core matches the raw, licence by licence and dollar by dollar")
    core = results["nucleo"]["nucleo"]
    RECORDER.check(bool(results["nucleo"]["reconciliacion"]["cuadra"].all()), "every reconciliation line matches")
    RECORDER.check(abs(core["s0_pipeline_usd"].sum() - raw["total_tr_usd"].sum()) < 1e-6 and abs(core["s0_renovados_usd"].sum() - raw["total_renewed_usd"].sum()) < 1e-6,
                   "Σ s0 pipeline $ and Σ s0 renewed $ = the raw's totals (the extract as it came, before wiping the future)")
    RECORDER.check(abs(core["s1_pipeline_usd"].sum() - raw["total_tr_usd"].sum()) < 1e-6, "Σ s1 pipeline $ (raw rows + explicit gaps) = the raw's pipeline")
    gaps = core[core["paso_alta"] == ROW_GAP]
    RECORDER.check(len(gaps) == int((results["units"]["sintetica"] == 1).sum()) and (gaps[["s1_pipeline_unidades", "s1_pipeline_usd", "s1_renovados_unidades"]] == 0).all().all(),
                   f"the {len(gaps)} gap months are explicit rows with measures 0 (information, not data added)")
    RECORDER.check(gaps["s0_id_fila"].isna().all() and core.loc[core["paso_alta"] == ROW_FROM_RAW, "s0_id_fila"].notna().all(), "added rows have no raw id; every raw row has one")


def test_columns_in_step_order(configuration, results) -> None:
    RECORDER.start_block("N2 · columns left to right, step by step")
    core = results["nucleo"]["nucleo"]
    steps = [int(re.match(r"s(\d)_", c).group(1)) for c in core.columns if re.match(r"s\d_", c) and c != "s0_id_fila"]
    RECORDER.check(steps == sorted(steps), "the step prefixes appear in non-decreasing order (s0 → s7)")
    RECORDER.check(not any(c.endswith("_key") for c in core.columns), "no hashed keys in the core: the dimensions themselves are the columns")
    RECORDER.check(core.columns.is_unique, "no duplicated column")


def test_aggregations_reproduce_answers(configuration, results) -> None:
    RECORDER.start_block("N3 · every study is an aggregation of the core")
    core = results["nucleo"]["nucleo"]
    business = results["forecast"]["business_summary"].set_index("anio")
    by_year = nucleo.agregar(core, ["anio"]).set_index("anio")
    bands = nucleo.banda_agregada(core, ["anio"]).set_index("anio")
    for year in business.index:
        RECORDER.check(abs(by_year.loc[year, "s6_usd_esperado"] - business.loc[year, "forecast_usd"]) < 1.0,
                       f"{year}: Σ s6 $ esperado = business_summary.forecast_usd (${business.loc[year, 'forecast_usd']:,.0f})")
        RECORDER.check(abs(bands.loc[year, "banda_total_high_usd"] - business.loc[year, "banda_high_usd"]) < 1.0 and abs(bands.loc[year, "banda_total_low_usd"] - business.loc[year, "banda_low_usd"]) < 1.0,
                       f"{year}: the band aggregated from the core (linear in pool × month, quadrature across, + common) = the business band")
    total = nucleo.agregar(core, [])
    future = core[core["s6_usd_esperado"].notna()]
    weighted_uplift = float(np.average(future["s5_uplift"], weights=future["s6_usd_esperado_precio_pipeline"]))
    RECORDER.check(abs(float(total["uplift"].iloc[0]) - weighted_uplift) < 1e-9, "the aggregate uplift is the ratio of sums (= the average weighted by $ at pipeline price)")
    closed = nucleo.agregar(core, ["s1_rol"]).set_index("s1_rol")
    RECORDER.check(pd.notna(closed.loc[ROLE_TRAIN, "tasa_real"]) and pd.isna(closed.loc[ROLE_PROJECTION, "tasa_real"]), "the realized rate exists for closed months and not for the future")


def test_pools_from_rungs(configuration, results) -> None:
    RECORDER.start_block("N4 · grouping by a rung column reproduces its pools")
    core = results["nucleo"]["nucleo"]
    ladder = results["parent_ladder"]
    closed = core[core["s1_rol"].isin(TRUTH_ROLES) & (core["s1_universo"] == UNIVERSE_NORMAL) & (core["s1_pipeline_unidades"] > 0)]
    compared, matched = 0, 0
    for rung in sorted(ladder["peldano"].unique()):
        column = f"s3_id_peldano_{int(rung)}"
        if column not in core.columns:
            continue
        pooled = closed.groupby(column).agg(ren=("s1_renovados_unidades", "sum"), pipe=("s1_pipeline_unidades", "sum"))
        pooled["tasa"] = pooled["ren"] / pooled["pipe"]
        expected = ladder[ladder["peldano"] == rung].drop_duplicates("padre_id").set_index("padre_id")["tasa_padre"].dropna()
        common = expected.index.intersection(pooled.index)
        compared += len(common)
        matched += int((np.abs(pooled.loc[common, "tasa"] - expected.loc[common]) < 1e-3).sum())
    RECORDER.check(compared > 0 and matched == compared, f"the pool rate of every rung pattern ({compared} pools) = Σ renewed / Σ pipeline over the rows with that pattern")


def test_added_rows(configuration, results) -> None:
    RECORDER.start_block("N5 · the rows the forecast adds")
    core = results["nucleo"]["nucleo"]
    projected = core[core["paso_alta"] == ROW_PROJECTED_REENTRY]
    simulated = core[core["paso_alta"] == ROW_SIMULATED_ACQUISITION]
    RECORDER.check(len(projected) > 0 and (projected["discount_pct"].fillna(0) == 0).all(), "projected re-entries renew at list price: discount 0")
    RECORDER.check(projected["s1_pipeline_usd"].isna().all() and (projected["s6_pipeline_usd"] > 0).all(), "their pipeline is not the raw's: s1 null, s6 filled")
    RECORDER.check(len(simulated) > 0 and (simulated["s6_origen_pipeline"] == PIPELINE_SIMULATED).all(), "simulated acquisition rows are labelled as such")
    RECORDER.check(np.allclose(core["s7_usd_esperado_ajustado"].dropna(), (core["s6_usd_esperado"] + core["s7_ajuste_senales_usd"])[core["s7_usd_esperado_ajustado"].notna()]),
                   "adjusted $ = expected $ + the maturation adjustment of the row")
    exam = core[core["s1_rol"] == ROLE_TEST]
    RECORDER.check(exam["s6_examen_tasa_h1"].notna().any() and core.loc[core["s1_rol"] != ROLE_TEST, "s6_examen_tasa_h1"].isna().all(),
                   "the exam prediction lives only on the exam rows (predicted vs real from the same table)")


def main() -> int:
    print("═" * 74 + "\nTEST núcleo · one table, the whole story\n" + "═" * 74)
    raw, configuration, results = run_synthetic()
    test_reconciliation(raw, configuration, results)
    test_columns_in_step_order(configuration, results)
    test_aggregations_reproduce_answers(configuration, results)
    test_pools_from_rungs(configuration, results)
    test_added_rows(configuration, results)
    return RECORDER.print_panel("NÚCLEO TEST")


if __name__ == "__main__":
    sys.exit(main())
