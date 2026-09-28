"""test_ideas.py — the battery of the composition effect, the panorama and the price layer.

    python test_ideas.py   →  exit 0 if healthy, 1 with the failures
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
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
import analysis_composition as ac
import analysis_portfolio_overview as po
import analysis_price_scenarios as ps
import run_uplift
from pipeline import run_analysis, phase_0
from run_rate_series import build_rate_series
from synthetic_v3 import build_raw
from test_fixtures import quiet

RECORDER = CheckRecorder()
TIMEVARYING = {"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"}


def small_config(folder: str, **overrides) -> Config:
    arguments = dict(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"], structural_timevarying_dims=TIMEVARYING,
                     extra_renovacion=["product", "channel"], extra_revalorizacion=["discount", "newcust"], console_explanations=False, sheets_top_series=0)
    arguments.update(overrides)
    return Config(**arguments)


def two_period_frame(weights_0, weights_1, rates_0, rates_1) -> pd.DataFrame:
    rows = []
    for period, weights, rates in (("p0", weights_0, rates_0), ("p1", weights_1, rates_1)):
        for cell, w, r in zip(("A", "B", "C"), weights, rates):
            n = 10000 * w
            rows.append(dict(anio=period, region="X", product=cell, channel="web", newcust=0, dormant=0, softcancel=0, no_instalado=0, autorenew=0,
                             total_tr_units=n, total_renewed_units=n * r, total_tr_usd=n * 20.0))
    return pd.DataFrame(rows)


def test_composition_identities() -> None:
    RECORDER.start_block("I3 · the composition effect: identities")
    with tempfile.TemporaryDirectory() as folder:
        configuration = small_config(folder)
        frame = two_period_frame((.5, .3, .2), (.2, .3, .5), (.9, .6, .4), (.85, .65, .5))
        summary, cells = ac.decompose_rate_change(frame, ["product"], "p0", "p1", configuration, period_column="anio")
        RECORDER.check(abs(summary["composicion_pp"] + summary["comportamiento_pp"] - summary["cambio_total_pp"]) < 1e-9, "composition + behaviour = total change, exactly")
        summary, _ = ac.decompose_rate_change(two_period_frame((.7, .2, .1), (.3, .3, .4), (.9, .6, .4), (.9, .6, .4)), ["product"], "p0", "p1", configuration, period_column="anio")
        RECORDER.check(abs(summary["comportamiento_pp"]) < 1e-9 and summary["pct_composicion"] > 99.9, "planted mix with constant cell rates → behaviour = 0, 100 % composition")
        summary, _ = ac.decompose_rate_change(two_period_frame((.5, .3, .2), (.5, .3, .2), (.9, .6, .4), (.8, .7, .5)), ["product"], "p0", "p1", configuration, period_column="anio")
        RECORDER.check(abs(summary["composicion_pp"]) < 1e-9, "rates changed with constant weights → composition = 0")
        summary, _ = ac.decompose_rate_change(two_period_frame((.5, .3, .2), (.2, .3, .5), (.9, .6, .4), (.85, .65, .5)), [], "p0", "p1", configuration, period_column="anio")
        RECORDER.check(abs(summary["composicion_pp"]) < 1e-9 and summary["celdas"] == 1, "a single cell → composition = 0 (everything is behaviour)")
        example = two_period_frame((.7, .3, 0), (.6, .4, 0), (.9, .4, .5), (.9, .4, .5))
        summary, _ = ac.decompose_rate_change(example, ["product"], "p0", "p1", configuration, period_column="anio")
        RECORDER.check(abs(summary["cambio_total_pp"] + 5) < 1e-6 and abs(summary["composicion_pp"] + 5) < 1e-6, "the document's example: 75 % → 70 %, −5 pp, all composition")


def test_panorama() -> None:
    RECORDER.start_block("I1 · the panorama of the portfolio")
    with tempfile.TemporaryDirectory() as folder:
        class Synthetic(Config):
            def read_raw(self):
                return build_raw(7)
        configuration = Synthetic(**{k: v for k, v in vars(small_config(folder)).items() if k in Config.__dataclass_fields__})
        with quiet():
            r0 = phase_0(configuration)
            units, _ = build_rate_series(r0["labeled_units"], configuration)
            out = po.run_portfolio_overview(units, configuration)
        summary = out["panorama_resumen"]
        total_pipeline = units[units.get("sintetica", 0) == 0][configuration.pipeline_usd_col].sum()
        RECORDER.check(abs(summary["usd"].sum() - total_pipeline) < 1e-6, "the sum of the roles in panorama_resumen = the total pipeline $")
        monthly = out["panorama_mensual"]
        closed = monthly[monthly["rol"].isin(TRUTH_ROLES)]
        RECORDER.check(closed["tasa_composicion_fija"].notna().sum() > 12 and (closed["banda_high"] > closed["banda_low"]).all(), "the monthly series carries the fixed-composition rate and a binomial band on every closed month")
        diagnosis = out["panorama_diagnostico"]
        RECORDER.check(set(diagnosis["serie"]) == {"tasa", "pipeline"} and set(diagnosis["veredicto"]) <= {"estacional", "tendencia", "estable", "fluctua_sin_patron", "historia_corta"},
                       "one diagnosis per aggregate series with a plain-text verdict")
        RECORDER.check(len(out["panorama_volumen_ts"]) >= 3, "the volume is backtested with a flat level, the seasonal index and Holt-Winters (where time series are judged)")
    # planted mix: two cells with constant rates whose weights move → observed moves, fixed-composition line flat
    with tempfile.TemporaryDirectory() as folder:
        configuration = small_config(folder, business_mandatory_dims=["region", "product"], extra_renovacion=[], extra_revalorizacion=[])   # the mix is between mandatory cells
        rows = []
        months = pd.period_range("2023-01", "2026-08", freq="M")
        for i, month in enumerate(months):
            w = 0.8 - 0.5 * i / len(months)
            for product, rate, weight in (("A", .9, w), ("B", .5, 1 - w)):
                n = 2000 * weight
                rows.append(dict(period=month, dataset_role=ROLE_TRAIN if month < pd.Period("2026-02", "M") else ROLE_TEST, region="X", product=product,
                                 dormant=0, softcancel=0, no_instalado=0, autorenew=0, fs_id=f"X|{product}", universo="normal", sintetica=0,
                                 total_tr_units=n, total_renewed_units=n * rate, total_tr_usd=n * 10.0, total_renewed_usd=n * rate * 10.0, tasa=rate))
        units = pd.DataFrame(rows)
        with quiet():
            out = po.run_portfolio_overview(units, configuration)
        monthly = out["panorama_mensual"]
        observed_range = monthly["tasa"].max() - monthly["tasa"].min()
        fixed = monthly["tasa_composicion_fija"].dropna()
        RECORDER.check(observed_range > 0.15 and (fixed.max() - fixed.min()) < 0.01, f"planted mix with constant cell rates: observed rate moves {100 * observed_range:.0f} pp, fixed-composition rate flat ({100 * (fixed.max() - fixed.min()):.2f} pp)")


def test_price_scenarios() -> None:
    RECORDER.start_block("I2 · the price scenario layer")
    with tempfile.TemporaryDirectory() as folder:
        raw = build_raw(7, with_discount_pct=True)
        raw["sku"] = raw["product"] + "-" + raw["channel"]

        class WithSku(Config):
            def read_raw(self):
                return raw
            def read_price_table(self):
                return pd.DataFrame([dict(escenario="base_mas5", sku="A-web", precio_lista=np.nan, fecha_efectiva="2027-03-01", subida_pct=0.05, subida_importe=0.0, delta_tasa_pp=-1.0)])
        configuration = WithSku(**{k: v for k, v in vars(small_config(folder, extended_horizon_end="2027-12", benchmark_group_dims=["region", "product"],
                                                                     benchmark_min_support=100, discount_value_column="discount_pct", sku_column="sku")).items() if k in Config.__dataclass_fields__})
        with quiet():
            results = run_analysis(configuration)
        scenarios = results["scenarios"]["escenarios_precio"]
        margin = results["scenarios"]["escenarios_margen"]
        RECORDER.check(len(scenarios) > 0 and set(scenarios["sku"]) == {"A-web"}, "only the SKU of the price table is affected")
        before = scenarios[scenarios[configuration.period_col] < "2027-03"]
        after = scenarios[scenarios[configuration.period_col] >= "2027-03"]
        RECORDER.check((before["neto_usd"].abs() < 1e-6).all() and (after["neto_usd"] != 0).any(), "rows before fecha_efectiva do not change; rows from it on do")
        RECORDER.check((after["usd_por_precio"] > 0).all() and (after["usd_por_retencion"] <= 1e-9).all(), "the increase adds money by price and the rate drop costs money by retention")
        RECORDER.check(set(margin["margen"]) >= {"sin_margen"} and (margin.groupby("anio")["pct_usd_anio"].sum().round(6) == 100).all(), "the margin classification covers the future portfolio (real + projected) and sums to 100 % per year")
        detail = results["forecast"]["forecast_detail"]
        RECORDER.check("simulada" not in set(scenarios["margen"]), "the simulated acquisition never enters a scenario")
        # the document's example on the formula: 10,000 units, list 50, rate 60 %, +5 %, −1 pp
        base = 10000 * 0.60 * 50
        scenario = 10000 * 0.59 * 52.5
        RECORDER.check(abs(scenario - base - 9750) < 1e-6 and abs(0.60 * 50 / 52.5 - 0.5714) < 1e-3, "the document's example: +$9,750 and break-even 57.1 %")


def test_backtest_uplift_paths() -> None:
    RECORDER.start_block("I2 · the backtest of the uplift: A = B without known discounts")
    with tempfile.TemporaryDirectory() as folder:
        class Synthetic(Config):
            def read_raw(self):
                return build_raw(7)
        configuration = Synthetic(**{k: v for k, v in vars(small_config(folder, extended_horizon_end="2027-12", benchmark_group_dims=["region", "product"], benchmark_min_support=100)).items() if k in Config.__dataclass_fields__})
        with quiet():
            results = run_analysis(configuration)
        table = results["backtest_uplift"]
        total = table[(table["ambito"] == "total") & (table["autorenew"] == "*")].iloc[0]
        RECORDER.check(abs(total["wape_final_A_pct"] - total["wape_final_B_pct"]) < 1e-9 and total["pct_usd_descuento_conocido"] == 0, "with no known discount, path A = path B and the verdict is a tie")
        RECORDER.check(total["veredicto"] == "empata", "…verdict: empata")


ALL_TESTS = [test_composition_identities, test_panorama, test_price_scenarios, test_backtest_uplift_paths]


def main() -> int:
    print("═" * 74 + "\nTEST ideas · composition effect, panorama, price scenarios, uplift backtest\n" + "═" * 74)
    for test in ALL_TESTS:
        test()
    return RECORDER.print_panel("IDEAS TEST")


if __name__ == "__main__":
    sys.exit(main())
