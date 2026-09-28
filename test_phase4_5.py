"""test_phase4_5.py — checks for the uplift (phase 4) and the assembly with extended
horizon (phase 5) of SFF v3.

    python test_phase4_5.py   →  exit 0 if healthy, 1 with the failures

Blocks:
  4   uplift            ratio of sums by hand · n = renewers · parent when below the floor ·
                        bootstrap band contains the ratio · the cap
  5   extended horizon  rows only beyond the known pipeline · units(m) = renewed(m−term) × factor ·
                        factor from history · simulated flag · fu_key minted
  5   assembly          every future row has a rate, uplift and $ · rate ≤ cap · the technique
                        of the estimation id at h · esperado = pipeline$ × tasa × uplift
  5   bands             asymmetric · contain the point · rows of one (id, month) add linearly,
                        across in quadrature · the total's band never narrows with h
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from test_fixtures import ladder_config, phase_0_units, quiet
import run_rate_series
import analysis_dimensions
import run_support_ladder
import analysis_seasonality_benchmark as benchmark_module
import analysis_backtest
import run_uplift
import run_forecast_assembly as assembly

RECORDER = CheckRecorder()


def run_to_assembly(configuration, horizons):
    fine, labeled = phase_0_units(configuration)
    with quiet():
        units, summary = run_rate_series.build_rate_series(labeled, configuration)
        dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
        estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
        bench = benchmark_module.run_seasonality_benchmark(card, units, fine, configuration)
        series = analysis_backtest.monthly_series_by_estimation_id(units, decision, ladder, configuration)
        dynamics = analysis_backtest.build_pool_reference(series, bench["decision_estacionalidad"], configuration)
        backtest = analysis_backtest.run_backtest_analysis(series, dynamics, configuration, horizons)
        uplift = run_uplift.run_uplift(fine, configuration)
        decisions = dict(decision_support=decision, decision_technique=backtest["decision_technique"], pool_reference=dynamics,
                         decision_error_bands=backtest["decision_error_bands"], decision_uplift=uplift)
        forecast = assembly.run_forecast_assembly(fine, units, estimates, card, decisions, series, configuration)
    return dict(fine=fine, units=units, estimates=estimates, card=card, decisions=decisions, series=series,
                dynamics=dynamics, uplift=uplift, forecast=forecast)


# ═══════════════════════════════════════════════════════════════════════════════════

def test_uplift() -> None:
    RECORDER.start_block("4 · uplift")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, _ = phase_0_units(configuration)
        renewers = run_uplift.renewer_rows(fine, configuration)
        by_hand = renewers["total_renewed_usd"].sum() / (renewers["total_renewed_units"] * renewers["auv_pipeline"]).sum()
        RECORDER.check(abs(run_uplift.ratio_of_sums(renewers, configuration) - by_hand) < 1e-12, "ratio of sums = Σ ren$ / Σ (ren units × pipeline AUV)")
        with quiet():
            decision = run_uplift.estimate_uplift_cells(renewers, configuration)
        cell = decision.set_index("uplift_cell_id").loc["EU|d0"]
        RECORDER.check(abs(cell["uplift_propio"] - 1.05) < 0.01 and cell["uplift_origen"] == "propia",
                       f"the cell's own uplift ≈ 1.05 as built ({cell['uplift_propio']:.4f}), origin 'propia'")
        RECORDER.check(cell["n_renovadores"] == renewers["total_renewed_units"].sum(), "n = the number of RENEWERS, not the pipeline")
        RECORDER.check(cell["banda_low"] <= cell["uplift"] <= cell["banda_high"] and cell["banda_high"] - cell["banda_low"] < 0.05,
                       "the bootstrap band contains the ratio and is tight with thousands of renewers")
        # a tiny cell falls to its parent, then to the mandatory cell
        tiny = renewers.copy()
        picked = tiny.index[:5]
        tiny.loc[picked, "discount"] = "d99"
        tiny.loc[picked, "total_renewed_units"] = 2.0                       # 10 renewers < floor 30
        tiny.loc[picked, "total_renewed_usd"] = 2.0 * 20.0 * 1.30           # own ratio 1.30, not credible
        tiny["uplift_cell_id"] = tiny["region"] + "|" + tiny["discount"]
        with quiet():
            decision_tiny = run_uplift.estimate_uplift_cells(tiny, configuration)
        small = decision_tiny.set_index("uplift_cell_id").loc["EU|d99"]
        RECORDER.check(small["n_renovadores"] < configuration.uplift_floor and small["uplift_origen"] in ("padre", "celda")
                       and abs(small["uplift"] - small["uplift_celda"]) < 1e-9,
                       "a cell below the floor takes its parent's / cell's ratio and says so")
        capped = renewers.copy()
        capped["total_renewed_usd"] = capped["total_renewed_usd"] * 10
        with quiet():
            decision_capped = run_uplift.estimate_uplift_cells(capped, configuration)
        RECORDER.check((decision_capped["uplift"] <= configuration.uplift_cap).all() and decision_capped["recortado"].all(),
                       "an implausible ratio is clipped at uplift_cap and flagged 'recortado'")


def test_extended_horizon() -> None:
    RECORDER.start_block("5 · extended horizon")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, extended_horizon_end="2026-09", renewal_term_months=12)
        results = run_to_assembly(configuration, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        extended = results["forecast"]["forecast_units_extended"]
        RECORDER.check(extended["period"].min() == pd.Period("2026-04", "M") and extended["period"].max() == pd.Period("2026-09", "M"),
                       "simulated rows cover only the months beyond the known pipeline (2026-04..2026-09)")
        RECORDER.check((extended["simulada"] == 1).all() and extended["total_renewed_units"].isna().all(),
                       "simulated rows are flagged and carry no results")
        fine, units = results["fine"], results["units"]
        s1_rows = extended[(extended["region"] == "EU") & (extended["softcancel"] == 0) & (extended["autorenew"] == 0)
                           & (extended["channel"] == "web") & (extended["period"] == pd.Period("2026-04", "M"))]
        s1_2026_04 = s1_rows.iloc[0]
        source = fine[(fine["region"] == "EU") & (fine["softcancel"] == 0) & (fine["autorenew"] == 0) & (fine["channel"] == "web")
                      & (fine["period"] == pd.Period("2025-04", "M"))].iloc[0]
        factor = s1_2026_04["factor_adquisicion"]
        RECORDER.check(abs(s1_rows["total_tr_units"].sum() - source["total_renewed_units"] * factor) < 1e-6,
                       "units(2026-04) = renewed(2025-04, observed) × acquisition factor (summed over the two origin rows)")
        RECORDER.check(set(s1_rows["origen_pipeline"]) == {"proyectada", "simulada"}
                       and abs(s1_rows.loc[s1_rows["origen_pipeline"] == "proyectada", "total_tr_units"].iloc[0] - source["total_renewed_units"]) < 1e-6,
                       "the re-entry is split: 'proyectada' = the renewals, 'simulada' = the acquisition (factor − 1)")
        RECORDER.check(abs(factor - 200 / (200 * 0.8)) < 0.15, f"the factor ≈ pipeline / renewed one term earlier ≈ 1.25 ({factor:.3f})")
        # the source (2025-04) has truth: the simulated pipeline is valued at the OBSERVED renewed AUV (20 × 1.05 = 21)
        RECORDER.check(abs(s1_2026_04["total_tr_usd"] / s1_2026_04["total_tr_units"] - 21.0) < 1e-9,
                       "the simulated row is valued at the renewed price (observed renewed AUV 21, not the pipeline AUV 20)")
        # a filter keeps multi-year rows out: with channel as a stand-in for the term column, only 'web' re-enters
        filtered = run_to_assembly(ladder_config(folder, extended_horizon_end="2026-09", extension_row_filter={"channel": ["web"]}), [1, 2, 3])
        RECORDER.check(set(filtered["forecast"]["forecast_units_extended"]["channel"]) == {"web"},
                       "extension_row_filter: only the allowed rows re-enter the simulated pipeline")
        RECORDER.check(extended["fu_comb_key"].is_unique and not extended["fu_key"].isin(fine["fu_key"]).any(),
                       "simulated rows get new keys that do not collide with the raw")
        # 2027-01 would need 2026-01 (a projection month): the expected renewals are used instead
        configuration_far = ladder_config(folder, extended_horizon_end="2027-02", renewal_term_months=12)
        far = run_to_assembly(configuration_far, [1, 2, 3])
        far_extended = far["forecast"]["forecast_units_extended"]
        rate_s1 = far["estimates"].set_index("fs_id").loc["EU|0|0|web", "tasa_estimada"]
        rows_2027 = far_extended[(far_extended["channel"] == "web") & (far_extended["softcancel"] == 0) & (far_extended["autorenew"] == 0)
                                 & (far_extended["period"] == pd.Period("2027-01", "M"))]
        RECORDER.check(abs(rows_2027["total_tr_units"].sum() - 200 * rate_s1 * rows_2027["factor_adquisicion"].iloc[0]) < 1e-6,
                       "a month whose source is a projection month uses pipeline × tasa_estimada as expected renewals")
        RECORDER.check(len(assembly.extend_forecast_units(fine, units, results["estimates"], ladder_config(folder))) == 0,
                       "with no extended_horizon_end nothing is simulated")


def test_assembly_and_bands() -> None:
    RECORDER.start_block("5 · assembly")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, extended_horizon_end="2026-09", rate_cap=0.90)
        results = run_to_assembly(configuration, list(range(1, 10)))
        detail, bands = results["forecast"]["forecast_detail"], results["forecast"]["forecast_bands"]
        fine = results["fine"]
        future_rows = (fine["dataset_role"] == ROLE_PROJECTION).sum() + len(results["forecast"]["forecast_units_extended"])
        RECORDER.check(len(detail) == future_rows and detail["esperado_usd"].notna().all(), "every future row (known + simulated) has a forecast")
        RECORDER.check(np.allclose(detail["esperado_usd"], detail["total_tr_usd"] * detail["tasa"] * detail["uplift"]),
                       "esperado_usd = pipeline$ × tasa × uplift")
        RECORDER.check((detail["tasa"] <= 0.90 + 1e-12).all() and (detail.loc[detail["fs_id"] == "EU|0|1|web", "tasa"] == 0.90).all(),
                       "the rate is saturated at rate_cap (the .95 series lands on .90)")
        from analysis_backtest import technique_for
        technique_at = technique_for(results["decisions"]["decision_technique"])
        expected = [technique_at.get((i, int(h)), "T2_mean") if pd.notna(h) else "T2_mean" for i, h in zip(detail["id_estimacion"], detail["h"])]
        RECORDER.check((detail["tecnica"] == expected).all(), "every row uses the technique decided for its estimation id and its horizon band")
        s1 = detail[detail["fs_id"] == "EU|0|0|web"].drop_duplicates("period").sort_values("period")
        RECORDER.check(list(s1["h"]) == list(range(1, 10)), "h counts from the last month with truth (2025-12): 2026-01 is h=1 … 2026-09 is h=9")
        RECORDER.check((s1["tasa_origen"] == "serie").all() and s1["tecnica_origen"].iloc[0] in ("campeon", "retador"),
                       "rows of a trainable series get their rate from the series' technique")
        RECORDER.check((bands["banda_low_pp"] <= 0).all() and (bands["banda_high_pp"] >= 0).all(), "bands contain the point forecast")
        RECORDER.check(np.allclose(bands["banda_low_usd"], bands["banda_low_pp"] * detail["total_tr_usd"] * detail["uplift"] / 100, atol=0.02),
                       "band dollars = band pp × pipeline$ × uplift")
        small, big = bands[detail["fs_id"] == "EU|0|0|tele"], bands[detail["fs_id"] == "EU|0|0|web"]
        RECORDER.check((small["banda_high_pp"] - small["banda_low_pp"]).mean() > (big["banda_high_pp"] - big["banda_low_pp"]).mean() * 2,
                       "a small series (n=10) has a much wider band in pp than a big one")
        RECORDER.check(small["banda_high_pp"].max() <= 100 * (0.90 - small.merge(detail[["fu_comb_key", "tasa"]], on="fu_comb_key")["tasa"]).max() + 1e-3,
                       "the high side of the band is clipped at the rate cap (asymmetry near the ceiling)")
    RECORDER.start_block("5 · aggregation")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        # a hand-built band table: two rows sharing (id, month) and one row apart
        table = pd.DataFrame(dict(id_estimacion=["X", "X", "Y"], period=[pd.Period("2026-01", "M")] * 3,
                                  esperado_usd=[100, 100, 100], banda_low_usd=[-3, -4, -12], banda_high_usd=[3, 4, 12], nivel_riesgo=["A"] * 3))
        total = assembly.aggregate_with_bands(table, ["nivel_riesgo"], configuration)
        RECORDER.check(abs(total["banda_high_usd"].iloc[0] - np.sqrt(7 ** 2 + 12 ** 2)) < 1e-9,
                       "rows sharing (id, month) add linearly (3+4=7), then quadrature with the other (√(7²+12²))")
        configuration = ladder_config(folder, extended_horizon_end="2026-09")
        results = run_to_assembly(configuration, list(range(1, 10)))
        horizon = results["forecast"]["horizon_report"]
        RECORDER.check(len(horizon) == 9 and (horizon["banda_monotona"] == 0).sum() <= 1,
                       "the total's relative band narrows beyond the mix tolerance in at most one month (composition of the simulated rows)")
        per_id_bands = results["decisions"]["decision_error_bands"].assign(w=lambda b: b["q_high_norm"] - b["q_low_norm"])
        RECORDER.check(all((g.sort_values("h")["w"].diff().dropna() >= -1e-9).all() for _, g in per_id_bands.groupby("id_estimacion")),
                       "per estimation id the band is monotone in h by construction")
        RECORDER.check(horizon["pct_simulado"].iloc[0] == 0 and horizon["pct_simulado"].iloc[-1] == 100, "% simulated: 0 on known months, 100 beyond the known pipeline")
        by_level = results["forecast"]["forecast_by_level"]
        RECORDER.check(abs(by_level["esperado_usd"].sum() - results["forecast"]["forecast_detail"]["esperado_usd"].sum()) < 1e-6,
                       "the forecast by level sums to the total")


ALL_TESTS = [test_uplift, test_extended_horizon, test_assembly_and_bands]


def main() -> int:
    print("═" * 74 + "\nTEST phases 4-5 · uplift, extended horizon, assembly\n" + "═" * 74)
    for test in ALL_TESTS:
        test()
    return RECORDER.print_panel("PHASES 4-5 TEST")


if __name__ == "__main__":
    sys.exit(main())
