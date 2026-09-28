"""test_statistics.py — the statistician's battery for SFF v3.

    python test_statistics.py   →  exit 0 if healthy, 1 with the failures

What a statistician checks: every claim of the framework that has a ground truth is
simulated with that truth known and the claim is measured.
  S1  the Wilson interval covers the true proportion ≈ 90 % of the time at small n
  S2  credibility: shrinkage is stronger for smaller n, zero for a series that speaks alone,
      and the blend is never outside [own, pool]
  S3  quadrature vs linear: independent errors add as √Σ², identical ones add linearly
      (measured on simulated errors, not assumed)
  S4  no leakage: the hold-out months are absent from the decision rows of the backtest,
      and changing what happens in the exam months does not change the champion
  S5  the benchmark rejects pure binomial noise as seasonal (false-positive rate on 20
      noise series) and accepts a planted season of 8 pp
  S6  Kitagawa closes: behaviour + composition = the change of the cell's rate, and a
      cell whose parts keep their rates while their weights move is ≈ 100 % composition
  S7  the uplift ratio of sums is unbiased for a known revaluation
  S8  the per-pool bands are calibrated on a simulated portfolio with known constant
      rates: the share of exam months inside the 90 % band is ≈ 90 %
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
import binomial_reference as ref
import analysis_seasonality_benchmark as bench
import analysis_dimensions
from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from test_fixtures import ladder_config, phase_0_units, quiet
import run_rate_series
import run_support_ladder
import analysis_backtest
import run_uplift

RECORDER = CheckRecorder()
RNG = np.random.default_rng(2026)


def test_wilson_coverage() -> None:
    RECORDER.start_block("S1 · Wilson interval coverage at small n")
    for n, p in ((10, 0.8), (30, 0.5), (30, 0.95), (100, 0.7)):
        inside = 0
        trials = 4000
        for _ in range(trials):
            k = RNG.binomial(n, p)
            low, high = ref.wilson_interval(k / n, n, 1.645)
            inside += low <= p <= high
        coverage = inside / trials
        RECORDER.check(0.86 <= coverage <= 0.98, f"n={n}, p={p}: the 90 % Wilson interval covers the truth {coverage:.1%} of the time (conservative at n=10: discreteness)")
    wald_inside, wald_over_one = 0, 0
    for _ in range(4000):
        k = RNG.binomial(10, 0.95)
        p_hat = k / 10
        se = np.sqrt(p_hat * (1 - p_hat) / 10)                      # the classic Wald, no variance floor
        wald_inside += (p_hat - 1.645 * se) <= 0.95 <= (p_hat + 1.645 * se)
        wald_over_one += (p_hat + 1.645 * se) > 1.0 or se == 0
    RECORDER.check(wald_inside / 4000 < 0.86 or wald_over_one / 4000 > 0.5,
                   f"…while the classic Wald interval fails at n=10, p=.95: covers {wald_inside / 4000:.1%}, degenerate (zero width or above 100 %) {wald_over_one / 4000:.0%} of the time")


def test_credibility_properties() -> None:
    RECORDER.start_block("S2 · credibility properties")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, own_rate_floor=271.0)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
            dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
            estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
        card = card.set_index("fs_id")
        s1, s2 = card.loc["EU|0|0|web"], card.loc["EU|0|0|tele"]
        RECORDER.check(s2["z"] < s1["z"], f"the smaller series trusts itself less (z tele {s2['z']} < z web {s1['z']})")
        for series_id in ("EU|0|0|web", "EU|0|0|tele"):
            row = card.loc[series_id]
            low, high = sorted([row["tasa_propia"], row["tasa_pariente"]])
            RECORDER.check(low - 1e-9 <= row["tasa_estimada"] <= high + 1e-9, f"{series_id}: the blend lies between its own rate and the pool's")
        with quiet():
            alone = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], ladder_config(folder, own_rate_floor=100.0))[1].set_index("fs_id")
        RECORDER.check(alone.loc["EU|0|0|web", "z"] == 1.0 and abs(alone.loc["EU|0|0|web", "tasa_estimada"] - alone.loc["EU|0|0|web", "tasa_propia"]) < 1e-12,
                       "above the precision floor a series speaks alone: z = 1, estimate = own rate")
        RECORDER.check((card["se_prediccion_pp"].dropna() >= card["se_estimacion_pp"].dropna()).all(),
                       "the prediction error never falls below the estimation error (the month still samples)")


def test_quadrature_measured() -> None:
    RECORDER.start_block("S3 · quadrature vs linear, measured on simulated errors")
    n_pieces, trials = 100, 3000
    independent = RNG.normal(0, 10_000, size=(trials, n_pieces)).sum(axis=1)
    common = np.repeat(RNG.normal(0, 10_000, size=(trials, 1)), n_pieces, axis=1).sum(axis=1)
    sd_independent, sd_common = independent.std(), common.std()
    RECORDER.check(abs(sd_independent - 10_000 * np.sqrt(n_pieces)) / (10_000 * np.sqrt(n_pieces)) < 0.05,
                   f"100 independent pieces of ±10,000 add to ±{sd_independent:,.0f} ≈ √100 × 10,000 (quadrature)")
    RECORDER.check(abs(sd_common - 10_000 * n_pieces) / (10_000 * n_pieces) < 0.05,
                   f"100 pieces missing TOGETHER add to ±{sd_common:,.0f} ≈ 100 × 10,000 (linear): the common component cannot be squared away")


def test_no_leakage_into_the_exam() -> None:
    RECORDER.start_block("S4 · no leakage: the exam does not influence the decision")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, seasonal_series=True, trend_series=True, backtest_test_start="2025-10",
                                      benchmark_group_dims=["region"], benchmark_min_support=100, benchmark_short_months=20)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
            dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
            estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
            benchmark = bench.run_seasonality_benchmark(card, units, fine, configuration)
            series = analysis_backtest.monthly_series_by_estimation_id(units, decision, ladder, configuration)
            reference = analysis_backtest.build_pool_reference(series, benchmark["decision_estacionalidad"], configuration)
            result = analysis_backtest.run_backtest_analysis(series, reference, configuration, [1, 6])
        long = result["backtest_long"]
        decision_rows = long[long["mes_objetivo"] < "2025-10"]
        RECORDER.check((decision_rows["mes_objetivo"] < "2025-10").all() and (result["backtest_holdout"]["mes_objetivo"] >= "2025-10").all(),
                       "decision rows are all before the cut; hold-out rows are all at or after it")
        # perturb the exam months' truth and re-run: the champion must not move
        perturbed = {k: v.copy() for k, v in series.items()}
        for monthly in perturbed.values():
            exam = monthly.index >= pd.Period("2025-10", "M")
            monthly.loc[exam, "ren"] = (monthly.loc[exam, "pipe"] * 0.5).round()
            monthly.loc[exam, "rate"] = monthly.loc[exam, "ren"] / monthly.loc[exam, "pipe"]
        with quiet():
            result_perturbed = analysis_backtest.run_backtest_analysis(perturbed, reference, ladder_config(folder, backtest_test_start="2025-10", benchmark_group_dims=["region"]), [1, 6])
        before = result["decision_technique"].set_index(["id_estimacion", "tramo_h"])["tecnica"]
        after = result_perturbed["decision_technique"].set_index(["id_estimacion", "tramo_h"])["tecnica"]
        RECORDER.check(before.equals(after.reindex(before.index)), "rewriting what happened in the exam months changes NO champion (the choice never saw them)")
        bands_before = result["decision_error_bands"].set_index(["id_estimacion", "h"])["q_high_norm"]
        bands_after = result_perturbed["decision_error_bands"].set_index(["id_estimacion", "h"])["q_high_norm"]
        RECORDER.check(np.allclose(bands_before, bands_after.reindex(bands_before.index)), "…and no band either")


def test_benchmark_false_positives_and_power() -> None:
    RECORDER.start_block("S5 · benchmark: rejects noise, accepts a planted season")
    months = pd.period_range("2023-01", periods=42, freq="M")
    false_positives, planted_found = 0, 0
    noise_series, planted_series = 20, 20
    for k in range(noise_series):
        n = 400
        rates = RNG.binomial(n, 0.75, size=len(months)) / n
        monthly = pd.DataFrame({"ren": (rates * n).round(), "pipe": float(n), "rate": rates}, index=months)
        regression = bench.month_effect_regression(monthly)
        panel, consistency = bench.month_year_panel(monthly)
        backtest = bench.predictive_backtest(monthly)
        high, low = regression["mes_alto"], regression["mes_bajo"]
        z_high, z_low = abs(panel.loc[panel["mes"] == high, "z"].mean()), abs(panel.loc[panel["mes"] == low, "z"].mean())
        seasonal = (regression["amplitud_pp"] >= 2.0 and consistency[high][0] >= 0.67 and consistency[low][0] >= 0.67 and z_high >= 1 and z_low >= 1
                    and pd.notna(backtest["mejora_h6_pct"]) and backtest["mejora_h6_pct"] >= 10 and backtest["mejora_h1_pct"] >= 0)
        false_positives += int(seasonal)
    for k in range(planted_series):
        n = 400
        season = 0.08 * np.sin(2 * np.pi * (months.month - 1) / 12)
        rates = RNG.binomial(n, np.clip(0.75 + season, 0.01, 0.99)) / n
        monthly = pd.DataFrame({"ren": (rates * n).round(), "pipe": float(n), "rate": rates}, index=months)
        regression = bench.month_effect_regression(monthly)
        panel, consistency = bench.month_year_panel(monthly)
        backtest = bench.predictive_backtest(monthly)
        high, low = regression["mes_alto"], regression["mes_bajo"]
        z_high, z_low = abs(panel.loc[panel["mes"] == high, "z"].mean()), abs(panel.loc[panel["mes"] == low, "z"].mean())
        seasonal = (regression["amplitud_pp"] >= 2.0 and consistency[high][0] >= 0.67 and consistency[low][0] >= 0.67 and z_high >= 1 and z_low >= 1
                    and pd.notna(backtest["mejora_h6_pct"]) and backtest["mejora_h6_pct"] >= 10 and backtest["mejora_h1_pct"] >= 0)
        planted_found += int(seasonal)
    RECORDER.check(false_positives <= 2, f"pure binomial noise (n=400, 42 months): declared seasonal {false_positives} of {noise_series} times (≤ 2 tolerated)")
    RECORDER.check(planted_found >= 16, f"a planted 8 pp season: found {planted_found} of {planted_series} times (≥ 16 required)")
    noisy_lrt_significant = 0
    for k in range(noise_series):
        n = 5000
        rates = RNG.binomial(n, 0.75, size=len(months)) / n
        monthly = pd.DataFrame({"ren": (rates * n).round(), "pipe": float(n), "rate": rates}, index=months)
        noisy_lrt_significant += int(bench.month_effect_regression(monthly)["lrt_pvalue"] < 0.05)
    RECORDER.check(noisy_lrt_significant <= 4, f"the LRT on noise at n=5000 is significant {noisy_lrt_significant}/20 times at 5 %: it is reported, it never decides")


def test_kitagawa_closes() -> None:
    RECORDER.start_block("S6 · Kitagawa: closes, and pure mix is ≈ 100 % composition")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
        counterfactual, decomposition = analysis_dimensions.counterfactual_and_decomposition(units, configuration)
        RECORDER.check(np.allclose(decomposition["delta_agregado_pp"], decomposition["delta_comportamiento_pp"] + decomposition["delta_composicion_pp"], atol=0.02),
                       "behaviour + composition = the change of the cell's rate, every cell-month")
    # a hand-built cell: two series with constant rates .9 and .5 whose weights swap over time
    months = pd.period_range("2024-01", periods=24, freq="M")
    rows = []
    for i, month in enumerate(months):
        w = 1.0 - i / 24
        for name, rate, weight in (("A", 0.9, w), ("B", 0.5, 1 - w)):
            pipe = 1000 * weight
            rows.append(dict(fs_id=f"X|{name}", period=month, region="X", universo="normal", sintetica=0, tasa=rate,
                             total_renewed_units=pipe * rate, total_tr_units=pipe, total_tr_usd=pipe * 10, dataset_role=ROLE_TRAIN))
    frame = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        _, decomposition = analysis_dimensions.counterfactual_and_decomposition(frame, configuration)
    share = decomposition["delta_composicion_pp"].abs().sum() / (decomposition["delta_composicion_pp"].abs().sum() + decomposition["delta_comportamiento_pp"].abs().sum())
    RECORDER.check(share > 0.98, f"constant rates + moving weights → {share:.0%} composition (the aggregate falls from 90 % to 50 % with nobody changing behaviour)")


def test_uplift_unbiased() -> None:
    RECORDER.start_block("S7 · uplift ratio of sums is unbiased")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, _ = phase_0_units(configuration)
        renewers = run_uplift.renewer_rows(fine, configuration)
        ratio = run_uplift.ratio_of_sums(renewers, configuration)
        RECORDER.check(abs(ratio - 1.05) < 0.005, f"the built-in revaluation of 5 % is recovered ({ratio:.4f})")
        with quiet():
            decision = run_uplift.estimate_uplift_cells(renewers, configuration)
        cell = decision.iloc[0]
        RECORDER.check(cell["banda_low"] <= 1.05 <= cell["banda_high"], "the bootstrap band contains the true revaluation")


def test_band_calibration_simulated() -> None:
    RECORDER.start_block("S8 · band calibration on a simulated portfolio with known constant rates")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, backtest_test_start="2025-10", benchmark_group_dims=["region"], benchmark_min_support=100)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
            dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
            estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
            benchmark = bench.run_seasonality_benchmark(card, units, fine, configuration)
            series = analysis_backtest.monthly_series_by_estimation_id(units, decision, ladder, configuration)
            reference = analysis_backtest.build_pool_reference(series, benchmark["decision_estacionalidad"], configuration)
            result = analysis_backtest.run_backtest_analysis(series, reference, configuration, [1, 6])
        holdout = result["backtest_holdout"]
        inside = holdout["dentro_banda"].mean()
        RECORDER.check(0.75 <= inside <= 1.0, f"with constant true rates, {inside:.0%} of exam predictions fall inside the 90 % band (3 exam months: wide tolerance)")
        RECORDER.check(holdout["err_pp"].abs().mean() < 12, f"the exam error on constant-rate series is sampling-sized ({holdout['err_pp'].abs().mean():.1f} pp)")


def test_discount_churn_planted() -> None:
    """A fine table where the discount lowers the rate 10 pp among neutral customers and
    does NOTHING among no_instalado customers (who renew at 20 % regardless): the analysis
    must find the neutral gap, find no gap for no_instalado, and rank the signal above the
    discount in importance."""
    RECORDER.start_block("S9 · discount and churn on a planted effect")
    import analysis_discount_churn as dc
    rows = []
    months = pd.period_range("2025-01", periods=12, freq="M")
    for month in months:
        for region in ("EU", "NA"):
            for state_flag in (0, 1):
                for bucket in ("d0", "d40"):
                    base = 0.20 if state_flag else (0.80 if bucket == "d0" else 0.70)
                    base += 0.03 if region == "NA" else 0.0
                    n = 500
                    k = RNG.binomial(n, base)
                    rows.append(dict(period=month, dataset_role="entrenamiento", region=region, no_instalado=state_flag, dormant=0, softcancel=0, autorenew=0,
                                     discount=bucket, newcust=0, total_tr_units=n, total_tr_usd=n * 30.0, total_renewed_units=k, total_renewed_usd=k * 33.0))
    fine = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as folder:
        configuration = Config(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"],
                               structural_timevarying_dims={"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"},
                               extra_renovacion=[], extra_revalorizacion=["discount", "newcust"], no_discount_value="d0", console_explanations=False)
        with quiet():
            result = dc.run_discount_churn(fine, configuration)
    by_state = result["by_state"]
    neutral = by_state[(by_state["estado"] == "neutro") & (by_state["tramo"] == "d40")].iloc[0]
    flagged = by_state[(by_state["estado"] == "no_instalado") & (by_state["tramo"] == "d40")].iloc[0]
    RECORDER.check(-12 < neutral["hueco_pp"] < -8 and neutral["z"] < -3, f"among neutral customers the discount gap is ≈ −10 pp and significant ({neutral['hueco_pp']:+.1f} pp, z {neutral['z']:+.1f})")
    RECORDER.check(abs(flagged["hueco_pp"]) < 3 and abs(flagged["z"]) < 2, f"among no_instalado customers there is no gap ({flagged['hueco_pp']:+.1f} pp, z {flagged['z']:+.1f}): the flag dominates")
    importance = result["importance"].set_index("grupo")
    RECORDER.check(importance.loc["estado", "r2_perdido_al_quitarlo"] > importance.loc["tramo", "r2_perdido_al_quitarlo"] * 5,
                   "the signal explains far more variance than the discount")
    effects = result["effects"]
    discount_effect = effects[(effects["variable"] == "tramo") & (effects["valor"] == "d40")].iloc[0]
    RECORDER.check(discount_effect["efecto_pp"] < 0 and abs(discount_effect["z"]) > 2, f"the adjusted discount effect is negative and significant ({discount_effect['efecto_pp']:+.1f} pp): an average over states that hides where it acts")


ALL_TESTS = [test_wilson_coverage, test_credibility_properties, test_quadrature_measured, test_no_leakage_into_the_exam,
             test_benchmark_false_positives_and_power, test_kitagawa_closes, test_uplift_unbiased, test_band_calibration_simulated,
             test_discount_churn_planted]


def main() -> int:
    print("═" * 74 + "\nTEST statistics · coverage, shrinkage, quadrature, leakage, benchmark power, identities\n" + "═" * 74)
    for test in ALL_TESTS:
        test()
    return RECORDER.print_panel("STATISTICS TEST")


if __name__ == "__main__":
    sys.exit(main())
