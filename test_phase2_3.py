"""test_phase2_3.py — checks for the binomial reference, the dynamics (phase 2), the
technique catalogue and the backtest (phase 3) of SFF v3.

    python test_phase2_3.py   →  exit 0 if healthy, 1 with the failures

Blocks:
  ref   binomial_reference     se, Wilson, the dial, logit round trip, φ on a designed series
  2     diagnose_dynamics      a seasonal series is 'estacional', a trending one 'tendencia',
                               a flat big one 'apto_promedio', a small one 'soporte'
  T     techniques             eligibility by history and labels · every technique returns a
                               rate in (0, 1) · seasonal index recovers the season · damped
                               trend saturates
  3     backtest               only history ≤ origin · errors normalized · the challenger
                               wins on a flat series · a seasonal technique wins on the
                               seasonal series · bands monotone in h · hold-out ≥ test start
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
import binomial_reference as ref
import run_rate_series
import analysis_dimensions
import run_support_ladder
import analysis_seasonality_benchmark as benchmark_module
import analysis_backtest
import techniques

RECORDER = CheckRecorder()


def run_to_dynamics(configuration):
    fine, labeled = phase_0_units(configuration)
    with quiet():
        units, summary = run_rate_series.build_rate_series(labeled, configuration)
        dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
        estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
        fine, _ = phase_0_units(configuration)
        bench = benchmark_module.run_seasonality_benchmark(card, units, fine, configuration)
        series = analysis_backtest.monthly_series_by_estimation_id(units, decision, ladder, configuration)
        dynamics = analysis_backtest.build_pool_reference(series, bench["decision_estacionalidad"], configuration)
    return dict(units=units, decision=decision, ladder=ladder, dynamics=dynamics, series=series, bench=bench, card=card)


# ═══════════════════════════════════════════════════════════════════════════════════

def test_binomial_reference() -> None:
    RECORDER.start_block("ref · binomial_reference")
    RECORDER.check(abs(ref.binomial_se_pp(0.8, 100) - 4.0) < 1e-9, "se(p=.8, n=100) = 4.0 pp")
    RECORDER.check(ref.binomial_se(1.0, 10) > 0, "p = 1 does not claim zero error (variance floor)")
    low, high = ref.wilson_interval(0.95, 20, 1.645)
    RECORDER.check(high <= 1.0 and (0.95 - low) > (high - 0.95), "Wilson stays inside [0, 1] and is asymmetric near the edge")
    RECORDER.check(round(ref.support_for_half_width(15, 1.645)) == 30 and round(ref.support_for_half_width(5, 1.645)) == 271,
                   "the dial: ±15 pp ↔ 30, ±5 pp ↔ 271")
    RECORDER.check(np.allclose(ref.inverse_logit(ref.logit([0.1, 0.5, 0.9])), [0.1, 0.5, 0.9]), "logit round trip")
    rng = np.random.default_rng(1)
    flat = rng.binomial(300, 0.8, size=36) / 300
    phi_flat, _, _ = ref.overdispersion_phi(flat, np.full(36, 300))
    moving = np.clip(0.8 + 0.1 * np.sin(np.arange(36) / 2) + rng.normal(0, 0.02, 36), 0, 1)
    phi_moving, _, _ = ref.overdispersion_phi(moving, np.full(36, 300))
    RECORDER.check(0.5 < phi_flat < 2.0 and phi_moving > 5, f"φ ≈ 1 on a binomial series ({phi_flat:.2f}), ≫ 1 with an engine ({phi_moving:.1f})")


def test_dynamics() -> None:
    RECORDER.start_block("2 · seasonality benchmark on big series")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, seasonal_series=True, trend_series=True, benchmark_group_dims=["region"],
                                      benchmark_min_support=100, benchmark_short_months=20)
        results = run_to_dynamics(configuration)
        decision = results["bench"]["decision_estacionalidad"].set_index("fs_id")
        RECORDER.check(set(decision.index) >= {"NA|0|0|web", "NA|0|0|tele", "EU|0|0|web"},
                       "the benchmark takes the big neutral series (≥ min support in every closed month)")
        RECORDER.check("EU|1|0|web" not in decision.index and "EU|0|0|tele" not in decision.index,
                       "signed and small series stay out of the benchmark")
        seasonal = decision.loc["NA|0|0|web"]
        RECORDER.check(seasonal["veredicto_estacional"] == 1 and seasonal["amplitud_pp"] > 10 and seasonal["consistencia_alto"] >= 0.67
                       and seasonal["mejora_h6_pct"] > 10, f"the seasonal series is declared seasonal (amplitude {seasonal['amplitud_pp']} pp, shape improves h=6 by {seasonal['mejora_h6_pct']:.0f}%)")
        RECORDER.check(seasonal["phi"] > 3, "…and its φ after the trend says it moves beyond sampling")
        flat = decision.loc["EU|0|0|web"]
        RECORDER.check(flat["veredicto_estacional"] == 0 and flat["phi"] < 2, "a flat big series is not seasonal and φ ≈ 1")
        trending = decision.loc["NA|0|0|tele"]
        RECORDER.check(trending["veredicto_tendencia"] == -1 and trending["pendiente_pp_anio"] < -3, "the declining series gets a trend verdict of −1")
        RECORDER.check(trending["veredicto_estacional"] == 0, "…and is not declared seasonal (the trend is removed before the month effects)")
        panel = results["bench"]["bench_panel"]
        RECORDER.check(set(panel.columns) >= {"fs_id", "mes", "anio", "z"} and panel["fs_id"].nunique() == len(decision),
                       "the month × year panel has one z per series × month × year")
        reference = results["dynamics"].set_index("id_estimacion")
        RECORDER.check(reference.loc["NA|0|0|web", "estacional"] == 1 and reference.loc["EU|0|0|web", "estacional"] == 0,
                       "the pool reference carries the benchmark's verdict per estimation id")
        RECORDER.check(reference.loc["EU|1|1|web", "gate"] == "soporte", "a pool under the floor is gated by 'soporte'")
        pooled = results["series"]["EU|SIG=neutro|*"]
        RECORDER.check(abs(pooled["pipe"].median() - 210) < 1, "the monthly series of a relative sums EVERY matching series (210 = S1 + S2)")


def test_techniques() -> None:
    RECORDER.start_block("T · techniques")
    RECORDER.check("T15_level_seasonal" not in techniques.eligible_techniques(24, dict(estacional=0, tendencia=0))
                   and "T15_level_seasonal" in techniques.eligible_techniques(24, dict(estacional=1, tendencia=0)),
                   "the only seasonal technique competes only where the benchmark declared month effects")
    RECORDER.check(len(techniques.CATALOGUE) == 10 and "T7_seasonal_idx" not in techniques.CATALOGUE
                   and "T11_holt_winters" not in techniques.eligible_techniques(36, dict(estacional=0)) and "T12_theta" in techniques.eligible_techniques(12, {}),
                   "10 techniques: level + Theta (always) + two seasonal ones gated by the benchmark; no self-found seasonality")
    RECORDER.check("T3_ma3" in techniques.eligible_techniques(3, {}), "the challenger is eligible with 3 months")
    months = pd.period_range("2023-01", periods=36, freq="M")
    seasonal = 0.7 + 0.1 * np.sin(2 * np.pi * (months.month - 1) / 12)
    labels = dict(estacional=2, tendencia=0, intermitente=0)
    for technique_id in techniques.CATALOGUE:
        value = techniques.predict(technique_id, seasonal, months, 3, labels)
        RECORDER.check(np.isfinite(value) and 0 < value < 1, f"{technique_id} returns a rate in (0, 1)")
    april_prediction = techniques.predict("T15_level_seasonal", seasonal, months, 4, labels)   # last = 2025-12 → h=4 = April
    RECORDER.check(abs(april_prediction - 0.8) < 0.02, f"the month effect on the recent level recovers April's .80 ({april_prediction:.3f})")
    trending = np.clip(0.9 - 0.01 * np.arange(36), 0.01, 0.99)
    far = techniques.predict("T10_holt_damped", trending, months, 24, dict(tendencia=-1))
    linear_extrapolation = trending[-1] - 0.01 * 24
    RECORDER.check(far > linear_extrapolation + 0.05, "the damped slope saturates: far horizon stays above the linear extrapolation")
    near_one = np.full(36, 0.99)
    RECORDER.check(techniques.predict("T2_mean", near_one, months, 1, {}) < 1.0, "the logit keeps predictions below 100 %")


def test_backtest() -> None:
    RECORDER.start_block("3 · backtest, technique, bands, hold-out")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, seasonal_series=True, trend_series=True, backtest_test_start="2025-10",
                                      benchmark_group_dims=["region"], benchmark_min_support=100, benchmark_short_months=20)
        results = run_to_dynamics(configuration)
        with quiet():
            backtest = analysis_backtest.run_backtest_analysis(results["series"], results["dynamics"], configuration, [1, 2, 3, 4])
        long = backtest["backtest_long"]
        RECORDER.check((pd.PeriodIndex(long["origen"], freq="M") + long["h"] == pd.PeriodIndex(long["mes_objetivo"], freq="M")).all(),
                       "origin + h = target for every prediction (only history ≤ origin is used)")
        RECORDER.check(np.allclose(long["err_norm"], long["err_pp"] / long["se_binom_pp"], atol=1e-2),
                       "err_norm = err_pp / binomial se of the target month")
        RECORDER.check(long["tecnica_id"].nunique() >= 6, "the level techniques compete where the history allows")
        seasonal_id = long[long["id_estimacion"] == "NA|0|0|web"]
        RECORDER.check("T15_level_seasonal" in set(seasonal_id["tecnica_id"]) and "T15_level_seasonal" not in set(long[long["id_estimacion"] == "EU|0|0|web"]["tecnica_id"]),
                       "the seasonal technique competes only on the series the benchmark declared seasonal")
        decision = backtest["decision_technique"]
        RECORDER.check(set(decision["tramo_h"]) == {"h1", "corto", "medio", "largo"} and (decision.groupby("id_estimacion").size() == 4).all(),
                       "one champion per estimation id AND horizon band (h1 / corto / medio / largo)")
        chosen = decision[decision["tramo_h"] == "h1"].set_index("id_estimacion")
        RECORDER.check(chosen.loc["EU|0|0|web", "tecnica"] == "T2_mean" and chosen.loc["EU|0|0|web", "tecnica_origen"] == "retador",
                       "on a flat series nobody beats the challenger by the margin → T2_mean, origin 'retador'")
        RECORDER.check(chosen.loc["NA|0|0|web", "tecnica_origen"] == "campeon" and chosen.loc["NA|0|0|web", "tecnica"] == "T15_level_seasonal",
                       f"on the seasonal series the month-effect technique wins ({chosen.loc['NA|0|0|web', 'tecnica']})")
        RECORDER.check(chosen.loc["NA|0|0|tele", "tecnica_origen"] == "campeon", "on the trending series a champion beats the mean")
        inherited = decision[decision["tecnica_origen"].str.endswith("_heredado")]
        RECORDER.check((inherited["tramo_h"] != "h1").all(), "a band with no screen horizon inside inherits the previous band's champion")
        bands = backtest["decision_error_bands"]
        widths = bands.assign(w=bands["q_high_norm"] - bands["q_low_norm"]).groupby("id_estimacion")["w"]
        RECORDER.check(all((group.diff().dropna() >= -1e-9).all() for _, group in widths), "band width never narrows with h")
        RECORDER.check((bands["q_low_norm"] <= 0).all() and (bands["q_high_norm"] >= 0).all(), "bands contain zero")
        holdout = backtest["backtest_holdout"]
        RECORDER.check(holdout["mes_objetivo"].min() >= "2025-10", "the hold-out contains only months ≥ backtest_test_start")
        RECORDER.check(set(holdout["tecnica"]) <= set(decision["tecnica"]), "the hold-out uses the chosen technique per id and band")
        inside = holdout["dentro_banda"].mean()
        RECORDER.check(0.75 <= inside <= 1.0, f"hold-out calibration: {inside:.0%} inside the band (target ≈ 90 %)")


ALL_TESTS = [test_binomial_reference, test_dynamics, test_techniques, test_backtest]


def main() -> int:
    print("═" * 74 + "\nTEST phases 2-3 · dynamics, techniques, backtest\n" + "═" * 74)
    for test in ALL_TESTS:
        test()
    return RECORDER.print_panel("PHASES 2-3 TEST")


if __name__ == "__main__":
    sys.exit(main())
