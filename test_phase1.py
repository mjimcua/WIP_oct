"""test_phase1.py — checks for phase 1 (SFF v3): rate series, dimensions, support ladder.

    python test_phase1.py   →  exit 0 if healthy, 1 with the failures

Built on the six-series raw of `test_fixtures.py`, where the expected ladder can be
verified by eye (see its docstring). Blocks:
  1.1  build_rate_series        fs_key · gaps with undefined rate · summary · sign
  1.2  dimension_separation     η² individual / unique / ω² · pairs · collapse order
  1.2  mix-shift                counterfactual and Kitagawa decomposition on a designed cell
  1.2  timevarying calibration  realized rate per flag and per sign
  1.3  build_relatives          the relative lists of every sign
  1.3  climb_ladder             who borrows from whom, and who stays alone
  1.3  estimate_rates           credibility only with a relative; the two errors
  1.3  risk levels              A / B / S / M and the money report
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
from run_support_ladder import build_relatives, collapse_order

RECORDER = CheckRecorder()


def run_to_ladder(configuration):
    fine, labeled = phase_0_units(configuration)
    with quiet():
        units, summary = run_rate_series.build_rate_series(labeled, configuration)
        dims = analysis_dimensions.run_dimension_analysis(units, summary, configuration)
        estimates, card, decision, ladder = run_support_ladder.run_support_ladder(units, summary, dims["decision_eta2"], configuration)
        composition = analysis_dimensions.run_composition_analysis(units, configuration)
    return dict(fine=fine, units=units, summary=summary, dims=dims, estimates=estimates, card=card,
                decision=decision, ladder=ladder)


# ═══════════════════════════════════════════════════════════════════════════════════

def test_rate_series() -> None:
    RECORDER.start_block("1.1 · build_rate_series")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, labeled = phase_0_units(configuration)
        # punch a gap into S1's history: remove 2024-06
        labeled = labeled[~((labeled["fs_id"] == "EU|0|0|web") & (labeled["period"].astype(str) == "2024-06"))]
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
        s1 = units[units["fs_id"] == "EU|0|0|web"].set_index("period")
        RECORDER.check(pd.Period("2024-06", "M") in s1.index and s1.loc[pd.Period("2024-06", "M"), "sintetica"] == 1,
                       "a missing month inside the history becomes a synthetic row")
        RECORDER.check(np.isnan(s1.loc[pd.Period("2024-06", "M"), "tasa"]),
                       "the synthetic row's rate is NaN (a month with no expirations says nothing)")
        RECORDER.check(s1[s1["dataset_role"] == ROLE_PROJECTION]["tasa"].isna().all(),
                       "projection rows have no rate")
        real = s1[(s1["sintetica"] == 0) & (s1["dataset_role"] != ROLE_PROJECTION)]
        RECORDER.check(np.allclose(real["tasa"], real["total_renewed_units"] / real["total_tr_units"]),
                       "real history rows: tasa = renewed / pipeline")
        row = summary.set_index("fs_id").loc["EU|0|0|web"]
        RECORDER.check(row["n_propio"] == 200 and row["huecos"] == 1 and row["meses_historia"] == 24,
                       "summary: n_propio = median monthly units (200), 1 gap, 24 history months")
        RECORDER.check(abs(row["tasa_propia"] - real["total_renewed_units"].sum() / real["total_tr_units"].sum()) < 1e-12,
                       "tasa_propia = Σ renewed / Σ pipeline over the real history")
        RECORDER.check(4.0 < row["error_binomial_pp"] < 5.5,
                       f"error_binomial_pp at n=200, p≈.8 is ≈ ±4.7 pp (found {row['error_binomial_pp']:.2f})")
        signs = summary.set_index("fs_id")["signo"]
        RECORDER.check(signs["EU|0|0|web"] == SIGN_NEUTRAL and signs["EU|1|0|web"] == SIGN_NEGATIVE
                       and signs["EU|0|1|web"] == SIGN_POSITIVE and signs["EU|1|1|web"] == SIGN_MIXED,
                       "sign per series: neutral / neg / pos / mixed")
        RECORDER.check(abs(row["usd_proyectado"] - 3 * 200 * 20) < 1e-9,
                       "usd_proyectado = the series' projected pipeline dollars (3 months × 200 × $20)")


def test_dimension_separation() -> None:
    RECORDER.start_block("1.2 · dimension_separation (η² factorial)")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, seasonal_series=True, trend_series=True)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
        decision, pairs = analysis_dimensions.dimension_separation(summary, units, configuration)
        by_dim = decision.set_index("dimension")
        RECORDER.check(set(by_dim.index) == {"region", "channel"}, "one row per non-timevarying dimension")
        RECORDER.check(by_dim.loc["channel", "contribucion_unica"] < by_dim.loc["region", "contribucion_unica"],
                       "region (EU .80 vs NA ≈ .74) contributes more than channel")
        RECORDER.check(by_dim.loc["region", "anulable"] == 0 and by_dim.loc["channel", "anulable"] == 1,
                       "mandatory dims are never annullable; extras are")
        RECORDER.check(((decision[["eta2_individual", "contribucion_unica", "omega2"]] >= 0)
                        & (decision[["eta2_individual", "contribucion_unica", "omega2"]] <= 1)).all().all(),
                       "every figure is inside [0, 1]")
        # a hand-built frame where B is redundant with A: unique contribution of B must be ≈ 0
        frame = pd.DataFrame({"A": list("xxyyzz") * 2, "B": list("ppqqrr") * 2,
                              "tasa_propia": [.2, .2, .5, .5, .8, .8] * 2, "n_propio": [50] * 12})
        r2_all = analysis_dimensions.weighted_r2_factorial(frame, ["A", "B"], "tasa_propia", "n_propio")
        r2_without_b = analysis_dimensions.weighted_r2_factorial(frame, ["A"], "tasa_propia", "n_propio")
        RECORDER.check(abs(r2_all - r2_without_b) < 1e-9 and r2_all > 0.99,
                       "a dimension redundant with another has zero unique contribution (type II)")
        RECORDER.check(abs(analysis_dimensions.weighted_eta2(frame, "B", "tasa_propia", "n_propio") - r2_all) < 1e-9,
                       "…while its individual η² is as high as the other's (the confounding)")
    RECORDER.start_block("1.2 · collapse order (sequential)")
    rng = np.random.default_rng(1)
    n = 600
    level_1 = rng.choice(["A", "B"], n)
    level_2 = np.array([f"{a}{rng.integers(0, 3)}" for a in level_1])
    level_3 = np.array([f"{b}{rng.integers(0, 2)}" for b in level_2])
    purchase = rng.choice(["new", "ren"], n)
    rate = 0.6 + 0.15 * (level_1 == "A") + 0.05 * (pd.Series(level_3).str[-1] == "1").to_numpy() + 0.2 * (purchase == "new") + rng.normal(0, .02, n)
    nested = pd.DataFrame(dict(regional_level_1=level_1, regional_level_2=level_2, regional_level_3=level_3, purchase_type=purchase,
                               tasa_propia=rate, n_propio=100.0))
    sequential = analysis_dimensions.sequential_collapse_order(nested, ["regional_level_1", "regional_level_2", "regional_level_3", "purchase_type"],
                                                               "tasa_propia", "n_propio")
    RECORDER.check([d for d, _ in sequential] == ["regional_level_3", "regional_level_2", "regional_level_1", "purchase_type"],
                   "nested hierarchy collapses finest first; the dim that separates most (purchase_type) collapses LAST")
    RECORDER.check(sequential[-1][1] > 0.5 and sequential[1][1] < 0.01,
                   "the sequential loss says what each collapse costs (level_2 ≈ 0 after level_3; purchase_type > .5)")
    RECORDER.start_block("1.2 · collapse order (legacy tie-break helper)")
    order = collapse_order(["a_level_1", "a_level_2", "b"], {"a_level_1": 0.01, "a_level_2": 0.9, "b": 0.05})
    RECORDER.check(order == ["b", "a_level_2", "a_level_1"],
                   "finest level of a family falls only after its deeper levels; families by unique contribution")


def test_mix_shift() -> None:
    RECORDER.start_block("1.4 · composition (mandatory-only cost + Kitagawa)")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
        counterfactual, decomposition = analysis_dimensions.counterfactual_and_decomposition(units, configuration)
        RECORDER.check(len(counterfactual) > 0 and set(counterfactual["celda_id"]) == {"EU"},
                       "one counterfactual row per cell × month of the window")
        pipeline_by_month = (units[units["tasa"].notna()].assign(mes=units["period"].astype(str))
                             .groupby("mes")["total_tr_usd"].sum())
        expected_saving = ((counterfactual["err_plano_pp"].abs() - counterfactual["err_seg_pp"].abs()) / 100
                           * counterfactual["mes"].map(pipeline_by_month))
        RECORDER.check(np.allclose(counterfactual["ahorro_usd"], expected_saving, atol=0.05),
                       "ahorro_usd = (|err_plano| − |err_seg|) × pipeline$ of the month")
        RECORDER.check(np.allclose(decomposition["delta_agregado_pp"],
                                   decomposition["delta_comportamiento_pp"] + decomposition["delta_composicion_pp"], atol=0.02),
                       "Kitagawa: Δ aggregate = behaviour term + composition term")
        # in this raw, weights are constant → composition term ≈ 0 every month
        RECORDER.check(decomposition["delta_composicion_pp"].abs().max() < 0.5,
                       "constant weights → composition term ≈ 0 (the movement is behaviour/sampling)")


def test_timevarying_calibration() -> None:
    RECORDER.start_block("1.2 · timevarying calibration")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        fine, labeled = phase_0_units(configuration)
        with quiet():
            units, summary = run_rate_series.build_rate_series(labeled, configuration)
        calibration = analysis_dimensions.timevarying_calibration(units, configuration)
        soft = calibration[(calibration["tipo"] == "flag") & (calibration["nombre"] == "softcancel")]
        RECORDER.check(len(soft) == 24 and 0.25 < soft["tasa_realizada"].mean() < 0.5,
                       "softcancel flag: 24 months, realized rate ≈ .35-.45 (S3, S4 and the mixed S6)")
        neg = calibration[(calibration["tipo"] == "signo") & (calibration["nombre"] == SIGN_NEGATIVE)]
        RECORDER.check(len(neg) == 24 and abs(neg["tasa_realizada"].mean() - 0.35) < 0.06,
                       "sign neg: only S3 and S4 (the mixed series is NOT counted as neg): ≈ .35")


def test_build_relatives() -> None:
    RECORDER.start_block("1.3 · build_relatives")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
    values = dict(region="EU", softcancel=1, autorenew=0, channel="web")
    signed = build_relatives(values, SIGN_NEGATIVE, configuration, "channel", ["region"])
    RECORDER.check([r[2] for r in signed] == ["EU|1|0|web", "EU|SIG=negativo|web", "EU|SIG=negativo|*", "EU|SIG=negativo|*"],
                   "signed series: itself → same sign → extra annulled → cell × sign; STOP (no mandatory collapse)")
    neutral = build_relatives(dict(region="EU", softcancel=0, autorenew=0, channel="web"), SIGN_NEUTRAL, configuration, "channel", ["region"])
    RECORDER.check([r[2] for r in neutral] == ["EU|0|0|web", "EU|SIG=neutro|*", "EU|SIG=neutro|*", "*|SIG=neutro|*"],
                   "neutral series: itself → extra annulled → cell (neutrals) → mandatory collapsed (neutrals)")
    RECORDER.check(all("SIG=neutro" in r[2] for r in neutral[1:]), "a neutral series never loses its 'neutral' sign")
    # with a loss cap, a signed series may collapse the mandatory dims that separate little, sign kept
    with tempfile.TemporaryDirectory() as folder:
        permissive = ladder_config(folder, signed_ladder_max_loss=0.05)
    climbing = build_relatives(values, SIGN_NEGATIVE, permissive, "channel", ["region"], collapse_loss={"region": 0.02})
    RECORDER.check([r[2] for r in climbing][-1] == "*|SIG=negativo|*" and "sign kept" in climbing[-1][1],
                   "signed_ladder_max_loss=0.05: 'region' (loses 0.02) collapses with the sign kept")
    stopped = build_relatives(values, SIGN_NEGATIVE, permissive, "channel", ["region"], collapse_loss={"region": 0.30})
    RECORDER.check([r[2] for r in stopped][-1] == "EU|SIG=negativo|*", "…but not when the loss (0.30) exceeds the cap")
    mixed = build_relatives(dict(region="EU", softcancel=1, autorenew=1, channel="web"), SIGN_MIXED, configuration, "channel", ["region"])
    RECORDER.check(len(mixed) == 1 and mixed[0][2] == "EU|1|1|web", "a mixed-sign series has no relative but itself")


def test_climb_and_estimate() -> None:
    RECORDER.start_block("1.3 · climb_ladder + estimate_rates (six series by eye)")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder)
        results = run_to_ladder(configuration)
        card = results["card"].set_index("fs_id")
        RECORDER.check(card.loc["EU|0|0|web", "peldano"] == 0 and card.loc["EU|0|0|web", "nivel_riesgo"] == "A_propio",
                       "S1 (n=200): its own rate, level A")
        RECORDER.check(card.loc["EU|0|0|tele", "peldano"] == 2 and card.loc["EU|0|0|tele", "id_estimacion"] == "EU|SIG=neutro|*"
                       and abs(card.loc["EU|0|0|tele", "n_efectivo"] - 210) < 1,
                       "S2 (n=10): channel annulled, pools WITH the big sibling S1 → n=210")
        s3_rung1 = results["ladder"][(results["ladder"]["fs_id"] == "EU|1|0|web") & (results["ladder"]["peldano"] == 1)].iloc[0]
        RECORDER.check(s3_rung1["padre_id"] == "EU|SIG=negativo|web" and s3_rung1["n_padre"] == 12 and s3_rung1["elegido"] == 0,
                       "S3 (soft=1 web, n=12): rung 1 'EU|SIG=negativo|web' is only itself (12 < 30) → not chosen, climbs")
        RECORDER.check(card.loc["EU|1|0|web", "peldano"] == 2 and card.loc["EU|1|0|web", "id_estimacion"] == "EU|SIG=negativo|*",
                       "S3 reaches rung 2 'EU|SIG=negativo|*' (channel annulled, sign kept)")
        RECORDER.check(card.loc["EU|1|0|tele", "id_estimacion"] == "EU|SIG=negativo|*" and abs(card.loc["EU|1|0|tele", "n_efectivo"] - 32) < 1,
                       "S4 reaches rung 2 'EU|SIG=negativo|*' = S3 + S4 = 32 ≥ 30 (never the neutral cell)")
        RECORDER.check(card.loc["EU|0|1|web", "nivel_riesgo"] == "S_signo_bajo_suelo" and card.loc["EU|0|1|web", "z"] == 1.0
                       and card.loc["EU|0|1|web", "id_estimacion"].startswith("EU|SIG=positivo"),
                       "S5 (autorenew, n=6): top of its sign ladder still < 30 → S level, no blend, stays in its sign")
        RECORDER.check(card.loc["EU|1|1|web", "nivel_riesgo"] == "M_signo_mixto" and card.loc["EU|1|1|web", "peldano"] == 0,
                       "S6 (mixed sign): alone, level M")
        # credibility arithmetic on S2: z = n/(n+k), rate = z·own + (1−z)·parent
        s2 = card.loc["EU|0|0|tele"]
        expected_rate = s2["z"] * s2["tasa_propia"] + (1 - s2["z"]) * s2["tasa_pariente"]
        RECORDER.check(abs(s2["tasa_estimada"] - expected_rate) < 1e-4 and abs(s2["z"] - 10 / (10 + s2["k"])) < 1e-3,
                       "S2: tasa_estimada = z·own + (1−z)·parent with z = n_propio/(n_propio + k)")
        RECORDER.check((card["se_prediccion_pp"].fillna(0) >= card["se_estimacion_pp"].fillna(0) - 1e-9).all(),
                       "se_prediccion ≥ se_estimacion for every series (the month still samples)")
        RECORDER.check(card.loc["EU|0|0|tele", "se_prediccion_pp"] > 10,
                       "S2 keeps a prediction error > 10 pp: n=10 stays noisy even with a good estimate")
        ladder = results["ladder"]
        RECORDER.check((ladder.groupby("fs_id")["elegido"].sum() == 1).all(), "exactly one chosen rung per series")
        signed = results["decision"][results["decision"]["signo"].isin([SIGN_NEGATIVE, SIGN_POSITIVE])]
        RECORDER.check((signed["peldano"] <= 3).all(), "no signed series climbs above its cell")
        report = run_support_ladder.risk_levels_report(results["card"], ROLE_TEST)
        RECORDER.check(abs(report["pct_usd"].sum() - 100) < 1e-6, "the money report sums to 100 %")
        RECORDER.check(set(report["nivel_riesgo"]) >= {"A_propio", "B_prestado", "S_signo_bajo_suelo", "M_signo_mixto"},
                       "the report shows levels A, B, S and M")


def test_own_rate_floor() -> None:
    RECORDER.start_block("1.3 · own_rate_floor (evidence vs precision)")
    with tempfile.TemporaryDirectory() as folder:
        configuration = ladder_config(folder, own_rate_floor=271.0)
        results = run_to_ladder(configuration)
        card = results["card"].set_index("fs_id")
        s1 = card.loc["EU|0|0|web"]
        RECORDER.check(s1["peldano"] == 2 and s1["id_estimacion"] == "EU|SIG=neutro|*" and 0 < s1["z"] < 1,
                       "S1 (n=200 < 271): climbs to its first relative with support and BLENDS (z between 0 and 1)")
        RECORDER.check(s1["nivel_riesgo"] == "A3_propio_reforzado", "S1 is level A3: own evidence, reinforced")
        RECORDER.check(abs(s1["z"] - 200 / (200 + s1["k"])) < 1e-3, "z = n/(n+k) with the series' own n")
        RECORDER.check(card.loc["EU|1|0|tele", "peldano"] == 2 and card.loc["EU|1|0|tele", "nivel_riesgo"] == "B_prestado",
                       "a series under the support floor behaves as before (borrows, level B)")
        big = ladder_config(folder, own_rate_floor=150.0)
        card_big = run_to_ladder(big)["card"].set_index("fs_id")
        RECORDER.check(card_big.loc["EU|0|0|web", "peldano"] == 0 and card_big.loc["EU|0|0|web", "z"] == 1.0
                       and card_big.loc["EU|0|0|web", "nivel_riesgo"] == "A_propio",
                       "with own_rate_floor=150, S1 (n=200) speaks alone: rung 0, z=1, level A")


ALL_TESTS = [test_rate_series, test_dimension_separation, test_mix_shift, test_timevarying_calibration,
             test_build_relatives, test_climb_and_estimate, test_own_rate_floor]


def main() -> int:
    print("═" * 74 + "\nTEST phase 1 · rate series, dimensions, support ladder\n" + "═" * 74)
    for test in ALL_TESTS:
        test()
    return RECORDER.print_panel("PHASE 1 TEST")


if __name__ == "__main__":
    sys.exit(main())
