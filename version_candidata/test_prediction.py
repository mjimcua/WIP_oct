"""
test_prediction.py — The single prediction module: the credibility shift, the band, the levels at an
origin, and that the forecast, the exam and the audit predict the same way.

    python test_prediction.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from prediction import (band_quantiles, credibility_shift, index_history, judged_horizon, levels_at_origins, months_known,
                        rate_band, shifted_rate)
from techniques import inverse_logit, logit
from test_helpers import check, console_of, finish, synthetic_with


def test_the_functions() -> None:
    print("A · the functions of prediction.py")
    shift = credibility_shift([0.25, 1.0, 0.5], [0.9, 0.9, np.nan], [0.8, 0.8, 0.8])
    check(abs(shift[0] - 0.75 * (logit(0.9) - logit(0.8))) < 1e-12 and shift[1] == 0 and shift[2] == 0,
          "the shift is (1 − z) × the difference of logit levels; 0 with z = 1 or a missing reference")
    rate, _ = shifted_rate([0.8], [0.0], [0.9], [0.8])
    check(abs(rate[0] - 0.9) < 1e-12, "with z = 0 the composition's prediction moves all the way to its reference level")
    check(np.all(credibility_shift([0.5], [0.9], [0.8], apply=False) == 0), "the shift can be switched off (apply_credibility_shift)")
    untouched, _ = shifted_rate([1.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan])
    check(list(untouched) == [1.0, 0.0], "without a shift the rate is the composition's prediction, untouched, also at exactly 1 or 0")
    low, high = rate_band([0.98, 0.5], [10, 100], np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    check(high[0] <= 1 and low[0] < 0.98 and abs((high[1] - 0.5) - 2 * np.sqrt(0.25 / 100)) < 1e-12,
          "the band: rate ± quantile × binomial error with the units due, never above 1")
    check(judged_horizon(1, [1, 6]) == 1 and judged_horizon(3, [1, 6]) == 6 and judged_horizon(9, [1, 6]) == 6,
          "a horizon takes the band of itself, the next judged one above, or the last")
    history = pd.DataFrame({"key": ["a"] * 3, "period": pd.period_range("2026-01", periods=3, freq="M"),
                            "due": [10, 10, 10], "renewed": [5, 8, 9]})
    levels = levels_at_origins(history, "key", pd.DataFrame({"key": ["a", "a"], "origin": [pd.Period("2026-01", "M"),
                                                                                         pd.Period("2026-02", "M")]}),
                               "period", "due", "renewed")
    check(list(levels["level"].round(4)) == [0.5, 0.65], "the level at an origin uses only the months up to it")


def test_one_way_of_predicting() -> None:
    print("B · the forecast, the exam and the audit predict the same way")
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    check("the chosen technique, applied to every series, gives the framework prediction of step 19" in console
          and "18 checks: 18 ok" in console.split("STEP AUD")[1],
          "the audit (from the backtest of step 14) and the exam (step 19) give the same prediction for every series")
    forecast = results["forecast"]["forecast"]
    traced = forecast.dropna(subset=["tasa_pool_h"])
    recomputed, _ = shifted_rate(traced["tasa_pool_h"], traced["z"], traced["ref_rate"], traced["group_rate"])
    check(np.allclose(recomputed, traced["tasa"]), "every forecast rate is its composition's prediction moved by prediction.py")


def test_the_lookups_at_scale() -> None:
    print("C · the lookups of the hot loops: the same values as the pandas way, on every path")
    configuration = synthetic_with()
    challenger = configuration.challenger_technique
    bands = pd.DataFrame({"tecnica": ["T2_mean", "T2_mean", challenger], "h": [1, 6, 1],
                          "q_low_norm": [-1.1, -2.2, -1.5], "q_high_norm": [1.3, 2.4, 1.6]})
    q_low, q_high = band_quantiles(bands, ["T2_mean", "T9_ses", "T9_ses", "own_level"], [1, 1, 6, 6], configuration)
    check(list(q_low) == [-1.1, -1.5, -configuration.z, -configuration.z] and list(q_high) == [1.3, 1.6, configuration.z, configuration.z],
          "band quantiles: the technique's own band; without one, the challenger's; without either, ±z")

    months = pd.period_range("2025-01", periods=6, freq="M")
    frame = pd.DataFrame({"key": ["b", "a", "b", "a", None, "b"], "period": [months[3], months[1], months[0], months[0], months[2], months[5]],
                          "due": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    index = index_history(frame, "key", "period", ["due"])
    origin = months[3]
    as_pandas = {key: block.sort_values("period")["due"].to_numpy() for key, block in frame[frame["period"] <= origin].groupby("key")}
    as_index = {key: entry["due"][:months_known(entry, origin)] for key, entry in index.items()}
    check(list(index) == ["a", "b"] and all(np.array_equal(as_pandas[key], as_index[key]) for key in as_pandas),
          "the history index: sorted keys, null keys out, and every prefix is what groupby + sort + the origin filter gave")
    try:
        index["a"]["due"][0] = 0.0
        written = True
    except ValueError:
        written = False
    check(not written, "the indexed histories are read-only: a technique cannot change the next origin's history")


if __name__ == "__main__":
    test_the_functions()
    test_one_way_of_predicting()
    test_the_lookups_at_scale()
    finish()
