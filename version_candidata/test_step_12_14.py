"""
test_step_12_14.py — Steps 12 and 14 on the synthetic: the pool series sum every matching
series; the backtest never peeks, chooses against the challenger, measures in the exam.

    python test_step_12_14.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from step_14_backtest import band_of_horizon
from techniques import CATALOGUE, inverse_logit, logit, predict_logit
from test_helpers import check, console_of, finish, synthetic_with


def run_everything():
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    return results, console, configuration


def test_the_techniques() -> None:
    print("A · the techniques")
    history = logit(np.array([0.8] * 24))
    months = np.array(list(range(1, 13)) * 2)
    check(all(abs(float(inverse_logit(predict_logit(technique_id, history, months, 1))) - 0.8) < 1e-9 for technique_id in CATALOGUE),
          "on a flat history every technique predicts the flat rate")
    rising = logit(np.linspace(0.5, 0.7, 24))
    check(inverse_logit(predict_logit("T12_theta", rising, months, 6)) > inverse_logit(predict_logit("T2_mean", rising, months, 6)),
          "on a rising history Theta (trend) predicts above the mean of the history")
    seasonal = logit(np.tile([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.9], 2))
    check(inverse_logit(predict_logit("T15_level_seasonal", seasonal, months, 12)) > 0.85
          and inverse_logit(predict_logit("T3_ma3", seasonal, months, 12)) < 0.8,
          "on a history with a December peak, level + month effect predicts the peak for December; ma3 does not")
    from techniques import eligible_techniques
    check("T11_holt_winters" not in eligible_techniques(23) and "T11_holt_winters" in eligible_techniques(24),
          "only the history limits a technique: Holt-Winters needs 24 months")
    check(band_of_horizon(1, {"corto": [1, 1], "medio_largo": [2, 6]}) == "corto"
          and band_of_horizon(9, {"corto": [1, 1], "medio_largo": [2, 6]}) == "medio_largo",
          "horizon 1 is corto; beyond the last band, the last band")


def test_pool_series_and_backtest() -> None:
    print("B · steps 12 and 14 on the synthetic")
    results, console, configuration = run_everything()
    pool_series, pool_reference = results["pool_series"], results["pool_reference"]
    own = pool_series[pool_series["composition_id"] == "EU|A|0|0|0|0|web"]
    units = results["rated_units"]
    history = units[(units["fs_id"] == "EU|A|0|0|0|0|web") & units["tasa"].notna()]
    check(own["renovadas"].sum() == history["total_renewed_units"].sum(),
          "the series of a rung-0 id is the series itself, month by month")
    check("5 checks: 5 ok" in console.split("STEP 12")[1] and "11 checks: 11 ok" in console.split("STEP 14")[1],
          "the checks of both steps pass")

    backtest = results["backtest"]
    predictions = backtest["predictions"]
    check((predictions["ultimo_mes_visto"] <= predictions["origen"]).all()
          and ((predictions["mes_objetivo"] - predictions["origen"]).map(lambda offset: offset.n) == predictions["h"]).all(),
          "every prediction saw only months up to target − h")
    check(set(predictions["proposito"]) == {"seleccion", "examen"}
          and predictions.loc[predictions["proposito"] == "examen", "mes_objetivo"].astype(str).isin(["2026-06", "2026-07", "2026-08"]).all(),
          "the exam targets are the exam months of the calendar")
    decision = backtest["decision"]
    unjudged = pool_reference.loc[pool_reference["gate"] == "soporte", "composition_id"]
    check((decision[decision["composition_id"].isin(unjudged)]["tecnica_origen"] == "sin_soporte").all(),
          "an id below the floor is not judged: it takes the challenger")
    champions = decision[decision["tecnica_origen"] == "campeon"]
    exam_rows = predictions[predictions["proposito"] == "seleccion"]
    one = champions.iloc[0]
    scores = (exam_rows[(exam_rows["composition_id"] == one["composition_id"]) & (exam_rows["tramo_h"] == one["tramo_h"])]
              .assign(abs_norm=lambda frame: frame["err_norm"].abs()).groupby("tecnica")["abs_norm"].mean())
    margin = configuration.challenger_margin_by_band[one["tramo_h"]]
    check(scores[one["tecnica"]] == scores.min() and scores[one["tecnica"]] < scores["T3_ma3"] - margin,
          "a champion has the lowest error of its id and band and beats the challenger by the margin")
    total = backtest["exam_total"]
    check(len(total) == 3 * len(configuration.backtest_horizons) and total["elegida_error_pct"].abs().max() < 0.2,
          "the exam of the total: one row per exam month and horizon")
    dimension = results["forecast_series"]
    check({"s14_tecnica_corto", "s14_tecnica_medio_largo", "s14_examen_err_pp_corto"} <= set(dimension.columns),
          "the forecast series dimension carries the chosen technique and the exam error of its composition")


if __name__ == "__main__":
    test_the_techniques()
    test_pool_series_and_backtest()
    finish()
