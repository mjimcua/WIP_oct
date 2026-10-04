"""
test_step_22.py — The adaptation to Power BI: the price increase monitor.

A · the monitor over the synthetic: columns, the exact base, no false flags
B · the detection over a hand-made core: a step is flagged and its cycle lasts CYCLE_MONTHS
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_22_power_bi import (CYCLE_MONTHS, MIN_REFERENCE_MONTHS, PERSISTENCE_MONTHS, PRICE_STEP_THRESHOLD,
                               price_increase_monitor)
from test_helpers import check, console_of, finish, synthetic_with
from main import run
from config import (CORE_ISOLATED_BAND_PREFIX, CORE_ISOLATED_RENEWED_PIPELINE_USD, CORE_ISOLATED_RENEWED_UNITS,
                    CORE_ISOLATED_RENEWED_USD, CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR, ISOLATED_RATIO_BANDS)


def test_the_monitor_over_the_synthetic() -> None:
    print("A · the monitor over the synthetic: columns, the exact base, no false flags")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))
    monitor = results["power_bi"]["price_monitor"]
    expected_columns = {"period", "uplift", "uplift_isolated", "share_due_isolated",
                        "rate_units_isolated", "uplift_previous_12", "uplift_step", "price_increase_flag",
                        "in_increase_cycle", "months_since_increase"}
    check(expected_columns <= set(monitor.columns), "one row per closed pipeline month with the detector columns")
    check(int(monitor["price_increase_flag"].sum()) == 0 and int(monitor["in_increase_cycle"].sum()) == 0,
          "the synthetic has no price increase: nothing is flagged (no false positives)")

    core = results["core"]
    closed = core[(core["forecast_status"] == "actual") & (core["forecast_universe"] == "pipeline")
                  & (core["forecast_to_renew_units"] > 0)]
    one_month = monitor["period"].iloc[-1]
    of_the_month = closed[closed["period"].astype(str) == str(one_month)]
    by_hand = of_the_month["forecast_renewed_USD"].sum() / of_the_month["forecast_to_renew_USD_renewed"].sum()
    check(abs(float(monitor.loc[monitor["period"] == one_month, "uplift"].iloc[0]) - by_hand) < 1e-9,
          "the uplift of a month = Σ renewed USD / Σ exact base, summed over the closed pipeline rows of the core")


def test_the_detection_over_a_hand_made_core() -> None:
    print("B · the detection: a +5 % step is flagged and its cycle lasts exactly CYCLE_MONTHS")
    months = pd.period_range("2023-01", periods=36, freq="M")
    step_starts = pd.Period("2025-01", freq="M")                     # month 25: well past the reference window
    rows = []
    for month in months:
        uplift = 1.10 if month < step_starts else 1.155              # a 5 % step that stays
        rows.append({"forecast_status": "actual", "forecast_universe": "pipeline", "period": month,
                     "forecast_to_renew_units": 100.0, "forecast_to_renew_USD": 1000.0,
                     "forecast_renewed_units": 70.0, "forecast_renewed_USD": 70.0 * 10.0 * uplift,
                     "forecast_to_renew_USD_renewed": 700.0})
    monitor = price_increase_monitor(pd.DataFrame(rows), synthetic_with())
    check(monitor.attrs["detector_series"] == "uplift" and monitor["uplift_isolated"].isna().all(),
          "without the isolated renewals the detector falls back to the plain uplift, and says so")
    check(monitor["uplift_previous_12"].isna().sum() == MIN_REFERENCE_MONTHS,
          "the first months have no reference yet (fewer than the minimum of previous months)")
    first_flagged = monitor[monitor["price_increase_flag"] == 1]["period"].iloc[0]
    check(first_flagged == step_starts, "the step is flagged in its first month")
    in_cycle = monitor[monitor["in_increase_cycle"] == 1]["period"]
    check(len(in_cycle) == CYCLE_MONTHS and in_cycle.iloc[0] == step_starts,
          f"the cycle lasts exactly {CYCLE_MONTHS} months from the flagged step")
    after_cycle = monitor[monitor["period"] >= step_starts + CYCLE_MONTHS]
    check((after_cycle["in_increase_cycle"] == 0).all(),
          "one cycle later everyone due already bought at the new tariff: the cycle is over")
    check(float(monitor.loc[monitor["period"] == step_starts, "uplift_step"].round(3).iloc[0]) == 0.05,
          "the step measures the size of the increase (+5 %) against the previous 12 months")

    noisy = pd.DataFrame(rows)
    spike_month = pd.Period("2024-06", freq="M")
    noisy.loc[noisy["period"] == spike_month, "forecast_renewed_USD"] *= 1.06     # ONE month 6 % up, then back
    noisy_monitor = price_increase_monitor(noisy, synthetic_with())
    spike = noisy_monitor[noisy_monitor["period"] == spike_month].iloc[0]
    check(spike["above_threshold"] == 1 and spike["price_increase_flag"] == 0
          and noisy_monitor.loc[noisy_monitor["period"] < step_starts, "in_increase_cycle"].sum() == 0,
          f"a single month above the threshold is not an increase (it needs {PERSISTENCE_MONTHS} in a row): no false cycle")


def test_the_dispersion_of_the_isolated_renewals() -> None:
    print("D · the dispersion: the std deviation and the shares, licence by licence, from the sums")
    rows = []
    for licence_ratios in ([1.00, 1.20], [1.10, 1.10]):                    # same mean 1.10, different spread
        base = 100.0 * len(licence_ratios)
        renewed = sum(100.0 * ratio for ratio in licence_ratios)
        second_moment = sum((100.0 * ratio) ** 2 / 100.0 for ratio in licence_ratios)
        bands = {short_name: 0.0 for _, short_name, _, _ in ISOLATED_RATIO_BANDS}
        for ratio in licence_ratios:
            for _, short_name, low, high in ISOLATED_RATIO_BANDS:
                if low <= ratio < high:
                    bands[short_name] += 100.0
        month = pd.Period("2026-01", freq="M") + len(rows)
        row = {"forecast_status": "actual", "forecast_universe": "pipeline", "period": month,
               "forecast_to_renew_units": 10.0, "forecast_to_renew_USD": 1000.0, "forecast_renewed_units": 2.0,
               "forecast_renewed_USD": renewed, CORE_ISOLATED_RENEWED_USD: renewed,
               CORE_ISOLATED_RENEWED_PIPELINE_USD: base, CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR: second_moment}
        row.update({CORE_ISOLATED_BAND_PREFIX + short_name: value for short_name, value in bands.items()})
        rows.append(row)
    monitor = price_increase_monitor(pd.DataFrame(rows), synthetic_with())
    check([round(value, 6) for value in monitor["desv_isolated"]] == [0.1, 0.0],
          "two licences at 1.00 and 1.20: std 0.10; two at 1.10: std 0 (same mean, the moment tells them apart)")
    check([round(value, 6) for value in monitor["share_isolated_ge110"]] == [0.5, 1.0]
          and [round(value, 6) for value in monitor["share_isolated_near_100"]] == [0.5, 0.0],
          "the shares near 1 and at or above 1.10 come from the bands, weighted by what the licences were worth")


def test_each_isolated_metric_needs_only_its_columns() -> None:
    print("C · without the isolated units due, the detector still runs: only the rate and the share go blank")
    months = pd.period_range("2024-01", periods=20, freq="M")
    rows = [{"forecast_status": "actual", "forecast_universe": "pipeline", "period": month,
             "forecast_to_renew_units": 100.0, "forecast_to_renew_USD": 1000.0,
             "forecast_renewed_units": 70.0, "forecast_renewed_USD": 735.0,
             CORE_ISOLATED_RENEWED_UNITS: 60.0, CORE_ISOLATED_RENEWED_USD: 618.0,
             CORE_ISOLATED_RENEWED_PIPELINE_USD: 600.0} for month in months]           # no isolated units due
    monitor = price_increase_monitor(pd.DataFrame(rows), synthetic_with())
    check(monitor.attrs["detector_series"] == "uplift_isolated" and (monitor["uplift_isolated"].round(6) == 1.03).all(),
          "the isolated uplift = Σ isolated renewed USD / Σ its exact base, and the detector runs on it")
    check(monitor["rate_units_isolated"].isna().all() and monitor["share_due_isolated"].isna().all(),
          "the metrics that need the isolated units due stay empty, and nothing else does")


if __name__ == "__main__":
    test_the_monitor_over_the_synthetic()
    test_the_detection_over_a_hand_made_core()
    test_each_isolated_metric_needs_only_its_columns()
    test_the_dispersion_of_the_isolated_renewals()
    finish()
