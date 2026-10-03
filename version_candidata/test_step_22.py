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


def test_the_monitor_over_the_synthetic() -> None:
    print("A · the monitor over the synthetic: columns, the exact base, no false flags")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))
    monitor = results["power_bi"]["price_monitor"]
    expected_columns = {"period", "uplift", "uplift_never_softcancel", "share_due_never_softcancel",
                        "rate_units_never_softcancel", "uplift_previous_12", "uplift_step", "price_increase_flag",
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
    check(monitor.attrs["detector_series"] == "uplift" and monitor["uplift_never_softcancel"].isna().all(),
          "without the never-softcancel columns the detector falls back to the plain uplift, and says so")
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


if __name__ == "__main__":
    test_the_monitor_over_the_synthetic()
    test_the_detection_over_a_hand_made_core()
    finish()
