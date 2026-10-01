"""
step_18_validation.py — The final validation: does everything agree, from the raw to the forecast?

Each step checked its own work. This step checks the chain ACROSS steps, once everything
has run, and leaves the result in one table:
  · the money: Σ USD due of the extract = Σ of the fine table = Σ of the forecast units;
    the future USD due of the extract = Σ USD due of the forecast
  · every series with money to predict has a rate and a risk level
  · every future row takes its rate from its series' estimation id when it has one
  · the forecast is coherent with the past: the expected rate of the future (USD) is within
    ±15 pp of the last closed year's (warning: a larger jump must be explained)
  · the precision measured in the exam of the portfolio (step 19, series by series): the
    error of the total renewals is within ±10 % in every exam month (warning: the forecast is
    less reliable than promised)

Actions (logged as they are done):
  1. the money along the chain
  2. the coverage of the series and the forecast rows
  3. the coherence of the forecast with the past and with the exam
  4. write the validation                                            check 7
  5. count the checks; stop if any failed

Checks (logged as they are made, numbered, at the level of their status):
   1. Σ USD due: extract = fine table = forecast units
   2. the future USD due of the extract = Σ USD due of the forecast
   3. every series with money to predict has a rate and a risk level
   4. every future row of a series with an estimation id takes its rate from the pool
   5. the expected rate of the future is within ±15 pp of the last closed year's   (warning only)
   6. the error of the total renewals in the exam is within ±10 %                   (warning only)
   7. table sff_validacion written and read back

Output: the validation (one row per check) · table sff_validacion.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL,
                         RATE_FROM_POOL, ROLE_PROJECTION, S0_PIPELINE_USD_COLUMN, SERIES_ID_COLUMN, TABLE_VALIDATION,
                         TRUTH_ROLES)


STEP_LABEL = "18"
STEP_NAME = "VALIDATION"
STEP_PURPOSE = ("check, once everything has run, that the chain agrees from the raw to the forecast: the money is "
                "conserved, every series and future row is covered, and the forecast is coherent with the past and "
                "with the precision measured in the exam")
STEP_ACTIONS = ["the money along the chain",
                "the coverage of the series and the forecast rows",
                "the coherence of the forecast with the past and with the exam",
                "write the validation (check 7)",
                "count the checks; stop if any failed"]
STEP_OUTPUT = "one row per check · table sff_validacion"

MONEY_TOLERANCE = 0.01
MAX_RATE_JUMP_PP = 15.0
MAX_EXAM_TOTAL_ERROR = 0.10


def validate_chain(raw: pd.DataFrame, results: dict, configuration: Config) -> pd.DataFrame:
    """The checks across steps; the table of the validation."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    fine, units, forecast = results["fine_table"], results["forecast_units"], results["forecast"]["forecast"]
    due = configuration.pipeline_usd_col

    # [1] the money
    configuration.log_action(STEP_LABEL, 1, "the money along the chain")
    time_series_due = results["time_series_rows"][due].fillna(0).sum() if results.get("time_series_rows") is not None else 0.0
    wiped = fine[S0_PIPELINE_USD_COLUMN].sum() - fine[due].sum()               # step 02: the pipeline not known yet
    totals = {"extracto": raw[due].sum() - time_series_due - wiped, "tabla_fina": fine[due].sum(), "forecast_units": units[due].sum()}
    configuration.log_check(STEP_LABEL, check_log, "Σ USD due: extract (without time_series and the pipeline not known yet) = fine table = forecast units",
                            max(totals.values()) - min(totals.values()) <= MONEY_TOLERANCE,
                            failure_detail=f"totals differ: {totals}", context=f"${totals['extracto']:,.0f}")
    future_due = fine.loc[fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION, due].sum()
    configuration.log_check(STEP_LABEL, check_log, "the future USD due of the extract = Σ USD due of the forecast's extract rows",
                            abs(future_due - forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum()) <= MONEY_TOLERANCE,
                            failure_detail=f"${future_due:,.0f} vs ${forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum():,.0f}",
                            context=f"${future_due:,.0f} (the extended horizon adds its own pipeline)")

    # [2] coverage
    configuration.log_action(STEP_LABEL, 2, "the coverage of the series and the forecast rows")
    estimate = results["series_estimate"]
    with_money = estimate[estimate["usd_por_predecir"] > 0]
    uncovered = with_money[with_money["nivel_riesgo"].isna()]
    forecast_series = set(forecast[SERIES_ID_COLUMN])
    configuration.log_check(STEP_LABEL, check_log, "every series with money to predict has a rate and a risk level",
                            uncovered.empty and set(with_money[SERIES_ID_COLUMN]) <= forecast_series,
                            failure_detail=f"{len(uncovered)} series without level", context=f"{len(with_money):,} series")
    pooled_ids = set(results["pool_reference"][ESTIMATION_ID_COLUMN])
    should_pool = forecast[ESTIMATION_ID_COLUMN].isin(pooled_ids)
    configuration.log_check(STEP_LABEL, check_log, "every future row of a series with an estimation id takes its rate from the pool",
                            bool((forecast.loc[should_pool, "origen_tasa"] == RATE_FROM_POOL).all()),
                            failure_detail="rows with a pool that took another rate",
                            context=f"{int(should_pool.sum()):,} of {len(forecast):,} rows from a pool")

    # [3] coherence
    configuration.log_action(STEP_LABEL, 3, "the coherence of the forecast with the past and with the exam")
    closed = fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    last_year = closed[configuration.period_col].max().year
    last_year_rows = closed[closed[configuration.period_col].map(lambda month: month.year) == last_year]
    past_rate = last_year_rows[configuration.renewed_usd_col].sum() / last_year_rows[due].sum()
    extract_rows = forecast[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL]
    future_rate = extract_rows["esperado_usd"].sum() / extract_rows[due].sum()
    configuration.log_check(STEP_LABEL, check_log, f"the expected rate of the future is within ±{MAX_RATE_JUMP_PP:.0f} pp of {last_year}'s",
                            abs(future_rate - past_rate) * 100 <= MAX_RATE_JUMP_PP,
                            failure_detail=f"future {future_rate:.1%} vs {last_year} {past_rate:.1%}: explain the jump",
                            context=f"future {future_rate:.1%} (USD, uplift included) vs {last_year} {past_rate:.1%}", blocking=False)
    exam = results.get("portfolio_exam")
    exam_error = (exam["framework_error_total"].abs().max() if exam is not None
                  else results["backtest"]["exam_total"]["elegida_error_pct"].abs().max())
    configuration.log_check(STEP_LABEL, check_log, f"the error of the total renewals in the exam is within ±{MAX_EXAM_TOTAL_ERROR:.0%}",
                            exam_error <= MAX_EXAM_TOTAL_ERROR,
                            failure_detail=f"the worst month misses by {exam_error:.1%}", context=f"worst month {exam_error:.1%}",
                            blocking=False)

    # [4] the table
    configuration.log_action(STEP_LABEL, 4, "writing the validation")
    validation = pd.DataFrame(check_log, columns=["estado", "comprobacion", "detalle"])
    configuration.write_table(STEP_LABEL, check_log, validation, TABLE_VALIDATION)

    # [5] the count of the checks
    configuration.log_action(STEP_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
    return validation
