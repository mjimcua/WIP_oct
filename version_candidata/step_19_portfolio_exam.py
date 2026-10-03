"""
step_19_portfolio_exam.py — The exam of the whole portfolio: how well does the framework
predict the renewals of every series, and how does it compare with the spreadsheet?

Step 14 judged every POOL on its own series. But pools overlap (a series belongs to its own
pool and to the pools of its relatives), so summing pools is not the portfolio. This step
examines the portfolio as the forecast predicts it: SERIES BY SERIES, in the exam months,
with only what was known h months before.

HOW THE TEST WORKS:
  · for every exam month T and horizon h (origin T − h), every series with renewals in T is
    predicted as step 17 would have predicted it at the origin:
        its estimation id's technique (step 14) on the id's months up to T − h, shifted by
        moved toward its credibility reference by (1 − z) × (reference level − group level),
        both levels measured up to T − h;
        no estimation id: the rate of its mandatory cell up to T − h (then the global one)
    predicted renewed units = predicted rate × its units due in T (the real pipeline)
  · THE SPREADSHEET: what the business does today — for every grain of baseline_grains, the
    rate of the last baseline_months closed months up to T − h (Σ renewed / Σ due), times the
    units due in T. Same series, same months, same horizons.
  · TWO ERRORS for every method: of the TOTAL (Σ predicted − Σ real) / Σ real, and BY SERIES
    (WAPE: Σ |predicted − real| over series / Σ real: a total can be right by compensation).
  · Everything in units: this exam judges the RATE; the price is step 16.

Actions (logged as they are done):
  1. the calendar of the exam and the series with renewals in it
  2. the framework's prediction of every series, month and horizon
  3. the spreadsheet's prediction for every grain
  4. the errors of every method, by month and on average
  5. check the exam                                                  checks 1-3
  6. write the exam                                                  checks 4-5
  7. count the checks; stop if any failed
  8. show the framework against the spreadsheet, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. no prediction used a month after its origin
   2. every method predicted every series of every exam month
   3. the real renewals of the exam equal the renewals of the exam months in the units
   4-5. tables sff_examen_cartera and sff_examen_cartera_resumen written and read back

Output: (the exam by month and horizon, the summary by method) · two tables.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from step_14_backtest import band_of_horizon
from techniques import inverse_logit, logit, predict_logit
from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM, TABLE_PORTFOLIO_EXAM_SUMMARY,
                         TRUTH_ROLES)


STEP_LABEL = "19"
STEP_NAME = "EXAM OF THE PORTFOLIO"
STEP_PURPOSE = ("examine the whole portfolio series by series, as the forecast predicts it, in the exam months with only "
                "what was known h months before; and compare it, on the same series and months, with the spreadsheet "
                "the business uses today (the rate of the last months per grain)")
STEP_ACTIONS = ["the calendar of the exam and the series with renewals in it",
                "the framework's prediction of every series, month and horizon",
                "the spreadsheet's prediction for every grain",
                "the errors of every method, by month and on average",
                "check the exam (checks 1-3)",
                "write the exam (checks 4-5)",
                "count the checks; stop if any failed",
                "show the framework against the spreadsheet, as tables"]
STEP_OUTPUT = "real vs predicted renewals by exam month, horizon and method · the mean errors · two tables"


def examine_portfolio(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                      backtest: dict, configuration: Config, reference_members: pd.DataFrame = None) -> tuple:
    """The framework and the spreadsheet predicting every series in the exam months."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    units_due, renewed = configuration.pipeline_units_col, configuration.renewed_units_col

    # [1] the exam: the real rows of the exam months
    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & (rated_units[CALENDAR_ROLE_COLUMN] == ROLE_TEST)
                       & (rated_units[units_due] > 0)].copy()
    exam_months = sorted(real[period_column].unique())
    configuration.log_action(STEP_LABEL, 1, f"exam months {exam_months[0]}..{exam_months[-1]} · {len(real):,} series-months · "
                                            f"horizons {configuration.backtest_horizons} · spreadsheet: last "
                                            f"{configuration.baseline_months} months per grain {configuration.baseline_grains}")

    closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                         & (rated_units[units_due] > 0)].copy()
    closed["_celda"] = join_columns(closed, configuration.business_mandatory_dims)
    real["_celda"] = join_columns(real, configuration.business_mandatory_dims)
    real = real.merge(series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "credibility_ref_id"]],
                      on=SERIES_ID_COLUMN, how="left")
    decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
    pool_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
    pool_ids = set(pool_truth[ESTIMATION_ID_COLUMN])

    rows, latest_month_used = [], []
    for target_month in exam_months:
        for horizon in configuration.backtest_horizons:
            origin = target_month - horizon
            known = closed[closed[period_column] <= origin]
            latest_month_used.append((known[period_column].max(), origin))
            month_rows = real[real[period_column] == target_month].copy()

            # [2] the framework
            month_rows["pred_" + METHOD_FRAMEWORK] = framework_rates(month_rows, known, pool_truth, pool_ids, decision,
                                                                    origin, horizon, configuration,
                                                                    reference_members) * month_rows[units_due]
            # [3] the spreadsheet, for every grain
            for grain in configuration.baseline_grains:
                month_rows["pred_hoja_" + grain] = spreadsheet_rates(month_rows, known, grain, origin, configuration) * month_rows[units_due]
            month_rows["h"] = horizon
            rows.append(month_rows)
    predictions = pd.concat(rows, ignore_index=True)
    methods = [METHOD_FRAMEWORK] + ["hoja_" + grain for grain in configuration.baseline_grains]
    configuration.log_action(STEP_LABEL, 2, f"framework: {len(predictions):,} series-month-horizon predictions")
    configuration.log_action(STEP_LABEL, 3, f"spreadsheet: {len(configuration.baseline_grains)} grains")

    # [4] the errors
    by_month, summary = exam_errors(predictions, methods, configuration)
    best = summary.sort_values("error_total_medio").iloc[0]
    configuration.log_action(STEP_LABEL, 4, f"best mean error of the total: {best['metodo']} (h = {best['h']}) "
                                            f"{best['error_total_medio']:.1%}")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the exam")
    peeked = [(used, origin) for used, origin in latest_month_used if pd.notna(used) and used > origin]
    configuration.log_check(STEP_LABEL, check_log, "no prediction used a month after its origin", not peeked,
                            failure_detail=f"{len(peeked)} (month, horizon) used later months",
                            context=f"{len(latest_month_used)} month × horizon exams")
    missing = predictions[["pred_" + method for method in methods]].isna().any(axis=1)
    configuration.log_check(STEP_LABEL, check_log, "every method predicted every series of every exam month", not missing.any(),
                            failure_detail=f"{int(missing.sum()):,} predictions missing",
                            context=f"{len(methods)} methods")
    real_total = real[renewed].sum() * len(configuration.backtest_horizons)
    configuration.log_check(STEP_LABEL, check_log, "the real renewals of the exam equal the renewals of the exam months",
                            abs(predictions[renewed].sum() - real_total) < 1e-6,
                            failure_detail="the exam lost or duplicated renewals",
                            context=f"{real[renewed].sum():,.0f} renewed units in the exam months")

    # [6] the tables
    configuration.log_action(STEP_LABEL, 6, "writing the exam")
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_PORTFOLIO_EXAM)
    configuration.write_table(STEP_LABEL, check_log, summary, TABLE_PORTFOLIO_EXAM_SUMMARY)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] the framework against the spreadsheet
    configuration.log_action(STEP_LABEL, 8, "the framework against the spreadsheet (error_total: of the sum of the portfolio; "
                                            "wape_series: series by series, no compensation):")
    configuration.show_table(summary)
    configuration.logger.doc(f"[{STEP_LABEL}] by exam month:")
    configuration.show_table(by_month)
    return by_month, summary


def framework_rates(month_rows: pd.DataFrame, known: pd.DataFrame, pool_truth: pd.DataFrame, pool_ids: set,
                    decision: pd.Series, origin, horizon: int, configuration: Config,
                    reference_members: pd.DataFrame = None) -> np.ndarray:
    """The rate the framework would have predicted at the origin for every series of the month."""
    band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
    period_column = configuration.period_col
    units_due, renewed = configuration.pipeline_units_col, configuration.renewed_units_col
    pool_known = pool_truth[pool_truth[period_column] <= origin]
    # the level of every credibility reference with what was known at the origin (all its series)
    reference_level = pd.Series(dtype=float)
    if reference_members is not None and len(reference_members):
        known_by_reference = known.merge(reference_members, on=SERIES_ID_COLUMN)
        grouped = known_by_reference.groupby("credibility_ref_id")
        reference_level = grouped[renewed].sum() / grouped[units_due].sum()
    cell_level = known.groupby("_celda")[renewed].sum() / known.groupby("_celda")[units_due].sum()
    global_level = known[renewed].sum() / known[units_due].sum()
    predicted_pool, pool_level = {}, {}
    for estimation_id in set(month_rows[ESTIMATION_ID_COLUMN].dropna()) & pool_ids:
        history = pool_known[pool_known[ESTIMATION_ID_COLUMN] == estimation_id].sort_values(period_column)
        if len(history) < 3:
            continue
        technique = decision.get((estimation_id, band_name), configuration.challenger_technique)
        rates = history[RATE_COLUMN].to_numpy(dtype=float)
        months = np.array([month.month for month in history[period_column]])
        value = predict_logit(technique, logit(rates), months, int(horizon))
        if not np.isfinite(value):
            value = predict_logit(configuration.challenger_technique, logit(rates), months, int(horizon))
        predicted_pool[estimation_id] = float(inverse_logit(value))
        pool_level[estimation_id] = history["renovadas"].sum() / history["vencen"].sum()
    rates = []
    for _, row in month_rows.iterrows():
        estimation_id = row[ESTIMATION_ID_COLUMN]
        if estimation_id in predicted_pool:
            rate = predicted_pool[estimation_id]
            reference = reference_level.get(row["credibility_ref_id"], np.nan) if pd.notna(row["credibility_ref_id"]) else np.nan
            if configuration.apply_credibility_shift and row["z"] < 1 and np.isfinite(reference) and 0 < reference < 1:
                rate = float(inverse_logit(logit(rate) + (1 - row["z"]) * (logit(reference) - logit(pool_level[estimation_id]))))
        else:
            rate = cell_level.get(row["_celda"], global_level)
        rates.append(rate)
    return np.array(rates, dtype=float)


def spreadsheet_rates(month_rows: pd.DataFrame, known: pd.DataFrame, grain: str, origin, configuration: Config) -> np.ndarray:
    """The rate of the last baseline_months closed months up to the origin, per grain (the global rate if the grain has none)."""
    period_column = configuration.period_col
    units_due, renewed = configuration.pipeline_units_col, configuration.renewed_units_col
    window = known[known[period_column] > origin - configuration.baseline_months]
    global_rate = window[renewed].sum() / window[units_due].sum()
    if grain == "global":
        return np.full(len(month_rows), global_rate)
    columns = configuration.business_mandatory_dims if grain == "mandatory" else grain.split("+")
    window_keys = join_columns(window, columns)
    grain_rate = window.groupby(window_keys)[renewed].sum() / window.groupby(window_keys)[units_due].sum()
    return join_columns(month_rows, columns).map(grain_rate).fillna(global_rate).to_numpy(dtype=float)


def exam_errors(predictions: pd.DataFrame, methods: list, configuration: Config) -> tuple:
    """By month and horizon: real, predicted and error of the total per method; the summary."""
    renewed = configuration.renewed_units_col
    month_rows, summary_rows = [], []
    for (target_month, horizon), block in predictions.groupby([configuration.period_col, "h"]):
        row = {"mes": target_month, "h": horizon, "renovadas_reales": block[renewed].sum()}
        for method in methods:
            predicted = block["pred_" + method]
            row[f"{method}_pred"] = predicted.sum()
            row[f"{method}_error_total"] = (predicted.sum() - block[renewed].sum()) / block[renewed].sum()
        month_rows.append(row)
    by_month = pd.DataFrame(month_rows)
    for horizon, block in predictions.groupby("h"):
        for method in methods:
            month_errors = by_month.loc[by_month["h"] == horizon, f"{method}_error_total"]
            summary_rows.append({"metodo": method, "h": horizon,
                                 "error_total_medio": float(month_errors.abs().mean()),
                                 "sesgo_total_medio": float(month_errors.mean()),
                                 "wape_series": float((block["pred_" + method] - block[renewed]).abs().sum() / block[renewed].sum())})
    return by_month, pd.DataFrame(summary_rows).sort_values(["h", "error_total_medio"]).reset_index(drop=True)
