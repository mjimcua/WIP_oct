"""
step_19_series_exam.py — The exam of every forecast series: how much better the framework
predicts than the forecast series alone (raw) and than the spreadsheet the business uses.

In the exam months, with only what was known h months before (origin T − h), every forecast series
with something due is predicted by three methods, on the same rows:

  raw          the forecast series alone, with its own history (no ladder): the technique of its
               composition on its own monthly rates (the challenger if it cannot; its own level if it
               has fewer than 3 months; its mandatory cell if it has none)
  framework    as the forecast predicts it (prediction.py): the technique of its composition on the
               composition's months, moved toward its credibility reference by (1 − z) × the difference
               of levels known at the origin; a series with no composition: its mandatory cell
  hoja_<grain> the spreadsheet: the rate of the last baseline_months closed months per grain

Every prediction keeps its steps (composition rate, shift, series rate) and, for raw and framework,
its INTERVAL, built as the forecast builds its band (prediction.py): the error quantiles of the
technique × the binomial error with the series' own units due. A prediction is IN THE INTERVAL when
the real rate falls inside it. Predicted units = rate × the series' own units due.

Outputs:
  sff_series_exam_detail      forecast series × exam month × h × method: every prediction, traced
  sff_series_exam             forecast series: per method, predictions, in the interval, units
                              predicted and real, error, coverage (the improvement over the raw)
  sff_examen_cartera(_resumen) the portfolio: the error of the total per exam month and method

Actions (logged as they are done):
  1. the exam: the calendar, the forecast series with something due in the exam months
  2. the three methods for every series, month and horizon (with the interval of raw and framework)
  3. the errors: per forecast series, and of the total per month
  4. check the exam                                               checks 1-4
  5. write the exam                                               checks 5-8
  6. count the checks; stop if any failed
  7. show the methods against each other, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. no prediction used a month after its origin
   2. every method predicted every series of every exam month
   3. the real renewals of the exam equal the renewals of the exam months (nothing counted twice)
   4. the summary per series adds up to the detail (counts and units)
   5-8. tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen written
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from prediction import band_quantiles, levels_at_origins, predict_composition, rate_band, shifted_rate
from step_14_backtest import band_of_horizon
from techniques import CATALOGUE, logit
from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM, TABLE_PORTFOLIO_EXAM_SUMMARY,
                         TABLE_SERIES_EXAM, TABLE_SERIES_EXAM_DETAIL, TRUTH_ROLES)


STEP_LABEL = "19"
STEP_NAME = "EXAM OF EVERY FORECAST SERIES"
STEP_PURPOSE = ("predict every forecast series in the exam months, with only what was known h months before, three ways "
                "on the same rows: alone with its own history (raw), as the framework does (composition and credibility) "
                "and as the spreadsheet does; every prediction traced and with its interval, so the improvement over the "
                "raw and over the spreadsheet can be measured series by series and summed")
STEP_ACTIONS = ["the exam: the calendar, the forecast series with something due in the exam months",
                "the three methods for every series, month and horizon (with the interval of raw and framework)",
                "the errors: per forecast series, and of the total per month",
                "check the exam (checks 1-4)",
                "write the exam (checks 5-8)",
                "count the checks; stop if any failed",
                "show the methods against each other, as tables"]
STEP_OUTPUT = ("every prediction of every forecast series, traced · the exam per series and method · the error of the total "
               "per month · tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen")

METHOD_RAW = "raw"
MIN_MONTHS_TO_PREDICT = 3          # fewer own months: the raw method takes its own level
UNITS_TOLERANCE = 1e-6


def examine_series(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                   backtest: dict, configuration: Config, reference_members: pd.DataFrame = None) -> dict:
    """The exam of every forecast series by the three methods.

    INPUT:   rated_units (step 08) · series_estimate (11: composition, z, reference) · the composition series
             (12) · the backtest (14: decision and error bands) · the members of every credibility reference (10).
    OUTPUT:  dict(detail, per_series, by_month, summary).
    RULES:   see the module header.
    EDGE CASES: a series with no composition (solo_historia, mixed) is predicted by its mandatory cell in the
             framework; its raw method still uses its own history. The spreadsheet has no interval.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period = configuration.period_col
    due, renewed = configuration.pipeline_units_col, configuration.renewed_units_col

    # [1] the exam
    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & (rated_units[CALENDAR_ROLE_COLUMN] == ROLE_TEST)
                       & (rated_units[due] > 0)].copy()
    closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                         & (rated_units[due] > 0)].copy()
    closed["_cell"] = join_columns(closed, configuration.business_mandatory_dims)
    real["_cell"] = join_columns(real, configuration.business_mandatory_dims)
    real = real.merge(series_estimate[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "z", "credibility_ref_id"]],
                      on=SERIES_ID_COLUMN, how="left")
    exam_months = sorted(real[period].unique())
    configuration.log_action(STEP_LABEL, 1, f"exam months {exam_months[0]}..{exam_months[-1]} · {real[SERIES_ID_COLUMN].nunique():,} "
                                            f"forecast series · {len(real):,} series × month · horizons {configuration.backtest_horizons}")

    # [2] the three methods
    decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    composition_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()
                                    & (pool_series["vencen"] > 0)]
    reference_history = (closed.merge(reference_members, on=SERIES_ID_COLUMN)
                         if reference_members is not None and len(reference_members) else closed.iloc[0:0].assign(credibility_ref_id=None))
    rows, latest_used = [], []
    for target_month in exam_months:
        for horizon in configuration.backtest_horizons:
            origin = target_month - horizon
            known = closed[closed[period] <= origin]
            latest_used.append((known[period].max(), composition_truth.loc[composition_truth[period] <= origin, period].max(), origin))
            month_rows = real[real[period] == target_month].copy()
            month_rows["h"], month_rows["origin"] = horizon, origin
            month_rows["tramo_h"] = band_of_horizon(int(horizon), configuration.horizon_bands)
            rows.append(predict_month(month_rows, known, composition_truth, reference_history, decision, origin, horizon,
                                      backtest["bands"], configuration))
    detail = pd.concat(rows, ignore_index=True)
    methods = sorted(detail["method"].unique(), key=lambda method: (method != METHOD_RAW, method != METHOD_FRAMEWORK, method))
    configuration.log_action(STEP_LABEL, 2, f"{len(detail):,} predictions · methods {methods}")

    # [3] the errors
    per_series = errors_per_series(detail, configuration)
    by_month, summary = errors_of_the_total(detail, methods, configuration)
    configuration.log_action(STEP_LABEL, 3, "per method, all series: " + " · ".join(
        f"{method} WAPE {row['wape']:.1%}" + (f", {row['in_interval']:.0%} in the interval" if pd.notna(row["in_interval"]) else "")
        for method, row in overall_by_method(detail, configuration).iterrows()))

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the exam")
    peeked = [origin for used, composition_used, origin in latest_used
              if (pd.notna(used) and used > origin) or (pd.notna(composition_used) and composition_used > origin)]
    configuration.log_check(STEP_LABEL, check_log, "no prediction used a month after its origin", not peeked,
                            failure_detail=f"{len(peeked)} exams used later months", context=f"{len(latest_used)} month × horizon exams")
    expected_rows = len(real) * len(configuration.backtest_horizons)
    per_method = detail.groupby("method").size()
    configuration.log_check(STEP_LABEL, check_log, "every method predicted every series of every exam month",
                            bool((per_method == expected_rows).all()) and detail["pred_rate"].notna().all(),
                            failure_detail=f"rows per method {per_method.to_dict()} for {expected_rows:,}",
                            context=f"{len(methods)} methods × {expected_rows:,}")
    real_total = real[renewed].sum() * len(configuration.backtest_horizons)
    adds_up = all(abs(detail.loc[detail["method"] == method, "real_units"].sum() - real_total) <= UNITS_TOLERANCE for method in methods)
    configuration.log_check(STEP_LABEL, check_log, "the real renewals of the exam equal the renewals of the exam months (nothing counted twice)",
                            adds_up, failure_detail="a method lost or duplicated renewals",
                            context=f"{real[renewed].sum():,.0f} renewed units in the exam months")
    framework = detail[detail["method"] == METHOD_FRAMEWORK]
    consistent = (int(per_series["framework_predictions"].sum()) == len(framework)
                  and abs(per_series["framework_real_units"].sum() - framework["real_units"].sum()) <= UNITS_TOLERANCE)
    configuration.log_check(STEP_LABEL, check_log, "the summary per series adds up to the detail (counts and units)", consistent,
                            failure_detail="sff_series_exam does not add up to sff_series_exam_detail")

    # [5] the tables
    configuration.log_action(STEP_LABEL, 5, "writing the exam")
    configuration.write_table(STEP_LABEL, check_log, detail, TABLE_SERIES_EXAM_DETAIL)
    configuration.write_table(STEP_LABEL, check_log, per_series, TABLE_SERIES_EXAM)
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_PORTFOLIO_EXAM)
    configuration.write_table(STEP_LABEL, check_log, summary, TABLE_PORTFOLIO_EXAM_SUMMARY)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the methods against each other
    configuration.log_action(STEP_LABEL, 7, "the methods on the same rows (error_total: of the sum of the portfolio; wape_series: "
                                            "series by series, no compensation):")
    configuration.show_table(summary)
    return dict(detail=detail, per_series=per_series, by_month=by_month, summary=summary)


def predict_month(month_rows: pd.DataFrame, known: pd.DataFrame, composition_truth: pd.DataFrame,
                  reference_history: pd.DataFrame, decision: pd.Series, origin, horizon: int, bands: pd.DataFrame,
                  configuration: Config) -> pd.DataFrame:
    """The predictions of one exam month and horizon, one row per series and method."""
    period, due, renewed = configuration.period_col, configuration.pipeline_units_col, configuration.renewed_units_col
    band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
    cell_level = known.groupby("_cell")[renewed].sum() / known.groupby("_cell")[due].sum()
    global_level = known[renewed].sum() / known[due].sum()
    month_rows = month_rows.assign(cell_level=month_rows["_cell"].map(cell_level).fillna(global_level))

    # the framework: the composition's prediction, moved toward its reference (levels known at the origin)
    composition_known = composition_truth[composition_truth[period] <= origin]
    composition_rate, composition_level, composition_technique = {}, {}, {}
    for composition_id, monthly in composition_known[composition_known[COMPOSITION_ID_COLUMN].isin(
            set(month_rows[COMPOSITION_ID_COLUMN].dropna()))].groupby(COMPOSITION_ID_COLUMN):
        if len(monthly) < MIN_MONTHS_TO_PREDICT:
            continue
        monthly = monthly.sort_values(period)
        technique = decision.get((composition_id, band_name), configuration.challenger_technique)
        technique, rate = predict_composition(logit(monthly[RATE_COLUMN].to_numpy(dtype=float)),
                                              np.array([month.month for month in monthly[period]]), technique, horizon,
                                              configuration.challenger_technique)
        composition_rate[composition_id], composition_technique[composition_id] = rate, technique
        composition_level[composition_id] = monthly["renovadas"].sum() / monthly["vencen"].sum()
    reference_level = levels_at_origins(reference_history, "credibility_ref_id",
                                        pd.DataFrame({"credibility_ref_id": month_rows["credibility_ref_id"].dropna().unique(),
                                                      "origin": origin}), period, due, renewed).set_index("credibility_ref_id")["level"]
    framework = month_rows.copy()
    framework["method"] = METHOD_FRAMEWORK
    framework["technique"] = framework[COMPOSITION_ID_COLUMN].map(composition_technique)
    framework["composition_rate"] = framework[COMPOSITION_ID_COLUMN].map(composition_rate)
    rate, shift = shifted_rate(framework["composition_rate"], framework["z"],
                               framework["credibility_ref_id"].map(reference_level),
                               framework[COMPOSITION_ID_COLUMN].map(composition_level), configuration.apply_credibility_shift)
    framework["credibility_shift_logit"] = shift
    framework["pred_rate"] = np.where(np.isfinite(rate), rate, framework["cell_level"])
    framework["technique"] = framework["technique"].fillna("mandatory_cell")

    # raw: the forecast series alone, with its own months
    own_known = known[known[SERIES_ID_COLUMN].isin(set(month_rows[SERIES_ID_COLUMN]))]
    own_rate, own_technique = {}, {}
    composition_of = month_rows.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN].to_dict()
    for series_id, monthly in own_known.groupby(SERIES_ID_COLUMN):
        monthly = monthly.sort_values(period)
        composition_id = composition_of.get(series_id)
        if len(monthly) < MIN_MONTHS_TO_PREDICT:
            own_rate[series_id], own_technique[series_id] = monthly[renewed].sum() / monthly[due].sum(), "own_level"
            continue
        technique = decision.get((composition_id, band_name), configuration.challenger_technique)
        if len(monthly) < int(CATALOGUE.get(technique, (None, None, None, 1))[3]):
            technique = configuration.challenger_technique
        monthly_rates = (monthly[renewed] / monthly[due]).clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
        technique, rate = predict_composition(logit(monthly_rates), np.array([month.month for month in monthly[period]]),
                                              technique, horizon, configuration.challenger_technique)
        own_rate[series_id], own_technique[series_id] = rate, technique
    raw = month_rows.copy()
    raw["method"] = METHOD_RAW
    raw["technique"] = raw[SERIES_ID_COLUMN].map(own_technique).fillna("mandatory_cell")
    raw["composition_rate"] = np.nan
    raw["credibility_shift_logit"] = 0.0
    raw["pred_rate"] = raw[SERIES_ID_COLUMN].map(own_rate).fillna(raw["cell_level"])

    # the interval of raw and framework, as the forecast builds its band
    for frame in (raw, framework):
        q_low, q_high = band_quantiles(bands, frame["technique"], frame["h"], configuration)
        frame["band_low"], frame["band_high"] = rate_band(frame["pred_rate"], frame[due], q_low, q_high)

    # the spreadsheet, for every grain (no interval)
    spreadsheets = []
    window = known[known[period] > origin - configuration.baseline_months]
    window_global = window[renewed].sum() / window[due].sum()
    for grain in configuration.baseline_grains:
        sheet = month_rows.copy()
        sheet["method"], sheet["technique"] = f"hoja_{grain}", f"last {configuration.baseline_months} months"
        if grain == "global":
            sheet["pred_rate"] = window_global
        else:
            columns = configuration.business_mandatory_dims if grain == "mandatory" else grain.split("+")
            keys = join_columns(window, columns)
            grain_rate = window.groupby(keys)[renewed].sum() / window.groupby(keys)[due].sum()
            sheet["pred_rate"] = join_columns(sheet, columns).map(grain_rate).fillna(window_global).to_numpy(dtype=float)
        sheet["composition_rate"], sheet["credibility_shift_logit"] = np.nan, 0.0
        sheet["band_low"], sheet["band_high"] = np.nan, np.nan
        spreadsheets.append(sheet)

    predictions = pd.concat([raw, framework] + spreadsheets, ignore_index=True)
    predictions["due_units"] = predictions[due]
    predictions["real_units"] = predictions[renewed]
    predictions["real_rate"] = predictions["real_units"] / predictions["due_units"]
    predictions["pred_units"] = predictions["pred_rate"] * predictions["due_units"]
    predictions["err_units"] = predictions["pred_units"] - predictions["real_units"]
    predictions["err_pp"] = 100 * (predictions["pred_rate"] - predictions["real_rate"])
    has_interval = predictions["band_low"].notna()
    predictions["in_band"] = np.where(has_interval, ((predictions["real_rate"] >= predictions["band_low"] - 1e-12)
                                                     & (predictions["real_rate"] <= predictions["band_high"] + 1e-12)).astype(float), np.nan)
    return predictions[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, period, "h", "tramo_h", "origin", "method", "technique",
                        "composition_rate", "credibility_shift_logit", "pred_rate", "band_low", "band_high", "real_rate",
                        "due_units", "pred_units", "real_units", "err_units", "err_pp", "in_band"]]


def errors_per_series(detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per forecast series and method (raw, framework): predictions, in the interval, units, error, coverage;
    side by side, so the improvement over the raw is one subtraction."""
    frames = []
    for method in (METHOD_RAW, METHOD_FRAMEWORK):
        block = detail[detail["method"] == method].assign(abs_err_units=lambda frame: frame["err_units"].abs(),
                                                          abs_err_pp=lambda frame: frame["err_pp"].abs())
        grouped = block.groupby(SERIES_ID_COLUMN)
        summary = pd.DataFrame({"predictions": grouped.size(), "in_band": grouped["in_band"].sum(),
                                "pred_units": grouped["pred_units"].sum(), "real_units": grouped["real_units"].sum(),
                                "abs_err_units": grouped["abs_err_units"].sum(), "mae_pp": grouped["abs_err_pp"].mean(),
                                "bias_pp": grouped["err_pp"].mean()})
        summary["wape"] = summary["abs_err_units"] / summary["real_units"].where(summary["real_units"] > 0)
        summary["coverage"] = summary["in_band"] / summary["predictions"]
        frames.append(summary.add_prefix(f"{method}_"))
    per_series = pd.concat(frames, axis=1).reset_index()
    per_series["improvement_mae_pp"] = per_series[f"{METHOD_RAW}_mae_pp"] - per_series[f"{METHOD_FRAMEWORK}_mae_pp"]
    return per_series


def overall_by_method(detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per method, over every series: WAPE and the share of predictions in their interval."""
    grouped = detail.assign(abs_err_units=detail["err_units"].abs()).groupby("method")
    return pd.DataFrame({"wape": grouped["abs_err_units"].sum() / grouped["real_units"].sum(),
                         "in_interval": grouped["in_band"].mean()})


def errors_of_the_total(detail: pd.DataFrame, methods: list, configuration: Config) -> tuple:
    """By month and horizon: real, predicted and error of the total per method; the summary per method."""
    period = configuration.period_col
    month_rows, summary_rows = [], []
    for (target_month, horizon), block in detail.groupby([period, "h"]):
        real_units = block.loc[block["method"] == methods[0], "real_units"].sum()
        row = {"mes": target_month, "h": horizon, "renovadas_reales": real_units}
        for method in methods:
            predicted = block.loc[block["method"] == method, "pred_units"].sum()
            row[f"{method}_pred"] = predicted
            row[f"{method}_error_total"] = (predicted - real_units) / real_units
        month_rows.append(row)
    by_month = pd.DataFrame(month_rows)
    for horizon in sorted(detail["h"].unique()):
        block = detail[detail["h"] == horizon]
        for method in methods:
            month_errors = by_month.loc[by_month["h"] == horizon, f"{method}_error_total"]
            rows = block[block["method"] == method]
            summary_rows.append({"metodo": method, "h": horizon, "error_total_medio": float(month_errors.abs().mean()),
                                 "sesgo_total_medio": float(month_errors.mean()),
                                 "wape_series": float(rows["err_units"].abs().sum() / rows["real_units"].sum()),
                                 "en_intervalo": float(rows["in_band"].mean()) if rows["in_band"].notna().any() else np.nan})
    return by_month, pd.DataFrame(summary_rows).sort_values(["h", "error_total_medio"]).reset_index(drop=True)
