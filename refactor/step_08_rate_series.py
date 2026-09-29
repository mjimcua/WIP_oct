"""
step_08_rate_series.py — The rate series: the renewal rate of every unit, and of every series.

A rate series is the monthly sequence of forecast units of one fs_id. Before estimating
anything later, this step leaves three things ready:
  · GAPS: a month with no unit INSIDE the history of a series (between its first and
    last closed month) gets a gap row, marked sintetica = 1, with nothing due. A month
    with no expirations says nothing about the rate: its rate is null, never 0 %.
    Only series that will be estimated get gaps (route predecible, universe normal).
  · THE RATE OF EVERY UNIT: renewed / due units, only in closed truth months
    (entrenamiento, examen) of real units with something due. Everywhere else it is
    null: no information, never 0 %.
  · THE SUMMARY OF EVERY SERIES: its support (n_propio: the median units due in its
    real closed months with something due — the size of a typical month, the one we
    will predict), its own rate (Σ renewed / Σ due over its closed months), the Wilson
    half-width of that rate at n_propio and z (the binomial error of one month), and its
    sign (from its ACTIVE timevarying flags: neutro, negativo, positivo or mixto).
    A series below the support floor will borrow support from its relatives later.

Actions (logged as they are done):
  1. find the series that get gaps and their history span
  2. add the gap rows
  3. the rate of every unit
  4. the summary of every series: support, own rate, binomial error
  5. the sign of every series
  6. check the gaps, the rates and the summary                      checks 1-4
  7. write the gap rows and the series summary                      checks 5-6
  8. count the checks; stop if any failed
  9. show the series by support and the largest ones, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. gap rows only inside the history of estimable series, never repeating a unit
   2. a rate only in real closed months with something due
   3. every rate between 0 and 1                                    (warning only)
   4. every series has its summary row
   5-6. tables sff_fu_huecos and sff_series_tasa written and read back

Output: (the forecast units with the gap rows and the columns tasa and sintetica,
the series summary) · tables sff_fu_huecos, sff_series_tasa.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ACTIVE_FLAG_VALUES, ID_FIELD_SEPARATOR, Config, hash_key
from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, FINE_ROWS_COLUMN, RATE_COLUMN, ROUTE_COLUMN,
                         ROUTE_PREDICTABLE, SERIES_ID_COLUMN, SERIES_KEY_COLUMN, SIGN_COLUMN, SIGN_MIXED,
                         SIGN_NEGATIVE, SIGN_NEUTRAL, SIGN_POSITIVE, SYNTHETIC_COLUMN, TABLE_GAPS, TABLE_SERIES_RATE,
                         TRUTH_ROLES, UNIT_ID_COLUMN, UNIT_KEY_COLUMN, UNIVERSE_COLUMN, UNIVERSE_NORMAL)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "08"
STEP_NAME = "RATE SERIES"
STEP_PURPOSE = ("prepare the renewal rate: fill the gaps inside the history of every estimable series (rate "
                "undefined, never 0 %), compute the rate of every unit only where it is truth, and summarise every "
                "series by its support, its own rate, its binomial error and its sign")
STEP_ACTIONS = ["find the series that get gaps and their history span",
                "add the gap rows",
                "the rate of every unit",
                "the summary of every series: support, own rate, binomial error",
                "the sign of every series",
                "check the gaps, the rates and the summary (checks 1-4)",
                "write the gap rows and the series summary (checks 5-6)",
                "count the checks; stop if any failed",
                "show the series by support and the largest ones, as tables"]
STEP_OUTPUT = ("the forecast units with gap rows and the rate of every unit · one row per series with its support, "
               "own rate, error and sign · tables sff_fu_huecos, sff_series_tasa")

# ─── named constants ─────────────────────────────────────────────────────────────
PERCENTAGE_POINTS = 100
NO_EVIDENCE_RATE = 0.5          # the rate the Wilson error is computed at when a series has no own rate
LARGEST_SERIES_SHOWN = 5


def wilson_half_width_pp(rates: np.ndarray, supports: np.ndarray, z: float) -> np.ndarray:
    """Half the Wilson interval, in pp, for arrays of rates and supports. Wilson is honest for
    small n and rates near 0 or 1 (the Wald p ± z·se crosses 0 and 1 and is too narrow there).
    A support of 0 gives the widest interval, [0, 1]: 50 pp."""
    rates = np.asarray(rates, dtype=float)
    supports = np.asarray(supports, dtype=float)
    safe_supports = np.where(supports > 0, supports, 1.0)
    z_squared = z * z
    denominator = 1 + z_squared / safe_supports
    centre = (rates + z_squared / (2 * safe_supports)) / denominator
    half_width = z * np.sqrt(rates * (1 - rates) / safe_supports + z_squared / (4 * safe_supports ** 2)) / denominator
    low = np.clip(centre - half_width, 0.0, 1.0)
    high = np.clip(centre + half_width, 0.0, 1.0)
    return np.where(supports > 0, PERCENTAGE_POINTS * (high - low) / 2, PERCENTAGE_POINTS / 2)


def build_rate_series(forecast_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config) -> tuple:
    """Gaps, the rate of every unit and the summary of every series; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the series that get gaps: estimated ones, of the normal universe; their closed span
    estimable_series = series_table.loc[(series_table[ROUTE_COLUMN] == ROUTE_PREDICTABLE)
                                        & (series_table[UNIVERSE_COLUMN] == UNIVERSE_NORMAL), SERIES_ID_COLUMN]
    truth_units = forecast_units[forecast_units[SERIES_ID_COLUMN].isin(set(estimable_series))
                                 & forecast_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    configuration.log_action(STEP_LABEL, 1, f"{len(estimable_series):,} estimable series (predecible, normal) with "
                                            f"{len(truth_units):,} units in closed months")

    # [2] the gap rows
    gap_rows = build_gap_rows(truth_units, configuration)
    rated_units = forecast_units.copy()
    rated_units[SYNTHETIC_COLUMN] = 0
    if len(gap_rows):
        rated_units = pd.concat([rated_units, gap_rows[rated_units.columns]], ignore_index=True)
    configuration.log_action(STEP_LABEL, 2, f"{len(gap_rows):,} gap rows added in "
                                            f"{gap_rows[SERIES_ID_COLUMN].nunique() if len(gap_rows) else 0:,} series")

    # [3] the rate of every unit: only where it is truth
    pipeline_units = rated_units[configuration.pipeline_units_col]
    rate_is_truth = ((rated_units[SYNTHETIC_COLUMN] == 0) & (pipeline_units > 0)
                     & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES))
    rated_units[RATE_COLUMN] = np.where(rate_is_truth,
                                        rated_units[configuration.renewed_units_col] / pipeline_units.where(pipeline_units > 0),
                                        np.nan)
    configuration.log_action(STEP_LABEL, 3, f"rate computed in {int(rate_is_truth.sum()):,} units; null in the other "
                                            f"{int((~rate_is_truth).sum()):,} (future, pending, gaps, nothing due)")

    # [4] the summary of every series
    series_rate = summarise_series(rated_units, series_table, configuration)
    with_history = series_rate["meses_historia"] > 0
    below_floor = with_history & (series_rate["bajo_suelo"] == 1)
    configuration.log_action(STEP_LABEL, 4, f"{len(series_rate):,} series summarised: {int(below_floor.sum()):,} with own "
                                            f"history below the support floor ({configuration.support_floor:.0f}) and "
                                            f"{int((~with_history).sum()):,} with no own history carry "
                                            f"${series_rate.loc[below_floor | ~with_history, 'usd_por_predecir'].sum():,.0f} "
                                            f"of ${series_rate['usd_por_predecir'].sum():,.0f} to predict: they will borrow")

    # [5] the sign of every series
    series_rate[SIGN_COLUMN] = sign_of_series(rated_units, series_rate[SERIES_ID_COLUMN], configuration)
    configuration.log_action(STEP_LABEL, 5, f"signs: {series_rate[SIGN_COLUMN].value_counts().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the gaps, the rates and the summary")
    check_rate_series(rated_units, gap_rows, truth_units, series_rate, series_table, configuration, check_log)

    # [7] the tables, written
    configuration.log_action(STEP_LABEL, 7, "writing the gap rows and the series summary")
    gap_table = (gap_rows[[UNIT_ID_COLUMN, UNIT_KEY_COLUMN, SERIES_ID_COLUMN, SERIES_KEY_COLUMN,
                           configuration.period_col, CALENDAR_ROLE_COLUMN]]
                 if len(gap_rows) else pd.DataFrame(columns=[UNIT_ID_COLUMN, UNIT_KEY_COLUMN, SERIES_ID_COLUMN,
                                                             SERIES_KEY_COLUMN, configuration.period_col,
                                                             CALENDAR_ROLE_COLUMN]))
    configuration.write_table(STEP_LABEL, check_log, gap_table, TABLE_GAPS)
    configuration.write_table(STEP_LABEL, check_log, series_rate, TABLE_SERIES_RATE)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] the series by support, and the largest ones
    log_rate_series_report(series_rate, configuration)
    return rated_units, series_rate


def build_gap_rows(truth_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per month missing between the first and the last closed month of a series.
    The row copies the description of the series, has nothing due, and takes the role of
    the previous real month. Built as one frame (months × series minus the existing), not
    row by row: with thousands of series the row-by-row version takes seconds per step."""
    period_column = configuration.period_col
    if truth_units.empty:
        return pd.DataFrame()

    # [1] every month of every series' span, minus the months it already has
    span = truth_units.groupby(SERIES_ID_COLUMN)[period_column].agg(["min", "max"])
    all_months = pd.concat([pd.DataFrame({SERIES_ID_COLUMN: series_id,
                                          period_column: pd.period_range(first_month, last_month, freq="M")})
                            for series_id, (first_month, last_month) in span.iterrows()], ignore_index=True)
    existing = truth_units[[SERIES_ID_COLUMN, period_column]].drop_duplicates()
    missing = all_months.merge(existing, on=[SERIES_ID_COLUMN, period_column], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if missing.empty:
        return pd.DataFrame()

    # [2] the gap rows: the series' description, nothing due, a new unit id
    template = (truth_units.sort_values(period_column).drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN))
    gap_rows = template.loc[missing[SERIES_ID_COLUMN]].reset_index()
    gap_rows[period_column] = missing[period_column].to_numpy()
    for measure_column in configuration.core_measures:
        gap_rows[measure_column] = 0.0
    gap_rows[CURRENT_MONTH_COLUMN] = 0
    gap_rows[FINE_ROWS_COLUMN] = 0
    gap_rows[SYNTHETIC_COLUMN] = 1
    gap_rows[UNIT_ID_COLUMN] = gap_rows[SERIES_ID_COLUMN] + ID_FIELD_SEPARATOR + gap_rows[period_column].astype(str)
    gap_rows[UNIT_KEY_COLUMN] = gap_rows[UNIT_ID_COLUMN].map(hash_key)

    # [3] the role of the previous real month of the same series
    role_by_month = truth_units[[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN]].copy()
    role_by_month["_month_number"] = role_by_month[period_column].map(lambda month: month.ordinal)
    gap_rows["_month_number"] = gap_rows[period_column].map(lambda month: month.ordinal)
    gap_rows = pd.merge_asof(gap_rows.drop(columns=[CALENDAR_ROLE_COLUMN]).sort_values("_month_number"),
                             role_by_month.drop(columns=[period_column]).sort_values("_month_number"),
                             on="_month_number", by=SERIES_ID_COLUMN, direction="backward")
    return gap_rows.drop(columns="_month_number")


def summarise_series(rated_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per series of the series table: support, history, own rate, binomial error."""
    labels = series_table[[SERIES_ID_COLUMN, SERIES_KEY_COLUMN, ROUTE_COLUMN, UNIVERSE_COLUMN, "usd_por_predecir"]]
    normal_series = set(series_table.loc[series_table[UNIVERSE_COLUMN] == UNIVERSE_NORMAL, SERIES_ID_COLUMN])
    history = rated_units[rated_units[SERIES_ID_COLUMN].isin(normal_series)
                          & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    real_months_with_pipeline = history[(history[SYNTHETIC_COLUMN] == 0)
                                        & (history[configuration.pipeline_units_col] > 0)]

    grouped_history = history.groupby(SERIES_ID_COLUMN)
    summary = pd.DataFrame({
        "n_propio": real_months_with_pipeline.groupby(SERIES_ID_COLUMN)[configuration.pipeline_units_col].median(),
        "meses_historia": grouped_history.size(),
        "huecos": grouped_history[SYNTHETIC_COLUMN].sum(),
        "renovadas": grouped_history[configuration.renewed_units_col].sum(),
        "vencen": grouped_history[configuration.pipeline_units_col].sum()})
    summary = labels.merge(summary, left_on=SERIES_ID_COLUMN, right_index=True, how="left")
    summary[["n_propio", "renovadas", "vencen"]] = summary[["n_propio", "renovadas", "vencen"]].fillna(0.0)
    summary[["meses_historia", "huecos"]] = summary[["meses_historia", "huecos"]].fillna(0).astype(int)

    summary["tasa_propia"] = summary["renovadas"] / summary["vencen"].where(summary["vencen"] > 0)
    summary["error_binomial_pp"] = wilson_half_width_pp(summary["tasa_propia"].fillna(NO_EVIDENCE_RATE).clip(0, 1),
                                                        summary["n_propio"], configuration.z)
    summary["bajo_suelo"] = (summary["n_propio"] < configuration.support_floor).astype(int)
    return summary


def sign_of_series(rated_units: pd.DataFrame, series_ids: pd.Series, configuration: Config) -> np.ndarray:
    """The sign of every series from its ACTIVE timevarying flags (constant inside a series:
    they are part of fs_id): neutro (none active), negativo, positivo, mixto (both kinds)."""
    one_row_per_series = rated_units.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN).loc[series_ids]
    negative_columns = [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "negative"]
    positive_columns = [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "positive"]
    negative_active = (one_row_per_series[negative_columns].isin(ACTIVE_FLAG_VALUES).any(axis=1).to_numpy()
                       if negative_columns else np.zeros(len(one_row_per_series), dtype=bool))
    positive_active = (one_row_per_series[positive_columns].isin(ACTIVE_FLAG_VALUES).any(axis=1).to_numpy()
                       if positive_columns else np.zeros(len(one_row_per_series), dtype=bool))
    return np.select([negative_active & positive_active, negative_active, positive_active],
                     [SIGN_MIXED, SIGN_NEGATIVE, SIGN_POSITIVE], default=SIGN_NEUTRAL)


def check_rate_series(rated_units: pd.DataFrame, gap_rows: pd.DataFrame, truth_units: pd.DataFrame,
                      series_rate: pd.DataFrame, series_table: pd.DataFrame, configuration: Config,
                      check_log: list) -> None:
    """Checks 1 to 4."""
    period_column = configuration.period_col

    # [1] gap rows inside the closed span of an estimable series, and never an existing unit
    gaps_outside_span = 0
    if len(gap_rows):
        span = truth_units.groupby(SERIES_ID_COLUMN)[period_column].agg(["min", "max"])
        gap_span = span.loc[gap_rows[SERIES_ID_COLUMN]]
        inside = ((gap_rows[period_column].to_numpy() > gap_span["min"].to_numpy())
                  & (gap_rows[period_column].to_numpy() < gap_span["max"].to_numpy()))
        gaps_outside_span = int((~inside).sum())
    repeated_units = int(rated_units[UNIT_ID_COLUMN].duplicated().sum())
    configuration.log_check(STEP_LABEL, check_log,
                            "gap rows only inside the history of estimable series, never repeating a unit",
                            gaps_outside_span == 0 and repeated_units == 0,
                            failure_detail=f"{gaps_outside_span:,} gap rows outside the span · {repeated_units:,} repeated units",
                            context=f"{len(gap_rows):,} gap rows")

    # [2] the rate exists only where it is truth
    rate_where_not_truth = int((rated_units[RATE_COLUMN].notna()
                                & ((rated_units[SYNTHETIC_COLUMN] == 1)
                                   | ~rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                                   | (rated_units[configuration.pipeline_units_col] <= 0))).sum())
    configuration.log_check(STEP_LABEL, check_log, "a rate only in real closed months with something due",
                            rate_where_not_truth == 0,
                            failure_detail=f"{rate_where_not_truth:,} units have a rate where there is no truth")

    # [3] rates are proportions (a rate above 1 comes from renewing more than fell due: step 01 warned)
    out_of_range = rated_units[(rated_units[RATE_COLUMN] < 0) | (rated_units[RATE_COLUMN] > 1)]
    configuration.log_check(STEP_LABEL, check_log, "every rate between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} units with a rate outside [0, 1]", blocking=False,
                            examples=out_of_range[[UNIT_ID_COLUMN, configuration.pipeline_units_col,
                                                   configuration.renewed_units_col, RATE_COLUMN]])

    # [4] every series of the series table has its summary
    missing_series = len(set(series_table[SERIES_ID_COLUMN]) - set(series_rate[SERIES_ID_COLUMN]))
    configuration.log_check(STEP_LABEL, check_log, "every series has its summary row", missing_series == 0,
                            failure_detail=f"{missing_series:,} series without a summary",
                            context=f"{len(series_rate):,} series")


def log_rate_series_report(series_rate: pd.DataFrame, configuration: Config) -> None:
    """Action 9: series and money to predict by support, and the largest series."""
    configuration.log_action(STEP_LABEL, 9, f"series by support (n_propio: units due in a typical closed month; "
                                            f"the floor is {configuration.support_floor:.0f}):")
    with_history = series_rate["meses_historia"] > 0
    groups = {"sin historia propia (solo_futuro o serie_temporal)": ~with_history,
              f"bajo el suelo (< {configuration.support_floor:.0f})": with_history & (series_rate["bajo_suelo"] == 1),
              f"sobre el suelo (≥ {configuration.support_floor:.0f})": with_history & (series_rate["bajo_suelo"] == 0)}
    total_usd = series_rate["usd_por_predecir"].sum()
    configuration.show_table(pd.DataFrame([{"soporte": group_name, "series": int(mask.sum()),
                                            "error_pp_mediano": series_rate.loc[mask, "error_binomial_pp"].median(),
                                            "usd_por_predecir": series_rate.loc[mask, "usd_por_predecir"].sum(),
                                            "pct_usd": series_rate.loc[mask, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0}
                                           for group_name, mask in groups.items()]))

    configuration.logger.doc(f"[{STEP_LABEL}] the {LARGEST_SERIES_SHOWN} series with the most to predict "
                             f"(own rate ± its binomial error of one month):")
    configuration.show_table(series_rate.nlargest(LARGEST_SERIES_SHOWN, "usd_por_predecir")[
        [SERIES_ID_COLUMN, ROUTE_COLUMN, SIGN_COLUMN, "n_propio", "meses_historia", "huecos", "tasa_propia",
         "error_binomial_pp", "usd_por_predecir"]])
