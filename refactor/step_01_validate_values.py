"""
step_01_validate_values.py — The values of the raw: money, dimensions and flags.

CONTENT, not structure: step 00 already checked that the columns and the months fit
the configuration.

Which months each check looks at:
  · the pipeline (units and USD falling due): every month, the future included
  · the renewals: only the months BEFORE the current one. From the current month on
    they are early results that step 02 wipes, so they are not checked.

Actions (logged as they are done):
  1. scope the renewal checks: renewals are checked only in months before the current one
     (from the current month on they are partial and step 02 wipes them)
  2. check the money: pipeline, every month; renewals, closed months        checks 1-10
  3. check the dimensions and the flags                                     checks 11-16
  4. check the exact discount, if declared                                  check 17
  5. look for what is plausible but worth a look (warnings)                 checks 18-20
  6. count the checks; stop if any failed
  7. report the money of the closed months

Checks (logged as they are made, numbered, at the level of their status):
  pipeline, every month          no nulls · no negatives (units and USD)
  renewals, closed months        no negatives (units and USD)
                                 how "no renewal" arrives: null or 0 (warning if null,
                                   with example rows shown as a table; step 02 reads a null as 0)
                                 units and USD are null together
                                 no renewal for free (units > 0, USD ≤ 0)
                                 no USD without units (USD > 0, units = 0)
  dimensions                     no null or empty value
  flags                          each timevarying column and the time_series flag is 0 / 1
  exact discount (if declared)   between 0 and 1, or null (unknown)
  warnings only                  no row renews more units than fall due (rate > 100 %)
                                 no row falls due with units but USD = 0
                                 no row repeated in every column (a double load)

Output: the raw, unchanged (nothing is written).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import ACTIVE_FLAG_VALUES, Config


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "01"
STEP_NAME = "VALIDATE VALUES"
STEP_PURPOSE = ("check that the values inside the raw can be computed with: money without nulls or negatives, "
                "renewals coherent with what fell due, dimensions never empty, flags 0 / 1, discount as a share")
STEP_ACTIONS = ["scope the renewal checks: renewals are checked only in months before the current one "
                "(from the current month on they are partial and step 02 wipes them)",
                "check the money: pipeline every month, renewals in closed months (checks 1-10)",
                "check the dimensions and the flags (checks 11-16)",
                "check the exact discount, if declared (check 17)",
                "look for what is plausible but worth a look: warnings (checks 18-20)",
                "count the checks; stop if any failed",
                "report the money of the closed months"]
STEP_OUTPUT = "the raw, unchanged (nothing is written)"

# ─── named constants ─────────────────────────────────────────────────────────────
# The only values a dichotomous column (timevarying, time_series flag) may carry.
DICHOTOMOUS_VALUES = {0, 1, True, False, "0", "1"}


def validate_values(validated: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the money, the dimensions and the flags of the raw, logging every action and check."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the scope of the renewal checks: the months before the current one. From the current
    #     month on the renewals are partial (the month is still running) and step 02 wipes them,
    #     so only the pipeline is checked there
    current_month = configuration.calendar_boundaries()["current"]
    closed_rows = validated[validated[configuration.period_col] < current_month]
    renewed_units = closed_rows[configuration.renewed_units_col]
    renewed_usd = closed_rows[configuration.renewed_usd_col]
    configuration.log_action(STEP_LABEL, 1, f"renewals checked in {len(closed_rows):,} rows before {current_month} · "
                                            f"{len(validated) - len(closed_rows):,} rows from {current_month} on: pipeline only, "
                                            f"renewals wiped in step 02")

    # [2] the money
    configuration.log_action(STEP_LABEL, 2, "checking the money")
    for pipeline_column in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
        check_no_nulls(configuration, check_log, validated, pipeline_column, "every month")
        check_no_negatives(configuration, check_log, validated, pipeline_column, "every month")
    for renewed_column in (configuration.renewed_units_col, configuration.renewed_usd_col):
        check_no_negatives(configuration, check_log, closed_rows, renewed_column, "closed months")

    # how "nobody renewed" arrives: null (a LEFT JOIN with no match) or 0. A null is not an
    # error but it is not a number either: step 02 reads it as 0.
    no_result_rows = closed_rows[renewed_units.isna() | (renewed_units == 0)]
    null_rows = int(renewed_units.isna().sum())
    zero_rows = int((renewed_units == 0).sum())
    convention_text = f"{null_rows:,} rows null · {zero_rows:,} rows 0 (of {len(closed_rows):,} closed rows)"
    configuration.log_check(STEP_LABEL, check_log, "rows with no renewal arrive as 0, not null, closed months",
                            null_rows == 0,
                            failure_detail=f"{convention_text}: a null means no renewal (read as 0 in step 02)",
                            context=convention_text, blocking=False,
                            examples=example_rows(no_result_rows.sort_values(configuration.renewed_units_col,
                                                                             na_position="first"), configuration))

    null_mismatch = closed_rows[renewed_units.isna() != renewed_usd.isna()]     # one null without the other: broken
    configuration.log_check(STEP_LABEL, check_log, "renewed units and USD are null together, closed months",
                            null_mismatch.empty,
                            failure_detail=f"{len(null_mismatch):,} closed rows have one of the two renewal columns null",
                            examples=example_rows(null_mismatch, configuration))

    free_renewals = closed_rows[(renewed_units > 0) & (renewed_usd <= 0)]
    configuration.log_check(STEP_LABEL, check_log, "no renewal for free (units > 0 with USD ≤ 0), closed months",
                            free_renewals.empty,
                            failure_detail=f"{len(free_renewals):,} closed rows renew units with USD ≤ 0 "
                                           f"(renewing for free does not exist)",
                            examples=example_rows(free_renewals, configuration))
    usd_without_units = closed_rows[(renewed_usd > 0) & (renewed_units == 0)]
    configuration.log_check(STEP_LABEL, check_log, "no renewed USD without renewed units, closed months",
                            usd_without_units.empty,
                            failure_detail=f"{len(usd_without_units):,} closed rows have renewed USD with 0 renewed units",
                            examples=example_rows(usd_without_units, configuration))

    # [3] the dimensions and the flags
    configuration.log_action(STEP_LABEL, 3, f"checking {len(dimension_columns(configuration))} dimensions and "
                                            f"{len(configuration.structural_timevarying_dims) + 1} flags")
    empty_by_dimension = {}
    for dimension_column in dimension_columns(configuration):
        values = validated[dimension_column]
        empty_rows = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        if empty_rows:
            empty_by_dimension[dimension_column] = empty_rows
    configuration.log_check(STEP_LABEL, check_log, "no dimension has a null or empty value", not empty_by_dimension,
                            failure_detail=f"empty values in dimensions (rows per column): {empty_by_dimension}",
                            context=f"{len(dimension_columns(configuration))} dimensions")

    dichotomous_columns = list(configuration.structural_timevarying_dims) + [configuration.flag_time_series_col]
    for dichotomous_column in dichotomous_columns:
        values = validated[dichotomous_column]
        unexpected_values = [value for value in values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
        has_nulls = bool(values.isna().any())
        configuration.log_check(STEP_LABEL, check_log, f"{dichotomous_column} is 0 / 1",
                                not unexpected_values and not has_nulls,
                                failure_detail=f"{dichotomous_column} must be 0 / 1: found {unexpected_values[:10]}"
                                               f"{' and nulls' if has_nulls else ''}",
                                context=f"{int(values.isin(ACTIVE_FLAG_VALUES).mean() * 100)} % of rows at 1")

    # [4] the exact discount: a share between 0 and 1, or null (unknown)
    if configuration.discount_value_column:
        discount_values = validated[configuration.discount_value_column]
        out_of_range = int(((discount_values < 0) | (discount_values > 1)).sum())
        configuration.log_action(STEP_LABEL, 4, f"checking {configuration.discount_value_column}")
        configuration.log_check(STEP_LABEL, check_log,
                                f"{configuration.discount_value_column} is between 0 and 1 (or null = unknown)",
                                out_of_range == 0,
                                failure_detail=f"{out_of_range:,} rows of {configuration.discount_value_column} "
                                               f"outside [0, 1] (the discount is a share: 0.25 = 25 %)",
                                context=f"{discount_values.isna().mean():.1%} unknown")
    else:
        configuration.log_action(STEP_LABEL, 4, "no exact discount declared: nothing to check")

    # [5] warnings: plausible in a real extract, worth a look
    configuration.log_action(STEP_LABEL, 5, "looking for what is plausible but worth a look")
    above_pipeline = closed_rows[renewed_units > closed_rows[configuration.pipeline_units_col]]
    excess_units = float((above_pipeline[configuration.renewed_units_col]
                          - above_pipeline[configuration.pipeline_units_col]).sum())
    configuration.log_check(STEP_LABEL, check_log, "no row renews more units than fall due, closed months",
                            above_pipeline.empty,
                            failure_detail=f"{len(above_pipeline):,} closed rows renew more units than fall due "
                                           f"(+{excess_units:,.0f} units: a rate above 100 %)",
                            blocking=False, examples=example_rows(above_pipeline, configuration))
    units_without_value = int(((validated[configuration.pipeline_units_col] > 0)
                               & (validated[configuration.pipeline_usd_col] == 0)).sum())
    configuration.log_check(STEP_LABEL, check_log, "no row falls due with units but USD = 0", units_without_value == 0,
                            failure_detail=f"{units_without_value:,} rows fall due with units but USD = 0",
                            blocking=False)
    repeated_rows = int(validated.duplicated().sum())
    configuration.log_check(STEP_LABEL, check_log, "no row repeated in every column", repeated_rows == 0,
                            failure_detail=f"{repeated_rows:,} rows repeated in every column (a possible double load)",
                            blocking=False)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the money of the closed months, and the rate it implies (a null renewal counts as 0)
    pipeline_units = closed_rows[configuration.pipeline_units_col].sum()
    pipeline_usd = closed_rows[configuration.pipeline_usd_col].sum()
    configuration.log_action(STEP_LABEL, 7, f"closed months: {pipeline_units:,.0f} units due · "
                                            f"{renewed_units.sum():,.0f} renewed ({renewed_units.sum() / pipeline_units:.1%}) · "
                                            f"${pipeline_usd:,.0f} due · ${renewed_usd.sum():,.0f} renewed")
    return validated


def check_no_nulls(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    null_rows = int(rows[column_name].isna().sum())
    configuration.log_check(STEP_LABEL, check_log, f"{column_name} has no nulls, {scope}", null_rows == 0,
                            failure_detail=f"{null_rows:,} nulls in {column_name} ({scope})")


def check_no_negatives(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    negative_rows = int((rows[column_name] < 0).sum())
    configuration.log_check(STEP_LABEL, check_log, f"{column_name} has no negatives, {scope}", negative_rows == 0,
                            failure_detail=f"{negative_rows:,} negative values in {column_name} ({scope})")


def example_rows(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The columns that identify a row and its money, to show a few rows under a check."""
    return rows[[configuration.period_col] + configuration.extract_mandatory_dims + configuration.core_measures]


def dimension_columns(configuration: Config) -> list:
    """Every dimension column, each once: mandatory, timevarying and both extra groups."""
    all_dimensions = (configuration.extract_mandatory_dims + list(configuration.structural_timevarying_dims)
                      + configuration.extra_renovacion + configuration.extra_revalorizacion)
    return list(dict.fromkeys(all_dimensions))
