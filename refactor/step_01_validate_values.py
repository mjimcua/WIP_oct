"""
step_01_validate_values.py — The values of the raw: money, dimensions and flags.

CONTENT, not structure: step 00 already checked that the columns and the months fit
the configuration.

Which months each check looks at:
  · the pipeline (units and USD falling due): every month, the future included
  · the renewals: only the months BEFORE the current one. From the current month on
    they are early results that step 02 wipes, so they are not checked.

The checks, each logged when it is made, numbered, at the level of its status:
  pipeline, every month          no nulls · no negatives (units and USD)
  renewals, closed months        no negatives (units and USD)
                                 how "no renewal" arrives: null or 0 (warning if null,
                                   with example rows; step 02 reads a null as 0)
                                 units and USD are null together
                                 no renewal for free (units > 0, USD ≤ 0)
                                 no USD without units (USD > 0, units = 0)
  dimensions                     no null or empty value
  flags                          each timevarying column and the time_series flag is 0 / 1
  exact discount (if declared)   between 0 and 1, or null (unknown)
  warnings only                  no row renews more units than fall due (rate > 100 %)
                                 no row falls due with units but USD = 0
                                 no row repeated in every column (a double load)

Returns the raw unchanged.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import logging
from typing import Optional

import pandas as pd

from config import Config


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "01"
STEP_NAME = "VALIDATE VALUES"
STEP_PURPOSE = ("check that the values inside the raw can be computed with: money without nulls or negatives, "
                "renewals coherent with what fell due, dimensions never empty, flags 0 / 1, discount as a share")

# ─── named constants ─────────────────────────────────────────────────────────────
# The only values a dichotomous column (timevarying, time_series flag) may carry.
DICHOTOMOUS_VALUES = {0, 1, True, False, "0", "1"}

# ─── logging the checks ──────────────────────────────────────────────────────────
# Every check is logged when it is made, numbered, at the level of its status:
#   ok (INFO) · WARN (WARNING: does not block) · FAIL (ERROR: blocks) · -- (WARNING: not evaluated)
STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED = "ok", "WARN", "FAIL", "--"
LOG_LEVEL_BY_STATUS = {STATUS_OK: logging.INFO, STATUS_WARNING: logging.WARNING,
                       STATUS_FAILED: logging.ERROR, STATUS_NOT_EVALUATED: logging.WARNING}
EXAMPLE_ROWS_SHOWN = 3      # example rows logged under a check


def validate_values(validated: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the money, the dimensions and the flags of the raw, logging every check; stop if any fails."""
    configuration.logger.info(f"[{STEP_LABEL}] ═══ STEP {STEP_LABEL} · {STEP_NAME} ═══")
    configuration.logger.info(f"[{STEP_LABEL}] purpose: {STEP_PURPOSE}")
    check_log = []
    closed_months = validated[configuration.period_col] < configuration.calendar_boundaries()["current"]
    closed_rows = validated[closed_months]
    renewed_units = closed_rows[configuration.renewed_units_col]
    renewed_usd = closed_rows[configuration.renewed_usd_col]

    # [1] the pipeline, every month: no nulls, no negatives
    for pipeline_column in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
        check_no_nulls(configuration, check_log, validated, pipeline_column, "every month")
        check_no_negatives(configuration, check_log, validated, pipeline_column, "every month")

    # [2] the renewals, closed months: no negatives
    for renewed_column in (configuration.renewed_units_col, configuration.renewed_usd_col):
        check_no_negatives(configuration, check_log, closed_rows, renewed_column, "closed months")

    # [3] how "nobody renewed" arrives: null (a LEFT JOIN with no match) or 0. A null is
    #     not an error but it is not a number either: step 02 reads it as 0.
    no_result_rows = closed_rows[renewed_units.isna() | (renewed_units == 0)]
    null_rows = int(renewed_units.isna().sum())
    zero_rows = int((renewed_units == 0).sum())
    convention_text = f"{null_rows:,} rows null · {zero_rows:,} rows 0 (of {len(closed_rows):,} closed rows)"
    log_check(configuration, check_log, "rows with no renewal arrive as 0, not null, closed months", null_rows == 0,
                  failure_detail=f"{convention_text}: a null means no renewal (read as 0 in step 02)",
                  context=convention_text, blocking=False,
                  examples=example_rows(no_result_rows.sort_values(configuration.renewed_units_col, na_position="first"),
                                        configuration))

    # [4] units and USD of a renewal are null together (one null without the other is a broken row)
    null_mismatch = closed_rows[renewed_units.isna() != renewed_usd.isna()]
    log_check(configuration, check_log, "renewed units and USD are null together, closed months", null_mismatch.empty,
                  failure_detail=f"{len(null_mismatch):,} closed rows have one of the two renewal columns null",
                  examples=example_rows(null_mismatch, configuration) if not null_mismatch.empty else None)

    # [5] units and USD of a renewal go together
    free_renewals = closed_rows[(renewed_units > 0) & (renewed_usd <= 0)]
    log_check(configuration, check_log, "no renewal for free (units > 0 with USD ≤ 0), closed months", free_renewals.empty,
                  failure_detail=f"{len(free_renewals):,} closed rows renew units with USD ≤ 0 (renewing for free does not exist)",
                  examples=example_rows(free_renewals, configuration) if not free_renewals.empty else None)
    usd_without_units = closed_rows[(renewed_usd > 0) & (renewed_units == 0)]
    log_check(configuration, check_log, "no renewed USD without renewed units, closed months", usd_without_units.empty,
                  failure_detail=f"{len(usd_without_units):,} closed rows have renewed USD with 0 renewed units",
                  examples=example_rows(usd_without_units, configuration) if not usd_without_units.empty else None)

    # [6] dimensions: no nulls, no empty text
    empty_by_dimension = {}
    for dimension_column in dimension_columns(configuration):
        values = validated[dimension_column]
        empty_rows = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        if empty_rows:
            empty_by_dimension[dimension_column] = empty_rows
    log_check(configuration, check_log, "no dimension has a null or empty value", not empty_by_dimension,
                  failure_detail=f"empty values in dimensions (rows per column): {empty_by_dimension}",
                  context=f"{len(dimension_columns(configuration))} dimensions")

    # [7] dichotomous columns: only 0 / 1, one line each
    dichotomous_columns = list(configuration.structural_timevarying_dims) + [configuration.flag_time_series_col]
    for dichotomous_column in dichotomous_columns:
        values = validated[dichotomous_column]
        unexpected_values = [value for value in values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
        has_nulls = bool(values.isna().any())
        log_check(configuration, check_log, f"{dichotomous_column} is 0 / 1", not unexpected_values and not has_nulls,
                      failure_detail=f"{dichotomous_column} must be 0 / 1: found {unexpected_values[:10]}"
                                     f"{' and nulls' if has_nulls else ''}",
                      context=f"{int(values.isin([1, True, '1']).mean() * 100)} % of rows at 1")

    # [8] the exact discount: a share between 0 and 1, or null (unknown)
    if configuration.discount_value_column:
        discount_values = validated[configuration.discount_value_column]
        out_of_range = int(((discount_values < 0) | (discount_values > 1)).sum())
        log_check(configuration, check_log, f"{configuration.discount_value_column} is between 0 and 1 (or null = unknown)", out_of_range == 0,
                      failure_detail=f"{out_of_range:,} rows of {configuration.discount_value_column} outside [0, 1] "
                                     f"(the discount is a share: 0.25 = 25 %)",
                      context=f"{discount_values.isna().mean():.1%} unknown")

    # [9] warnings: plausible in a real extract, worth a look
    pipeline_units = closed_rows[configuration.pipeline_units_col]
    above_pipeline = closed_rows[renewed_units > pipeline_units]
    excess_units = float((above_pipeline[configuration.renewed_units_col] - above_pipeline[configuration.pipeline_units_col]).sum())
    log_check(configuration, check_log, "no row renews more units than fall due, closed months", above_pipeline.empty,
                  failure_detail=f"{len(above_pipeline):,} closed rows renew more units than fall due "
                                 f"(+{excess_units:,.0f} units: a rate above 100 %)", blocking=False,
                  examples=example_rows(above_pipeline, configuration) if not above_pipeline.empty else None)
    units_without_value = int(((validated[configuration.pipeline_units_col] > 0)
                               & (validated[configuration.pipeline_usd_col] == 0)).sum())
    log_check(configuration, check_log, "no row falls due with units but USD = 0", units_without_value == 0,
                  failure_detail=f"{units_without_value:,} rows fall due with units but USD = 0", blocking=False)
    repeated_rows = int(validated.duplicated().sum())
    log_check(configuration, check_log, "no row repeated in every column", repeated_rows == 0,
                  failure_detail=f"{repeated_rows:,} rows repeated in every column (a possible double load)",
                  blocking=False)

    # [10] the count of the checks; stop if anything failed; then the numbers
    log_summary_and_stop(configuration, check_log)
    log_values_report(validated, closed_rows, configuration)
    return validated


def check_no_nulls(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    null_rows = int(rows[column_name].isna().sum())
    log_check(configuration, check_log, f"{column_name} has no nulls, {scope}", null_rows == 0,
                  failure_detail=f"{null_rows:,} nulls in {column_name} ({scope})")


def check_no_negatives(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    negative_rows = int((rows[column_name] < 0).sum())
    log_check(configuration, check_log, f"{column_name} has no negatives, {scope}", negative_rows == 0,
                  failure_detail=f"{negative_rows:,} negative values in {column_name} ({scope})")


def example_rows(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The columns that identify a row and its money, to show a few rows under a check."""
    shown_columns = ([configuration.period_col] + configuration.business_mandatory_dims
                     + configuration.core_measures)
    return rows[shown_columns]


def dimension_columns(configuration: Config) -> list:
    """Every dimension column, each once: mandatory, timevarying and both extra groups."""
    all_dimensions = (configuration.business_mandatory_dims + list(configuration.structural_timevarying_dims)
                      + configuration.extra_renovacion + configuration.extra_revalorizacion)
    return list(dict.fromkeys(all_dimensions))


def log_values_report(validated: pd.DataFrame, closed_rows: pd.DataFrame, configuration: Config) -> None:
    """After the checks: the money of the closed months and the grain."""

    # [1] the money of the closed months, and the rate it implies (a null renewal counts as 0)
    pipeline_units = closed_rows[configuration.pipeline_units_col].sum()
    renewed_units = closed_rows[configuration.renewed_units_col].sum()
    pipeline_usd = closed_rows[configuration.pipeline_usd_col].sum()
    renewed_usd = closed_rows[configuration.renewed_usd_col].sum()
    configuration.logger.info(f"[01] closed months: {pipeline_units:,.0f} units due · {renewed_units:,.0f} renewed "
                f"({renewed_units / pipeline_units:.1%}) · ${pipeline_usd:,.0f} due · ${renewed_usd:,.0f} renewed")

    # [2] the grain: rows sharing month + every dimension are summed in step 03
    grain_columns = [configuration.period_col] + dimension_columns(configuration)
    rows_sharing_grain = int(validated.duplicated(grain_columns, keep=False).sum())
    configuration.logger.info(f"[01] grain: {rows_sharing_grain:,} rows share month + dimensions with another row (step 03 sums them)")


def log_check(configuration: Config, check_log: list, description: str, passed: bool,
              failure_detail: str = "", context: str = "", blocking: bool = True,
              examples: Optional[pd.DataFrame] = None) -> bool:
    """Log one check, numbered, at the level of its status; keep its status for the summary.
    `context` is shown when it passes, `failure_detail` when it fails."""
    if passed:
        status, detail = STATUS_OK, context
    else:
        status, detail = (STATUS_FAILED if blocking else STATUS_WARNING), failure_detail
    check_log.append((status, description, detail))
    level = LOG_LEVEL_BY_STATUS[status]
    detail_text = f"   {detail}" if detail else ""
    configuration.logger.log(level, f"[{STEP_LABEL}]  {len(check_log):>2}. {status:<4}  {description}{detail_text}")
    if examples is not None and len(examples):
        for table_line in examples.head(EXAMPLE_ROWS_SHOWN).to_string(index=False).splitlines():
            configuration.logger.log(level, f"[{STEP_LABEL}]          {table_line}")
    return passed


def log_not_evaluated(configuration: Config, check_log: list, description: str, reason: str) -> None:
    """A check that cannot be made because an earlier one failed."""
    check_log.append((STATUS_NOT_EVALUATED, description, reason))
    configuration.logger.warning(f"[{STEP_LABEL}]  {len(check_log):>2}. {STATUS_NOT_EVALUATED:<4}  {description}   {reason}")


def log_summary_and_stop(configuration: Config, check_log: list) -> None:
    """The count of the checks, at the level of the worst one; stop if any FAILED."""
    counts = {status: sum(1 for logged in check_log if logged[0] == status)
              for status in (STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED)}
    if counts[STATUS_FAILED]:
        summary_level = logging.ERROR
    elif counts[STATUS_WARNING] or counts[STATUS_NOT_EVALUATED]:
        summary_level = logging.WARNING
    else:
        summary_level = logging.INFO
    configuration.logger.log(summary_level, f"[{STEP_LABEL}] {len(check_log)} checks: {counts[STATUS_OK]} ok · "
                                            f"{counts[STATUS_WARNING]} warnings · {counts[STATUS_FAILED]} failed · "
                                            f"{counts[STATUS_NOT_EVALUATED]} not evaluated")

    failed_checks = [logged for logged in check_log if logged[0] == STATUS_FAILED]
    if failed_checks:
        numbered_failures = "\n".join(f"  {number}. {description} — {detail}"
                                      for number, (status, description, detail) in enumerate(failed_checks, 1))
        raise ValueError(f"step {STEP_LABEL} · {STEP_NAME}: {len(failed_checks)} checks failed\n{numbered_failures}")
