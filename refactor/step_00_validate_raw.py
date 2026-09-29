"""
step_00_validate_raw.py — The first look at the raw: does it fit the configuration?

STRUCTURE, not content: the columns and the months. The values inside the measures
and the dimensions are step 01.

The checks, each logged when it is made, numbered, at the level of its status:
   1. the raw has rows
   2. no column name appears twice
   3. every column of the raw has a role in the Config
   4. every declared column is in the raw (the ignored ones may be absent)
   5. every row has a month
   6. every value of the period column is a month
   7. current_month is declared
   8. the current month is inside the months of the raw
   9. the calendar leaves months to train
  10. every month between the first and the last has rows            (warning only)

Returns a copy of the raw with the period as a monthly Period. Nothing else changes.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import logging

import pandas as pd

from config import COLUMN_ROLE_IGNORE, Config, parse_month


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "00"
STEP_NAME = "VALIDATE RAW"
STEP_PURPOSE = ("check that the raw fits the configuration before reading any value: every column has a role "
                "and is there, and the months of the raw can hold the calendar (current month, exam, training)")

# ─── logging the checks ──────────────────────────────────────────────────────────
# Every check is logged when it is made, numbered, at the level of its status:
#   ok (INFO) · WARN (WARNING: does not block) · FAIL (ERROR: blocks) · -- (WARNING: not evaluated)
STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED = "ok", "WARN", "FAIL", "--"
LOG_LEVEL_BY_STATUS = {STATUS_OK: logging.INFO, STATUS_WARNING: logging.WARNING,
                       STATUS_FAILED: logging.ERROR, STATUS_NOT_EVALUATED: logging.WARNING}


def validate_raw(raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the columns and the months of the raw against the Config, logging every check; stop if any fails."""
    configuration.logger.info(f"[{STEP_LABEL}] ═══ STEP {STEP_LABEL} · {STEP_NAME} ═══")
    configuration.logger.info(f"[{STEP_LABEL}] purpose: {STEP_PURPOSE}")
    check_log = []
    column_roles = configuration.column_roles()
    period_column = configuration.period_col

    # [1] the raw has rows
    log_check(configuration, check_log, "the raw has rows", not raw.empty,
                  failure_detail="the raw has no rows", context=f"{len(raw):,} rows")

    # [2] no column name twice (a SQL join can return two columns with the same name)
    duplicated_columns = sorted(set(raw.columns[raw.columns.duplicated()]))
    log_check(configuration, check_log, "no column name appears twice", not duplicated_columns,
                  failure_detail=f"duplicated column names in the raw: {duplicated_columns}")

    # [3] every raw column has a role
    columns_without_role = [column_name for column_name in raw.columns if column_name not in column_roles]
    log_check(configuration, check_log, "every column of the raw has a role in the Config", not columns_without_role,
                  failure_detail=f"columns with no role in the Config: {columns_without_role} "
                                 f"(declare their role or add them to ignore_cols)",
                  context=f"{len(raw.columns)} columns")

    # [4] every declared column is in the raw (the ignored ones may be absent)
    missing_columns = [column_name for column_name, role in column_roles.items()
                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns]
    log_check(configuration, check_log, "every declared column is in the raw (ignored ones may be absent)", not missing_columns,
                  failure_detail=f"columns declared in the Config but missing from the raw: {missing_columns}")

    # [5..10] the months: they parse, and the calendar fits them
    raw_months = []
    period_is_usable = period_column in raw.columns and period_column not in duplicated_columns
    if period_is_usable:
        raw_months = check_months(raw, configuration, check_log)
    else:
        log_not_evaluated(configuration, check_log, "the months and the calendar", f"no usable '{period_column}' column")

    # [11] the count of the checks; stop if anything failed
    log_summary_and_stop(configuration, check_log)

    # [12] the period as a monthly Period, and what the raw is
    validated = raw.copy()
    validated[period_column] = pd.PeriodIndex([parse_month(value) for value in validated[period_column]], freq="M")
    print_raw_report(validated, configuration, column_roles, raw_months)
    return validated


def check_months(raw: pd.DataFrame, configuration: Config, check_log: list) -> list:
    """Checks 5 to 10 of step 00. Returns the sorted months of the raw."""
    period_column = configuration.period_col
    period_values = raw[period_column]

    # [5] no row without a month
    rows_without_month = int(period_values.isna().sum())
    log_check(configuration, check_log, "every row has a month", rows_without_month == 0,
                  failure_detail=f"{rows_without_month:,} rows with no {period_column}")

    # [6] every distinct value is a month (checked once per distinct value)
    months_by_value, unparseable_values = {}, []
    for distinct_value in period_values.dropna().unique():
        try:
            months_by_value[distinct_value] = parse_month(distinct_value)
        except ValueError:
            unparseable_values.append(distinct_value)
    raw_months = sorted(set(months_by_value.values()))
    log_check(configuration, check_log, f"every value of {period_column} is a month", not unparseable_values,
                  failure_detail=f"values that are not a month: {unparseable_values[:10]}",
                  context=f"{len(raw_months)} months, {raw_months[0]}..{raw_months[-1]}" if raw_months else "")
    if not raw_months:
        log_not_evaluated(configuration, check_log, "the calendar against the months of the raw", "no month could be read")
        return raw_months

    # [7] the current month is declared (the calendar has no default date)
    try:
        boundaries = configuration.calendar_boundaries()
        log_check(configuration, check_log, "current_month is declared", True, context=str(boundaries["current"]))
    except ValueError as error:
        log_check(configuration, check_log, "current_month is declared", False, failure_detail=str(error))
        log_not_evaluated(configuration, check_log, "the calendar against the months of the raw", "no current month")
        return raw_months

    # [8] the current month is inside the raw
    first_month, last_month = raw_months[0], raw_months[-1]
    log_check(configuration, check_log, "the current month is inside the months of the raw",
                  first_month <= boundaries["current"] <= last_month,
                  failure_detail=f"current_month {boundaries['current']} is outside the months of the raw "
                                 f"({first_month}..{last_month}): the extract must carry the pipeline from the current month on",
                  context=f"{boundaries['current']} in {first_month}..{last_month}")

    # [9] the exam leaves months to train
    log_check(configuration, check_log, "the calendar leaves months to train", boundaries["test_start"] > first_month,
                  failure_detail=f"the calendar leaves no month to train: the exam starts at {boundaries['test_start']} "
                                 f"and the raw starts at {first_month} (lower test_months or pending_close_months)",
                  context=f"training {first_month}..{boundaries['test_start'] - 1}")

    # [10] a month with no row at all is worth a look, not a stop
    expected_months = pd.period_range(first_month, last_month, freq="M")
    months_without_rows = [str(month) for month in expected_months if month not in set(raw_months)]
    log_check(configuration, check_log, "every month between the first and the last has rows", not months_without_rows,
                  failure_detail=f"months with no row at all: {months_without_rows}", blocking=False)
    return raw_months


def print_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
    """After the checks: the columns by role and the calendar, rows per role."""

    # [1] the columns, grouped by role
    columns_by_role = {}
    for column_name, role in column_roles.items():
        if column_name in validated.columns:
            columns_by_role.setdefault(role, []).append(column_name)
    configuration.logger.info("[00] the columns, by role")
    for role, role_columns in columns_by_role.items():
        configuration.logger.info(f"[00]         {role:<38} {', '.join(role_columns)}")

    # [2] the calendar, and how many months and rows fall in each role
    configuration.logger.info(f"[00] calendar: {configuration.calendar_description()}")
    rows_per_role = pd.Series(configuration.role_of_months(validated[configuration.period_col])).value_counts()
    months_per_role = pd.Series(configuration.role_of_months(pd.Series(raw_months))).value_counts()
    for role, row_count in rows_per_role.items():
        configuration.logger.info(f"[00]         {role:<18} {int(months_per_role.get(role, 0)):>3} months · {int(row_count):>8,} rows")


def log_check(configuration: Config, check_log: list, description: str, passed: bool,
              failure_detail: str = "", context: str = "", blocking: bool = True) -> bool:
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
