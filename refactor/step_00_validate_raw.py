"""
step_00_validate_raw.py — The first look at the raw: does it fit the configuration?

STRUCTURE, not content: the columns and the months. The values inside the measures
and the dimensions are step 01.

Actions (logged as they are done):
  1. read the roles the Config declares
  2. compare the columns of the raw with them                       checks 1-4
  3. read the months of the raw and fit the calendar to them        checks 5-10
  4. count the checks; stop if any failed
  5. convert the period to a monthly Period
  6. show the columns by role and the calendar, as tables

Checks (logged as they are made, numbered, at the level of their status):
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

Output: a copy of the raw, same rows and columns, with the period as a monthly Period.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import COLUMN_ROLE_IGNORE, Config, parse_month
from vocabulario import CALENDAR_ROLE_COLUMN


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "00"
STEP_NAME = "VALIDATE RAW"
STEP_PURPOSE = ("check that the raw fits the configuration before reading any value: every column has a role "
                "and is there, and the months of the raw can hold the calendar (current month, exam, training)")
STEP_ACTIONS = ["read the roles the Config declares",
                "compare the columns of the raw with them (checks 1-4)",
                "read the months of the raw and fit the calendar to them (checks 5-10)",
                "count the checks; stop if any failed",
                "convert the period to a monthly Period",
                "show the columns by role and the calendar, as tables"]
STEP_OUTPUT = "the raw, same rows and columns, with the period as a monthly Period (nothing is written)"


def validate_raw(raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the columns and the months of the raw against the Config, logging every action and check."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the roles the Config declares
    column_roles = configuration.column_roles()
    configuration.log_action(STEP_LABEL, 1, f"the Config declares {len(column_roles)} columns in "
                                            f"{len(set(column_roles.values()))} roles")

    # [2] the columns of the raw against the declared ones
    configuration.log_action(STEP_LABEL, 2, f"the raw has {len(raw.columns)} columns and {len(raw):,} rows")
    configuration.log_check(STEP_LABEL, check_log, "the raw has rows", not raw.empty,
                            failure_detail="the raw has no rows", context=f"{len(raw):,} rows")

    duplicated_columns = sorted(set(raw.columns[raw.columns.duplicated()]))     # a SQL join can repeat a name
    configuration.log_check(STEP_LABEL, check_log, "no column name appears twice", not duplicated_columns,
                            failure_detail=f"duplicated column names in the raw: {duplicated_columns}")

    columns_without_role = [column_name for column_name in raw.columns if column_name not in column_roles]
    configuration.log_check(STEP_LABEL, check_log, "every column of the raw has a role in the Config",
                            not columns_without_role,
                            failure_detail=f"columns with no role in the Config: {columns_without_role} "
                                           f"(declare their role or add them to ignore_cols)",
                            context=f"{len(raw.columns)} columns")

    missing_columns = [column_name for column_name, role in column_roles.items()
                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns]
    configuration.log_check(STEP_LABEL, check_log, "every declared column is in the raw (ignored ones may be absent)",
                            not missing_columns,
                            failure_detail=f"columns declared in the Config but missing from the raw: {missing_columns}")

    # [3] the months, and the calendar on them
    raw_months = []
    if period_column in raw.columns and period_column not in duplicated_columns:
        raw_months = check_months(raw, configuration, check_log)
    else:
        configuration.log_action(STEP_LABEL, 3, f"no usable '{period_column}' column: the months cannot be read")
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the months and the calendar",
                                        f"no usable '{period_column}' column")

    # [4] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 4, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [5] the period as a monthly Period
    validated = raw.copy()
    validated[period_column] = pd.PeriodIndex([parse_month(value) for value in validated[period_column]], freq="M")
    configuration.log_action(STEP_LABEL, 5, f"'{period_column}' converted to a monthly Period")

    # [6] what the raw is
    log_raw_report(validated, configuration, column_roles, raw_months)
    return validated


def check_months(raw: pd.DataFrame, configuration: Config, check_log: list) -> list:
    """Checks 5 to 10: the months can be read and the calendar fits them. Returns the sorted months."""
    period_column = configuration.period_col
    period_values = raw[period_column]

    # [5] no row without a month
    rows_without_month = int(period_values.isna().sum())
    configuration.log_check(STEP_LABEL, check_log, "every row has a month", rows_without_month == 0,
                            failure_detail=f"{rows_without_month:,} rows with no {period_column}")

    # [6] every distinct value is a month (read once per distinct value)
    months_by_value, unparseable_values = {}, []
    for distinct_value in period_values.dropna().unique():
        try:
            months_by_value[distinct_value] = parse_month(distinct_value)
        except ValueError:
            unparseable_values.append(distinct_value)
    raw_months = sorted(set(months_by_value.values()))
    months_text = f"{len(raw_months)} months, {raw_months[0]}..{raw_months[-1]}" if raw_months else "no month"
    configuration.log_action(STEP_LABEL, 3, f"the months of the raw: {months_text}")
    configuration.log_check(STEP_LABEL, check_log, f"every value of {period_column} is a month", not unparseable_values,
                            failure_detail=f"values that are not a month: {unparseable_values[:10]}",
                            context=months_text)
    if not raw_months:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the calendar against the months of the raw",
                                        "no month could be read")
        return raw_months

    # [7] the current month is declared (the calendar has no default date)
    try:
        boundaries = configuration.calendar_boundaries()
    except ValueError as error:
        configuration.log_check(STEP_LABEL, check_log, "current_month is declared", False, failure_detail=str(error))
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the calendar against the months of the raw",
                                        "no current month")
        return raw_months
    configuration.log_check(STEP_LABEL, check_log, "current_month is declared", True, context=str(boundaries["current"]))

    # [8] the current month is inside the raw
    first_month, last_month = raw_months[0], raw_months[-1]
    configuration.log_check(STEP_LABEL, check_log, "the current month is inside the months of the raw",
                            first_month <= boundaries["current"] <= last_month,
                            failure_detail=f"current_month {boundaries['current']} is outside the months of the raw "
                                           f"({first_month}..{last_month}): the extract must carry the pipeline "
                                           f"from the current month on",
                            context=f"{boundaries['current']} in {first_month}..{last_month}")

    # [9] the exam leaves months to train
    configuration.log_check(STEP_LABEL, check_log, "the calendar leaves months to train",
                            boundaries["test_start"] > first_month,
                            failure_detail=f"the exam starts at {boundaries['test_start']} and the raw starts at "
                                           f"{first_month}: no month to train (lower test_months)",
                            context=f"training {first_month}..{boundaries['test_start'] - 1}")

    # [10] a month with no row at all is worth a look, not a stop
    expected_months = pd.period_range(first_month, last_month, freq="M")
    months_without_rows = [str(month) for month in expected_months if month not in set(raw_months)]
    configuration.log_check(STEP_LABEL, check_log, "every month between the first and the last has rows",
                            not months_without_rows,
                            failure_detail=f"months with no row at all: {months_without_rows}", blocking=False)
    return raw_months


def log_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
    """Action 6: the columns by role and the calendar, rows per role, as tables."""
    configuration.log_action(STEP_LABEL, 6, "the columns of the raw, by role:")
    columns_by_role = {}
    for column_name, role in column_roles.items():
        if column_name in validated.columns:
            columns_by_role.setdefault(role, []).append(column_name)
    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "columnas": len(role_columns), "nombres": ", ".join(role_columns)}
                                           for role, role_columns in columns_by_role.items()]))

    configuration.logger.doc(f"[{STEP_LABEL}] the calendar of the Config on these months: "
                             f"{configuration.calendar_description()}")
    rows_per_role = pd.Series(configuration.role_of_months(validated[configuration.period_col])).value_counts()
    months_per_role = pd.Series(configuration.role_of_months(pd.Series(raw_months))).value_counts()
    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "meses": int(months_per_role.get(role, 0)), "filas": int(row_count)}
                                           for role, row_count in rows_per_role.items()]))
