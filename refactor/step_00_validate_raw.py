"""
step_00_validate_raw.py — The first look at the raw: its columns and its months.

Before computing anything, the raw must fit the configuration:
  · it has rows
  · no column name appears twice
  · every column has a role in the Config, and every declared column is there
    (the ignored ones may be absent)
  · every month parses, and the calendar of the Config fits the months of the raw:
    the current month is inside the raw and the exam leaves months to train

Every problem is collected and reported at once; if there is any, the run stops.
Gaps in the months (a month with no row at all) are reported, not stopped.

Returns a copy of the raw with the period as a monthly Period. Nothing else changes.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import COLUMN_ROLE_IGNORE, Config, parse_month


def validate_raw(raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the columns and the months of the raw against the Config; stop with every problem."""
    problems = []
    column_roles = configuration.column_roles()
    period_column = configuration.period_col

    # [1] the raw has rows
    if raw.empty:
        problems.append("the raw has no rows")

    # [2] no column name twice (a SQL join can return two columns with the same name)
    duplicated_columns = sorted(set(raw.columns[raw.columns.duplicated()]))
    if duplicated_columns:
        problems.append(f"duplicated column names in the raw: {duplicated_columns}")

    # [3] every raw column has a role
    columns_without_role = [column_name for column_name in raw.columns if column_name not in column_roles]
    if columns_without_role:
        problems.append(f"columns with no role in the Config: {columns_without_role} "
                        f"(declare their role or add them to ignore_cols)")

    # [4] every declared column is in the raw (the ignored ones may be absent)
    missing_columns = [column_name for column_name, role in column_roles.items()
                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns]
    if missing_columns:
        problems.append(f"columns declared in the Config but missing from the raw: {missing_columns}")

    # [5] the months: they parse, and the calendar fits them
    validated = raw.copy()
    raw_months = []
    period_is_usable = period_column in raw.columns and period_column not in duplicated_columns
    if period_is_usable:
        month_problems, raw_months = check_months(validated, configuration)
        problems.extend(month_problems)
        if not month_problems:
            validated[period_column] = pd.PeriodIndex(
                [parse_month(value) for value in validated[period_column]], freq="M")

    # [6] stop with the whole list
    if problems:
        numbered_problems = "\n".join(f"  {number}. {problem}" for number, problem in enumerate(problems, 1))
        raise ValueError(f"step 00 · the raw does not fit the configuration:\n{numbered_problems}")

    # [7] what the raw is, in numbers
    print_raw_report(validated, configuration, column_roles, raw_months)
    return validated


def check_months(raw: pd.DataFrame, configuration: Config) -> tuple:
    """The month checks of step 00. Returns (problems, sorted list of the raw's months)."""
    problems = []
    period_values = raw[configuration.period_col]

    # [1] no row without a month
    rows_without_month = int(period_values.isna().sum())
    if rows_without_month:
        problems.append(f"{rows_without_month:,} rows with no {configuration.period_col}")

    # [2] every distinct value is a month (checked once per distinct value)
    months_by_value, unparseable_values = {}, []
    for distinct_value in period_values.dropna().unique():
        try:
            months_by_value[distinct_value] = parse_month(distinct_value)
        except ValueError:
            unparseable_values.append(distinct_value)
    if unparseable_values:
        problems.append(f"values of {configuration.period_col} that are not a month: {unparseable_values[:10]}")
    raw_months = sorted(set(months_by_value.values()))
    if not raw_months:
        return problems, raw_months

    # [3] the calendar fits the raw: current month inside it, exam leaving months to train
    try:
        boundaries = configuration.calendar_boundaries()
    except ValueError as error:
        problems.append(str(error))
        return problems, raw_months
    first_month, last_month = raw_months[0], raw_months[-1]
    if not (first_month <= boundaries["current"] <= last_month):
        problems.append(f"current_month {boundaries['current']} is outside the months of the raw "
                        f"({first_month}..{last_month}): the extract must carry the pipeline from the current month on")
    if boundaries["test_start"] <= first_month:
        problems.append(f"the calendar leaves no month to train: the exam starts at {boundaries['test_start']} "
                        f"and the raw starts at {first_month} (lower test_months or pending_close_months)")
    return problems, raw_months


def print_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
    """The console of step 00: columns by role, months, calendar, rows per role."""
    print("[00] the raw fits the configuration")

    # [1] the columns, grouped by role
    columns_by_role = {}
    for column_name, role in column_roles.items():
        if column_name in validated.columns:
            columns_by_role.setdefault(role, []).append(column_name)
    print(f"[00] {len(validated.columns)} columns · {len(validated):,} rows")
    for role, role_columns in columns_by_role.items():
        print(f"       {role:<38} {', '.join(role_columns)}")

    # [2] the months, and any month with no row at all
    first_month, last_month = raw_months[0], raw_months[-1]
    expected_months = pd.period_range(first_month, last_month, freq="M")
    months_without_rows = [str(month) for month in expected_months if month not in set(raw_months)]
    print(f"[00] months {first_month}..{last_month} ({len(raw_months)} with data)")
    if months_without_rows:
        print(f"[00] WARNING months with no row at all: {months_without_rows}")

    # [3] the calendar, and how many months and rows fall in each role
    print(f"[00] calendar: {configuration.calendar_description()}")
    role_of_each_row = pd.Series(configuration.role_of_months(validated[configuration.period_col]))
    rows_per_role = role_of_each_row.value_counts()
    months_per_role = pd.Series(configuration.role_of_months(pd.Series(raw_months))).value_counts()
    for role, row_count in rows_per_role.items():
        print(f"       {role:<18} {int(months_per_role.get(role, 0)):>3} months · {int(row_count):>8,} rows")
