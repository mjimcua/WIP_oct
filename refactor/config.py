"""
config.py — The configuration of SFF (mirror version, grown step by step).

What it holds today (steps 00 to 03):
  · THE COLUMN CONTRACT: which role every column of the raw plays. A column has one
    role (the two extra groups may share a column); the raw may not carry a column
    with no role, nor miss a declared one.
  · THE CALENDAR: the current month and the number of exam months. The role of every
    month is generated from them; the raw's own role columns are never read.

  · THE LOG: level, colours and an optional file. The logger is configured once, when
    the Config is built, and every step logs through it:
      configuration.log_step_start(...)   title, purpose, actions and output     (blue)
      configuration.log_action(...)       one action of the step, as it is done  (blue)
      configuration.log_check(...)        one numbered check: ok / WARN / FAIL   (green / yellow / red)
      configuration.log_check_summary()   the count of the checks; stops if any FAILED
      configuration.logger.doc(...)       any explanation or summary line        (blue)
      configuration.show_table(...)       concrete rows, as a pandas table (never logged)
  · THE IDS AND KEYS: how a row names its series, its forecast unit and its
    revaluation combination (joined text ids), and their keys (a hash of each id).
  · THE TABLES: where they are written (a SQL engine, or CSV files when there is none).
    Every table is read back after writing and its row count checked.

What it does not hold yet: parameters of later steps, keys and ids, persistence. Each
arrives with the step that needs it.

Usage: subclass Config in main.py, override `read_raw()` and declare the columns there.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import datetime
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from logging_helpers import DOC_LEVEL, LoggerManager
from vocabulario import ROLE_PENDING, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN


# ─── named constants ─────────────────────────────────────────────────────────────
# The roles a column of the raw can play.
COLUMN_ROLE_PERIOD = "period"
COLUMN_ROLE_MEASURE = "measure"
COLUMN_ROLE_TIME_SERIES_FLAG = "time_series_flag"
COLUMN_ROLE_MANDATORY = "mandatory"
COLUMN_ROLE_TIMEVARYING = "timevarying"
COLUMN_ROLE_EXTRA_RENOVACION = "extra_renovacion"
COLUMN_ROLE_EXTRA_REVALORIZACION = "extra_revalorizacion"
COLUMN_ROLE_BOTH_EXTRAS = "extra_renovacion+extra_revalorizacion"
COLUMN_ROLE_FORMULA_INPUT = "formula_input"
COLUMN_ROLE_IGNORE = "ignore"

# The name of the framework's logger (a named logger: other libraries keep their own).
LOGGER_NAME = "sff"

# How the ids are built: fields joined with "|"; a unit and its combination with "||".
ID_FIELD_SEPARATOR = "|"
COMBINED_ID_SEPARATOR = "||"
NO_COMBINATION_ID = "na"          # the combination id when no extra_revalorizacion is declared
HASH_HEX_DIGITS = 12              # 48 bits: fits a SQL bigint

# The status of a check, and the level it is logged at.
STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED = "ok", "WARN", "FAIL", "--"
LOG_LEVEL_BY_STATUS = {STATUS_OK: logging.INFO, STATUS_WARNING: logging.WARNING,
                       STATUS_FAILED: logging.ERROR, STATUS_NOT_EVALUATED: logging.WARNING}
EXAMPLE_ROWS_SHOWN = 3      # example rows shown under a check

# A timevarying column rotates towards churn (negative) or towards renewal (positive).
VALID_TIMEVARYING_SIGNS = ("negative", "positive")

# The three accepted spellings of a month.
MONTH_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}$")              # 2026-09
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")          # 2026-09-01
DAY_FIRST_DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")    # 01/09/2026


def parse_month(month_value) -> pd.Period:
    """A month as a monthly Period, from "2026-09", "2026-09-01", "01/09/2026" (day
    first) or a date-like object. Anything else raises ValueError: an ambiguous date
    never becomes a silent wrong month."""
    if isinstance(month_value, (pd.Period, pd.Timestamp, datetime.date)):
        return pd.Period(month_value, freq="M")

    month_text = str(month_value).strip()
    if MONTH_ONLY_PATTERN.match(month_text):
        return pd.Period(month_text, freq="M")
    if ISO_DATE_PATTERN.match(month_text):
        return pd.Period(month_text[:7], freq="M")
    if DAY_FIRST_DATE_PATTERN.match(month_text):
        day_text, month_number_text, year_text = month_text.split("/")
        return pd.Period(f"{year_text}-{month_number_text}", freq="M")

    raise ValueError(f"'{month_value}' is not a month: use 'YYYY-MM', 'YYYY-MM-DD' or 'DD/MM/YYYY'")


def join_columns(frame: pd.DataFrame, columns: list) -> pd.Series:
    """The "|"-joined id of every row from several columns, in the given order (the order
    is part of the id). Built on arrays, not Series, so a duplicated index cannot misalign it."""
    joined_ids = frame[columns[0]].astype(str).to_numpy(dtype=object)
    for column_name in columns[1:]:
        joined_ids = joined_ids + ID_FIELD_SEPARATOR + frame[column_name].astype(str).to_numpy(dtype=object)
    return pd.Series(joined_ids, index=frame.index)


def hash_key(identifier) -> int:
    """The key of an id: the first 12 hex digits of MD5(id) as an integer. The same id
    gives the same key on any machine and in any run (keys are derived, never assigned)."""
    return int(hashlib.md5(str(identifier).encode("utf-8")).hexdigest()[:HASH_HEX_DIGITS], 16)


def running_in_notebook() -> bool:
    """True inside a Jupyter notebook (where display() renders an HTML table)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:

    # ─── the columns of the raw, by role ────────────────────────────────────────────
    period_col: str = "period"                              # the month of the row
    pipeline_units_col: str = "total_tr_units"              # subscriptions falling due
    pipeline_usd_col: str = "total_tr_usd"                  # their value at due date
    renewed_units_col: str = "total_renewed_units"          # subscriptions renewed
    renewed_usd_col: str = "total_renewed_usd"              # their value at renewal
    extra_measure_cols: list = field(default_factory=list)  # other measures (reacquisitions, AUVs): declared, not used yet
    flag_time_series_col: str = "flag_time_series"          # marks the rows of the time_series universe

    business_mandatory_dims: list = field(default_factory=list)       # open the series and the uplift cell
    structural_timevarying_dims: dict = field(default_factory=dict)   # column → "negative" | "positive"
    extra_renovacion: list = field(default_factory=list)              # enter the rate series only
    extra_revalorizacion: list = field(default_factory=list)          # enter the uplift cell only

    discount_value_column: Optional[str] = None   # the exact discount (formula input, never a dimension)
    sku_column: Optional[str] = None              # the SKU (formula input for the price scenarios)
    ignore_cols: list = field(default_factory=list)   # read by nobody; may be absent from the raw

    # ─── the calendar (the role of every month is generated from it) ────────────────
    current_month: Optional[str] = None   # first month of the future; no default on purpose
    test_months: int = 3                  # closed months before it that only evaluate
    pending_close_months: int = 0         # months before the exam not closed yet (0 = all closed)

    # ─── where the tables are written ───────────────────────────────────────────────
    sql_engine: Optional[object] = None   # a SQLAlchemy engine; None → CSV files in output_folder
    sql_schema: Optional[str] = None      # "dbo" in SQL Server; None in SQLite
    table_prefix: str = "sff_"            # every table the framework writes starts with it
    output_folder: str = "salida"         # where the CSV tables go when there is no engine

    # ─── the log (configured once, when the Config is built) ────────────────────────
    log_level: str = "INFO"               # DEBUG · INFO · WARNING · ERROR
    log_colors: bool = True               # colours on the console (off in tests)
    log_file: Optional[str] = None        # also write the log to this file, without colours

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE RAW SOURCE
    # ═══════════════════════════════════════════════════════════════════════════════

    def read_raw(self) -> pd.DataFrame:
        """The raw extract. Overridden by the Config of main.py (SQL, CSV, synthetic)."""
        raise NotImplementedError("override read_raw() in your Config subclass (main.py)")

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE COLUMN CONTRACT
    # ═══════════════════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        """The logger is configured, and a misdeclared Config stops here, before any data is read."""
        # [0] the log: one configuration for the whole run
        LoggerManager(self.log_level, use_colors=self.log_colors, log_file=self.log_file).get_logger_configured(LOGGER_NAME)

        # [1] every timevarying column carries a valid sign
        invalid_signs = {column_name: sign for column_name, sign in self.structural_timevarying_dims.items()
                         if sign not in VALID_TIMEVARYING_SIGNS}
        if invalid_signs:
            raise ValueError(f"timevarying signs must be one of {VALID_TIMEVARYING_SIGNS}: {invalid_signs}")

        # [2] no column declared with two roles (built and checked by column_roles)
        self.column_roles()

        # [3] the calendar parameters are well formed
        if self.current_month is not None:
            parse_month(self.current_month)
        if int(self.test_months) < 1:
            raise ValueError(f"test_months must be at least 1 (found {self.test_months}): without an exam nothing is evaluated")
        if int(self.pending_close_months) < 0:
            raise ValueError(f"pending_close_months cannot be negative (found {self.pending_close_months})")

    @property
    def logger(self) -> logging.Logger:
        """The framework's logger: every step logs through it."""
        return logging.getLogger(LOGGER_NAME)

    def log_step_start(self, step_label: str, step_name: str, step_purpose: str,
                       step_actions: list, step_output: str) -> None:
        """The opening of a step, in blue: its title, its purpose, the actions it will
        take (numbered, as log_action will report them) and what it returns or writes."""
        self.logger.log(DOC_LEVEL, f"[{step_label}] ═══ STEP {step_label} · {step_name} ═══")
        self.logger.log(DOC_LEVEL, f"[{step_label}] purpose: {step_purpose}")
        self.logger.log(DOC_LEVEL, f"[{step_label}] actions:")
        for action_number, action_text in enumerate(step_actions, 1):
            self.logger.log(DOC_LEVEL, f"[{step_label}]    {action_number}. {action_text}")
        self.logger.log(DOC_LEVEL, f"[{step_label}] output: {step_output}")

    def log_action(self, step_label: str, action_number: int, action_text: str) -> None:
        """One action of a step, when it is done, with its result (blue)."""
        self.logger.log(DOC_LEVEL, f"[{step_label}] ▸ {action_number}. {action_text}")

    def log_check(self, step_label: str, check_log: list, description: str, passed: bool,
                  failure_detail: str = "", context: str = "", blocking: bool = True,
                  examples: Optional[pd.DataFrame] = None) -> bool:
        """One check of a step, numbered, logged at the level of its status (ok green, WARN
        yellow, FAIL red); its status is kept in `check_log` for the summary. `context` is
        shown when it passes, `failure_detail` when it fails; `examples` (a few concrete
        rows) are shown under the line as a pandas table."""
        if passed:
            status, detail = STATUS_OK, context
        else:
            status, detail = (STATUS_FAILED if blocking else STATUS_WARNING), failure_detail
        check_log.append((status, description, detail))

        has_examples = examples is not None and len(examples) > 0
        detail_text = f"   {detail}" if detail else ""
        examples_text = f"   · {min(len(examples), EXAMPLE_ROWS_SHOWN)} example rows below" if has_examples else ""
        self.logger.log(LOG_LEVEL_BY_STATUS[status],
                        f"[{step_label}]  {len(check_log):>2}. {status:<4}  {description}{detail_text}{examples_text}")
        if has_examples:
            self.show_table(examples.head(EXAMPLE_ROWS_SHOWN))
        return passed

    def log_not_evaluated(self, step_label: str, check_log: list, description: str, reason: str) -> None:
        """A check that cannot be made because an earlier one failed (yellow)."""
        check_log.append((STATUS_NOT_EVALUATED, description, reason))
        self.logger.warning(f"[{step_label}]  {len(check_log):>2}. {STATUS_NOT_EVALUATED:<4}  {description}   {reason}")

    def log_check_summary(self, step_label: str, step_name: str, check_log: list) -> None:
        """The count of the checks of a step, at the level of the worst one; stops with a
        ValueError naming the failed checks if any FAILED."""
        counts = {status: sum(1 for logged in check_log if logged[0] == status)
                  for status in (STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED)}
        if counts[STATUS_FAILED]:
            summary_level = logging.ERROR
        elif counts[STATUS_WARNING] or counts[STATUS_NOT_EVALUATED]:
            summary_level = logging.WARNING
        else:
            summary_level = logging.INFO
        self.logger.log(summary_level, f"[{step_label}] {len(check_log)} checks: {counts[STATUS_OK]} ok · "
                                       f"{counts[STATUS_WARNING]} warnings · {counts[STATUS_FAILED]} failed · "
                                       f"{counts[STATUS_NOT_EVALUATED]} not evaluated")

        failed_checks = [logged for logged in check_log if logged[0] == STATUS_FAILED]
        if failed_checks:
            numbered_failures = "\n".join(f"  {number}. {description} — {detail}"
                                          for number, (status, description, detail) in enumerate(failed_checks, 1))
            raise ValueError(f"step {step_label} · {step_name}: {len(failed_checks)} checks failed\n{numbered_failures}")

    def write_table(self, step_label: str, check_log: list, frame: pd.DataFrame, table_name: str) -> None:
        """Write a table, read it back and log a check on its row count.

        The physical name is table_prefix + table_name. With an engine the table is
        replaced in SQL (schema sql_schema); without one it is a CSV in output_folder.
        Monthly Periods are written as text ("2026-09"). The check fails if the rows read
        back are not the rows written.
        """
        physical_name = f"{self.table_prefix}{table_name}"
        persisted_frame = frame.copy()
        for column_name in persisted_frame.columns:
            if isinstance(persisted_frame[column_name].dtype, pd.PeriodDtype):
                persisted_frame[column_name] = persisted_frame[column_name].astype(str)

        write_start_time = time.time()
        if self.sql_engine is not None:
            persisted_frame.to_sql(physical_name, self.sql_engine, schema=self.sql_schema,
                                   if_exists="replace", index=False, chunksize=10_000)
            qualified_name = f"{self.sql_schema}.{physical_name}" if self.sql_schema else physical_name
            rows_read_back = int(pd.read_sql(f"SELECT COUNT(*) AS row_count FROM {qualified_name}",
                                             self.sql_engine)["row_count"].iloc[0])
            destination = f"SQL {qualified_name}"
        else:
            os.makedirs(self.output_folder, exist_ok=True)
            destination = os.path.join(self.output_folder, f"{physical_name}.csv")
            persisted_frame.to_csv(destination, index=False)
            rows_read_back = len(pd.read_csv(destination))
        elapsed_seconds = time.time() - write_start_time

        self.log_check(step_label, check_log, f"table {physical_name} written and read back",
                       rows_read_back == len(persisted_frame),
                       failure_detail=f"{len(persisted_frame):,} rows written but {rows_read_back:,} read back from {destination}",
                       context=f"{rows_read_back:,} rows × {len(persisted_frame.columns)} columns → "
                               f"{destination} ({elapsed_seconds:.1f}s)")

    def show_table(self, table: pd.DataFrame) -> None:
        """Concrete rows (examples) as a pandas table, never through the logger: rendered
        with display() in a notebook, printed aligned (to_string) in a terminal. The log
        line that introduces the table says how many rows follow."""
        sys.stdout.flush()      # the log lines already written come before the table
        shown_table = table.reset_index(drop=True)
        if running_in_notebook():
            from IPython.display import display
            display(shown_table)
        else:
            print(shown_table.to_string())

    @property
    def rate_series_columns(self) -> list:
        """The columns that define ONE renewal-rate series: mandatory + timevarying +
        extra_renovacion, in this order (the field order of fs_id and fu_id)."""
        return (self.business_mandatory_dims + list(self.structural_timevarying_dims)
                + self.extra_renovacion)

    @property
    def core_measures(self) -> list:
        """The four measures every step computes with."""
        return [self.pipeline_units_col, self.pipeline_usd_col, self.renewed_units_col, self.renewed_usd_col]

    def column_roles(self) -> dict:
        """Every declared column with its role: {column: role}.

        A column may have ONE role. The only exception: the two extra groups may share a
        column (its role is then "extra_renovacion+extra_revalorizacion"). Any other
        column declared twice raises ValueError naming every conflict at once.
        """
        # [1] every (column, role) pair the Config declares
        declared_pairs = [(self.period_col, COLUMN_ROLE_PERIOD),
                          (self.flag_time_series_col, COLUMN_ROLE_TIME_SERIES_FLAG)]
        for measure_column in self.core_measures + list(self.extra_measure_cols):
            declared_pairs.append((measure_column, COLUMN_ROLE_MEASURE))
        for mandatory_column in self.business_mandatory_dims:
            declared_pairs.append((mandatory_column, COLUMN_ROLE_MANDATORY))
        for timevarying_column in self.structural_timevarying_dims:
            declared_pairs.append((timevarying_column, COLUMN_ROLE_TIMEVARYING))
        for extra_column in self.extra_renovacion:
            declared_pairs.append((extra_column, COLUMN_ROLE_EXTRA_RENOVACION))
        for extra_column in self.extra_revalorizacion:
            declared_pairs.append((extra_column, COLUMN_ROLE_EXTRA_REVALORIZACION))
        for formula_input_column in (self.discount_value_column, self.sku_column):
            if formula_input_column:
                declared_pairs.append((formula_input_column, COLUMN_ROLE_FORMULA_INPUT))
        for ignored_column in self.ignore_cols:
            declared_pairs.append((ignored_column, COLUMN_ROLE_IGNORE))

        # [2] group the roles of each column
        roles_by_column = {}
        for column_name, role in declared_pairs:
            roles_by_column.setdefault(column_name, [])
            if role not in roles_by_column[column_name]:
                roles_by_column[column_name].append(role)

        # [3] one role per column, except a column shared by the two extra groups
        both_extras = {COLUMN_ROLE_EXTRA_RENOVACION, COLUMN_ROLE_EXTRA_REVALORIZACION}
        column_roles, conflicts = {}, {}
        for column_name, roles in roles_by_column.items():
            if len(roles) == 1:
                column_roles[column_name] = roles[0]
            elif set(roles) == both_extras:
                column_roles[column_name] = COLUMN_ROLE_BOTH_EXTRAS
            else:
                conflicts[column_name] = roles
        if conflicts:
            raise ValueError(f"columns declared with more than one role: {conflicts}")
        return column_roles

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE CALENDAR
    # ═══════════════════════════════════════════════════════════════════════════════

    def calendar_boundaries(self) -> dict:
        """The three months that cut the calendar: current (first month of projection),
        pending_start (first pending month; = current when there is none) and test_start
        (first exam month). Raises ValueError when current_month is not declared."""
        if self.current_month is None:
            raise ValueError("current_month is not declared: set it in your Config "
                             "(e.g. current_month=\"2026-09\" or \"01/09/2026\")")
        current = parse_month(self.current_month)
        pending_start = current - int(self.pending_close_months)
        test_start = pending_start - int(self.test_months)
        return dict(current=current, pending_start=pending_start, test_start=test_start)

    def role_of_months(self, periods) -> np.ndarray:
        """The role of every month: ≥ current → proyeccion · ≥ pending_start →
        pendiente_cierre · ≥ test_start → examen · earlier → entrenamiento."""
        boundaries = self.calendar_boundaries()
        month_values = pd.Series(periods).reset_index(drop=True)
        # np.select takes the FIRST condition that holds, so the order is the rule
        conditions = [month_values >= boundaries["current"],
                      month_values >= boundaries["pending_start"],
                      month_values >= boundaries["test_start"]]
        return np.select(conditions, [ROLE_PROJECTION, ROLE_PENDING, ROLE_TEST], default=ROLE_TRAIN)

    def calendar_description(self) -> str:
        """The calendar in one line, for the console."""
        boundaries = self.calendar_boundaries()
        current, pending_start, test_start = boundaries["current"], boundaries["pending_start"], boundaries["test_start"]
        if self.pending_close_months > 0:
            pending_text = f"{ROLE_PENDING} {pending_start}..{current - 1}"
        else:
            pending_text = f"{ROLE_PENDING}: none"
        return (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{pending_start - 1} · "
                f"{pending_text} · {ROLE_PROJECTION} ≥ {current}")
