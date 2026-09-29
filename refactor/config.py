"""
config.py — The configuration of SFF (mirror version, grown step by step).

What it holds today (steps 00 and 01):
  · THE COLUMN CONTRACT: which role every column of the raw plays. A column has one
    role (the two extra groups may share a column); the raw may not carry a column
    with no role, nor miss a declared one.
  · THE CALENDAR: the current month and the number of exam months. The role of every
    month is generated from them; the raw's own role columns are never read.

  · THE LOG: level, colours and an optional file. The logger is configured once, when
    the Config is built, and every step logs through `configuration.logger`.

What it does not hold yet: parameters of later steps, keys and ids, persistence. Each
arrives with the step that needs it.

Usage: subclass Config in main.py, override `read_raw()` and declare the columns there.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import datetime
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from logging_helpers import LoggerManager
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
