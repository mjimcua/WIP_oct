"""
config.py — The configuration of SFF (mirror version, grown step by step).

What it holds (grows with the steps that need it):
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
  · THE IDS AND KEYS: how a row names its rate series, its forecast unit and its
    uplift cell (joined text ids), and their keys (a hash of each id).
  · THE DISCOUNT: ONE column of the extract (the exact discount, a share: 0.25 = 25 %)
    serves both sides. As it is, it feeds the contract rule of the uplift
    (1 / (1 − discount)) and is part of the identity of a fine row. Cut into buckets
    (discount_bucket_edges), it is a dimension of the uplift cell. A null discount is
    unknown, not 0 %: its bucket is "sin_dato" and it takes the statistical uplift.
  · THE STATISTICAL PARAMETERS: the z of the bands and the support floor.
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
from vocabulario import ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN


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

# How the ids are built: fields joined with "|"; a unit and its revaluation values with "||".
ID_FIELD_SEPARATOR = "|"
COMBINED_ID_SEPARATOR = "||"
NO_REVALUATION_VALUES = "na"      # the revaluation part of a fine row when no extra_revalorizacion is declared
HASH_HEX_DIGITS = 12              # 48 bits: fits a SQL bigint
NULL_ID_TEXT = "null"             # how a null value is written inside an id (e.g. an unknown exact discount)

# The status of a check, and the level it is logged at.
STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED = "ok", "WARN", "FAIL", "--"
LOG_LEVEL_BY_STATUS = {STATUS_OK: logging.INFO, STATUS_WARNING: logging.WARNING,
                       STATUS_FAILED: logging.ERROR, STATUS_NOT_EVALUATED: logging.WARNING}
EXAMPLE_ROWS_SHOWN = 3      # example rows shown under a check

# The values that mean "this flag is on" in a timevarying column or the time_series flag.
ACTIVE_FLAG_VALUES = (1, True, "1")

# A timevarying column rotates towards churn (negative) or towards renewal (positive).
VALID_TIMEVARYING_SIGNS = ("negative", "positive")

# The bucket of a row whose exact discount is null (unknown, not 0 %).
UNKNOWN_DISCOUNT_BUCKET = "sin_dato"

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


def discount_bucket_labels(discount_values: pd.Series, bucket_edges_pct: list) -> pd.Series:
    """The bucket of every exact discount, labelled like the extract's discount_interval:
    "07 _ 60-70%" for a discount of 0.65. Buckets are [low, high) in percent, the last one
    closed ([90, 100]); the number is the bucket's position, from 01. A null discount is
    unknown: "sin_dato". A value outside the edges is left null (step 01 already stops a
    discount outside [0, 1])."""
    percent_values = (pd.to_numeric(discount_values, errors="coerce") * 100).round(6)
    labels = [f"{position:02d} _ {low:g}-{high:g}%"
              for position, (low, high) in enumerate(zip(bucket_edges_pct[:-1], bucket_edges_pct[1:]), 1)]
    bucket_codes = pd.cut(percent_values, bins=bucket_edges_pct, labels=labels, right=False, include_lowest=True)
    buckets = bucket_codes.astype(object)
    at_the_top = percent_values == bucket_edges_pct[-1]           # 100 % closes the last bucket
    buckets[at_the_top] = labels[-1]
    buckets[percent_values.isna()] = UNKNOWN_DISCOUNT_BUCKET
    return buckets


def is_one_year(rows: pd.DataFrame, configuration) -> pd.Series:
    """The rows of a 1-year licence (term_column = one_year_term_value); every row when no term_column is
    configured. term_column is a mandatory dim (checked when the Config is built): it is in every step."""
    if not configuration.term_column:
        return pd.Series(True, index=rows.index)
    return rows[configuration.term_column].astype(str) == str(configuration.one_year_term_value)


ROLE_PURPOSES = [
    (COLUMN_ROLE_PERIOD, "the month of the row"),
    (COLUMN_ROLE_TIME_SERIES_FLAG, "1 = the time_series universe (retail to subscription), projected apart (step 20)"),
    (COLUMN_ROLE_MEASURE, "summed: units and USD due, renewed, reacquired"),
    (COLUMN_ROLE_MANDATORY, "open the forecast series and the uplift cell; collapsed in stage 3 of the ladder"),
    (COLUMN_ROLE_TIMEVARYING, "signals of the customer, summarised by their sign in stage 1"),
    (COLUMN_ROLE_EXTRA_RENOVACION, "open the forecast series; annulled in stage 2"),
    (COLUMN_ROLE_EXTRA_REVALORIZACION, "open the uplift cell (the price), not the rate"),
    (COLUMN_ROLE_BOTH_EXTRAS, "both extras"),
    (COLUMN_ROLE_FORMULA_INPUT, "inputs of a formula (exact discount, SKU), not a dimension"),
    (COLUMN_ROLE_IGNORE, "read and not used")]
LEVELED_ROLE = "niveles generados (01)"
GENERATED_LEVEL_SUFFIX = "_level_1"           # the coarse level the library generates for a leveled dim
LEVEL_TYPES = ("ordinal", "nominal")          # ordinal: only neighbouring values merge · nominal: any two


def roles_overview(columns, configuration) -> pd.DataFrame:
    """Every role of the Config, in a fixed order, with its columns (0 when the role is empty), what it is for,
    and the dimensions that get generated levels (level_1 grouped by the library, level_2 the raw value)."""
    columns = list(columns)
    present = set(columns)
    roles = configuration.column_roles()                 # in the order the Config declares them
    rows = []
    for role, purpose in ROLE_PURPOSES:
        role_columns = [column for column, column_role in roles.items() if column_role == role and column in present]
        if role == COLUMN_ROLE_BOTH_EXTRAS and not role_columns:
            continue
        rows.append({"rol": role, "columnas": len(role_columns), "nombres": ", ".join(role_columns), "para_que": purpose})
    leveled = [f"{column_name} → {column_name}{GENERATED_LEVEL_SUFFIX} ({level_type})"
               + (" · generado" if f"{column_name}{GENERATED_LEVEL_SUFFIX}" in present else "")
               for column_name, level_type in configuration.leveled_dims.items()]
    if leveled:
        rows.append({"rol": LEVELED_ROLE, "columnas": len(leveled), "nombres": "; ".join(leveled),
                     "para_que": "the column keeps its raw value (fine level); _level_1 groups its values by their standardised "
                                 "rate (JSON in levels_path); the ladder collapses the fine level first"})
    return pd.DataFrame(rows)


def region_columns_of(configuration) -> list:
    """The region levels of the time_series universe, coarse to fine (the first mandatory dim if none is
    declared). Used by step 01 (it separates the universe) and step 20 (it projects it)."""
    return list(configuration.ts_region_columns) or [configuration.business_mandatory_dims[0]]


def forecast_column_of_closed_month_measure(measure_column: str) -> str:
    """The name in the final block of the core of a closed-month measure, like the rest of the block: total_
    becomes forecast_, tr becomes to_renew and usd becomes USD (total_tr_units_without_softcancel →
    forecast_to_renew_units_without_softcancel; total_renewed_usd_without_softcancel →
    forecast_renewed_USD_without_softcancel)."""
    name = measure_column[len("total_"):] if measure_column.startswith("total_") else measure_column
    renamed_words = {"usd": "USD", "tr": "to_renew"}
    words = [renamed_words.get(word, word) for word in name.split("_")]
    return "forecast_" + "_".join(words)


def total_column_of_closed_month_measure(measure_column: str, configuration) -> str:
    """The measure a closed-month measure is a part of, by its name: the due or the renewed side, in units or
    in USD (total_tr_units_without_softcancel is a part of the units due). Step 01 checks it is not above it."""
    words = measure_column.lower().split("_")
    in_usd = "usd" in words
    if "tr" in words and "renewed" in words and in_usd and configuration.renewed_pipeline_usd_col:
        return configuration.renewed_pipeline_usd_col        # the renewers' due value of a subset: a part of the exact base
    if "tr" in words:
        return configuration.pipeline_usd_col if in_usd else configuration.pipeline_units_col
    return configuration.renewed_usd_col if in_usd else configuration.renewed_units_col


def join_columns(frame: pd.DataFrame, columns: list) -> pd.Series:
    """The "|"-joined id of every row from several columns, in the given order (the order
    is part of the id); a null value is written "null". Built on arrays, not Series, so a
    duplicated index cannot misalign it."""
    joined_ids = id_text(frame[columns[0]])
    for column_name in columns[1:]:
        joined_ids = joined_ids + ID_FIELD_SEPARATOR + id_text(frame[column_name])
    return pd.Series(joined_ids, index=frame.index)


def id_text(values: pd.Series) -> np.ndarray:
    """The values of one column as the text they take inside an id; null → "null"."""
    texts = values.astype(str).to_numpy(dtype=object)
    texts[values.isna().to_numpy()] = NULL_ID_TEXT
    return texts


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

    # ─── where the raw comes from ───────────────────────────────────────────────────
    raw_extract_sql: Optional[str] = None # the SQL of the extract (or a table name): read_raw may use it, and the
                                          # diagnostics write their reproduction queries on top of it

    # ─── the columns of the raw, by role ────────────────────────────────────────────
    period_col: str = "period"                              # the month of the row
    pipeline_units_col: str = "total_tr_units"              # subscriptions falling due
    pipeline_usd_col: str = "total_tr_usd"                  # their value at due date
    renewed_units_col: str = "total_renewed_units"          # subscriptions renewed
    renewed_usd_col: str = "total_renewed_usd"              # their value at renewal
    extra_measure_cols: list = field(default_factory=list)  # other measures (reacquisitions, AUVs): declared, not used yet
    renewed_pipeline_usd_col: Optional[str] = None  # the USD that was FALLING DUE of the contracts that RENEWED
                                                    # (built per licence, then summed): the exact base of the uplift.
                                                    # Only known once the month closes, so it gets the closed-month
                                                    # treatment below. None = not in the extract: the uplift falls back
                                                    # to renewed units × the row's average due price (an approximation
                                                    # that is exact only if renewers were worth the row's average)
    closed_month_measure_cols: list = field(default_factory=list)  # measures only known once the month CLOSES (e.g. the units
                                                                   # due, renewed units and renewed USD of those who never had a
                                                                   # softcancel, before, during or after the renewal): treated like
                                                                   # the renewals (null in a closed month = 0, wiped from the
                                                                   # current month on) and carried to the core as forecast_* in
                                                                   # the closed months
    flag_time_series_col: str = "flag_time_series"          # marks the rows of the time_series universe

    business_mandatory_dims: list = field(default_factory=list)       # open the series and the uplift cell
    leveled_dims: dict = field(default_factory=dict)  # mandatory dims that get a generated coarse level (step 01):
                                                      # {column: "ordinal" | "nominal"}. The column keeps its raw value
                                                      # (the fine level); <column>_level_1 groups its values by their
                                                      # standardised renewal rate and is added to the mandatory dims
                                                      # right after it, when the Config is built
    levels_path: Optional[str] = None                 # the JSON of the generated groups (None: <output_folder>/sff_levels.json);
                                                      # a later run reuses it; delete it to regenerate
    level_merge_max_pp: Optional[float] = None        # two neighbouring values merge while their standardised rates differ
                                                      # by at most this (pp). None: the binomial noise of a series at the
                                                      # support floor, 100·√(p(1−p)/support_floor), p = the training rate
    save_checkpoints: bool = True                     # every step saves its tables, so a later run can start from any step
    checkpoint_folder: Optional[str] = None           # where (None: <output_folder>/checkpoints); one .pkl per table + manifest
    structural_timevarying_dims: dict = field(default_factory=dict)   # column → "negative" | "positive"
    extra_renovacion: list = field(default_factory=list)              # enter the rate series only
    extra_revalorizacion: list = field(default_factory=list)          # enter the uplift cell only

    uplift_mandatory_dims: Optional[list] = None      # mandatory dims of the uplift cell; None = all of them
    # the discount: one column, both sides (see the module header). None = no discount in the extract
    discount_value_column: Optional[str] = "discount"          # the exact discount, a share (0.25 = 25 %)
    discount_bucket_edges: list = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # in %
    discount_bucket_column: str = "tramo_descuento"            # the derived bucket (a dimension of the uplift cell)
    sku_column: Optional[str] = None              # the SKU (formula input for the price scenarios)
    ignore_cols: list = field(default_factory=list)   # read by nobody; may be absent from the raw

    # ─── the calendar (the role of every month is generated from it) ────────────────
    current_month: Optional[str] = None   # first month of the future; no default on purpose. If the last months
                                          # are not mature yet (late renewals still arriving), set current_month to the
                                          # first immature one: it becomes proyeccion, is predicted, and its partial
                                          # renewals are wiped in step 02. "Closed" always means: before current_month
    test_months: int = 3                  # closed months before it that only evaluate

    # ─── statistical parameters ────────────────────────────────────────────────────
    z: float = 1.645                      # 90 % two-sided: every band and every binomial error uses it
    support_floor: float = 30.0           # contracts in a typical month below which a series' own rate
                                          # moves ±15 pp by chance (90 %, p = 0.5): it will borrow support

    dimension_pairs_shown: int = 20       # step 09: pairs of dimensions kept, the ones with the most interaction
    # ─── the ladder (steps 10 and 11) ───
    own_rate_floor: float = 271.0
    collapse_passes: int = 2            # stage 3 merges at most this many mandatory dims (collapse order); the passes
                                        # after them are only searched for a credibility reference (stage 4)         # contracts in a typical month to predict ALONE: ±5 pp at 90 % (p = 0.5).
                                          # 30 says who may speak; 271 who may speak alone
    k_cred: float = 60.0                  # Bühlmann k when a relative has too few siblings to estimate it: a series
                                          # with n = 30 keeps 33 % of its own rate ("twice the floor to be believed half")
    close_relative_max_rung: int = 2      # rungs 1-2 share every mandatory dim (close: level B); 3 and up are far (C)
    own_level_min_history_months: int = 12    # a full year to be level A (a seasonal series has seen every season)
    signed_ladder_max_loss: float = 0.05  # a signed series may collapse mandatory dims (sign kept) while the cumulative
                                          # R² lost (step 09) stays ≤ this; 0 = its ladder ends at the cell × sign
    # ─── the dynamics of the rate (step 13): descriptive, it does NOT restrict any technique ───
    dynamics_min_months: int = 24         # months a pool needs to measure its month effect (two of each month)
    dynamics_significance: float = 0.05   # a month effect or a trend is declared when its p-value is below this
    # ─── the uplift (steps 15 and 16) ───
    uplift_floor: float = 30.0            # renewers a cell needs to use its own uplift; below, its parent's (tramo kept)
    uplift_cap: float = 3.0               # an uplift is clipped to [0.01, 3]: a renewer never pays 3× what fell due
    uplift_bootstrap_samples: int = 200   # resamples of the renewer rows for the uplift band (p5-p95)
    random_seed: int = 42                 # the bootstrap is reproducible
    contract_apply_realization_ratio: bool = False   # contract path: 1/(1−d) as is (False) or × the cell's observed ratio
    # ─── the forecast (step 17) ───
    # the SIMULATION WINDOW: from the current month (included) to simulation_end. What happens in it
    # (renewals of 1-year licences, acquisitions, time_series conversions) falls due 12 months later
    simulation_end: Optional[str] = None          # the last month simulated; None = December of the current month's year
    term_column: Optional[str] = None             # the column with the term of the licence; None = every licence is of 1 year
    one_year_term_value: str = "1 year"           # the value of term_column that means a 1-year licence (only those re-enter)
    renewal_reentry_discount: float = 0.0         # a renewal falls due again at the price it renewed: no discount
    acquisition_column: Optional[str] = None      # the column that says a row is an acquisition; None = no acquisition simulated
    acquisition_values: list = field(default_factory=list)      # its values that are acquisition: each one is simulated apart
                                                                # (they have different proportions)
    acquisition_discount: float = 0.4             # an acquisition is sold with this discount (it renews without it)
    acquisition_level_window_months: int = 3      # acquisition level: last 3 closed months vs the same months a year before
    acquisition_auv_window_months: int = 12       # acquisition value per unit: Σ value / Σ units of the last 12 closed months
    # the confidence of a future row (step 17): how sure the forecast is of its rate
    confidence_high_band_pp: float = 5.0      # high: its rate band within ±5 pp (the precision of own_rate_floor)…
    confidence_high_exam_pp: float = 5.0      # …its composition judged by the backtest, with an exam error ≤ 5 pp
    confidence_medium_band_pp: float = 10.0   # medium: judged, and a band within ±10 pp; low: the rest (or not judged)
    apply_credibility_shift: bool = True  # a group below own_rate_floor moves its predicted rate toward its credibility
                                          # reference by (1 − z) of their difference of level (logit scale); False = the
                                          # group's prediction as is

    # ─── the spreadsheet baseline (step 19): what the business does today ───
    baseline_months: int = 12             # the rate of the last N closed months… (what the business does: 12 months per cell)
    baseline_grains: list = field(default_factory=lambda: ["mandatory", "global"])   # …per grain: "global", "mandatory"
                                          # (every mandatory dim) or columns joined with "+" (e.g. "region+product")
    # ─── the time_series universe: retail to subscription (step 20) ───
    ts_region_columns: list = field(default_factory=list)   # the region levels of a time_series row, coarse to fine;
                                              # [] = the first mandatory dim. The projection is made at the FINEST level and
                                              # the renewal rate climbs fine → coarse → global; the same columns must be in
                                              # the normal pipeline (its rate is read there)
    ts_level_window_months: int = 3           # the level: the last 3 closed months of this year vs the same months of last
                                              # year (recent enough to follow the campaigns, long enough not to be one month)
    ts_auv_window_months: int = 12            # the value per unit: Σ value / Σ units of the last 12 closed months (a year)
    ts_rate_window_months: int = 12           # the renewal rate of the region in the normal pipeline: last 12 closed months
    ts_acquisition_discount: float = 0.4      # retail-to-subscription offers: acquired at 40 % off, renewed at 100 %
    # ─── the backtest of the rate (step 14) ───
    challenger_technique: str = "T3_ma3"  # the technique to beat: the moving average of 3 months (what a spreadsheet does)
    backtest_selection_months: int = 6    # closed months BEFORE the exam used as targets to CHOOSE the technique
    backtest_horizons: list = field(default_factory=lambda: [1, 6])   # months ahead judged: next month, and six months
    horizon_bands: dict = field(default_factory=lambda: {"corto": [1, 1], "medio_largo": [2, 6]})   # a horizon between
                                          # two judged ones takes the band of the judged one above it (the wider)
    challenger_margin_by_band: dict = field(default_factory=lambda: {"corto": 0.10, "medio_largo": 0.0})  # how much a
                                          # technique must beat the challenger (in |error| / binomial error) to replace it
    backtest_min_history_months: int = 12 # months a technique must see before predicting (every technique can compete)
    backtest_min_predictions: int = 3     # predictions a technique needs in a band to be chosen
    band_low_quantile: float = 0.05       # the error band: 5 % and 95 % quantiles of the normalised error (90 %)
    band_high_quantile: float = 0.95

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

    def read_time_series(self) -> Optional[pd.DataFrame]:
        """The history of the time_series universe (retail-to-subscription conversions), one row
        per region × month with the columns: period, region, unidades, valor. Overridden by the
        Config of main.py when it comes from its own query. None (the default) = step 20 takes it
        from the rows of the raw with the time_series flag."""
        return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE COLUMN CONTRACT
    # ═══════════════════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        """The logger is configured, and a misdeclared Config stops here, before any data is read."""
        # [0] the log: one configuration for the whole run; the count of the checks of every
        #     step is kept for the report
        LoggerManager(self.log_level, use_colors=self.log_colors, log_file=self.log_file).get_logger_configured(LOGGER_NAME)
        self.check_history = []

        # [1] every timevarying column carries a valid sign
        invalid_signs = {column_name: sign for column_name, sign in self.structural_timevarying_dims.items()
                         if sign not in VALID_TIMEVARYING_SIGNS}
        if invalid_signs:
            raise ValueError(f"timevarying signs must be one of {VALID_TIMEVARYING_SIGNS}: {invalid_signs}")

        # [2] the leveled dims: mandatory, with a valid type; their generated level is a mandatory dim from
        #     here on (step 01 creates it before any step reads the dims)
        not_mandatory = [column_name for column_name in self.leveled_dims if column_name not in self.business_mandatory_dims]
        if not_mandatory:
            raise ValueError(f"leveled_dims must be mandatory dims: {not_mandatory}")
        invalid_types = {column_name: level_type for column_name, level_type in self.leveled_dims.items()
                         if level_type not in LEVEL_TYPES}
        if invalid_types:
            raise ValueError(f"leveled_dims types must be one of {LEVEL_TYPES}: {invalid_types}")
        expanded = []
        for column_name in self.business_mandatory_dims:
            if column_name in self.generated_columns:
                continue                                   # re-inserted right after its column
            expanded.append(column_name)
            if column_name in self.leveled_dims:
                expanded.append(f"{column_name}{GENERATED_LEVEL_SUFFIX}")
        self.business_mandatory_dims = expanded

        # [3] the term of a licence is a declared mandatory dim (it exists in every step)
        if self.term_column and self.term_column not in self.business_mandatory_dims:
            raise ValueError(f"term_column '{self.term_column}' must be one of the mandatory dims")

        # [4] no column declared with two roles (built and checked by column_roles), and
        #     the uplift cell takes its mandatory dims from the declared ones
        self.column_roles()
        unknown_uplift_dims = [column_name for column_name in (self.uplift_mandatory_dims or [])
                               if column_name not in self.business_mandatory_dims]
        if unknown_uplift_dims:
            raise ValueError(f"uplift_mandatory_dims must be mandatory dims: {unknown_uplift_dims}")

        # [5] the calendar parameters are well formed
        if self.current_month is not None:
            parse_month(self.current_month)
        edges = list(self.discount_bucket_edges)
        if edges != sorted(set(edges)) or edges[0] != 0 or edges[-1] != 100:
            raise ValueError(f"discount_bucket_edges must go up strictly from 0 to 100 (in %): {edges}")
        for discount_name in ("acquisition_discount", "renewal_reentry_discount"):
            if not (0 <= float(getattr(self, discount_name)) < 1):
                raise ValueError(f"{discount_name} must be in [0, 1) (found {getattr(self, discount_name)})")
        if not (0 <= float(self.ts_acquisition_discount) < 1):
            raise ValueError(f"ts_acquisition_discount must be in [0, 1) (found {self.ts_acquisition_discount})")
        if int(self.test_months) < 1:
            raise ValueError(f"test_months must be at least 1 (found {self.test_months}): without an exam nothing is evaluated")

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
        self.check_history.append({"paso": step_label, "nombre": step_name, "comprobaciones": len(check_log),
                                   "ok": counts[STATUS_OK], "avisos": counts[STATUS_WARNING],
                                   "fallos": counts[STATUS_FAILED], "no_evaluadas": counts[STATUS_NOT_EVALUATED],
                                   "avisos_detalle": " · ".join(description for status, description, _ in check_log
                                                                if status in (STATUS_WARNING, STATUS_FAILED))})
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
        back are not the rows written. If a blocking check of the step has already failed,
        nothing is written and the write is logged as not evaluated.
        """
        physical_name = f"{self.table_prefix}{table_name}"

        # a table built from data that already failed a blocking check is not written:
        # it would replace a good table with a wrong one, and it costs time for nothing
        if any(logged[0] == STATUS_FAILED for logged in check_log):
            self.log_not_evaluated(step_label, check_log, f"table {physical_name} written and read back",
                                   "not written: an earlier check of this step failed")
            return

        persisted_frame = frame.copy()
        for column_name in persisted_frame.columns:
            if isinstance(persisted_frame[column_name].dtype, pd.PeriodDtype):
                persisted_frame[column_name] = persisted_frame[column_name].astype(str)

        write_start_time = time.time()
        if self.sql_engine is not None:
            # chunks of 10,000 rows; with an engine created with fast_executemany=True (see main.py),
            # pyodbc sends each chunk in one round trip instead of one INSERT per row
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

    def show_query(self, step_label: str, what: str, sql: str) -> None:
        """The SQL that reproduces an aggregated report from the tables the run writes (report_queries.py)."""
        self.logger.doc(f"[{step_label}] {what}: the SQL that reproduces it from the tables of the run:\n{sql}")

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
    def generated_columns(self) -> list:
        """The columns the library adds to the raw (step 01): the coarse level of every leveled dim."""
        return [f"{column_name}{GENERATED_LEVEL_SUFFIX}" for column_name in self.leveled_dims]

    @property
    def extract_mandatory_dims(self) -> list:
        """The mandatory dims the extract brings (the steps that validate the extract, 00 and 01, read these;
        from step 01 on, business_mandatory_dims, with the generated levels)."""
        return [column_name for column_name in self.business_mandatory_dims if column_name not in self.generated_columns]

    @property
    def all_closed_month_measure_cols(self) -> list:
        """Every measure with the closed-month treatment (null closed = 0, wiped from the current month on,
        carried to the core as forecast_*): the declared list plus the uplift base, when there is one."""
        exact_uplift_base = [self.renewed_pipeline_usd_col] if self.renewed_pipeline_usd_col else []
        return list(self.closed_month_measure_cols) + exact_uplift_base

    @property
    def rate_series_columns(self) -> list:
        """The columns that define ONE renewal-rate series: mandatory + timevarying +
        extra_renovacion, in this order (the field order of fs_id and fu_id)."""
        return (self.business_mandatory_dims + list(self.structural_timevarying_dims)
                + self.extra_renovacion)

    @property
    def dimension_columns(self) -> list:
        """Every dimension, each once, in the order of the taxonomy: mandatory, timevarying,
        extra_renovacion, extra_revalorizacion."""
        return list(dict.fromkeys(self.business_mandatory_dims + list(self.structural_timevarying_dims)
                                  + self.extra_renovacion + self.extra_revalorizacion))

    @property
    def simulation_window(self) -> list:
        """The months simulated: from the current month (included) to simulation_end (December by default)."""
        current = self.calendar_boundaries()["current"]
        end = parse_month(self.simulation_end) if self.simulation_end else pd.Period(f"{current.year}-12", freq="M")
        return list(pd.period_range(current, end, freq="M")) if end >= current else []

    @property
    def uplift_cell_columns(self) -> list:
        """The columns of ONE uplift cell (the price context): the uplift mandatory dims
        (all mandatory unless uplift_mandatory_dims says fewer) + extra_revalorizacion +
        the discount bucket derived from the exact discount (when there is one)."""
        uplift_mandatory = self.uplift_mandatory_dims if self.uplift_mandatory_dims is not None else self.business_mandatory_dims
        bucket = [self.discount_bucket_column] if self.discount_value_column else []
        return list(uplift_mandatory) + self.extra_revalorizacion + bucket

    @property
    def fine_row_columns(self) -> list:
        """What makes a fine row unique inside its forecast unit: the extra_revalorizacion
        values and the exact discount (the extract comes one row per exact discount)."""
        exact_discount = [self.discount_value_column] if self.discount_value_column else []
        return self.extra_revalorizacion + exact_discount

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
        for measure_column in self.core_measures + list(self.extra_measure_cols) + self.all_closed_month_measure_cols:
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
        """The two months that cut the calendar: current (the first month of projection; every
        month before it is closed) and test_start (the first exam month: current − test_months).
        Raises ValueError when current_month is not declared."""
        if self.current_month is None:
            raise ValueError("current_month is not declared: set it in your Config "
                             "(e.g. current_month=\"2026-09\" or \"01/09/2026\")")
        current = parse_month(self.current_month)
        test_start = current - int(self.test_months)
        return dict(current=current, test_start=test_start)

    def role_of_months(self, periods) -> np.ndarray:
        """The role of every month: ≥ current → proyeccion · ≥ test_start → examen ·
        earlier → entrenamiento."""
        boundaries = self.calendar_boundaries()
        month_values = pd.Series(periods).reset_index(drop=True)
        # np.select takes the FIRST condition that holds, so the order is the rule
        conditions = [month_values >= boundaries["current"],
                      month_values >= boundaries["test_start"]]
        return np.select(conditions, [ROLE_PROJECTION, ROLE_TEST], default=ROLE_TRAIN)

    def calendar_description(self) -> str:
        """The calendar in one line, for the console."""
        boundaries = self.calendar_boundaries()
        current, test_start = boundaries["current"], boundaries["test_start"]
        return (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{current - 1} · "
                f"{ROLE_PROJECTION} ≥ {current}")
