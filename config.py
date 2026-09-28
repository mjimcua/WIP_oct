"""config.py — SFF v3 · the configuration contract.

This module is the single place where the framework learns four things:

  1. TAXONOMY AND VALIDATION   which raw column plays which role (DISENO_V2 §3):
                               mandatory · timevarying (with sign) · extra_renovacion ·
                               extra_revalorizacion · measures · ignore. Every raw
                               column must be declared in exactly one role, or the
                               program stops (principle P3, the exhaustive contract).
  2. SERIES AND CELL COLUMNS   which columns define a rate series and an uplift cell.
                               DERIVED from the taxonomy, never typed:
                               rate_series_columns = mandatory + timevarying + extra_renovacion
                               uplift_cell_columns = mandatory + extra_revalorizacion
  3. KEYS AND IDS              how a row becomes an id ('|'-joined fields) and how an
                               id becomes a deterministic int64 key (P7).
  4. SQL PERSISTENCE           how a star-schema table is written back: physical name
                               registry with the `sff_` prefix, process_date and
                               execution_id stamped on every row, TRUNCATE/replace
                               semantics, fast_executemany on the pyodbc cursor.

  RAW SOURCE                   `read_raw()` returns the raw DataFrame and MUST be
                               overridden by the caller's main (production: its own
                               query; tests: the synthetic dataset). Nothing in `run/`
                               reads a file or a query.

Every tunable parameter lives in the TUNABLE PARAMETERS block of `Config`, grouped by
phase, each with its purpose, the reason for its default and what to look at before
changing it. PARAMETROS.md is the reader-facing version of that block.

Vocabulary: `extra_renovacion` / `extra_revalorizacion` are the doctrine names of the
two extra groups (the word "covariates" is forbidden). Persisted column names and
values stay in Spanish because they are the contract of the reference output and of the BI.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import datetime
import hashlib
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.types import NVARCHAR


# ─── named constants ─────────────────────────────────────────────────────────────
# Accepted signs for a timevarying dimension: "negative" rotates towards churn (e.g.
# softcancel), "positive" rotates towards renewal (e.g. autorenew).
VALID_TIMEVARYING_SIGNS = ("negative", "positive")

# Separator between the fields of an id. Ids are '|'-joined in the order
# `mandatory | timevarying | extra_renovacion | month` for a forecast unit.
ID_FIELD_SEPARATOR = "|"

# Separator between fu_id and comb_id when they are fused into one single-column key
# (`fu_comb_key`), so the BI can join the raw anchor to the key bridge with ONE
# relationship.
COMBINED_ID_SEPARATOR = "||"

# The key of an id is the first 12 hex digits of its MD5, read as an integer: it fits
# a SQL bigint and is reproducible across runs, machines and languages (P7).
HASH_HEX_DIGITS = 12

# Length of the execution identifier stamped on every persisted row.
EXECUTION_ID_LENGTH = 10

# Sizing rule for NVARCHAR columns on SQL Server: observed max length × 1.3 + 4,
# never below 8 and never above the NVARCHAR(4000) ceiling.
NVARCHAR_GROWTH_FACTOR = 1.3
NVARCHAR_GROWTH_PADDING = 4
NVARCHAR_MIN_LENGTH = 8
NVARCHAR_MAX_LENGTH = 4000

# Floor of the write stopwatch so rows/second never divides by zero.
MIN_ELAPSED_SECONDS = 0.01

# Logical table name → physical suffix. The physical table is `sql_table_prefix` +
# suffix (e.g. `sff_fu_summary`), unless `sql_table_names` overrides the logical name.
# This registry is a datum of the contract: the reference output and the BI expect these names.
PHYSICAL_TABLE_NAMES = {
    # ── phase 0 · raw
    "fact_fu": "fact_fu",
    "fact_fine": "fact_fine",
    "fact_fu_gaps": "fact_fu_gaps",
    "lookup_fu": "lookup_fu",
    "lookup_comb": "lookup_comb",
    "forecast_units_raw_summary": "fu_summary",
    "raw_profile": "raw_profile",
    "dimension_domains": "dim_domains",
    "fu_profile": "fu_profile",
    "dial_buckets": "dial_buckets",
    # ── phase 1 · series, ladder, card
    "forecast_series_raw_summary": "fs_summary",
    "series_card": "series_card",
    "risk_levels_report": "risk_levels",
    "parent_ladder": "parent_ladder",
    "support_chain": "support_chain",
    "decision_support": "decision_support",
    "decision_eta2": "decision_eta2",
    "decision_eta2_pairs": "decision_eta2_pairs",
    "mandatory_only_cost": "mandatory_only_cost",
    "mix_shift_decomposition": "mix_shift",
    "timevarying_calibration": "tv_calibration",
    # ── phase 2 · seasonality benchmark and pool reference
    "decision_estacionalidad": "decision_estacionalidad",
    "bench_panel_mes_anio": "bench_panel",
    "bench_flags": "bench_flags",
    "pool_reference": "pool_reference",
    # ── phase 3 · backtest
    "dim_tecnica": "dim_tecnica",
    "backtest_predictions": "backtest_pred",
    "backtest_holdout": "backtest_holdout",
    "backtest_holdout_aggregate": "backtest_holdout_agg",
    "backtest_aggregate_error": "backtest_agg_error",
    "decision_aggregate_bands": "decision_agg_bands",
    "decision_technique": "decision_technique",
    "decision_error_bands": "decision_error_bands",
    # ── phase 4 · uplift
    "uplift_chain": "uplift_chain",
    "decision_uplift": "decision_uplift",
    # ── phase 5 · assembly
    "key_bridge": "key_bridge",
    "forecast_units_extended": "fu_extended",
    "forecast_detail": "forecast_detail",
    "forecast_bands": "forecast_bands",
    "horizon_report_total": "horizon_report_total",
    "forecast_by_level": "forecast_by_level",
    "pipeline_summary": "pipeline_summary",
    "business_summary": "business_summary",
    "forecast_by_region": "forecast_by_region",
    "top_movers": "top_movers",
    "signal_composition": "signal_composition",
    "signal_snapshot": "signal_snapshot",
    "signal_final_composition": "signal_final_comp",
    "signal_adjustment": "signal_adjustment",
    "signal_alerts": "signal_alerts",
    "metric_legend": "metric_legend",
    "discount_churn_by_bucket": "discount_churn_bucket",
    "discount_churn_by_state": "discount_churn_state",
    "discount_churn_adjusted": "discount_churn_adjusted",
    "discount_churn_importance": "discount_churn_importance",
    "discount_churn_price": "discount_churn_price",
    "uplift_contract_check": "uplift_contract_check",
    "uplift_missing_check": "uplift_missing_check",
    "backtest_uplift": "backtest_uplift",
    "escenarios_precio": "escenarios_precio",
    "escenarios_margen": "escenarios_margen",
    "nucleo": "nucleo",
    "topologia": "topologia",
    "nucleo_reconciliacion": "nucleo_reconciliacion",
    "panorama_resumen": "panorama_resumen",
    "panorama_mensual": "panorama_mensual",
    "panorama_mercado": "panorama_mercado",
    "panorama_diagnostico": "panorama_diagnostico",
    "panorama_volumen_ts": "panorama_volumen_ts",
    "efecto_composicion": "efecto_composicion",
    "efecto_composicion_celda": "efecto_composicion_celda",
    "efecto_composicion_futuro": "efecto_composicion_fut",
    "efecto_composicion_celda_futuro": "efecto_composicion_celda_fut",
    "baseline_forecast": "baseline_forecast",
    "baseline_summary": "baseline_summary",
    "validation_report": "validation_report",
}


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 3 · KEYS AND IDS
# (module-level functions: every phase imports them; they do not depend on Config)
# ═══════════════════════════════════════════════════════════════════════════════════

def join_columns(frame: pd.DataFrame, columns: list) -> pd.Series:
    """Build the '|'-joined id of every row from several columns, vectorized.

    INPUT:   frame — any DataFrame · columns — ordered list of column names to join.
    OUTPUT:  Series of str, one id per row, indexed like `frame`.
    RULES:   fields are joined in the order given, cast with `astype(str)`, with
             ID_FIELD_SEPARATOR between them. The ORDER is contractual: the same
             columns in another order produce a different id and a different key.
    EDGE CASES: `columns` must be non-empty. Works on NumPy arrays, never on Series:
             adding Series aligns by index, and a frame coming out of a merge can carry
             duplicate index labels (it would raise or silently misalign). Measured at
             600k×10 rows: ~3 s here versus ~46 s with the row-by-row
             `frame[cols].astype(str).agg("|".join, axis=1)` (x16).
    CONSOLE: nothing.
    STEPS:
      [1] Start from the first column as an object array of strings.
      [2] Append every remaining column with the separator in front.
      [3] Return a Series aligned to the frame's index.
    """
    # [1] first field: the seed of the id, as a plain object array
    joined_ids = frame[columns[0]].astype(str).to_numpy(dtype=object)

    # [2] every further field is glued after a separator, still on arrays
    for column_name in columns[1:]:
        next_field = frame[column_name].astype(str).to_numpy(dtype=object)
        joined_ids = joined_ids + ID_FIELD_SEPARATOR + next_field

    # [3] back to a Series that carries the caller's index
    return pd.Series(joined_ids, index=frame.index)


# What each table is FOR: "nucleo" (THE table: the whole story, one row per fine
# combination × month, steps s0 → s7 left to right — Power BI needs only this one to
# rebuild every rate by aggregation), "producto" (model outputs and studies the report
# reads), "bi" (the older fact/lookup tables, kept while the reports migrate to the core);
# everything else is "intermedia" (decisions and traces the framework reads back).
TABLE_KIND = {
    "nucleo": ["nucleo", "nucleo_reconciliacion"],
    "producto": ["pipeline_summary", "business_summary", "forecast_by_region", "forecast_by_level", "horizon_report_total",
                 "top_movers", "risk_levels_report", "decision_estacionalidad", "baseline_summary", "signal_adjustment",
                 "signal_alerts", "validation_report", "dial_buckets", "metric_legend", "panorama_resumen", "panorama_mensual",
                 "panorama_mercado", "panorama_diagnostico", "panorama_volumen_ts", "efecto_composicion", "efecto_composicion_futuro",
                 "backtest_uplift", "escenarios_precio", "escenarios_margen"],
    "bi": ["fact_fu", "fact_fine", "lookup_fu", "lookup_comb", "key_bridge", "forecast_detail",
           "forecast_bands", "series_card", "forecast_units_extended", "signal_snapshot", "mix_shift_decomposition"],
}


def table_kind(logical_name: str) -> str:
    """'producto', 'bi' or 'intermedia' for a logical table name."""
    for kind, names in TABLE_KIND.items():
        if logical_name in names:
            return kind
    return "intermedia"


def explain(configuration, *lines: str) -> None:
    """Console: how to read the block just printed (two or three lines, indented). Silent
    when `console_explanations` is off."""
    if getattr(configuration, "console_explanations", True):
        for line in lines:
            print(f"       > {line}")


def hash_key(identifier) -> int:
    """Deterministic int64 key of an id (P7: keys are derived, never assigned).

    INPUT:   identifier — anything; it is cast with `str()` first.
    OUTPUT:  int, the first HASH_HEX_DIGITS (12) hex digits of MD5(str(identifier)),
             read as a base-16 integer.
    RULES:   the same id always yields the same key, on any machine and in any run.
             12 hex digits = 48 bits, so the key fits a SQL bigint.
    EDGE CASES: collisions are possible in theory (48 bits) and have never been observed;
             fu_id uniqueness is asserted in phase 0 on the id, not on the key.
    CONSOLE: nothing.
    STEPS:
      [1] Cast to str and encode as UTF-8 bytes.
      [2] MD5 hex digest, keep the first 12 hex digits.
      [3] Parse them as an integer.
    """
    # [1] canonical text form of the id
    identifier_bytes = str(identifier).encode()

    # [2] first 12 hex digits of the MD5 digest
    digest_prefix = hashlib.md5(identifier_bytes).hexdigest()[:HASH_HEX_DIGITS]

    # [3] the key is that prefix read in base 16
    return int(digest_prefix, 16)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """The single declaration of what the raw data IS and where the results GO.

    PURPOSE. Every phase of the framework needs to answer the same four questions, and
    none of them should answer any of them on its own: which column plays which role,
    which columns define a series and a cell in each branch, how a row becomes an id and a key, and under
    what name a result table is persisted. `Config` is the one place where those
    answers live, so that a change of taxonomy, of series definition or of destination is a change
    in ONE file and not a search across five phases.

    WHAT IT GUARANTEES.
      · Exhaustiveness (P3): a raw column with no declared role stops the run. There is
        no "unknown column" path, and therefore no column silently ignored.
      · Consistency of the taxonomy (DISENO_V2 §3): the exclusion rules are checked at
        construction, so a misdeclared Config never reaches a phase.
      · Derived definitions: the columns of a rate series and of an uplift cell are
        computed from the taxonomy by formula, never typed by hand; they cannot drift
        apart from it.
      · Reproducible identity: the same row yields the same id and the same key in any
        run, on any machine — the keys in SQL survive a re-execution.
      · Traceable persistence: every written row carries `process_date` and
        `execution_id`, and every table lands under the physical name of the registry.

    WHAT IT DOES NOT DO. It does not know where the raw comes from (`read_raw` is
    overridden by the caller's main), does not compute
    any forecast quantity, and does not decide anything measured: η², chosen technique
    and error bands are decisions of ANALYSIS, persisted in the decision tables
    (DISENO_SPLIT §3). Only the versioned business parameters live here
    (the TUNABLE PARAMETERS block below, grouped by phase):
    they are decisions of the repository, not measurements of a run.

    USAGE. Subclass it in your main and override `read_raw()`; `Config()` fields with
    no arguments are production. Tests pass a smaller taxonomy and inject a sqlite
    engine. Field names are the public contract used by every phase, by the runners and
    by the regeneration prompts; they do not change.
    """

    # ─── raw column names (defaults = production, from the user's notebook) ─────────
    period_col: str = "period"
    dataset_role_col: str = "dataset_role"
    pipeline_units_col: str = "total_tr_units"
    pipeline_usd_col: str = "total_tr_usd"
    renewed_units_col: str = "total_renewed_units"
    renewed_usd_col: str = "total_renewed_usd"
    # reacquisitions are outside the renewal study but present in the raw: declared
    reacq_units_col: str = "total_reacquired_units"
    reacq_usd_col: str = "total_reacquired_usd"
    # AUV columns, if present in the raw, are declared as measures (never computed from)
    auv_pipeline_col: str = "TR_AUV"
    auv_renewed_col: str = "REN_AUV"
    auv_reacq_col: str = "ReAC_AUV"
    current_month_col: str = "is_current_month"
    flag_time_series_col: str = "flag_time_series"
    # revenue column of the time_series universe; declared for the universe contract,
    # not read by any phase yet (reserved for the time_series treatment)
    ts_revenue_col: str = "total_renewed_usd"

    # ─── taxonomy (DISENO_V2 §3) ────────────────────────────────────────────────────
    # mandatory: opens both the rate series and the uplift cell; never collapsed in pools;
    # exclusive with every other group
    business_mandatory_dims: list = field(default_factory=lambda: [
        "tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
        "tr_product_level_1", "tr_product_level_2", "tr_origin_type_SKU_based",
        "tr_term_level_1", "tr_term_level_2", "tr_band_level_1", "tr_band_level_2"])
    # timevarying: dichotomous columns that rotate with the calendar; each carries its
    # sign; handled in L1; exclusive with mandatory and with both extra groups
    structural_timevarying_dims: dict = field(default_factory=lambda: {"softcancel": "negative"})
    # which raw values mean "flag is on" for a timevarying column
    timevarying_positive_values: list = field(default_factory=lambda: [1, "1", True])
    # extra_renovacion: enters the rate series only; may overlap extra_revalorizacion
    extra_renovacion: list = field(default_factory=list)
    # extra_revalorizacion: enters the uplift cell only (former pricing columns)
    extra_revalorizacion: list = field(default_factory=lambda: [
        "price_cap", "msrp_increased", "discount_interval", "prev_OperationGroup"])
    # ignore: present in the raw, read by nobody
    ignore_cols: list = field(default_factory=lambda: ["dummy_field", "row_id", "_filter"])
    # optional semantic labels (column, value, label); empty on purpose in production
    semantic_labels: list = field(default_factory=list)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TUNABLE PARAMETERS — grouped by phase. Every field says WHAT it controls, WHY the
    # default, and WHAT TO WATCH before changing it. They are versioned decisions of the
    # repository (DISENO_SPLIT §3): a change here is a change of contract, documented
    # in ASUNCIONES.md. Parameters that are pure algorithm constants (smoothing alphas
    # of the techniques, the logit clip, the variance floor) stay in their modules and
    # are listed in PARAMETROS.md, not here.
    # ═══════════════════════════════════════════════════════════════════════════════

    # ─── the binomial reference (every phase) ──────────────────────────────────────
    # Confidence multiplier for every interval and bound. 1.645 = 90 % two-sided: the
    # level the business reads ("nine times out of ten"). 1.96 would give 95 % and bands
    # ~20 % wider; the hold-out calibration (P3.2) tells whether 90 % is honoured.
    z: float = 1.645
    # Renewal rates above this are saturated. A 95 % ceiling keeps a small series with a
    # lucky 100 % month from forecasting 100 %; no real cohort renews above it. Raise it
    # only if the calibration table (tv_calibration) shows real cohorts above 95 %.
    rate_cap: float = 0.95

    # ─── phase 0 · the calendar of roles (from the current month, not from the raw) ──
    # The raw's dataset_role is OVERWRITTEN from the current month (`is_current_month`):
    #   current month and later      → projection (results wiped: the future has not started)
    #   the month(s) just before     → pending_close: not closed yet (people renew after
    #                                  expiry); NOT used to evaluate, NOT used to learn;
    #                                  forecast like a projection month, reported apart
    #   the `test_months` before     → test: the exam (evaluation only)
    #   everything earlier           → train
    # Forecasting learns from train AND test (every closed month); only the choice of
    # technique and its bands are decided without the test months.
    pending_close_months: int = 1
    test_months: int = 6

    # ─── phase 1.1 · rate series ───────────────────────────────────────────────────

    # ─── phase 1.2 · dimension separation and mix-shift (ANALYSIS) ─────────────────
    # Months with truth judged by the walk-forward valuation of the mandatory-only view
    # (one rate per cell vs the segmented series, each month against what happened). 12 = one full year of verdicts, so a
    # seasonal cell is judged in every season. More months = more evidence, older past.
    counterfactual_window_months: int = 12
    # A cell needs this many past months before its first verdict; below it the "flat"
    # rate is itself noise and the comparison says nothing. 6 = half a year.
    counterfactual_min_history_months: int = 6
    # Pairs of dimensions kept in decision_eta2_pairs (pairs explode combinatorially with
    # many dims); the top by interaction. 15 is what fits on one screen.
    eta2_max_pairs: int = 15

    # ─── phase 1.3 · support ladder and credibility (RUN) ──────────────────────────
    # The support floor: median monthly pipeline units a relative needs to be "enough".
    # 30 is the dial at ±15 pp (90 %, p=.5): below it a month's rate says almost nothing.
    # It is the one parameter that moves the whole ladder: raise it and more series
    # borrow (and level B/C grow); lower it and more series keep noisy own rates. The
    # money by risk level (risk_levels) is the table to look at before touching it.
    support_floor: float = 30.0
    # The precision floor: a series with n_propio ≥ this predicts ALONE (z = 1); below it,
    # even with support of its own, it climbs to its first relative with support and blends
    # with z = n/(n+k). 271 is the dial at ±5 pp: the promise to the business. Between 30
    # and 271 there is evidence but not precision: it is used, weighted, and completed
    # with the pool. 30 says who may speak; 271 says who may speak alone.
    own_rate_floor: float = 271.0
    # Default credibility k (Bühlmann-Straub) when a relative has fewer than 3 siblings
    # with history, so no between/within variance can be estimated. z = n/(n+k): with
    # k=60 a series with n=30 keeps 33 % of its own rate; with n=12, 17 %. 60 ≈ two floors:
    # "you need twice the floor to be believed half". Estimated k's (decision_support.k)
    # override it wherever there are siblings.
    k_cred: float = 60.0
    # Risk level "B_prestado" vs "C_lejano": a relative at rung ≤ 2 (same sign / extra
    # annulled) is close; from rung 3 (the mandatory cell or above) it is far. The
    # distinction is the money report's, not the estimate's.
    close_relative_max_rung: int = 2
    # Months of history a series needs to be level "A_propio" even when it has support:
    # one full year, so a seasonal series has seen every season.
    own_level_min_history_months: int = 12
    # How far a series WITH SIGN may climb beyond its mandatory cell × sign, keeping the
    # sign: it may collapse mandatory dims in the sequential order while the CUMULATIVE R²
    # lost (decision_eta2.perdida_secuencial) stays ≤ this. 0.0 = the strict rule (the
    # ladder of a signed series ends at the cell × sign). 0.05 (default) lets it collapse
    # the dims that separate almost nothing (in Kamelot: band_2, band_1, product_2,
    # product_1), i.e. cohorts nearly identical — pooled signal, not mixed cohorts. With 10
    # mandatory dims the strict rule left 4,237 signed series ($8.3M) without a pool.
    signed_ladder_max_loss: float = 0.05

    # ─── phase 4 · the contract path of the uplift ─────────────────────────────────
    # Where the customer's CURRENT discount is known, the renewal price is not estimated:
    # the contract fixes it. `discount_value_column` names the raw column with the exact
    # discount in tanto por 1 (0.30 = 30 %); 0 = list price, null = unknown (never read as 0).
    # It is a formula input, not a cell dimension: it rides along every raw row (nulls
    # allowed), it is not part of any id, and it does not cut the support. The row's
    # uplift = price_increase(period) / (1 − discount) × realization ratio of its cell;
    # rows with an unknown discount (or one above `discount_cap`, near-free licences) take
    # the statistical uplift of their cell as before. None = every row is statistical.
    discount_value_column: Optional[str] = None
    # List-price increases live ONLY in the scenario layer (read_price_table): the base
    # forecast renews at 100 % of the current list price. Kept for the scenario functions.
    price_increase_by_period: dict = field(default_factory=dict)
    # Discounts above this are treated as unknown (1/(1−d) explodes for near-free licences).
    discount_cap: float = 0.9
    # The realization ratio (observed uplift / rule uplift, dollar-weighted, per uplift cell)
    # is CONTROL, not correction: it is measured and reported in uplift_contract_check
    # (where renewal offers make customers renew below list, the ratio shows it). Set to
    # True to apply it as a correction where the cell has ≥ uplift_floor known-discount renewers.
    contract_apply_realization_ratio: bool = False
    # Estimate the statistical uplift only with rows whose discount is unknown. The
    # hypothesis behind it: the missing discount is not random (channel, region), so mixing
    # both populations biases the average that the unknown rows will receive. It is
    # VERIFIED with data every run (`uplift_missing_check`: per cell, the uplift of the
    # unknown-discount rows against the one of the known-discount rows and of all rows,
    # with the bootstrap band); the console says in how many cells and how much money the
    # two populations differ beyond the band. True by default: the two paths coexist and
    # each is estimated on its own population.
    statistical_uplift_from_unknown_only: bool = True

    # ─── the price scenario layer (applied on the finished forecast) ──────────────
    # The SKU column of the raw (None = no scenarios by SKU). The price table comes from
    # `read_price_table()` (override it like read_raw): columns escenario, sku,
    # precio_lista (informative), fecha_efectiva, subida_pct, subida_importe, delta_tasa_pp.
    sku_column: Optional[str] = None
    # Both increase columns are applied (pct, then importe); an empty one counts as 0.   [por confirmar]
    scenario_use_pct_and_importe: bool = True
    # Rows with an UNKNOWN discount get the increase on their estimated price.         [por confirmar]
    scenario_increase_unknown_discount: bool = True
    # The rate delta applies only to NEUTRAL rows (no dormant / softcancel): elsewhere the
    # signal already carries the rejection.                                           [por confirmar]
    scenario_delta_neutral_only: bool = True

    # ─── phase 0.5 · the panorama of the portfolio (ANALYSIS) ──────────────────────
    # The year whose cell composition is the reference for the "rate at fixed
    # composition" line (None = the last complete closed year).
    panorama_reference_year: Optional[int] = None

    # ─── phase 1.5 · discount and churn (ANALYSIS) ─────────────────────────────────
    # The column with the discount bucket (None = the first extra de revalorización whose
    # name contains "disc") and the value that means "no discount" (None = the first
    # bucket in sorted order). The analysis compares every bucket to that reference.
    discount_column: Optional[str] = None
    no_discount_value: Optional[str] = None

    # ─── phase 2 · the seasonality benchmark (ANALYSIS) ────────────────────────────
    # Months of history the TECHNIQUES see (backtest, forecast): None = all. The ladder
    # and the pools always use the whole history (support is support); this only cuts
    # what the techniques learn from. Try 24 and compare the hold-out of the total.
    technique_history_months: Optional[int] = None
    # One decision for the whole portfolio, taken where the test has power: the biggest
    # NEUTRAL series (no flag), fully segmented, with support ≥ benchmark_min_support in
    # EVERY closed month (±5 pp floor or better), top N by money inside each group of
    # benchmark_group_dims (None = the first two mandatory dims). Seasonal if amplitude ≥
    # benchmark_amplitude_pp AND both extreme months keep their sign in ≥ benchmark_consistency
    # of the years AND the seasonal shape improves the recent level by ≥ benchmark_improvement_pct
    # at h=6 without worsening it at h=1. If the seasonal series carry less than
    # benchmark_material_share_pct of the sample's money, the rate has NO material
    # seasonality and the rate branch keeps level techniques only.
    benchmark_group_dims: list = field(default_factory=list)
    benchmark_top_n: int = 5
    benchmark_min_support: float = 271.0
    benchmark_min_months: int = 36
    benchmark_short_months: int = 24
    benchmark_min_years: int = 2
    benchmark_amplitude_pp: float = 2.0
    benchmark_consistency: float = 0.67
    benchmark_improvement_pct: float = 10.0
    benchmark_material_share_pct: float = 10.0
    # The extreme months must ALSO stand out of the noise: their mean standardized
    # residual |z| (rate − trend, in binomial errors) ≥ this. Without it, on 40 simulated
    # pure-noise series the criteria declared 4 seasonal (10 %); with it, 0, and the power
    # on an 8 pp planted season stayed at 34/40. Measured, not assumed (test_statistics S5).
    benchmark_min_extreme_z: float = 1.0

    # ─── phase 3 · backtest, technique, bands (ANALYSIS) ───────────────────────────
    # First month of the hold-out: the months ≥ this are NEVER used to choose techniques
    # or to measure bands; they are the exam. None = the first month with role 'test' in
    # the extract (the extract's own split), else the last `holdout_default_months`.
    # Training always uses the whole history before each origin.
    backtest_test_start: Optional[str] = None
    holdout_default_months: int = 6
    # Months of history before the first origin. 8 = enough for every non-seasonal
    # technique to be eligible (ma6, ses, drift); seasonal ones wait for 13 anyway.
    backtest_min_history_months: int = 8
    # The judge uses only the most recent target months: the business changes, so the
    # evidence must be as recent as possible. 6 = the last half year of verdicts.
    backtest_max_targets: int = 6
    # Horizons judged. Sparse on purpose (bands are monotone in h, so the nearest lower
    # judged horizon is a safe band for the ones in between); the forecast horizon H is
    # added automatically. [1,2,3,4] for the operational months, 6/9/12 for the year.
    # TWO test batteries, two views: the SHORT one (predict next month with data up to
    # the previous month: h=1) and the MEDIUM-LONG one (predict a month with data up to
    # six months before: h=6). Months further than six ahead use the six-month evidence
    # and are re-forecast every month.
    backtest_horizons: list = field(default_factory=lambda: [1, 6])
    # Two-stage judge: every eligible technique is screened at these horizons to choose
    # the champion; then only champion + challenger are judged at every horizon. {1,3,6}
    # covers the operational month, the quarter and the half-year with ~3× less cost.
    backtest_screen_horizons: list = field(default_factory=lambda: [1, 6])
    # Horizon bands: ONE champion per band, not one per pool. A 3-month average wins the
    # near months and knows nothing about January twelve months out; a seasonal or mixed
    # technique may lose at h=1 and win at h=12. Each band is judged with the screen
    # horizons that fall inside it (so every band needs at least one screen horizon).
    # h=1 is a band of its own: the current month is the forecast the business trusts
    # first, so its technique is chosen on its own evidence. Beyond `backtest_horizon_cap`
    # nothing is judged: a month 16 ahead is predicted as if 12 ahead (same technique,
    # same band). The error there is large and declared, not measured.
    backtest_horizon_bands: dict = field(default_factory=lambda: {"corto": [1, 1], "medio_largo": [2, 999]})
    backtest_horizon_cap: int = 6
    # What to persist of the long backtest table (`backtest_pred`): "chosen" = only the
    # rows of each band's champion and the challenger (~25 % of the rows: enough to audit
    # the decision and the hold-out); "all" = every technique (the full error-by-horizon
    # figure from SQL; in Kamelot 2M rows, ~3 minutes of writing); "none". The full table
    # stays in memory (results["backtest"]["backtest_long"]) during the session either way.
    backtest_persist: str = "chosen"
    # Parallel workers for the backtest (1 = sequential; identical result). Useful with
    # thousands of estimation ids on a multi-core machine; harmless otherwise.
    backtest_workers: int = 1
    # Minimum predictions a technique needs to dethrone the challenger: 6 = at least half
    # a year of verdicts at the screen horizons.
    backtest_min_predictions: int = 6
    # The challenger: the technique a champion must beat. The 3-month average ("what
    # happened last quarter"): in Kamelot it beat the whole-history mean in every horizon
    # band (2.1 vs 3.6 binomial units near, 4.5 vs 4.7 far) because the rate moves by
    # level changes, not by season. A champion has to beat THAT to be a champion.
    challenger_technique: str = "T3_ma3"
    # Margin, in units of the binomial error, by which a champion must beat the
    # challenger (and within which techniques tie → the richer family wins). 0.10 = a
    # tenth of a sampling error: enough to ignore luck, small enough to let real signal
    # through. The leaderboard (P3.1) shows how far apart techniques really are.
    challenger_margin_normalized: float = 0.10
    # Margin per horizon band, overriding the one above where set. Far from now the
    # challenger (a short window) carries no information about the shape of the future,
    # so a technique that merely TIES it there should be allowed to win when it uses more
    # history: the margin to dethrone the challenger shrinks with the horizon, and within
    # the margin the technique with the longest MEMORY wins (see techniques.MEMORY_MONTHS).
    # Never below zero: a technique still has to be at least as good as the challenger.
    challenger_margin_by_band: dict = field(default_factory=lambda: {"corto": 0.10, "medio_largo": 0.0})
    # Among techniques within the margin of the best, the technique with more memory (and
    # then the richer family: time series > smoothing > average) wins — but only when the
    # id has at least this many months of history: a month-effect technique chosen on 14
    # months is a story, not a model. 24 = two full cycles. Below it, the simplest wins.
    richer_family_min_history_months: int = 24
    # Predictions an (id, h) needs for its OWN error quantiles; below it the band comes
    # from the family (same technique, every id). 20 predictions make a p5/p95 that is
    # not just the extremes.
    band_min_predictions: int = 20
    # The band quantiles of the signed normalized error: p5 / p95 → a 90 % band, matching
    # z. Widen to .025/.975 for 95 %. The hold-out "% inside band" is the check.
    band_low_quantile: float = 0.05
    band_high_quantile: float = 0.95
    # Share of near-zero months (rate < 2 %) above which a series is "intermittent" and
    # the SBA technique competes. 30 %: below it, the zeros are just bad months.
    intermittent_zero_share: float = 0.30

    # ─── phase 4 · uplift (RUN) ─────────────────────────────────────────────────────
    # The mandatory dims that open an uplift cell. None = every mandatory dim (cell =
    # mandatory + extra_revalorizacion). With 10 mandatory dims that made 25,710 cells in
    # Kamelot, 70 % below the floor: a subset (e.g. regional_level_1, product_level_1,
    # purchase_type, term_level_2) keeps the revaluation drivers and gives cells with
    # enough renewers. Every dim listed must be a mandatory dim.
    uplift_mandatory_dims: Optional[list] = None
    # Renewers a cell needs to use its own ratio; below it the parent's (starting-point
    # extras kept) or the mandatory cell's. 30, like the rate floor: an uplift is a
    # ratio of the money of ~30 renewers before it stops jumping.
    uplift_floor: float = 30.0
    # Ratios above this are clipped and flagged `recortado`. 3.0: a renewer paying three
    # times the pipeline AUV is a data problem (a bundle, a currency), not a revaluation.
    uplift_cap: float = 3.0
    # extra_revalorizacion columns KEPT in the parent cell: the "starting point" (e.g.
    # newcust) that a small cell must not lose when it borrows. Empty = the parent is
    # the mandatory cell.
    uplift_parent_keep_columns: list = field(default_factory=list)
    # Months of history the uplift is estimated on: None = all. The uplift tracks the
    # discount mix, recovery campaigns (lower it) and price rises (raise it): a price rise
    # is a step, and the whole-history ratio averages before and after. 12 keeps the
    # current price regime. The parent/cell fallbacks use the same window.
    uplift_window_months: Optional[int] = None
    # Bootstrap resamples for the uplift band. 200 gives a stable p5/p95 in milliseconds
    # (numpy resampling); 1000 changes the third decimal.
    uplift_bootstrap_samples: int = 200

    # ─── phase 5 · assembly and extended horizon (RUN) ─────────────────────────────
    # None = derived: months from the last month with truth to the end of the forecast
    # (known projection or extended horizon). Set it only to cut the forecast short.
    max_forecast_horizon: Optional[int] = None
    # Simulate the pipeline beyond the known projection up to this month ("2027-12").
    # None = no extension. Everything built on simulated rows is flagged (simulada = 1)
    # and reported (horizon_report_total.pct_simulado).
    extended_horizon_end: Optional[str] = None
    # A renewed contract re-enters the pipeline after its term. `renewal_term_months` is
    # the default (12 = yearly). A mixed-term portfolio declares the column that carries
    # the term and the months of each value: e.g. term_column = "term_level_2",
    # term_months_by_value = {"1 year": 12, "2 year": 24, "3 year": 36}; values not in the
    # map fall back to the default. Used by the extended horizon and the acquisition factor.
    renewal_term_months: int = 12
    term_column: Optional[str] = None
    term_months_by_value: dict = field(default_factory=dict)
    # pipeline(m) = renewed(m − term) × factor. None = estimated from history per series
    # (median of pipeline(t) / renewed(t − term) = 1 + acquisitions / renewals; global
    # fallback). Set a number to impose a business assumption on acquisition.
    acquisition_factor: Optional[float] = None
    # Only rows matching this filter re-enter the simulated pipeline: column → allowed
    # values, e.g. {"term_level_2": ["1 year"]}. Multi-year contracts renewed now fall due
    # beyond the horizon and their known expirations are already in the pipeline; only
    # the 12-month ones (renewals AND acquisitions) shape next year. {} = every row.
    extension_row_filter: dict = field(default_factory=dict)
    # The simulated pipeline is valued at the RENEWED price: a contract renewed in 2026 at
    # pipeline AUV × uplift is worth that when it falls due in 2027 (observed renewed AUV
    # where there is truth, pipeline AUV × the cell's uplift where there is not). The 2027
    # forecast then applies rate × uplift again on that revalued pipeline.
    # Pairs (t, t − term) a series needs for its own acquisition factor; below it the
    # global one. 3 = a median that is not a single point.
    acquisition_min_pairs: int = 3
    # The total's relative band may narrow from one month to the next when the mix leans
    # toward well-supported series; a narrowing beyond this many percentage points of the
    # total is flagged in horizon_report_total.banda_monotona. Per id the band never
    # narrows (by construction); this is a mix signal, not a calibration one.
    band_narrowing_tolerance_pct: float = 1.0

    # ─── phase 5.9 · maturation of the signals (RUN) ────────────────────────────────
    # The final composition of a cell (neutral / softcancel / dormant…) is measured over
    # the last N closed months; the pending maturation of a future month is that final
    # share minus today's share. 12 = a full year of expiries.
    signal_final_window_months: int = 12

    # ─── sheets drawn at the end of every analysis ─────────────────────────────────
    # After the forecast, the sheet (six-panel figure + story) of the N series with the
    # most projected money is drawn into <outdir>/diagnostics/. 5 by default; 0 disables.
    sheets_top_series: int = 5

    # ─── console ───────────────────────────────────────────────────────────────────
    # Rows printed per listing (cells, champions, candidates...). The tables hold
    # everything; the console shows the top by support or money. 15 fits a screen.
    console_top_rows: int = 15
    # Print, after every block of numbers, two or three lines that say how to read them
    # (what a binomial unit is, what a band promises, why the mean is not the forecast).
    # Off for the delegated monthly run once the reader knows the framework.
    console_explanations: bool = True

    # ─── baseline (ANALYSIS) ────────────────────────────────────────────────────────
    # Grains of the spreadsheet baseline: "global" (one dollar rate for the portfolio),
    # "mandatory" (every mandatory dim), or a '+'-joined list of columns (a coarse cut,
    # e.g. "regional_level_1+product_level_1+purchase_type"). Each grain × window (1, 3,
    # 12 months) gives a forecast and its own walk-forward error, next to the framework's.
    baseline_grains: list = field(default_factory=lambda: ["global", "mandatory"])

    # ─── governance ────────────────────────────────────────────────────────────────
    # Version (as-of) of the model that produces each timevarying flag, stamped on the
    # calibration table so a flag's realized rate can be compared across versions.
    timevarying_model_version: dict = field(default_factory=dict)
    # The monthly run warns when the decision tables are older than this. 6 months: two
    # seasons; the ladder and the champions should be re-judged at least twice a year.
    decision_max_age_months: int = 6
    # Seed of every random draw (uplift bootstrap): the same run gives the same band.
    random_seed: int = 7

    # ─── output and persistence ────────────────────────────────────────────────────
    outdir: str = "./salida"

    # ─── SQL persistence (SQLAlchemy) ───────────────────────────────────────────────
    sql_server: Optional[str] = None
    sql_database: Optional[str] = None
    sql_driver: str = "ODBC Driver 17 for SQL Server"
    sql_username: Optional[str] = None
    sql_password: Optional[str] = None
    sql_trusted: bool = True
    sql_schema: Optional[str] = "dbo"        # the schema is a config datum
    sql_write_mode: str = "replace"          # replace (DROP+CREATE) | truncate (keeps DDL)
    sql_chunksize: int = 50_000
    sql_engine: object = None                # or inject an already-built engine
    sql_table_prefix: str = "sff_"
    sql_table_names: dict = field(default_factory=dict)   # logical name → physical override

    # ═══════════════════════════════════════════════════════════════════════════════
    # RAW SOURCE (override in your own main; run/ never reads a file or a query)
    # ═══════════════════════════════════════════════════════════════════════════════

    def read_price_table(self) -> pd.DataFrame:
        """The price scenarios (escenario, sku, precio_lista, fecha_efectiva, subida_pct,
        subida_importe, delta_tasa_pp). Override it like read_raw. Default: no scenarios."""
        return pd.DataFrame(columns=["escenario", "sku", "precio_lista", "fecha_efectiva", "subida_pct", "subida_importe", "delta_tasa_pp"])

    def read_raw(self) -> pd.DataFrame:
        """Return the raw extract as a DataFrame. MUST be overridden by the caller.

        INPUT:   none (the subclass decides: a SQL query, a file, a generator).
        OUTPUT:  a DataFrame with every column of the extract; the pipeline validates
                 it against the column contract before computing anything.
        RULES:   the framework assumes nothing about where the raw comes from. The
                 production main subclasses Config and reads its query; the test main
                 subclasses Config and builds the synthetic dataset. `run/` contains no
                 loading code of any kind.
        EDGE CASES: the base implementation raises NotImplementedError with the
                 instruction, so a Config used without an override fails at the first
                 step with a sentence, not a downstream AttributeError.
        CONSOLE: nothing.
        STEPS:
          [1] Refuse: this method only exists to be overridden.
        """
        # [1] there is no default source, on purpose
        raise NotImplementedError(
            "Config.read_raw() must be overridden: subclass Config in your main and "
            "return the raw DataFrame from your own query or dataset")

    # ═══════════════════════════════════════════════════════════════════════════════
    # BLOCK 1 · TAXONOMY AND VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        """Validate the taxonomy and prepare the execution context.

        INPUT:   the dataclass fields as given by the caller.
        OUTPUT:  none; sets `execution_id` and creates `outdir`.
        RULES:   the exclusion rules of DISENO_V2 §3 are checked here, once, at
                 construction: a misdeclared taxonomy never reaches a phase.
        EDGE CASES: an invalid sign or an overlap raises ValueError naming the culprit.
        CONSOLE: nothing.
        STEPS:
          [1] Check the taxonomy (signs and exclusions).
          [2] Mint the execution identifier stamped on every persisted row.
          [3] Make sure the output folder exists.
        """
        # [1] taxonomy first: everything downstream assumes it is consistent
        self._validate_taxonomy()

        # [2] one short random id per execution, shared by all the tables it writes
        self.execution_id = uuid.uuid4().hex[:EXECUTION_ID_LENGTH]

        # [3] the CSV fallback of `write` needs the folder
        os.makedirs(self.outdir, exist_ok=True)

    def _validate_taxonomy(self) -> None:
        """Enforce the exclusion rules of the 4-group taxonomy (DISENO_V2 §3).

        INPUT:   the taxonomy fields of this Config.
        OUTPUT:  none.
        RULES:   every timevarying sign is "negative" or "positive" · mandatory is
                 exclusive with timevarying and with both extra groups · timevarying is
                 exclusive with both extra groups · the two extra groups MAY overlap.
        EDGE CASES: raises ValueError with the offending column names.
        CONSOLE: nothing.
        STEPS:
          [1] Every timevarying dimension carries a valid sign.
          [2] Mandatory does not overlap timevarying, extra_renovacion or
              extra_revalorizacion.
          [3] Timevarying does not overlap the extra groups.
        """
        # [1] a timevarying without a valid sign cannot be grouped in L1
        for dimension_name, sign in self.structural_timevarying_dims.items():
            if sign not in VALID_TIMEVARYING_SIGNS:
                raise ValueError(
                    f"timevarying['{dimension_name}']='{sign}': "
                    f"the sign must be one of {VALID_TIMEVARYING_SIGNS}")

        # [2] mandatory is exclusive with every other dimension group
        mandatory_set = set(self.business_mandatory_dims)
        timevarying_set = set(self.structural_timevarying_dims)
        extra_renovacion_set = set(self.extra_renovacion)
        extra_revalorizacion_set = set(self.extra_revalorizacion)
        mandatory_overlap = (mandatory_set & timevarying_set
                             | mandatory_set & extra_renovacion_set
                             | mandatory_set & extra_revalorizacion_set)
        if mandatory_overlap:
            raise ValueError(
                f"mandatory is exclusive with timevarying and with the extra groups; "
                f"overlapping columns: {sorted(mandatory_overlap)}")

        # [3] timevarying is exclusive with the extra groups (the extras may overlap
        #     each other: a column can serve both the rate and the uplift)
        timevarying_overlap = timevarying_set & (extra_renovacion_set | extra_revalorizacion_set)
        if timevarying_overlap:
            raise ValueError(
                f"timevarying is exclusive with the extra groups; "
                f"overlapping columns: {sorted(timevarying_overlap)}")

    def validate_column_contract(self, frame: pd.DataFrame) -> None:
        """The exhaustive contract (P3): every raw column has exactly one role.

        INPUT:   frame — the raw extract as loaded.
        OUTPUT:  none.
        RULES:   the set of declared columns is: period, role, current-month flag,
                 time-series flag, the four core measures, the declared measures, the
                 rate series columns, extra_revalorizacion and ignore_cols. Any raw column outside
                 that set is an orphan and stops the run. Declared columns MISSING from
                 the raw are not checked here: the phases fail on first access.
        EDGE CASES: raises ValueError listing every orphan column at once.
        CONSOLE: nothing.
        STEPS:
          [1] Assemble the set of columns that have a role.
          [2] List the raw columns outside that set.
          [3] Stop if there is any orphan.
        """
        # [1] every column the taxonomy knows about
        context_columns = [self.period_col, self.dataset_role_col,
                           self.current_month_col, self.flag_time_series_col]
        declared_columns = set(context_columns
                               + self.core_measures
                               + self.declared_measures
                               + self.rate_series_columns
                               + self.extra_revalorizacion
                               + self.ignore_cols
                               + ([self.discount_value_column] if self.discount_value_column else [])    # a formula input, not a dim
                               + ([self.sku_column] if self.sku_column else []))                          # carried for the scenarios

        # [2] the orphans: present in the raw, declared nowhere
        orphan_columns = [column_name for column_name in frame.columns
                          if column_name not in declared_columns]

        # [3] an orphan is a contract violation, not a warning
        if orphan_columns:
            raise ValueError(
                f"columns with no role in config: {orphan_columns} — "
                f"declare their group or add them to ignore_cols")

    # ═══════════════════════════════════════════════════════════════════════════════
    # BLOCK 2 · SERIES AND CELL COLUMNS (derived from the taxonomy, never typed)
    # ═══════════════════════════════════════════════════════════════════════════════

    @property
    def rate_series_columns(self) -> list:
        """The columns that define ONE renewal-rate series (a forecast series, `fs_id`).

        = mandatory + timevarying + extra_renovacion. Every distinct combination of these
        columns is one series; with the month appended it is one forecast unit (`fu_id`).
        The order is contractual: it is the field order of `fu_id` and `fs_id`.
        """
        return (self.business_mandatory_dims
                + list(self.structural_timevarying_dims)
                + self.extra_renovacion)

    @property
    def uplift_cell_columns(self) -> list:
        """The columns that define ONE uplift cell (`gu` / `uplift_cell_key`).

        = mandatory + extra_revalorizacion. The uplift has no time axis: it is estimated
        over every month at once, so its unit is a static cell, not a series.
        """
        mandatory = self.uplift_mandatory_dims if self.uplift_mandatory_dims else self.business_mandatory_dims
        unknown = [d for d in mandatory if d not in self.business_mandatory_dims]
        if unknown:
            raise ValueError(f"uplift_mandatory_dims must be mandatory dims; not mandatory: {unknown}")
        return list(mandatory) + [c for c in self.extra_revalorizacion if c != self.discount_value_column]

    @property
    def core_measures(self) -> list:
        """The four measures the forecast computes with: pipeline and renewed, units and USD."""
        return [self.pipeline_units_col, self.pipeline_usd_col,
                self.renewed_units_col, self.renewed_usd_col]

    @property
    def declared_measures(self) -> list:
        """Measures of the raw outside the computation (reacquisitions, AUVs).

        They are declared so the contract accepts them; nobody computes from them.
        A column configured as None or "" is simply not expected in the raw.
        """
        candidate_columns = (self.reacq_units_col, self.reacq_usd_col,
                             self.auv_pipeline_col, self.auv_renewed_col, self.auv_reacq_col)
        present_columns = [column_name for column_name in candidate_columns if column_name]
        return present_columns

    # ═══════════════════════════════════════════════════════════════════════════════
    # BLOCK 4 · SQL PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════════

    @property
    def engine(self):
        """The SQLAlchemy engine, built lazily from the SQL fields or injected.

        INPUT:   sql_engine (injected) or sql_server + sql_database + auth fields.
        OUTPUT:  an Engine with fast_executemany guaranteed on mssql, or None when no
                 engine is injected and no server is configured (CSV fallback).
        RULES:   an injected engine wins; a built engine is cached in `sql_engine`.
        EDGE CASES: trusted connection when `sql_trusted`, else UID/PWD.
        CONSOLE: nothing.
        STEPS:
          [1] Injected engine: reuse it, only making sure fast_executemany is on.
          [2] No engine but a server: build the mssql+pyodbc engine once and cache it.
          [3] Neither: None (the writer falls back to CSV).
        """
        # [1] an injected engine (tests use sqlite, production may inject mssql)
        if self.sql_engine is not None:
            return self._ensure_fast_executemany(self.sql_engine)

        # [2] build from the connection fields, once
        if self.sql_server:
            if self.sql_trusted:
                auth_fragment = "Trusted_Connection=yes;"
            else:
                auth_fragment = f"UID={self.sql_username};PWD={self.sql_password};"
            odbc_connection_string = (f"DRIVER={{{self.sql_driver}}};SERVER={self.sql_server};"
                                      f"DATABASE={self.sql_database};{auth_fragment}")
            sqlalchemy_url = ("mssql+pyodbc:///?odbc_connect="
                              + urllib.parse.quote_plus(odbc_connection_string))
            self.sql_engine = create_engine(sqlalchemy_url, fast_executemany=True)

        # [3] None when nothing is configured
        return self.sql_engine

    @staticmethod
    def _ensure_fast_executemany(sqlalchemy_engine):
        """Force `fast_executemany` on every executemany of an mssql engine.

        INPUT:   sqlalchemy_engine — built here or injected by the caller.
        OUTPUT:  the same engine, with the cursor event registered at most once.
        RULES:   fast_executemany is an attribute of the pyodbc CURSOR, not of the
                 engine; the only way to guarantee it for an injected engine is a
                 `before_cursor_execute` listener. Non-mssql engines are returned as is.
        EDGE CASES: idempotent: a private flag on the engine prevents double registration.
        CONSOLE: nothing.
        STEPS:
          [1] Skip if already done or if the dialect is not mssql.
          [2] Register the listener that flips the cursor flag on executemany.
          [3] Mark the engine as done.
        """
        # [1] nothing to do for sqlite or for an engine already patched
        already_patched = getattr(sqlalchemy_engine, "_sff_fast", False)
        is_mssql = str(sqlalchemy_engine.url).startswith("mssql")
        if already_patched or not is_mssql:
            return sqlalchemy_engine

        # [2] the listener runs before every statement; only executemany is affected
        @event.listens_for(sqlalchemy_engine, "before_cursor_execute")
        def enable_fast_executemany(connection, cursor, statement, parameters,
                                    context, executemany):
            if executemany:
                cursor.fast_executemany = True

        # [3] remember it, so a second call is a no-op
        sqlalchemy_engine._sff_fast = True
        return sqlalchemy_engine

    @staticmethod
    def _sanitize_for_persistence(frame: pd.DataFrame) -> pd.DataFrame:
        """Make a frame writable by pandas `to_sql` / `to_csv` without losing information.

        INPUT:   frame — any table produced by a phase.
        OUTPUT:  a copy where Period columns become str (plus a `<col>_date` timestamp
                 column, if not already present) and Interval columns become str.
        RULES:   the original frame is never modified. A `<col>_date` companion is
                 added ONLY for columns with a PeriodDtype; a column of Period objects
                 in an object column is cast to str without a companion (legacy
                 behaviour, kept for equivalence).
        EDGE CASES: a column holding nested objects (DataFrame, Series, list, dict, set)
                 raises a named error instead of a cryptic driver failure.
        CONSOLE: nothing.
        STEPS:
          [1] Copy the frame.
          [2] Period dtype → str + `<col>_date`; Interval dtype → str.
          [3] Object columns: Period objects → str; nested objects → error.
        """
        # [1] never touch the phase's frame
        sanitized_frame = frame.copy()

        for column_name in list(sanitized_frame.columns):
            column_dtype = sanitized_frame[column_name].dtype

            # [2] typed Period / Interval columns
            if isinstance(column_dtype, pd.PeriodDtype):
                companion_column = f"{column_name}_date"
                if companion_column not in sanitized_frame.columns:
                    sanitized_frame[companion_column] = sanitized_frame[column_name].dt.to_timestamp()
                sanitized_frame[column_name] = sanitized_frame[column_name].astype(str)
                continue
            if isinstance(column_dtype, pd.IntervalDtype):
                sanitized_frame[column_name] = sanitized_frame[column_name].astype(str)
                continue

            # [3] object columns: inspect the first non-null value
            first_non_null = sanitized_frame[column_name].dropna().head(1)
            if len(first_non_null) == 0:
                continue
            sample_value = first_non_null.iloc[0]
            if isinstance(sample_value, pd.Period):
                sanitized_frame[column_name] = sanitized_frame[column_name].astype(str)
            elif isinstance(sample_value, (pd.DataFrame, pd.Series, list, dict, set)):
                raise ValueError(
                    f"write: column '{column_name}' holds nested objects — not writable")

        return sanitized_frame

    @staticmethod
    def _nvarchar_dtypes(frame: pd.DataFrame) -> dict:
        """Size every text column as NVARCHAR(n) for SQL Server.

        INPUT:   frame — already sanitized.
        OUTPUT:  dict column → NVARCHAR(length) for object/str/string columns only.
        RULES:   length = max observed × NVARCHAR_GROWTH_FACTOR + NVARCHAR_GROWTH_PADDING,
                 clamped to [NVARCHAR_MIN_LENGTH, NVARCHAR_MAX_LENGTH]. An all-null
                 text column gets the minimum.
        EDGE CASES: numeric and datetime columns are left to pandas' default mapping.
        CONSOLE: nothing.
        STEPS:
          [1] For every text column, measure the longest value.
          [2] Apply the growth rule and the clamp.
        """
        dtype_by_column = {}
        for column_name in frame.columns:
            is_text_column = str(frame[column_name].dtype) in ("object", "str", "string")
            if not is_text_column:
                continue

            # [1] longest observed text (1 when the column is entirely null)
            non_null_text = frame[column_name].dropna().astype(str)
            if len(non_null_text) > 0:
                longest_observed = int(non_null_text.str.len().max())
            else:
                longest_observed = 1

            # [2] leave headroom for future values, within SQL Server's limits
            grown_length = int(longest_observed * NVARCHAR_GROWTH_FACTOR) + NVARCHAR_GROWTH_PADDING
            clamped_length = min(max(grown_length, NVARCHAR_MIN_LENGTH), NVARCHAR_MAX_LENGTH)
            dtype_by_column[column_name] = NVARCHAR(clamped_length)
        return dtype_by_column

    def _resolve_physical_table_name(self, logical_table_name: str) -> str:
        """Map a logical table name to its physical name.

        INPUT:   logical_table_name — as used by the phases ("fact_fu", ...).
        OUTPUT:  the physical table name.
        RULES:   an explicit override in `sql_table_names` wins; otherwise prefix +
                 registry suffix; a logical name absent from the registry keeps its
                 own name (prefixed).
        EDGE CASES: none.
        CONSOLE: nothing.
        STEPS:
          [1] Override, if configured.
          [2] Prefix + registry suffix (or the logical name itself).
        """
        # [1] the caller may pin any physical name
        if logical_table_name in self.sql_table_names:
            return self.sql_table_names[logical_table_name]

        # [2] the registry, with the logical name as fallback suffix
        registry_suffix = PHYSICAL_TABLE_NAMES.get(logical_table_name, logical_table_name)
        return self.sql_table_prefix + registry_suffix

    def _qualified_table_name(self, physical_table_name: str) -> str:
        """`[schema].[table]` when a schema is configured, else `[table]` (sqlite)."""
        if self.sql_schema:
            return f"[{self.sql_schema}].[{physical_table_name}]"
        return f"[{physical_table_name}]"

    def _empty_table_keeping_definition(self, qualified_table_name: str) -> None:
        """Empty an existing table with TRUNCATE, falling back to DELETE.

        INPUT:   qualified_table_name — `[schema].[table]`.
        OUTPUT:  none; the table is empty and its DDL untouched.
        RULES:   TRUNCATE is minimally logged and instantaneous; 600k rows of DELETE
                 mean lock escalation and hundreds of MB of log. DELETE is the LAST
                 resort, used only when TRUNCATE fails (no permission, referencing FKs).
        EDGE CASES: any exception from TRUNCATE triggers the DELETE path.
        CONSOLE: nothing.
        STEPS:
          [1] Try TRUNCATE inside a transaction.
          [2] On failure, DELETE inside the same transaction.
        """
        with self.engine.begin() as connection:
            try:
                # [1] the cheap path
                connection.execute(text(f"TRUNCATE TABLE {qualified_table_name}"))
            except Exception:
                # [2] the expensive but always-allowed path
                connection.execute(text(f"DELETE FROM {qualified_table_name}"))

    def _write_to_sql(self, sanitized_frame: pd.DataFrame, physical_table_name: str) -> str:
        """Write one table to the engine honouring `sql_write_mode`.

        INPUT:   sanitized_frame (traceability columns already stamped) ·
                 physical_table_name.
        OUTPUT:  the effective write mode used ("truncate" or "replace").
        RULES:   "truncate" keeps the table definition when the table exists (empty +
                 append); when it does not exist yet, the first write is a "replace".
                 "replace" drops and recreates. Text columns are sized with
                 `_nvarchar_dtypes`; rows go in chunks of `sql_chunksize`.
        EDGE CASES: a "truncate" on a missing table silently becomes "replace".
        CONSOLE: nothing (the caller prints the stopwatch line).
        STEPS:
          [1] Truncate mode on an existing table: empty it, then append.
          [2] Otherwise: replace.
        """
        effective_write_mode = self.sql_write_mode
        qualified_table_name = self._qualified_table_name(physical_table_name)
        text_column_types = self._nvarchar_dtypes(sanitized_frame)

        # [1] keep the DDL (indexes, permissions, BI bindings): empty and append
        if effective_write_mode == "truncate":
            table_exists = inspect(self.engine).has_table(physical_table_name, schema=self.sql_schema)
            if table_exists:
                self._empty_table_keeping_definition(qualified_table_name)
                sanitized_frame.to_sql(physical_table_name, self.engine, schema=self.sql_schema,
                                       if_exists="append", index=False,
                                       chunksize=self.sql_chunksize, dtype=text_column_types)
                return "truncate"
            effective_write_mode = "replace"

        # [2] drop and recreate
        sanitized_frame.to_sql(physical_table_name, self.engine, schema=self.sql_schema,
                               if_exists="replace", index=False,
                               chunksize=self.sql_chunksize, dtype=text_column_types)
        return effective_write_mode

    # Id column → key column stamped on every written table that carries the id, so every
    # table joins to the audit dimensions (series, estimation id, uplift cell, mandatory
    # cell) on the same short keys in the BI. The ids stay: they are human-readable.
    DERIVED_KEY_COLUMNS = {"fs_id": "fs_key", "id_estimacion": "estimacion_key",
                           "uplift_cell_id": "uplift_cell_key", "celda_id": "celda_key"}

    def stamp_derived_keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the key column of every id column present (hash_key), if not already there."""
        stamped = frame.copy()
        for id_column, key_column in self.DERIVED_KEY_COLUMNS.items():
            if id_column in stamped.columns and key_column not in stamped.columns:
                stamped[key_column] = stamped[id_column].astype(str).map(hash_key)
        return stamped

    def write(self, frame: pd.DataFrame, logical_table_name: str, append: bool = False) -> pd.DataFrame:
        """Persist one star-schema table with automatic traceability.

        `append=True` ADDS the rows to the existing table instead of replacing it: the
        monthly photos (`signal_snapshot`) accumulate across runs; every row carries
        its process_date and execution_id, so a photo can be told from the next.

        INPUT:   frame — the table as produced by a phase · logical_table_name — the
                 LOGICAL name (PHYSICAL_TABLE_NAMES maps it to the physical one).
        OUTPUT:  the sanitized frame that was written (with process_date and
                 execution_id), so callers can keep using it.
        RULES:   every row is stamped with `process_date` (ISO, seconds) and
                 `execution_id`; with an engine the table goes to SQL, without one it
                 goes to `<outdir>/<physical>.csv`. Period columns become str with a
                 `<col>_date` companion (see `_sanitize_for_persistence`).
        EDGE CASES: nested objects in a column stop the write with a named error.
        CONSOLE: one line per table: rows, destination, mode, seconds and rows/second.
        STEPS:
          [1] Sanitize the frame for the driver.
          [2] Stamp traceability.
          [3] Resolve the physical name.
          [4] Write to SQL (truncate/replace) or to CSV, timing it.
          [5] Report rows, destination and throughput.
        """
        # [1] Period/Interval → str, nested objects → error
        sanitized_frame = self.stamp_derived_keys(self._sanitize_for_persistence(frame))

        # [2] every persisted row knows when and by which execution it was written
        sanitized_frame["process_date"] = datetime.datetime.now().isoformat(timespec="seconds")
        sanitized_frame["execution_id"] = self.execution_id

        # [3] logical → physical
        physical_table_name = self._resolve_physical_table_name(logical_table_name)
        row_count = len(sanitized_frame)

        # [4] SQL when there is an engine, CSV otherwise
        write_start_time = time.time()
        if self.engine is not None and append:
            sanitized_frame.to_sql(physical_table_name, self.engine, schema=self.sql_schema, if_exists="append", index=False,
                                   chunksize=self.sql_chunksize, dtype=self._nvarchar_dtypes(sanitized_frame))
            destination = self._qualified_table_name(physical_table_name)
            mode_label = "SQL append"
        elif self.engine is not None:
            effective_write_mode = self._write_to_sql(sanitized_frame, physical_table_name)
            destination = self._qualified_table_name(physical_table_name)
            mode_label = "truncate" if effective_write_mode == "truncate" else f"SQL {effective_write_mode}"
        elif append:
            csv_path = os.path.join(self.outdir, f"{physical_table_name}.csv")
            os.makedirs(self.outdir, exist_ok=True)
            sanitized_frame.to_csv(csv_path, index=False, mode="a", header=not os.path.exists(csv_path))
            destination = csv_path
            mode_label = "CSV append"
        else:
            csv_path = os.path.join(self.outdir, f"{physical_table_name}.csv")
            sanitized_frame.to_csv(csv_path, index=False)
            destination = f"{physical_table_name}.csv"
            mode_label = "no engine: CSV"

        # [5] the stopwatch line: what was written, where, and how fast
        elapsed_seconds = time.time() - write_start_time
        rows_per_second = row_count / max(elapsed_seconds, MIN_ELAPSED_SECONDS)
        print(f"[write] {row_count:,} rows → {destination} "
              f"({mode_label}, {elapsed_seconds:.1f}s, {rows_per_second:,.0f} rows/s)")
        return sanitized_frame
