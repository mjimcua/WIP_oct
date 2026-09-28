"""support_reference.py — SFF v2 · ANALYSIS · phase 0.4: the immutable reference.

Before any rate is estimated, before any pool is formed, this table records for every
forecast unit how much support exists and what binomial error to expect in the worst
case. It is the yardstick everything later is compared against: a series whose band
after repair is not narrower than its own worst-case bound has gained nothing.

It is ANALYSIS, not RUN: no phase consumes it, no forecast number depends on it. It is
the artifact the analyst keeps to audit the monthly run without re-running anything.

Persisted as `forecast_units_raw_summary` (physical name `sff_fu_summary`).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)


# ─── named constants ─────────────────────────────────────────────────────────────
# The worst-case variance of a proportion: p(1-p) is maximal at p = 0.5, so the bound
# computed here is honest for ANY rate, including the ones not yet estimated.
WORST_CASE_PROPORTION_VARIANCE = 0.25

# A unit with no pipeline still gets a bound: its support is clipped to 1 so the
# formula does not divide by zero (the bound is then ±82 pp at 90%, i.e. "no evidence").
MIN_SUPPORT_UNITS = 1

# Percentage points per unit of proportion.
PERCENTAGE_POINTS = 100

# The persisted table (logical name; the physical one is `sff_fu_summary`).
REFERENCE_TABLE_NAME = "forecast_units_raw_summary"


def add_support_reference(labeled_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The three bound columns (se_pp_max, moe_pp_max, moe_usd_max) added to the units
    frame itself, so `fact_fu` carries them and no separate table is needed."""
    reference = compute_support_reference(labeled_units, configuration)
    return labeled_units.merge(reference[["fu_id", "se_pp_max", "moe_pp_max", "moe_usd_max"]], on="fu_id", how="left")


def build_support_reference(labeled_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Kept for the tests and notebooks that call it: computes and PERSISTS the reference table."""
    reference = compute_support_reference(labeled_units, configuration)
    configuration.write(reference, "forecast_units_raw_summary")
    return reference


def compute_support_reference(labeled_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Persist the immutable reference: support and worst-case binomial error per unit.

    INPUT:   labeled_units — from `label_universe_and_routes` · configuration — uses
             period_col, dataset_role_col, current_month_col, pipeline_*_col, z, write.
    OUTPUT:  the persisted frame: fu_key, fu_id, period (as text), role, current-month
             flag, universo, ruta, pipeline units and USD, se_pp_max, moe_pp_max,
             moe_usd_max (+ process_date, execution_id from write).
    RULES:   se_pp_max = 100 · √(0.25 / n) with n = pipeline units clipped to ≥ 1 ·
             moe_pp_max = z · se_pp_max · moe_usd_max = moe_pp_max / 100 · pipeline USD.
             It is a BOUND at p = 0.5, not an estimate: no rate exists yet.
    EDGE CASES: n = 0 → n = 1 (bound of "no evidence"); NaN pipeline USD → NaN bound in $.
    CONSOLE: the `[write]` line of the persisted table.
    STEPS:
      [1] Identity, context and support of every unit.
      [2] The binomial bound in percentage points.
      [3] The same bound in dollars over the unit's own pipeline.
      [4] Persist.
    """
    # [1] what identifies the unit, what context it sits in, how much support it has
    reference = labeled_units[["fu_key", "fu_id", configuration.period_col,
                               configuration.dataset_role_col, configuration.current_month_col,
                               "universo", "ruta",
                               configuration.pipeline_units_col, configuration.pipeline_usd_col]].copy()

    # [2] the honest worst case: p = 0.5, n = the unit's own pipeline
    support_units = reference[configuration.pipeline_units_col].clip(lower=MIN_SUPPORT_UNITS)
    reference["se_pp_max"] = PERCENTAGE_POINTS * np.sqrt(WORST_CASE_PROPORTION_VARIANCE / support_units)
    reference["moe_pp_max"] = configuration.z * reference["se_pp_max"]

    # [3] percentage points mean nothing to a budget: the bound over the unit's money
    reference["moe_usd_max"] = (reference["moe_pp_max"] / PERCENTAGE_POINTS
                                * reference[configuration.pipeline_usd_col])

    # [4] the period as text (Period is not a SQL type); write stamps the traceability
    reference[configuration.period_col] = reference[configuration.period_col].astype(str)
    return configuration.write(reference, REFERENCE_TABLE_NAME)
