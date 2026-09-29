"""
step_07_support_bound.py — How much can chance alone move the rate of every forecast unit?

Before any rate is estimated, the support of a unit (its contracts falling due, n)
already bounds its error: the rate of n contracts moves, by chance alone, up to

    se_pp_max  = 100 · √(0.25 / n)            (the binomial error at p = 0.5, the worst case)
    moe_pp_max = z · se_pp_max                (at the confidence of the Config, 90 %)
    moe_usd_max = moe_pp_max / 100 · USD due  (the same bound in money)

It is a BOUND, not an estimate: no rate exists yet. It is the yardstick the later steps
are compared against: a series whose band after borrowing support is not narrower than
its own bound has gained nothing. Nothing later computes with it; it is kept to audit.

Actions (logged as they are done):
  1. take the support (units due) of every forecast unit; a unit with none counts as 1
  2. the worst-case binomial error, in percentage points and at z
  3. the same bound in dollars over the unit's own pipeline
  4. check the bounds                                               check 1
  5. write the bounds table                                         check 2
  6. count the checks; stop if any failed
  7. show the bound per role, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. every forecast unit has its bound
   2. table sff_fu_soporte written and read back

Output: one row per forecast unit with its three bounds · table sff_fu_soporte.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, ROLES_IN_ORDER, SERIES_KEY_COLUMN, TABLE_UNIT_SUPPORT, UNIT_ID_COLUMN,
                         UNIT_KEY_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "07"
STEP_NAME = "SUPPORT BOUND"
STEP_PURPOSE = ("bound, before estimating anything, how much chance alone can move the rate of every forecast unit "
                "given its support (worst case p = 0.5): the yardstick every later band is compared against")
STEP_ACTIONS = ["take the support (units due) of every forecast unit; a unit with none counts as 1",
                "the worst-case binomial error, in percentage points and at z",
                "the same bound in dollars over the unit's own pipeline",
                "check the bounds (check 1)",
                "write the bounds table (check 2)",
                "count the checks; stop if any failed",
                "show the bound per role, as a table"]
STEP_OUTPUT = "one row per forecast unit with se_pp_max, moe_pp_max, moe_usd_max · table sff_fu_soporte"

# ─── named constants ─────────────────────────────────────────────────────────────
WORST_CASE_PROPORTION_VARIANCE = 0.25     # p (1 − p) at p = 0.5: the largest a binomial variance can be
MIN_SUPPORT_UNITS = 1                     # a unit with nothing due is bounded as if it had one contract
PERCENTAGE_POINTS = 100


def build_support_bound(forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The worst-case binomial bound of every forecast unit, checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the support of every unit
    support_bound = forecast_units[[UNIT_ID_COLUMN, UNIT_KEY_COLUMN, SERIES_KEY_COLUMN, configuration.period_col,
                                    CALENDAR_ROLE_COLUMN, configuration.pipeline_units_col,
                                    configuration.pipeline_usd_col]].copy()
    support_units = support_bound[configuration.pipeline_units_col].clip(lower=MIN_SUPPORT_UNITS)
    units_counted_as_one = int((support_bound[configuration.pipeline_units_col] < MIN_SUPPORT_UNITS).sum())
    configuration.log_action(STEP_LABEL, 1, f"support of {len(support_bound):,} units; {units_counted_as_one:,} with "
                                            f"nothing due counted as 1")

    # [2] the worst-case binomial error, in percentage points
    support_bound["se_pp_max"] = PERCENTAGE_POINTS * np.sqrt(WORST_CASE_PROPORTION_VARIANCE / support_units)
    support_bound["moe_pp_max"] = configuration.z * support_bound["se_pp_max"]
    configuration.log_action(STEP_LABEL, 2, f"se_pp_max = 100·√(0.25/n) · moe_pp_max = {configuration.z} · se_pp_max: "
                                            f"median ±{support_bound['moe_pp_max'].median():.1f} pp")

    # [3] the same bound in dollars
    support_bound["moe_usd_max"] = (support_bound["moe_pp_max"] / PERCENTAGE_POINTS
                                    * support_bound[configuration.pipeline_usd_col])
    configuration.log_action(STEP_LABEL, 3, "moe_usd_max = moe_pp_max / 100 × USD due")

    # [4] the check
    configuration.log_action(STEP_LABEL, 4, "checking the bounds")
    units_without_bound = int(support_bound["moe_pp_max"].isna().sum())
    configuration.log_check(STEP_LABEL, check_log, "every forecast unit has its bound", units_without_bound == 0,
                            failure_detail=f"{units_without_bound:,} units without a bound",
                            context=f"from ±{support_bound['moe_pp_max'].min():.1f} pp to "
                                    f"±{support_bound['moe_pp_max'].max():.1f} pp")

    # [5] the bounds table, written
    configuration.log_action(STEP_LABEL, 5, "writing the bounds table")
    configuration.write_table(STEP_LABEL, check_log, support_bound, TABLE_UNIT_SUPPORT)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the bound per role
    log_bound_report(support_bound, configuration)
    return support_bound


def log_bound_report(support_bound: pd.DataFrame, configuration: Config) -> None:
    """Action 7: per role, the typical bound of a unit and the bound of the role's total."""
    configuration.log_action(STEP_LABEL, 7, "the bound per role. moe_pp_mediana: the typical unit · "
                                            "moe_usd_total: the bound of the role's total if the units' errors "
                                            "are independent (√Σ moe_usd²), not their sum")
    role_rows = []
    for role in ROLES_IN_ORDER:
        role_units = support_bound[support_bound[CALENDAR_ROLE_COLUMN] == role]
        role_usd = role_units[configuration.pipeline_usd_col].sum()
        independent_total = float(np.sqrt((role_units["moe_usd_max"] ** 2).sum()))
        role_rows.append({"rol": role, "unidades": len(role_units),
                          "moe_pp_mediana": role_units["moe_pp_max"].median() if len(role_units) else np.nan,
                          "usd_vence": role_usd, "moe_usd_total": independent_total,
                          "moe_pct_total": independent_total / role_usd if role_usd else np.nan})
    configuration.show_table(pd.DataFrame(role_rows))
