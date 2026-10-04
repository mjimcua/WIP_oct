"""
step_04_forecast_units.py — The forecast units: one row per rate series in its month.

The fine table has one row per unit AND revaluation values. The renewal rate does not
depend on them (they only change the price), so the rate is
studied and forecast per FORECAST UNIT: the fine rows of a unit are added up here.
Measures are SUMMED, never averaged: every rate later is Σ renewed / Σ due, computed from
these sums. A unit whose renewals are all null (the projection) keeps them null.

Actions (logged as they are done):
  1. group the fine rows by forecast unit
  2. take the description of each unit (series, month, role, flag) from its first row
  3. add up the four measures and count the fine rows of each unit
  4. check the units and that the money is conserved                checks 1-4
  5. write the units table                                          check 5
  6. count the checks; stop if any failed
  7. show the units per role and the fine rows per unit, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. the time_series flag is the same in every fine row of a unit
   2. the four measures are conserved: Σ fine = Σ units
   3. renewals are numbers in closed months and null from the current month on
   4. every unit has something falling due in the extract (the ones the calendar wiped on purpose are
      counted apart)                                                  (warning only)
   5. table sff_fact_fu written and read back

Output: the forecast units (one row per fu_id: ids, keys, series columns, month, role,
current-month mark, time_series flag, number of fine rows, the four measures summed)
· table sff_fact_fu.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, FINE_ROWS_COLUMN, ROLES_IN_ORDER,
                         ROLE_PROJECTION, S0_PIPELINE_UNITS_COLUMN, SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_UNITS,
                         UNIT_ID_COLUMN, UNIT_KEY_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "04"
STEP_NAME = "FORECAST UNITS"
STEP_PURPOSE = ("add up the fine rows of every forecast unit (a rate series in its month): the renewal rate is "
                "studied per unit, because the revaluation values change the price, not the rate")
STEP_ACTIONS = ["group the fine rows by forecast unit",
                "take the description of each unit (series, month, role, flag) from its first row",
                "add up the four measures and count the fine rows of each unit",
                "check the units and that the money is conserved (checks 1-4)",
                "write the units table (check 5)",
                "count the checks; stop if any failed",
                "show the units per role and the fine rows per unit, as tables"]
STEP_OUTPUT = "one row per forecast unit with its four measures summed · table sff_fact_fu"

# ─── named constants ─────────────────────────────────────────────────────────────
# Money is conserved when the difference is below a cent or a billionth of the total
# (float sums over a million rows differ in the last digits).


def build_forecast_units(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Add up the fine rows of every forecast unit; check conservation; write the units table."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    measures = configuration.core_measures
    flag_column = configuration.flag_time_series_col

    # [1] the fine rows of each unit
    grouped_by_unit = fine_table.groupby(UNIT_ID_COLUMN, sort=False)
    configuration.log_action(STEP_LABEL, 1, f"{len(fine_table):,} fine rows grouped into "
                                            f"{grouped_by_unit.ngroups:,} forecast units")

    # [2] the description of each unit: constant inside it (the flag is checked below)
    description_columns = ([SERIES_ID_COLUMN, SERIES_KEY_COLUMN, UNIT_KEY_COLUMN] + configuration.rate_series_columns
                           + [configuration.period_col, CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, flag_column])
    forecast_units = grouped_by_unit[description_columns].first()
    configuration.log_action(STEP_LABEL, 2, f"each unit described by {len(description_columns)} columns "
                                            f"(series, month, role, flag)")

    # [3] the measures, summed (a unit whose values are all null stays null), and the fine rows
    forecast_units[measures] = grouped_by_unit[measures].sum(min_count=1)
    forecast_units[FINE_ROWS_COLUMN] = grouped_by_unit.size()
    forecast_units = forecast_units.reset_index()
    forecast_units = forecast_units[[UNIT_ID_COLUMN] + description_columns + [FINE_ROWS_COLUMN] + measures]
    configuration.log_action(STEP_LABEL, 3, f"measures summed; {forecast_units[FINE_ROWS_COLUMN].mean():.2f} "
                                            f"fine rows per unit on average")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the units and that the money is conserved")
    check_flag_constant_inside_units(fine_table, grouped_by_unit, configuration, check_log)
    check_money_conserved(fine_table, forecast_units, configuration, check_log)
    check_renewals_by_month(forecast_units, configuration, check_log)
    # a unit at 0 units due: either the calendar wiped it on purpose (step 02: the pipeline of a 1-year
    # licence sold or renewed from the current month on; step 17 projects it), or the extract brings it at 0
    units_without_pipeline = forecast_units[forecast_units[configuration.pipeline_units_col] == 0]
    if S0_PIPELINE_UNITS_COLUMN in fine_table.columns:
        due_in_extract = fine_table.groupby(UNIT_ID_COLUMN)[S0_PIPELINE_UNITS_COLUMN].sum()
        wiped_on_purpose = units_without_pipeline[UNIT_ID_COLUMN].map(due_in_extract).fillna(0) > 0
    else:
        wiped_on_purpose = pd.Series(False, index=units_without_pipeline.index)
    zero_in_extract = units_without_pipeline[~wiped_on_purpose]
    configuration.log_action(STEP_LABEL, 4, f"units at 0 units due: {int(wiped_on_purpose.sum()):,} wiped on purpose by the calendar "
                                            f"(step 02, projected in step 17) · {len(zero_in_extract):,} at 0 in the extract")
    configuration.log_check(STEP_LABEL, check_log, "every unit has something falling due in the extract", zero_in_extract.empty,
                            failure_detail=f"{len(zero_in_extract):,} units at 0 units due in the extract itself: their rate is undefined",
                            blocking=False,
                            examples=zero_in_extract[[UNIT_ID_COLUMN, configuration.period_col] + measures])

    # [5] the units table, written
    configuration.log_action(STEP_LABEL, 5, "writing the units table")
    configuration.write_table(STEP_LABEL, check_log, forecast_units, TABLE_UNITS)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] what the units look like
    log_units_report(forecast_units, configuration)
    return forecast_units


def check_flag_constant_inside_units(fine_table: pd.DataFrame, grouped_by_unit, configuration: Config,
                                     check_log: list) -> None:
    """Check 1: the fine rows of a unit agree on the time_series flag. If they did not, the
    unit would belong to two universes at once (an inconsistent extract). Month, role and
    current-month mark are constant by construction: they come from the month."""
    flag_column = configuration.flag_time_series_col
    flag_values_per_unit = grouped_by_unit[flag_column].nunique(dropna=False)
    mixed_units = flag_values_per_unit[flag_values_per_unit > 1].index
    mixed_rows = fine_table[fine_table[UNIT_ID_COLUMN].isin(mixed_units)]
    configuration.log_check(STEP_LABEL, check_log, f"{flag_column} is the same in every fine row of a unit",
                            len(mixed_units) == 0,
                            failure_detail=f"{len(mixed_units):,} units mix rows with {flag_column} 0 and 1 "
                                           f"(one unit, two universes: inconsistent extract)",
                            examples=mixed_rows.sort_values(UNIT_ID_COLUMN)[
                                [UNIT_ID_COLUMN] + configuration.extra_revalorizacion + [flag_column]
                                + configuration.core_measures])


def check_money_conserved(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, configuration: Config,
                          check_log: list) -> None:
    """Check 2: not a cent appears or disappears when the fine rows are added up."""
    differences = {}
    for measure_column in configuration.core_measures:
        fine_total = float(fine_table[measure_column].sum())
        units_total = float(forecast_units[measure_column].sum())
        tolerance = max(configuration.money_tolerance, configuration.money_relative_tolerance * abs(fine_total))
        if abs(units_total - fine_total) > tolerance:
            differences[measure_column] = units_total - fine_total
    pipeline_usd = float(forecast_units[configuration.pipeline_usd_col].sum())
    configuration.log_check(STEP_LABEL, check_log, "the four measures are conserved: Σ fine = Σ units", not differences,
                            failure_detail=f"measures that changed when summed (units − fine): {differences}",
                            context=f"${pipeline_usd:,.0f} due in both tables")


def check_renewals_by_month(forecast_units: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 3: after step 02, a closed unit has a renewal (0 or more) and a projection unit
    has none (null). A null closed unit or a projection unit with a number means the
    fine table lost the calendar on the way."""
    renewed_units = forecast_units[configuration.renewed_units_col]
    projection_units = forecast_units[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    null_closed_units = int((~projection_units & renewed_units.isna()).sum())
    booked_projection_units = int((projection_units & renewed_units.notna()).sum())
    configuration.log_check(STEP_LABEL, check_log,
                            "renewals are numbers in closed months and null from the current month on",
                            null_closed_units == 0 and booked_projection_units == 0,
                            failure_detail=f"{null_closed_units:,} closed units with a null renewal · "
                                           f"{booked_projection_units:,} projection units with a renewal",
                            context=f"{int((~projection_units).sum()):,} closed units · "
                                    f"{int(projection_units.sum()):,} projection units")


def log_units_report(forecast_units: pd.DataFrame, configuration: Config) -> None:
    """Action 7: units per role and fine rows per unit, as tables."""
    configuration.log_action(STEP_LABEL, 7, "units per role (series = distinct rate series with a unit in that role):")
    per_role_rows = []
    for role in ROLES_IN_ORDER:
        role_units = forecast_units[forecast_units[CALENDAR_ROLE_COLUMN] == role]
        per_role_rows.append({CALENDAR_ROLE_COLUMN: role, "unidades": len(role_units), "series": role_units[SERIES_ID_COLUMN].nunique(),
                              "unidades_vencen": role_units[configuration.pipeline_units_col].sum(),
                              "usd_vence": role_units[configuration.pipeline_usd_col].sum()})
    configuration.show_table(pd.DataFrame(per_role_rows))

    configuration.logger.doc(f"[{STEP_LABEL}] fine rows per unit (how many fine rows were added into one unit):")
    fine_rows_count = (forecast_units[FINE_ROWS_COLUMN].value_counts().sort_index()
                          .rename_axis(FINE_ROWS_COLUMN).reset_index(name="unidades"))
    configuration.show_table(fine_rows_count)
