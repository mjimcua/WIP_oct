"""
step_01_values_and_levels.py — The content of the raw: the two universes, the values, and the
dimensions with a generated level.

CONTENT, not structure: step 00 already checked that the columns and the months fit the
configuration. This step reads what is inside, in three parts, as one step:

  THE UNIVERSES   the rows flagged time_series (retail to subscription) carry only the region and
                  the result; the other dimensions and the pipeline are empty. They are separated
                  first, so every check below reads the renewal universe only. Step 20 receives
                  them at the end.

  THE VALUES      money without nulls or negatives, renewals coherent with what fell due,
                  dimensions never empty, flags 0 / 1, discount as a share. Which months each
                  check looks at:
                    · the pipeline (units and USD falling due): every month, the future included
                    · the renewals: only the months BEFORE the current one. From the current month
                      on they are early results that step 02 wipes, so they are not checked.

  THE LEVELS      some mandatory dimensions come with many values (a band, a term). The Config
                  declares them in leveled_dims ({column: "ordinal" | "nominal"}); the column keeps
                  its raw value (the fine level) and this step adds ONE column, <column>_level_1,
                  with its values grouped by their renewal rate. The Config already counts
                  <column>_level_1 as a mandatory dim, right after its column, from the moment it
                  is built; this step only fills it, before the calendar, so every later step sees
                  the same columns. The ladder collapses the fine level first, then the coarse one
                  (step 09).

HOW THE GROUPS ARE MADE (only the TRAINING months, taken from the calendar of the Config)
  1. the STANDARDISED rate of every value: renewed / expected, where expected = the rate of its cell
     (every other mandatory dim of the extract) × its units due, times the global rate. A value is
     compared with the others inside the same cell, not in bulk.
  2. values whose monthly support (median of the monthly units due) is below support_floor go to a
     RESIDUAL group: too little to tell their rate.
  3. the others are ordered (ordinal: by their value, numbers as numbers; nominal: by their rate) and
     the two NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
     the MERGE THRESHOLD.
  4. the rate of every value and of every group, year by year, goes with them as evidence.

THE MERGE THRESHOLD (the same principle as the ladder: merge while the bias it adds is smaller than the
noise it removes). The generated level is only used by the series that climb the ladder, the ones below
the support floor; their monthly rate carries at least the binomial noise of a series AT the floor,
√(p(1−p)/support_floor). Two values whose rates differ by less than that cannot be told apart by any
series that will use the merge: merging them adds less bias than the noise it removes. With p = 0.64 and
a floor of 30, the threshold is 8.8 pp. level_merge_max_pp fixes another value (5 pp = the noise of a
series of about 90 contracts a month). A dimension that ends in a single group does not separate the
renewal rate: its level_1 is constant and the ladder gives it no pass (step 09).

THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
the ids of the forecast series do not move from one month to the next. A dimension missing from the file
(or with another type or another merge threshold) is generated and added. To regenerate everything, delete
the file. A value the file does not know goes to the residual group, with a warning.

Actions (logged as they are done):
   1. check the time_series flag and separate the two universes
   2. scope the renewal checks: renewals are checked only in months before the current one
   3. check the money: pipeline, every month; renewals, closed months
   4. check the dimensions and the timevarying flags
   5. check the exact discount, if declared
   6. look for what is plausible but worth a look (warnings)
   7. the dimensions with a generated level and the file: use it, or generate
   8. generate the groups of every dimension that needs them (training months)
   9. fill <column>_level_1
  10. check the values the file knows
  11. write the levels tables
  12. count the checks; stop if any failed
  13. report the money of the closed months
  14. show the criterion, the groups (rate per year), the values and the merge decisions, and the
      columns by role, as tables

Checks (logged as they are made, numbered in order, at the level of their status):
  universes                      the time_series flag is 0 / 1
                                 every time_series row has every region level
                                 every closed time_series row has its units and value     (warning only)
  pipeline, every month          no nulls · no negatives (units and USD)
  renewals, closed months        no negatives (units and USD)
                                 how "no renewal" arrives: null or 0 (warning if null,
                                   with example rows shown as a table; step 02 reads a null as 0)
                                 units and USD are null together
                                 no renewal for free (units > 0, USD ≤ 0)
                                 no USD without units (USD > 0, units = 0)
  dimensions                     no null or empty value
  flags                          each timevarying column is 0 / 1
  exact discount (if declared)   between 0 and 1, or null (unknown)
  warnings only                  no row renews more units than fall due (rate > 100 %)
                                 no row falls due with units but USD = 0
                                 no row repeated in every column (a double load)
  levels (if declared)           no value of the extract is unknown to the file           (warning only)
                                 tables sff_dimension_levels and sff_dimension_level_values written

Output: (the raw of the renewal universe with <column>_level_1 for every leveled column, the time_series
rows). Without leveled_dims the renewal raw comes back with its rows and columns unchanged.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import (ACTIVE_FLAG_VALUES, GENERATED_LEVEL_SUFFIX, ISOLATED_RATIO_BANDS, STATUS_FAILED, Config,
                    join_columns, region_columns_of, roles_overview)
from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVEL_VALUES, TABLE_DIMENSION_LEVELS


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "01"
STEP_NAME = "VALUES AND DIMENSION LEVELS"
STEP_PURPOSE = ("separate the time_series universe; check that the values inside the raw can be computed with "
                "(money without nulls or negatives, renewals coherent with what fell due, dimensions never empty, "
                "flags 0 / 1, discount as a share); and give every dimension declared in leveled_dims its coarse "
                "level, <column>_level_1: its values grouped by their standardised renewal rate in the training "
                "months, persisted in a JSON that later runs reuse so the forecast series keep their ids")
STEP_ACTIONS = ["check the time_series flag and separate the two universes",
                "scope the renewal checks: renewals are checked only in months before the current one "
                "(from the current month on they are partial and step 02 wipes them)",
                "check the money: pipeline every month, renewals in closed months",
                "check the dimensions and the timevarying flags",
                "check the exact discount, if declared",
                "look for what is plausible but worth a look: warnings",
                "the dimensions with a generated level and the file: use it, or generate",
                "generate the groups of every dimension that needs them (training months)",
                "fill <column>_level_1",
                "check the values the file knows",
                "write the levels tables",
                "count the checks; stop if any failed",
                "report the money of the closed months",
                "show the criterion, the groups (rate per year), the values and the merge decisions, and the "
                "columns by role, as tables"]
STEP_OUTPUT = ("the renewal raw with <column>_level_1 · the time_series rows · the JSON of the groups · tables "
               "sff_dimension_levels (groups) and sff_dimension_level_values (values)")

# ─── named constants ─────────────────────────────────────────────────────────────
# The only values a dichotomous column (timevarying, time_series flag) may carry.
DICHOTOMOUS_VALUES = {0, 1, True, False, "0", "1"}
RESIDUAL_GROUP = "residual"


def validate_values_and_build_levels(validated: pd.DataFrame, configuration: Config) -> tuple:
    """The content of the raw: the two universes, the values and the generated levels.

    INPUT:   the raw validated by step 00 · the Config (flags, measures, dimensions, discount, leveled_dims,
             levels_path, level_merge_max_pp, support_floor, the calendar).
    OUTPUT:  (the renewal raw with <column>_level_1 for every leveled column, the time_series rows).
    RULES:   see the module header.
    EDGE CASES: no time_series row → an empty frame, and step 20 only builds the total. No leveled_dims →
             actions 8-11 and 14 have nothing to do, nothing is written. A dimension with a single value →
             one group. All values below the floor → a single residual group.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the two universes
    renewal_raw, time_series_rows = separate_the_universes(validated, configuration, check_log)

    # [2-6] the values of the renewal universe
    closed_rows = check_the_values(renewal_raw, configuration, check_log)

    # [7-11] the dimensions with a generated level
    leveled_raw, levels_evidence = build_the_levels(renewal_raw, configuration, check_log)

    # [12] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 12, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [13] the money of the closed months, and the rate it implies (a null renewal counts as 0)
    report_the_closed_months(closed_rows, configuration)

    # [14] the evidence of the levels and the columns by role
    if levels_evidence is not None:
        show_the_evidence(*levels_evidence, configuration)
        configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, with the generated levels:")
        configuration.show_table(roles_overview(leveled_raw.columns, configuration))
    else:
        configuration.log_action(STEP_LABEL, 14, "no dimension declared in leveled_dims: no evidence to show")
    return leveled_raw, time_series_rows


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 1 · THE UNIVERSES
# ═══════════════════════════════════════════════════════════════════════════════════

def separate_the_universes(validated: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
    """Action 1: the time_series flag is checked first (a wrong value would send a row to the wrong
    universe), then the flagged rows are taken out. They carry only the region and the result; the rest of
    their dimensions and their pipeline may be empty."""
    flag_column = configuration.flag_time_series_col
    flag_values = validated[flag_column]
    unexpected_values = [value for value in flag_values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
    has_nulls = bool(flag_values.isna().any())
    configuration.log_check(STEP_LABEL, check_log, f"{flag_column} is 0 / 1",
                            not unexpected_values and not has_nulls,
                            failure_detail=f"{flag_column} must be 0 / 1: found {unexpected_values[:10]}"
                                           f"{' and nulls' if has_nulls else ''}",
                            context=f"{flag_values.isin(ACTIVE_FLAG_VALUES).mean():.0%} of rows at 1")

    flagged = flag_values.isin(ACTIVE_FLAG_VALUES)
    time_series_rows = validated[flagged].copy()
    renewal_raw = validated[~flagged].copy()
    region_columns = region_columns_of(configuration)
    configuration.log_action(STEP_LABEL, 1, f"{int(flagged.sum()):,} time_series rows (retail to subscription) out of "
                                            f"{len(validated):,}: they go to step 20 · {len(renewal_raw):,} rows of the "
                                            f"renewal universe go on · region levels {region_columns}")

    missing_region = int(time_series_rows[region_columns].isna().any(axis=1).sum()) if len(time_series_rows) else 0
    finest_regions = time_series_rows[region_columns].drop_duplicates().shape[0] if len(time_series_rows) else 0
    configuration.log_check(STEP_LABEL, check_log, "every time_series row has every region level", missing_region == 0,
                            failure_detail=f"{missing_region:,} time_series rows without some region level",
                            context=f"{finest_regions} finest regions")

    if len(time_series_rows):
        closed = time_series_rows[configuration.period_col] < configuration.calendar_boundaries()["current"]
        result_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]
        missing_closed = int(time_series_rows.loc[closed, result_columns].isna().any(axis=1).sum())
    else:
        missing_closed = 0
    configuration.log_check(STEP_LABEL, check_log, "every closed time_series row has its units and value",
                            missing_closed == 0,
                            failure_detail=f"{missing_closed:,} closed rows without units or value (read as 0)",
                            blocking=False)
    return renewal_raw, time_series_rows


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 2 · THE VALUES
# ═══════════════════════════════════════════════════════════════════════════════════

def check_the_values(renewal_raw: pd.DataFrame, configuration: Config, check_log: list) -> pd.DataFrame:
    """Actions 2-6: the money, the dimensions, the timevarying flags, the discount and the warnings of
    the renewal universe. Returns the rows of the closed months (action 13 reports their money)."""

    # [2] the scope of the renewal checks: the months before the current one. From the current
    #     month on the renewals are partial (the month is still running) and step 02 wipes them,
    #     so only the pipeline is checked there
    current_month = configuration.calendar_boundaries()["current"]
    closed_rows = renewal_raw[renewal_raw[configuration.period_col] < current_month]
    renewed_units = closed_rows[configuration.renewed_units_col]
    renewed_usd = closed_rows[configuration.renewed_usd_col]
    configuration.log_action(STEP_LABEL, 2, f"renewals checked in {len(closed_rows):,} rows before {current_month} · "
                                            f"{len(renewal_raw) - len(closed_rows):,} rows from {current_month} on: "
                                            f"pipeline only, renewals wiped in step 02")

    # [3] the money
    configuration.log_action(STEP_LABEL, 3, "checking the money")
    for pipeline_column in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
        check_no_nulls(configuration, check_log, renewal_raw, pipeline_column, "every month")
        check_no_negatives(configuration, check_log, renewal_raw, pipeline_column, "every month")
    for renewed_column in (configuration.renewed_units_col, configuration.renewed_usd_col):
        check_no_negatives(configuration, check_log, closed_rows, renewed_column, "closed months")
    for measure in configuration.closed_month_measures:
        closed_month_column = measure.raw_column
        check_no_negatives(configuration, check_log, closed_rows, closed_month_column, "closed months")
        # a part cannot be more than its whole: the measure it is a part of, as the Config declares it
        total_column = measure.part_of
        if total_column is None:
            continue                                  # not a part of anything (the second moment): checked below
        above_total = closed_rows[closed_rows[closed_month_column] > closed_rows[total_column] + configuration.money_tolerance]
        configuration.log_check(STEP_LABEL, check_log, f"{closed_month_column} is not above {total_column}, closed months",
                                above_total.empty,
                                failure_detail=f"{len(above_total):,} closed rows with {closed_month_column} above "
                                               f"{total_column} (a part of the renewed above the renewed)",
                                blocking=False, examples=example_rows(above_total, configuration))

    # the dispersion of the isolated renewals: coherent with its base and renewed USD
    base_column, renewed_column = configuration.isolated_renewed_pipeline_usd_col, configuration.isolated_renewed_usd_col
    if configuration.isolated_renewed_usd_sq_over_tr_col:
        moment_column = configuration.isolated_renewed_usd_sq_over_tr_col
        # Σ renewed² / base ≥ (Σ renewed)² / Σ base always holds (a variance cannot be negative): below it, the
        # column was not built as Σ renewed_usd² / due_usd licence by licence
        with_base = closed_rows[closed_rows[base_column] > 0]
        negative_variance = with_base[with_base[moment_column] * with_base[base_column]
                                      < with_base[renewed_column] ** 2 * (1 - configuration.measure_relative_tolerance)
                                      - configuration.money_tolerance]
        configuration.log_check(STEP_LABEL, check_log, f"{moment_column} gives a variance ≥ 0, closed months",
                                negative_variance.empty,
                                failure_detail=f"{len(negative_variance):,} closed rows where Σ renewed² / base is below "
                                               f"(Σ renewed)² / Σ base: not built licence by licence as renewed² / due",
                                blocking=False, examples=example_rows(negative_variance, configuration))
    band_columns = [band[0] for band in configuration.isolated_ratio_bands]
    if len(band_columns) == len(ISOLATED_RATIO_BANDS):
        # the six bands cover every ratio: together they are the isolated base
        band_total = closed_rows[band_columns].fillna(0).sum(axis=1)
        off_base = closed_rows[(band_total - closed_rows[base_column].fillna(0)).abs()
                               > configuration.money_tolerance
                               + configuration.measure_relative_tolerance * closed_rows[base_column].fillna(0).abs()]
        configuration.log_check(STEP_LABEL, check_log, f"the {len(band_columns)} ratio bands add up to {base_column}, closed months",
                                off_base.empty,
                                failure_detail=f"{len(off_base):,} closed rows whose bands do not add up to the isolated "
                                               f"base (a licence in no band, or in two)",
                                blocking=False, examples=example_rows(off_base, configuration))

    # how "nobody renewed" arrives: null (a LEFT JOIN with no match) or 0. A null is not an
    # error but it is not a number either: step 02 reads it as 0.
    no_result_rows = closed_rows[renewed_units.isna() | (renewed_units == 0)]
    null_rows = int(renewed_units.isna().sum())
    zero_rows = int((renewed_units == 0).sum())
    convention_text = f"{null_rows:,} rows null · {zero_rows:,} rows 0 (of {len(closed_rows):,} closed rows)"
    configuration.log_check(STEP_LABEL, check_log, "rows with no renewal arrive as 0, not null, closed months",
                            null_rows == 0,
                            failure_detail=f"{convention_text}: a null means no renewal (read as 0 in step 02)",
                            context=convention_text, blocking=False,
                            examples=example_rows(no_result_rows.sort_values(configuration.renewed_units_col,
                                                                             na_position="first"), configuration))

    null_mismatch = closed_rows[renewed_units.isna() != renewed_usd.isna()]     # one null without the other: broken
    configuration.log_check(STEP_LABEL, check_log, "renewed units and USD are null together, closed months",
                            null_mismatch.empty,
                            failure_detail=f"{len(null_mismatch):,} closed rows have one of the two renewal columns null",
                            examples=example_rows(null_mismatch, configuration))

    free_renewals = closed_rows[(renewed_units > 0) & (renewed_usd <= 0)]
    configuration.log_check(STEP_LABEL, check_log, "no renewal for free (units > 0 with USD ≤ 0), closed months",
                            free_renewals.empty,
                            failure_detail=f"{len(free_renewals):,} closed rows renew units with USD ≤ 0 "
                                           f"(renewing for free does not exist)",
                            examples=example_rows(free_renewals, configuration))
    usd_without_units = closed_rows[(renewed_usd > 0) & (renewed_units == 0)]
    configuration.log_check(STEP_LABEL, check_log, "no renewed USD without renewed units, closed months",
                            usd_without_units.empty,
                            failure_detail=f"{len(usd_without_units):,} closed rows have renewed USD with 0 renewed units",
                            examples=example_rows(usd_without_units, configuration))

    if configuration.renewed_pipeline_usd_col:
        base_column = configuration.renewed_pipeline_usd_col
        base_values = closed_rows[base_column]
        value_without_renewers = closed_rows[(renewed_units == 0) & (base_values > 0)]
        configuration.log_check(STEP_LABEL, check_log, f"{base_column} is 0 where nothing renewed, closed months",
                                value_without_renewers.empty,
                                failure_detail=f"{len(value_without_renewers):,} closed rows carry a renewers' due value "
                                               f"with 0 renewed units",
                                blocking=False, examples=example_rows(value_without_renewers, configuration))
        renewers_without_value = closed_rows[(renewed_units > 0) & (base_values.fillna(0) == 0)]
        configuration.log_check(STEP_LABEL, check_log, f"{base_column} is above 0 where something renewed, closed months",
                                renewers_without_value.empty,
                                failure_detail=f"{len(renewers_without_value):,} closed rows renew units with no renewers' "
                                               f"due value (their uplift cannot be computed: they are left out of it)",
                                blocking=False, examples=example_rows(renewers_without_value, configuration))

    # [4] the dimensions and the timevarying flags
    configuration.log_action(STEP_LABEL, 4, f"checking {len(dimension_columns(configuration))} dimensions and "
                                            f"{len(configuration.structural_timevarying_dims)} timevarying flags")
    empty_by_dimension = {}
    for dimension_column in dimension_columns(configuration):
        values = renewal_raw[dimension_column]
        empty_rows = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        if empty_rows:
            empty_by_dimension[dimension_column] = empty_rows
    configuration.log_check(STEP_LABEL, check_log, "no dimension has a null or empty value", not empty_by_dimension,
                            failure_detail=f"empty values in dimensions (rows per column): {empty_by_dimension}",
                            context=f"{len(dimension_columns(configuration))} dimensions")

    # the share of rows AND the share of the USD due: a flag on many small rows weighs little in money
    usd_due = renewal_raw[configuration.pipeline_usd_col]
    total_usd_due = float(usd_due.sum())

    def share_of_money(rows_mask) -> str:
        return f"{usd_due[rows_mask].sum() / total_usd_due:.0%} of the USD due" if total_usd_due else "no USD due"

    for timevarying_column in configuration.structural_timevarying_dims:
        values = renewal_raw[timevarying_column]
        unexpected_values = [value for value in values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
        has_nulls = bool(values.isna().any())
        configuration.log_check(STEP_LABEL, check_log, f"{timevarying_column} is 0 / 1",
                                not unexpected_values and not has_nulls,
                                failure_detail=f"{timevarying_column} must be 0 / 1: found {unexpected_values[:10]}"
                                               f"{' and nulls' if has_nulls else ''}",
                                context=f"{values.isin(ACTIVE_FLAG_VALUES).mean():.0%} of rows at 1 · "
                                        f"{share_of_money(values.isin(ACTIVE_FLAG_VALUES))}")

    # [5] the exact discount: a share between 0 and 1, or null (unknown)
    if configuration.discount_value_column:
        discount_values = renewal_raw[configuration.discount_value_column]
        out_of_range = int(((discount_values < 0) | (discount_values > 1)).sum())
        configuration.log_action(STEP_LABEL, 5, f"checking {configuration.discount_value_column}")
        configuration.log_check(STEP_LABEL, check_log,
                                f"{configuration.discount_value_column} is between 0 and 1 (or null = unknown)",
                                out_of_range == 0,
                                failure_detail=f"{out_of_range:,} rows of {configuration.discount_value_column} "
                                               f"outside [0, 1] (the discount is a share: 0.25 = 25 %)",
                                context=f"{discount_values.isna().mean():.1%} of rows unknown · "
                                        f"{share_of_money(discount_values.isna())}")
    else:
        configuration.log_action(STEP_LABEL, 5, "no exact discount declared: nothing to check")

    # [6] warnings: plausible in a real extract, worth a look
    configuration.log_action(STEP_LABEL, 6, "looking for what is plausible but worth a look")
    above_pipeline = closed_rows[renewed_units > closed_rows[configuration.pipeline_units_col]]
    excess_units = float((above_pipeline[configuration.renewed_units_col]
                          - above_pipeline[configuration.pipeline_units_col]).sum())
    configuration.log_check(STEP_LABEL, check_log, "no row renews more units than fall due, closed months",
                            above_pipeline.empty,
                            failure_detail=f"{len(above_pipeline):,} closed rows renew more units than fall due "
                                           f"(+{excess_units:,.0f} units: a rate above 100 %)",
                            blocking=False, examples=example_rows(above_pipeline, configuration))
    units_without_value = int(((renewal_raw[configuration.pipeline_units_col] > 0)
                               & (renewal_raw[configuration.pipeline_usd_col] == 0)).sum())
    configuration.log_check(STEP_LABEL, check_log, "no row falls due with units but USD = 0", units_without_value == 0,
                            failure_detail=f"{units_without_value:,} rows fall due with units but USD = 0",
                            blocking=False)
    repeated_rows = int(renewal_raw.duplicated().sum())
    configuration.log_check(STEP_LABEL, check_log, "no row repeated in every column", repeated_rows == 0,
                            failure_detail=f"{repeated_rows:,} rows repeated in every column (a possible double load)",
                            blocking=False)
    return closed_rows


def report_the_closed_months(closed_rows: pd.DataFrame, configuration: Config) -> None:
    """Action 13: the money of the closed months, and the rate it implies (a null renewal counts as 0)."""
    pipeline_units = closed_rows[configuration.pipeline_units_col].sum()
    pipeline_usd = closed_rows[configuration.pipeline_usd_col].sum()
    renewed_units = closed_rows[configuration.renewed_units_col].sum()
    renewed_usd = closed_rows[configuration.renewed_usd_col].sum()
    renewal_rate_text = f"{renewed_units / pipeline_units:.1%}" if pipeline_units else "no units due"
    configuration.log_action(STEP_LABEL, 13, f"closed months: {pipeline_units:,.0f} units due · "
                                             f"{renewed_units:,.0f} renewed ({renewal_rate_text}) · "
                                             f"${pipeline_usd:,.0f} due · ${renewed_usd:,.0f} renewed")


def check_no_nulls(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    null_rows = int(rows[column_name].isna().sum())
    configuration.log_check(STEP_LABEL, check_log, f"{column_name} has no nulls, {scope}", null_rows == 0,
                            failure_detail=f"{null_rows:,} nulls in {column_name} ({scope})")


def check_no_negatives(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    negative_rows = int((rows[column_name] < 0).sum())
    configuration.log_check(STEP_LABEL, check_log, f"{column_name} has no negatives, {scope}", negative_rows == 0,
                            failure_detail=f"{negative_rows:,} negative values in {column_name} ({scope})")


def example_rows(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The columns that identify a row and its money, to show a few rows under a check."""
    return rows[[configuration.period_col] + configuration.extract_mandatory_dims + configuration.core_measures]


def dimension_columns(configuration: Config) -> list:
    """Every dimension column, each once: mandatory, timevarying and both extra groups."""
    all_dimensions = (configuration.extract_mandatory_dims + list(configuration.structural_timevarying_dims)
                      + configuration.extra_renovacion + configuration.extra_revalorizacion)
    return list(dict.fromkeys(all_dimensions))


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 3 · THE DIMENSIONS WITH A GENERATED LEVEL
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_levels(renewal_raw: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
    """Actions 7-11: the generated coarse level of every dimension in leveled_dims. Returns (the raw with
    <column>_level_1, the evidence to show in action 14 — None when no dimension is leveled)."""
    if not configuration.leveled_dims:
        configuration.log_action(STEP_LABEL, 7, "no dimension declared in leveled_dims: nothing to generate")
        return renewal_raw, None

    # the groups are made from the values: if a check of the values already failed, they would be made
    # from wrong data (the step stops in action 12 anyway)
    if any(logged_check[0] == STATUS_FAILED for logged_check in check_log):
        configuration.log_action(STEP_LABEL, 7, "the levels are not built: a check of the values failed")
        return renewal_raw, None
    raw = renewal_raw.copy()

    # [7] the declared dimensions and the file
    path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
    stored = read_levels_file(path)
    to_generate = [column_name for column_name, level_type in configuration.leveled_dims.items()
                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type
                   or stored["dims"][column_name].get("criterion", {}).get("fixed_pp") != configuration.level_merge_max_pp]
    configuration.log_action(STEP_LABEL, 7, f"{len(configuration.leveled_dims)} dimensions with a generated level "
                                            f"{list(configuration.leveled_dims)} · file {path}: "
                                            + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
                                            + (f"generating {to_generate}" if to_generate else "every dimension in it, reused"))

    # [8] generate what the file does not have
    if to_generate:
        stored.setdefault("dims", {})
        for column_name in to_generate:
            stored["dims"][column_name] = generate_levels(raw, column_name, configuration.leveled_dims[column_name], configuration)
        stored["created"] = stored.get("created") or datetime.now().isoformat(timespec="seconds")
        stored["updated"] = datetime.now().isoformat(timespec="seconds")
        stored["current_month"] = str(configuration.current_month)
        write_levels_file(path, stored)
        configuration.log_action(STEP_LABEL, 8, f"groups generated from the training months and written to {path}")
    else:
        configuration.log_action(STEP_LABEL, 8, "nothing to generate")

    # [9] fill the coarse level
    unknown_values = {}
    for column_name in configuration.leveled_dims:
        mapping = stored["dims"][column_name]["mapping"]
        values = raw[column_name].astype(str)
        unknown = sorted(set(values.unique()) - set(mapping))
        if unknown:
            unknown_values[column_name] = unknown
        raw[f"{column_name}{GENERATED_LEVEL_SUFFIX}"] = values.map(mapping).fillna(RESIDUAL_GROUP)
    configuration.log_action(STEP_LABEL, 9, "filled: " + ", ".join(configuration.generated_columns))

    # [10] the check
    configuration.log_action(STEP_LABEL, 10, "checking the values the file knows")
    configuration.log_check(STEP_LABEL, check_log, "no value of the extract is unknown to the file", not unknown_values,
                            failure_detail="values sent to the residual: " + "; ".join(f"{column_name}: {values[:10]}"
                                                                                     for column_name, values in unknown_values.items()),
                            blocking=False)

    # [11] the tables
    configuration.log_action(STEP_LABEL, 11, "writing the levels")
    levels_table, values_table, decisions_table = levels_as_tables(stored, configuration)
    configuration.write_table(STEP_LABEL, check_log, levels_table, TABLE_DIMENSION_LEVELS)
    configuration.write_table(STEP_LABEL, check_log, values_table, TABLE_DIMENSION_LEVEL_VALUES)
    return raw, (stored, levels_table, values_table, decisions_table)


def natural_key(value: str) -> list:
    """'2 devices' before '10 devices': numbers compared as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def merge_threshold_pp(global_rate: float, configuration: Config) -> tuple:
    """(threshold in pp, how it was set): the binomial noise of a series at the support floor, or the fixed value."""
    if configuration.level_merge_max_pp is not None:
        return float(configuration.level_merge_max_pp), f"fixed in the Config (level_merge_max_pp = {configuration.level_merge_max_pp})"
    noise = 100 * np.sqrt(global_rate * (1 - global_rate) / configuration.support_floor)
    return float(noise), (f"the binomial noise of a series at the support floor: 100·√(p(1−p)/{configuration.support_floor:.0f}) "
                          f"with p = {global_rate:.2f}")


def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
    """The groups of one dimension, from the training months: standardised rate per value, residual for the
    rare ones, neighbours merged while their rates differ by at most the merge threshold. Keeps the evidence:
    every value, every group year by year, every merge decision."""
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
    train = raw[training & (raw[due].to_numpy() > 0)].copy()
    train["_value"] = train[column_name].astype(str)
    other_dims = [other for other in configuration.business_mandatory_dims
                  if other != column_name and other not in configuration.generated_columns]
    train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
    cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
    train["_expected"] = train["_cell"].map(cell_rate) * train[due]
    train["_year"] = train[period].map(lambda month: month.year)
    global_rate = float(train[renewed].sum() / train[due].sum())
    threshold_pp, threshold_rule = merge_threshold_pp(global_rate, configuration)

    per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
    per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
    rare = per_value["monthly_support"] < configuration.support_floor

    def standardised(values: list, rows: pd.DataFrame = None) -> float:
        if rows is None:
            block = per_value.loc[values]
            return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)
        expected = rows["_expected"].sum()
        return float(global_rate * rows[renewed].sum() / expected) if expected > 0 else np.nan

    def by_year(values: list) -> dict:
        block = train[train["_value"].isin(values)]
        return {str(year): round(standardised(values, rows), 4) for year, rows in block.groupby("_year")}

    groups = [[value] for value in per_value.index[~rare]]
    if level_type == "ordinal":
        groups.sort(key=lambda group: natural_key(group[0]))
    else:
        groups.sort(key=lambda group: standardised(group))

    def label(group: list) -> str:
        ordered = sorted(group, key=natural_key)
        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"

    # merge the closest neighbours while they differ by at most the threshold; every decision is kept
    decisions = []
    while len(groups) > 1:
        differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
        closest = int(np.argmin(differences))
        if 100 * differences[closest] > threshold_pp:
            break
        decisions.append({"left": label(groups[closest]), "right": label(groups[closest + 1]),
                          "difference_pp": round(100 * differences[closest], 2), "decision": "merged"})
        groups[closest:closest + 2] = [groups[closest] + groups[closest + 1]]
    for index in range(len(groups) - 1):                       # the neighbours that stay apart, and by how much
        decisions.append({"left": label(groups[index]), "right": label(groups[index + 1]),
                          "difference_pp": round(100 * abs(standardised(groups[index]) - standardised(groups[index + 1])), 2),
                          "decision": "kept apart"})

    mapping, group_evidence, value_evidence = {}, [], []
    final_groups = [(label(group), group) for group in groups]
    if rare.any():
        final_groups.append((RESIDUAL_GROUP, list(per_value.index[rare])))
    for group_label, values in final_groups:
        for value in values:
            mapping[value] = group_label
            value_rate = standardised([value])
            support = float(per_value.loc[value, "monthly_support"])
            value_evidence.append({"value": value, "group": group_label, "units_due": float(per_value.loc[value, "units_due"]),
                                   "monthly_support": support, "rate_std": round(value_rate, 4),
                                   "noise_pp": round(100 * np.sqrt(max(value_rate * (1 - value_rate), 0) / max(support, 1)), 2),
                                   "rate_by_year": by_year([value])})
        block = train[train["_value"].isin(values)]
        group_evidence.append({"group": group_label, "values": sorted(values, key=natural_key),
                               "units_due": float(per_value.loc[values, "units_due"].sum()),
                               "monthly_support": float(block.groupby(period)[due].sum().median()),
                               "rate_std": round(standardised(values), 4), "rate_by_year": by_year(values)})
    value_evidence.sort(key=lambda row: natural_key(row["value"]))
    return {"type": level_type,
            "criterion": {"threshold_pp": round(threshold_pp, 2), "rule": threshold_rule, "fixed_pp": configuration.level_merge_max_pp,
                          "global_rate": round(global_rate, 4), "support_floor": configuration.support_floor},
            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping,
            "groups": group_evidence, "values": value_evidence, "decisions": decisions}


def read_levels_file(path: str) -> dict:
    """The stored groups, or an empty dict when there is no file yet."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_levels_file(path: str, content: dict) -> None:
    """The groups, as indented JSON (readable and editable by hand)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)


def levels_as_tables(stored: dict, configuration: Config) -> tuple:
    """The evidence of the JSON as tables: groups, values and merge decisions (one row each), with the rate
    of every year as a column of its own, readable on screen and in Power BI."""
    group_rows, value_rows, decision_rows = [], [], []
    for column_name in configuration.leveled_dims:
        stored_dimension = stored["dims"][column_name]
        threshold = stored_dimension.get("criterion", {}).get("threshold_pp")
        for group in stored_dimension["groups"]:
            group_rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
                               "n_values": len(group["values"]), "units_due": group["units_due"],
                               "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                               **{f"rate_{year}": rate for year, rate in group["rate_by_year"].items()}})
        for value in stored_dimension.get("values", []):
            value_rows.append({"dimension": column_name, "value": value["value"], "group": value["group"],
                               "units_due": value["units_due"], "monthly_support": value["monthly_support"],
                               "rate_std": value["rate_std"], "noise_pp": value["noise_pp"],
                               **{f"rate_{year}": rate for year, rate in value["rate_by_year"].items()}})
        for decision in stored_dimension.get("decisions", []):
            decision_rows.append({"dimension": column_name, **decision, "threshold_pp": threshold,
                                  "within_5_pp": "yes" if decision["difference_pp"] <= 5 else "no"})
    return pd.DataFrame(group_rows), pd.DataFrame(value_rows), pd.DataFrame(decision_rows)


def show_the_evidence(stored: dict, levels_table: pd.DataFrame, values_table: pd.DataFrame,
                      decisions_table: pd.DataFrame, configuration: Config) -> None:
    """Action 14: the criterion, then the groups year by year, the values and the merge decisions."""
    for column_name in configuration.leveled_dims:
        criterion = stored["dims"][column_name].get("criterion", {})
        groups = [group for group in stored["dims"][column_name]["groups"] if group["group"] != RESIDUAL_GROUP]
        configuration.log_action(STEP_LABEL, 14, f"{column_name} ({stored['dims'][column_name]['type']}): neighbours merge while their "
                                                f"standardised rates differ by ≤ {criterion.get('threshold_pp', '?')} pp — "
                                                f"{criterion.get('rule', '?')} · {len(groups)} groups"
                                                + (" · ONE group: the dimension does not separate the renewal rate beyond that "
                                                   "noise; its level_1 is constant and gets no pass of the ladder (step 09)"
                                                   if len(groups) == 1 else ""))
    rate = next(iter(stored["dims"].values())).get("criterion", {}).get("global_rate", 0.64)
    noise_rows = [{"contracts_a_month": support, "binomial_noise_pp": round(100 * np.sqrt(rate * (1 - rate) / support), 1)}
                  for support in (10, 30, 50, 100, 271, 1000)]
    configuration.logger.doc(f"[{STEP_LABEL}] what a difference in pp means: the noise of the monthly rate of a series of n contracts "
                             f"(p = {rate:.2f}); a merge helps the series whose noise is larger than the difference it adds:")
    configuration.show_table(pd.DataFrame(noise_rows))
    configuration.logger.doc(f"[{STEP_LABEL}] the groups (rate_std: standardised by cell; rate_<year>: the same, year by year — "
                             f"a group is stable when its years stay close):")
    configuration.show_table(levels_table)
    configuration.logger.doc(f"[{STEP_LABEL}] every value (noise_pp: the binomial noise of its own monthly rate; two values whose "
                             f"rates differ by less than their noise cannot be told apart month by month):")
    configuration.show_table(values_table)
    configuration.logger.doc(f"[{STEP_LABEL}] the merge decisions, in order (within_5_pp: what the old fixed 5 pp would have said):")
    configuration.show_table(decisions_table)
