"""
step_12_pool_series.py — The monthly series of every final group: what the backtest judges.

After the ladder (steps 10-11) every estimable series belongs to ONE final group, and the
group lends its rate to its series. The rate that will be forecast is the rate of that
group, month by month. This step builds that monthly series: for every final group, the
units due and renewed of ITS series (the partition of step 10: every series in one group)
are summed month by month, and its rate is Σ renewed / Σ due of the month. Grouping by the
final group id, the totals add up to the raw.

Only closed truth months enter (entrenamiento, examen): the gaps and the future have no
rate. Every id also gets a reference row: its months, its support (the median units due
of a month), its rate over all its months, how many series take their rate from it, the
money they have to predict, and its GATE:
  · nivel    support ≥ support_floor: the backtest will judge its techniques
  · soporte  support below the floor: not judged, it will take the challenger

Actions (logged as they are done):
  1. the compositions of step 10 and every series in their rate (users and lenders)
  2. the monthly series of every id (units due and renewed summed; rate)
  3. the reference of every id: months, support, rate, series, money, gate
  4. check the series and the reference                             checks 1-3
  5. write the series and the reference                             checks 4-5
  6. count the checks; stop if any failed
  7. show the ids by gate and the largest ones, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every chosen estimation id has a monthly series
   2. the support of every group equals its support in step 10
   3. every monthly rate is between 0 and 1                         (warning only)
   4-5. tables sff_pool_serie and sff_pool_referencia written and read back

Output: (the monthly series of every id, long: id × month; the reference of every id)
· tables sff_pool_serie, sff_pool_referencia.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT, RATE_COLUMN,
                         SERIES_ID_COLUMN, TABLE_POOL_REFERENCE, TABLE_POOL_SERIES)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "12"
STEP_NAME = "POOL SERIES"
STEP_PURPOSE = ("build the monthly series of the rate of every final group of the ladder, summing every "
                "series in it; this is the series the backtest judges and the forecast extends")
STEP_ACTIONS = ["the compositions of step 10 and every series in their rate (users and lenders)",
                "the monthly series of every id (units due and renewed summed; rate)",
                "the reference of every id: months, support, rate, series, money, gate",
                "check the series and the reference (checks 1-3)",
                "write the series and the reference (checks 4-5)",
                "count the checks; stop if any failed",
                "show the ids by gate and the largest ones, as tables"]
STEP_OUTPUT = "one row per id × closed month (units due, renewed, rate) · one row per id · tables sff_pool_serie, sff_pool_referencia"

SUPPORT_TOLERANCE = 1e-9
LARGEST_SHOWN = 5


def build_pool_series(rated_units: pd.DataFrame, ladder: dict, series_estimate: pd.DataFrame,
                      configuration: Config) -> tuple:
    """The monthly series and the reference of every final group; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the compositions of step 10 and every series in their rate
    # every series in the rate of every composition: the ones that use it and the ones that only lend their history
    membership = ladder["composition_members"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]]
    chosen = series_estimate[series_estimate[SERIES_ID_COLUMN].isin(set(membership[SERIES_ID_COLUMN]))]
    chosen_ids = set(membership[COMPOSITION_ID_COLUMN])
    configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} compositions · {len(membership):,} series in their rates "
                                            f"({membership[SERIES_ID_COLUMN].nunique():,} distinct: a series may lend its history "
                                            f"to a composition it does not use)")

    # [2] the monthly series: every matching series summed, month by month
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN,
                                                             configuration.renewed_units_col, configuration.pipeline_units_col]]
    joined = history.merge(membership, on=SERIES_ID_COLUMN)
    pool_series = (joined.groupby([COMPOSITION_ID_COLUMN, period_column])
                   .agg(rol=(CALENDAR_ROLE_COLUMN, "first"),
                        renovadas=(configuration.renewed_units_col, "sum"),
                        vencen=(configuration.pipeline_units_col, "sum"),
                        series_en_el_mes=(SERIES_ID_COLUMN, "nunique"))
                   .reset_index().sort_values([COMPOSITION_ID_COLUMN, period_column]).reset_index(drop=True))
    pool_series[RATE_COLUMN] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
    configuration.log_action(STEP_LABEL, 2, f"{len(pool_series):,} group × month rows; a group has "
                                            f"{pool_series.groupby(COMPOSITION_ID_COLUMN).size().median():.0f} months (median)")

    # [3] the reference of every id
    pool_reference = reference_of_pools(pool_series, chosen, membership, configuration)
    gate_counts = pool_reference["gate"].value_counts().to_dict()
    configuration.log_action(STEP_LABEL, 3, f"gates: {gate_counts} · the judged ids (nivel) carry "
                                            f"{pool_reference.loc[pool_reference['gate'] == GATE_LEVEL, 'usd_por_predecir'].sum() / max(pool_reference['usd_por_predecir'].sum(), 1):.0%} "
                                            f"of the money their series have to predict")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the series and the reference")
    missing_ids = chosen_ids - set(pool_series[COMPOSITION_ID_COLUMN])
    configuration.log_check(STEP_LABEL, check_log, "every final group has a monthly series", not missing_ids,
                            failure_detail=f"{len(missing_ids):,} ids without months: {sorted(missing_ids)[:5]}",
                            context=f"{len(chosen_ids):,} ids")
    support_in_step_10 = (ladder["groups"].drop_duplicates(COMPOSITION_ID_COLUMN)
                          .set_index(COMPOSITION_ID_COLUMN)["group_support"].rename("n_pool_paso_10"))
    compared = pool_reference.join(support_in_step_10, on=COMPOSITION_ID_COLUMN)
    mismatched = compared[(compared["n_pool"] - compared["n_pool_paso_10"]).abs() > SUPPORT_TOLERANCE]
    configuration.log_check(STEP_LABEL, check_log, "the support of every group equals its support in step 10",
                            mismatched.empty,
                            failure_detail=f"{len(mismatched):,} groups whose support differs from step 10",
                            examples=mismatched[[COMPOSITION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
    out_of_range = pool_series[(pool_series[RATE_COLUMN] < 0) | (pool_series[RATE_COLUMN] > 1)]
    configuration.log_check(STEP_LABEL, check_log, "every monthly rate is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} months with a rate outside [0, 1]", blocking=False,
                            examples=out_of_range)

    # [5] the tables, written
    configuration.log_action(STEP_LABEL, 5, "writing the series and the reference")
    configuration.write_table(STEP_LABEL, check_log, pool_series, TABLE_POOL_SERIES)
    configuration.write_table(STEP_LABEL, check_log, pool_reference, TABLE_POOL_REFERENCE)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the ids by gate, and the largest ones
    configuration.log_action(STEP_LABEL, 7, f"final groups by gate (nivel: support ≥ {configuration.support_floor:.0f}, "
                                            f"judged by the backtest; soporte: below it, takes the challenger):")
    configuration.show_table(pool_reference.groupby("gate")
                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), series_que_lo_usan=("series_que_lo_usan", "sum"),
                                  usd_por_predecir=("usd_por_predecir", "sum"), meses_mediana=("meses", "median"))
                             .reset_index())
    configuration.logger.doc(f"[{STEP_LABEL}] the {LARGEST_SHOWN} groups with the most money to predict:")
    configuration.show_table(pool_reference.nlargest(LARGEST_SHOWN, "usd_por_predecir"))
    return pool_series, pool_reference


def reference_of_pools(pool_series: pd.DataFrame, chosen: pd.DataFrame, membership: pd.DataFrame,
                       configuration: Config) -> pd.DataFrame:
    """One row per id: months, support, rate, first and last month, series, money, gate."""
    period_column = configuration.period_col
    with_pipeline = pool_series[pool_series["vencen"] > 0]
    grouped = with_pipeline.groupby(COMPOSITION_ID_COLUMN)
    reference = pd.DataFrame({
        "meses": grouped.size(),
        "primer_mes": grouped[period_column].min(),
        "ultimo_mes": grouped[period_column].max(),
        "n_pool": grouped["vencen"].median(),
        "tasa_pool": grouped["renovadas"].sum() / grouped["vencen"].sum(),
        "series_en_el_pool": membership.groupby(COMPOSITION_ID_COLUMN)[SERIES_ID_COLUMN].nunique()})
    reference["series_que_lo_usan"] = chosen.groupby(COMPOSITION_ID_COLUMN).size()
    reference["usd_por_predecir"] = chosen.groupby(COMPOSITION_ID_COLUMN)["usd_por_predecir"].sum()
    reference = reference.reset_index().rename(columns={"index": COMPOSITION_ID_COLUMN})
    reference[["series_que_lo_usan", "usd_por_predecir"]] = reference[["series_que_lo_usan", "usd_por_predecir"]].fillna(0)
    reference["gate"] = np.where(reference["n_pool"] >= configuration.support_floor, GATE_LEVEL, GATE_SUPPORT)
    return reference
