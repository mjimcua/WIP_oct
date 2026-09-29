"""
step_12_pool_series.py — The monthly series of every estimation id: what the backtest judges.

After the ladder (step 11) every estimable series takes its rate from an ESTIMATION ID:
itself (rung 0) or the pattern of the relative it chose. The rate that will be forecast
is the rate of that estimation id, month by month. This step builds that monthly series:
for every chosen estimation id, the units due and renewed of EVERY series that matches its
pattern are summed month by month (big siblings included: the pattern decides who computes
the number), and its rate is Σ renewed / Σ due of the month.

Only closed truth months enter (entrenamiento, examen): the gaps and the future have no
rate. Every id also gets a reference row: its months, its support (the median units due
of a month), its rate over all its months, how many series take their rate from it, the
money they have to predict, and its GATE:
  · nivel    support ≥ support_floor: the backtest will judge its techniques
  · soporte  support below the floor: not judged, it will take the challenger

Actions (logged as they are done):
  1. the estimation ids chosen in step 11, and the series that match each one
  2. the monthly series of every id (units due and renewed summed; rate)
  3. the reference of every id: months, support, rate, series, money, gate
  4. check the series and the reference                             checks 1-3
  5. write the series and the reference                             checks 4-5
  6. count the checks; stop if any failed
  7. show the ids by gate and the largest ones, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every chosen estimation id has a monthly series
   2. the support of every id equals the support of its pool in step 10
   3. every monthly rate is between 0 and 1                         (warning only)
   4-5. tables sff_pool_serie and sff_pool_referencia written and read back

Output: (the monthly series of every id, long: id × month; the reference of every id)
· tables sff_pool_serie, sff_pool_referencia.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT, PATTERN_COLUMN,
                         RATE_COLUMN, SERIES_ID_COLUMN, TABLE_POOL_REFERENCE, TABLE_POOL_SERIES)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "12"
STEP_NAME = "POOL SERIES"
STEP_PURPOSE = ("build the monthly series of the rate of every estimation id chosen by the ladder, summing every "
                "series that matches it; this is the series the backtest judges and the forecast extends")
STEP_ACTIONS = ["the estimation ids chosen in step 11, and the series that match each one",
                "the monthly series of every id (units due and renewed summed; rate)",
                "the reference of every id: months, support, rate, series, money, gate",
                "check the series and the reference (checks 1-3)",
                "write the series and the reference (checks 4-5)",
                "count the checks; stop if any failed",
                "show the ids by gate and the largest ones, as tables"]
STEP_OUTPUT = "one row per id × closed month (units due, renewed, rate) · one row per id · tables sff_pool_serie, sff_pool_referencia"

SUPPORT_TOLERANCE = 1e-9
LARGEST_SHOWN = 5


def build_pool_series(rated_units: pd.DataFrame, relatives: pd.DataFrame, pools: pd.DataFrame,
                      series_estimate: pd.DataFrame, configuration: Config) -> tuple:
    """The monthly series and the reference of every chosen estimation id; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the chosen ids (only estimable series have a pattern) and who matches each
    chosen = series_estimate[series_estimate[ESTIMATION_ID_COLUMN].isin(set(relatives[PATTERN_COLUMN]))]
    chosen_ids = set(chosen[ESTIMATION_ID_COLUMN])
    membership = (relatives[relatives[PATTERN_COLUMN].isin(chosen_ids)][[SERIES_ID_COLUMN, PATTERN_COLUMN]]
                  .drop_duplicates().rename(columns={PATTERN_COLUMN: ESTIMATION_ID_COLUMN}))
    configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} estimation ids chosen by {len(chosen):,} series; "
                                            f"{len(membership):,} (series, id) memberships")

    # [2] the monthly series: every matching series summed, month by month
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN,
                                                             configuration.renewed_units_col, configuration.pipeline_units_col]]
    joined = history.merge(membership, on=SERIES_ID_COLUMN)
    pool_series = (joined.groupby([ESTIMATION_ID_COLUMN, period_column])
                   .agg(rol=(CALENDAR_ROLE_COLUMN, "first"),
                        renovadas=(configuration.renewed_units_col, "sum"),
                        vencen=(configuration.pipeline_units_col, "sum"),
                        series_en_el_mes=(SERIES_ID_COLUMN, "nunique"))
                   .reset_index().sort_values([ESTIMATION_ID_COLUMN, period_column]).reset_index(drop=True))
    pool_series["tasa"] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
    configuration.log_action(STEP_LABEL, 2, f"{len(pool_series):,} id × month rows; a series has "
                                            f"{pool_series.groupby(ESTIMATION_ID_COLUMN).size().median():.0f} months (median)")

    # [3] the reference of every id
    pool_reference = reference_of_pools(pool_series, chosen, membership, configuration)
    gate_counts = pool_reference["gate"].value_counts().to_dict()
    configuration.log_action(STEP_LABEL, 3, f"gates: {gate_counts} · the judged ids (nivel) carry "
                                            f"{pool_reference.loc[pool_reference['gate'] == GATE_LEVEL, 'usd_por_predecir'].sum() / max(pool_reference['usd_por_predecir'].sum(), 1):.0%} "
                                            f"of the money their series have to predict")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the series and the reference")
    missing_ids = chosen_ids - set(pool_series[ESTIMATION_ID_COLUMN])
    configuration.log_check(STEP_LABEL, check_log, "every chosen estimation id has a monthly series", not missing_ids,
                            failure_detail=f"{len(missing_ids):,} ids without months: {sorted(missing_ids)[:5]}",
                            context=f"{len(chosen_ids):,} ids")
    compared = pool_reference.merge(pools[[PATTERN_COLUMN, "n_pool"]].rename(columns={PATTERN_COLUMN: ESTIMATION_ID_COLUMN,
                                                                                      "n_pool": "n_pool_paso_10"}),
                                    on=ESTIMATION_ID_COLUMN, how="left")
    mismatched = compared[(compared["n_pool"] - compared["n_pool_paso_10"]).abs() > SUPPORT_TOLERANCE]
    configuration.log_check(STEP_LABEL, check_log, "the support of every id equals the support of its pool in step 10",
                            mismatched.empty,
                            failure_detail=f"{len(mismatched):,} ids whose support differs from step 10",
                            examples=mismatched[[ESTIMATION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
    out_of_range = pool_series[(pool_series["tasa"] < 0) | (pool_series["tasa"] > 1)]
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
    configuration.log_action(STEP_LABEL, 7, f"estimation ids by gate (nivel: support ≥ {configuration.support_floor:.0f}, "
                                            f"judged by the backtest; soporte: below it, takes the challenger):")
    configuration.show_table(pool_reference.groupby("gate")
                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), series_que_lo_usan=("series_que_lo_usan", "sum"),
                                  usd_por_predecir=("usd_por_predecir", "sum"), meses_mediana=("meses", "median"))
                             .reset_index())
    configuration.logger.doc(f"[{STEP_LABEL}] the {LARGEST_SHOWN} ids with the most money to predict:")
    configuration.show_table(pool_reference.nlargest(LARGEST_SHOWN, "usd_por_predecir"))
    return pool_series, pool_reference


def reference_of_pools(pool_series: pd.DataFrame, chosen: pd.DataFrame, membership: pd.DataFrame,
                       configuration: Config) -> pd.DataFrame:
    """One row per id: months, support, rate, first and last month, series, money, gate."""
    period_column = configuration.period_col
    with_pipeline = pool_series[pool_series["vencen"] > 0]
    grouped = with_pipeline.groupby(ESTIMATION_ID_COLUMN)
    reference = pd.DataFrame({
        "meses": grouped.size(),
        "primer_mes": grouped[period_column].min(),
        "ultimo_mes": grouped[period_column].max(),
        "n_pool": grouped["vencen"].median(),
        "tasa_pool": grouped["renovadas"].sum() / grouped["vencen"].sum(),
        "series_en_el_pool": membership.groupby(ESTIMATION_ID_COLUMN)[SERIES_ID_COLUMN].nunique()})
    reference["series_que_lo_usan"] = chosen.groupby(ESTIMATION_ID_COLUMN).size()
    reference["usd_por_predecir"] = chosen.groupby(ESTIMATION_ID_COLUMN)["usd_por_predecir"].sum()
    reference = reference.reset_index().rename(columns={"index": ESTIMATION_ID_COLUMN})
    reference[["series_que_lo_usan", "usd_por_predecir"]] = reference[["series_que_lo_usan", "usd_por_predecir"]].fillna(0)
    reference["gate"] = np.where(reference["n_pool"] >= configuration.support_floor, GATE_LEVEL, GATE_SUPPORT)
    return reference
