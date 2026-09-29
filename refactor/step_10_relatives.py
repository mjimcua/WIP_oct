"""
step_10_relatives.py — The relatives of every series, and the support and rate of each.

A series below the support floor cannot predict with its own rate: it borrows from a
RELATIVE, a pool of series like it. This step builds, for every estimable series, the
ORDERED LIST of its relatives, from the closest to the farthest, and computes the pool of
every relative with ALL the series that match it (big siblings included: the pattern
decides who computes the number; step 11 decides who receives it).

A relative is a PATTERN: the rate series columns in order, where
  · the timevarying block is summarised as the sign of the series (SIG=neutro / negativo /
    positivo): THE SIGN IS NEVER LOST, a series only pools with series of its sign
  · an annulled or collapsed dimension is written '*'

The rungs:
    series WITH sign (negativo, positivo)          series NEUTRO (no active flag)
    0  itself                                      0  itself
    1  same dims, flags summarised as the sign     —
    2  the annullable extra set to '*'             2  the annullable extra set to '*'
    3  the mandatory cell × sign                   3  the mandatory cell, neutral only
    4… mandatory dims collapsed (step 09 order)    4… mandatory dims collapsed (step 09 order)
       while the cumulative R² lost ≤                 until the total of the neutral series
       signed_ladder_max_loss
A series with flags of both signs (mixto) has only rung 0: it is never pooled.
The annullable extra is the extra_renovacion that separates the rate the least (step 09).

The support of a pool (n_pool) is the median, over its closed months, of the units due
summed over its series: the size of a typical month of the pool, the same measure as
n_propio. Its rate is Σ renewed / Σ due.

Actions (logged as they are done):
  1. what the ladder reads from step 09: the collapse order, the sequential losses and
     the annullable extra
  2. the estimable series (predecible, normal universe) and their columns
  3. the patterns of every rung of every series
  4. the pool of every pattern: support, rate, series in it
  5. check the relatives and the pools                                checks 1-3
  6. write the relatives and the pools                                checks 4-5
  7. count the checks; stop if any failed
  8. show the rungs: how many series have each and how many pools reach the floor

Checks (logged as they are made, numbered, at the level of their status):
   1. rung 0 of every series is itself: its pool has its own support (= n_propio of step 08)
   2. every rung pools at least the series of the rung below (the relatives grow)
   3. a pool never mixes signs
   4-5. tables sff_parientes and sff_pools written and read back

Output: (relatives: fs_id × rung → pattern; pools: pattern → support, rate, series)
· tables sff_parientes, sff_pools.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ID_FIELD_SEPARATOR, Config, id_text
from vocabulario import (PATTERN_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, RUNG_COLUMN, SERIES_ID_COLUMN,
                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_POOLS, TABLE_RELATIVES,
                         UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "10"
STEP_NAME = "RELATIVES AND POOLS"
STEP_PURPOSE = ("build, for every estimable series, the ordered list of the relatives it may borrow support from "
                "(same sign always, then dimensions annulled or collapsed in the order of step 09), and compute the "
                "support and the rate of every relative with all the series that match it")
STEP_ACTIONS = ["read from step 09 the collapse order, the sequential losses and the annullable extra",
                "the estimable series (predecible, normal universe) and their columns",
                "the patterns of every rung of every series",
                "the pool of every pattern: support, rate, series in it",
                "check the relatives and the pools (checks 1-3)",
                "write the relatives and the pools (checks 4-5)",
                "count the checks; stop if any failed",
                "show the rungs: how many series have each and how many reach the floor"]
STEP_OUTPUT = "every series × rung with its relative's pattern · every pattern with its support and rate · tables sff_parientes, sff_pools"

# ─── named constants ─────────────────────────────────────────────────────────────
RUNG_ITSELF, RUNG_SAME_SIGN, RUNG_EXTRA_ANNULLED, RUNG_MANDATORY_CELL = 0, 1, 2, 3
FIRST_COLLAPSE_RUNG = 4
SUPPORT_TOLERANCE = 1e-9


def build_relatives(rated_units: pd.DataFrame, series_rate: pd.DataFrame, series_lookup: pd.DataFrame,
                    dimension_decision: pd.DataFrame, configuration: Config) -> tuple:
    """The relatives of every estimable series and the pool of every relative; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] what step 09 decided
    mandatory_decision = dimension_decision[dimension_decision["orden_colapso"] > 0].sort_values("orden_colapso")
    collapse_order = list(mandatory_decision["dimension"])
    collapse_order += [dimension for dimension in configuration.business_mandatory_dims if dimension not in collapse_order]
    collapse_loss = dict(zip(mandatory_decision["dimension"], mandatory_decision["perdida_secuencial"].fillna(1.0)))
    extras = dimension_decision[dimension_decision["anulable"] == 1]
    annullable_extra = extras.sort_values(["contribucion_unica", "dimension"])["dimension"].iloc[0] if len(extras) else None
    configuration.log_action(STEP_LABEL, 1, f"collapse order {' → '.join(collapse_order)} · annullable extra "
                                            f"{annullable_extra!r} · a signed series may lose up to "
                                            f"{configuration.signed_ladder_max_loss} of R²")

    # [2] the estimable series, with their columns and sign
    estimable = series_rate[(series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE)
                            & (series_rate[UNIVERSE_COLUMN] == UNIVERSE_NORMAL)][[SERIES_ID_COLUMN, SIGN_COLUMN]]
    estimable = estimable.merge(series_lookup[[SERIES_ID_COLUMN] + configuration.rate_series_columns], on=SERIES_ID_COLUMN)
    configuration.log_action(STEP_LABEL, 2, f"{len(estimable):,} estimable series · signs "
                                            f"{estimable[SIGN_COLUMN].value_counts().to_dict()}")

    # [3] the patterns of every rung
    relatives = relative_patterns(estimable, collapse_order, collapse_loss, annullable_extra, configuration)
    configuration.log_action(STEP_LABEL, 3, f"{len(relatives):,} rungs for {relatives[SERIES_ID_COLUMN].nunique():,} "
                                            f"series; {relatives[PATTERN_COLUMN].nunique():,} distinct patterns")

    # [4] the pool of every pattern, with every series that matches it
    pools = pool_support(rated_units, relatives, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{len(pools):,} pools computed; {int((pools['n_pool'] >= configuration.support_floor).sum()):,} "
                                            f"reach the support floor ({configuration.support_floor:.0f})")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the relatives and the pools")
    check_relatives(relatives, pools, series_rate, rated_units, configuration, check_log)

    # [6] the tables, written
    configuration.log_action(STEP_LABEL, 6, "writing the relatives and the pools")
    configuration.write_table(STEP_LABEL, check_log, relatives, TABLE_RELATIVES)
    configuration.write_table(STEP_LABEL, check_log, pools, TABLE_POOLS)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] the rungs
    log_rungs_report(relatives, pools, configuration)
    return relatives, pools


def pattern_of(estimable: pd.DataFrame, configuration: Config, summarise_sign: bool, starred: set) -> pd.Series:
    """The pattern of one rung for every series: the rate series columns in order; the
    timevarying block replaced by SIG=<sign> when summarised; '*' on the starred dims."""
    timevarying = list(configuration.structural_timevarying_dims)
    fields = []
    sign_written = False
    for column_name in configuration.rate_series_columns:
        if column_name in timevarying and summarise_sign:
            if not sign_written:
                fields.append(SIGN_TOKEN + estimable[SIGN_COLUMN].astype(str).to_numpy(dtype=object))
                sign_written = True
            continue
        if column_name in starred:
            fields.append(np.full(len(estimable), WILDCARD, dtype=object))
        else:
            fields.append(id_text(estimable[column_name]))
    if summarise_sign and not sign_written:            # no timevarying declared: the sign still separates pools
        fields.append(SIGN_TOKEN + estimable[SIGN_COLUMN].astype(str).to_numpy(dtype=object))
    joined = fields[0]
    for field in fields[1:]:
        joined = joined + ID_FIELD_SEPARATOR + field
    return pd.Series(joined, index=estimable.index)


def relative_patterns(estimable: pd.DataFrame, collapse_order: list, collapse_loss: dict, annullable_extra,
                      configuration: Config) -> pd.DataFrame:
    """Every (series, rung, description, pattern), built rung by rung for all the series at once."""
    signed = estimable[SIGN_COLUMN].isin([sign for sign in estimable[SIGN_COLUMN].unique()
                                         if sign not in (SIGN_NEUTRAL, SIGN_MIXED)])
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    all_extras = set(configuration.extra_renovacion)
    rung_frames = []

    def add_rung(rung, description, applies, summarise_sign, starred):
        if applies.any():
            rung_series = estimable[applies]
            rung_frames.append(pd.DataFrame({SERIES_ID_COLUMN: rung_series[SERIES_ID_COLUMN].to_numpy(),
                                             RUNG_COLUMN: rung, "descripcion": description,
                                             PATTERN_COLUMN: pattern_of(rung_series, configuration, summarise_sign,
                                                                        starred).to_numpy()}))

    # rung 0: every series is its own first relative (a mixed series has nothing else)
    add_rung(RUNG_ITSELF, "itself", estimable[SERIES_ID_COLUMN].notna(), False, set())
    # rung 1: a signed series with its flags summarised as the sign
    if configuration.structural_timevarying_dims:
        add_rung(RUNG_SAME_SIGN, "same sign", signed, True, set())
    # rung 2: the annullable extra set to '*'
    if annullable_extra:
        add_rung(RUNG_EXTRA_ANNULLED, f"extra '{annullable_extra}' annulled", ~mixed, True, {annullable_extra})
    # rung 3: the mandatory cell × sign (every extra annulled)
    add_rung(RUNG_MANDATORY_CELL, "mandatory cell × sign", ~mixed, True, all_extras)
    # rungs 4…: the mandatory dims collapsed in order; a signed series only while the loss is small
    collapsed, cumulative_loss = set(), 0.0
    for rung, dimension in enumerate(collapse_order, start=FIRST_COLLAPSE_RUNG):
        collapsed = collapsed | {dimension}
        cumulative_loss += float(collapse_loss.get(dimension, 1.0))
        signed_may_climb = (configuration.signed_ladder_max_loss > 0
                            and cumulative_loss <= configuration.signed_ladder_max_loss + SUPPORT_TOLERANCE)
        applies = (estimable[SIGN_COLUMN] == SIGN_NEUTRAL) | (signed & signed_may_climb)
        add_rung(rung, f"mandatory '{dimension}' collapsed", applies, True, all_extras | collapsed)

    relatives = pd.concat(rung_frames, ignore_index=True)
    return relatives.sort_values([SERIES_ID_COLUMN, RUNG_COLUMN]).reset_index(drop=True)


def pool_support(rated_units: pd.DataFrame, relatives: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Support and rate of every pattern with ALL the series that match it: the units due
    summed by month over the pool, n_pool = the median of the months with something due."""
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, configuration.period_col,
                                                             configuration.renewed_units_col,
                                                             configuration.pipeline_units_col]]
    membership = relatives[[SERIES_ID_COLUMN, PATTERN_COLUMN]].drop_duplicates()
    joined = history.merge(membership, on=SERIES_ID_COLUMN)
    monthly = (joined.groupby([PATTERN_COLUMN, configuration.period_col])
               [[configuration.renewed_units_col, configuration.pipeline_units_col]].sum().reset_index())
    months_with_pipeline = monthly[monthly[configuration.pipeline_units_col] > 0]
    pools = pd.DataFrame({
        "n_pool": months_with_pipeline.groupby(PATTERN_COLUMN)[configuration.pipeline_units_col].median(),
        "renovadas_pool": monthly.groupby(PATTERN_COLUMN)[configuration.renewed_units_col].sum(),
        "vencen_pool": monthly.groupby(PATTERN_COLUMN)[configuration.pipeline_units_col].sum(),
        "meses_pool": months_with_pipeline.groupby(PATTERN_COLUMN).size(),
        "series_pool": membership.groupby(PATTERN_COLUMN).size()}).reset_index().rename(columns={"index": PATTERN_COLUMN})
    pools[["n_pool", "renovadas_pool", "vencen_pool", "meses_pool"]] = pools[["n_pool", "renovadas_pool", "vencen_pool",
                                                                              "meses_pool"]].fillna(0)
    pools["tasa_pool"] = pools["renovadas_pool"] / pools["vencen_pool"].where(pools["vencen_pool"] > 0)
    return pools


def check_relatives(relatives: pd.DataFrame, pools: pd.DataFrame, series_rate: pd.DataFrame, rated_units: pd.DataFrame,
                    configuration: Config, check_log: list) -> None:
    """Checks 1 to 3."""
    ladder = relatives.merge(pools, on=PATTERN_COLUMN, how="left")

    # [1] rung 0 is the series itself: same support as step 08 measured
    own = ladder[ladder[RUNG_COLUMN] == RUNG_ITSELF].merge(series_rate[[SERIES_ID_COLUMN, "n_propio"]], on=SERIES_ID_COLUMN)
    mismatched = own[(own["n_pool"].fillna(0) - own["n_propio"]).abs() > SUPPORT_TOLERANCE]
    configuration.log_check(STEP_LABEL, check_log, "rung 0 of every series is itself: its pool has its own support",
                            mismatched.empty and (own["series_pool"] == 1).all(),
                            failure_detail=f"{len(mismatched):,} series whose rung-0 pool differs from n_propio",
                            context=f"{len(own):,} series",
                            examples=mismatched[[SERIES_ID_COLUMN, "n_pool", "n_propio", "series_pool"]])

    # [2] the relatives grow: a rung pools at least the series of the rung below
    ordered = ladder.sort_values([SERIES_ID_COLUMN, RUNG_COLUMN])
    shrinking = ordered[ordered.groupby(SERIES_ID_COLUMN)["series_pool"].diff() < 0]
    configuration.log_check(STEP_LABEL, check_log, "every rung pools at least the series of the rung below",
                            shrinking.empty,
                            failure_detail=f"{len(shrinking):,} rungs with fewer series than the rung below",
                            examples=shrinking[[SERIES_ID_COLUMN, RUNG_COLUMN, PATTERN_COLUMN, "series_pool"]])

    # [3] a pool never mixes signs
    signs_per_pool = (relatives[relatives[RUNG_COLUMN] > RUNG_ITSELF]
                      .merge(series_rate[[SERIES_ID_COLUMN, SIGN_COLUMN]], on=SERIES_ID_COLUMN)
                      .groupby(PATTERN_COLUMN)[SIGN_COLUMN].nunique())
    mixing_pools = signs_per_pool[signs_per_pool > 1]
    configuration.log_check(STEP_LABEL, check_log, "a pool never mixes signs", mixing_pools.empty,
                            failure_detail=f"{len(mixing_pools):,} pools with series of more than one sign",
                            context=f"{len(signs_per_pool):,} pools above rung 0")


def log_rungs_report(relatives: pd.DataFrame, pools: pd.DataFrame, configuration: Config) -> None:
    """Action 8: per rung, how many series have it and how many of their relatives reach the floor."""
    configuration.log_action(STEP_LABEL, 8, f"the rungs (a relative 'reaches the floor' when its typical month has "
                                            f"≥ {configuration.support_floor:.0f} units due):")
    ladder = relatives.merge(pools, on=PATTERN_COLUMN, how="left")
    configuration.show_table(ladder.groupby([RUNG_COLUMN, "descripcion"], sort=True)
                             .agg(series=(SERIES_ID_COLUMN, "size"),
                                  patrones=(PATTERN_COLUMN, "nunique"),
                                  n_pool_mediano=("n_pool", "median"),
                                  pct_llega_al_suelo=("n_pool", lambda n: float((n >= configuration.support_floor).mean())))
                             .reset_index())
