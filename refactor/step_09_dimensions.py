"""
step_09_dimensions.py — Which dimensions separate the renewal rate, and in which order the
ladder will let the mandatory ones go.

The ladder of step 10 lends support to a small series by collapsing it with its relatives:
it drops mandatory dimensions one by one until there is enough evidence. WHICH one it drops
first matters: dropping a dimension that separates the rate a lot mixes series that renew
differently. This step measures it, on the series whose own rate is informative:
route predecible, own rate known, sign neutro (a series with an active signal would
contaminate the rates: the signals are treated apart), weighted by n_propio.

For every mandatory and extra_renovacion dimension:
  · eta2_individual     η²: the share of the variance of the rate between its values, alone
  · contribucion_unica  the R² lost when it is removed from the model with all dimensions
  · omega2              ω²: η² corrected for the number of values (a dimension with many
                        values separates by chance too)
  · anulable            1 for extra_renovacion: the ladder annuls them before any mandatory
For the mandatory dimensions, the COLLAPSE ORDER: at every round, among the ones that can
go (a *_level_N only once its *_level_(N+1) has gone), the one that loses the least R²
given the ones that remain. That sequential loss is the question the ladder asks at each
rung (the unique contribution is not: in a nested hierarchy every coarser level has
unique ≈ 0 while the finer one is there).
For every pair of dimensions: the η² of the pair and its interaction (η² of the pair −
the larger η² of the two): pairs that separate more together than apart.

Actions (logged as they are done):
  1. the base: the series whose own rate is informative, with their dimension values
  2. η², unique contribution and ω² of every dimension
  3. the collapse order of the mandatory dimensions
  4. the pairs of dimensions and their interaction
  5. check the figures and the order                                 checks 1-4
  6. write the two decision tables                                   checks 5-6
  7. count the checks; stop if any failed
  8. show the dimensions in collapse order and the top pairs, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. the base has at least 3 series (else every figure is 0)       (warning only)
   2. every figure is between 0 and 1
   3. every mandatory dimension has one collapse position, 1 to M
   4. a *_level_N never collapses before its *_level_(N+1)
   5-6. tables sff_decision_eta2 and sff_decision_eta2_pares written and read back

Output: (the decision per dimension, the top pairs) · tables sff_decision_eta2,
sff_decision_eta2_pares. The ladder (step 10) reads the collapse order.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import itertools

import numpy as np
import pandas as pd

from config import Config
from vocabulario import (ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN, SIGN_COLUMN, SIGN_NEUTRAL,
                         TABLE_DIMENSION_PAIRS, TABLE_DIMENSIONS)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "09"
STEP_NAME = "DIMENSIONS"
STEP_PURPOSE = ("measure how much every dimension separates the renewal rate (η², unique contribution, ω²) and "
                "decide the order in which the ladder will collapse the mandatory ones: first the one whose loss "
                "costs the least R² given the ones that remain")
STEP_ACTIONS = ["the base: the series whose own rate is informative, with their dimension values",
                "η², unique contribution and ω² of every dimension",
                "the collapse order of the mandatory dimensions",
                "the pairs of dimensions and their interaction",
                "check the figures and the order (checks 1-4)",
                "write the two decision tables (checks 5-6)",
                "count the checks; stop if any failed",
                "show the dimensions in collapse order and the top pairs, as tables"]
STEP_OUTPUT = ("one row per dimension (η², unique contribution, ω², collapse order, sequential loss) · the top pairs · "
               "tables sff_decision_eta2, sff_decision_eta2_pares")

# ─── named constants ─────────────────────────────────────────────────────────────
MIN_SERIES_IN_BASE = 3          # with fewer series every figure is 0: no evidence, declared
VALUE_COLUMN = "tasa_propia"
WEIGHT_COLUMN = "n_propio"
LEVEL_MARK = "_level_"          # the naming of a hierarchy: product_level_1 ⊂ product_level_2
FIGURE_DECIMALS = 4


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STATISTICS (weighted by the support of each series)
# ═══════════════════════════════════════════════════════════════════════════════════

def weighted_eta2(frame: pd.DataFrame, group_column: str) -> float:
    """η²: weighted sum of squares BETWEEN the values of the column / total. 0 when the
    rate does not vary at all."""
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    if total <= 0:
        return 0.0
    grouped = frame.assign(_weighted_value=weights * values).groupby(group_column, observed=True)
    group_weights = grouped[WEIGHT_COLUMN].sum().to_numpy(dtype=float)
    group_means = grouped["_weighted_value"].sum().to_numpy(dtype=float) / group_weights
    between = float(np.sum(group_weights * (group_means - grand_mean) ** 2))
    return between / total


def weighted_omega2(frame: pd.DataFrame, group_column: str) -> float:
    """ω² = (SSB − (k − 1) · MSW) / (SST + MSW): η² corrected for the number of values k.
    0 when there are not more series than values."""
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    series_count = len(frame)
    value_count = frame[group_column].nunique()
    if total <= 0 or series_count <= value_count:
        return 0.0
    between = weighted_eta2(frame, group_column) * total
    mean_square_within = (total - between) / (series_count - value_count)
    return float(max(0.0, (between - (value_count - 1) * mean_square_within) / (total + mean_square_within)))


def weighted_r2(frame: pd.DataFrame, dimensions: list) -> float:
    """R² of a weighted least-squares fit of the rate on the dimensions as categories
    (additive model). A design with redundant columns is solved by lstsq."""
    if not dimensions or len(frame) < MIN_SERIES_IN_BASE:
        return 0.0
    design = pd.get_dummies(frame[dimensions].astype(str), drop_first=True).astype(float)
    design.insert(0, "intercept", 1.0)
    root_weights = np.sqrt(frame[WEIGHT_COLUMN].to_numpy(dtype=float))
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design.to_numpy() * root_weights[:, None], values * root_weights, rcond=None)[0]
    fitted = design.to_numpy() @ coefficients
    grand_mean = np.average(values, weights=root_weights ** 2)
    total = float(np.sum(root_weights ** 2 * (values - grand_mean) ** 2))
    residual = float(np.sum(root_weights ** 2 * (values - fitted) ** 2))
    return float(max(0.0, 1 - residual / total)) if total > 0 else 0.0


def family_and_level(dimension_name: str) -> tuple:
    """("product", 2) for "product_level_2"; (name, 1) for a dimension outside a hierarchy."""
    if LEVEL_MARK in dimension_name:
        family, _, level_text = dimension_name.rpartition(LEVEL_MARK)
        if level_text.isdigit():
            return family, int(level_text)
    return dimension_name, 1


def sequential_collapse_order(base: pd.DataFrame, mandatory_dims: list) -> list:
    """The collapse order of the mandatory dims, greedy and sequential: at every round, among
    the droppable dims (a *_level_N only when no *_level_(N+1) remains), drop the one whose
    removal from the dims that REMAIN loses the least R². Returns [(dim, loss), ...] from
    the first to collapse to the last; ties go to the name, so the order is reproducible."""
    remaining = list(mandatory_dims)
    collapse_order = []
    current_r2 = weighted_r2(base, remaining)
    while remaining:
        droppable = [dimension for dimension in remaining
                     if not any(family_and_level(other) == (family_and_level(dimension)[0], family_and_level(dimension)[1] + 1)
                                for other in remaining)]
        losses = {}
        for dimension in droppable:
            r2_without = weighted_r2(base, [other for other in remaining if other != dimension])
            losses[dimension] = max(0.0, current_r2 - r2_without)
        next_dimension = min(droppable, key=lambda dimension: (losses[dimension], dimension))
        collapse_order.append((next_dimension, round(losses[next_dimension], FIGURE_DECIMALS)))
        remaining.remove(next_dimension)
        current_r2 = weighted_r2(base, remaining)
    return collapse_order


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def analyse_dimensions(series_rate: pd.DataFrame, series_lookup: pd.DataFrame, configuration: Config) -> tuple:
    """η², unique contribution, ω² and the collapse order; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    mandatory_dims = configuration.business_mandatory_dims
    dimensions = mandatory_dims + configuration.extra_renovacion

    # [1] the base: series with an informative own rate, and their dimension values
    base = series_rate[(series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE) & series_rate[VALUE_COLUMN].notna()
                       & (series_rate[SIGN_COLUMN] == SIGN_NEUTRAL) & (series_rate[WEIGHT_COLUMN] > 0)]
    base = base.merge(series_lookup[[SERIES_ID_COLUMN] + dimensions], on=SERIES_ID_COLUMN)
    predictable_usd = series_rate.loc[series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE, "usd_por_predecir"].sum()
    configuration.log_action(STEP_LABEL, 1, f"base: {len(base):,} series (predecible, own rate, neutro), "
                                            f"{base['usd_por_predecir'].sum() / predictable_usd if predictable_usd else 0:.0%} "
                                            f"of the money of the predictable series")
    has_evidence = len(base) >= MIN_SERIES_IN_BASE

    # [2] the three figures of every dimension
    r2_all = weighted_r2(base, dimensions)
    dimension_rows = []
    for dimension in dimensions:
        dimension_rows.append({
            "dimension": dimension,
            "grupo": "mandatory" if dimension in mandatory_dims else "extra_renovacion",
            "valores": int(base[dimension].nunique()) if has_evidence else 0,
            "eta2_individual": round(weighted_eta2(base, dimension), FIGURE_DECIMALS) if has_evidence else 0.0,
            "contribucion_unica": round(max(0.0, r2_all - weighted_r2(base, [other for other in dimensions if other != dimension])),
                                        FIGURE_DECIMALS) if has_evidence else 0.0,
            "omega2": round(weighted_omega2(base, dimension), FIGURE_DECIMALS) if has_evidence else 0.0,
            "anulable": int(dimension in configuration.extra_renovacion)})
    decision = pd.DataFrame(dimension_rows)
    configuration.log_action(STEP_LABEL, 2, f"{len(dimensions)} dimensions measured; R² of all of them together "
                                            f"{r2_all:.3f}")

    # [3] the collapse order of the mandatory dimensions
    collapse_order = sequential_collapse_order(base, mandatory_dims)
    position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
    loss_of = dict(collapse_order)
    decision["orden_colapso"] = decision["dimension"].map(position_of).fillna(0).astype(int)
    decision["perdida_secuencial"] = decision["dimension"].map(loss_of)
    configuration.log_action(STEP_LABEL, 3, "collapse order: " + " → ".join(dimension for dimension, _ in collapse_order))

    # [4] the pairs and their interaction
    pairs = dimension_pairs(base, decision, dimensions, has_evidence, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{len(list(itertools.combinations(dimensions, 2)))} pairs measured; "
                                            f"the top {len(pairs)} by interaction are kept")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the figures and the order")
    check_dimensions(base, decision, collapse_order, configuration, check_log)

    # [6] the two tables, written
    configuration.log_action(STEP_LABEL, 6, "writing the two decision tables")
    configuration.write_table(STEP_LABEL, check_log, decision, TABLE_DIMENSIONS)
    configuration.write_table(STEP_LABEL, check_log, pairs, TABLE_DIMENSION_PAIRS)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] the dimensions in collapse order, and the top pairs
    configuration.log_action(STEP_LABEL, 8, "the dimensions (mandatory in collapse order: 1 goes first; extras are "
                                            "annulled before any mandatory):")
    shown = decision.assign(_order=decision["orden_colapso"].where(decision["orden_colapso"] > 0, 0))
    configuration.show_table(shown.sort_values(["anulable", "_order"], ascending=[False, True]).drop(columns="_order"))
    if len(pairs):
        configuration.logger.doc(f"[{STEP_LABEL}] the pairs that separate more together than apart:")
        configuration.show_table(pairs.head(10))
    return decision, pairs


def dimension_pairs(base: pd.DataFrame, decision: pd.DataFrame, dimensions: list, has_evidence: bool,
                    configuration: Config) -> pd.DataFrame:
    """η² of every pair of dimensions and its interaction (η² of the pair − the larger of the
    two individual η²); the top ones by interaction."""
    pair_columns = ["par", "eta2_par", "interaccion"]
    if not has_evidence:
        return pd.DataFrame(columns=pair_columns)
    individual = decision.set_index("dimension")["eta2_individual"]
    pair_rows = []
    for first_dimension, second_dimension in itertools.combinations(dimensions, 2):
        joint = base.assign(_pair=base[first_dimension].astype(str) + "|" + base[second_dimension].astype(str))
        pair_eta2 = weighted_eta2(joint, "_pair")
        pair_rows.append({"par": f"{first_dimension}×{second_dimension}",
                          "eta2_par": round(pair_eta2, FIGURE_DECIMALS),
                          "interaccion": round(pair_eta2 - max(individual[first_dimension], individual[second_dimension]),
                                               FIGURE_DECIMALS)})
    pairs = pd.DataFrame(pair_rows, columns=pair_columns)
    return (pairs.sort_values("interaccion", ascending=False).head(configuration.dimension_pairs_shown)
            .reset_index(drop=True))


def check_dimensions(base: pd.DataFrame, decision: pd.DataFrame, collapse_order: list, configuration: Config,
                     check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] enough series to measure anything
    configuration.log_check(STEP_LABEL, check_log, f"the base has at least {MIN_SERIES_IN_BASE} series",
                            len(base) >= MIN_SERIES_IN_BASE,
                            failure_detail=f"only {len(base)} series: every figure is 0 and the collapse order is by name",
                            context=f"{len(base):,} series", blocking=False)

    # [2] the figures are proportions
    figure_columns = ["eta2_individual", "contribucion_unica", "omega2"]
    out_of_range = decision[((decision[figure_columns] < 0) | (decision[figure_columns] > 1)).any(axis=1)]
    configuration.log_check(STEP_LABEL, check_log, "every figure is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range)} dimensions with a figure outside [0, 1]",
                            examples=out_of_range)

    # [3] one position per mandatory dimension, 1 to M
    mandatory_dims = configuration.business_mandatory_dims
    positions = sorted(decision.loc[decision["grupo"] == "mandatory", "orden_colapso"])
    configuration.log_check(STEP_LABEL, check_log, "every mandatory dimension has one collapse position, 1 to M",
                            positions == list(range(1, len(mandatory_dims) + 1)),
                            failure_detail=f"positions found: {positions}",
                            context=f"{len(mandatory_dims)} mandatory dimensions")

    # [4] the hierarchy: a coarser level never goes before a finer one of its family
    position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
    broken = []
    for dimension in mandatory_dims:
        family, level = family_and_level(dimension)
        finer = [other for other in mandatory_dims if family_and_level(other) == (family, level + 1)]
        for finer_dimension in finer:
            if position_of[dimension] < position_of[finer_dimension]:
                broken.append(f"{dimension} before {finer_dimension}")
    configuration.log_check(STEP_LABEL, check_log, "a *_level_N never collapses before its *_level_(N+1)", not broken,
                            failure_detail=f"hierarchy broken: {broken}")
