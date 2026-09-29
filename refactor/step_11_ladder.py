"""
step_11_ladder.py — Every series climbs its ladder, and gets its estimated rate, its two
errors and its risk level.

"The series borrows support from the nearest relative that has enough, and trusts it in
proportion to how little it has itself."

THE CLIMB (per estimable series, over the rungs of step 10):
  · rung 0 (itself) is chosen when its own support reaches own_rate_floor (271): it is
    precise enough to predict alone;
  · otherwise, the first relative (rung ≥ 1) whose pool reaches support_floor (30);
  · if none does: itself when its own support reaches support_floor (own rate, nobody to
    reinforce it), else the LAST rung (the best available), marked as not reaching the floor.
THE CREDIBILITY (Bühlmann-Straub, per chosen relative): k = within variance / between
variance of the rates of the series that chose it (with fewer than 3 of them, k_cred).
A homogeneous pool gets a large k (its series take the pool's rate); a heterogeneous one a
small k (its series keep their own).
THE RATE: tasa = z · own + (1 − z) · relative, z = n_propio / (n_propio + k), when the series
reached the floor on a relative (rung ≥ 1) and has an own rate; otherwise the best rate
found alone (z = 1). Two errors, in pp:
  · se_estimacion: the error of the ESTIMATE, √(z²·se_own² + (1−z)²·se_relative²)
  · se_prediccion: √(se_estimacion² + se_binomial(rate, n_propio)²): next month still
    samples with the series' OWN size. It never shrinks below its own binomial noise.
THE RISK LEVEL: how the rate was obtained (A propio … D sin historia, N sin impacto), with
the money each level carries.

Series that are not estimable (solo_historia, solo_futuro, time_series, mixto) get their
row too: rung 0, no relative, their own rate if they have one, and their level.

Actions (logged as they are done):
  1. the ladder of every estimable series with the support and rate of each rung
  2. the climb: the chosen relative of every series
  3. the credibility k of every chosen relative
  4. the rate and the two errors of every series (the others: their own rate, if any)
  5. the risk level of every series
  6. check the decisions, the rates and the errors                  checks 1-4
  7. write the estimate of every series and the money by level      checks 5-6
  8. count the checks; stop if any failed
  9. show the money by risk level and what each level means, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every series has exactly one decision
   2. the choice follows the rule (a series marked as reaching the floor stands on ≥ support_floor)
   3. every estimated rate and every z is between 0 and 1
   4. the prediction error is never below the estimation error
   5-6. tables sff_series_estimacion and sff_niveles_riesgo written and read back

Output: one row per series with its chosen relative, k, z, estimated rate, two errors and
risk level · the money by risk level · tables sff_series_estimacion, sff_niveles_riesgo.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (ESTIMATION_ID_COLUMN, LEVEL_BORROWED, LEVEL_FAR, LEVEL_MIXED, LEVEL_NO_HISTORY,
                         LEVEL_NO_IMPACT, LEVEL_OWN, LEVEL_OWN_REINFORCED, LEVEL_OWN_SHORT, LEVEL_SIGNED_UNDER_FLOOR,
                         LEVEL_TIME_SERIES, PATTERN_COLUMN, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY,
                         RUNG_COLUMN, SERIES_ID_COLUMN, SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, TABLE_RISK_LEVELS,
                         TABLE_SERIES_ESTIMATE, UNIVERSE_COLUMN, UNIVERSE_NORMAL)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "11"
STEP_NAME = "LADDER"
STEP_PURPOSE = ("let every series climb to the nearest relative with enough support, blend its own rate with the "
                "relative's in proportion to its own support (Bühlmann credibility), and give it its rate, the error "
                "of that estimate, the error of next month's prediction and its risk level")
STEP_ACTIONS = ["the ladder of every estimable series with the support and rate of each rung",
                "the climb: the chosen relative of every series",
                "the credibility k of every chosen relative",
                "the rate and the two errors of every series (the others: their own rate, if any)",
                "the risk level of every series",
                "check the decisions, the rates and the errors (checks 1-4)",
                "write the estimate of every series and the money by level (checks 5-6)",
                "count the checks; stop if any failed",
                "show the money by risk level and what each level means, as tables"]
STEP_OUTPUT = ("one row per series: chosen relative, k, z, estimated rate, se_estimacion, se_prediccion, risk level · "
               "the money by level · tables sff_series_estimacion, sff_niveles_riesgo")

# ─── named constants ─────────────────────────────────────────────────────────────
PERCENTAGE_POINTS = 100
MIN_SIBLINGS_FOR_K = 3              # below it, the k of the Config
HOMOGENEOUS_POOL_FACTOR = 10        # a pool with no between variance gets k = 10 × k_cred: its series take its rate
MIN_BETWEEN_VARIANCE = 1e-6
ROUNDING_TOLERANCE = 1e-9

# What each risk level means, for the table of action 9 (the rule, then its meaning).
LEVEL_DEFINITIONS = [
    (LEVEL_OWN, "n ≥ own_rate_floor and ≥ 12 months", "own precision (±5 pp or better): predicts alone, z = 1"),
    (LEVEL_OWN_SHORT, "n ≥ own_rate_floor and < 12 months", "own precision but a short history: it has not seen every season"),
    (LEVEL_OWN_REINFORCED, "support_floor ≤ n < own_rate_floor", "own evidence without precision: blended with its first relative with support"),
    (LEVEL_BORROWED, "rung 1-2 reaches the floor", "below the floor; borrows from a relative that shares EVERY mandatory dim"),
    (LEVEL_FAR, "rung ≥ 3 reaches the floor (or none does)", "below the floor; the relative is the mandatory cell or coarser"),
    (LEVEL_SIGNED_UNDER_FLOOR, "signed and no rung reaches the floor", "the best rate of ITS sign, noisy: it may not mix with unsigned series"),
    (LEVEL_MIXED, "flags of both signs", "never pooled: keeps its own rate; should be ≈ 0"),
    (LEVEL_NO_HISTORY, "solo_futuro", "no rate to estimate: it will take a rate from its cell later"),
    (LEVEL_NO_IMPACT, "solo_historia", "nothing to predict: kept for the pools and the backtest"),
    (LEVEL_TIME_SERIES, "time_series universe", "labelled only: treated apart"),
]


def binomial_se_pp(rates, supports) -> np.ndarray:
    """The binomial standard error of a rate, in pp: 100 · √(p(1 − p) / n), n at least 1."""
    rates = np.asarray(rates, dtype=float)
    supports = np.maximum(np.asarray(supports, dtype=float), 1.0)
    return PERCENTAGE_POINTS * np.sqrt(rates * (1 - rates) / supports)


def climb_the_ladder(series_rate: pd.DataFrame, relatives: pd.DataFrame, pools: pd.DataFrame,
                     configuration: Config) -> tuple:
    """The chosen relative, k, rate, errors and risk level of every series; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the ladder: every rung with its pool
    ladder = relatives.merge(pools[[PATTERN_COLUMN, "n_pool", "tasa_pool"]], on=PATTERN_COLUMN, how="left")
    ladder["n_pool"] = ladder["n_pool"].fillna(0.0)
    configuration.log_action(STEP_LABEL, 1, f"{len(ladder):,} rungs of {ladder[SERIES_ID_COLUMN].nunique():,} estimable series")

    # [2] the climb
    decisions = choose_relatives(ladder, configuration)
    reached_share = decisions["alcanzo_suelo"].mean() if len(decisions) else 0.0
    configuration.log_action(STEP_LABEL, 2, f"{reached_share:.0%} of the estimable series reached the floor; chosen rungs "
                                            f"{decisions[RUNG_COLUMN].value_counts().sort_index().to_dict()}")

    # [3] the credibility of every chosen relative
    k_by_relative = credibility_k(decisions, series_rate, configuration)
    decisions["k"] = decisions[ESTIMATION_ID_COLUMN].map(k_by_relative).fillna(configuration.k_cred)
    configuration.log_action(STEP_LABEL, 3, f"k estimated for {int((k_by_relative != configuration.k_cred).sum()):,} "
                                            f"relatives; the others use k_cred = {configuration.k_cred:.0f}")

    # [4] the rate and the two errors of every series
    series_estimate = estimate_rates(series_rate, decisions, configuration)
    configuration.log_action(STEP_LABEL, 4, f"rate estimated for {int(series_estimate['tasa_estimada'].notna().sum()):,} "
                                            f"of {len(series_estimate):,} series")

    # [5] the risk level
    series_estimate["nivel_riesgo"] = risk_levels(series_estimate, configuration)
    configuration.log_action(STEP_LABEL, 5, f"levels: {series_estimate['nivel_riesgo'].value_counts().sort_index().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the decisions, the rates and the errors")
    check_ladder(series_estimate, series_rate, configuration, check_log)

    # [7] the tables, written
    configuration.log_action(STEP_LABEL, 7, "writing the estimate of every series and the money by level")
    money_by_level = risk_levels_report(series_estimate)
    configuration.write_table(STEP_LABEL, check_log, series_estimate, TABLE_SERIES_ESTIMATE)
    configuration.write_table(STEP_LABEL, check_log, money_by_level, TABLE_RISK_LEVELS)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] the money by level and what each level means
    configuration.log_action(STEP_LABEL, 9, "money to predict by risk level (error_pp: next month's prediction error "
                                            "of the level, weighted by money):")
    configuration.show_table(money_by_level)
    configuration.logger.doc(f"[{STEP_LABEL}] what each level means:")
    configuration.show_table(pd.DataFrame(LEVEL_DEFINITIONS, columns=["nivel", "regla", "significado"]))
    return series_estimate, money_by_level


def choose_relatives(ladder: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The chosen rung of every series, vectorised: own (rung 0) if it reaches own_rate_floor;
    else the first rung ≥ 1 reaching support_floor; else own if it reaches support_floor;
    else the last rung, marked as not reaching the floor."""
    own = ladder[ladder[RUNG_COLUMN] == 0].set_index(SERIES_ID_COLUMN)
    last = ladder.sort_values(RUNG_COLUMN).groupby(SERIES_ID_COLUMN).tail(1).set_index(SERIES_ID_COLUMN)
    first_with_support = (ladder[(ladder[RUNG_COLUMN] > 0) & (ladder["n_pool"] >= configuration.support_floor)]
                          .sort_values(RUNG_COLUMN).groupby(SERIES_ID_COLUMN).head(1).set_index(SERIES_ID_COLUMN))

    own_is_precise = own["n_pool"] >= configuration.own_rate_floor
    own_has_evidence = own["n_pool"] >= configuration.support_floor
    has_relative = own.index.isin(first_with_support.index)

    chosen = last.copy()                                                  # nobody reaches the floor
    chosen.loc[own_has_evidence] = own.loc[own_has_evidence]              # own, with nobody to reinforce it
    relative_ids = own.index[has_relative & ~own_is_precise]
    chosen.loc[relative_ids] = first_with_support.loc[relative_ids]       # the first relative with support
    chosen.loc[own_is_precise] = own.loc[own_is_precise]                  # precise: alone
    chosen["alcanzo_suelo"] = (own_is_precise | has_relative | own_has_evidence).astype(int)

    decisions = chosen.reset_index().rename(columns={PATTERN_COLUMN: ESTIMATION_ID_COLUMN, "n_pool": "n_efectivo",
                                                     "tasa_pool": "tasa_pariente", "descripcion": "descripcion_peldano"})
    return decisions[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, RUNG_COLUMN, "descripcion_peldano", "n_efectivo",
                      "tasa_pariente", "alcanzo_suelo"]]


def credibility_k(decisions: pd.DataFrame, series_rate: pd.DataFrame, configuration: Config) -> pd.Series:
    """Bühlmann-Straub k per chosen relative from the series that chose it:
    within = mean of p(1 − p); between = weighted variance of p minus its sampling part;
    k = within / between. Fewer than 3 siblings with a rate: k_cred. No between variance:
    10 × k_cred (a homogeneous pool: its series take its rate)."""
    siblings = decisions.merge(series_rate[[SERIES_ID_COLUMN, "tasa_propia", "vencen"]], on=SERIES_ID_COLUMN)
    siblings = siblings[siblings["tasa_propia"].notna() & (siblings["vencen"] > 0)].copy()
    rates, weights = siblings["tasa_propia"], siblings["vencen"]
    siblings["_w"] = weights
    siblings["_wp"] = weights * rates
    grouped = siblings.groupby(ESTIMATION_ID_COLUMN)
    pooled_rate = grouped["_wp"].sum() / grouped["_w"].sum()
    siblings["_pooled"] = siblings[ESTIMATION_ID_COLUMN].map(pooled_rate)
    siblings["_within"] = rates * (1 - rates)
    siblings["_w_sq_dev"] = weights * (rates - siblings["_pooled"]) ** 2
    siblings["_w_sampling"] = weights * rates * (1 - rates) / np.maximum(weights, 1)
    grouped = siblings.groupby(ESTIMATION_ID_COLUMN)
    count = grouped.size()
    within = grouped["_within"].mean()
    between = (grouped["_w_sq_dev"].sum() - grouped["_w_sampling"].sum()) / grouped["_w"].sum()
    k = pd.Series(np.where(between > MIN_BETWEEN_VARIANCE, within / between.where(between > MIN_BETWEEN_VARIANCE),
                           HOMOGENEOUS_POOL_FACTOR * configuration.k_cred), index=count.index)
    k[count < MIN_SIBLINGS_FOR_K] = configuration.k_cred
    return k


def estimate_rates(series_rate: pd.DataFrame, decisions: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The rate and the two errors of every series (estimable or not)."""
    estimate = series_rate.merge(decisions, on=SERIES_ID_COLUMN, how="left")
    not_estimable = estimate[ESTIMATION_ID_COLUMN].isna()
    estimate.loc[not_estimable, ESTIMATION_ID_COLUMN] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
    estimate.loc[not_estimable, RUNG_COLUMN] = 0
    estimate.loc[not_estimable, "n_efectivo"] = estimate.loc[not_estimable, "n_propio"]
    estimate.loc[not_estimable, "alcanzo_suelo"] = 0
    estimate["k"] = estimate["k"].fillna(configuration.k_cred)
    estimate[RUNG_COLUMN] = estimate[RUNG_COLUMN].astype(int)
    estimate["alcanzo_suelo"] = estimate["alcanzo_suelo"].astype(int)

    own_rate, own_n = estimate["tasa_propia"], estimate["n_propio"]
    relative_rate, relative_n = estimate["tasa_pariente"], estimate["n_efectivo"]
    blends = ((estimate["alcanzo_suelo"] == 1) & (estimate[RUNG_COLUMN] > 0)
              & relative_rate.notna() & own_rate.notna())

    # blended series: credibility
    z = np.where(blends, own_n / (own_n + estimate["k"]), 1.0)
    blended_rate = z * own_rate.fillna(0) + (1 - z) * relative_rate.fillna(0)
    blended_se = np.sqrt((z * binomial_se_pp(own_rate.fillna(0.5), own_n)) ** 2
                         + ((1 - z) * binomial_se_pp(relative_rate.fillna(0.5), relative_n)) ** 2)

    # the others: their best rate found alone (own at rung 0, the relative's above it)
    alone_rate = np.where(estimate[RUNG_COLUMN] == 0, own_rate, relative_rate.fillna(own_rate))
    alone_n = np.where(estimate[RUNG_COLUMN] == 0, own_n, relative_n)
    alone_se = binomial_se_pp(np.nan_to_num(alone_rate, nan=0.5), alone_n)

    estimate["z"] = np.round(z, 3)
    estimate["tasa_estimada"] = np.where(blends, blended_rate, alone_rate)
    estimate["se_estimacion_pp"] = np.where(blends, blended_se, np.where(pd.notna(alone_rate), alone_se, np.nan))
    own_noise = binomial_se_pp(np.nan_to_num(estimate["tasa_estimada"], nan=0.5), np.maximum(own_n, 1))
    estimate["se_prediccion_pp"] = np.where(estimate["tasa_estimada"].notna(),
                                            np.sqrt(estimate["se_estimacion_pp"] ** 2 + own_noise ** 2), np.nan)
    return estimate


def risk_levels(estimate: pd.DataFrame, configuration: Config) -> np.ndarray:
    """The risk level of every series: route and universe first, then sign, then whether
    the floor was reached, then the rung. The first condition that holds wins."""
    conditions = [
        estimate[ROUTE_COLUMN] == ROUTE_HISTORY_ONLY,
        estimate[UNIVERSE_COLUMN] != UNIVERSE_NORMAL,
        (estimate[ROUTE_COLUMN] == ROUTE_FUTURE_ONLY) | (estimate["meses_historia"] == 0),
        estimate[SIGN_COLUMN] == SIGN_MIXED,
        (estimate["alcanzo_suelo"] == 0) & (estimate[SIGN_COLUMN] != SIGN_NEUTRAL),
        estimate["alcanzo_suelo"] == 0,
        (estimate["n_propio"] >= configuration.support_floor) & (estimate["n_propio"] < configuration.own_rate_floor),
        (estimate[RUNG_COLUMN] == 0) & (estimate["meses_historia"] >= configuration.own_level_min_history_months),
        estimate[RUNG_COLUMN] == 0,
        estimate[RUNG_COLUMN] <= configuration.close_relative_max_rung]
    levels = [LEVEL_NO_IMPACT, LEVEL_TIME_SERIES, LEVEL_NO_HISTORY, LEVEL_MIXED, LEVEL_SIGNED_UNDER_FLOOR, LEVEL_FAR,
              LEVEL_OWN_REINFORCED, LEVEL_OWN, LEVEL_OWN_SHORT, LEVEL_BORROWED]
    return np.select(conditions, levels, default=LEVEL_FAR)


def risk_levels_report(estimate: pd.DataFrame) -> pd.DataFrame:
    """Money to predict by risk level, with the level's prediction error weighted by money."""
    total_usd = estimate["usd_por_predecir"].sum()
    report_rows = []
    for level, level_series in estimate.groupby("nivel_riesgo"):
        with_error = level_series[level_series["se_prediccion_pp"].notna() & (level_series["usd_por_predecir"] > 0)]
        error_pp = (np.average(with_error["se_prediccion_pp"], weights=with_error["usd_por_predecir"])
                    if len(with_error) else np.nan)
        report_rows.append({"nivel_riesgo": level, "series": len(level_series),
                            "usd_por_predecir": level_series["usd_por_predecir"].sum(),
                            "pct_usd": level_series["usd_por_predecir"].sum() / total_usd if total_usd else 0.0,
                            "error_pp": error_pp})
    return pd.DataFrame(report_rows).sort_values("nivel_riesgo").reset_index(drop=True)


def check_ladder(estimate: pd.DataFrame, series_rate: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] one decision per series
    configuration.log_check(STEP_LABEL, check_log, "every series has exactly one decision",
                            len(estimate) == len(series_rate) and estimate[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(estimate):,} decisions for {len(series_rate):,} series",
                            context=f"{len(estimate):,} series")

    # [2] the rule: a series marked as reaching the floor stands on enough support
    marked = estimate[estimate["alcanzo_suelo"] == 1]
    breaking = marked[marked["n_efectivo"] < configuration.support_floor - ROUNDING_TOLERANCE]
    configuration.log_check(STEP_LABEL, check_log, "the choice follows the rule (reaching the floor = standing on ≥ support_floor)",
                            breaking.empty,
                            failure_detail=f"{len(breaking):,} series marked as reaching the floor below it",
                            examples=breaking[[SERIES_ID_COLUMN, RUNG_COLUMN, "n_efectivo"]])

    # [3] proportions
    out_of_range = estimate[(estimate["tasa_estimada"] < 0) | (estimate["tasa_estimada"] > 1)
                            | (estimate["z"] < 0) | (estimate["z"] > 1)]
    configuration.log_check(STEP_LABEL, check_log, "every estimated rate and every z is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} series out of [0, 1]",
                            examples=out_of_range[[SERIES_ID_COLUMN, "tasa_estimada", "z"]])

    # [4] the prediction never sheds the series' own noise
    shrinking = estimate[estimate["se_prediccion_pp"] < estimate["se_estimacion_pp"] - ROUNDING_TOLERANCE]
    configuration.log_check(STEP_LABEL, check_log, "the prediction error is never below the estimation error",
                            shrinking.empty,
                            failure_detail=f"{len(shrinking):,} series whose prediction error is below the estimate's")
