"""
step_11_ladder.py — Every series takes the rate of its final group (step 10), reinforced by
credibility with the group's reference; and gets its two errors and its risk level.

"The group lends its rate to its series; a group without precision trusts its own rate in
proportion to how much support it has, and the rest comes from a wider reference."

THE RATE OF THE GROUP: Σ renewed / Σ due of all the series of the final group (step 10).
THE CREDIBILITY (only for a final group below own_rate_floor, 271, with a reference):
  z = n / (n + k), n = the support of the group, k = Bühlmann-Straub k of its reference,
  estimated from the final groups that share that reference: within variance / between
  variance of their rates (fewer than 3 groups: k_cred; no between variance: 10 × k_cred).
  estimated rate = z · rate of the group + (1 − z) · rate of the reference
  A group at or above own_rate_floor predicts alone: z = 1.
Every series of a group gets the same estimated rate (the group lends it). Two errors, in pp:
  · se_estimacion: the error of the ESTIMATE, √(z²·se_group² + (1−z)²·se_reference²)
  · se_prediccion: √(se_estimacion² + se_binomial(rate, n_propio)²): next month still samples
    with the series' OWN size. It never shrinks below its own binomial noise.
THE RISK LEVEL: how the rate was obtained, by the pass where the group of the series was formed:
  A  the series itself (pass 0) with own precision (≥ own_rate_floor), a year of history
  A2 the same, less than a year
  A3 the series itself with evidence (≥ support_floor) but not precision: credibility
  B  merged in the sign or extra passes: its group shares EVERY mandatory dim with it
  C  merged in a mandatory pass (a mandatory dim collapsed), or never reached the floor
  S  signed and its group never reached the floor
  M  mixed signs: never merged · D solo_futuro · N solo_historia · T time_series universe

Series that are not estimable (solo_historia, solo_futuro, time_series, mixto) get their row
too: their own group, their own rate if they have one, and their level.

Actions (logged as they are done):
  1. the final group and the reference of every estimable series (step 10)
  2. the credibility k of every reference, from the groups that share it
  3. the rate and the two errors of every series (the others: their own rate, if any)
  4. the risk level of every series
  5. check the rates and the errors                                   checks 1-4
  6. write the estimate of every series and the money by level        checks 5-6
  7. count the checks; stop if any failed
  8. show the money by risk level and what each level means, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every series has exactly one estimate
   2. every series of a group has the same estimated rate
   3. every estimated rate and every z is between 0 and 1
   4. the prediction error is never below the estimation error
   5-6. tables sff_series_estimacion and sff_niveles_riesgo written and read back

Output: one row per series with its final group, reference, k, z, estimated rate, two errors
and risk level · the money by risk level · tables sff_series_estimacion, sff_niveles_riesgo.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import (ESTIMATION_ID_COLUMN, LEVEL_BORROWED, LEVEL_FAR, LEVEL_MIXED, LEVEL_NO_HISTORY,
                         LEVEL_NO_IMPACT, LEVEL_OWN, LEVEL_OWN_REINFORCED, LEVEL_OWN_SHORT, LEVEL_SIGNED_UNDER_FLOOR,
                         LEVEL_TIME_SERIES, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, SERIES_ID_COLUMN,
                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, TABLE_RISK_LEVELS, TABLE_SERIES_ESTIMATE,
                         UNIVERSE_COLUMN, UNIVERSE_NORMAL)


STEP_LABEL = "11"
STEP_NAME = "LADDER"
STEP_PURPOSE = ("give every series the rate of its final group, reinforced by credibility with the group's reference "
                "when the group has no precision of its own; the error of that estimate, the error of next month's "
                "prediction, and the risk level of how the rate was obtained")
STEP_ACTIONS = ["the final group and the reference of every estimable series (step 10)",
                "the credibility k of every reference, from the groups that share it",
                "the rate and the two errors of every series (the others: their own rate, if any)",
                "the risk level of every series",
                "check the rates and the errors (checks 1-4)",
                "write the estimate of every series and the money by level (checks 5-6)",
                "count the checks; stop if any failed",
                "show the money by risk level and what each level means, as tables"]
STEP_OUTPUT = ("one row per series: final group, reference, k, z, estimated rate, se_estimacion, se_prediccion, risk level · "
               "the money by level · tables sff_series_estimacion, sff_niveles_riesgo")

PERCENTAGE_POINTS = 100
MIN_SIBLINGS_FOR_K = 3              # below it, the k of the Config
HOMOGENEOUS_POOL_FACTOR = 10        # a reference with no between variance gets k = 10 × k_cred: its groups take its rate
MIN_BETWEEN_VARIANCE = 1e-6
ROUNDING_TOLERANCE = 1e-9

# What each risk level means, for the table of action 8 (the rule, then its meaning).
LEVEL_DEFINITIONS = [
    (LEVEL_OWN, "itself, n ≥ own_rate_floor, ≥ 12 months", "own precision: predicts alone, z = 1"),
    (LEVEL_OWN_SHORT, "itself, n ≥ own_rate_floor, < 12 months", "own precision but a short history: it has not seen every season"),
    (LEVEL_OWN_REINFORCED, "itself, support_floor ≤ n < own_rate_floor", "own evidence without precision: credibility with its reference"),
    (LEVEL_BORROWED, "merged in the sign or extra passes", "its group shares EVERY mandatory dim with it"),
    (LEVEL_FAR, "merged in a mandatory pass, or never reached the floor", "its group collapsed a mandatory dim"),
    (LEVEL_SIGNED_UNDER_FLOOR, "signed, its group never reached the floor", "the best rate of ITS sign, noisy: it may not mix with unsigned series"),
    (LEVEL_MIXED, "flags of both signs", "never merged: keeps its own rate; should be ≈ 0"),
    (LEVEL_NO_HISTORY, "solo_futuro", "no rate to estimate: it will take a rate from its cell later"),
    (LEVEL_NO_IMPACT, "solo_historia", "nothing to predict: kept for the groups and the backtest"),
    (LEVEL_TIME_SERIES, "time_series universe", "labelled only: treated apart"),
]


def binomial_se_pp(rates, supports) -> np.ndarray:
    """The binomial standard error of a rate, in pp: 100 · √(p(1 − p) / n), n at least 1."""
    rates = np.asarray(rates, dtype=float)
    supports = np.maximum(np.asarray(supports, dtype=float), 1.0)
    return PERCENTAGE_POINTS * np.sqrt(rates * (1 - rates) / supports)


def climb_the_ladder(series_rate: pd.DataFrame, ladder: dict, configuration: Config) -> tuple:
    """The rate of every series from its final group and its reference; the errors and the risk level.

    INPUT:   series_rate (step 08) · the ladder of step 10 (groups, summary) · the Config.
    OUTPUT:  (series_estimate: one row per series, money_by_level).
    RULES:   see the module header.
    EDGE CASES: a group with no reference, or at or above own_rate_floor, has z = 1. A series that
             is not estimable is its own group with its own rate.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the final group and the reference of every estimable series
    groups = ladder["groups"]
    step_names = dict(zip(ladder["summary"]["ladder_step"], ladder["summary"]["step_name"]))
    configuration.log_action(STEP_LABEL, 1, f"{len(groups):,} estimable series in {groups['final_group_id'].nunique():,} "
                                            f"final groups; {int(groups['credibility_ref_id'].notna().sum()):,} of them "
                                            f"with a credibility reference")

    # [2] the credibility k of every reference
    k_by_reference = credibility_k(groups, configuration)
    configuration.log_action(STEP_LABEL, 2, f"k estimated for {int((k_by_reference != configuration.k_cred).sum()):,} "
                                            f"references; the others use k_cred = {configuration.k_cred:.0f}")

    # [3] the rate and the two errors of every series
    series_estimate = estimate_rates(series_rate, groups, k_by_reference, configuration)
    configuration.log_action(STEP_LABEL, 3, f"rate estimated for {int(series_estimate['tasa_estimada'].notna().sum()):,} "
                                            f"of {len(series_estimate):,} series · z < 1 in "
                                            f"{int((series_estimate['z'] < 1).sum()):,}")

    # [4] the risk level
    series_estimate["nivel_riesgo"] = risk_levels(series_estimate, step_names, configuration)
    configuration.log_action(STEP_LABEL, 4, f"levels: {series_estimate['nivel_riesgo'].value_counts().sort_index().to_dict()}")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the rates and the errors")
    check_ladder(series_estimate, series_rate, configuration, check_log)

    # [6] the tables, written
    configuration.log_action(STEP_LABEL, 6, "writing the estimate of every series and the money by level")
    money_by_level = risk_levels_report(series_estimate)
    configuration.write_table(STEP_LABEL, check_log, series_estimate, TABLE_SERIES_ESTIMATE)
    configuration.write_table(STEP_LABEL, check_log, money_by_level, TABLE_RISK_LEVELS)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] the money by level and what each level means
    configuration.log_action(STEP_LABEL, 8, "money to predict by risk level (error_pp: next month's prediction error "
                                            "of the level, weighted by money):")
    configuration.show_table(money_by_level)
    configuration.logger.doc(f"[{STEP_LABEL}] what each level means:")
    configuration.show_table(pd.DataFrame(LEVEL_DEFINITIONS, columns=["nivel", "regla", "significado"]))
    return series_estimate, money_by_level


def credibility_k(groups: pd.DataFrame, configuration: Config) -> pd.Series:
    """Bühlmann-Straub k per reference, from the final groups that share it (one row per group):
    within = mean of p(1 − p); between = weighted variance of p minus its sampling part;
    k = within / between. Fewer than 3 groups: k_cred. No between variance: 10 × k_cred."""
    siblings = groups.drop_duplicates("final_group_id").dropna(subset=["credibility_ref_id", "group_rate"]).copy()
    siblings = siblings[siblings["group_support"] > 0]
    if siblings.empty:
        return pd.Series(dtype=float)
    rates, weights = siblings["group_rate"], siblings["group_support"]
    siblings["_w"] = weights
    siblings["_wp"] = weights * rates
    grouped = siblings.groupby("credibility_ref_id")
    pooled_rate = grouped["_wp"].sum() / grouped["_w"].sum()
    siblings["_pooled"] = siblings["credibility_ref_id"].map(pooled_rate)
    siblings["_within"] = rates * (1 - rates)
    siblings["_w_sq_dev"] = weights * (rates - siblings["_pooled"]) ** 2
    siblings["_w_sampling"] = weights * rates * (1 - rates) / np.maximum(weights, 1)
    grouped = siblings.groupby("credibility_ref_id")
    count = grouped.size()
    within = grouped["_within"].mean()
    between = (grouped["_w_sq_dev"].sum() - grouped["_w_sampling"].sum()) / grouped["_w"].sum()
    k = pd.Series(np.where(between > MIN_BETWEEN_VARIANCE, within / between.where(between > MIN_BETWEEN_VARIANCE),
                           HOMOGENEOUS_POOL_FACTOR * configuration.k_cred), index=count.index)
    k[count < MIN_SIBLINGS_FOR_K] = configuration.k_cred
    return k


def estimate_rates(series_rate: pd.DataFrame, groups: pd.DataFrame, k_by_reference: pd.Series,
                   configuration: Config) -> pd.DataFrame:
    """The rate of the group (blended with its reference below own_rate_floor) and the two errors, per series."""
    estimate = series_rate.merge(groups, on=SERIES_ID_COLUMN, how="left")

    # a series that is not estimable is its own group, with its own rate
    not_estimable = estimate["final_group_id"].isna()
    estimate.loc[not_estimable, "final_group_id"] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
    estimate.loc[not_estimable, "final_step"] = 0
    estimate.loc[not_estimable, "group_series"] = 1
    estimate.loc[not_estimable, "group_support"] = estimate.loc[not_estimable, "n_propio"]
    estimate.loc[not_estimable, "group_rate"] = estimate.loc[not_estimable, "tasa_propia"]
    estimate["final_step"] = estimate["final_step"].astype(int)
    estimate["group_series"] = estimate["group_series"].astype(int)
    estimate[ESTIMATION_ID_COLUMN] = estimate["final_group_id"]

    group_rate, group_support = estimate["group_rate"], estimate["group_support"].fillna(0.0)
    reference_rate, reference_support = estimate["ref_rate"], estimate["ref_support"]
    estimate["k"] = estimate["credibility_ref_id"].map(k_by_reference).fillna(configuration.k_cred)
    blends = (estimate["credibility_ref_id"].notna() & reference_rate.notna() & group_rate.notna()
              & (group_support < configuration.own_rate_floor - ROUNDING_TOLERANCE))

    z = np.where(blends, group_support / (group_support + estimate["k"]), 1.0)
    blended_rate = z * group_rate.fillna(0) + (1 - z) * reference_rate.fillna(0)
    blended_se = np.sqrt((z * binomial_se_pp(group_rate.fillna(0.5), group_support)) ** 2
                         + ((1 - z) * binomial_se_pp(reference_rate.fillna(0.5), reference_support.fillna(1))) ** 2)
    alone_se = binomial_se_pp(group_rate.fillna(0.5), group_support)

    estimate["z"] = np.round(z, 3)
    estimate["tasa_estimada"] = np.where(blends, blended_rate, group_rate)
    estimate["se_estimacion_pp"] = np.where(blends, blended_se, np.where(group_rate.notna(), alone_se, np.nan))
    own_noise = binomial_se_pp(np.nan_to_num(estimate["tasa_estimada"], nan=0.5), np.maximum(estimate["n_propio"], 1))
    estimate["se_prediccion_pp"] = np.where(estimate["tasa_estimada"].notna(),
                                            np.sqrt(estimate["se_estimacion_pp"] ** 2 + own_noise ** 2), np.nan)
    estimate["alcanzo_suelo"] = (group_support >= configuration.support_floor - ROUNDING_TOLERANCE).astype(int)
    return estimate


def risk_levels(estimate: pd.DataFrame, step_names: dict, configuration: Config) -> np.ndarray:
    """The risk level of every series: route and universe first, then sign, then whether the group
    reached the floor, then the pass where its group was formed. The first condition that holds wins."""
    formed_in = estimate["final_step"].map(step_names).fillna("itself")
    merged_close = formed_in.str.startswith("sign") | formed_in.str.startswith("extra")
    itself = estimate["final_step"] == 0
    conditions = [
        estimate[ROUTE_COLUMN] == ROUTE_HISTORY_ONLY,
        estimate[UNIVERSE_COLUMN] != UNIVERSE_NORMAL,
        (estimate[ROUTE_COLUMN] == ROUTE_FUTURE_ONLY) | (estimate["meses_historia"] == 0),
        estimate[SIGN_COLUMN] == SIGN_MIXED,
        (estimate["alcanzo_suelo"] == 0) & (estimate[SIGN_COLUMN] != SIGN_NEUTRAL),
        estimate["alcanzo_suelo"] == 0,
        itself & (estimate["group_support"] >= configuration.own_rate_floor)
        & (estimate["meses_historia"] >= configuration.own_level_min_history_months),
        itself & (estimate["group_support"] >= configuration.own_rate_floor),
        itself,
        merged_close]
    levels = [LEVEL_NO_IMPACT, LEVEL_TIME_SERIES, LEVEL_NO_HISTORY, LEVEL_MIXED, LEVEL_SIGNED_UNDER_FLOOR, LEVEL_FAR,
              LEVEL_OWN, LEVEL_OWN_SHORT, LEVEL_OWN_REINFORCED, LEVEL_BORROWED]
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
    # [1] one estimate per series
    configuration.log_check(STEP_LABEL, check_log, "every series has exactly one estimate",
                            len(estimate) == len(series_rate) and estimate[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(estimate):,} estimates for {len(series_rate):,} series",
                            context=f"{len(estimate):,} series")

    # [2] the group lends its rate: every series of a group has the same estimated rate
    spread = estimate.dropna(subset=["tasa_estimada"]).groupby("final_group_id")["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
    configuration.log_check(STEP_LABEL, check_log, "every series of a group has the same estimated rate",
                            bool((spread <= ROUNDING_TOLERANCE).all()),
                            failure_detail=f"{int((spread > ROUNDING_TOLERANCE).sum()):,} groups with different rates")

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
