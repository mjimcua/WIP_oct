"""
step_10_ladder_groups.py — The ladder: the series are merged pass by pass into bigger groups
until every group has enough support, and every group gets a credibility reference.

THE IDEA: the forecast series of the raw (with its gaps) are the starting point. Each PASS gives
every series a group id. Grouping the numbers by the id of any pass gives the same totals as
the raw, but fewer and bigger series, with better binomial support. The report of the passes
shows how much each pass improves the support.

THE PASSES, in this order (a pass is skipped if it has nothing to do):
  0            itself: the group id is the series itself
  1            sign: the timevarying flags are summarised as their sign (SIG=negativo, …)
  extras       one pass per extra_renovacion, the least informative first (unique contribution
               of step 09): its value becomes '*'
  mandatory    one pass per mandatory dim, in the collapse order of step 09 (the dim whose loss
               costs the least R² first; a *_level_N before its *_level_N−1): its value becomes '*'

THE RULE OF A PASS: group the series by the id of the previous pass. A group with a support
≥ support_floor (30) is CLOSED: it keeps its id in every later pass. The series of an open group
take the id of this pass (one more dim set to '*'). So every pass is a partition: each series
is in one group, and the totals by group always add up to the raw.
  · a signed group (negativo / positivo) keeps its sign: it only goes through the mandatory
    passes while the cumulative loss of R² is ≤ signed_ladder_max_loss
  · a mixed series (flags of both signs) is never merged: it stays itself
  · support of a group = the median, over its months with something due, of the units due
    summed over its series

THE FINAL GROUP of a series is its id after the last pass. The group lends its rate to every
series in it (step 11). A final group below own_rate_floor (271) gets a CREDIBILITY REFERENCE:
a wider group whose rate is blended with the group's own, z = n / (n + k). The reference is the
first of these candidates that has more series than the group and reaches support_floor
(else the widest one that has more series):
  · the group's own id, counted over ALL the series (also the ones closed in earlier passes,
    the big ones included)
  · the id of each later pass its sign allows, counted over all the series
A reference is a source of rate, not a group of the partition: the same series can be in the
reference of many groups. Every series has ONE reference, so the series still add up by it,
but the rate of a reference is computed with every series that matches it.

Actions (logged as they are done):
  1. the plan of the passes, from the decision of step 09
  2. the estimable series with their sign and their columns
  3. the id of every series at every pass (and of every possible pattern, for the references)
  4. the passes: close the groups that reach the floor, move the others one pass up
  5. the final group of every series, and the credibility reference of every final group
  6. check the partition, the signs, the closed groups and the references      checks 1-4
  7. write the passes, their summary and the final groups                       checks 5-7
  8. count the checks; stop if any failed
  9. show how every pass improves the support, as a table

Output: (the passes: series × pass with its group id, the summary by pass, the final group of
every series with its reference, the members of every reference) · tables sff_ladder_steps,
sff_ladder_summary, sff_ladder_groups.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ID_FIELD_SEPARATOR, Config, id_text
from vocabulario import (RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN, SIGN_COLUMN, SIGN_MIXED,
                         SIGN_NEUTRAL, SIGN_TOKEN, TABLE_LADDER_GROUPS, TABLE_LADDER_STEPS, TABLE_LADDER_SUMMARY,
                         UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)


STEP_LABEL = "10"
STEP_NAME = "LADDER GROUPS"
STEP_PURPOSE = ("merge the forecast series pass by pass (sign, extras, mandatory dims in the collapse order) until every "
                "group has enough support: every pass is a partition, so the totals always add up and the series get "
                "fewer and bigger; give every final group below the own-rate floor a wider credibility reference")
STEP_ACTIONS = ["the plan of the passes, from the decision of step 09",
                "the estimable series with their sign and their columns",
                "the id of every series at every pass (and of every possible pattern, for the references)",
                "the passes: close the groups that reach the floor, move the others one pass up",
                "the final group of every series, and the credibility reference of every final group",
                "check the partition, the signs, the closed groups and the references (checks 1-4)",
                "write the passes, their summary and the final groups (checks 5-7)",
                "count the checks; stop if any failed",
                "show how every pass improves the support, as a table"]
STEP_OUTPUT = ("series × pass with its group id · the summary by pass · the final group and the reference of every series · "
               "tables sff_ladder_steps, sff_ladder_summary, sff_ladder_groups")

SUPPORT_TOLERANCE = 1e-9
MONEY_TOLERANCE = 0.01


def build_ladder_groups(rated_units: pd.DataFrame, series_rate: pd.DataFrame, series_lookup: pd.DataFrame,
                        dimension_decision: pd.DataFrame, configuration: Config) -> dict:
    """The passes of the ladder, their summary, the final group and reference of every series.

    INPUT:   rated_units (step 08: the rate of every unit, gaps included) · series_rate (step 08:
             route, universe, sign, support of every series) · the series lookup (step 05: the
             columns of every series) · the dimension decision (step 09) · the Config.
    OUTPUT:  dict(steps, summary, groups, reference_members).
    RULES:   see the module header.
    EDGE CASES: no timevarying dim → no sign pass. No extra → no extra pass. The passes stop as
             soon as no group is open. A group that never reaches the floor keeps its last id.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the plan of the passes
    plan = plan_of_the_passes(dimension_decision, configuration)
    configuration.log_action(STEP_LABEL, 1, f"{len(plan)} passes: " + " → ".join(step["name"] for step in plan)
                             + f" · a signed group may lose up to {configuration.signed_ladder_max_loss} of R² "
                               f"· floor {configuration.support_floor:.0f} · own-rate floor {configuration.own_rate_floor:.0f}")

    # [2] the estimable series
    estimable = series_rate[(series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE)
                            & (series_rate[UNIVERSE_COLUMN] == UNIVERSE_NORMAL)][[SERIES_ID_COLUMN, SIGN_COLUMN, "usd_por_predecir"]]
    estimable = estimable.merge(series_lookup[[SERIES_ID_COLUMN] + configuration.rate_series_columns], on=SERIES_ID_COLUMN)
    estimable = estimable.set_index(SERIES_ID_COLUMN, drop=False)
    configuration.log_action(STEP_LABEL, 2, f"{len(estimable):,} estimable series · signs "
                                            f"{estimable[SIGN_COLUMN].value_counts().to_dict()}")

    # [3] the id of every series at every pass, as if it moved every time
    patterns = pd.DataFrame({step["step"]: pattern_of(estimable, configuration, step["summarise_sign"], step["starred"])
                             for step in plan})
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, configuration.period_col,
                                                              configuration.renewed_units_col, configuration.pipeline_units_col]]
    history = history[history[SERIES_ID_COLUMN].isin(estimable.index)]
    configuration.log_action(STEP_LABEL, 3, f"{len(plan)} patterns per series · {len(history):,} history months")

    # [4] the passes
    steps, summary, final = run_the_passes(estimable, patterns, plan, history, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{summary['ladder_step'].max() + 1} passes run · groups "
                                            f"{summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
                                            f"closed at the end: {int(final['closed'].sum()):,} of {len(final):,} series")

    # [5] the final groups and their references
    groups, reference_members = credibility_references(estimable, patterns, plan, final, history, configuration)
    with_reference = groups["credibility_ref_id"].notna()
    configuration.log_action(STEP_LABEL, 5, f"{groups['final_group_id'].nunique():,} final groups · "
                                            f"{int(with_reference.sum()):,} series take a credibility reference "
                                            f"({reference_members['credibility_ref_id'].nunique():,} references)")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the partition, the signs, the closed groups and the references")
    check_the_ladder(steps, summary, groups, estimable, history, configuration, check_log)

    # [7] the tables
    configuration.log_action(STEP_LABEL, 7, "writing the passes, their summary and the final groups")
    configuration.write_table(STEP_LABEL, check_log, steps, TABLE_LADDER_STEPS)
    configuration.write_table(STEP_LABEL, check_log, summary, TABLE_LADDER_SUMMARY)
    configuration.write_table(STEP_LABEL, check_log, groups, TABLE_LADDER_GROUPS)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] how every pass improves the support
    configuration.log_action(STEP_LABEL, 9, "how every pass improves the support (groups: fewer and bigger; the units due "
                                            "add up in every pass; pct_usd_*: money to predict in groups that reach the floor):")
    configuration.show_table(summary)
    return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PLAN AND THE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════════

def plan_of_the_passes(dimension_decision: pd.DataFrame, configuration: Config) -> list:
    """The passes in order: itself, sign, one per extra, one per mandatory dim. Each pass says whether
    the sign is summarised, which dims are '*' so far, and whether a signed group may take it."""
    mandatory_decision = dimension_decision[dimension_decision["orden_colapso"] > 0].sort_values("orden_colapso")
    collapse_order = list(mandatory_decision["dimension"])
    collapse_order += [dimension for dimension in configuration.business_mandatory_dims if dimension not in collapse_order]
    collapse_loss = dict(zip(mandatory_decision["dimension"], mandatory_decision["perdida_secuencial"].fillna(1.0)))
    extras = dimension_decision[dimension_decision["grupo"] == "extra_renovacion"] if "grupo" in dimension_decision else dimension_decision.iloc[0:0]
    extra_order = list(extras.sort_values(["contribucion_unica", "dimension"])["dimension"])
    extra_order += [extra for extra in configuration.extra_renovacion if extra not in extra_order]

    plan = [dict(step=0, name="itself", summarise_sign=False, starred=frozenset(), signed_allowed=True)]
    if configuration.structural_timevarying_dims:
        plan.append(dict(step=len(plan), name="sign", summarise_sign=True, starred=frozenset(), signed_allowed=True))
    starred = set()
    for extra in extra_order:
        starred = starred | {extra}
        plan.append(dict(step=len(plan), name=f"extra {extra}", summarise_sign=True, starred=frozenset(starred),
                         signed_allowed=True))
    cumulative_loss = 0.0
    for dimension in collapse_order:
        starred = starred | {dimension}
        cumulative_loss += float(collapse_loss.get(dimension, 1.0))
        signed_allowed = (configuration.signed_ladder_max_loss > 0
                          and cumulative_loss <= configuration.signed_ladder_max_loss + SUPPORT_TOLERANCE)
        plan.append(dict(step=len(plan), name=f"without {dimension}", summarise_sign=True, starred=frozenset(starred),
                         signed_allowed=signed_allowed))
    return plan


def pattern_of(estimable: pd.DataFrame, configuration: Config, summarise_sign: bool, starred: set) -> pd.Series:
    """The group id of every series at one pass: the rate series columns in order; the timevarying
    block replaced by SIG=<sign> when summarised; '*' on the starred dims."""
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
            fields.append(np.asarray(id_text(estimable[column_name]), dtype=object))
    if summarise_sign and not sign_written:            # no timevarying declared: the sign still separates groups
        fields.append(SIGN_TOKEN + estimable[SIGN_COLUMN].astype(str).to_numpy(dtype=object))
    joined = fields[0]
    for field in fields[1:]:
        joined = joined + ID_FIELD_SEPARATOR + field
    return pd.Series(joined, index=estimable.index)


def support_of_groups(history: pd.DataFrame, group_of_series: pd.Series, configuration: Config) -> pd.DataFrame:
    """Per group: support (median of the monthly units due summed over its series), units due and
    renewed in total, and its number of series."""
    joined = history.assign(_group=history[SERIES_ID_COLUMN].map(group_of_series)).dropna(subset=["_group"])
    monthly = (joined.groupby(["_group", configuration.period_col])
               [[configuration.renewed_units_col, configuration.pipeline_units_col]].sum().reset_index())
    with_pipeline = monthly[monthly[configuration.pipeline_units_col] > 0]
    result = pd.DataFrame({
        "support": with_pipeline.groupby("_group")[configuration.pipeline_units_col].median(),
        "renewed": monthly.groupby("_group")[configuration.renewed_units_col].sum(),
        "due": monthly.groupby("_group")[configuration.pipeline_units_col].sum(),
        "series": group_of_series.dropna().groupby(group_of_series.dropna()).size()})
    result[["support", "renewed", "due"]] = result[["support", "renewed", "due"]].fillna(0.0)
    result["rate"] = result["renewed"] / result["due"].where(result["due"] > 0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PASSES
# ═══════════════════════════════════════════════════════════════════════════════════

def run_the_passes(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, history: pd.DataFrame,
                   configuration: Config) -> tuple:
    """Pass after pass: the open groups move one pass up, the closed ones keep their id."""
    floor = configuration.support_floor
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    neutral = estimable[SIGN_COLUMN] == SIGN_NEUTRAL
    group = patterns[0].copy()                      # pass 0: every series is its own group
    assigned_at = pd.Series(0, index=estimable.index)
    step_rows, summary_rows = [], []
    usd_to_predict = estimable["usd_por_predecir"]
    total_usd = usd_to_predict.sum()

    for step in plan:
        if step["step"] > 0:
            support_before = support_of_groups(history, group, configuration)["support"]
            open_series = group.map(support_before).fillna(0.0) < floor - SUPPORT_TOLERANCE
            may_move = open_series & ~mixed & (neutral | step["signed_allowed"])
            if not may_move.any():
                break
            group[may_move] = patterns.loc[may_move, step["step"]]
            assigned_at[may_move] = step["step"]
        groups_now = support_of_groups(history, group, configuration)
        support_of_series = group.map(groups_now["support"]).fillna(0.0)
        closed = support_of_series >= floor - SUPPORT_TOLERANCE
        step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"],
                                       "step_name": step["name"], "group_id": group.to_numpy(),
                                       "group_support": support_of_series.to_numpy(), "closed": closed.astype(int).to_numpy()}))
        summary_rows.append({
            "ladder_step": step["step"], "step_name": step["name"], "groups": int(group.nunique()),
            "open_groups": int((groups_now["support"] < floor - SUPPORT_TOLERANCE).sum()),
            "median_group_support": float(groups_now["support"].median()),
            "units_due": float(groups_now["due"].sum()),
            "pct_usd_floor": float(usd_to_predict[closed].sum() / total_usd) if total_usd else np.nan,
            "pct_usd_own_rate": float(usd_to_predict[support_of_series >= configuration.own_rate_floor].sum() / total_usd)
                                if total_usd else np.nan})
    steps = pd.concat(step_rows, ignore_index=True)
    final = pd.DataFrame({"final_group_id": group, "final_step": assigned_at,
                          "closed": group.map(support_of_groups(history, group, configuration)["support"]).fillna(0.0)
                                    >= floor - SUPPORT_TOLERANCE})
    return steps, pd.DataFrame(summary_rows), final


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CREDIBILITY REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════════

def credibility_references(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, final: pd.DataFrame,
                           history: pd.DataFrame, configuration: Config) -> tuple:
    """The final group of every series with its support and rate, and, below the own-rate floor,
    its credibility reference with its support and rate; and the members of every reference."""
    floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
    groups_support = support_of_groups(history, final["final_group_id"], configuration)
    not_mixed = estimable[SIGN_COLUMN] != SIGN_MIXED

    # every pattern of every pass, counted over ALL the series (the closed and the big ones too)
    all_patterns = {}
    for step in plan:
        pattern_of_series = patterns[step["step"]].where(not_mixed)
        all_patterns[step["step"]] = support_of_groups(history, pattern_of_series, configuration)

    group_rows = []
    for group_id, members in final.groupby("final_group_id"):
        group_support = float(groups_support.loc[group_id, "support"]) if group_id in groups_support.index else 0.0
        group_series = int(len(members))
        first_member = members.index[0]
        sign = estimable.loc[first_member, SIGN_COLUMN]
        chosen_step, chosen_id = None, None
        if group_support < own_rate_floor - SUPPORT_TOLERANCE and sign != SIGN_MIXED:
            assigned_step = int(members["final_step"].iloc[0])
            candidates = [assigned_step] + [step["step"] for step in plan if step["step"] > assigned_step
                                            and (sign == SIGN_NEUTRAL or step["signed_allowed"])]
            widest = None
            for candidate in candidates:
                candidate_id = patterns.loc[first_member, candidate]
                candidate_stats = all_patterns[candidate].loc[candidate_id]
                if candidate_stats["series"] > group_series:
                    widest = (candidate, candidate_id)
                    if candidate_stats["support"] >= floor - SUPPORT_TOLERANCE:
                        break
            if widest is not None:
                chosen_step, chosen_id = widest
        row = {"final_group_id": group_id, "final_step": int(members["final_step"].iloc[0]),
               "group_series": group_series, "group_support": group_support,
               "group_rate": float(groups_support.loc[group_id, "rate"]) if group_id in groups_support.index else np.nan,
               "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
        if chosen_id is not None:
            stats = all_patterns[chosen_step].loc[chosen_id]
            row.update(ref_series=int(stats["series"]), ref_support=float(stats["support"]), ref_rate=float(stats["rate"]))
        group_rows.append(row)
    by_group = pd.DataFrame(group_rows)
    for column_name in ("ref_series", "ref_support", "ref_rate"):
        if column_name not in by_group.columns:
            by_group[column_name] = np.nan

    groups = final[["final_group_id"]].reset_index().merge(by_group, on="final_group_id", how="left")
    reference_rows = []
    for (ref_step, ref_id), _ in by_group.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
        members = patterns.index[(patterns[int(ref_step)] == ref_id) & not_mixed]
        reference_rows.append(pd.DataFrame({"credibility_ref_id": ref_id, SERIES_ID_COLUMN: members}))
    reference_members = (pd.concat(reference_rows, ignore_index=True) if reference_rows
                         else pd.DataFrame(columns=["credibility_ref_id", SERIES_ID_COLUMN]))
    return groups, reference_members


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════

def check_the_ladder(steps: pd.DataFrame, summary: pd.DataFrame, groups: pd.DataFrame, estimable: pd.DataFrame,
                     history: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] every pass is a partition: one group per series, and the units due add up to the raw
    raw_units_due = float(history[configuration.pipeline_units_col].sum())
    one_group = steps.groupby("ladder_step")[SERIES_ID_COLUMN].agg(lambda ids: ids.is_unique and len(ids) == len(estimable)).all()
    adds_up = bool(((summary["units_due"] - raw_units_due).abs() <= MONEY_TOLERANCE).all())
    configuration.log_check(STEP_LABEL, check_log, "every pass is a partition: one group per series, the units due add up",
                            bool(one_group) and adds_up,
                            failure_detail=f"units due by pass {summary['units_due'].tolist()} vs {raw_units_due:,.0f}",
                            context=f"{len(summary)} passes · {raw_units_due:,.0f} units due in every one")

    # [2] a group never mixes signs
    sign_of_series = estimable[SIGN_COLUMN]
    signs_per_group = steps.assign(_sign=steps[SERIES_ID_COLUMN].map(sign_of_series)).groupby(["ladder_step", "group_id"])["_sign"].nunique()
    configuration.log_check(STEP_LABEL, check_log, "a group never mixes signs", bool((signs_per_group <= 1).all()),
                            failure_detail=f"{int((signs_per_group > 1).sum()):,} groups with more than one sign")

    # [3] a closed group keeps its id in every later pass, and the groups only get fewer
    ordered = steps.sort_values([SERIES_ID_COLUMN, "ladder_step"])
    previous_closed = ordered.groupby(SERIES_ID_COLUMN)["closed"].shift(1).fillna(0).astype(int)
    previous_id = ordered.groupby(SERIES_ID_COLUMN)["group_id"].shift(1)
    reopened = ordered[(previous_closed == 1) & (ordered["group_id"] != previous_id)]
    fewer = bool((summary["groups"].diff().fillna(0) <= 0).all())
    configuration.log_check(STEP_LABEL, check_log, "a closed group keeps its id, and every pass has fewer or equal groups",
                            reopened.empty and fewer,
                            failure_detail=f"{len(reopened):,} series left a closed group · groups by pass {summary['groups'].tolist()}")

    # [4] a reference is wider than its group
    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates("final_group_id")
    narrower = with_reference[with_reference["ref_series"] <= with_reference["group_series"]]
    configuration.log_check(STEP_LABEL, check_log, "every credibility reference has more series than its group", narrower.empty,
                            failure_detail=f"{len(narrower):,} references not wider than their group",
                            context=f"{len(with_reference):,} groups with a reference")
