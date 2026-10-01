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

AN ID ONLY CHANGES WHEN THE GROUP CHANGES: when a pass sets one more dim to '*' but the group
keeps exactly the same series, it keeps the id it had. So reading the ids pass after pass, an id
changes only where the series really joined others.

THE STAGES: the passes are summarised in the 4 stages business reads (one column pair per stage
in the core, stage_id and stage_support):
  stage 0  raw      the forecast series itself (its gaps filled)
  stage 1  sign     after the sign pass
  stage 2  extras   after the last extra pass
  stage 3  collapse after the last mandatory pass: the COMPOSITION, what is predicted
  (stage 4, credibility, is step 11: it blends rates, it does not merge series)
A stage that adds nothing repeats the id and the support of the stage before it.

THE COMPOSITION of a series is its id after the last pass. The group lends its rate to every
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
from vocabulario import (COMPOSITION_ID_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_COMPOSITION_MEMBERS,
                         TABLE_LADDER_GROUPS, TABLE_LADDER_MERGES, TABLE_LADDER_STEPS, TABLE_LADDER_SUMMARY,
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
    merging = [step["name"] for step in plan if step["merges"]]
    reference_only = [step["name"] for step in plan if not step["merges"]]
    configuration.log_action(STEP_LABEL, 1, f"{len(merging)} merging passes: " + " → ".join(merging)
                             + (f" · searched only for a credibility reference: {', '.join(reference_only)}" if reference_only else "")
                             + f" · stage 3 merges at most {configuration.collapse_passes} dims · a signed group may lose up to "
                               f"{configuration.signed_ladder_max_loss} of R² · floor {configuration.support_floor:.0f} · "
                               f"own-rate floor {configuration.own_rate_floor:.0f}")

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
    passes = run_the_passes(estimable, patterns, plan, history, configuration)
    steps, summary, final = passes["steps"], passes["summary"], passes["final"]
    stages = stages_of_the_series(steps, plan)
    members = composition_members(estimable, patterns, final)
    merges = ladder_merges(steps, series_rate, passes["pattern_stats"], patterns)
    lenders = members.merge(final[[COMPOSITION_ID_COLUMN]].reset_index(), on=[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN], how="left",
                            indicator=True)
    members["role"] = np.where(lenders["_merge"].to_numpy() == "both", "uses", "lends")
    improving = int(merges["improves"].sum()) if len(merges) else 0
    configuration.log_action(STEP_LABEL, 4, f"{len(summary)} passes run · ids {summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
                                            f"{int(final['closed'].sum()):,} of {len(final):,} series reach the floor · "
                                            f"{len(merges):,} merges, {improving:,} improve the error of the series' monthly rate · "
                                            f"{int((members['role'] == 'lends').sum()):,} series lend their history to a composition they do not use")

    # [5] the final groups and their references
    groups, reference_members = credibility_references(estimable, patterns, plan, final, passes["pattern_stats"], configuration)
    with_reference = groups["credibility_ref_id"].notna()
    configuration.log_action(STEP_LABEL, 5, f"{groups[COMPOSITION_ID_COLUMN].nunique():,} final groups · "
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
    configuration.write_table(STEP_LABEL, check_log, merges, TABLE_LADDER_MERGES)
    configuration.write_table(STEP_LABEL, check_log, members, TABLE_COMPOSITION_MEMBERS)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] how every pass improves the support
    configuration.log_action(STEP_LABEL, 9, "how every pass improves the support (groups: fewer and bigger; the units due "
                                            "add up in every pass; pct_usd_*: money to predict in groups that reach the floor):")
    configuration.show_table(summary)
    configuration.logger.doc(f"[{STEP_LABEL}] the 4 stages (0 raw · 1 sign · 2 extras · 3 collapse: the composition), "
                             f"grouping the forecast series by the id of each stage:")
    summary_by_stage = stage_summary(stages, estimable, history, configuration)
    configuration.show_table(summary_by_stage)
    return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members, stages=stages,
                stage_summary=summary_by_stage, composition_members=members, merges=merges)


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
    for collapse_index, dimension in enumerate(collapse_order, start=1):
        starred = starred | {dimension}
        cumulative_loss += float(collapse_loss.get(dimension, 1.0))
        signed_allowed = (configuration.signed_ladder_max_loss > 0
                          and cumulative_loss <= configuration.signed_ladder_max_loss + SUPPORT_TOLERANCE)
        # stage 3 merges at most collapse_passes dims; the passes after them are only searched for a
        # credibility reference (stage 4), never to merge
        plan.append(dict(step=len(plan), name=f"without {dimension}", summarise_sign=True, starred=frozenset(starred),
                         signed_allowed=signed_allowed, merges=collapse_index <= configuration.collapse_passes))
    for step in plan:
        step.setdefault("merges", True)
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
                   configuration: Config) -> dict:
    """The mechanical rule: at every pass EVERY forecast series is grouped by the same pattern (its dims
    with the ones removed so far set to '*'); each series uses the first pass whose group reaches the
    support floor. A series that reaches it on its own (pass 0) keeps its own id, but its history still
    counts in the groups of the others. Mixed series never merge. A series that never reaches the floor
    keeps the widest group it was allowed."""
    floor = configuration.support_floor
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    neutral = estimable[SIGN_COLUMN] == SIGN_NEUTRAL
    merge_passes = [step for step in plan if step["merges"]]

    # the support, rate and number of series of every pattern of every pass, over ALL the series
    pattern_stats = {}
    for step in plan:
        pattern_of_series = patterns[step["step"]] if step["step"] == 0 else patterns[step["step"]].where(~mixed)
        pattern_stats[step["step"]] = support_of_groups(history, pattern_of_series, configuration)

    # the pass every series uses: the first one (allowed for its sign) whose group reaches the floor
    chosen_step = pd.Series(np.nan, index=estimable.index)
    last_allowed = pd.Series(0, index=estimable.index)
    for step in merge_passes:
        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
        last_allowed[allowed] = step["step"]
        support = patterns[step["step"]].map(pattern_stats[step["step"]]["support"]).fillna(0.0)
        reaches = allowed & chosen_step.isna() & (support >= floor - SUPPORT_TOLERANCE)
        chosen_step[reaches] = step["step"]
    reached = chosen_step.notna()
    final_step = chosen_step.fillna(last_allowed).astype(int)

    # the id of every series at every pass: the pattern while it climbs, frozen once it has its group;
    # an id only changes when the group really gets new series (the same series: the previous id)
    ids, step_rows, summary_rows = {}, [], []
    usd_to_predict = estimable["usd_por_predecir"]
    total_usd = usd_to_predict.sum()
    previous_id, previous_count = None, None
    for step in merge_passes:
        climbing = step["step"] <= final_step
        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
        moving = climbing & allowed
        pattern = patterns[step["step"]]
        count = pattern.map(pattern_stats[step["step"]]["series"]).fillna(1)
        if previous_id is None:
            current_id = pattern.copy()
        else:
            current_id = previous_id.copy()
            new_series_joined = moving & (count > previous_count)
            current_id[new_series_joined] = pattern[new_series_joined]
        current_count = count.where(moving, previous_count if previous_count is not None else count)
        support = pattern.map(pattern_stats[step["step"]]["support"]).fillna(0.0)
        if previous_id is not None:
            support = support.where(moving, step_rows[-1].set_index(SERIES_ID_COLUMN)["group_support"])
        ids[step["step"]] = current_id
        has_group = step["step"] >= final_step
        step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"], "step_name": step["name"],
                                       "group_id": current_id.to_numpy(copy=True), "group_support": support.to_numpy(copy=True),
                                       "series_in_group": current_count.to_numpy(copy=True),
                                       "closed": (has_group & (support >= floor - SUPPORT_TOLERANCE)).astype(int).to_numpy()}))
        partition = support_of_groups(history, current_id, configuration)
        group_support = pd.Series(support.to_numpy(), index=current_id.to_numpy()).groupby(level=0).first()
        summary_rows.append({
            "ladder_step": step["step"], "step_name": step["name"], "groups": int(current_id.nunique()),
            "open_groups": int((group_support < floor - SUPPORT_TOLERANCE).sum()),
            "median_group_support": float(group_support.median()),
            "units_due": float(partition["due"].sum()),
            "pct_usd_floor": float(usd_to_predict[support >= floor - SUPPORT_TOLERANCE].sum() / total_usd) if total_usd else np.nan,
            "pct_usd_own_rate": float(usd_to_predict[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
        previous_id, previous_count = current_id, current_count

    # the composition of every series: its id at the pass it uses, with the series in its rate
    composition = ids[merge_passes[-1]["step"]]
    final_pattern = pd.Series(patterns.to_numpy()[np.arange(len(patterns)), patterns.columns.get_indexer(final_step.to_numpy())],
                              index=estimable.index)
    final = pd.DataFrame({COMPOSITION_ID_COLUMN: composition, "final_step": final_step, "final_pattern": final_pattern,
                          "closed": reached})
    return dict(steps=pd.concat(step_rows, ignore_index=True), summary=pd.DataFrame(summary_rows), final=final,
                pattern_stats=pattern_stats)


def composition_members(estimable: pd.DataFrame, patterns: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Every series in the rate of every composition: the series whose pattern, at the pass the composition
    uses, is the composition's (the series that use it and the ones that only lend their history)."""
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    rows = []
    for step_number, compositions in final.drop_duplicates([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]).groupby("final_step"):
        # every series of that pass with its pattern, then joined to the compositions that use that pass
        candidates = patterns[step_number] if step_number == 0 else patterns[step_number][~mixed]
        series_of_pattern = pd.DataFrame({"final_pattern": candidates.to_numpy(), SERIES_ID_COLUMN: candidates.index})
        rows.append(compositions[[COMPOSITION_ID_COLUMN, "final_pattern"]].merge(series_of_pattern, on="final_pattern")
                    [[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN]])
    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)


def ladder_merges(steps: pd.DataFrame, series_rate: pd.DataFrame, pattern_stats: dict, patterns: pd.DataFrame) -> pd.DataFrame:
    """Every merge of every forecast series (a pass where its id changed): the group before and after,
    and whether it improves the prediction of its monthly rate: error alone √(p(1−p)/n) against error
    merged √(bias² + q(1−q)/N), bias = rate of the new group − its own rate."""
    ordered = steps.sort_values([SERIES_ID_COLUMN, "ladder_step"])
    ordered = ordered.assign(previous_id=ordered.groupby(SERIES_ID_COLUMN)["group_id"].shift(1),
                             previous_support=ordered.groupby(SERIES_ID_COLUMN)["group_support"].shift(1))
    changes = ordered[ordered["previous_id"].notna() & (ordered["group_id"] != ordered["previous_id"])].copy()
    if changes.empty:
        return changes
    own = series_rate.set_index(SERIES_ID_COLUMN)[["n_propio", "tasa_propia"]]
    changes = changes.join(own, on=SERIES_ID_COLUMN)
    changes["group_rate"] = [pattern_stats[int(step_number)]["rate"].get(patterns.loc[series_id, int(step_number)], np.nan)
                             for series_id, step_number in zip(changes[SERIES_ID_COLUMN], changes["ladder_step"])]
    own_rate, own_n = changes["tasa_propia"], changes["n_propio"].clip(lower=1)
    group_rate, group_n = changes["group_rate"], changes["group_support"].clip(lower=1)
    changes["bias_pp"] = 100 * (group_rate - own_rate)
    changes["error_alone_pp"] = 100 * np.sqrt(own_rate * (1 - own_rate) / own_n)
    changes["error_merged_pp"] = 100 * np.sqrt((group_rate - own_rate) ** 2 + group_rate * (1 - group_rate) / group_n)
    changes["improves"] = (changes["error_merged_pp"] < changes["error_alone_pp"]).astype(int)
    return changes[[SERIES_ID_COLUMN, "ladder_step", "step_name", "previous_id", "previous_support", "group_id", "group_support",
                    "series_in_group", "n_propio", "tasa_propia", "group_rate", "bias_pp", "error_alone_pp", "error_merged_pp",
                    "improves"]].rename(columns={"n_propio": "own_support", "tasa_propia": "own_rate"})


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGES
# ═══════════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = {0: "raw", 1: "sign", 2: "extras", 3: "collapse"}


def stage_of_each_pass(plan: list) -> dict:
    """The stage of every pass: 0 itself, 1 the sign, 2 the extras, 3 the mandatory dims."""
    stage_of_pass = {}
    for step in plan:
        if step["step"] == 0:
            stage_of_pass[step["step"]] = 0
        elif step["name"] == "sign":
            stage_of_pass[step["step"]] = 1
        elif step["name"].startswith("extra"):
            stage_of_pass[step["step"]] = 2
        else:
            stage_of_pass[step["step"]] = 3
    return stage_of_pass


def stages_of_the_series(steps: pd.DataFrame, plan: list) -> pd.DataFrame:
    """Per forecast series: the id and the support at the end of every stage (the last pass run that
    belongs to the stage or an earlier one; a stage that adds nothing repeats the one before)."""
    stage_of_pass = stage_of_each_pass(plan)
    steps = steps.assign(stage=steps["ladder_step"].map(stage_of_pass))
    stage_columns = {}
    for stage in sorted(STAGE_NAMES):
        last_pass = steps[steps["stage"] <= stage].sort_values("ladder_step").groupby(SERIES_ID_COLUMN).tail(1)
        last_pass = last_pass.set_index(SERIES_ID_COLUMN)
        stage_columns[f"stage{stage}_id"] = last_pass["group_id"]
        stage_columns[f"stage{stage}_support"] = last_pass["group_support"]
    return pd.DataFrame(stage_columns).reset_index()


def stage_summary(stages: pd.DataFrame, estimable: pd.DataFrame, history: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per stage: groups, median support, money to predict in groups that reach the floor and the own-rate
    floor, and the units due (the same in every stage: each stage is a partition of the raw)."""
    usd = estimable.set_index(SERIES_ID_COLUMN)["usd_por_predecir"] if SERIES_ID_COLUMN in estimable.columns else estimable["usd_por_predecir"]
    total_usd = usd.sum()
    rows = []
    for stage, name in STAGE_NAMES.items():
        ids = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_id"]
        support = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_support"]
        groups = support_of_groups(history, ids, configuration)
        rows.append({"stage": stage, "stage_name": name, "groups": int(ids.nunique()),
                     "median_group_support": float(groups["support"].median()),
                     "units_due": float(groups["due"].sum()),
                     "pct_usd_floor": float(usd.reindex(ids.index)[support >= configuration.support_floor].sum() / total_usd) if total_usd else np.nan,
                     "pct_usd_own_rate": float(usd.reindex(ids.index)[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CREDIBILITY REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════════

def credibility_references(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, final: pd.DataFrame,
                           pattern_stats: dict, configuration: Config) -> tuple:
    """The composition of every series with its support, rate and series (every series in its rate) and,
    below the own-rate floor, its credibility reference: the first later pass (merging or not: stage 4 may
    look wider than stage 3 merges) allowed for its sign whose group has more series and reaches the floor,
    else the widest with more series. And the members of every reference."""
    floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    users = final.groupby(COMPOSITION_ID_COLUMN).size()
    group_rows = []
    for (composition_id, step_number, pattern), members in final.groupby([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]):
        stats = pattern_stats[int(step_number)].loc[pattern]
        first_member = members.index[0]
        sign = estimable.loc[first_member, SIGN_COLUMN]
        chosen_step, chosen_id = None, None
        if stats["support"] < own_rate_floor - SUPPORT_TOLERANCE and sign != SIGN_MIXED:
            widest = None
            for step in plan:
                if step["step"] <= step_number or not (sign == SIGN_NEUTRAL or step["signed_allowed"]):
                    continue
                candidate_id = patterns.loc[first_member, step["step"]]
                candidate = pattern_stats[step["step"]].loc[candidate_id]
                if candidate["series"] > stats["series"]:
                    widest = (step["step"], candidate_id)
                    if candidate["support"] >= floor - SUPPORT_TOLERANCE:
                        break
            if widest is not None:
                chosen_step, chosen_id = widest
        row = {COMPOSITION_ID_COLUMN: composition_id, "final_step": int(step_number), "group_series": int(stats["series"]),
               "composition_users": int(users.get(composition_id, 0)), "group_support": float(stats["support"]),
               "group_rate": float(stats["rate"]), "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
        if chosen_id is not None:
            reference = pattern_stats[chosen_step].loc[chosen_id]
            row.update(ref_series=int(reference["series"]), ref_support=float(reference["support"]), ref_rate=float(reference["rate"]))
        group_rows.append(row)
    by_composition = pd.DataFrame(group_rows)
    for column_name in ("ref_series", "ref_support", "ref_rate"):
        if column_name not in by_composition.columns:
            by_composition[column_name] = np.nan

    groups = final[[COMPOSITION_ID_COLUMN]].reset_index().merge(by_composition, on=COMPOSITION_ID_COLUMN, how="left")
    reference_rows = []
    for (ref_step, ref_id), _ in by_composition.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
        members = patterns.index[(patterns[int(ref_step)] == ref_id) & ~mixed]
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
    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
    narrower = with_reference[with_reference["ref_series"] <= with_reference["group_series"]]
    configuration.log_check(STEP_LABEL, check_log, "every credibility reference has more series than its group", narrower.empty,
                            failure_detail=f"{len(narrower):,} references not wider than their group",
                            context=f"{len(with_reference):,} groups with a reference")
