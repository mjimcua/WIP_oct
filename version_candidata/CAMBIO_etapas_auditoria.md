# Cambio · etapas, composición, confianza y tablas de auditoría
Verificación: 18 ficheros de test en verde (13 checks nuevos en test_step_audit.py), 17 comprobaciones del paso AUD en verde y las cifras del sintético idénticas a las de antes (total por año, tasas estimadas, z, forecast, examen de cartera).

## config.py

```diff
--- antes/config.py	2026-10-01 06:11:03.860789254 +0000
+++ config.py	2026-10-01 06:13:12.314069889 +0000
@@ -248,6 +248,10 @@
     acquisition_discount: float = 0.4             # an acquisition is sold with this discount (it renews without it)
     acquisition_level_window_months: int = 3      # acquisition level: last 3 closed months vs the same months a year before
     acquisition_auv_window_months: int = 12       # acquisition value per unit: Σ value / Σ units of the last 12 closed months
+    # the confidence of a future row (step 17): how sure the forecast is of its rate
+    confidence_high_band_pp: float = 5.0      # high: its rate band within ±5 pp (the precision of own_rate_floor)…
+    confidence_high_exam_pp: float = 5.0      # …its composition judged by the backtest, with an exam error ≤ 5 pp
+    confidence_medium_band_pp: float = 10.0   # medium: judged, and a band within ±10 pp; low: the rest (or not judged)
     apply_credibility_shift: bool = True  # a group below own_rate_floor moves its predicted rate toward its credibility
                                           # reference by (1 − z) of their difference of level (logit scale); False = the
                                           # group's prediction as is
```

## main.py

```diff
--- antes/main.py	2026-10-01 06:11:03.861085443 +0000
+++ main.py	2026-10-01 06:16:37.528129220 +0000
@@ -38,6 +38,7 @@
 from step_18_validation import validate_chain
 from step_19_portfolio_exam import examine_portfolio
 from step_20_time_series import build_time_series_and_total, split_time_series_rows
+from step_audit import build_audit_tables
 from step_nucleo import build_core_table
 from step_informe import build_report
 
@@ -189,7 +190,9 @@
         series_table=series_table, series_rate=series_rate, series_estimate=series_estimate, pool_dynamics=pool_dynamics,
         technique_decision=backtest["decision"], exam_by_pool=backtest["exam_by_pool"], forecast=forecast["forecast"],
         time_series_rows=time_series_rows, time_series_table=results["time_series"],
-        forecast_total=results["forecast_total"])                                                  # the core: every row, every decision
+        forecast_total=results["forecast_total"], ladder_stages=ladder["stages"])
+    results["audit"] = build_audit_tables(ladder, series_rate, series_estimate, rated_units, pool_series, pool_reference,
+                                          pool_dynamics, backtest, forecast["forecast"], configuration)   # the satellites                                                  # the core: every row, every decision
     results["validation"] = validate_chain(raw, results, configuration)                            # step 18 (after 19: it reads its exam)
     results["card"] = build_report(raw, results, configuration)                                   # the report, last
     return results
```

## step_10_ladder_groups.py

```diff
--- antes/step_10_ladder_groups.py	2026-10-01 06:11:04.196009306 +0000
+++ step_10_ladder_groups.py	2026-10-01 06:18:30.448955987 +0000
@@ -25,7 +25,20 @@
   · support of a group = the median, over its months with something due, of the units due
     summed over its series
 
-THE FINAL GROUP of a series is its id after the last pass. The group lends its rate to every
+AN ID ONLY CHANGES WHEN THE GROUP CHANGES: when a pass sets one more dim to '*' but the group
+keeps exactly the same series, it keeps the id it had. So reading the ids pass after pass, an id
+changes only where the series really joined others.
+
+THE STAGES: the passes are summarised in the 4 stages business reads (one column pair per stage
+in the core, stage_id and stage_support):
+  stage 0  raw      the forecast series itself (its gaps filled)
+  stage 1  sign     after the sign pass
+  stage 2  extras   after the last extra pass
+  stage 3  collapse after the last mandatory pass: the COMPOSITION, what is predicted
+  (stage 4, credibility, is step 11: it blends rates, it does not merge series)
+A stage that adds nothing repeats the id and the support of the stage before it.
+
+THE COMPOSITION of a series is its id after the last pass. The group lends its rate to every
 series in it (step 11). A final group below own_rate_floor (271) gets a CREDIBILITY REFERENCE:
 a wider group whose rate is blended with the group's own, z = n / (n + k). The reference is the
 first of these candidates that has more series than the group and reaches support_floor
@@ -58,7 +71,7 @@
 import pandas as pd
 
 from config import ID_FIELD_SEPARATOR, Config, id_text
-from vocabulario import (ESTIMATION_ID_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
+from vocabulario import (COMPOSITION_ID_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
                          SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_LADDER_GROUPS, TABLE_LADDER_STEPS,
                          TABLE_LADDER_SUMMARY, UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)
 
@@ -123,6 +136,7 @@
 
     # [4] the passes
     steps, summary, final = run_the_passes(estimable, patterns, plan, history, configuration)
+    stages = stages_of_the_series(steps, plan)
     configuration.log_action(STEP_LABEL, 4, f"{summary['ladder_step'].max() + 1} passes run · groups "
                                             f"{summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
                                             f"closed at the end: {int(final['closed'].sum()):,} of {len(final):,} series")
@@ -130,7 +144,7 @@
     # [5] the final groups and their references
     groups, reference_members = credibility_references(estimable, patterns, plan, final, history, configuration)
     with_reference = groups["credibility_ref_id"].notna()
-    configuration.log_action(STEP_LABEL, 5, f"{groups[ESTIMATION_ID_COLUMN].nunique():,} final groups · "
+    configuration.log_action(STEP_LABEL, 5, f"{groups[COMPOSITION_ID_COLUMN].nunique():,} final groups · "
                                             f"{int(with_reference.sum()):,} series take a credibility reference "
                                             f"({reference_members['credibility_ref_id'].nunique():,} references)")
 
@@ -152,7 +166,12 @@
     configuration.log_action(STEP_LABEL, 9, "how every pass improves the support (groups: fewer and bigger; the units due "
                                             "add up in every pass; pct_usd_*: money to predict in groups that reach the floor):")
     configuration.show_table(summary)
-    return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members)
+    configuration.logger.doc(f"[{STEP_LABEL}] the 4 stages (0 raw · 1 sign · 2 extras · 3 collapse: the composition), "
+                             f"grouping the forecast series by the id of each stage:")
+    summary_by_stage = stage_summary(stages, estimable, history, configuration)
+    configuration.show_table(summary_by_stage)
+    return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members, stages=stages,
+                stage_summary=summary_by_stage)
 
 
 # ═══════════════════════════════════════════════════════════════════════════════════
@@ -247,19 +266,29 @@
     total_usd = usd_to_predict.sum()
 
     for step in plan:
+        previous_assigned_at = assigned_at.copy()
         if step["step"] > 0:
             support_before = support_of_groups(history, group, configuration)["support"]
             open_series = group.map(support_before).fillna(0.0) < floor - SUPPORT_TOLERANCE
             may_move = open_series & ~mixed & (neutral | step["signed_allowed"])
             if not may_move.any():
                 break
+            previous_group = group.copy()
             group[may_move] = patterns.loc[may_move, step["step"]]
             assigned_at[may_move] = step["step"]
+            # a group that keeps exactly the same series keeps the id it had: its id only changes
+            # when it really joins other series
+            unchanged = groups_that_did_not_change(previous_group, group, may_move)
+            for new_id, old_id in unchanged.items():
+                same_group = may_move & (group == new_id)
+                group[same_group] = old_id
+                assigned_at[same_group] = previous_assigned_at[same_group]
         groups_now = support_of_groups(history, group, configuration)
         support_of_series = group.map(groups_now["support"]).fillna(0.0)
         closed = support_of_series >= floor - SUPPORT_TOLERANCE
         step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"],
-                                       "step_name": step["name"], "group_id": group.to_numpy(),
+                                       "step_name": step["name"], "group_id": group.to_numpy(copy=True),   # a copy: the
+                                       # group of later passes is written over the same Series
                                        "group_support": support_of_series.to_numpy(), "closed": closed.astype(int).to_numpy()}))
         summary_rows.append({
             "ladder_step": step["step"], "step_name": step["name"], "groups": int(group.nunique()),
@@ -270,12 +299,77 @@
             "pct_usd_own_rate": float(usd_to_predict[support_of_series >= configuration.own_rate_floor].sum() / total_usd)
                                 if total_usd else np.nan})
     steps = pd.concat(step_rows, ignore_index=True)
-    final = pd.DataFrame({ESTIMATION_ID_COLUMN: group, "final_step": assigned_at,
+    final = pd.DataFrame({COMPOSITION_ID_COLUMN: group, "final_step": assigned_at,
                           "closed": group.map(support_of_groups(history, group, configuration)["support"]).fillna(0.0)
                                     >= floor - SUPPORT_TOLERANCE})
     return steps, pd.DataFrame(summary_rows), final
 
 
+def groups_that_did_not_change(previous_group: pd.Series, group: pd.Series, moved: pd.Series) -> dict:
+    """{new id: old id} for every new group made of exactly one old group, whole (the same series)."""
+    moved_rows = pd.DataFrame({"new_id": group[moved], "old_id": previous_group[moved]})
+    if moved_rows.empty:
+        return {}
+    per_new_group = moved_rows.groupby("new_id")["old_id"].agg(["nunique", "first", "size"])
+    size_of_old_group = previous_group.value_counts()
+    same_series = (per_new_group["nunique"] == 1) & (per_new_group["size"] == per_new_group["first"].map(size_of_old_group))
+    return dict(zip(per_new_group.index[same_series], per_new_group.loc[same_series, "first"]))
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# THE STAGES
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+STAGE_NAMES = {0: "raw", 1: "sign", 2: "extras", 3: "collapse"}
+
+
+def stage_of_each_pass(plan: list) -> dict:
+    """The stage of every pass: 0 itself, 1 the sign, 2 the extras, 3 the mandatory dims."""
+    stage_of_pass = {}
+    for step in plan:
+        if step["step"] == 0:
+            stage_of_pass[step["step"]] = 0
+        elif step["name"] == "sign":
+            stage_of_pass[step["step"]] = 1
+        elif step["name"].startswith("extra"):
+            stage_of_pass[step["step"]] = 2
+        else:
+            stage_of_pass[step["step"]] = 3
+    return stage_of_pass
+
+
+def stages_of_the_series(steps: pd.DataFrame, plan: list) -> pd.DataFrame:
+    """Per forecast series: the id and the support at the end of every stage (the last pass run that
+    belongs to the stage or an earlier one; a stage that adds nothing repeats the one before)."""
+    stage_of_pass = stage_of_each_pass(plan)
+    steps = steps.assign(stage=steps["ladder_step"].map(stage_of_pass))
+    stage_columns = {}
+    for stage in sorted(STAGE_NAMES):
+        last_pass = steps[steps["stage"] <= stage].sort_values("ladder_step").groupby(SERIES_ID_COLUMN).tail(1)
+        last_pass = last_pass.set_index(SERIES_ID_COLUMN)
+        stage_columns[f"stage{stage}_id"] = last_pass["group_id"]
+        stage_columns[f"stage{stage}_support"] = last_pass["group_support"]
+    return pd.DataFrame(stage_columns).reset_index()
+
+
+def stage_summary(stages: pd.DataFrame, estimable: pd.DataFrame, history: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """Per stage: groups, median support, money to predict in groups that reach the floor and the own-rate
+    floor, and the units due (the same in every stage: each stage is a partition of the raw)."""
+    usd = estimable.set_index(SERIES_ID_COLUMN)["usd_por_predecir"] if SERIES_ID_COLUMN in estimable.columns else estimable["usd_por_predecir"]
+    total_usd = usd.sum()
+    rows = []
+    for stage, name in STAGE_NAMES.items():
+        ids = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_id"]
+        support = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_support"]
+        groups = support_of_groups(history, ids, configuration)
+        rows.append({"stage": stage, "stage_name": name, "groups": int(ids.nunique()),
+                     "median_group_support": float(groups["support"].median()),
+                     "units_due": float(groups["due"].sum()),
+                     "pct_usd_floor": float(usd.reindex(ids.index)[support >= configuration.support_floor].sum() / total_usd) if total_usd else np.nan,
+                     "pct_usd_own_rate": float(usd.reindex(ids.index)[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
+    return pd.DataFrame(rows)
+
+
 # ═══════════════════════════════════════════════════════════════════════════════════
 # THE CREDIBILITY REFERENCES
 # ═══════════════════════════════════════════════════════════════════════════════════
@@ -285,7 +379,7 @@
     """The final group of every series with its support and rate, and, below the own-rate floor,
     its credibility reference with its support and rate; and the members of every reference."""
     floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
-    groups_support = support_of_groups(history, final[ESTIMATION_ID_COLUMN], configuration)
+    groups_support = support_of_groups(history, final[COMPOSITION_ID_COLUMN], configuration)
     not_mixed = estimable[SIGN_COLUMN] != SIGN_MIXED
 
     # every pattern of every pass, counted over ALL the series (the closed and the big ones too)
@@ -295,7 +389,7 @@
         all_patterns[step["step"]] = support_of_groups(history, pattern_of_series, configuration)
 
     group_rows = []
-    for group_id, members in final.groupby(ESTIMATION_ID_COLUMN):
+    for group_id, members in final.groupby(COMPOSITION_ID_COLUMN):
         group_support = float(groups_support.loc[group_id, "support"]) if group_id in groups_support.index else 0.0
         group_series = int(len(members))
         first_member = members.index[0]
@@ -315,7 +409,7 @@
                         break
             if widest is not None:
                 chosen_step, chosen_id = widest
-        row = {ESTIMATION_ID_COLUMN: group_id, "final_step": int(members["final_step"].iloc[0]),
+        row = {COMPOSITION_ID_COLUMN: group_id, "final_step": int(members["final_step"].iloc[0]),
                "group_series": group_series, "group_support": group_support,
                "group_rate": float(groups_support.loc[group_id, "rate"]) if group_id in groups_support.index else np.nan,
                "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
@@ -328,7 +422,7 @@
         if column_name not in by_group.columns:
             by_group[column_name] = np.nan
 
-    groups = final[[ESTIMATION_ID_COLUMN]].reset_index().merge(by_group, on=ESTIMATION_ID_COLUMN, how="left")
+    groups = final[[COMPOSITION_ID_COLUMN]].reset_index().merge(by_group, on=COMPOSITION_ID_COLUMN, how="left")
     reference_rows = []
     for (ref_step, ref_id), _ in by_group.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
         members = patterns.index[(patterns[int(ref_step)] == ref_id) & not_mixed]
@@ -371,7 +465,7 @@
                             failure_detail=f"{len(reopened):,} series left a closed group · groups by pass {summary['groups'].tolist()}")
 
     # [4] a reference is wider than its group
-    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates(ESTIMATION_ID_COLUMN)
+    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
     narrower = with_reference[with_reference["ref_series"] <= with_reference["group_series"]]
     configuration.log_check(STEP_LABEL, check_log, "every credibility reference has more series than its group", narrower.empty,
                             failure_detail=f"{len(narrower):,} references not wider than their group",
```

## step_11_ladder.py

```diff
--- antes/step_11_ladder.py	2026-10-01 06:11:04.087966663 +0000
+++ step_11_ladder.py	2026-10-01 06:12:43.325593269 +0000
@@ -54,7 +54,7 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (ESTIMATION_ID_COLUMN, LEVEL_BORROWED, LEVEL_FAR, LEVEL_MIXED, LEVEL_NO_HISTORY,
+from vocabulario import (COMPOSITION_ID_COLUMN, LEVEL_BORROWED, LEVEL_FAR, LEVEL_MIXED, LEVEL_NO_HISTORY,
                          LEVEL_NO_IMPACT, LEVEL_OWN, LEVEL_OWN_REINFORCED, LEVEL_OWN_SHORT, LEVEL_SIGNED_UNDER_FLOOR,
                          LEVEL_TIME_SERIES, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, SERIES_ID_COLUMN,
                          SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, TABLE_RISK_LEVELS, TABLE_SERIES_ESTIMATE,
@@ -120,7 +120,7 @@
     # [1] the final group and the reference of every estimable series
     groups = ladder["groups"]
     step_names = dict(zip(ladder["summary"]["ladder_step"], ladder["summary"]["step_name"]))
-    configuration.log_action(STEP_LABEL, 1, f"{len(groups):,} estimable series in {groups[ESTIMATION_ID_COLUMN].nunique():,} "
+    configuration.log_action(STEP_LABEL, 1, f"{len(groups):,} estimable series in {groups[COMPOSITION_ID_COLUMN].nunique():,} "
                                             f"final groups; {int(groups['credibility_ref_id'].notna().sum()):,} of them "
                                             f"with a credibility reference")
 
@@ -162,14 +162,19 @@
     return series_estimate, money_by_level
 
 
-def credibility_k(groups: pd.DataFrame, configuration: Config) -> pd.Series:
-    """Bühlmann-Straub k per reference, from the final groups that share it (one row per group):
-    within = mean of p(1 − p); between = weighted variance of p minus its sampling part;
-    k = within / between. Fewer than 3 groups: k_cred. No between variance: 10 × k_cred."""
-    siblings = groups.drop_duplicates(ESTIMATION_ID_COLUMN).dropna(subset=["credibility_ref_id", "group_rate"]).copy()
+def credibility_k_detail(groups: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """Bühlmann-Straub k per credibility reference, with the numbers it is made of, from the
+    compositions that share the reference (one row per composition):
+      within  = mean of p(1 − p): how much a composition's rate moves by chance
+      between = weighted variance of p minus its sampling part: how much the compositions differ for real
+      k = within / between · source 'estimated'
+      fewer than 3 compositions: k_cred · source 'default'
+      no real difference between them: 10 × k_cred (they take the reference's rate) · source 'homogeneous'"""
+    siblings = groups.drop_duplicates(COMPOSITION_ID_COLUMN).dropna(subset=["credibility_ref_id", "group_rate"]).copy()
     siblings = siblings[siblings["group_support"] > 0]
+    columns = ["credibility_ref_id", "compositions_using", "within_variance", "between_variance", "k", "k_source"]
     if siblings.empty:
-        return pd.Series(dtype=float)
+        return pd.DataFrame(columns=columns)
     rates, weights = siblings["group_rate"], siblings["group_support"]
     siblings["_w"] = weights
     siblings["_wp"] = weights * rates
@@ -180,13 +185,23 @@
     siblings["_w_sq_dev"] = weights * (rates - siblings["_pooled"]) ** 2
     siblings["_w_sampling"] = weights * rates * (1 - rates) / np.maximum(weights, 1)
     grouped = siblings.groupby("credibility_ref_id")
-    count = grouped.size()
-    within = grouped["_within"].mean()
-    between = (grouped["_w_sq_dev"].sum() - grouped["_w_sampling"].sum()) / grouped["_w"].sum()
-    k = pd.Series(np.where(between > MIN_BETWEEN_VARIANCE, within / between.where(between > MIN_BETWEEN_VARIANCE),
-                           HOMOGENEOUS_POOL_FACTOR * configuration.k_cred), index=count.index)
-    k[count < MIN_SIBLINGS_FOR_K] = configuration.k_cred
-    return k
+    detail = pd.DataFrame({"compositions_using": grouped.size(),
+                           "within_variance": grouped["_within"].mean(),
+                           "between_variance": (grouped["_w_sq_dev"].sum() - grouped["_w_sampling"].sum()) / grouped["_w"].sum()})
+    has_between = detail["between_variance"] > MIN_BETWEEN_VARIANCE
+    detail["k"] = np.where(has_between, detail["within_variance"] / detail["between_variance"].where(has_between),
+                           HOMOGENEOUS_POOL_FACTOR * configuration.k_cred)
+    detail["k_source"] = np.where(has_between, "estimated", "homogeneous")
+    too_few = detail["compositions_using"] < MIN_SIBLINGS_FOR_K
+    detail.loc[too_few, "k"] = configuration.k_cred
+    detail.loc[too_few, "k_source"] = "default"
+    return detail.reset_index()[columns]
+
+
+def credibility_k(groups: pd.DataFrame, configuration: Config) -> pd.Series:
+    """The k of every credibility reference (see credibility_k_detail)."""
+    detail = credibility_k_detail(groups, configuration)
+    return detail.set_index("credibility_ref_id")["k"] if len(detail) else pd.Series(dtype=float)
 
 
 def estimate_rates(series_rate: pd.DataFrame, groups: pd.DataFrame, k_by_reference: pd.Series,
@@ -195,15 +210,15 @@
     estimate = series_rate.merge(groups, on=SERIES_ID_COLUMN, how="left")
 
     # a series that is not estimable is its own group, with its own rate
-    not_estimable = estimate[ESTIMATION_ID_COLUMN].isna()
-    estimate.loc[not_estimable, ESTIMATION_ID_COLUMN] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
+    not_estimable = estimate[COMPOSITION_ID_COLUMN].isna()
+    estimate.loc[not_estimable, COMPOSITION_ID_COLUMN] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
     estimate.loc[not_estimable, "final_step"] = 0
     estimate.loc[not_estimable, "group_series"] = 1
     estimate.loc[not_estimable, "group_support"] = estimate.loc[not_estimable, "n_propio"]
     estimate.loc[not_estimable, "group_rate"] = estimate.loc[not_estimable, "tasa_propia"]
     estimate["final_step"] = estimate["final_step"].astype(int)
     estimate["group_series"] = estimate["group_series"].astype(int)
-    estimate[ESTIMATION_ID_COLUMN] = estimate[ESTIMATION_ID_COLUMN]
+    estimate[COMPOSITION_ID_COLUMN] = estimate[COMPOSITION_ID_COLUMN]
 
     group_rate, group_support = estimate["group_rate"], estimate["group_support"].fillna(0.0)
     reference_rate, reference_support = estimate["ref_rate"], estimate["ref_support"]
@@ -224,6 +239,8 @@
     estimate["se_prediccion_pp"] = np.where(estimate["tasa_estimada"].notna(),
                                             np.sqrt(estimate["se_estimacion_pp"] ** 2 + own_noise ** 2), np.nan)
     estimate["alcanzo_suelo"] = (group_support >= configuration.support_floor - ROUNDING_TOLERANCE).astype(int)
+    # how much the credibility moves the rate of the composition (pp): 0 when it predicts alone
+    estimate["credibility_effect_pp"] = PERCENTAGE_POINTS * (estimate["tasa_estimada"] - group_rate)
     return estimate
 
 
@@ -274,7 +291,7 @@
                             context=f"{len(estimate):,} series")
 
     # [2] the group lends its rate: every series of a group has the same estimated rate
-    spread = estimate.dropna(subset=["tasa_estimada"]).groupby(ESTIMATION_ID_COLUMN)["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
+    spread = estimate.dropna(subset=["tasa_estimada"]).groupby(COMPOSITION_ID_COLUMN)["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
     configuration.log_check(STEP_LABEL, check_log, "every series of a group has the same estimated rate",
                             bool((spread <= ROUNDING_TOLERANCE).all()),
                             failure_detail=f"{int((spread > ROUNDING_TOLERANCE).sum()):,} groups with different rates")
```

## step_12_pool_series.py

```diff
--- antes/step_12_pool_series.py	2026-10-01 06:11:04.146143522 +0000
+++ step_12_pool_series.py	2026-10-01 06:11:04.374475869 +0000
@@ -39,7 +39,7 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT, RATE_COLUMN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT, RATE_COLUMN,
                          SERIES_ID_COLUMN, TABLE_POOL_REFERENCE, TABLE_POOL_SERIES)
 
 
@@ -69,9 +69,9 @@
     period_column = configuration.period_col
 
     # [1] the final groups of step 10 (the estimable series) and their series: a partition
-    membership = ladder["groups"][[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN]].rename(columns={ESTIMATION_ID_COLUMN: ESTIMATION_ID_COLUMN})
+    membership = ladder["groups"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]].rename(columns={COMPOSITION_ID_COLUMN: COMPOSITION_ID_COLUMN})
     chosen = series_estimate[series_estimate[SERIES_ID_COLUMN].isin(set(membership[SERIES_ID_COLUMN]))]
-    chosen_ids = set(membership[ESTIMATION_ID_COLUMN])
+    chosen_ids = set(membership[COMPOSITION_ID_COLUMN])
     configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} final groups of {len(chosen):,} series "
                                             f"(each series in one group)")
 
@@ -79,15 +79,15 @@
     history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN,
                                                              configuration.renewed_units_col, configuration.pipeline_units_col]]
     joined = history.merge(membership, on=SERIES_ID_COLUMN)
-    pool_series = (joined.groupby([ESTIMATION_ID_COLUMN, period_column])
+    pool_series = (joined.groupby([COMPOSITION_ID_COLUMN, period_column])
                    .agg(rol=(CALENDAR_ROLE_COLUMN, "first"),
                         renovadas=(configuration.renewed_units_col, "sum"),
                         vencen=(configuration.pipeline_units_col, "sum"),
                         series_en_el_mes=(SERIES_ID_COLUMN, "nunique"))
-                   .reset_index().sort_values([ESTIMATION_ID_COLUMN, period_column]).reset_index(drop=True))
+                   .reset_index().sort_values([COMPOSITION_ID_COLUMN, period_column]).reset_index(drop=True))
     pool_series[RATE_COLUMN] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
     configuration.log_action(STEP_LABEL, 2, f"{len(pool_series):,} group × month rows; a group has "
-                                            f"{pool_series.groupby(ESTIMATION_ID_COLUMN).size().median():.0f} months (median)")
+                                            f"{pool_series.groupby(COMPOSITION_ID_COLUMN).size().median():.0f} months (median)")
 
     # [3] the reference of every id
     pool_reference = reference_of_pools(pool_series, chosen, membership, configuration)
@@ -98,18 +98,18 @@
 
     # [4] the checks
     configuration.log_action(STEP_LABEL, 4, "checking the series and the reference")
-    missing_ids = chosen_ids - set(pool_series[ESTIMATION_ID_COLUMN])
+    missing_ids = chosen_ids - set(pool_series[COMPOSITION_ID_COLUMN])
     configuration.log_check(STEP_LABEL, check_log, "every final group has a monthly series", not missing_ids,
                             failure_detail=f"{len(missing_ids):,} ids without months: {sorted(missing_ids)[:5]}",
                             context=f"{len(chosen_ids):,} ids")
-    support_in_step_10 = (ladder["groups"].drop_duplicates(ESTIMATION_ID_COLUMN)
-                          .set_index(ESTIMATION_ID_COLUMN)["group_support"].rename("n_pool_paso_10"))
-    compared = pool_reference.join(support_in_step_10, on=ESTIMATION_ID_COLUMN)
+    support_in_step_10 = (ladder["groups"].drop_duplicates(COMPOSITION_ID_COLUMN)
+                          .set_index(COMPOSITION_ID_COLUMN)["group_support"].rename("n_pool_paso_10"))
+    compared = pool_reference.join(support_in_step_10, on=COMPOSITION_ID_COLUMN)
     mismatched = compared[(compared["n_pool"] - compared["n_pool_paso_10"]).abs() > SUPPORT_TOLERANCE]
     configuration.log_check(STEP_LABEL, check_log, "the support of every group equals its support in step 10",
                             mismatched.empty,
                             failure_detail=f"{len(mismatched):,} groups whose support differs from step 10",
-                            examples=mismatched[[ESTIMATION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
+                            examples=mismatched[[COMPOSITION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
     out_of_range = pool_series[(pool_series[RATE_COLUMN] < 0) | (pool_series[RATE_COLUMN] > 1)]
     configuration.log_check(STEP_LABEL, check_log, "every monthly rate is between 0 and 1", out_of_range.empty,
                             failure_detail=f"{len(out_of_range):,} months with a rate outside [0, 1]", blocking=False,
@@ -128,7 +128,7 @@
     configuration.log_action(STEP_LABEL, 7, f"final groups by gate (nivel: support ≥ {configuration.support_floor:.0f}, "
                                             f"judged by the backtest; soporte: below it, takes the challenger):")
     configuration.show_table(pool_reference.groupby("gate")
-                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), series_que_lo_usan=("series_que_lo_usan", "sum"),
+                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), series_que_lo_usan=("series_que_lo_usan", "sum"),
                                   usd_por_predecir=("usd_por_predecir", "sum"), meses_mediana=("meses", "median"))
                              .reset_index())
     configuration.logger.doc(f"[{STEP_LABEL}] the {LARGEST_SHOWN} groups with the most money to predict:")
@@ -141,17 +141,17 @@
     """One row per id: months, support, rate, first and last month, series, money, gate."""
     period_column = configuration.period_col
     with_pipeline = pool_series[pool_series["vencen"] > 0]
-    grouped = with_pipeline.groupby(ESTIMATION_ID_COLUMN)
+    grouped = with_pipeline.groupby(COMPOSITION_ID_COLUMN)
     reference = pd.DataFrame({
         "meses": grouped.size(),
         "primer_mes": grouped[period_column].min(),
         "ultimo_mes": grouped[period_column].max(),
         "n_pool": grouped["vencen"].median(),
         "tasa_pool": grouped["renovadas"].sum() / grouped["vencen"].sum(),
-        "series_en_el_pool": membership.groupby(ESTIMATION_ID_COLUMN)[SERIES_ID_COLUMN].nunique()})
-    reference["series_que_lo_usan"] = chosen.groupby(ESTIMATION_ID_COLUMN).size()
-    reference["usd_por_predecir"] = chosen.groupby(ESTIMATION_ID_COLUMN)["usd_por_predecir"].sum()
-    reference = reference.reset_index().rename(columns={"index": ESTIMATION_ID_COLUMN})
+        "series_en_el_pool": membership.groupby(COMPOSITION_ID_COLUMN)[SERIES_ID_COLUMN].nunique()})
+    reference["series_que_lo_usan"] = chosen.groupby(COMPOSITION_ID_COLUMN).size()
+    reference["usd_por_predecir"] = chosen.groupby(COMPOSITION_ID_COLUMN)["usd_por_predecir"].sum()
+    reference = reference.reset_index().rename(columns={"index": COMPOSITION_ID_COLUMN})
     reference[["series_que_lo_usan", "usd_por_predecir"]] = reference[["series_que_lo_usan", "usd_por_predecir"]].fillna(0)
     reference["gate"] = np.where(reference["n_pool"] >= configuration.support_floor, GATE_LEVEL, GATE_SUPPORT)
     return reference
```

## step_13_dynamics.py

```diff
--- antes/step_13_dynamics.py	2026-10-01 06:11:04.146405623 +0000
+++ step_13_dynamics.py	2026-10-01 06:11:11.602120249 +0000
@@ -38,10 +38,10 @@
 Checks (logged as they are made, numbered, at the level of their status):
    1. every pool measured has its attributes (φ ≥ 0, p-values between 0 and 1)
    2. the portfolio profile covers the 12 calendar months
-   3-4. tables sff_dinamica_pool and sff_estacionalidad_cartera written and read back
+   3-4. tables sff_composition_dynamics and sff_estacionalidad_cartera written and read back
 
 Output: (the dynamics of every pool, the month profile of the portfolio) · tables
-sff_dinamica_pool, sff_estacionalidad_cartera.
+sff_composition_dynamics, sff_estacionalidad_cartera.
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
@@ -50,7 +50,7 @@
 from scipy import stats
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, TABLE_POOL_DYNAMICS,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, TABLE_POOL_DYNAMICS,
                          TABLE_PORTFOLIO_SEASONALITY, TRUTH_ROLES)
 
 
@@ -67,7 +67,7 @@
                 "write the pools' dynamics and the portfolio's profile (checks 3-4)",
                 "count the checks; stop if any failed",
                 "show the general verdict: the portfolio's profile and how many pools show a trend, a season, φ > 1"]
-STEP_OUTPUT = "one row per pool (φ, trend, season, months high/low) · the portfolio's month profile · tables sff_dinamica_pool, sff_estacionalidad_cartera"
+STEP_OUTPUT = "one row per pool (φ, trend, season, months high/low) · the portfolio's month profile · tables sff_composition_dynamics, sff_estacionalidad_cartera"
 
 # ─── named constants ─────────────────────────────────────────────────────────────
 PERCENTAGE_POINTS = 100
@@ -84,8 +84,8 @@
     truth_months = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (pool_series["vencen"] > 0)]
 
     # [1] the pools measured
-    months_per_pool = truth_months.groupby(ESTIMATION_ID_COLUMN).size()
-    measured_ids = [estimation_id for estimation_id in pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, ESTIMATION_ID_COLUMN]
+    months_per_pool = truth_months.groupby(COMPOSITION_ID_COLUMN).size()
+    measured_ids = [estimation_id for estimation_id in pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN]
                     if months_per_pool.get(estimation_id, 0) >= configuration.dynamics_min_months]
     configuration.log_action(STEP_LABEL, 1, f"{len(measured_ids):,} pools measured (support ≥ {configuration.support_floor:.0f} "
                                             f"and ≥ {configuration.dynamics_min_months} months) of {len(pool_reference):,}")
@@ -93,12 +93,12 @@
     # [2] the attributes of every pool
     dynamics_rows = []
     for estimation_id in measured_ids:
-        monthly = truth_months[truth_months[ESTIMATION_ID_COLUMN] == estimation_id].sort_values(period_column)
-        dynamics_rows.append({ESTIMATION_ID_COLUMN: estimation_id,
+        monthly = truth_months[truth_months[COMPOSITION_ID_COLUMN] == estimation_id].sort_values(period_column)
+        dynamics_rows.append({COMPOSITION_ID_COLUMN: estimation_id,
                               **dynamics_of_one_series(monthly, period_column, configuration)})
     pool_dynamics = pd.DataFrame(dynamics_rows)
     if len(pool_dynamics):
-        pool_dynamics = pool_dynamics.merge(pool_reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
+        pool_dynamics = pool_dynamics.merge(pool_reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
     configuration.log_action(STEP_LABEL, 2, f"φ median {pool_dynamics['phi'].median():.2f} · "
                                             f"{int(pool_dynamics['estacional'].sum())} seasonal · "
                                             f"{int((pool_dynamics['tendencia'] != 0).sum())} with a trend"
```

## step_14_backtest.py

```diff
--- antes/step_14_backtest.py	2026-10-01 06:11:04.146242627 +0000
+++ step_14_backtest.py	2026-10-01 06:11:04.376158511 +0000
@@ -69,7 +69,7 @@
 
 from config import Config
 from techniques import CATALOGUE, eligible_techniques, inverse_logit, logit, predict_logit, technique_table
-from vocabulario import (CHALLENGER_ORIGIN, CHAMPION_ORIGIN, ESTIMATION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM,
+from vocabulario import (CHALLENGER_ORIGIN, CHAMPION_ORIGIN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM,
                          PURPOSE_SELECTION, RATE_COLUMN, TABLE_BACKTEST_PREDICTIONS, TABLE_ERROR_BANDS,
                          TABLE_EXAM_BY_POOL, TABLE_EXAM_TOTAL, TABLE_TECHNIQUES, TABLE_TECHNIQUE_DECISION)
 
@@ -96,7 +96,7 @@
 PERCENTAGE_POINTS = 100
 MIN_EXAM_COVERAGE = 0.80
 SUPPORT_ORIGIN = "sin_soporte"      # an id below the floor: not judged, it takes the challenger
-PREDICTION_COLUMNS = [ESTIMATION_ID_COLUMN, "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
+PREDICTION_COLUMNS = [COMPOSITION_ID_COLUMN, "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
                       "tasa_pred", "tasa_real", "vencen_real", "err_pp", "se_binom_pp", "err_norm"]
 
 
@@ -118,10 +118,10 @@
     log_how_the_test_works(selection_months, exam_months, pool_reference, configuration)
 
     # [2] the predictions
-    judged_ids = set(pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, ESTIMATION_ID_COLUMN])
+    judged_ids = set(pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN])
     predictions = predict_every_target(pool_series, judged_ids, selection_months, exam_months, configuration)
     predictions["tramo_h"] = predictions["h"].map(lambda h: band_of_horizon(int(h), configuration.horizon_bands))
-    configuration.log_action(STEP_LABEL, 2, f"{len(predictions):,} predictions: {predictions[ESTIMATION_ID_COLUMN].nunique():,} "
+    configuration.log_action(STEP_LABEL, 2, f"{len(predictions):,} predictions: {predictions[COMPOSITION_ID_COLUMN].nunique():,} "
                                             f"ids × {predictions['mes_objetivo'].nunique()} target months × "
                                             f"{len(configuration.backtest_horizons)} horizons × up to {len(CATALOGUE)} techniques "
                                             f"(each where its history allows)")
@@ -137,7 +137,7 @@
 
     # [5] the exam: per id and for the total
     exam_by_pool, exam_total, exam_rows = exam_precision(predictions, decision, bands, configuration)
-    configuration.log_action(STEP_LABEL, 5, f"exam measured on {exam_by_pool[ESTIMATION_ID_COLUMN].nunique():,} ids and "
+    configuration.log_action(STEP_LABEL, 5, f"exam measured on {exam_by_pool[COMPOSITION_ID_COLUMN].nunique():,} ids and "
                                             f"{exam_total['mes_objetivo'].nunique()} months")
 
     # [6] the checks
@@ -218,10 +218,10 @@
     period_column = configuration.period_col
     purpose_of = {month.ordinal: PURPOSE_SELECTION for month in selection_months}
     purpose_of.update({month.ordinal: PURPOSE_EXAM for month in exam_months})
-    judged_series = pool_series[pool_series[ESTIMATION_ID_COLUMN].isin(judged_ids) & pool_series[RATE_COLUMN].notna()
+    judged_series = pool_series[pool_series[COMPOSITION_ID_COLUMN].isin(judged_ids) & pool_series[RATE_COLUMN].notna()
                                 & (pool_series["vencen"] > 0)]
     rows = []
-    for estimation_id, monthly in judged_series.groupby(ESTIMATION_ID_COLUMN, sort=False):
+    for estimation_id, monthly in judged_series.groupby(COMPOSITION_ID_COLUMN, sort=False):
         monthly = monthly.sort_values(period_column)
         month_ordinals = np.array([month.ordinal for month in monthly[period_column]])
         calendar_months = np.array([month.month for month in monthly[period_column]])
@@ -269,11 +269,11 @@
     catalogue_order = {technique_id: position for position, technique_id in enumerate(CATALOGUE)}
     selection = predictions[predictions["proposito"] == PURPOSE_SELECTION].assign(
         abs_norm=lambda frame: frame["err_norm"].abs(), abs_pp=lambda frame: frame["err_pp"].abs())
-    scores = (selection.groupby([ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"])
+    scores = (selection.groupby([COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
               .agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"), n_predicciones=("abs_norm", "size"))
               .reset_index())
     decision_rows = []
-    for (estimation_id, band_name), block in scores.groupby([ESTIMATION_ID_COLUMN, "tramo_h"]):
+    for (estimation_id, band_name), block in scores.groupby([COMPOSITION_ID_COLUMN, "tramo_h"]):
         block = block.set_index("tecnica")
         challenger_score = block.loc[challenger, "err_norm_medio"] if challenger in block.index else np.inf
         margin = float(configuration.challenger_margin_by_band.get(band_name, 0.0))
@@ -286,7 +286,7 @@
                 chosen, origin = best, CHAMPION_ORIGIN
         chosen_row = block.loc[chosen] if chosen in block.index else pd.Series(dict(err_norm_medio=np.nan, err_pp_medio=np.nan,
                                                                                     n_predicciones=0))
-        decision_rows.append({ESTIMATION_ID_COLUMN: estimation_id, "tramo_h": band_name, "tecnica": chosen,
+        decision_rows.append({COMPOSITION_ID_COLUMN: estimation_id, "tramo_h": band_name, "tecnica": chosen,
                               "tecnica_origen": origin, "err_norm_seleccion": chosen_row["err_norm_medio"],
                               "err_pp_seleccion": chosen_row["err_pp_medio"], "n_predicciones": int(chosen_row["n_predicciones"]),
                               "retador_err_norm_seleccion": challenger_score})
@@ -294,9 +294,9 @@
 
     # every id and band gets a decision: the unjudged ones (and a judged id with no selection
     # month in a band) take the challenger
-    all_pairs = pd.MultiIndex.from_product([pool_reference[ESTIMATION_ID_COLUMN], list(configuration.horizon_bands)],
-                                           names=[ESTIMATION_ID_COLUMN, "tramo_h"]).to_frame(index=False)
-    decision = all_pairs.merge(decision, on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
+    all_pairs = pd.MultiIndex.from_product([pool_reference[COMPOSITION_ID_COLUMN], list(configuration.horizon_bands)],
+                                           names=[COMPOSITION_ID_COLUMN, "tramo_h"]).to_frame(index=False)
+    decision = all_pairs.merge(decision, on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
     unjudged = decision["tecnica"].isna()
     decision.loc[unjudged, "tecnica"] = challenger
     decision.loc[unjudged, "tecnica_origen"] = SUPPORT_ORIGIN
@@ -318,7 +318,7 @@
     """The chosen technique and the challenger in the exam months: per id and for the total."""
     challenger = configuration.challenger_technique
     exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
-    chosen_rows = exam.merge(decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"]], on=[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"])
+    chosen_rows = exam.merge(decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"]], on=[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
     chosen_rows = chosen_rows.merge(bands[["tecnica", "h", "q_low_norm", "q_high_norm"]], on=["tecnica", "h"], how="left")
     chosen_rows["dentro_banda"] = ((chosen_rows["err_norm"] >= chosen_rows["q_low_norm"])
                                    & (chosen_rows["err_norm"] <= chosen_rows["q_high_norm"])).astype(int)
@@ -326,15 +326,15 @@
 
     def summary(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
         return (rows.assign(abs_pp=rows["err_pp"].abs(), abs_norm=rows["err_norm"].abs())
-                .groupby([ESTIMATION_ID_COLUMN, "tramo_h"])
+                .groupby([COMPOSITION_ID_COLUMN, "tramo_h"])
                 .agg(**{f"{prefix}_err_pp_medio": ("abs_pp", "mean"), f"{prefix}_sesgo_pp": ("err_pp", "mean"),
                         f"{prefix}_err_norm_medio": ("abs_norm", "mean")}))
 
     exam_by_pool = summary(chosen_rows, "elegida").join(summary(challenger_rows, "retador"), how="left").reset_index()
-    exam_by_pool = exam_by_pool.merge(decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]],
-                                      on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
+    exam_by_pool = exam_by_pool.merge(decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]],
+                                      on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
     exam_by_pool["mejora_pp"] = exam_by_pool["retador_err_pp_medio"] - exam_by_pool["elegida_err_pp_medio"]
-    exam_by_pool["dentro_banda"] = chosen_rows.groupby([ESTIMATION_ID_COLUMN, "tramo_h"])["dentro_banda"].mean().to_numpy()
+    exam_by_pool["dentro_banda"] = chosen_rows.groupby([COMPOSITION_ID_COLUMN, "tramo_h"])["dentro_banda"].mean().to_numpy()
 
     # the TOTAL: Σ predicted renewals vs Σ real renewals of every judged id, per exam month and horizon
     def total_of(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
@@ -367,7 +367,7 @@
                             context=f"{len(pool_reference):,} ids × {len(configuration.horizon_bands)} bands")
 
     # [3] the challenger is always there to compare with
-    targets = predictions.groupby([ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].apply(set)
+    targets = predictions.groupby([COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].apply(set)
     without_challenger = int(sum(configuration.challenger_technique not in techniques for techniques in targets))
     configuration.log_check(STEP_LABEL, check_log, "the challenger was predicted wherever another technique was",
                             without_challenger == 0,
@@ -378,7 +378,7 @@
     configuration.log_check(STEP_LABEL, check_log, "the ranking compares the techniques on common targets",
                             len(common) > 0,
                             failure_detail="no target where every technique competed: the ranking would mix different targets",
-                            context=f"{len(common):,} of {predictions.groupby([ESTIMATION_ID_COLUMN, 'mes_objetivo', 'h']).ngroups:,} "
+                            context=f"{len(common):,} of {predictions.groupby([COMPOSITION_ID_COLUMN, 'mes_objetivo', 'h']).ngroups:,} "
                                     f"targets have every technique; the others lack a history of 24 months")
 
     # [5] the band holds what it promises in the exam
@@ -394,8 +394,8 @@
 
 def common_targets(predictions: pd.DataFrame) -> pd.DataFrame:
     """The (id, target, horizon) where every technique of the catalogue competed."""
-    techniques_per_target = predictions.groupby([ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].nunique()
-    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[[ESTIMATION_ID_COLUMN, "mes_objetivo", "h"]]
+    techniques_per_target = predictions.groupby([COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].nunique()
+    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[[COMPOSITION_ID_COLUMN, "mes_objetivo", "h"]]
 
 
 def log_backtest_report(predictions: pd.DataFrame, decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
@@ -405,7 +405,7 @@
                                             "as close as chance allows; same targets for every technique):")
     selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
     common = common_targets(predictions)
-    selection = selection.merge(common, on=[ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])
+    selection = selection.merge(common, on=[COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])
     ranking = (selection.assign(abs_norm=selection["err_norm"].abs(), abs_pp=selection["err_pp"].abs())
                .groupby(["tramo_h", "tecnica"]).agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"),
                                                     predicciones=("abs_norm", "size"))
@@ -413,15 +413,15 @@
     configuration.show_table(ranking)
 
     configuration.logger.doc(f"[{STEP_LABEL}] the chosen technique per band (ids and the money their series predict):")
-    chosen_money = decision.merge(pool_reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
+    chosen_money = decision.merge(pool_reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
     configuration.show_table(chosen_money.groupby(["tramo_h", "tecnica", "tecnica_origen"])
-                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum"))
+                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum"))
                              .reset_index().sort_values(["tramo_h", "usd_por_predecir"], ascending=[True, False]))
 
     configuration.logger.doc(f"[{STEP_LABEL}] precision in the EXAM, per band (mean over the judged ids; err in pp of rate; "
                              f"mejora = challenger's error − chosen's error):")
     configuration.show_table(exam_by_pool.groupby("tramo_h")
-                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
+                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
                                   retador_err_pp=("retador_err_pp_medio", "mean"), mejora_pp=("mejora_pp", "mean"),
                                   elegida_sesgo_pp=("elegida_sesgo_pp", "mean"), dentro_banda=("dentro_banda", "mean"))
                              .reset_index())
```

## step_17_forecast.py

```diff
--- antes/step_17_forecast.py	2026-10-01 06:11:04.147114905 +0000
+++ step_17_forecast.py	2026-10-01 06:13:12.326050133 +0000
@@ -79,7 +79,7 @@
 from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
 from step_14_backtest import band_of_horizon
 from techniques import inverse_logit, logit, predict_logit
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PATH_CONTRACT, PATH_STATISTICAL,
                          PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_COLUMN,
                          RATE_FROM_CELL, RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
                          TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNIT_ID_COLUMN,
@@ -150,6 +150,12 @@
                                             f"${extension[configuration.pipeline_usd_col].sum() if len(extension) else 0:,.0f} due · "
                                             f"${extension['esperado_usd'].sum() if len(extension) else 0:,.0f} expected")
 
+    # the confidence of every row: how sure the forecast is of its rate
+    future["confidence"] = confidence_of_rows(future, context, configuration)
+    configuration.log_action(STEP_LABEL, 6, "confidence (USD expected): " + " · ".join(
+        f"{label} {share:.0%}" for label, share in
+        (future.groupby("confidence")["esperado_usd"].sum() / max(future["esperado_usd"].sum(), 1e-9)).items()))
+
     # [7] the totals
     by_month, by_year = totals(future, fine_table, configuration)
     configuration.log_action(STEP_LABEL, 7, f"{len(by_month)} months · {len(by_year)} years")
@@ -161,8 +167,8 @@
     # [9] the tables
     configuration.log_action(STEP_LABEL, 9, "writing the forecast, the months and the summary")
     forecast_columns = ([period_column, "h", PIPELINE_ORIGIN_COLUMN, SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
-                         ESTIMATION_ID_COLUMN, configuration.pipeline_units_col, configuration.pipeline_usd_col, "origen_tasa",
-                         "tecnica", "tasa", "tasa_baja", "tasa_alta", "via_uplift", "uplift", "uplift_bajo", "uplift_alto",
+                         COMPOSITION_ID_COLUMN, configuration.pipeline_units_col, configuration.pipeline_usd_col, "origen_tasa",
+                         "tecnica", "tasa", "tasa_baja", "tasa_alta", "confidence", "via_uplift", "uplift", "uplift_bajo", "uplift_alto",
                          "esperado_unidades", "esperado_usd", "esperado_usd_bajo", "esperado_usd_alto"]
                         + configuration.rate_series_columns + configuration.extra_revalorizacion
                         + ([configuration.discount_value_column, configuration.discount_bucket_column]
@@ -186,6 +192,29 @@
     return dict(forecast=future, by_month=by_month, by_year=by_year)
 
 
+CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW = "high", "medium", "low"
+
+
+def confidence_of_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> np.ndarray:
+    """The confidence of every future row:
+      high    the half-width of its rate band ≤ confidence_high_band_pp (5), its composition judged by
+              the backtest, and the exam error of the chosen technique for its horizon band ≤
+              confidence_high_exam_pp (5)
+      medium  its composition judged and the half-width of its band ≤ confidence_medium_band_pp (10)
+      low     the rest: a wider band, a composition below the support floor (not judged), or a rate
+              from the mandatory cell or the global rate (no composition)"""
+    band_half_width_pp = 100 * (rows["tasa_alta"] - rows["tasa_baja"]) / 2
+    judged_ids = set(context["pool_reference"].loc[context["pool_reference"]["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN])
+    judged = rows[COMPOSITION_ID_COLUMN].isin(judged_ids) & (rows["origen_tasa"] == RATE_FROM_POOL)
+    exam = context["backtest"]["exam_by_pool"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["elegida_err_pp_medio"]
+    horizon_band = rows["h"].map(lambda horizon: band_of_horizon(int(horizon), configuration.horizon_bands))
+    exam_error = pd.Series(list(zip(rows[COMPOSITION_ID_COLUMN], horizon_band)), index=rows.index).map(exam)
+    high = (judged & (band_half_width_pp <= configuration.confidence_high_band_pp)
+            & (exam_error <= configuration.confidence_high_exam_pp))
+    medium = judged & (band_half_width_pp <= configuration.confidence_medium_band_pp)
+    return np.select([high, medium], [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM], default=CONFIDENCE_LOW)
+
+
 def predict_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> pd.DataFrame:
     """Rate, uplift and expected renewals (with bands) of future rows, of the extract or extended."""
     rows = rows.copy()
@@ -360,10 +389,10 @@
 def pool_predictions(pool_series: pd.DataFrame, backtest: dict, horizons: list, configuration: Config) -> pd.DataFrame:
     """The rate of every estimation id at every future horizon, with the technique of its band,
     learning from every closed month of the id."""
-    decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
+    decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
     truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
     rows = []
-    for estimation_id, monthly in truth.groupby(ESTIMATION_ID_COLUMN):
+    for estimation_id, monthly in truth.groupby(COMPOSITION_ID_COLUMN):
         monthly = monthly.sort_values(configuration.period_col)
         history = logit(monthly[RATE_COLUMN].to_numpy(dtype=float))
         calendar_months = np.array([month.month for month in monthly[configuration.period_col]])
@@ -375,20 +404,20 @@
                 technique = configuration.challenger_technique
                 predicted = predict_logit(technique, history, calendar_months, int(horizon))
             rows.append((estimation_id, int(horizon), technique, float(inverse_logit(predicted))))
-    return pd.DataFrame(rows, columns=[ESTIMATION_ID_COLUMN, "h", "tecnica", "tasa_pool_h"])
+    return pd.DataFrame(rows, columns=[COMPOSITION_ID_COLUMN, "h", "tecnica", "tasa_pool_h"])
 
 
 def rate_of_rows(future: pd.DataFrame, series_estimate: pd.DataFrame, pool_rates: pd.DataFrame, pool_reference: pd.DataFrame,
                  rated_units: pd.DataFrame, backtest: dict, forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
     """The rate of every future row: its group's prediction (moved toward its credibility reference),
     mandatory cell, or global; and its band."""
-    estimate = series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "group_rate", "ref_rate"]]
-    future = future.drop(columns=[column for column in (ESTIMATION_ID_COLUMN, "z", "group_rate", "ref_rate", "tecnica",
+    estimate = series_estimate[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "z", "group_rate", "ref_rate"]]
+    future = future.drop(columns=[column for column in (COMPOSITION_ID_COLUMN, "z", "group_rate", "ref_rate", "tecnica",
                                                         "tasa_pool_h", "tasa_pool", "tasa", "origen_tasa")
                                   if column in future.columns])
     future = future.merge(estimate, on=SERIES_ID_COLUMN, how="left")
-    future = future.merge(pool_rates, on=[ESTIMATION_ID_COLUMN, "h"], how="left")
-    future = future.merge(pool_reference[[ESTIMATION_ID_COLUMN, "tasa_pool"]], on=ESTIMATION_ID_COLUMN, how="left")
+    future = future.merge(pool_rates, on=[COMPOSITION_ID_COLUMN, "h"], how="left")
+    future = future.merge(pool_reference[[COMPOSITION_ID_COLUMN, "tasa_pool"]], on=COMPOSITION_ID_COLUMN, how="left")
 
     # the group's predicted rate, moved toward its credibility reference by (1 − z) of the difference
     # of levels (logit scale): the same blend as step 11, applied to the prediction
```

## step_18_validation.py

```diff
--- antes/step_18_validation.py	2026-10-01 06:11:04.147266031 +0000
+++ step_18_validation.py	2026-10-01 06:11:04.386547113 +0000
@@ -37,7 +37,7 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL,
                          RATE_FROM_POOL, ROLE_PROJECTION, S0_PIPELINE_USD_COLUMN, SERIES_ID_COLUMN, TABLE_VALIDATION,
                          TRUTH_ROLES)
 
@@ -89,8 +89,8 @@
     configuration.log_check(STEP_LABEL, check_log, "every series with money to predict has a rate and a risk level",
                             uncovered.empty and set(with_money[SERIES_ID_COLUMN]) <= forecast_series,
                             failure_detail=f"{len(uncovered)} series without level", context=f"{len(with_money):,} series")
-    pooled_ids = set(results["pool_reference"][ESTIMATION_ID_COLUMN])
-    should_pool = forecast[ESTIMATION_ID_COLUMN].isin(pooled_ids)
+    pooled_ids = set(results["pool_reference"][COMPOSITION_ID_COLUMN])
+    should_pool = forecast[COMPOSITION_ID_COLUMN].isin(pooled_ids)
     configuration.log_check(STEP_LABEL, check_log, "every future row of a series with an estimation id takes its rate from the pool",
                             bool((forecast.loc[should_pool, "origen_tasa"] == RATE_FROM_POOL).all()),
                             failure_detail="rows with a pool that took another rate",
```

## step_19_portfolio_exam.py

```diff
--- antes/step_19_portfolio_exam.py	2026-10-01 06:11:04.195000508 +0000
+++ step_19_portfolio_exam.py	2026-10-01 06:11:04.385780354 +0000
@@ -48,7 +48,7 @@
 from config import Config, join_columns
 from step_14_backtest import band_of_horizon
 from techniques import inverse_logit, logit, predict_logit
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
                          SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM, TABLE_PORTFOLIO_EXAM_SUMMARY,
                          TRUTH_ROLES)
 
@@ -89,11 +89,11 @@
                          & (rated_units[units_due] > 0)].copy()
     closed["_celda"] = join_columns(closed, configuration.business_mandatory_dims)
     real["_celda"] = join_columns(real, configuration.business_mandatory_dims)
-    real = real.merge(series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "credibility_ref_id"]],
+    real = real.merge(series_estimate[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "z", "credibility_ref_id"]],
                       on=SERIES_ID_COLUMN, how="left")
-    decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
+    decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
     pool_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
-    pool_ids = set(pool_truth[ESTIMATION_ID_COLUMN])
+    pool_ids = set(pool_truth[COMPOSITION_ID_COLUMN])
 
     rows, latest_month_used = [], []
     for target_month in exam_months:
@@ -174,8 +174,8 @@
     cell_level = known.groupby("_celda")[renewed].sum() / known.groupby("_celda")[units_due].sum()
     global_level = known[renewed].sum() / known[units_due].sum()
     predicted_pool, pool_level = {}, {}
-    for estimation_id in set(month_rows[ESTIMATION_ID_COLUMN].dropna()) & pool_ids:
-        history = pool_known[pool_known[ESTIMATION_ID_COLUMN] == estimation_id].sort_values(period_column)
+    for estimation_id in set(month_rows[COMPOSITION_ID_COLUMN].dropna()) & pool_ids:
+        history = pool_known[pool_known[COMPOSITION_ID_COLUMN] == estimation_id].sort_values(period_column)
         if len(history) < 3:
             continue
         technique = decision.get((estimation_id, band_name), configuration.challenger_technique)
@@ -188,7 +188,7 @@
         pool_level[estimation_id] = history["renovadas"].sum() / history["vencen"].sum()
     rates = []
     for _, row in month_rows.iterrows():
-        estimation_id = row[ESTIMATION_ID_COLUMN]
+        estimation_id = row[COMPOSITION_ID_COLUMN]
         if estimation_id in predicted_pool:
             rate = predicted_pool[estimation_id]
             reference = reference_level.get(row["credibility_ref_id"], np.nan) if pd.notna(row["credibility_ref_id"]) else np.nan
```

## step_audit.py

```diff
--- antes/step_audit.py	1970-01-01 00:00:00.000000000 +0000
+++ step_audit.py	2026-10-01 06:17:02.438530762 +0000
@@ -0,0 +1,540 @@
+"""
+step_audit.py — The audit tables: the satellites of the core (sff_nucleo) for the drill-down.
+
+The core keeps, for every forecast series, what the report needs. When a number is in doubt,
+these tables explain the whole way it came from, joined to the core by its keys:
+
+  table                          one row per                                  key in the core
+  sff_composition                id of any stage (0 raw … 3 composition)      s10_stage0_id … s10_stage3_id
+  sff_composition_techniques     composition × horizon band × technique       s10_stage3_id
+  sff_credibility                credibility reference                        s11_credibility_ref_id
+  sff_credibility_members        reference × forecast series in its rate      s11_credibility_ref_id
+  sff_series_dynamics            forecast series                              s03_fs_id
+  sff_series_backtest            forecast series × exam month × h × technique s03_fs_id
+  sff_series_technique_summary   forecast series × band × technique           s03_fs_id
+  sff_composition_forecast_all   composition × future horizon × technique     s10_stage3_id
+
+The 4 stages only give the forecast better conditions to predict: the backtest, the exam and the
+forecast are referred back to every forecast series (its own units due, its own renewals).
+
+NOTHING IS COUNTED TWICE: every table is checked against the table it comes from (the checks
+below). A forecast series is in one id per stage, in one composition, and has one reference; a
+reference may hold series of many compositions, so its members are not summed across references.
+
+Actions (logged as they are done):
+  1. sff_composition: every id of every stage                         checks 1-2
+  2. sff_composition_techniques: every technique of every composition check 3
+  3. sff_credibility and its members                                  check 4
+  4. sff_series_dynamics                                              check 5
+  5. sff_series_backtest and its summary                              checks 6-8
+  6. sff_composition_forecast_all                                     check 9
+  7. write the audit tables                                           checks 10-17
+  8. count the checks; stop if any failed
+
+Checks (logged as they are made, numbered, at the level of their status):
+   1. every stage is a partition: the units due of its ids add up to the raw's
+   2. every stage holds every estimable series once
+   3. every composition and band has exactly one chosen technique, the one of step 14
+   4. the support and the rate of every reference, recomputed from its members, are the ones used
+   5. one dynamics row per estimable forecast series
+   6. the series of a composition add up to its exam months (units due and renewed)
+   7. without credibility, the series' predictions add up to the composition's prediction
+   8. every forecast series and band has exactly one chosen technique in its summary
+   9. the chosen technique gives the rate step 17 used (rows with no credibility shift)
+   10-17. the eight tables written and read back
+
+Output: dict of the audit tables · the tables listed above.
+"""
+
+# ─── imports ─────────────────────────────────────────────────────────────────────
+import numpy as np
+import pandas as pd
+
+from config import Config
+from step_11_ladder import credibility_k_detail
+from step_13_dynamics import dynamics_of_one_series
+from step_14_backtest import band_of_horizon
+from techniques import CATALOGUE, inverse_logit, logit, predict_logit
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM, PURPOSE_SELECTION,
+                         RATE_COLUMN, RATE_FROM_POOL, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_COMPOSITION,
+                         TABLE_COMPOSITION_FORECAST_ALL, TABLE_COMPOSITION_TECHNIQUES, TABLE_CREDIBILITY,
+                         TABLE_CREDIBILITY_MEMBERS, TABLE_SERIES_BACKTEST, TABLE_SERIES_DYNAMICS,
+                         TABLE_SERIES_TECHNIQUE_SUMMARY, TECHNIQUE_COMPOSITION_BELOW_FLOOR,
+                         TECHNIQUE_NOT_ENOUGH_HISTORY, TECHNIQUE_TESTED, TRUTH_ROLES)
+
+
+STEP_LABEL = "AUD"
+STEP_NAME = "AUDIT TABLES"
+STEP_PURPOSE = ("build the satellites of the core for the drill-down: every id of every stage, every technique of every "
+                "composition, every credibility reference and its members, the dynamics of every forecast series, the "
+                "backtest of every technique referred back to every forecast series, and the future rate of every "
+                "technique; each one checked against the table it comes from, so nothing is counted twice")
+STEP_ACTIONS = ["sff_composition: every id of every stage (checks 1-2)",
+                "sff_composition_techniques: every technique of every composition (check 3)",
+                "sff_credibility and its members (check 4)",
+                "sff_series_dynamics (check 5)",
+                "sff_series_backtest and its summary (checks 6-8)",
+                "sff_composition_forecast_all (check 9)",
+                "write the audit tables (checks 10-17)",
+                "count the checks; stop if any failed"]
+STEP_OUTPUT = "eight audit tables joined to the core by fs_id, the stage ids and the credibility reference"
+
+UNITS_TOLERANCE = 1e-6
+RATE_TOLERANCE = 1e-9
+PERCENTAGE_POINTS = 100
+TREND_MIN_MONTHS = 12          # a trend is measured with at least a year of history
+SEASONALITY_MIN_MONTHS = 24    # a month effect with at least two of every calendar month
+STAGES = (0, 1, 2, 3)
+
+
+def build_audit_tables(ladder: dict, series_rate: pd.DataFrame, series_estimate: pd.DataFrame,
+                       rated_units: pd.DataFrame, pool_series: pd.DataFrame, pool_reference: pd.DataFrame,
+                       pool_dynamics: pd.DataFrame, backtest: dict, forecast_rows: pd.DataFrame,
+                       configuration: Config) -> dict:
+    """The eight audit tables, checked against their sources and written.
+
+    INPUT:   the ladder (step 10), series_rate (08), series_estimate (11), rated_units (08), the
+             composition series and reference (12), their dynamics (13), the backtest (14) and the
+             forecast rows (17).
+    OUTPUT:  dict(composition, composition_techniques, credibility, credibility_members,
+             series_dynamics, series_backtest, series_technique_summary, composition_forecast_all).
+    RULES:   see the module header.
+    EDGE CASES: a composition below the support floor is not judged by the backtest: its
+             techniques have status composition_below_floor and its series have no backtest rows.
+    """
+    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
+    check_log = []
+    history = real_history(rated_units, configuration)
+    groups = ladder["groups"]
+
+    # [1] every id of every stage
+    composition = composition_table(ladder["stages"], groups, series_estimate, pool_reference, backtest, history,
+                                    forecast_rows, configuration)
+    configuration.log_action(STEP_LABEL, 1, f"{len(composition):,} ids over the 4 stages · "
+                                            f"{int(composition['is_composition'].sum()):,} compositions")
+    check_stages(ladder["stages"], history, configuration, check_log)
+
+    # [2] every technique of every composition
+    techniques = composition_techniques(pool_series, pool_reference, backtest, configuration)
+    configuration.log_action(STEP_LABEL, 2, f"{len(techniques):,} composition × band × technique rows · status "
+                                            f"{techniques['status'].value_counts().to_dict()}")
+    check_chosen_once(techniques, backtest["decision"], [COMPOSITION_ID_COLUMN, "tramo_h"],
+                      "every composition and band has exactly one chosen technique, the one of step 14",
+                      configuration, check_log)
+
+    # [3] the credibility references and their members
+    credibility, members = credibility_tables(groups, ladder["reference_members"], series_rate, configuration)
+    configuration.log_action(STEP_LABEL, 3, f"{len(credibility):,} references · {len(members):,} reference × series rows "
+                                            f"({members['role'].value_counts().to_dict() if len(members) else {}})")
+    check_references(credibility, members, history, configuration, check_log)
+
+    # [4] the dynamics of every forecast series
+    series_dynamics = series_dynamics_table(groups, series_rate, history, pool_dynamics, configuration)
+    configuration.log_action(STEP_LABEL, 4, f"{len(series_dynamics):,} forecast series · measurable "
+                                            f"{series_dynamics['measurable'].value_counts().to_dict()} · "
+                                            f"{int(series_dynamics['differs_from_composition'].sum()):,} differ from their composition")
+    configuration.log_check(STEP_LABEL, check_log, "one dynamics row per estimable forecast series",
+                            len(series_dynamics) == len(groups) and series_dynamics[SERIES_ID_COLUMN].is_unique,
+                            failure_detail=f"{len(series_dynamics):,} rows for {len(groups):,} series")
+
+    # [5] the backtest referred back to every forecast series
+    series_backtest = series_backtest_table(backtest["predictions"], backtest["decision"], groups, series_estimate,
+                                            ladder["reference_members"], pool_series, history, configuration)
+    summary = series_technique_summary(series_backtest)
+    configuration.log_action(STEP_LABEL, 5, f"{len(series_backtest):,} forecast series × exam month × h × technique rows · "
+                                            f"{len(summary):,} series × band × technique")
+    check_series_backtest(series_backtest, backtest["predictions"], summary, configuration, check_log)
+
+    # [6] the future rate of every technique
+    forecast_all = composition_forecast_all(pool_series, backtest["decision"], forecast_rows, configuration)
+    configuration.log_action(STEP_LABEL, 6, f"{len(forecast_all):,} composition × horizon × technique rates")
+    check_forecast_all(forecast_all, forecast_rows, series_estimate, configuration, check_log)
+
+    # [7] the tables
+    configuration.log_action(STEP_LABEL, 7, "writing the audit tables")
+    tables = dict(composition=composition, composition_techniques=techniques, credibility=credibility,
+                  credibility_members=members, series_dynamics=series_dynamics, series_backtest=series_backtest,
+                  series_technique_summary=summary, composition_forecast_all=forecast_all)
+    for name, table_name in (("composition", TABLE_COMPOSITION), ("composition_techniques", TABLE_COMPOSITION_TECHNIQUES),
+                             ("credibility", TABLE_CREDIBILITY), ("credibility_members", TABLE_CREDIBILITY_MEMBERS),
+                             ("series_dynamics", TABLE_SERIES_DYNAMICS), ("series_backtest", TABLE_SERIES_BACKTEST),
+                             ("series_technique_summary", TABLE_SERIES_TECHNIQUE_SUMMARY),
+                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL)):
+        configuration.write_table(STEP_LABEL, check_log, tables[name], table_name)
+
+    # [8] the count of the checks; stop if anything failed
+    configuration.log_action(STEP_LABEL, 8, "counting the checks")
+    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
+    return tables
+
+
+def real_history(rated_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """The real closed months of every forecast series with something due: its units due and renewed."""
+    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
+                       & (rated_units[configuration.pipeline_units_col] > 0)]
+    return real[[SERIES_ID_COLUMN, configuration.period_col, CALENDAR_ROLE_COLUMN,
+                 configuration.pipeline_units_col, configuration.renewed_units_col]].copy()
+
+
+def monthly_support(history: pd.DataFrame, members: pd.DataFrame, key: str, configuration: Config) -> pd.DataFrame:
+    """Per key: support (median of the monthly units due summed over its members), units due, renewed, rate."""
+    joined = history.merge(members, on=SERIES_ID_COLUMN)
+    monthly = joined.groupby([key, configuration.period_col])[[configuration.pipeline_units_col,
+                                                               configuration.renewed_units_col]].sum().reset_index()
+    grouped = monthly.groupby(key)
+    result = pd.DataFrame({"support": grouped[configuration.pipeline_units_col].median(),
+                           "units_due": grouped[configuration.pipeline_units_col].sum(),
+                           "units_renewed": grouped[configuration.renewed_units_col].sum()})
+    result["rate"] = result["units_renewed"] / result["units_due"].where(result["units_due"] > 0)
+    return result
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 1 · THE COMPOSITIONS (every id of every stage)
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def composition_table(stages: pd.DataFrame, groups: pd.DataFrame, series_estimate: pd.DataFrame,
+                      pool_reference: pd.DataFrame, backtest: dict, history: pd.DataFrame, forecast_rows: pd.DataFrame,
+                      configuration: Config) -> pd.DataFrame:
+    """One row per id of any stage: the stages it appears in, its series, support, rate; and, for a
+    composition (stage 3), how it is predicted."""
+    long = pd.concat([pd.DataFrame({SERIES_ID_COLUMN: stages[SERIES_ID_COLUMN], "stage": stage,
+                                    "id": stages[f"stage{stage}_id"]}) for stage in STAGES], ignore_index=True)
+    # an id that repeats in several stages has the same series in all of them: one membership per id
+    membership = long.drop_duplicates(["id", SERIES_ID_COLUMN])[["id", SERIES_ID_COLUMN]]
+    support = monthly_support(history, membership, "id", configuration)
+    table = (long.groupby("id").agg(first_stage=("stage", "min"), last_stage=("stage", "max"))
+             .join(membership.groupby("id").size().rename("series")).join(support).reset_index())
+    table["is_composition"] = (table["last_stage"] == 3).astype(int)
+
+    # the compositions: how they are predicted
+    per_composition = (series_estimate.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
+                       .set_index(COMPOSITION_ID_COLUMN)[["credibility_ref_id", "ref_support", "ref_rate", "k", "z"]])
+    estimated = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["tasa_estimada"]
+    table = table.join(per_composition, on="id").join(estimated.rename("estimated_rate"), on="id")
+    table = table.join(pool_reference.set_index(COMPOSITION_ID_COLUMN)[["gate", "usd_por_predecir"]], on="id")
+    decision = backtest["decision"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="tecnica").add_prefix("technique_")
+    exam = backtest["exam_by_pool"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="elegida_err_pp_medio").add_prefix("exam_err_pp_")
+    table = table.join(decision, on="id").join(exam, on="id")
+    if forecast_rows is not None and len(forecast_rows):
+        expected = forecast_rows.groupby([COMPOSITION_ID_COLUMN, "confidence"])["esperado_usd"].sum().unstack(fill_value=0.0)
+        expected = expected.add_prefix("expected_usd_")
+        table = table.join(expected, on="id")
+    return table.rename(columns={"id": COMPOSITION_ID_COLUMN}).sort_values(["first_stage", COMPOSITION_ID_COLUMN]).reset_index(drop=True)
+
+
+def check_stages(stages: pd.DataFrame, history: pd.DataFrame, configuration: Config, check_log: list) -> None:
+    """Checks 1 and 2: every stage is a partition of the raw."""
+    raw_due = float(history[history[SERIES_ID_COLUMN].isin(stages[SERIES_ID_COLUMN])][configuration.pipeline_units_col].sum())
+    due_by_stage = {}
+    for stage in STAGES:
+        members = stages[[SERIES_ID_COLUMN, f"stage{stage}_id"]].rename(columns={f"stage{stage}_id": "id"})
+        due_by_stage[stage] = float(monthly_support(history, members, "id", configuration)["units_due"].sum())
+    configuration.log_check(STEP_LABEL, check_log, "every stage is a partition: the units due of its ids add up to the raw's",
+                            all(abs(due - raw_due) <= UNITS_TOLERANCE for due in due_by_stage.values()),
+                            failure_detail=f"units due by stage {due_by_stage} vs {raw_due:,.0f}",
+                            context=f"{raw_due:,.0f} units due in every stage")
+    configuration.log_check(STEP_LABEL, check_log, "every stage holds every estimable series once",
+                            stages[SERIES_ID_COLUMN].is_unique and stages[[f"stage{stage}_id" for stage in STAGES]].notna().all().all(),
+                            failure_detail="a series without an id in some stage, or twice",
+                            context=f"{len(stages):,} series")
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 2 · THE TECHNIQUES OF EVERY COMPOSITION
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def composition_techniques(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict,
+                           configuration: Config) -> pd.DataFrame:
+    """Composition × band × technique: status and why, months, errors in selection and exam, rank, chosen."""
+    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()]
+    months_available = truth.groupby(COMPOSITION_ID_COLUMN).size()
+    gate = pool_reference.set_index(COMPOSITION_ID_COLUMN)["gate"]
+    predictions = backtest["predictions"].assign(abs_err_pp=lambda frame: frame["err_pp"].abs(),
+                                                 abs_err_norm=lambda frame: frame["err_norm"].abs())
+    by_purpose = (predictions.groupby([COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "proposito"])
+                  [["abs_err_pp", "abs_err_norm"]].mean().unstack("proposito"))
+    by_purpose.columns = [f"{purpose}_{measure}" for measure, purpose in by_purpose.columns]
+    chosen = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
+
+    rows = []
+    for composition_id in pool_reference[COMPOSITION_ID_COLUMN]:
+        for band_name in configuration.horizon_bands:
+            for technique_id, technique in CATALOGUE.items():
+                required = int(technique[3])
+                available = int(months_available.get(composition_id, 0))
+                if gate.get(composition_id) != GATE_LEVEL:
+                    status = TECHNIQUE_COMPOSITION_BELOW_FLOOR
+                elif available < required:
+                    status = TECHNIQUE_NOT_ENOUGH_HISTORY
+                else:
+                    status = TECHNIQUE_TESTED
+                rows.append({COMPOSITION_ID_COLUMN: composition_id, "tramo_h": band_name, "tecnica": technique_id,
+                             "status": status, "months_available": available, "months_required": required,
+                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
+    table = pd.DataFrame(rows).join(by_purpose, on=[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
+    table = table.rename(columns={f"{PURPOSE_SELECTION}_abs_err_norm": "selection_err_norm",
+                                  f"{PURPOSE_SELECTION}_abs_err_pp": "selection_err_pp",
+                                  f"{PURPOSE_EXAM}_abs_err_norm": "exam_err_norm",
+                                  f"{PURPOSE_EXAM}_abs_err_pp": "exam_err_pp"})
+    tested = table["status"] == TECHNIQUE_TESTED
+    table["rank"] = np.nan
+    table.loc[tested, "rank"] = (table[tested].groupby([COMPOSITION_ID_COLUMN, "tramo_h"])["selection_err_norm"]
+                                 .rank(method="min"))
+    return table
+
+
+def check_chosen_once(table: pd.DataFrame, decision: pd.DataFrame, keys: list, name: str, configuration: Config,
+                      check_log: list) -> None:
+    """Exactly one chosen technique per key, and it is the technique of step 14's decision."""
+    chosen = table[table["is_chosen"] == 1]
+    per_key = chosen.groupby(keys).size()
+    matches = chosen.merge(decision[keys + ["tecnica"]], on=keys, suffixes=("", "_decision"))
+    configuration.log_check(STEP_LABEL, check_log, name,
+                            bool((per_key == 1).all()) and len(per_key) == len(decision)
+                            and bool((matches["tecnica"] == matches["tecnica_decision"]).all()),
+                            failure_detail=f"{int((per_key != 1).sum())} keys without exactly one chosen · "
+                                           f"{len(per_key)} keys with a choice for {len(decision)} decisions",
+                            context=f"{len(decision):,} composition × band")
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 3 · THE CREDIBILITY
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def credibility_tables(groups: pd.DataFrame, reference_members: pd.DataFrame, series_rate: pd.DataFrame,
+                       configuration: Config) -> tuple:
+    """One row per reference (support, rate, k and its parts), and one per reference × forecast series."""
+    references = (groups.dropna(subset=["credibility_ref_id"])
+                  .drop_duplicates("credibility_ref_id")[["credibility_ref_id", "credibility_ref_step", "ref_series",
+                                                          "ref_support", "ref_rate"]])
+    k_detail = credibility_k_detail(groups, configuration)
+    credibility = references.merge(k_detail, on="credibility_ref_id", how="left")
+    credibility["compositions_using"] = credibility["compositions_using"].fillna(0).astype(int)
+    credibility["k"] = credibility["k"].fillna(configuration.k_cred)
+    credibility["k_source"] = credibility["k_source"].fillna("default")
+
+    composition_of = groups.set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN]
+    reference_of_composition = groups.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["credibility_ref_id"]
+    members = reference_members.merge(series_rate[[SERIES_ID_COLUMN, "n_propio", "tasa_propia"]], on=SERIES_ID_COLUMN, how="left")
+    members[COMPOSITION_ID_COLUMN] = members[SERIES_ID_COLUMN].map(composition_of)
+    borrows = members[COMPOSITION_ID_COLUMN].map(reference_of_composition) == members["credibility_ref_id"]
+    members["role"] = np.where(borrows, "borrower", "lender")
+    members = members.rename(columns={"n_propio": "support", "tasa_propia": "own_rate"})
+    return credibility, members
+
+
+def check_references(credibility: pd.DataFrame, members: pd.DataFrame, history: pd.DataFrame, configuration: Config,
+                     check_log: list) -> None:
+    """Check 4: every reference recomputed from its members."""
+    if credibility.empty:
+        configuration.log_not_evaluated(STEP_LABEL, check_log, "every reference recomputed from its members", "no reference")
+        return
+    recomputed = monthly_support(history, members[["credibility_ref_id", SERIES_ID_COLUMN]], "credibility_ref_id", configuration)
+    compared = credibility.join(recomputed, on="credibility_ref_id")
+    compared["members"] = compared["credibility_ref_id"].map(members.groupby("credibility_ref_id").size())
+    wrong = compared[((compared["support"] - compared["ref_support"]).abs() > UNITS_TOLERANCE)
+                     | ((compared["rate"] - compared["ref_rate"]).abs() > RATE_TOLERANCE)
+                     | (compared["members"] != compared["ref_series"])]
+    configuration.log_check(STEP_LABEL, check_log, "the support, rate and series of every reference, recomputed from its members, "
+                            "are the ones used", wrong.empty, failure_detail=f"{len(wrong)} references differ",
+                            context=f"{len(credibility):,} references",
+                            examples=wrong[["credibility_ref_id", "ref_support", "support", "ref_rate", "rate"]])
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 4 · THE DYNAMICS OF EVERY FORECAST SERIES
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def series_dynamics_table(groups: pd.DataFrame, series_rate: pd.DataFrame, history: pd.DataFrame,
+                          pool_dynamics: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """φ, trend and seasonality of every estimable forecast series on its own months, whether they can
+    be trusted, and how they compare with its composition's."""
+    support = series_rate.set_index(SERIES_ID_COLUMN)["n_propio"]
+    monthly = history.rename(columns={configuration.pipeline_units_col: "vencen", configuration.renewed_units_col: "renovadas"})
+    composition_dynamics = (pool_dynamics.set_index(COMPOSITION_ID_COLUMN)[["tendencia", "estacional"]]
+                            if pool_dynamics is not None and len(pool_dynamics) else pd.DataFrame(columns=["tendencia", "estacional"]))
+    months_of_series = {series_id: block.sort_values(configuration.period_col)
+                        for series_id, block in monthly.groupby(SERIES_ID_COLUMN)}   # one pass over the history
+    no_months = monthly.iloc[0:0]
+    rows = []
+    for series_id, composition_id in zip(groups[SERIES_ID_COLUMN], groups[COMPOSITION_ID_COLUMN]):
+        months = months_of_series.get(series_id, no_months)
+        own_support = float(support.get(series_id, 0.0))
+        row = {SERIES_ID_COLUMN: series_id, COMPOSITION_ID_COLUMN: composition_id, "months": len(months),
+               "own_support": own_support}
+        if len(months) < TREND_MIN_MONTHS:
+            row["measurable"] = "short_history"
+        else:
+            row["measurable"] = "yes" if own_support >= configuration.support_floor else "low_support"
+            measured = dynamics_of_one_series(months, configuration.period_col, configuration)
+            row.update(phi=measured["phi"], trend=measured["tendencia"], trend_pp_year=measured["tendencia_pp_ano"],
+                       trend_p_value=measured["p_valor_tendencia"])
+            if len(months) >= SEASONALITY_MIN_MONTHS:
+                row.update(seasonal=int(measured["estacional"]), seasonal_p_value=measured["p_valor_mes"],
+                           amplitude_pp=measured["amplitud_pp"], high_months=measured["meses_alto"],
+                           low_months=measured["meses_bajo"])
+        if composition_id in composition_dynamics.index:
+            row["composition_trend"] = composition_dynamics.loc[composition_id, "tendencia"]
+            row["composition_seasonal"] = int(composition_dynamics.loc[composition_id, "estacional"])
+        rows.append(row)
+    table = pd.DataFrame(rows)
+    for column_name in ("phi", "trend", "trend_pp_year", "trend_p_value", "seasonal", "seasonal_p_value", "amplitude_pp",
+                        "high_months", "low_months", "composition_trend", "composition_seasonal"):
+        if column_name not in table.columns:
+            table[column_name] = np.nan
+    # a series that behaves differently from its composition (only when both are measured and trusted)
+    trusted = (table["measurable"] == "yes") & table["composition_trend"].notna()
+    different_trend = table["trend"].fillna(0).ne(0) & table["trend"].ne(table["composition_trend"])
+    different_season = table["seasonal"].notna() & table["seasonal"].ne(table["composition_seasonal"])
+    table["differs_from_composition"] = (trusted & (different_trend | different_season)).astype(int)
+    return table
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 5 · THE BACKTEST REFERRED BACK TO EVERY FORECAST SERIES
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def series_backtest_table(predictions: pd.DataFrame, decision: pd.DataFrame, groups: pd.DataFrame,
+                          series_estimate: pd.DataFrame, reference_members: pd.DataFrame, pool_series: pd.DataFrame,
+                          history: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """Every exam prediction of a composition applied to each of its forecast series: the composition's
+    rate (moved toward the reference by 1 − z, levels known at the origin) × the series' own units due,
+    against what the series really renewed."""
+    exam = predictions[predictions["proposito"] == PURPOSE_EXAM].copy()
+    exam["origin"] = exam["ultimo_mes_visto"]
+    series_of = groups[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]].merge(
+        series_estimate[[SERIES_ID_COLUMN, "z", "credibility_ref_id"]], on=SERIES_ID_COLUMN, how="left")
+    rows = exam.merge(series_of, on=COMPOSITION_ID_COLUMN)
+    real = history.rename(columns={configuration.period_col: "mes_objetivo", configuration.pipeline_units_col: "due_units",
+                                   configuration.renewed_units_col: "real_units"})[[SERIES_ID_COLUMN, "mes_objetivo", "due_units", "real_units"]]
+    rows = rows.merge(real, on=[SERIES_ID_COLUMN, "mes_objetivo"])
+
+    # the levels known at the origin: the composition's and its reference's
+    composition_level = levels_at_origins(pool_series[pool_series[RATE_COLUMN].notna()].rename(columns={"vencen": "_due", "renovadas": "_renewed"}),
+                                          COMPOSITION_ID_COLUMN, rows[[COMPOSITION_ID_COLUMN, "origin"]].drop_duplicates(), configuration)
+    reference_history = history.merge(reference_members, on=SERIES_ID_COLUMN).rename(
+        columns={configuration.pipeline_units_col: "_due", configuration.renewed_units_col: "_renewed"})
+    reference_level = levels_at_origins(reference_history, "credibility_ref_id",
+                                        rows[["credibility_ref_id", "origin"]].dropna().drop_duplicates(), configuration)
+    rows = rows.merge(composition_level.rename(columns={"level": "composition_level"}), on=[COMPOSITION_ID_COLUMN, "origin"], how="left")
+    rows = rows.merge(reference_level.rename(columns={"level": "reference_level"}), on=["credibility_ref_id", "origin"], how="left")
+
+    shift = np.zeros(len(rows))
+    if configuration.apply_credibility_shift:
+        moves = (rows["z"] < 1) & rows["reference_level"].between(0, 1, inclusive="neither") & rows["composition_level"].between(0, 1, inclusive="neither")
+        shift = np.where(moves, (1 - rows["z"]) * (logit(rows["reference_level"].clip(1e-6, 1 - 1e-6))
+                                                    - logit(rows["composition_level"].clip(1e-6, 1 - 1e-6))), 0.0)
+    rows["pred_rate"] = inverse_logit(logit(rows["tasa_pred"]) + shift)
+    rows["real_rate"] = rows["real_units"] / rows["due_units"]
+    rows["pred_units"] = rows["pred_rate"] * rows["due_units"]
+    rows["err_units"] = rows["pred_units"] - rows["real_units"]
+    rows["err_pp"] = PERCENTAGE_POINTS * (rows["pred_rate"] - rows["real_rate"])
+    rows["se_binom_pp"] = PERCENTAGE_POINTS * np.sqrt(rows["pred_rate"] * (1 - rows["pred_rate"]) / rows["due_units"])
+    rows["err_norm"] = rows["err_pp"] / rows["se_binom_pp"].where(rows["se_binom_pp"] > 0)
+    chosen = decision.set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
+    rows["is_chosen"] = (pd.Series(list(zip(rows[COMPOSITION_ID_COLUMN], rows["tramo_h"])), index=rows.index).map(chosen)
+                         == rows["tecnica"]).astype(int)
+    rows["shifted_by_credibility"] = (np.abs(shift) > 0).astype(int)
+    return rows[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "mes_objetivo", "h", "tramo_h", "origin", "tecnica", "is_chosen",
+                 "shifted_by_credibility", "tasa_pred", "pred_rate", "real_rate", "due_units", "pred_units", "real_units",
+                 "err_units", "err_pp", "se_binom_pp", "err_norm"]]
+
+
+def levels_at_origins(history: pd.DataFrame, key: str, wanted: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """The rate (Σ renewed / Σ due) of every key with what was known at every origin it needs."""
+    rows = []
+    history_of_key = dict(tuple(history.groupby(key)))                              # one pass over the history
+    for key_value, origins in wanted.groupby(key)["origin"]:
+        own = history_of_key.get(key_value, history.iloc[0:0])
+        for origin in origins:
+            known = own[own[configuration.period_col] <= origin]
+            due = known["_due"].sum()
+            rows.append({key: key_value, "origin": origin, "level": known["_renewed"].sum() / due if due > 0 else np.nan})
+    return pd.DataFrame(rows, columns=[key, "origin", "level"])
+
+
+def series_technique_summary(series_backtest: pd.DataFrame) -> pd.DataFrame:
+    """Forecast series × band × technique: errors, bias, WAPE, rank within the series, chosen, best."""
+    grouped = series_backtest.assign(abs_err_pp=series_backtest["err_pp"].abs(), abs_err_units=series_backtest["err_units"].abs(),
+                                     abs_err_norm=series_backtest["err_norm"].abs()).groupby([SERIES_ID_COLUMN, "tramo_h", "tecnica"])
+    summary = pd.DataFrame({"predictions": grouped.size(), "mean_abs_err_pp": grouped["abs_err_pp"].mean(),
+                            "bias_pp": grouped["err_pp"].mean(), "mean_abs_err_norm": grouped["abs_err_norm"].mean(),
+                            "wape": grouped["abs_err_units"].sum() / grouped["real_units"].sum().where(grouped["real_units"].sum() > 0),
+                            "is_chosen": grouped["is_chosen"].max(),
+                            COMPOSITION_ID_COLUMN: grouped[COMPOSITION_ID_COLUMN].first()}).reset_index()
+    summary["rank"] = summary.groupby([SERIES_ID_COLUMN, "tramo_h"])["mean_abs_err_norm"].rank(method="min")
+    summary["best_for_series"] = (summary["rank"] == 1).astype(int)
+    chosen_rank = summary[summary["is_chosen"] == 1].set_index([SERIES_ID_COLUMN, "tramo_h"])["rank"]
+    summary["chosen_rank"] = pd.Series(list(zip(summary[SERIES_ID_COLUMN], summary["tramo_h"])), index=summary.index).map(chosen_rank)
+    return summary
+
+
+def check_series_backtest(series_backtest: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame,
+                          configuration: Config, check_log: list) -> None:
+    """Checks 6 to 8: the series add up to their composition; one chosen technique per series and band."""
+    exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
+    keys = [COMPOSITION_ID_COLUMN, "mes_objetivo", "h", "tecnica"]
+    summed = series_backtest.groupby(keys)[["due_units", "real_units", "pred_units"]].sum()
+    shifted = series_backtest.groupby(keys)["shifted_by_credibility"].max()
+    compared = exam.set_index(keys)[["vencen_real", "tasa_real", "tasa_pred"]].join(summed, how="inner").join(shifted)
+    adds_up = (((compared["due_units"] - compared["vencen_real"]).abs() <= UNITS_TOLERANCE)
+               & ((compared["real_units"] - compared["tasa_real"] * compared["vencen_real"]).abs() <= UNITS_TOLERANCE))
+    configuration.log_check(STEP_LABEL, check_log, "the series of a composition add up to its exam months (units due and renewed)",
+                            bool(adds_up.all()) and len(compared) == len(exam),
+                            failure_detail=f"{int((~adds_up).sum())} composition × month × h × technique differ · "
+                                           f"{len(compared)} of {len(exam)} exam predictions covered",
+                            context=f"{len(compared):,} composition × month × h × technique")
+    unshifted = compared[compared["shifted_by_credibility"] == 0]
+    same_prediction = ((unshifted["pred_units"] - unshifted["tasa_pred"] * unshifted["vencen_real"]).abs() <= UNITS_TOLERANCE)
+    configuration.log_check(STEP_LABEL, check_log, "without credibility, the series' predictions add up to the composition's prediction",
+                            bool(same_prediction.all()),
+                            failure_detail=f"{int((~same_prediction).sum())} differ", context=f"{len(unshifted):,} checked")
+    per_series_band = summary.groupby([SERIES_ID_COLUMN, "tramo_h"])["is_chosen"].sum()
+    configuration.log_check(STEP_LABEL, check_log, "every forecast series and band has exactly one chosen technique in its summary",
+                            bool((per_series_band == 1).all()),
+                            failure_detail=f"{int((per_series_band != 1).sum())} series × band without exactly one",
+                            context=f"{len(per_series_band):,} series × band")
+
+
+# ═══════════════════════════════════════════════════════════════════════════════════
+# 6 · THE FUTURE RATE OF EVERY TECHNIQUE
+# ═══════════════════════════════════════════════════════════════════════════════════
+
+def composition_forecast_all(pool_series: pd.DataFrame, decision: pd.DataFrame, forecast_rows: pd.DataFrame,
+                             configuration: Config) -> pd.DataFrame:
+    """Composition × future horizon × technique: the rate each technique gives with the whole history
+    (before the credibility shift), and which one was chosen for that horizon's band."""
+    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()
+                        & (pool_series["vencen"] > 0)]
+    horizons = sorted(int(horizon) for horizon in forecast_rows["h"].dropna().unique()) if forecast_rows is not None else []
+    chosen = decision.set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
+    rows = []
+    for composition_id, monthly in truth.groupby(COMPOSITION_ID_COLUMN):
+        monthly = monthly.sort_values(configuration.period_col)
+        history_logit = logit(monthly[RATE_COLUMN].to_numpy(dtype=float))
+        months = np.array([month.month for month in monthly[configuration.period_col]])
+        for horizon in horizons:
+            band_name = band_of_horizon(horizon, configuration.horizon_bands)
+            for technique_id, technique in CATALOGUE.items():
+                enough = len(monthly) >= int(technique[3])
+                value = predict_logit(technique_id, history_logit, months, horizon) if enough else np.nan
+                rows.append({COMPOSITION_ID_COLUMN: composition_id, "h": horizon, "tramo_h": band_name,
+                             "tecnica": technique_id, "months_available": len(monthly),
+                             "rate": float(inverse_logit(value)) if np.isfinite(value) else np.nan,
+                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
+    return pd.DataFrame(rows)
+
+
+def check_forecast_all(forecast_all: pd.DataFrame, forecast_rows: pd.DataFrame, series_estimate: pd.DataFrame,
+                       configuration: Config, check_log: list) -> None:
+    """Check 9: the chosen technique gives the rate step 17 used, on the rows with no credibility shift."""
+    z_of = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["z"]
+    rows = forecast_rows[(forecast_rows["origen_tasa"] == RATE_FROM_POOL)]
+    rows = rows[rows[COMPOSITION_ID_COLUMN].map(z_of).fillna(1) >= 1]
+    chosen = forecast_all[forecast_all["is_chosen"] == 1].set_index([COMPOSITION_ID_COLUMN, "h"])["rate"]
+    expected = pd.Series(list(zip(rows[COMPOSITION_ID_COLUMN], rows["h"].astype(int))), index=rows.index).map(chosen)
+    differs = (expected - rows[RATE_COLUMN]).abs() > RATE_TOLERANCE
+    configuration.log_check(STEP_LABEL, check_log, "the chosen technique gives the rate step 17 used (rows with no credibility shift)",
+                            not bool(differs.any()), failure_detail=f"{int(differs.sum())} future rows differ",
+                            context=f"{len(rows):,} future rows compared")
```

## step_informe.py

```diff
--- antes/step_informe.py	2026-10-01 06:11:04.146725926 +0000
+++ step_informe.py	2026-10-01 06:18:30.450161393 +0000
@@ -40,7 +40,7 @@
 import pandas as pd
 
 from config import Config, UNKNOWN_DISCOUNT_BUCKET
-from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, LEVEL_OWN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, LEVEL_OWN,
                          METHOD_FRAMEWORK, PURPOSE_SELECTION, REPORT_FILE_NAME, ROLES_IN_ORDER, ROLE_PROJECTION,
                          S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN,
                          SYNTHETIC_COLUMN, TABLE_SERIES_CARD, TOTAL_ORIGIN_TOTAL, TRUTH_ROLES, UPLIFT_CELL_ID_COLUMN)
@@ -151,16 +151,16 @@
                                             on=SERIES_ID_COLUMN, how="left")
     dynamics = results.get("pool_dynamics")
     if dynamics is not None and len(dynamics):
-        card = card.merge(dynamics[[ESTIMATION_ID_COLUMN, "phi", "tendencia", "tendencia_pp_ano", "estacional",
-                                    "amplitud_pp", "meses_alto", "meses_bajo"]], on=ESTIMATION_ID_COLUMN, how="left")
+        card = card.merge(dynamics[[COMPOSITION_ID_COLUMN, "phi", "tendencia", "tendencia_pp_ano", "estacional",
+                                    "amplitud_pp", "meses_alto", "meses_bajo"]], on=COMPOSITION_ID_COLUMN, how="left")
     backtest = results.get("backtest")
     if backtest is not None:
-        per_band = backtest["decision"][[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]].merge(
-            backtest["exam_by_pool"][[ESTIMATION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio"]],
-            on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
-        wide = per_band.pivot(index=ESTIMATION_ID_COLUMN, columns="tramo_h")
+        per_band = backtest["decision"][[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]].merge(
+            backtest["exam_by_pool"][[COMPOSITION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio"]],
+            on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
+        wide = per_band.pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h")
         wide.columns = [f"{name}_{band}" for name, band in wide.columns]
-        card = card.merge(wide, left_on=ESTIMATION_ID_COLUMN, right_index=True, how="left")
+        card = card.merge(wide, left_on=COMPOSITION_ID_COLUMN, right_index=True, how="left")
     card["error_estimacion_pp"] = configuration.z * card["se_estimacion_pp"]
     return card
 
@@ -268,13 +268,16 @@
              "su tasa se conoce a ±5 pp y puede ir sola. **Antes**: cada serie con su propio soporte. **Después**: la "
              "escalera junta las series pasada a pasada (signo, extras y dimensiones mandatory en el orden de colapso) hasta "
              "que cada grupo llega a 30; el grupo presta su tasa a sus series y, por debajo de 271, la mezcla con la de una "
-             "referencia más amplia por credibilidad. Ver `DOC_escalera.md`.", "",
+             "referencia más amplia por credibilidad (etapa 4). Ver `DOC_escalera.md` y `DOC_modelo_datos.md`.", "",
              "**Antes · el dinero por soporte propio (el dial):**", "", markdown_table(before),
              "**Antes y después · el error con el que se CONOCE la tasa de cada serie** (antes: su error binomial con su "
              "propio soporte; después: el error de la estimación de la escalera). La predicción de un mes concreto conserva "
              "además el ruido de su propio tamaño, que ninguna escalera elimina: está en el nivel de riesgo.", "",
              markdown_table(comparison, 1),
-             "**Pasada a pasada · cómo mejora el soporte** (cada pasada es un reparto: las unidades que vencen suman lo "
+             "**Etapa a etapa · cómo mejora el soporte** (0 raw · 1 signo · 2 extras · 3 colapso = la composición con la que "
+             "se predice; agrupando por el id de cada etapa, las unidades que vencen suman lo mismo):", "",
+             markdown_table(results["ladder"]["stage_summary"], 2) if results.get("ladder") else "",
+             "**Pasada a pasada · el detalle dentro de cada etapa** (cada pasada es un reparto: las unidades que vencen suman lo "
              "mismo en todas; los grupos son menos y más grandes; pct_usd_floor / pct_usd_own_rate: dinero por predecir en "
              "grupos que llegan a 30 / a 271):", "",
              markdown_table(results["ladder"]["summary"], 2) if results.get("ladder") else "",
@@ -303,7 +306,7 @@
     if dynamics is not None and len(dynamics):
         lines += ["**Los pools con soporte, uno a uno:**", "",
                   markdown_table(dynamics.sort_values("usd_por_predecir", ascending=False).head(TOP_ROWS)
-                                 [[ESTIMATION_ID_COLUMN, "meses", "phi", "tendencia_pp_ano", "estacional", "amplitud_pp",
+                                 [[COMPOSITION_ID_COLUMN, "meses", "phi", "tendencia_pp_ano", "estacional", "amplitud_pp",
                                    "meses_alto", "meses_bajo", "usd_por_predecir"]])]
     return "\n".join(lines) + "\n"
 
@@ -319,7 +322,7 @@
     judged_share = (reference.loc[reference["gate"] == GATE_LEVEL, "usd_por_predecir"].sum()
                     / max(reference["usd_por_predecir"].sum(), 1))
 
-    exam_by_band = exam_by_pool.merge(reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
+    exam_by_band = exam_by_pool.merge(reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
     band_rows = []
     for band_name, rows in exam_by_band.groupby("tramo_h"):
         weights = rows["usd_por_predecir"] + 1e-9
@@ -330,9 +333,9 @@
                           "dentro_banda": rows["dentro_banda"].mean()})
     by_band = pd.DataFrame(band_rows)
 
-    card = results["series_estimate"][[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "nivel_riesgo", "usd_por_predecir"]]
+    card = results["series_estimate"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "nivel_riesgo", "usd_por_predecir"]]
     short = exam_by_pool[exam_by_pool["tramo_h"] == list(configuration.horizon_bands)[0]]
-    by_level = card.merge(short[[ESTIMATION_ID_COLUMN, "elegida_err_pp_medio", "retador_err_pp_medio"]], on=ESTIMATION_ID_COLUMN, how="left")
+    by_level = card.merge(short[[COMPOSITION_ID_COLUMN, "elegida_err_pp_medio", "retador_err_pp_medio"]], on=COMPOSITION_ID_COLUMN, how="left")
     by_level = (by_level.dropna(subset=["elegida_err_pp_medio"]).groupby("nivel_riesgo")
                 .apply(lambda rows: pd.Series({"series": int(len(rows)), "usd_por_predecir": rows["usd_por_predecir"].sum(),
                                                "error_elegida_pp": np.average(rows["elegida_err_pp_medio"], weights=rows["usd_por_predecir"] + 1e-9),
```

## step_nucleo.py

```diff
--- antes/step_nucleo.py	2026-10-01 06:11:04.088963837 +0000
+++ step_nucleo.py	2026-10-01 06:13:52.444757383 +0000
@@ -53,7 +53,7 @@
 import pandas as pd
 
 from config import Config, join_columns
-from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, ESTIMATION_ID_COLUMN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, COMPOSITION_ID_COLUMN,
                          FINE_ROWS_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_SIMULATED, ROLE_TEST,
                          ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
                          S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN,
@@ -109,6 +109,7 @@
                       ("tasa", "s17_tasa", "17", "the predicted rate of the row (units)"),
                       ("tasa_baja", "s17_tasa_baja", "17", "low end of the rate band"),
                       ("tasa_alta", "s17_tasa_alta", "17", "high end of the rate band"),
+                      ("confidence", "s17_confidence", "17", "high · medium · low: how sure the forecast is of the rate of the row"),
                       ("via_uplift", "s15_via_uplift", "15", "contrato (1/(1 − d)) · estadistica (its cell)"),
                       ("uplift", "s15_uplift", "15", "the uplift applied to the row"),
                       ("esperado_unidades", "s17_esperado_unidades", "17", "expected renewed units (SUM)"),
@@ -129,14 +130,22 @@
                          ("error_binomial_pp", "s08_error_binomial_pp", "08", "Wilson half-width of one month at n_propio (pp)"),
                          (SIGN_COLUMN, "s08_signo", "08", "neutro · negativo · positivo · mixto"),
                          ("bajo_suelo", "s08_bajo_suelo", "08", "1 when n_propio is below support_floor")]
-SERIES_VALUES_STEP_11 = [(ESTIMATION_ID_COLUMN, "s11_final_group_id", "11", "the final group of the series in the ladder: it lends its rate ('*' = collapsed)"),
-                         ("final_step", "s11_final_step", "11", "the pass of the ladder where its group was formed (0 = itself)"),
-                         ("group_series", "s11_group_series", "11", "series in its final group"),
-                         ("group_support", "s11_group_support", "11", "support of its final group (units due in a typical month)"),
-                         ("group_rate", "s11_group_rate", "11", "rate of its final group"),
+# step 10, per forecast series: the 4 stages of the ladder (grouping by any stage id adds up to the raw)
+SERIES_VALUES_STEP_10 = [("stage0_id", "s10_stage0_id", "10", "stage 0 · raw: the forecast series itself"),
+                         ("stage0_support", "s10_stage0_support", "10", "stage 0 · its own support (units due in a typical month)"),
+                         ("stage1_id", "s10_stage1_id", "10", "stage 1 · sign: its group after the signs are merged (repeats stage 0 if nothing changed)"),
+                         ("stage1_support", "s10_stage1_support", "10", "stage 1 · the support of that group"),
+                         ("stage2_id", "s10_stage2_id", "10", "stage 2 · extras: its group after the extras are annulled"),
+                         ("stage2_support", "s10_stage2_support", "10", "stage 2 · the support of that group"),
+                         ("stage3_id", "s10_stage3_id", "10", "stage 3 · collapse: its COMPOSITION, the group that is predicted"),
+                         ("stage3_support", "s10_stage3_support", "10", "stage 3 · the support of its composition")]
+# step 11, per forecast series: the rate of its composition and the credibility (stage 4)
+SERIES_VALUES_STEP_11 = [("group_series", "s11_composition_series", "11", "forecast series in its composition"),
+                         ("group_rate", "s11_composition_rate", "11", "rate of its composition"),
                          ("credibility_ref_id", "s11_credibility_ref_id", "11", "the wider group its rate is blended with (below own_rate_floor)"),
                          ("ref_support", "s11_ref_support", "11", "support of the credibility reference"),
                          ("ref_rate", "s11_ref_rate", "11", "rate of the credibility reference"),
+                         ("credibility_effect_pp", "s11_credibility_effect_pp", "11", "how much the credibility moves the rate of its composition (pp); 0 when it predicts alone"),
                          ("alcanzo_suelo", "s11_alcanzo_suelo", "11", "1 when its final group reaches the support floor"),
                          ("k", "s11_k", "11", "Bühlmann k of the reference"),
                          ("z", "s11_z", "11", "credibility of the group's own rate: n / (n + k); 1 without a reference"),
@@ -166,7 +175,8 @@
                      series_estimate: pd.DataFrame = None, pool_dynamics: pd.DataFrame = None,
                      technique_decision: pd.DataFrame = None, exam_by_pool: pd.DataFrame = None,
                      forecast: pd.DataFrame = None, time_series_rows: pd.DataFrame = None,
-                     time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None) -> tuple:
+                     time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None,
+                     ladder_stages: pd.DataFrame = None) -> tuple:
     """The core table with every block whose step has run, and its legend; checked and written.
 
     INPUT:   the fine table and the results of every step that has run (None = not run).
@@ -209,18 +219,19 @@
     configuration.log_action(STEP_LABEL, 3, f"unit blocks added: {[specs[0][2] for specs in blocks_present] or 'none'}")
 
     # [4] the values of the series and of its estimation id
+    ladder_stages = stages_of_every_series(ladder_stages, series_rate)
     series_blocks = [(series_table, SERIES_VALUES_STEP_06), (series_rate, SERIES_VALUES_STEP_08),
-                     (series_estimate, SERIES_VALUES_STEP_11)]
+                     (ladder_stages, SERIES_VALUES_STEP_10), (series_estimate, SERIES_VALUES_STEP_11)]
     for block_frame, block_specs in series_blocks:
         if block_frame is not None:
             core = add_block(core, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
             blocks_present.append(block_specs)
-    if pool_dynamics is not None and len(pool_dynamics) and "s11_final_group_id" in core.columns:
-        dynamics_values = pool_dynamics[[ESTIMATION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
+    if pool_dynamics is not None and len(pool_dynamics) and "s10_stage3_id" in core.columns:
+        dynamics_values = pool_dynamics[[COMPOSITION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
         dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_13})
-        core = core.merge(dynamics_values, left_on="s11_final_group_id", right_on=ESTIMATION_ID_COLUMN, how="left").drop(columns=ESTIMATION_ID_COLUMN)
+        core = core.merge(dynamics_values, left_on="s10_stage3_id", right_on=COMPOSITION_ID_COLUMN, how="left").drop(columns=COMPOSITION_ID_COLUMN)
         blocks_present.append(ESTIMATION_VALUES_STEP_13)
-    if technique_decision is not None and "s11_final_group_id" in core.columns:
+    if technique_decision is not None and "s10_stage3_id" in core.columns:
         core, backtest_specs = add_backtest_block(core, technique_decision, exam_by_pool, configuration)
         blocks_present.append(backtest_specs)
     if forecast is not None:
@@ -273,11 +284,11 @@
 def add_backtest_block(core: pd.DataFrame, technique_decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                        configuration: Config) -> tuple:
     """The chosen technique and the exam error of the row's estimation id, one column per horizon band."""
-    per_band = technique_decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
+    per_band = technique_decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
     if exam_by_pool is not None:
-        per_band = per_band.merge(exam_by_pool[[ESTIMATION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio",
-                                                "dentro_banda"]], on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
-    wide = per_band.pivot(index=ESTIMATION_ID_COLUMN, columns="tramo_h")
+        per_band = per_band.merge(exam_by_pool[[COMPOSITION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio",
+                                                "dentro_banda"]], on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
+    wide = per_band.pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h")
     specs = []
     renamed = {}
     for source, name, step, description in ESTIMATION_VALUES_STEP_14:
@@ -287,7 +298,7 @@
                 specs.append((source, f"{name}_{band_name}", step, f"{description} (band {band_name})"))
     wide = wide[list(renamed)]
     wide.columns = [renamed[column] for column in wide.columns]
-    core = core.merge(wide, left_on="s11_final_group_id", right_index=True, how="left")
+    core = core.merge(wide, left_on="s10_stage3_id", right_index=True, how="left")
     return core, specs
 
 
@@ -394,6 +405,21 @@
     return core.drop(columns=[name for name in core.columns if name.startswith("_ts_")])
 
 
+def stages_of_every_series(ladder_stages, series_rate):
+    """The 4 stages of every forecast series: the estimable ones as the ladder left them; the others
+    (solo_historia, solo_futuro: they do not climb) are themselves in every stage, with their own support."""
+    if ladder_stages is None or series_rate is None:
+        return ladder_stages
+    missing = series_rate[~series_rate[SERIES_ID_COLUMN].isin(ladder_stages[SERIES_ID_COLUMN])]
+    if missing.empty:
+        return ladder_stages
+    themselves = pd.DataFrame({SERIES_ID_COLUMN: missing[SERIES_ID_COLUMN].to_numpy()})
+    for stage in range(4):
+        themselves[f"stage{stage}_id"] = missing[SERIES_ID_COLUMN].to_numpy()
+        themselves[f"stage{stage}_support"] = missing["n_propio"].to_numpy()
+    return pd.concat([ladder_stages, themselves], ignore_index=True)
+
+
 def core_dimension_columns(configuration: Config) -> list:
     """The dimension columns of the core, as the extract names them (they are the slicers):
     the rate series columns, the revaluation extras, the exact discount and its bucket."""
```

## test_step_10_11.py

```diff
--- antes/test_step_10_11.py	2026-10-01 06:11:03.861637483 +0000
+++ test_step_10_11.py	2026-10-01 06:11:04.379615882 +0000
@@ -88,7 +88,7 @@
     check(abs(row["z"] - round(expected_z, 3)) < 1e-9
           and abs(row["tasa_estimada"] - (expected_z * row["group_rate"] + (1 - expected_z) * row["ref_rate"])) < 1e-9,
           "z = n / (n + k) with the group's support; rate = z · group + (1 − z) · reference")
-    rates_per_group = estimate.dropna(subset=["tasa_estimada"]).groupby("final_group_id")["tasa_estimada"].nunique()
+    rates_per_group = estimate.dropna(subset=["tasa_estimada"]).groupby("composition_id")["tasa_estimada"].nunique()
     check((rates_per_group == 1).all(), "every series of a group has the same rate: the group lends it")
     check((estimate["se_prediccion_pp"].dropna() >= estimate["se_estimacion_pp"].dropna() - 1e-9).all()
           and np.allclose(binomial_se_pp([0.5], [100]), [5.0]),
```

## test_step_12_14.py

```diff
--- antes/test_step_12_14.py	2026-10-01 06:11:04.146391083 +0000
+++ test_step_12_14.py	2026-10-01 06:11:04.378787970 +0000
@@ -47,7 +47,7 @@
     print("B · steps 12 and 14 on the synthetic")
     results, console, configuration = run_everything()
     pool_series, pool_reference = results["pool_series"], results["pool_reference"]
-    own = pool_series[pool_series["final_group_id"] == "EU|A|0|0|0|0|web"]
+    own = pool_series[pool_series["composition_id"] == "EU|A|0|0|0|0|web"]
     units = results["rated_units"]
     history = units[(units["fs_id"] == "EU|A|0|0|0|0|web") & units["tasa"].notna()]
     check(own["renovadas"].sum() == history["total_renewed_units"].sum(),
@@ -64,13 +64,13 @@
           and predictions.loc[predictions["proposito"] == "examen", "mes_objetivo"].astype(str).isin(["2026-06", "2026-07", "2026-08"]).all(),
           "the exam targets are the exam months of the calendar")
     decision = backtest["decision"]
-    unjudged = pool_reference.loc[pool_reference["gate"] == "soporte", "final_group_id"]
-    check((decision[decision["final_group_id"].isin(unjudged)]["tecnica_origen"] == "sin_soporte").all(),
+    unjudged = pool_reference.loc[pool_reference["gate"] == "soporte", "composition_id"]
+    check((decision[decision["composition_id"].isin(unjudged)]["tecnica_origen"] == "sin_soporte").all(),
           "an id below the floor is not judged: it takes the challenger")
     champions = decision[decision["tecnica_origen"] == "campeon"]
     exam_rows = predictions[predictions["proposito"] == "seleccion"]
     one = champions.iloc[0]
-    scores = (exam_rows[(exam_rows["final_group_id"] == one["final_group_id"]) & (exam_rows["tramo_h"] == one["tramo_h"])]
+    scores = (exam_rows[(exam_rows["composition_id"] == one["composition_id"]) & (exam_rows["tramo_h"] == one["tramo_h"])]
               .assign(abs_norm=lambda frame: frame["err_norm"].abs()).groupby("tecnica")["abs_norm"].mean())
     margin = configuration.challenger_margin_by_band[one["tramo_h"]]
     check(scores[one["tecnica"]] == scores.min() and scores[one["tecnica"]] < scores["T3_ma3"] - margin,
```

## test_step_audit.py

```diff
--- antes/test_step_audit.py	1970-01-01 00:00:00.000000000 +0000
+++ test_step_audit.py	2026-10-01 06:17:37.675640605 +0000
@@ -0,0 +1,78 @@
+"""
+test_step_audit.py — The 4 stages of the ladder and the audit tables: every stage adds up, an id
+only changes when its group changes, and every satellite adds up to the table it comes from.
+
+    python test_step_audit.py
+"""
+
+# ─── imports ─────────────────────────────────────────────────────────────────────
+import numpy as np
+
+from main import run
+from test_helpers import check, console_of, finish, synthetic_with
+
+
+def run_synthetic():
+    results = {}
+    console = console_of(lambda: results.update(run(synthetic_with())))
+    return results, console
+
+
+def test_the_stages() -> None:
+    print("A · the 4 stages of every forecast series")
+    results, console = run_synthetic()
+    stages, steps = results["ladder"]["stages"], results["ladder"]["steps"]
+    tele = steps[steps["fs_id"] == "NA|A|0|0|0|0|tele"].sort_values("ladder_step")
+    check(list(tele["group_id"][:3]) == ["NA|A|0|0|0|0|tele"] * 3 and tele["group_id"].iloc[-1] == "*|*|SIG=neutro|*",
+          "an id only changes when the group really changes: NA|A|tele keeps its own id until it joins others")
+    negative = stages[stages["fs_id"] == "EU|A|0|0|1|0|web"].iloc[0]
+    check(negative["stage0_id"] == "EU|A|0|0|1|0|web" and negative["stage0_support"] == 10
+          and negative["stage1_id"] == "EU|A|SIG=negativo|web" and negative["stage1_support"] == 42
+          and negative["stage3_id"] == negative["stage2_id"],
+          "a negative series: own support 10 at stage 0, 42 after the sign; nothing more in stages 2 and 3")
+    core = results["core"]
+    raw = core[core["origen_fila"] == "raw"]
+    check(all(raw.groupby(f"s10_stage{stage}_id")["s00_vencen_usd"].sum().sum() == raw["s00_vencen_usd"].sum() for stage in range(4)),
+          "grouping the core by the id of any stage, the USD due adds up to the same total")
+    check(raw["s10_stage3_id"].nunique() <= raw["s10_stage0_id"].nunique(), "fewer groups at stage 3 than at stage 0")
+    check({"s17_confidence", "s11_credibility_effect_pp", "s10_stage3_support"} <= set(core.columns),
+          "the core carries the stages, the credibility effect and the confidence")
+
+
+def test_the_audit_tables() -> None:
+    print("B · the audit tables add up to their sources")
+    results, console = run_synthetic()
+    check("17 checks: 17 ok" in console.split("STEP AUD")[1], "the 17 checks of the audit step pass")
+    audit = results["audit"]
+    techniques = audit["composition_techniques"]
+    check(set(techniques["status"]) <= {"tested", "not_enough_history", "composition_below_floor"}
+          and (techniques.groupby(["composition_id", "tramo_h"])["is_chosen"].sum() == 1).all(),
+          "every composition and band: one chosen technique; every technique with its status")
+    members = audit["credibility_members"]
+    check(set(members["role"]) <= {"borrower", "lender"} and len(audit["credibility"]) == members["credibility_ref_id"].nunique(),
+          "every reference with its members, borrowers and lenders")
+    backtest = audit["series_backtest"]
+    one_series = backtest[backtest["fs_id"] == "EU|A|0|0|0|0|web"]
+    check(one_series["tecnica"].nunique() == 10 and (one_series.groupby(["mes_objetivo", "h"])["is_chosen"].sum() == 1).all(),
+          "selecting a forecast series: every technique tested on its months, the chosen one marked")
+    row = backtest.iloc[0]
+    check(abs(row["pred_units"] - row["pred_rate"] * row["due_units"]) < 1e-9
+          and abs(row["err_pp"] - 100 * (row["pred_rate"] - row["real_rate"])) < 1e-9,
+          "a series' prediction is the rate times its own units due; the error is in pp of its own rate")
+    summary = audit["series_technique_summary"]
+    check((summary.groupby(["fs_id", "tramo_h"])["best_for_series"].sum() >= 1).all() and summary["chosen_rank"].notna().all(),
+          "the summary ranks the techniques within every series and says where the chosen one ranks")
+    dynamics = audit["series_dynamics"]
+    check(set(dynamics["measurable"]) <= {"yes", "low_support", "short_history"}
+          and dynamics.loc[dynamics["measurable"] == "short_history", "phi"].isna().all(),
+          "every forecast series has its dynamics, with whether it can be trusted")
+    forecast_all = audit["composition_forecast_all"]
+    check(forecast_all.groupby(["composition_id", "h"])["is_chosen"].sum().max() == 1 and forecast_all["rate"].between(0, 1).all()
+          if forecast_all["rate"].notna().any() else False,
+          "the future rate of every technique, one chosen per composition and horizon")
+
+
+if __name__ == "__main__":
+    test_the_stages()
+    test_the_audit_tables()
+    finish()
```

## vocabulario.py

```diff
--- antes/vocabulario.py	2026-10-01 06:11:03.861959558 +0000
+++ vocabulario.py	2026-10-01 06:14:36.797217494 +0000
@@ -81,7 +81,8 @@
 # ─── the ladder (steps 10 and 11) ────────────────────────────────────────────────
 SIGN_TOKEN = "SIG="                          # inside a relative's pattern: the timevarying block summarised as its sign
 WILDCARD = "*"                               # inside a relative's pattern: a dimension collapsed or annulled
-ESTIMATION_ID_COLUMN = "final_group_id"      # the final group of a series in the ladder (step 10): it lends its rate
+COMPOSITION_ID_COLUMN = "composition_id"     # the composition of a forecast series: its group after the 3 merging stages of
+                                             # the ladder (step 10); it is what is predicted, and it lends its rate
 
 # The risk level of a series: how its rate is estimated, from best to worst.
 LEVEL_OWN = "A_propio"                        # its own support is precise (≥ own_rate_floor) and it has a full year
@@ -128,7 +129,7 @@
 
 # ─── step 13: the dynamics of the rate ───────────────────────────────────────────
 TABLE_PORTFOLIO_SEASONALITY = "estacionalidad_cartera"   # step 13: the month effect of the whole portfolio
-TABLE_POOL_DYNAMICS = "dinamica_pool"                    # step 13: φ, trend, seasonality of every estimation id
+TABLE_POOL_DYNAMICS = "composition_dynamics"   # step 13: φ, trend and seasonality of every composition
 
 # ─── the report ──────────────────────────────────────────────────────────────────
 TABLE_SERIES_CARD = "ficha_serie"            # one row per series: every attribute the framework knows about it
@@ -184,3 +185,16 @@
 TOTAL_ORIGIN_TOTAL = "TOTAL"
 TABLE_TIME_SERIES = "time_series"            # step 20: region × month: origin, units, value, level, AUV, discount, rate
 TABLE_FORECAST_TOTAL = "forecast_total"      # step 20: year × origin and the total of every year
+
+# ─── step AUD: the audit tables (satellites of the core, for the drill-down) ───────
+TABLE_COMPOSITION = "composition"                        # every id of every stage: who forms it, support, rate, prediction
+TABLE_COMPOSITION_TECHNIQUES = "composition_techniques"  # composition × band × technique: status, errors, rank, chosen
+TABLE_CREDIBILITY = "credibility"                        # every credibility reference: support, rate, k and its parts
+TABLE_CREDIBILITY_MEMBERS = "credibility_members"        # reference × forecast series: what goes into its rate
+TABLE_SERIES_DYNAMICS = "series_dynamics"                # every forecast series: φ, trend, seasonality vs its composition
+TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
+TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
+TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
+TECHNIQUE_TESTED = "tested"
+TECHNIQUE_NOT_ENOUGH_HISTORY = "not_enough_history"
+TECHNIQUE_COMPOSITION_BELOW_FLOOR = "composition_below_floor"
```
