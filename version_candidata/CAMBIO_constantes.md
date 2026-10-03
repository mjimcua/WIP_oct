# Cambio · nombres de columnas y valores escritos a mano → constantes de vocabulario.py
Cada nombre que ya tenía constante en `vocabulario.py` se usa ahora a través de ella. Un renombrado futuro se hace en una sola línea. Verificación: 17 ficheros de test en verde y las salidas del sintético idénticas a las de antes del cambio (núcleo, total, estimación de series, referencia de grupos, examen de cartera, decisión del backtest, forecast).

## step_00_validate_raw.py

```diff
--- antes/step_00_validate_raw.py	2026-10-01 05:40:53.746836686 +0000
+++ step_00_validate_raw.py	2026-10-01 05:41:33.201526931 +0000
@@ -31,6 +31,7 @@
 import pandas as pd
 
 from config import COLUMN_ROLE_IGNORE, Config, parse_month
+from vocabulario import CALENDAR_ROLE_COLUMN
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -173,12 +174,12 @@
     for column_name, role in column_roles.items():
         if column_name in validated.columns:
             columns_by_role.setdefault(role, []).append(column_name)
-    configuration.show_table(pd.DataFrame([{"rol": role, "columnas": len(role_columns), "nombres": ", ".join(role_columns)}
+    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "columnas": len(role_columns), "nombres": ", ".join(role_columns)}
                                            for role, role_columns in columns_by_role.items()]))
 
     configuration.logger.doc(f"[{STEP_LABEL}] the calendar of the Config on these months: "
                              f"{configuration.calendar_description()}")
     rows_per_role = pd.Series(configuration.role_of_months(validated[configuration.period_col])).value_counts()
     months_per_role = pd.Series(configuration.role_of_months(pd.Series(raw_months))).value_counts()
-    configuration.show_table(pd.DataFrame([{"rol": role, "meses": int(months_per_role.get(role, 0)), "filas": int(row_count)}
+    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "meses": int(months_per_role.get(role, 0)), "filas": int(row_count)}
                                            for role, row_count in rows_per_role.items()]))
```

## step_02_apply_calendar.py

```diff
--- antes/step_02_apply_calendar.py	2026-10-01 05:40:53.898443303 +0000
+++ step_02_apply_calendar.py	2026-10-01 05:41:33.202005183 +0000
@@ -274,7 +274,7 @@
         renewed_units = role_rows[configuration.renewed_units_col].sum(min_count=1)
         pipeline_units = role_rows[configuration.pipeline_units_col].sum()
         summary_rows.append({
-            "rol": role,
+            CALENDAR_ROLE_COLUMN: role,
             "meses": len(role_months),
             "desde": str(role_months[0]) if role_months else "",
             "hasta": str(role_months[-1]) if role_months else "",
```

## step_04_forecast_units.py

```diff
--- antes/step_04_forecast_units.py	2026-10-01 05:40:53.746539253 +0000
+++ step_04_forecast_units.py	2026-10-01 05:41:33.202269584 +0000
@@ -167,7 +167,7 @@
     per_role_rows = []
     for role in ROLES_IN_ORDER:
         role_units = forecast_units[forecast_units[CALENDAR_ROLE_COLUMN] == role]
-        per_role_rows.append({"rol": role, "unidades": len(role_units), "series": role_units[SERIES_ID_COLUMN].nunique(),
+        per_role_rows.append({CALENDAR_ROLE_COLUMN: role, "unidades": len(role_units), "series": role_units[SERIES_ID_COLUMN].nunique(),
                               "unidades_vencen": role_units[configuration.pipeline_units_col].sum(),
                               "usd_vence": role_units[configuration.pipeline_usd_col].sum()})
     configuration.show_table(pd.DataFrame(per_role_rows))
```

## step_06_series_routes.py

```diff
--- antes/step_06_series_routes.py	2026-10-01 05:40:53.666070857 +0000
+++ step_06_series_routes.py	2026-10-01 05:41:33.205719811 +0000
@@ -34,8 +34,8 @@
 import pandas as pd
 
 from config import ACTIVE_FLAG_VALUES, Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN,
-                         ROLES_IN_ORDER, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, ROUTE_PREDICTABLE,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ROLES_IN_ORDER, ROLE_PROJECTION, ROLE_TEST,
+                         ROLE_TRAIN, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, ROUTE_PREDICTABLE,
                          SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_SERIES, UNIT_ID_COLUMN, UNIVERSE_COLUMN,
                          UNIVERSE_MIXED, UNIVERSE_NORMAL, UNIVERSE_TIME_SERIES)
 
@@ -180,6 +180,6 @@
     for route in ROUTES_IN_ORDER:
         route_series = series_table[series_table[ROUTE_COLUMN] == route]
         route_usd = route_series["usd_por_predecir"].sum()
-        route_rows.append({"ruta": route, "series": len(route_series), "usd_por_predecir": route_usd,
+        route_rows.append({ROUTE_COLUMN: route, "series": len(route_series), "usd_por_predecir": route_usd,
                            "pct_usd": route_usd / total_usd if total_usd else 0.0})
     configuration.show_table(pd.DataFrame(route_rows))
```

## step_07_support_bound.py

```diff
--- antes/step_07_support_bound.py	2026-10-01 05:40:53.746776813 +0000
+++ step_07_support_bound.py	2026-10-01 05:41:33.202509403 +0000
@@ -113,7 +113,7 @@
         role_units = support_bound[support_bound[CALENDAR_ROLE_COLUMN] == role]
         role_usd = role_units[configuration.pipeline_usd_col].sum()
         independent_total = float(np.sqrt((role_units["moe_usd_max"] ** 2).sum()))
-        role_rows.append({"rol": role, "unidades": len(role_units),
+        role_rows.append({CALENDAR_ROLE_COLUMN: role, "unidades": len(role_units),
                           "moe_pp_mediana": role_units["moe_pp_max"].median() if len(role_units) else np.nan,
                           "usd_vence": role_usd, "moe_usd_total": independent_total,
                           "moe_pct_total": independent_total / role_usd if role_usd else np.nan})
```

## step_10_ladder_groups.py

```diff
--- antes/step_10_ladder_groups.py	2026-10-01 05:40:53.898597317 +0000
+++ step_10_ladder_groups.py	2026-10-01 05:40:53.924059287 +0000
@@ -58,9 +58,9 @@
 import pandas as pd
 
 from config import ID_FIELD_SEPARATOR, Config, id_text
-from vocabulario import (RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN, SIGN_COLUMN, SIGN_MIXED,
-                         SIGN_NEUTRAL, SIGN_TOKEN, TABLE_LADDER_GROUPS, TABLE_LADDER_STEPS, TABLE_LADDER_SUMMARY,
-                         UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)
+from vocabulario import (ESTIMATION_ID_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
+                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_LADDER_GROUPS, TABLE_LADDER_STEPS,
+                         TABLE_LADDER_SUMMARY, UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)
 
 
 STEP_LABEL = "10"
@@ -130,7 +130,7 @@
     # [5] the final groups and their references
     groups, reference_members = credibility_references(estimable, patterns, plan, final, history, configuration)
     with_reference = groups["credibility_ref_id"].notna()
-    configuration.log_action(STEP_LABEL, 5, f"{groups['final_group_id'].nunique():,} final groups · "
+    configuration.log_action(STEP_LABEL, 5, f"{groups[ESTIMATION_ID_COLUMN].nunique():,} final groups · "
                                             f"{int(with_reference.sum()):,} series take a credibility reference "
                                             f"({reference_members['credibility_ref_id'].nunique():,} references)")
 
@@ -270,7 +270,7 @@
             "pct_usd_own_rate": float(usd_to_predict[support_of_series >= configuration.own_rate_floor].sum() / total_usd)
                                 if total_usd else np.nan})
     steps = pd.concat(step_rows, ignore_index=True)
-    final = pd.DataFrame({"final_group_id": group, "final_step": assigned_at,
+    final = pd.DataFrame({ESTIMATION_ID_COLUMN: group, "final_step": assigned_at,
                           "closed": group.map(support_of_groups(history, group, configuration)["support"]).fillna(0.0)
                                     >= floor - SUPPORT_TOLERANCE})
     return steps, pd.DataFrame(summary_rows), final
@@ -285,7 +285,7 @@
     """The final group of every series with its support and rate, and, below the own-rate floor,
     its credibility reference with its support and rate; and the members of every reference."""
     floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
-    groups_support = support_of_groups(history, final["final_group_id"], configuration)
+    groups_support = support_of_groups(history, final[ESTIMATION_ID_COLUMN], configuration)
     not_mixed = estimable[SIGN_COLUMN] != SIGN_MIXED
 
     # every pattern of every pass, counted over ALL the series (the closed and the big ones too)
@@ -295,7 +295,7 @@
         all_patterns[step["step"]] = support_of_groups(history, pattern_of_series, configuration)
 
     group_rows = []
-    for group_id, members in final.groupby("final_group_id"):
+    for group_id, members in final.groupby(ESTIMATION_ID_COLUMN):
         group_support = float(groups_support.loc[group_id, "support"]) if group_id in groups_support.index else 0.0
         group_series = int(len(members))
         first_member = members.index[0]
@@ -315,7 +315,7 @@
                         break
             if widest is not None:
                 chosen_step, chosen_id = widest
-        row = {"final_group_id": group_id, "final_step": int(members["final_step"].iloc[0]),
+        row = {ESTIMATION_ID_COLUMN: group_id, "final_step": int(members["final_step"].iloc[0]),
                "group_series": group_series, "group_support": group_support,
                "group_rate": float(groups_support.loc[group_id, "rate"]) if group_id in groups_support.index else np.nan,
                "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
@@ -328,7 +328,7 @@
         if column_name not in by_group.columns:
             by_group[column_name] = np.nan
 
-    groups = final[["final_group_id"]].reset_index().merge(by_group, on="final_group_id", how="left")
+    groups = final[[ESTIMATION_ID_COLUMN]].reset_index().merge(by_group, on=ESTIMATION_ID_COLUMN, how="left")
     reference_rows = []
     for (ref_step, ref_id), _ in by_group.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
         members = patterns.index[(patterns[int(ref_step)] == ref_id) & not_mixed]
@@ -371,7 +371,7 @@
                             failure_detail=f"{len(reopened):,} series left a closed group · groups by pass {summary['groups'].tolist()}")
 
     # [4] a reference is wider than its group
-    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates("final_group_id")
+    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates(ESTIMATION_ID_COLUMN)
     narrower = with_reference[with_reference["ref_series"] <= with_reference["group_series"]]
     configuration.log_check(STEP_LABEL, check_log, "every credibility reference has more series than its group", narrower.empty,
                             failure_detail=f"{len(narrower):,} references not wider than their group",
```

## step_11_ladder.py

```diff
--- antes/step_11_ladder.py	2026-10-01 05:40:53.810132338 +0000
+++ step_11_ladder.py	2026-10-01 05:41:33.205959740 +0000
@@ -92,8 +92,8 @@
     (LEVEL_FAR, "merged in a mandatory pass, or never reached the floor", "its group collapsed a mandatory dim"),
     (LEVEL_SIGNED_UNDER_FLOOR, "signed, its group never reached the floor", "the best rate of ITS sign, noisy: it may not mix with unsigned series"),
     (LEVEL_MIXED, "flags of both signs", "never merged: keeps its own rate; should be ≈ 0"),
-    (LEVEL_NO_HISTORY, "solo_futuro", "no rate to estimate: it will take a rate from its cell later"),
-    (LEVEL_NO_IMPACT, "solo_historia", "nothing to predict: kept for the groups and the backtest"),
+    (LEVEL_NO_HISTORY, ROUTE_FUTURE_ONLY, "no rate to estimate: it will take a rate from its cell later"),
+    (LEVEL_NO_IMPACT, ROUTE_HISTORY_ONLY, "nothing to predict: kept for the groups and the backtest"),
     (LEVEL_TIME_SERIES, "time_series universe", "labelled only: treated apart"),
 ]
 
@@ -120,7 +120,7 @@
     # [1] the final group and the reference of every estimable series
     groups = ladder["groups"]
     step_names = dict(zip(ladder["summary"]["ladder_step"], ladder["summary"]["step_name"]))
-    configuration.log_action(STEP_LABEL, 1, f"{len(groups):,} estimable series in {groups['final_group_id'].nunique():,} "
+    configuration.log_action(STEP_LABEL, 1, f"{len(groups):,} estimable series in {groups[ESTIMATION_ID_COLUMN].nunique():,} "
                                             f"final groups; {int(groups['credibility_ref_id'].notna().sum()):,} of them "
                                             f"with a credibility reference")
 
@@ -166,7 +166,7 @@
     """Bühlmann-Straub k per reference, from the final groups that share it (one row per group):
     within = mean of p(1 − p); between = weighted variance of p minus its sampling part;
     k = within / between. Fewer than 3 groups: k_cred. No between variance: 10 × k_cred."""
-    siblings = groups.drop_duplicates("final_group_id").dropna(subset=["credibility_ref_id", "group_rate"]).copy()
+    siblings = groups.drop_duplicates(ESTIMATION_ID_COLUMN).dropna(subset=["credibility_ref_id", "group_rate"]).copy()
     siblings = siblings[siblings["group_support"] > 0]
     if siblings.empty:
         return pd.Series(dtype=float)
@@ -195,15 +195,15 @@
     estimate = series_rate.merge(groups, on=SERIES_ID_COLUMN, how="left")
 
     # a series that is not estimable is its own group, with its own rate
-    not_estimable = estimate["final_group_id"].isna()
-    estimate.loc[not_estimable, "final_group_id"] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
+    not_estimable = estimate[ESTIMATION_ID_COLUMN].isna()
+    estimate.loc[not_estimable, ESTIMATION_ID_COLUMN] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
     estimate.loc[not_estimable, "final_step"] = 0
     estimate.loc[not_estimable, "group_series"] = 1
     estimate.loc[not_estimable, "group_support"] = estimate.loc[not_estimable, "n_propio"]
     estimate.loc[not_estimable, "group_rate"] = estimate.loc[not_estimable, "tasa_propia"]
     estimate["final_step"] = estimate["final_step"].astype(int)
     estimate["group_series"] = estimate["group_series"].astype(int)
-    estimate[ESTIMATION_ID_COLUMN] = estimate["final_group_id"]
+    estimate[ESTIMATION_ID_COLUMN] = estimate[ESTIMATION_ID_COLUMN]
 
     group_rate, group_support = estimate["group_rate"], estimate["group_support"].fillna(0.0)
     reference_rate, reference_support = estimate["ref_rate"], estimate["ref_support"]
@@ -274,7 +274,7 @@
                             context=f"{len(estimate):,} series")
 
     # [2] the group lends its rate: every series of a group has the same estimated rate
-    spread = estimate.dropna(subset=["tasa_estimada"]).groupby("final_group_id")["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
+    spread = estimate.dropna(subset=["tasa_estimada"]).groupby(ESTIMATION_ID_COLUMN)["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
     configuration.log_check(STEP_LABEL, check_log, "every series of a group has the same estimated rate",
                             bool((spread <= ROUNDING_TOLERANCE).all()),
                             failure_detail=f"{int((spread > ROUNDING_TOLERANCE).sum()):,} groups with different rates")
```

## step_12_pool_series.py

```diff
--- antes/step_12_pool_series.py	2026-10-01 05:40:53.897769726 +0000
+++ step_12_pool_series.py	2026-10-01 05:41:33.203265245 +0000
@@ -39,8 +39,8 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT,
-                         RATE_COLUMN, SERIES_ID_COLUMN, TABLE_POOL_REFERENCE, TABLE_POOL_SERIES)
+from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, GATE_SUPPORT, RATE_COLUMN,
+                         SERIES_ID_COLUMN, TABLE_POOL_REFERENCE, TABLE_POOL_SERIES)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -69,7 +69,7 @@
     period_column = configuration.period_col
 
     # [1] the final groups of step 10 (the estimable series) and their series: a partition
-    membership = ladder["groups"][[SERIES_ID_COLUMN, "final_group_id"]].rename(columns={"final_group_id": ESTIMATION_ID_COLUMN})
+    membership = ladder["groups"][[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN]].rename(columns={ESTIMATION_ID_COLUMN: ESTIMATION_ID_COLUMN})
     chosen = series_estimate[series_estimate[SERIES_ID_COLUMN].isin(set(membership[SERIES_ID_COLUMN]))]
     chosen_ids = set(membership[ESTIMATION_ID_COLUMN])
     configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} final groups of {len(chosen):,} series "
@@ -85,7 +85,7 @@
                         vencen=(configuration.pipeline_units_col, "sum"),
                         series_en_el_mes=(SERIES_ID_COLUMN, "nunique"))
                    .reset_index().sort_values([ESTIMATION_ID_COLUMN, period_column]).reset_index(drop=True))
-    pool_series["tasa"] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
+    pool_series[RATE_COLUMN] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
     configuration.log_action(STEP_LABEL, 2, f"{len(pool_series):,} group × month rows; a group has "
                                             f"{pool_series.groupby(ESTIMATION_ID_COLUMN).size().median():.0f} months (median)")
 
@@ -102,15 +102,15 @@
     configuration.log_check(STEP_LABEL, check_log, "every final group has a monthly series", not missing_ids,
                             failure_detail=f"{len(missing_ids):,} ids without months: {sorted(missing_ids)[:5]}",
                             context=f"{len(chosen_ids):,} ids")
-    support_in_step_10 = (ladder["groups"].drop_duplicates("final_group_id")
-                          .set_index("final_group_id")["group_support"].rename("n_pool_paso_10"))
+    support_in_step_10 = (ladder["groups"].drop_duplicates(ESTIMATION_ID_COLUMN)
+                          .set_index(ESTIMATION_ID_COLUMN)["group_support"].rename("n_pool_paso_10"))
     compared = pool_reference.join(support_in_step_10, on=ESTIMATION_ID_COLUMN)
     mismatched = compared[(compared["n_pool"] - compared["n_pool_paso_10"]).abs() > SUPPORT_TOLERANCE]
     configuration.log_check(STEP_LABEL, check_log, "the support of every group equals its support in step 10",
                             mismatched.empty,
                             failure_detail=f"{len(mismatched):,} groups whose support differs from step 10",
                             examples=mismatched[[ESTIMATION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
-    out_of_range = pool_series[(pool_series["tasa"] < 0) | (pool_series["tasa"] > 1)]
+    out_of_range = pool_series[(pool_series[RATE_COLUMN] < 0) | (pool_series[RATE_COLUMN] > 1)]
     configuration.log_check(STEP_LABEL, check_log, "every monthly rate is between 0 and 1", out_of_range.empty,
                             failure_detail=f"{len(out_of_range):,} months with a rate outside [0, 1]", blocking=False,
                             examples=out_of_range)
```

## step_13_dynamics.py

```diff
--- antes/step_13_dynamics.py	2026-10-01 05:40:53.897919703 +0000
+++ step_13_dynamics.py	2026-10-01 05:41:33.202749182 +0000
@@ -50,8 +50,8 @@
 from scipy import stats
 
 from config import Config
-from vocabulario import (ESTIMATION_ID_COLUMN, GATE_LEVEL, TABLE_POOL_DYNAMICS, TABLE_PORTFOLIO_SEASONALITY,
-                         TRUTH_ROLES)
+from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, TABLE_POOL_DYNAMICS,
+                         TABLE_PORTFOLIO_SEASONALITY, TRUTH_ROLES)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -81,7 +81,7 @@
     configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
     check_log = []
     period_column = configuration.period_col
-    truth_months = pool_series[pool_series["rol"].isin(TRUTH_ROLES) & (pool_series["vencen"] > 0)]
+    truth_months = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (pool_series["vencen"] > 0)]
 
     # [1] the pools measured
     months_per_pool = truth_months.groupby(ESTIMATION_ID_COLUMN).size()
```

## step_14_backtest.py

```diff
--- antes/step_14_backtest.py	2026-10-01 05:40:53.897789959 +0000
+++ step_14_backtest.py	2026-10-01 05:41:33.203687736 +0000
@@ -70,8 +70,8 @@
 from config import Config
 from techniques import CATALOGUE, eligible_techniques, inverse_logit, logit, predict_logit, technique_table
 from vocabulario import (CHALLENGER_ORIGIN, CHAMPION_ORIGIN, ESTIMATION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM,
-                         PURPOSE_SELECTION, TABLE_BACKTEST_PREDICTIONS, TABLE_ERROR_BANDS, TABLE_EXAM_BY_POOL,
-                         TABLE_EXAM_TOTAL, TABLE_TECHNIQUE_DECISION, TABLE_TECHNIQUES)
+                         PURPOSE_SELECTION, RATE_COLUMN, TABLE_BACKTEST_PREDICTIONS, TABLE_ERROR_BANDS,
+                         TABLE_EXAM_BY_POOL, TABLE_EXAM_TOTAL, TABLE_TECHNIQUES, TABLE_TECHNIQUE_DECISION)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -96,7 +96,7 @@
 PERCENTAGE_POINTS = 100
 MIN_EXAM_COVERAGE = 0.80
 SUPPORT_ORIGIN = "sin_soporte"      # an id below the floor: not judged, it takes the challenger
-PREDICTION_COLUMNS = ["final_group_id", "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
+PREDICTION_COLUMNS = [ESTIMATION_ID_COLUMN, "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
                       "tasa_pred", "tasa_real", "vencen_real", "err_pp", "se_binom_pp", "err_norm"]
 
 
@@ -218,15 +218,15 @@
     period_column = configuration.period_col
     purpose_of = {month.ordinal: PURPOSE_SELECTION for month in selection_months}
     purpose_of.update({month.ordinal: PURPOSE_EXAM for month in exam_months})
-    judged_series = pool_series[pool_series["final_group_id"].isin(judged_ids) & pool_series["tasa"].notna()
+    judged_series = pool_series[pool_series[ESTIMATION_ID_COLUMN].isin(judged_ids) & pool_series[RATE_COLUMN].notna()
                                 & (pool_series["vencen"] > 0)]
     rows = []
-    for estimation_id, monthly in judged_series.groupby("final_group_id", sort=False):
+    for estimation_id, monthly in judged_series.groupby(ESTIMATION_ID_COLUMN, sort=False):
         monthly = monthly.sort_values(period_column)
         month_ordinals = np.array([month.ordinal for month in monthly[period_column]])
         calendar_months = np.array([month.month for month in monthly[period_column]])
         months = list(monthly[period_column])
-        rates = monthly["tasa"].to_numpy(dtype=float)
+        rates = monthly[RATE_COLUMN].to_numpy(dtype=float)
         units_due = monthly["vencen"].to_numpy(dtype=float)
         logit_rates = logit(rates)
         for target_position, target_ordinal in enumerate(month_ordinals):
@@ -269,11 +269,11 @@
     catalogue_order = {technique_id: position for position, technique_id in enumerate(CATALOGUE)}
     selection = predictions[predictions["proposito"] == PURPOSE_SELECTION].assign(
         abs_norm=lambda frame: frame["err_norm"].abs(), abs_pp=lambda frame: frame["err_pp"].abs())
-    scores = (selection.groupby(["final_group_id", "tramo_h", "tecnica"])
+    scores = (selection.groupby([ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"])
               .agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"), n_predicciones=("abs_norm", "size"))
               .reset_index())
     decision_rows = []
-    for (estimation_id, band_name), block in scores.groupby(["final_group_id", "tramo_h"]):
+    for (estimation_id, band_name), block in scores.groupby([ESTIMATION_ID_COLUMN, "tramo_h"]):
         block = block.set_index("tecnica")
         challenger_score = block.loc[challenger, "err_norm_medio"] if challenger in block.index else np.inf
         margin = float(configuration.challenger_margin_by_band.get(band_name, 0.0))
@@ -286,7 +286,7 @@
                 chosen, origin = best, CHAMPION_ORIGIN
         chosen_row = block.loc[chosen] if chosen in block.index else pd.Series(dict(err_norm_medio=np.nan, err_pp_medio=np.nan,
                                                                                     n_predicciones=0))
-        decision_rows.append({"final_group_id": estimation_id, "tramo_h": band_name, "tecnica": chosen,
+        decision_rows.append({ESTIMATION_ID_COLUMN: estimation_id, "tramo_h": band_name, "tecnica": chosen,
                               "tecnica_origen": origin, "err_norm_seleccion": chosen_row["err_norm_medio"],
                               "err_pp_seleccion": chosen_row["err_pp_medio"], "n_predicciones": int(chosen_row["n_predicciones"]),
                               "retador_err_norm_seleccion": challenger_score})
@@ -295,8 +295,8 @@
     # every id and band gets a decision: the unjudged ones (and a judged id with no selection
     # month in a band) take the challenger
     all_pairs = pd.MultiIndex.from_product([pool_reference[ESTIMATION_ID_COLUMN], list(configuration.horizon_bands)],
-                                           names=["final_group_id", "tramo_h"]).to_frame(index=False)
-    decision = all_pairs.merge(decision, on=["final_group_id", "tramo_h"], how="left")
+                                           names=[ESTIMATION_ID_COLUMN, "tramo_h"]).to_frame(index=False)
+    decision = all_pairs.merge(decision, on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
     unjudged = decision["tecnica"].isna()
     decision.loc[unjudged, "tecnica"] = challenger
     decision.loc[unjudged, "tecnica_origen"] = SUPPORT_ORIGIN
@@ -318,7 +318,7 @@
     """The chosen technique and the challenger in the exam months: per id and for the total."""
     challenger = configuration.challenger_technique
     exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
-    chosen_rows = exam.merge(decision[["final_group_id", "tramo_h", "tecnica"]], on=["final_group_id", "tramo_h", "tecnica"])
+    chosen_rows = exam.merge(decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"]], on=[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica"])
     chosen_rows = chosen_rows.merge(bands[["tecnica", "h", "q_low_norm", "q_high_norm"]], on=["tecnica", "h"], how="left")
     chosen_rows["dentro_banda"] = ((chosen_rows["err_norm"] >= chosen_rows["q_low_norm"])
                                    & (chosen_rows["err_norm"] <= chosen_rows["q_high_norm"])).astype(int)
@@ -326,15 +326,15 @@
 
     def summary(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
         return (rows.assign(abs_pp=rows["err_pp"].abs(), abs_norm=rows["err_norm"].abs())
-                .groupby(["final_group_id", "tramo_h"])
+                .groupby([ESTIMATION_ID_COLUMN, "tramo_h"])
                 .agg(**{f"{prefix}_err_pp_medio": ("abs_pp", "mean"), f"{prefix}_sesgo_pp": ("err_pp", "mean"),
                         f"{prefix}_err_norm_medio": ("abs_norm", "mean")}))
 
     exam_by_pool = summary(chosen_rows, "elegida").join(summary(challenger_rows, "retador"), how="left").reset_index()
-    exam_by_pool = exam_by_pool.merge(decision[["final_group_id", "tramo_h", "tecnica", "tecnica_origen"]],
-                                      on=["final_group_id", "tramo_h"], how="left")
+    exam_by_pool = exam_by_pool.merge(decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]],
+                                      on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
     exam_by_pool["mejora_pp"] = exam_by_pool["retador_err_pp_medio"] - exam_by_pool["elegida_err_pp_medio"]
-    exam_by_pool["dentro_banda"] = chosen_rows.groupby(["final_group_id", "tramo_h"])["dentro_banda"].mean().to_numpy()
+    exam_by_pool["dentro_banda"] = chosen_rows.groupby([ESTIMATION_ID_COLUMN, "tramo_h"])["dentro_banda"].mean().to_numpy()
 
     # the TOTAL: Σ predicted renewals vs Σ real renewals of every judged id, per exam month and horizon
     def total_of(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
@@ -367,7 +367,7 @@
                             context=f"{len(pool_reference):,} ids × {len(configuration.horizon_bands)} bands")
 
     # [3] the challenger is always there to compare with
-    targets = predictions.groupby(["final_group_id", "mes_objetivo", "h"])["tecnica"].apply(set)
+    targets = predictions.groupby([ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].apply(set)
     without_challenger = int(sum(configuration.challenger_technique not in techniques for techniques in targets))
     configuration.log_check(STEP_LABEL, check_log, "the challenger was predicted wherever another technique was",
                             without_challenger == 0,
@@ -378,7 +378,7 @@
     configuration.log_check(STEP_LABEL, check_log, "the ranking compares the techniques on common targets",
                             len(common) > 0,
                             failure_detail="no target where every technique competed: the ranking would mix different targets",
-                            context=f"{len(common):,} of {predictions.groupby(['final_group_id', 'mes_objetivo', 'h']).ngroups:,} "
+                            context=f"{len(common):,} of {predictions.groupby([ESTIMATION_ID_COLUMN, 'mes_objetivo', 'h']).ngroups:,} "
                                     f"targets have every technique; the others lack a history of 24 months")
 
     # [5] the band holds what it promises in the exam
@@ -394,8 +394,8 @@
 
 def common_targets(predictions: pd.DataFrame) -> pd.DataFrame:
     """The (id, target, horizon) where every technique of the catalogue competed."""
-    techniques_per_target = predictions.groupby(["final_group_id", "mes_objetivo", "h"])["tecnica"].nunique()
-    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[["final_group_id", "mes_objetivo", "h"]]
+    techniques_per_target = predictions.groupby([ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].nunique()
+    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[[ESTIMATION_ID_COLUMN, "mes_objetivo", "h"]]
 
 
 def log_backtest_report(predictions: pd.DataFrame, decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
@@ -405,7 +405,7 @@
                                             "as close as chance allows; same targets for every technique):")
     selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
     common = common_targets(predictions)
-    selection = selection.merge(common, on=["final_group_id", "mes_objetivo", "h"])
+    selection = selection.merge(common, on=[ESTIMATION_ID_COLUMN, "mes_objetivo", "h"])
     ranking = (selection.assign(abs_norm=selection["err_norm"].abs(), abs_pp=selection["err_pp"].abs())
                .groupby(["tramo_h", "tecnica"]).agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"),
                                                     predicciones=("abs_norm", "size"))
@@ -415,13 +415,13 @@
     configuration.logger.doc(f"[{STEP_LABEL}] the chosen technique per band (ids and the money their series predict):")
     chosen_money = decision.merge(pool_reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
     configuration.show_table(chosen_money.groupby(["tramo_h", "tecnica", "tecnica_origen"])
-                             .agg(ids=("final_group_id", "size"), usd_por_predecir=("usd_por_predecir", "sum"))
+                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum"))
                              .reset_index().sort_values(["tramo_h", "usd_por_predecir"], ascending=[True, False]))
 
     configuration.logger.doc(f"[{STEP_LABEL}] precision in the EXAM, per band (mean over the judged ids; err in pp of rate; "
                              f"mejora = challenger's error − chosen's error):")
     configuration.show_table(exam_by_pool.groupby("tramo_h")
-                             .agg(ids=("final_group_id", "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
+                             .agg(ids=(ESTIMATION_ID_COLUMN, "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
                                   retador_err_pp=("retador_err_pp_medio", "mean"), mejora_pp=("mejora_pp", "mean"),
                                   elegida_sesgo_pp=("elegida_sesgo_pp", "mean"), dentro_banda=("dentro_banda", "mean"))
                              .reset_index())
```

## step_16_uplift_backtest.py

```diff
--- antes/step_16_uplift_backtest.py	2026-10-01 05:40:53.898118441 +0000
+++ step_16_uplift_backtest.py	2026-10-01 05:41:33.202949222 +0000
@@ -39,7 +39,8 @@
 
 from config import Config
 from step_15_uplift import estimate_cell_uplifts, renewer_rows
-from vocabulario import PATH_CONTRACT, PATH_STATISTICAL, ROLE_TEST, TABLE_UPLIFT_BACKTEST, UPLIFT_CELL_ID_COLUMN
+from vocabulario import (CALENDAR_ROLE_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, ROLE_TEST, TABLE_UPLIFT_BACKTEST,
+                         UPLIFT_CELL_ID_COLUMN)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -72,7 +73,7 @@
 
     # [2] the renewals of the exam, predicted by both paths
     exam_rows = renewer_rows(fine_table, configuration)
-    exam_rows = exam_rows[exam_rows["rol"] == ROLE_TEST].copy()
+    exam_rows = exam_rows[exam_rows[CALENDAR_ROLE_COLUMN] == ROLE_TEST].copy()
     exam_rows["pred_" + PATH_STATISTICAL] = exam_rows["_denominador"] * exam_rows[UPLIFT_CELL_ID_COLUMN].map(
         cells.set_index(UPLIFT_CELL_ID_COLUMN)["uplift"])
     discount = configuration.discount_value_column
```

## step_17_forecast.py

```diff
--- antes/step_17_forecast.py	2026-10-01 05:40:53.898146936 +0000
+++ step_17_forecast.py	2026-10-01 05:41:33.204279091 +0000
@@ -79,9 +79,9 @@
 from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
 from step_14_backtest import band_of_horizon
 from techniques import inverse_logit, logit, predict_logit
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, PIPELINE_ORIGIN_COLUMN,
-                         PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_FROM_CELL,
-                         RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL,
+                         PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_COLUMN,
+                         RATE_FROM_CELL, RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
                          TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNIT_ID_COLUMN,
                          UPLIFT_CELL_ID_COLUMN)
 
@@ -361,11 +361,11 @@
     """The rate of every estimation id at every future horizon, with the technique of its band,
     learning from every closed month of the id."""
     decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
-    truth = pool_series[pool_series["rol"].isin(TRUTH_ROLES) & pool_series["tasa"].notna() & (pool_series["vencen"] > 0)]
+    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
     rows = []
     for estimation_id, monthly in truth.groupby(ESTIMATION_ID_COLUMN):
         monthly = monthly.sort_values(configuration.period_col)
-        history = logit(monthly["tasa"].to_numpy(dtype=float))
+        history = logit(monthly[RATE_COLUMN].to_numpy(dtype=float))
         calendar_months = np.array([month.month for month in monthly[configuration.period_col]])
         for horizon in horizons:
             band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
@@ -402,7 +402,7 @@
     future["origen_tasa"] = np.where(pool_rate.notna(), RATE_FROM_POOL, None)
 
     # no pool: the rate of the mandatory cell, then the global rate (closed months)
-    truth = rated_units[rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & rated_units["tasa"].notna()].copy()
+    truth = rated_units[rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & rated_units[RATE_COLUMN].notna()].copy()
     truth["_celda"] = join_columns(truth, configuration.business_mandatory_dims)
     cell_rates = truth.groupby("_celda")[configuration.renewed_units_col].sum() / truth.groupby("_celda")[configuration.pipeline_units_col].sum()
     global_rate = truth[configuration.renewed_units_col].sum() / truth[configuration.pipeline_units_col].sum()
```

## step_18_validation.py

```diff
--- antes/step_18_validation.py	2026-10-01 05:40:53.898169101 +0000
+++ step_18_validation.py	2026-10-01 05:41:33.204902111 +0000
@@ -37,7 +37,8 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, RATE_FROM_POOL, ROLE_PROJECTION, TABLE_VALIDATION,
+from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL,
+                         RATE_FROM_POOL, ROLE_PROJECTION, S0_PIPELINE_USD_COLUMN, SERIES_ID_COLUMN, TABLE_VALIDATION,
                          TRUTH_ROLES)
 
 
@@ -68,15 +69,15 @@
     # [1] the money
     configuration.log_action(STEP_LABEL, 1, "the money along the chain")
     time_series_due = results["time_series_rows"][due].fillna(0).sum() if results.get("time_series_rows") is not None else 0.0
-    wiped = fine["s0_vencen_usd"].sum() - fine[due].sum()               # step 02: the pipeline not known yet
+    wiped = fine[S0_PIPELINE_USD_COLUMN].sum() - fine[due].sum()               # step 02: the pipeline not known yet
     totals = {"extracto": raw[due].sum() - time_series_due - wiped, "tabla_fina": fine[due].sum(), "forecast_units": units[due].sum()}
     configuration.log_check(STEP_LABEL, check_log, "Σ USD due: extract (without time_series and the pipeline not known yet) = fine table = forecast units",
                             max(totals.values()) - min(totals.values()) <= MONEY_TOLERANCE,
                             failure_detail=f"totals differ: {totals}", context=f"${totals['extracto']:,.0f}")
     future_due = fine.loc[fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION, due].sum()
     configuration.log_check(STEP_LABEL, check_log, "the future USD due of the extract = Σ USD due of the forecast's extract rows",
-                            abs(future_due - forecast.loc[forecast["origen_pipeline"] == "real", due].sum()) <= MONEY_TOLERANCE,
-                            failure_detail=f"${future_due:,.0f} vs ${forecast.loc[forecast['origen_pipeline'] == 'real', due].sum():,.0f}",
+                            abs(future_due - forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum()) <= MONEY_TOLERANCE,
+                            failure_detail=f"${future_due:,.0f} vs ${forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum():,.0f}",
                             context=f"${future_due:,.0f} (the extended horizon adds its own pipeline)")
 
     # [2] coverage
@@ -84,12 +85,12 @@
     estimate = results["series_estimate"]
     with_money = estimate[estimate["usd_por_predecir"] > 0]
     uncovered = with_money[with_money["nivel_riesgo"].isna()]
-    forecast_series = set(forecast["fs_id"])
+    forecast_series = set(forecast[SERIES_ID_COLUMN])
     configuration.log_check(STEP_LABEL, check_log, "every series with money to predict has a rate and a risk level",
-                            uncovered.empty and set(with_money["fs_id"]) <= forecast_series,
+                            uncovered.empty and set(with_money[SERIES_ID_COLUMN]) <= forecast_series,
                             failure_detail=f"{len(uncovered)} series without level", context=f"{len(with_money):,} series")
-    pooled_ids = set(results["pool_reference"]["final_group_id"])
-    should_pool = forecast["final_group_id"].isin(pooled_ids)
+    pooled_ids = set(results["pool_reference"][ESTIMATION_ID_COLUMN])
+    should_pool = forecast[ESTIMATION_ID_COLUMN].isin(pooled_ids)
     configuration.log_check(STEP_LABEL, check_log, "every future row of a series with an estimation id takes its rate from the pool",
                             bool((forecast.loc[should_pool, "origen_tasa"] == RATE_FROM_POOL).all()),
                             failure_detail="rows with a pool that took another rate",
@@ -101,7 +102,7 @@
     last_year = closed[configuration.period_col].max().year
     last_year_rows = closed[closed[configuration.period_col].map(lambda month: month.year) == last_year]
     past_rate = last_year_rows[configuration.renewed_usd_col].sum() / last_year_rows[due].sum()
-    extract_rows = forecast[forecast["origen_pipeline"] == "real"]
+    extract_rows = forecast[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL]
     future_rate = extract_rows["esperado_usd"].sum() / extract_rows[due].sum()
     configuration.log_check(STEP_LABEL, check_log, f"the expected rate of the future is within ±{MAX_RATE_JUMP_PP:.0f} pp of {last_year}'s",
                             abs(future_rate - past_rate) * 100 <= MAX_RATE_JUMP_PP,
```

## step_19_portfolio_exam.py

```diff
--- antes/step_19_portfolio_exam.py	2026-10-01 05:40:53.898355534 +0000
+++ step_19_portfolio_exam.py	2026-10-01 05:41:33.204636497 +0000
@@ -48,8 +48,9 @@
 from config import Config, join_columns
 from step_14_backtest import band_of_horizon
 from techniques import inverse_logit, logit, predict_logit
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, METHOD_FRAMEWORK, ROLE_TEST, SERIES_ID_COLUMN,
-                         SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM, TABLE_PORTFOLIO_EXAM_SUMMARY, TRUTH_ROLES)
+from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
+                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM, TABLE_PORTFOLIO_EXAM_SUMMARY,
+                         TRUTH_ROLES)
 
 
 STEP_LABEL = "19"
@@ -91,7 +92,7 @@
     real = real.merge(series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "credibility_ref_id"]],
                       on=SERIES_ID_COLUMN, how="left")
     decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
-    pool_truth = pool_series[pool_series["rol"].isin(TRUTH_ROLES) & pool_series["tasa"].notna() & (pool_series["vencen"] > 0)]
+    pool_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
     pool_ids = set(pool_truth[ESTIMATION_ID_COLUMN])
 
     rows, latest_month_used = [], []
@@ -178,7 +179,7 @@
         if len(history) < 3:
             continue
         technique = decision.get((estimation_id, band_name), configuration.challenger_technique)
-        rates = history["tasa"].to_numpy(dtype=float)
+        rates = history[RATE_COLUMN].to_numpy(dtype=float)
         months = np.array([month.month for month in history[period_column]])
         value = predict_logit(technique, logit(rates), months, int(horizon))
         if not np.isfinite(value):
```

## step_informe.py

```diff
--- antes/step_informe.py	2026-10-01 05:40:53.897955735 +0000
+++ step_informe.py	2026-10-01 05:41:33.205452925 +0000
@@ -40,9 +40,10 @@
 import pandas as pd
 
 from config import Config, UNKNOWN_DISCOUNT_BUCKET
-from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, LEVEL_OWN, PURPOSE_SELECTION, REPORT_FILE_NAME,
-                         ROLE_PROJECTION, ROLES_IN_ORDER, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
-                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_SERIES_CARD, TRUTH_ROLES)
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ESTIMATION_ID_COLUMN, GATE_LEVEL, LEVEL_OWN,
+                         METHOD_FRAMEWORK, PURPOSE_SELECTION, REPORT_FILE_NAME, ROLES_IN_ORDER, ROLE_PROJECTION,
+                         S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN,
+                         SYNTHETIC_COLUMN, TABLE_SERIES_CARD, TOTAL_ORIGIN_TOTAL, TRUTH_ROLES, UPLIFT_CELL_ID_COLUMN)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -146,7 +147,7 @@
 def series_card(results: dict, configuration: Config) -> pd.DataFrame:
     """One row per series: route, support, own rate, ladder, estimate, level, the dynamics and
     the chosen techniques of its estimation id, and their exam error."""
-    card = results["series_estimate"].merge(results["series"][[SERIES_ID_COLUMN, "cobertura", "primer_mes", "ultimo_mes"]],
+    card = results["series_estimate"].merge(results["series"][[SERIES_ID_COLUMN, COVERAGE_COLUMN, "primer_mes", "ultimo_mes"]],
                                             on=SERIES_ID_COLUMN, how="left")
     dynamics = results.get("pool_dynamics")
     if dynamics is not None and len(dynamics):
@@ -204,7 +205,7 @@
              f"{early[S0_RENEWED_UNITS_COLUMN].sum():,.0f} unidades y ${early[S0_RENEWED_USD_COLUMN].sum():,.0f}. "
              f"El raw original queda en las columnas s0_.",
              ]
-    wiped_pipeline = fine["s0_vencen_usd"] - fine[configuration.pipeline_usd_col]
+    wiped_pipeline = fine[S0_PIPELINE_USD_COLUMN] - fine[configuration.pipeline_usd_col]
     lines.append(f"- Pipeline de licencias de 1 año vendidas o renovadas desde el mes en curso (vence desde "
                  f"{boundaries['current'] + 12}): **aún no se conoce**, se borra y se proyecta: **{int((wiped_pipeline != 0).sum()):,} "
                  f"filas**, ${wiped_pipeline.sum():,.0f} (el raw la conserva en s0_vencen_*).")
@@ -315,7 +316,7 @@
                                                        backtest["exam_by_pool"], backtest["exam_total"])
     boundaries = configuration.calendar_boundaries()
     reference = results["pool_reference"]
-    judged_share = (reference.loc[reference["gate"] == "nivel", "usd_por_predecir"].sum()
+    judged_share = (reference.loc[reference["gate"] == GATE_LEVEL, "usd_por_predecir"].sum()
                     / max(reference["usd_por_predecir"].sum(), 1))
 
     exam_by_band = exam_by_pool.merge(reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
@@ -343,8 +344,8 @@
     portfolio_summary = results.get("portfolio_exam_summary")
     if portfolio_summary is not None:
         for horizon, block in portfolio_summary.groupby("h"):
-            framework = block[block["metodo"] == "framework"].iloc[0]
-            spreadsheet = block[block["metodo"] != "framework"].sort_values("error_total_medio").iloc[0]
+            framework = block[block["metodo"] == METHOD_FRAMEWORK].iloc[0]
+            spreadsheet = block[block["metodo"] != METHOD_FRAMEWORK].sort_values("error_total_medio").iloc[0]
             headline.append((f"error del TOTAL en el examen, h = {horizon}: framework vs mejor hoja de cálculo",
                              f"{framework['error_total_medio']:.1%} vs {spreadsheet['error_total_medio']:.1%} ({spreadsheet['metodo']})"))
     if len(by_band):
@@ -383,7 +384,7 @@
     comparison, verdict = results.get("uplift_backtest"), results.get("uplift_verdict")
     if cells is None:
         return "## 6 · La revalorización\n\n_(pasos 15-16 no ejecutados)_\n"
-    by_origin = cells.groupby("uplift_origen").agg(celdas=("uplift_cell_id", "size"), uplift_medio=("uplift", "mean"),
+    by_origin = cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), uplift_medio=("uplift", "mean"),
                                                    renovadores=("renovadores", "sum")).reset_index()
     lines = ["## 6 · La revalorización: a qué precio se renueva", "",
              "El uplift es lo que paga quien renueva respecto a lo que vencía (1,00 = mismo precio). Dos vías: la "
@@ -430,7 +431,7 @@
              "**De dónde sale la tasa de las filas futuras:**", "", markdown_table(origins, 0)]
     total = results.get("forecast_total")
     if total is not None and len(total):
-        for _, row in total[total["origen"] == "TOTAL"].iterrows():
+        for _, row in total[total["origen"] == TOTAL_ORIGIN_TOTAL].iterrows():
             headline.append((f"TOTAL {int(row['ano'])} renovado + revenue time_series (pipeline {row['usd_vence']:,.0f} $)",
                              f"${row['usd_renovado']:,.0f}"))
         lines += ["**El total del forecast por año y origen** (paso 20; es la SUMA de `sff_nucleo` por `fin_ano` y `fin_origen`, "
```

## step_nucleo.py

```diff
--- antes/step_nucleo.py	2026-10-01 05:40:53.810154543 +0000
+++ step_nucleo.py	2026-10-01 05:41:33.209065920 +0000
@@ -53,11 +53,14 @@
 import pandas as pd
 
 from config import Config, join_columns
-from vocabulario import (TRUTH_ROLES as TRUTH_ROLES_OF_CORE, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, ESTIMATION_ID_COLUMN,
-                         FINE_ROWS_COLUMN, ROLE_TEST, ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_RAW,
-                         ROW_ORIGIN_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
-                         SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_CORE, TABLE_CORE_LEGEND,
-                         UNIT_ID_COLUMN, UNIVERSE_COLUMN, UPLIFT_CELL_ID_COLUMN)
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, ESTIMATION_ID_COLUMN,
+                         FINE_ROWS_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_SIMULATED, ROLE_TEST,
+                         ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
+                         S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN,
+                         S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_CORE,
+                         TABLE_CORE_LEGEND, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
+                         TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL, TS_PROJECTED, TS_REAL, TS_REENTRY,
+                         UNIT_ID_COLUMN, UNIVERSE_COLUMN, UPLIFT_CELL_ID_COLUMN, TRUTH_ROLES as TRUTH_ROLES_OF_CORE)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -97,7 +100,7 @@
                      ("s02_renovado_usd", "02", "renewed USD after the calendar: null → 0 in closed months, wiped from the current month on")]
 
 # steps 15-17, per FUTURE ROW (a value of the row: the money columns add up)
-ROW_VALUES_STEP_17 = [("origen_pipeline", "s17_origen_pipeline", "17", "real (extract) · proyectada · simulada"),
+ROW_VALUES_STEP_17 = [(PIPELINE_ORIGIN_COLUMN, "s17_origen_pipeline", "17", "real (extract) · proyectada · simulada"),
                       ("_vencen_unidades", "s17_vencen_unidades", "17", "units due of the future row: the extract's or the extended horizon's (SUM)"),
                       ("_vencen_usd", "s17_vencen_usd", "17", "USD due of the future row: the extract's or the extended horizon's (SUM)"),
                       ("h", "s17_h", "17", "months from the last closed month"),
@@ -213,9 +216,9 @@
             core = add_block(core, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
             blocks_present.append(block_specs)
     if pool_dynamics is not None and len(pool_dynamics) and "s11_final_group_id" in core.columns:
-        dynamics_values = pool_dynamics[["final_group_id"] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
+        dynamics_values = pool_dynamics[[ESTIMATION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
         dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_13})
-        core = core.merge(dynamics_values, left_on="s11_final_group_id", right_on="final_group_id", how="left").drop(columns="final_group_id")
+        core = core.merge(dynamics_values, left_on="s11_final_group_id", right_on=ESTIMATION_ID_COLUMN, how="left").drop(columns=ESTIMATION_ID_COLUMN)
         blocks_present.append(ESTIMATION_VALUES_STEP_13)
     if technique_decision is not None and "s11_final_group_id" in core.columns:
         core, backtest_specs = add_backtest_block(core, technique_decision, exam_by_pool, configuration)
@@ -270,11 +273,11 @@
 def add_backtest_block(core: pd.DataFrame, technique_decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                        configuration: Config) -> tuple:
     """The chosen technique and the exam error of the row's estimation id, one column per horizon band."""
-    per_band = technique_decision[["final_group_id", "tramo_h", "tecnica", "tecnica_origen"]]
+    per_band = technique_decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
     if exam_by_pool is not None:
-        per_band = per_band.merge(exam_by_pool[["final_group_id", "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio",
-                                                "dentro_banda"]], on=["final_group_id", "tramo_h"], how="left")
-    wide = per_band.pivot(index="final_group_id", columns="tramo_h")
+        per_band = per_band.merge(exam_by_pool[[ESTIMATION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio",
+                                                "dentro_banda"]], on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
+    wide = per_band.pivot(index=ESTIMATION_ID_COLUMN, columns="tramo_h")
     specs = []
     renamed = {}
     for source, name, step, description in ESTIMATION_VALUES_STEP_14:
@@ -293,7 +296,7 @@
 FINAL_ORIGIN_COLUMN = "fin_origen"
 FINAL_STATE_COLUMN = "fin_estado"
 FINAL_YEAR_COLUMN = "fin_ano"
-UNIVERSE_PIPELINE, UNIVERSE_TIME_SERIES = "pipeline", "time_series"
+FINAL_UNIVERSE_PIPELINE, FINAL_UNIVERSE_TIME_SERIES = "pipeline", "time_series"
 STATE_REAL, STATE_EXPECTED = "real", "previsto"
 TS_WITHOUT_RESULT = "ts_sin_resultado"       # an original time_series row of a month not closed yet (its result is projected)
 FINAL_VALUES = [
@@ -325,7 +328,7 @@
             if column_name in time_series_rows.columns and column_name not in original.columns:
                 original[column_name] = time_series_rows[column_name].to_numpy()
         closed = original[configuration.period_col] < current
-        original[ROW_ORIGIN_COLUMN] = np.where(closed, "ts_real", TS_WITHOUT_RESULT)
+        original[ROW_ORIGIN_COLUMN] = np.where(closed, TS_REAL, TS_WITHOUT_RESULT)
         original["s00_vencen_unidades"] = time_series_rows[configuration.pipeline_units_col].fillna(0).to_numpy()
         original["s00_vencen_usd"] = time_series_rows[configuration.pipeline_usd_col].fillna(0).to_numpy()
         original["s00_renovadas_unidades"] = time_series_rows[configuration.renewed_units_col].fillna(0).to_numpy()
@@ -336,7 +339,7 @@
         original["_ts_usd"] = np.where(closed, original["s00_renovado_usd"], 0.0)
         frames.append(original)
     if time_series_table is not None and len(time_series_table):
-        synthetic = time_series_table[time_series_table["origen"].isin(["ts_proyectado", "ts_reentrada"])]
+        synthetic = time_series_table[time_series_table["origen"].isin([TS_PROJECTED, TS_REENTRY])]
         rows = pd.DataFrame({ROW_ORIGIN_COLUMN: synthetic["origen"].to_numpy(),
                              configuration.period_col: synthetic[configuration.period_col].to_numpy()})
         for column_name in core_columns:
@@ -345,7 +348,7 @@
         for measure_name, _, _ in RAW_MEASURES:
             rows[measure_name] = 0.0
         rows["s02_rol"] = configuration.role_of_months(rows[configuration.period_col])
-        reentry = (synthetic["origen"] == "ts_reentrada").to_numpy()
+        reentry = (synthetic["origen"] == TS_REENTRY).to_numpy()
         rows["_ts_vence_unidades"] = np.where(reentry, synthetic["unidades"].to_numpy(), 0.0)
         rows["_ts_vence_usd"] = np.where(reentry, synthetic["valor"].to_numpy(), 0.0)      # the pipeline: value at 40 % off
         rows["_ts_renovadas"] = np.where(reentry, synthetic.get("unidades_renovadas", pd.Series(0.0, index=synthetic.index)).fillna(0).to_numpy(),
@@ -366,13 +369,13 @@
     column = lambda name: core[name].fillna(0.0) if name in core.columns else zeros
 
     raw_closed, raw_future = (origin == ROW_FROM_RAW) & closed, (origin == ROW_FROM_RAW) & ~closed
-    extended = origin.isin(["proyectada", "simulada"])
-    core[FINAL_UNIVERSE_COLUMN] = np.where(is_time_series, UNIVERSE_TIME_SERIES, UNIVERSE_PIPELINE)
+    extended = origin.isin([PIPELINE_PROJECTED, PIPELINE_SIMULATED])
+    core[FINAL_UNIVERSE_COLUMN] = np.where(is_time_series, FINAL_UNIVERSE_TIME_SERIES, FINAL_UNIVERSE_PIPELINE)
     core[FINAL_ORIGIN_COLUMN] = np.select(
-        [raw_closed, raw_future, origin == "proyectada", origin == "simulada", origin == ROW_FROM_GAP],
-        ["pipeline_renovado_real", "pipeline_real_esperado", "pipeline_proyectada", "pipeline_simulada", "hueco"],
+        [raw_closed, raw_future, origin == PIPELINE_PROJECTED, origin == PIPELINE_SIMULATED, origin == ROW_FROM_GAP],
+        [TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED, ROW_FROM_GAP],
         default=origin.astype(str))
-    core[FINAL_STATE_COLUMN] = np.where(raw_closed | (origin == ROW_FROM_GAP) | origin.isin(["ts_real", TS_WITHOUT_RESULT]),
+    core[FINAL_STATE_COLUMN] = np.where(raw_closed | (origin == ROW_FROM_GAP) | origin.isin([TS_REAL, TS_WITHOUT_RESULT]),
                                         STATE_REAL, STATE_EXPECTED)
     core[FINAL_YEAR_COLUMN] = core[configuration.period_col].map(lambda month: month.year)
 
@@ -435,7 +438,7 @@
     extension = forecast[forecast["_fila"].isna()]
     if extension.empty:
         return pd.DataFrame(columns=core_columns)
-    rows = pd.DataFrame({ROW_ORIGIN_COLUMN: extension["origen_pipeline"].to_numpy(),
+    rows = pd.DataFrame({ROW_ORIGIN_COLUMN: extension[PIPELINE_ORIGIN_COLUMN].to_numpy(),
                          configuration.period_col: extension[configuration.period_col].to_numpy()})
     for column_name in core_columns:
         if column_name in extension.columns and column_name not in rows.columns and not column_name.startswith("s"):
@@ -524,7 +527,7 @@
                             context=" · ".join(f"{origin} {count:,}" for origin, count in origin_counts.items()))
 
     # [2] the money of the pipeline reconciles with the extract (the time_series rows are apart)
-    pipeline_rows = core[FINAL_UNIVERSE_COLUMN] == UNIVERSE_PIPELINE
+    pipeline_rows = core[FINAL_UNIVERSE_COLUMN] == FINAL_UNIVERSE_PIPELINE
     reconciliation = {"s00_vencen_unidades": fine_table[S0_PIPELINE_UNITS_COLUMN].sum(),
                       "s00_vencen_usd": fine_table[S0_PIPELINE_USD_COLUMN].sum(),
                       "s02_vencen_unidades": fine_table[configuration.pipeline_units_col].sum(),
@@ -563,7 +566,7 @@
         configuration.log_not_evaluated(STEP_LABEL, check_log, "the core summed by year and origin = sff_forecast_total",
                                         "step 20 has not run")
         return
-    parts = forecast_total[forecast_total["origen"] != "TOTAL"].set_index(["ano", "origen"])
+    parts = forecast_total[forecast_total["origen"] != TOTAL_ORIGIN_TOTAL].set_index(["ano", "origen"])
     summed = core.groupby([FINAL_YEAR_COLUMN, FINAL_ORIGIN_COLUMN])[["fin_vence_usd", "fin_renovado_usd"]].sum()
     summed.index = summed.index.set_names(["ano", "origen"])
     compared = parts.join(summed, how="left").fillna(0.0)
```
