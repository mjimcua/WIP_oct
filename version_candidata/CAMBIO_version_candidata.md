# Cambio · versión candidata (1-oct-2026)

Verificación: 21 ficheros de test en verde (288 comprobaciones); el sintético corre entero por el orquestador, 25 pasos con
0 comprobaciones fallidas. El módulo único de predicción deja el forecast idéntico; la escalera mecánica lo mueve por diseño
(2026: $422.599 → $422.481; 2027: $101.477 → $101.317), porque las series "tele" predicen ahora con su hermana web.

## Qué cambia

| Pieza | Ficheros |
|---|---|
| Módulo único de predicción | `prediction.py` (nuevo); lo usan `step_17_forecast.py`, `step_19_series_exam.py`, `step_audit.py` |
| Examen único serie a serie (raw, framework, hoja) | `step_19_series_exam.py` (nuevo, sustituye a `step_19_portfolio_exam.py`) |
| Escalera mecánica, `collapse_passes`, fusiones y miembros | `step_10_ladder_groups.py`, `step_12_pool_series.py`, `config.py` |
| Niveles generados con JSON | `step_02b_dimension_levels.py` (nuevo), `config.py` |
| Hechos + dimensión | `step_nucleo.py` |
| Contratos y orquestador | `pipeline.py` (nuevo); `main.py` lo usa |
| Auditoría | `step_audit.py`: origen T − h (corregido), comprobaciones 6-7 con series que prestan, 10 contra el paso 19 |
| Informe | `step_informe.py`: framework frente a hoja y frente a la serie sola |
| Tests nuevos | `test_prediction.py`, `test_step_02b.py`, `test_pipeline.py` |

## Parámetros nuevos de la Config

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `collapse_passes` | 2 | la etapa 3 junta como mucho estas dimensiones; las pasadas siguientes solo buscan referencia de credibilidad |
| `leveled_dims` | `{}` | dimensiones con niveles generados: `{nombre: {"source": columna raw, "type": "ordinal" o "nominal"}}` |
| `levels_path` | `None` (= `<output_folder>/sff_levels.json`) | el JSON de los grupos; se reutiliza; se borra para regenerar |
| `level_merge_max_pp` | 5.0 | dos valores vecinos se juntan mientras sus tasas difieran como mucho esto |

En la Config de producción: `leveled_dims = {"tr_term": {"source": "tr_term_level_2", "type": "ordinal"},
"tr_band": {"source": "tr_band_level_2", "type": "ordinal"}}`. El `_level_1` del extracto se sobrescribe con el generado.

## Tablas nuevas

`sff_forecast_series` (dimensión), `sff_series_exam_detail`, `sff_series_exam` (ahora del paso 19), `sff_ladder_merges`,
`sff_composition_members`, `sff_dimension_levels`. `sff_nucleo` pasa de 117 a 53 columnas: lo de nivel serie vive en la
dimensión.

## En el notebook

```python
import importlib, pipeline, main
importlib.reload(pipeline); importlib.reload(main)

configuration = main.production_configuration()
orchestrator = pipeline.Orchestrator(configuration)
results = orchestrator.run()                    # todo, en el orden de sus dependencias

orchestrator.run_step("ladder")                 # un paso suelto: descarta lo que habían producido los posteriores
orchestrator.run_step("credibility")            # … y hay que volver a ejecutarlos en orden
orchestrator.context["forecast_series"]         # cualquier tabla de la ejecución
```

Si se ejecuta un paso sin sus entradas, se detiene diciendo qué paso hay que ejecutar antes.

## Ficheros

Nuevos: pipeline.py, prediction.py, step_02b_dimension_levels.py, step_19_series_exam.py, test_pipeline.py, test_prediction.py, test_step_02b.py  
Modificados: config.py, main.py, step_10_ladder_groups.py, step_12_pool_series.py, step_17_forecast.py, step_audit.py, step_informe.py, step_nucleo.py, test_step_10_11.py, test_step_12_14.py, test_step_13_informe.py, test_step_15_18.py, test_step_19.py, test_step_audit.py, test_step_nucleo.py, vocabulario.py  
Eliminados: step_19_portfolio_exam.py

## config.py

```diff
--- antes/config.py	2026-10-01 08:08:00.671203186 +0000
+++ config.py	2026-10-01 08:19:08.898490006 +0000
@@ -192,6 +192,13 @@
     flag_time_series_col: str = "flag_time_series"          # marks the rows of the time_series universe
 
     business_mandatory_dims: list = field(default_factory=list)       # open the series and the uplift cell
+    leveled_dims: dict = field(default_factory=dict)  # dims given two generated levels (step 02b): {name: {"source": raw
+                                                      # column (default: name), "type": "ordinal" | "nominal"}};
+                                                      # <name>_level_2 = the raw value, <name>_level_1 = values grouped
+                                                      # by their standardised renewal rate. The source must be mandatory
+    levels_path: Optional[str] = None                 # the JSON of the generated groups (None: <output_folder>/sff_levels.json);
+                                                      # a later run reuses it; delete it to regenerate
+    level_merge_max_pp: float = 5.0                   # two neighbouring values merge while their rates differ by at most this
     structural_timevarying_dims: dict = field(default_factory=dict)   # column → "negative" | "positive"
     extra_renovacion: list = field(default_factory=list)              # enter the rate series only
     extra_revalorizacion: list = field(default_factory=list)          # enter the uplift cell only
@@ -218,7 +225,9 @@
 
     dimension_pairs_shown: int = 20       # step 09: pairs of dimensions kept, the ones with the most interaction
     # ─── the ladder (steps 10 and 11) ───
-    own_rate_floor: float = 271.0         # contracts in a typical month to predict ALONE: ±5 pp at 90 % (p = 0.5).
+    own_rate_floor: float = 271.0
+    collapse_passes: int = 2            # stage 3 merges at most this many mandatory dims (collapse order); the passes
+                                        # after them are only searched for a credibility reference (stage 4)         # contracts in a typical month to predict ALONE: ±5 pp at 90 % (p = 0.5).
                                           # 30 says who may speak; 271 who may speak alone
     k_cred: float = 60.0                  # Bühlmann k when a relative has too few siblings to estimate it: a series
                                           # with n = 30 keeps 33 % of its own rate ("twice the floor to be believed half")
```

## main.py

```diff
--- antes/main.py	2026-10-01 08:08:00.671554774 +0000
+++ main.py	2026-10-01 08:23:26.598691850 +0000
@@ -11,36 +11,12 @@
 # ─── imports ─────────────────────────────────────────────────────────────────────
 import os
 import sys
-import time
 
 import pandas as pd
 from sqlalchemy import create_engine
 
+from pipeline import Orchestrator
 from config import Config
-from step_00_validate_raw import validate_raw
-from step_01_validate_values import validate_values
-from step_02_apply_calendar import apply_calendar
-from step_03_fine_table import build_fine_table
-from step_04_forecast_units import build_forecast_units
-from step_05_lookups import build_lookups
-from step_06_series_routes import build_series_routes
-from step_07_support_bound import build_support_bound
-from step_08_rate_series import build_rate_series
-from step_09_dimensions import analyse_dimensions
-from step_10_ladder_groups import build_ladder_groups
-from step_11_ladder import climb_the_ladder
-from step_12_pool_series import build_pool_series
-from step_13_dynamics import measure_dynamics
-from step_14_backtest import run_backtest
-from step_15_uplift import estimate_uplift
-from step_16_uplift_backtest import backtest_uplift
-from step_17_forecast import assemble_forecast
-from step_18_validation import validate_chain
-from step_19_portfolio_exam import examine_portfolio
-from step_20_time_series import build_time_series_and_total, split_time_series_rows
-from step_audit import build_audit_tables
-from step_nucleo import build_core_table
-from step_informe import build_report
 
 
 # ─── named constants ─────────────────────────────────────────────────────────────
@@ -93,6 +69,10 @@
                                  "tr_product_level_1", "tr_product_level_2", "tr_purchase_type", "tr_renewal_type",
                                  "tr_term_level_1", "tr_term_level_2", "tr_band_level_1", "tr_band_level_2",
                                  "tr_master_partner_code"],
+        # term and band: level_2 = the raw value of the extract, level_1 = generated by the library (step 02b;
+        # the extract's own level_1 is overwritten). The groups are kept in salida/sff_levels.json
+        leveled_dims={"tr_term": {"source": "tr_term_level_2", "type": "ordinal"},
+                      "tr_band": {"source": "tr_band_level_2", "type": "ordinal"}},
         structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                      "not_installed": "negative"},            # [por confirmar] the signs
         extra_renovacion=["net_new", "prev_OperationGroup"],
@@ -142,62 +122,8 @@
 # ═══════════════════════════════════════════════════════════════════════════════════
 
 def run(configuration: Config) -> dict:
-    """The steps, in order. Each step adds its line here when it is built."""
-    configuration.logger.doc("[main] ═══ SFF · reading the raw ═══")
-    read_start_time = time.time()
-    raw = configuration.read_raw()
-    configuration.logger.doc(f"[main] raw read: {len(raw):,} rows × {len(raw.columns)} columns "
-                             f"({time.time() - read_start_time:.1f}s)")
-
-    validated_raw = validate_raw(raw, configuration)                     # step 00
-    validated_raw, time_series_rows = split_time_series_rows(validated_raw, configuration)   # the time_series universe waits for step 20
-    validated_raw = validate_values(validated_raw, configuration)        # step 01
-    calendared_raw = apply_calendar(validated_raw, configuration)        # step 02
-    fine_table = build_fine_table(calendared_raw, configuration)         # step 03
-    forecast_units = build_forecast_units(fine_table, configuration)     # step 04
-    lookups = build_lookups(fine_table, forecast_units, configuration)   # step 05
-    series_table = build_series_routes(forecast_units, configuration)    # step 06
-    support_bound = build_support_bound(forecast_units, configuration)   # step 07
-    rated_units, series_rate = build_rate_series(forecast_units, series_table, configuration)   # step 08
-    dimension_decision, dimension_pairs = analyse_dimensions(series_rate, lookups["lookup_fs"], configuration)   # step 09
-    ladder = build_ladder_groups(rated_units, series_rate, lookups["lookup_fs"], dimension_decision,
-                                 configuration)                                                   # step 10
-    series_estimate, money_by_level = climb_the_ladder(series_rate, ladder, configuration)   # step 11
-    pool_series, pool_reference = build_pool_series(rated_units, ladder, series_estimate, configuration)   # step 12
-    pool_dynamics, portfolio_profile, portfolio_dynamics = measure_dynamics(pool_series, pool_reference, configuration)   # step 13
-    backtest = run_backtest(pool_series, pool_reference, configuration)                            # step 14
-    uplift_cells, contract_check = estimate_uplift(fine_table, configuration)                       # step 15
-    uplift_backtest, uplift_verdict = backtest_uplift(fine_table, configuration)                     # step 16
-    forecast = assemble_forecast(fine_table, forecast_units, series_estimate, pool_series, pool_reference, backtest,
-                                 uplift_cells, uplift_verdict, rated_units, configuration)          # step 17
-    results = dict(pool_series=pool_series, pool_reference=pool_reference,
-                   backtest=backtest, pool_dynamics=pool_dynamics, portfolio_profile=portfolio_profile,
-                   portfolio_dynamics=portfolio_dynamics, ladder=ladder,
-                   series_estimate=series_estimate, money_by_level=money_by_level,
-                   dimension_decision=dimension_decision, dimension_pairs=dimension_pairs,
-                   raw=calendared_raw, fine_table=fine_table, forecast_units=forecast_units, lookups=lookups,
-                   series=series_table, support_bound=support_bound, rated_units=rated_units, series_rate=series_rate)
-    results.update(uplift_cells=uplift_cells, contract_check=contract_check, uplift_backtest=uplift_backtest,
-                   uplift_verdict=uplift_verdict, forecast=forecast)
-    results["portfolio_exam"], results["portfolio_exam_summary"] = examine_portfolio(
-        rated_units, series_estimate, pool_series, backtest, configuration,
-        reference_members=ladder["reference_members"])                                               # step 19
-    results["time_series"], results["forecast_total"] = build_time_series_and_total(
-        time_series_rows, fine_table, forecast["forecast"], configuration)                      # step 20
-    results["time_series_rows"] = time_series_rows
-    results["audit"] = build_audit_tables(ladder, series_rate, series_estimate, rated_units, pool_series, pool_reference,
-                                          pool_dynamics, backtest, forecast["forecast"], configuration)   # the satellites
-    results["core"], results["core_legend"] = build_core_table(
-        fine_table, configuration, forecast_units=forecast_units, support_bound=support_bound, rated_units=rated_units,
-        series_table=series_table, series_rate=series_rate, series_estimate=series_estimate, pool_dynamics=pool_dynamics,
-        technique_decision=backtest["decision"], exam_by_pool=backtest["exam_by_pool"], forecast=forecast["forecast"],
-        time_series_rows=time_series_rows, time_series_table=results["time_series"],
-        forecast_total=results["forecast_total"], ladder_stages=ladder["stages"],
-        series_dynamics=results["audit"]["series_dynamics"], series_exam=results["audit"]["series_exam"])
-                                                 # the core: every row, every decision
-    results["validation"] = validate_chain(raw, results, configuration)                            # step 18 (after 19: it reads its exam)
-    results["card"] = build_report(raw, results, configuration)                                   # the report, last
-    return results
+    """Every step, in the order of their dependencies (pipeline.py: contracts, step interface, orchestrator)."""
+    return Orchestrator(configuration).run()
 
 
 if __name__ == "__main__":
```

## step_10_ladder_groups.py

```diff
--- antes/step_10_ladder_groups.py	2026-10-01 08:08:00.952287250 +0000
+++ step_10_ladder_groups.py	2026-10-01 08:28:30.285969274 +0000
@@ -72,8 +72,9 @@
 
 from config import ID_FIELD_SEPARATOR, Config, id_text
 from vocabulario import (COMPOSITION_ID_COLUMN, RATE_COLUMN, ROUTE_COLUMN, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
-                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_LADDER_GROUPS, TABLE_LADDER_STEPS,
-                         TABLE_LADDER_SUMMARY, UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)
+                         SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, TABLE_COMPOSITION_MEMBERS,
+                         TABLE_LADDER_GROUPS, TABLE_LADDER_MERGES, TABLE_LADDER_STEPS, TABLE_LADDER_SUMMARY,
+                         UNIVERSE_COLUMN, UNIVERSE_NORMAL, WILDCARD)
 
 
 STEP_LABEL = "10"
@@ -114,9 +115,13 @@
 
     # [1] the plan of the passes
     plan = plan_of_the_passes(dimension_decision, configuration)
-    configuration.log_action(STEP_LABEL, 1, f"{len(plan)} passes: " + " → ".join(step["name"] for step in plan)
-                             + f" · a signed group may lose up to {configuration.signed_ladder_max_loss} of R² "
-                               f"· floor {configuration.support_floor:.0f} · own-rate floor {configuration.own_rate_floor:.0f}")
+    merging = [step["name"] for step in plan if step["merges"]]
+    reference_only = [step["name"] for step in plan if not step["merges"]]
+    configuration.log_action(STEP_LABEL, 1, f"{len(merging)} merging passes: " + " → ".join(merging)
+                             + (f" · searched only for a credibility reference: {', '.join(reference_only)}" if reference_only else "")
+                             + f" · stage 3 merges at most {configuration.collapse_passes} dims · a signed group may lose up to "
+                               f"{configuration.signed_ladder_max_loss} of R² · floor {configuration.support_floor:.0f} · "
+                               f"own-rate floor {configuration.own_rate_floor:.0f}")
 
     # [2] the estimable series
     estimable = series_rate[(series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE)
@@ -135,14 +140,22 @@
     configuration.log_action(STEP_LABEL, 3, f"{len(plan)} patterns per series · {len(history):,} history months")
 
     # [4] the passes
-    steps, summary, final = run_the_passes(estimable, patterns, plan, history, configuration)
+    passes = run_the_passes(estimable, patterns, plan, history, configuration)
+    steps, summary, final = passes["steps"], passes["summary"], passes["final"]
     stages = stages_of_the_series(steps, plan)
-    configuration.log_action(STEP_LABEL, 4, f"{summary['ladder_step'].max() + 1} passes run · groups "
-                                            f"{summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
-                                            f"closed at the end: {int(final['closed'].sum()):,} of {len(final):,} series")
+    members = composition_members(estimable, patterns, final)
+    merges = ladder_merges(steps, series_rate, passes["pattern_stats"], patterns)
+    lenders = members.merge(final[[COMPOSITION_ID_COLUMN]].reset_index(), on=[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN], how="left",
+                            indicator=True)
+    members["role"] = np.where(lenders["_merge"].to_numpy() == "both", "uses", "lends")
+    improving = int(merges["improves"].sum()) if len(merges) else 0
+    configuration.log_action(STEP_LABEL, 4, f"{len(summary)} passes run · ids {summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
+                                            f"{int(final['closed'].sum()):,} of {len(final):,} series reach the floor · "
+                                            f"{len(merges):,} merges, {improving:,} improve the error of the series' monthly rate · "
+                                            f"{int((members['role'] == 'lends').sum()):,} series lend their history to a composition they do not use")
 
     # [5] the final groups and their references
-    groups, reference_members = credibility_references(estimable, patterns, plan, final, history, configuration)
+    groups, reference_members = credibility_references(estimable, patterns, plan, final, passes["pattern_stats"], configuration)
     with_reference = groups["credibility_ref_id"].notna()
     configuration.log_action(STEP_LABEL, 5, f"{groups[COMPOSITION_ID_COLUMN].nunique():,} final groups · "
                                             f"{int(with_reference.sum()):,} series take a credibility reference "
@@ -157,6 +170,8 @@
     configuration.write_table(STEP_LABEL, check_log, steps, TABLE_LADDER_STEPS)
     configuration.write_table(STEP_LABEL, check_log, summary, TABLE_LADDER_SUMMARY)
     configuration.write_table(STEP_LABEL, check_log, groups, TABLE_LADDER_GROUPS)
+    configuration.write_table(STEP_LABEL, check_log, merges, TABLE_LADDER_MERGES)
+    configuration.write_table(STEP_LABEL, check_log, members, TABLE_COMPOSITION_MEMBERS)
 
     # [8] the count of the checks; stop if anything failed
     configuration.log_action(STEP_LABEL, 8, "counting the checks")
@@ -171,7 +186,7 @@
     summary_by_stage = stage_summary(stages, estimable, history, configuration)
     configuration.show_table(summary_by_stage)
     return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members, stages=stages,
-                stage_summary=summary_by_stage)
+                stage_summary=summary_by_stage, composition_members=members, merges=merges)
 
 
 # ═══════════════════════════════════════════════════════════════════════════════════
@@ -198,13 +213,17 @@
         plan.append(dict(step=len(plan), name=f"extra {extra}", summarise_sign=True, starred=frozenset(starred),
                          signed_allowed=True))
     cumulative_loss = 0.0
-    for dimension in collapse_order:
+    for collapse_index, dimension in enumerate(collapse_order, start=1):
         starred = starred | {dimension}
         cumulative_loss += float(collapse_loss.get(dimension, 1.0))
         signed_allowed = (configuration.signed_ladder_max_loss > 0
                           and cumulative_loss <= configuration.signed_ladder_max_loss + SUPPORT_TOLERANCE)
+        # stage 3 merges at most collapse_passes dims; the passes after them are only searched for a
+        # credibility reference (stage 4), never to merge
         plan.append(dict(step=len(plan), name=f"without {dimension}", summarise_sign=True, starred=frozenset(starred),
-                         signed_allowed=signed_allowed))
+                         signed_allowed=signed_allowed, merges=collapse_index <= configuration.collapse_passes))
+    for step in plan:
+        step.setdefault("merges", True)
     return plan
 
 
@@ -254,66 +273,121 @@
 # ═══════════════════════════════════════════════════════════════════════════════════
 
 def run_the_passes(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, history: pd.DataFrame,
-                   configuration: Config) -> tuple:
-    """Pass after pass: the open groups move one pass up, the closed ones keep their id."""
+                   configuration: Config) -> dict:
+    """The mechanical rule: at every pass EVERY forecast series is grouped by the same pattern (its dims
+    with the ones removed so far set to '*'); each series uses the first pass whose group reaches the
+    support floor. A series that reaches it on its own (pass 0) keeps its own id, but its history still
+    counts in the groups of the others. Mixed series never merge. A series that never reaches the floor
+    keeps the widest group it was allowed."""
     floor = configuration.support_floor
     mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
     neutral = estimable[SIGN_COLUMN] == SIGN_NEUTRAL
-    group = patterns[0].copy()                      # pass 0: every series is its own group
-    assigned_at = pd.Series(0, index=estimable.index)
-    step_rows, summary_rows = [], []
-    usd_to_predict = estimable["usd_por_predecir"]
-    total_usd = usd_to_predict.sum()
+    merge_passes = [step for step in plan if step["merges"]]
 
+    # the support, rate and number of series of every pattern of every pass, over ALL the series
+    pattern_stats = {}
     for step in plan:
-        previous_assigned_at = assigned_at.copy()
-        if step["step"] > 0:
-            support_before = support_of_groups(history, group, configuration)["support"]
-            open_series = group.map(support_before).fillna(0.0) < floor - SUPPORT_TOLERANCE
-            may_move = open_series & ~mixed & (neutral | step["signed_allowed"])
-            if not may_move.any():
-                break
-            previous_group = group.copy()
-            group[may_move] = patterns.loc[may_move, step["step"]]
-            assigned_at[may_move] = step["step"]
-            # a group that keeps exactly the same series keeps the id it had: its id only changes
-            # when it really joins other series
-            unchanged = groups_that_did_not_change(previous_group, group, may_move)
-            for new_id, old_id in unchanged.items():
-                same_group = may_move & (group == new_id)
-                group[same_group] = old_id
-                assigned_at[same_group] = previous_assigned_at[same_group]
-        groups_now = support_of_groups(history, group, configuration)
-        support_of_series = group.map(groups_now["support"]).fillna(0.0)
-        closed = support_of_series >= floor - SUPPORT_TOLERANCE
-        step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"],
-                                       "step_name": step["name"], "group_id": group.to_numpy(copy=True),   # a copy: the
-                                       # group of later passes is written over the same Series
-                                       "group_support": support_of_series.to_numpy(), "closed": closed.astype(int).to_numpy()}))
+        pattern_of_series = patterns[step["step"]] if step["step"] == 0 else patterns[step["step"]].where(~mixed)
+        pattern_stats[step["step"]] = support_of_groups(history, pattern_of_series, configuration)
+
+    # the pass every series uses: the first one (allowed for its sign) whose group reaches the floor
+    chosen_step = pd.Series(np.nan, index=estimable.index)
+    last_allowed = pd.Series(0, index=estimable.index)
+    for step in merge_passes:
+        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
+        last_allowed[allowed] = step["step"]
+        support = patterns[step["step"]].map(pattern_stats[step["step"]]["support"]).fillna(0.0)
+        reaches = allowed & chosen_step.isna() & (support >= floor - SUPPORT_TOLERANCE)
+        chosen_step[reaches] = step["step"]
+    reached = chosen_step.notna()
+    final_step = chosen_step.fillna(last_allowed).astype(int)
+
+    # the id of every series at every pass: the pattern while it climbs, frozen once it has its group;
+    # an id only changes when the group really gets new series (the same series: the previous id)
+    ids, step_rows, summary_rows = {}, [], []
+    usd_to_predict = estimable["usd_por_predecir"]
+    total_usd = usd_to_predict.sum()
+    previous_id, previous_count = None, None
+    for step in merge_passes:
+        climbing = step["step"] <= final_step
+        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
+        moving = climbing & allowed
+        pattern = patterns[step["step"]]
+        count = pattern.map(pattern_stats[step["step"]]["series"]).fillna(1)
+        if previous_id is None:
+            current_id = pattern.copy()
+        else:
+            current_id = previous_id.copy()
+            new_series_joined = moving & (count > previous_count)
+            current_id[new_series_joined] = pattern[new_series_joined]
+        current_count = count.where(moving, previous_count if previous_count is not None else count)
+        support = pattern.map(pattern_stats[step["step"]]["support"]).fillna(0.0)
+        if previous_id is not None:
+            support = support.where(moving, step_rows[-1].set_index(SERIES_ID_COLUMN)["group_support"])
+        ids[step["step"]] = current_id
+        has_group = step["step"] >= final_step
+        step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"], "step_name": step["name"],
+                                       "group_id": current_id.to_numpy(copy=True), "group_support": support.to_numpy(copy=True),
+                                       "series_in_group": current_count.to_numpy(copy=True),
+                                       "closed": (has_group & (support >= floor - SUPPORT_TOLERANCE)).astype(int).to_numpy()}))
+        partition = support_of_groups(history, current_id, configuration)
+        group_support = pd.Series(support.to_numpy(), index=current_id.to_numpy()).groupby(level=0).first()
         summary_rows.append({
-            "ladder_step": step["step"], "step_name": step["name"], "groups": int(group.nunique()),
-            "open_groups": int((groups_now["support"] < floor - SUPPORT_TOLERANCE).sum()),
-            "median_group_support": float(groups_now["support"].median()),
-            "units_due": float(groups_now["due"].sum()),
-            "pct_usd_floor": float(usd_to_predict[closed].sum() / total_usd) if total_usd else np.nan,
-            "pct_usd_own_rate": float(usd_to_predict[support_of_series >= configuration.own_rate_floor].sum() / total_usd)
-                                if total_usd else np.nan})
-    steps = pd.concat(step_rows, ignore_index=True)
-    final = pd.DataFrame({COMPOSITION_ID_COLUMN: group, "final_step": assigned_at,
-                          "closed": group.map(support_of_groups(history, group, configuration)["support"]).fillna(0.0)
-                                    >= floor - SUPPORT_TOLERANCE})
-    return steps, pd.DataFrame(summary_rows), final
-
-
-def groups_that_did_not_change(previous_group: pd.Series, group: pd.Series, moved: pd.Series) -> dict:
-    """{new id: old id} for every new group made of exactly one old group, whole (the same series)."""
-    moved_rows = pd.DataFrame({"new_id": group[moved], "old_id": previous_group[moved]})
-    if moved_rows.empty:
-        return {}
-    per_new_group = moved_rows.groupby("new_id")["old_id"].agg(["nunique", "first", "size"])
-    size_of_old_group = previous_group.value_counts()
-    same_series = (per_new_group["nunique"] == 1) & (per_new_group["size"] == per_new_group["first"].map(size_of_old_group))
-    return dict(zip(per_new_group.index[same_series], per_new_group.loc[same_series, "first"]))
+            "ladder_step": step["step"], "step_name": step["name"], "groups": int(current_id.nunique()),
+            "open_groups": int((group_support < floor - SUPPORT_TOLERANCE).sum()),
+            "median_group_support": float(group_support.median()),
+            "units_due": float(partition["due"].sum()),
+            "pct_usd_floor": float(usd_to_predict[support >= floor - SUPPORT_TOLERANCE].sum() / total_usd) if total_usd else np.nan,
+            "pct_usd_own_rate": float(usd_to_predict[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
+        previous_id, previous_count = current_id, current_count
+
+    # the composition of every series: its id at the pass it uses, with the series in its rate
+    composition = ids[merge_passes[-1]["step"]]
+    final_pattern = pd.Series(patterns.to_numpy()[np.arange(len(patterns)), patterns.columns.get_indexer(final_step.to_numpy())],
+                              index=estimable.index)
+    final = pd.DataFrame({COMPOSITION_ID_COLUMN: composition, "final_step": final_step, "final_pattern": final_pattern,
+                          "closed": reached})
+    return dict(steps=pd.concat(step_rows, ignore_index=True), summary=pd.DataFrame(summary_rows), final=final,
+                pattern_stats=pattern_stats)
+
+
+def composition_members(estimable: pd.DataFrame, patterns: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
+    """Every series in the rate of every composition: the series whose pattern, at the pass the composition
+    uses, is the composition's (the series that use it and the ones that only lend their history)."""
+    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
+    rows = []
+    for step_number, compositions in final.drop_duplicates([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]).groupby("final_step"):
+        # every series of that pass with its pattern, then joined to the compositions that use that pass
+        candidates = patterns[step_number] if step_number == 0 else patterns[step_number][~mixed]
+        series_of_pattern = pd.DataFrame({"final_pattern": candidates.to_numpy(), SERIES_ID_COLUMN: candidates.index})
+        rows.append(compositions[[COMPOSITION_ID_COLUMN, "final_pattern"]].merge(series_of_pattern, on="final_pattern")
+                    [[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN]])
+    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
+
+
+def ladder_merges(steps: pd.DataFrame, series_rate: pd.DataFrame, pattern_stats: dict, patterns: pd.DataFrame) -> pd.DataFrame:
+    """Every merge of every forecast series (a pass where its id changed): the group before and after,
+    and whether it improves the prediction of its monthly rate: error alone √(p(1−p)/n) against error
+    merged √(bias² + q(1−q)/N), bias = rate of the new group − its own rate."""
+    ordered = steps.sort_values([SERIES_ID_COLUMN, "ladder_step"])
+    ordered = ordered.assign(previous_id=ordered.groupby(SERIES_ID_COLUMN)["group_id"].shift(1),
+                             previous_support=ordered.groupby(SERIES_ID_COLUMN)["group_support"].shift(1))
+    changes = ordered[ordered["previous_id"].notna() & (ordered["group_id"] != ordered["previous_id"])].copy()
+    if changes.empty:
+        return changes
+    own = series_rate.set_index(SERIES_ID_COLUMN)[["n_propio", "tasa_propia"]]
+    changes = changes.join(own, on=SERIES_ID_COLUMN)
+    changes["group_rate"] = [pattern_stats[int(step_number)]["rate"].get(patterns.loc[series_id, int(step_number)], np.nan)
+                             for series_id, step_number in zip(changes[SERIES_ID_COLUMN], changes["ladder_step"])]
+    own_rate, own_n = changes["tasa_propia"], changes["n_propio"].clip(lower=1)
+    group_rate, group_n = changes["group_rate"], changes["group_support"].clip(lower=1)
+    changes["bias_pp"] = 100 * (group_rate - own_rate)
+    changes["error_alone_pp"] = 100 * np.sqrt(own_rate * (1 - own_rate) / own_n)
+    changes["error_merged_pp"] = 100 * np.sqrt((group_rate - own_rate) ** 2 + group_rate * (1 - group_rate) / group_n)
+    changes["improves"] = (changes["error_merged_pp"] < changes["error_alone_pp"]).astype(int)
+    return changes[[SERIES_ID_COLUMN, "ladder_step", "step_name", "previous_id", "previous_support", "group_id", "group_support",
+                    "series_in_group", "n_propio", "tasa_propia", "group_rate", "bias_pp", "error_alone_pp", "error_merged_pp",
+                    "improves"]].rename(columns={"n_propio": "own_support", "tasa_propia": "own_rate"})
 
 
 # ═══════════════════════════════════════════════════════════════════════════════════
@@ -375,57 +449,49 @@
 # ═══════════════════════════════════════════════════════════════════════════════════
 
 def credibility_references(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, final: pd.DataFrame,
-                           history: pd.DataFrame, configuration: Config) -> tuple:
-    """The final group of every series with its support and rate, and, below the own-rate floor,
-    its credibility reference with its support and rate; and the members of every reference."""
+                           pattern_stats: dict, configuration: Config) -> tuple:
+    """The composition of every series with its support, rate and series (every series in its rate) and,
+    below the own-rate floor, its credibility reference: the first later pass (merging or not: stage 4 may
+    look wider than stage 3 merges) allowed for its sign whose group has more series and reaches the floor,
+    else the widest with more series. And the members of every reference."""
     floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
-    groups_support = support_of_groups(history, final[COMPOSITION_ID_COLUMN], configuration)
-    not_mixed = estimable[SIGN_COLUMN] != SIGN_MIXED
-
-    # every pattern of every pass, counted over ALL the series (the closed and the big ones too)
-    all_patterns = {}
-    for step in plan:
-        pattern_of_series = patterns[step["step"]].where(not_mixed)
-        all_patterns[step["step"]] = support_of_groups(history, pattern_of_series, configuration)
-
+    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
+    users = final.groupby(COMPOSITION_ID_COLUMN).size()
     group_rows = []
-    for group_id, members in final.groupby(COMPOSITION_ID_COLUMN):
-        group_support = float(groups_support.loc[group_id, "support"]) if group_id in groups_support.index else 0.0
-        group_series = int(len(members))
+    for (composition_id, step_number, pattern), members in final.groupby([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]):
+        stats = pattern_stats[int(step_number)].loc[pattern]
         first_member = members.index[0]
         sign = estimable.loc[first_member, SIGN_COLUMN]
         chosen_step, chosen_id = None, None
-        if group_support < own_rate_floor - SUPPORT_TOLERANCE and sign != SIGN_MIXED:
-            assigned_step = int(members["final_step"].iloc[0])
-            candidates = [assigned_step] + [step["step"] for step in plan if step["step"] > assigned_step
-                                            and (sign == SIGN_NEUTRAL or step["signed_allowed"])]
+        if stats["support"] < own_rate_floor - SUPPORT_TOLERANCE and sign != SIGN_MIXED:
             widest = None
-            for candidate in candidates:
-                candidate_id = patterns.loc[first_member, candidate]
-                candidate_stats = all_patterns[candidate].loc[candidate_id]
-                if candidate_stats["series"] > group_series:
-                    widest = (candidate, candidate_id)
-                    if candidate_stats["support"] >= floor - SUPPORT_TOLERANCE:
+            for step in plan:
+                if step["step"] <= step_number or not (sign == SIGN_NEUTRAL or step["signed_allowed"]):
+                    continue
+                candidate_id = patterns.loc[first_member, step["step"]]
+                candidate = pattern_stats[step["step"]].loc[candidate_id]
+                if candidate["series"] > stats["series"]:
+                    widest = (step["step"], candidate_id)
+                    if candidate["support"] >= floor - SUPPORT_TOLERANCE:
                         break
             if widest is not None:
                 chosen_step, chosen_id = widest
-        row = {COMPOSITION_ID_COLUMN: group_id, "final_step": int(members["final_step"].iloc[0]),
-               "group_series": group_series, "group_support": group_support,
-               "group_rate": float(groups_support.loc[group_id, "rate"]) if group_id in groups_support.index else np.nan,
-               "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
+        row = {COMPOSITION_ID_COLUMN: composition_id, "final_step": int(step_number), "group_series": int(stats["series"]),
+               "composition_users": int(users.get(composition_id, 0)), "group_support": float(stats["support"]),
+               "group_rate": float(stats["rate"]), "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
         if chosen_id is not None:
-            stats = all_patterns[chosen_step].loc[chosen_id]
-            row.update(ref_series=int(stats["series"]), ref_support=float(stats["support"]), ref_rate=float(stats["rate"]))
+            reference = pattern_stats[chosen_step].loc[chosen_id]
+            row.update(ref_series=int(reference["series"]), ref_support=float(reference["support"]), ref_rate=float(reference["rate"]))
         group_rows.append(row)
-    by_group = pd.DataFrame(group_rows)
+    by_composition = pd.DataFrame(group_rows)
     for column_name in ("ref_series", "ref_support", "ref_rate"):
-        if column_name not in by_group.columns:
-            by_group[column_name] = np.nan
+        if column_name not in by_composition.columns:
+            by_composition[column_name] = np.nan
 
-    groups = final[[COMPOSITION_ID_COLUMN]].reset_index().merge(by_group, on=COMPOSITION_ID_COLUMN, how="left")
+    groups = final[[COMPOSITION_ID_COLUMN]].reset_index().merge(by_composition, on=COMPOSITION_ID_COLUMN, how="left")
     reference_rows = []
-    for (ref_step, ref_id), _ in by_group.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
-        members = patterns.index[(patterns[int(ref_step)] == ref_id) & not_mixed]
+    for (ref_step, ref_id), _ in by_composition.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
+        members = patterns.index[(patterns[int(ref_step)] == ref_id) & ~mixed]
         reference_rows.append(pd.DataFrame({"credibility_ref_id": ref_id, SERIES_ID_COLUMN: members}))
     reference_members = (pd.concat(reference_rows, ignore_index=True) if reference_rows
                          else pd.DataFrame(columns=["credibility_ref_id", SERIES_ID_COLUMN]))
```

## step_12_pool_series.py

```diff
--- antes/step_12_pool_series.py	2026-10-01 08:08:00.826758866 +0000
+++ step_12_pool_series.py	2026-10-01 08:17:19.193173351 +0000
@@ -16,7 +16,7 @@
   · soporte  support below the floor: not judged, it will take the challenger
 
 Actions (logged as they are done):
-  1. the final groups of step 10 and their series
+  1. the compositions of step 10 and every series in their rate (users and lenders)
   2. the monthly series of every id (units due and renewed summed; rate)
   3. the reference of every id: months, support, rate, series, money, gate
   4. check the series and the reference                             checks 1-3
@@ -48,7 +48,7 @@
 STEP_NAME = "POOL SERIES"
 STEP_PURPOSE = ("build the monthly series of the rate of every final group of the ladder, summing every "
                 "series in it; this is the series the backtest judges and the forecast extends")
-STEP_ACTIONS = ["the final groups of step 10 and their series",
+STEP_ACTIONS = ["the compositions of step 10 and every series in their rate (users and lenders)",
                 "the monthly series of every id (units due and renewed summed; rate)",
                 "the reference of every id: months, support, rate, series, money, gate",
                 "check the series and the reference (checks 1-3)",
@@ -68,12 +68,14 @@
     check_log = []
     period_column = configuration.period_col
 
-    # [1] the final groups of step 10 (the estimable series) and their series: a partition
-    membership = ladder["groups"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]].rename(columns={COMPOSITION_ID_COLUMN: COMPOSITION_ID_COLUMN})
+    # [1] the compositions of step 10 and every series in their rate
+    # every series in the rate of every composition: the ones that use it and the ones that only lend their history
+    membership = ladder["composition_members"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]]
     chosen = series_estimate[series_estimate[SERIES_ID_COLUMN].isin(set(membership[SERIES_ID_COLUMN]))]
     chosen_ids = set(membership[COMPOSITION_ID_COLUMN])
-    configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} final groups of {len(chosen):,} series "
-                                            f"(each series in one group)")
+    configuration.log_action(STEP_LABEL, 1, f"{len(chosen_ids):,} compositions · {len(membership):,} series in their rates "
+                                            f"({membership[SERIES_ID_COLUMN].nunique():,} distinct: a series may lend its history "
+                                            f"to a composition it does not use)")
 
     # [2] the monthly series: every matching series summed, month by month
     history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN,
```

## step_17_forecast.py

```diff
--- antes/step_17_forecast.py	2026-10-01 08:08:00.901081703 +0000
+++ step_17_forecast.py	2026-10-01 08:08:51.391555799 +0000
@@ -78,6 +78,7 @@
 
 from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
 from step_14_backtest import band_of_horizon
+from prediction import band_quantiles, predict_composition, rate_band, shifted_rate
 from techniques import inverse_logit, logit, predict_logit
 from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PATH_CONTRACT, PATH_STATISTICAL,
                          PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_COLUMN,
@@ -380,30 +381,22 @@
 # THE RATE
 # ═══════════════════════════════════════════════════════════════════════════════════
 
-def judged_horizon(horizon: int, judged: list) -> int:
-    """The judged horizon whose band a horizon takes: itself, the next one above, or the last."""
-    above = [judged_h for judged_h in sorted(judged) if judged_h >= horizon]
-    return above[0] if above else max(judged)
-
-
 def pool_predictions(pool_series: pd.DataFrame, backtest: dict, horizons: list, configuration: Config) -> pd.DataFrame:
     """The rate of every estimation id at every future horizon, with the technique of its band,
     learning from every closed month of the id."""
     decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
     truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
     rows = []
-    for estimation_id, monthly in truth.groupby(COMPOSITION_ID_COLUMN):
+    for composition_id, monthly in truth.groupby(COMPOSITION_ID_COLUMN):
         monthly = monthly.sort_values(configuration.period_col)
         history = logit(monthly[RATE_COLUMN].to_numpy(dtype=float))
         calendar_months = np.array([month.month for month in monthly[configuration.period_col]])
         for horizon in horizons:
             band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
-            technique = decision.get((estimation_id, band_name), configuration.challenger_technique)
-            predicted = predict_logit(technique, history, calendar_months, int(horizon))
-            if not np.isfinite(predicted):
-                technique = configuration.challenger_technique
-                predicted = predict_logit(technique, history, calendar_months, int(horizon))
-            rows.append((estimation_id, int(horizon), technique, float(inverse_logit(predicted))))
+            technique = decision.get((composition_id, band_name), configuration.challenger_technique)
+            technique, rate = predict_composition(history, calendar_months, technique, int(horizon),
+                                                  configuration.challenger_technique)
+            rows.append((composition_id, int(horizon), technique, rate))
     return pd.DataFrame(rows, columns=[COMPOSITION_ID_COLUMN, "h", "tecnica", "tasa_pool_h"])
 
 
@@ -419,15 +412,13 @@
     future = future.merge(pool_rates, on=[COMPOSITION_ID_COLUMN, "h"], how="left")
     future = future.merge(pool_reference[[COMPOSITION_ID_COLUMN, "tasa_pool"]], on=COMPOSITION_ID_COLUMN, how="left")
 
-    # the group's predicted rate, moved toward its credibility reference by (1 − z) of the difference
-    # of levels (logit scale): the same blend as step 11, applied to the prediction
+    # the composition's predicted rate, moved toward its credibility reference (prediction.py: the same
+    # computation as the exam and the audit)
     pool_rate = future["tasa_pool_h"]
-    shift = np.zeros(len(future))
-    if configuration.apply_credibility_shift:
-        blended = (future["z"] < 1) & future["ref_rate"].notna() & future["group_rate"].notna()
-        shift = np.where(blended, (1 - future["z"].fillna(1)) * (logit(future["ref_rate"].fillna(0.5))
-                                                                  - logit(future["group_rate"].fillna(0.5))), 0.0)
-    future["tasa"] = np.where(pool_rate.notna(), inverse_logit(logit(pool_rate.fillna(0.5)) + shift), np.nan)
+    rate, shift = shifted_rate(pool_rate, future["z"], future["ref_rate"], future["group_rate"],
+                               configuration.apply_credibility_shift)
+    future["credibility_shift_logit"] = shift
+    future["tasa"] = np.where(pool_rate.notna(), rate, np.nan)
     future["origen_tasa"] = np.where(pool_rate.notna(), RATE_FROM_POOL, None)
 
     # no pool: the rate of the mandatory cell, then the global rate (closed months)
@@ -444,20 +435,11 @@
     future["tasa"] = future["tasa"].fillna(global_rate)
     future["tecnica"] = future["tecnica"].fillna(configuration.challenger_technique)
 
-    # the band: quantiles of the normalised error × binomial error with the unit's units due
-    bands = backtest["bands"].set_index(["tecnica", "h"])
-    judged = list(configuration.backtest_horizons)
+    # the band: quantiles of the normalised error × binomial error with the unit's units due (prediction.py)
     unit_units = future[UNIT_ID_COLUMN].map(forecast_units.set_index(UNIT_ID_COLUMN)[configuration.pipeline_units_col])
     unit_units = unit_units.fillna(future[configuration.pipeline_units_col])     # an extended row: its own units
-    se_rate = np.sqrt(future["tasa"] * (1 - future["tasa"]) / unit_units.clip(lower=1))
-    keys = list(zip(future["tecnica"], future["h"].map(lambda horizon: judged_horizon(int(horizon), judged))))
-    challenger_keys = [(configuration.challenger_technique, judged_h) for _, judged_h in keys]
-    q_low = np.array([bands["q_low_norm"].get(key, bands["q_low_norm"].get(fallback, -configuration.z))
-                      for key, fallback in zip(keys, challenger_keys)])
-    q_high = np.array([bands["q_high_norm"].get(key, bands["q_high_norm"].get(fallback, configuration.z))
-                       for key, fallback in zip(keys, challenger_keys)])
-    future["tasa_baja"] = np.clip(future["tasa"] + np.minimum(q_low, 0) * se_rate, 0, 1)
-    future["tasa_alta"] = np.clip(future["tasa"] + np.maximum(q_high, 0) * se_rate, 0, 1)
+    q_low, q_high = band_quantiles(backtest["bands"], future["tecnica"], future["h"], configuration)
+    future["tasa_baja"], future["tasa_alta"] = rate_band(future["tasa"], unit_units, q_low, q_high)
     return future.drop(columns=["_celda"])
 
 
```

## step_audit.py

```diff
--- antes/step_audit.py	2026-10-01 08:08:00.827181865 +0000
+++ step_audit.py	2026-10-01 08:17:19.192797224 +0000
@@ -13,7 +13,6 @@
   sff_series_backtest            forecast series × exam month × h × technique s03_fs_id
   sff_series_technique_summary   forecast series × band × technique           s03_fs_id
   sff_composition_forecast_all   composition × future horizon × technique     s10_stage3_id
-  sff_series_exam                forecast series (the chosen technique)       s03_fs_id
 
 EVERY TEST HAS ITS INTERVAL, built as the forecast builds its band: the error quantiles of the
 technique in the backtest (step 14) × the binomial error with the series' own units due. A test
@@ -35,8 +34,8 @@
   4. sff_series_dynamics                                              check 5
   5. sff_series_backtest and its summary                              checks 6-8
   6. sff_composition_forecast_all                                     check 9
-  7. sff_series_exam: the exam of the chosen technique per series     checks 10-11
-  8. write the audit tables                                           checks 12-20
+  7. the chosen technique against step 19's framework (one way of predicting)  check 10
+  8. write the audit tables                                           checks 11-18
   9. count the checks; stop if any failed
 
 Checks (logged as they are made, numbered, at the level of their status):
@@ -45,13 +44,12 @@
    3. every composition and band has exactly one chosen technique, the one of step 14
    4. the support and the rate of every reference, recomputed from its members, are the ones used
    5. one dynamics row per estimable forecast series
-   6. the series of a composition add up to its exam months (units due and renewed)
-   7. without credibility, the series' predictions add up to the composition's prediction
+   6. every composition is the sum of every series in its rate, in every exam month (users and lenders)
+   7. without credibility, every series predicts exactly its composition's rate
    8. every forecast series and band has exactly one chosen technique in its summary
    9. the chosen technique gives the rate step 17 used (rows with no credibility shift)
-   10. the exam of every series adds up to its chosen predictions (counts and units)
-   11. the predictions in the interval reach 80 % in the exam                      (warning only)
-   12-20. the nine tables written and read back
+   10. the chosen technique, applied to every series, gives the framework prediction of step 19
+   11-18. the eight tables written and read back
 
 Output: dict of the audit tables · the tables listed above.
 """
@@ -64,6 +62,7 @@
 from step_11_ladder import credibility_k_detail
 from step_13_dynamics import dynamics_of_one_series
 from step_14_backtest import band_of_horizon
+from prediction import band_quantiles, levels_at_origins, rate_band, shifted_rate
 from techniques import CATALOGUE, inverse_logit, logit, predict_logit
 from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM, PURPOSE_SELECTION,
                          RATE_COLUMN, RATE_FROM_POOL, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_COMPOSITION,
@@ -85,10 +84,10 @@
                 "sff_series_dynamics (check 5)",
                 "sff_series_backtest and its summary (checks 6-8)",
                 "sff_composition_forecast_all (check 9)",
-                "sff_series_exam: the exam of the chosen technique per series (checks 10-11)",
-                "write the audit tables (checks 12-20)",
+                "the chosen technique against step 19's framework (check 10)",
+                "write the audit tables (checks 11-18)",
                 "count the checks; stop if any failed"]
-STEP_OUTPUT = "nine audit tables joined to the core by fs_id, the stage ids and the credibility reference"
+STEP_OUTPUT = "eight audit tables joined to the core by fs_id, the stage ids and the credibility reference"
 
 UNITS_TOLERANCE = 1e-6
 RATE_TOLERANCE = 1e-9
@@ -102,7 +101,7 @@
 def build_audit_tables(ladder: dict, series_rate: pd.DataFrame, series_estimate: pd.DataFrame,
                        rated_units: pd.DataFrame, pool_series: pd.DataFrame, pool_reference: pd.DataFrame,
                        pool_dynamics: pd.DataFrame, backtest: dict, forecast_rows: pd.DataFrame,
-                       configuration: Config) -> dict:
+                       configuration: Config, series_exam_detail: pd.DataFrame = None) -> dict:
     """The eight audit tables, checked against their sources and written.
 
     INPUT:   the ladder (step 10), series_rate (08), series_estimate (11), rated_units (08), the
@@ -155,32 +154,28 @@
     summary = series_technique_summary(series_backtest)
     configuration.log_action(STEP_LABEL, 5, f"{len(series_backtest):,} forecast series × exam month × h × technique rows · "
                                             f"{len(summary):,} series × band × technique")
-    check_series_backtest(series_backtest, backtest["predictions"], summary, configuration, check_log)
+    check_series_backtest(series_backtest, backtest["predictions"], summary, ladder["composition_members"], rated_units,
+                          configuration, check_log)
 
     # [6] the future rate of every technique
     forecast_all = composition_forecast_all(pool_series, backtest["decision"], forecast_rows, configuration)
     configuration.log_action(STEP_LABEL, 6, f"{len(forecast_all):,} composition × horizon × technique rates")
     check_forecast_all(forecast_all, forecast_rows, series_estimate, configuration, check_log)
 
-    # [7] the exam of the chosen technique, per forecast series
-    series_exam = series_exam_table(series_backtest, series_rate, groups, pool_reference)
-    tested = series_exam[series_exam["exam_status"] == TECHNIQUE_TESTED]
-    configuration.log_action(STEP_LABEL, 7, f"exam per forecast series: {series_exam['exam_status'].value_counts().to_dict()} · "
-                                            f"{int(tested['exam_in_band'].sum()):,} of {int(tested['exam_predictions'].sum()):,} "
-                                            f"predictions in their interval · WAPE "
-                                            f"{tested['exam_abs_err_units'].sum() / max(tested['exam_real_units'].sum(), 1e-9):.1%}")
-    check_series_exam(series_exam, series_backtest, configuration, check_log)
+    # [7] the chosen technique of every composition, applied to its forecast series, is step 19's framework
+    configuration.log_action(STEP_LABEL, 7, "the chosen technique of every composition against the framework of step 19")
+    check_against_series_exam(series_backtest, series_exam_detail, configuration, check_log)
 
     # [8] the tables
     configuration.log_action(STEP_LABEL, 8, "writing the audit tables")
     tables = dict(composition=composition, composition_techniques=techniques, credibility=credibility,
                   credibility_members=members, series_dynamics=series_dynamics, series_backtest=series_backtest,
-                  series_technique_summary=summary, composition_forecast_all=forecast_all, series_exam=series_exam)
+                  series_technique_summary=summary, composition_forecast_all=forecast_all)
     for name, table_name in (("composition", TABLE_COMPOSITION), ("composition_techniques", TABLE_COMPOSITION_TECHNIQUES),
                              ("credibility", TABLE_CREDIBILITY), ("credibility_members", TABLE_CREDIBILITY_MEMBERS),
                              ("series_dynamics", TABLE_SERIES_DYNAMICS), ("series_backtest", TABLE_SERIES_BACKTEST),
                              ("series_technique_summary", TABLE_SERIES_TECHNIQUE_SUMMARY),
-                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL), ("series_exam", TABLE_SERIES_EXAM)):
+                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL)):
         configuration.write_table(STEP_LABEL, check_log, tables[name], table_name)
 
     # [9] the count of the checks; stop if anything failed
@@ -423,7 +418,9 @@
     rate (moved toward the reference by 1 − z, levels known at the origin) × the series' own units due,
     against what the series really renewed."""
     exam = predictions[predictions["proposito"] == PURPOSE_EXAM].copy()
-    exam["origin"] = exam["ultimo_mes_visto"]
+    # the origin is what was known h months before the target (T − h), not the last month the composition
+    # happened to have data: a gap just before the origin must not hide the months its reference did have
+    exam["origin"] = exam["mes_objetivo"] - exam["h"].astype(int)
     series_of = groups[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]].merge(
         series_estimate[[SERIES_ID_COLUMN, "z", "credibility_ref_id"]], on=SERIES_ID_COLUMN, how="left")
     rows = exam.merge(series_of, on=COMPOSITION_ID_COLUMN)
@@ -433,20 +430,18 @@
 
     # the levels known at the origin: the composition's and its reference's
     composition_level = levels_at_origins(pool_series[pool_series[RATE_COLUMN].notna()].rename(columns={"vencen": "_due", "renovadas": "_renewed"}),
-                                          COMPOSITION_ID_COLUMN, rows[[COMPOSITION_ID_COLUMN, "origin"]].drop_duplicates(), configuration)
+                                          COMPOSITION_ID_COLUMN, rows[[COMPOSITION_ID_COLUMN, "origin"]].drop_duplicates(),
+                                          configuration.period_col, "_due", "_renewed")
     reference_history = history.merge(reference_members, on=SERIES_ID_COLUMN).rename(
         columns={configuration.pipeline_units_col: "_due", configuration.renewed_units_col: "_renewed"})
     reference_level = levels_at_origins(reference_history, "credibility_ref_id",
-                                        rows[["credibility_ref_id", "origin"]].dropna().drop_duplicates(), configuration)
+                                        rows[["credibility_ref_id", "origin"]].dropna().drop_duplicates(),
+                                        configuration.period_col, "_due", "_renewed")
     rows = rows.merge(composition_level.rename(columns={"level": "composition_level"}), on=[COMPOSITION_ID_COLUMN, "origin"], how="left")
     rows = rows.merge(reference_level.rename(columns={"level": "reference_level"}), on=["credibility_ref_id", "origin"], how="left")
 
-    shift = np.zeros(len(rows))
-    if configuration.apply_credibility_shift:
-        moves = (rows["z"] < 1) & rows["reference_level"].between(0, 1, inclusive="neither") & rows["composition_level"].between(0, 1, inclusive="neither")
-        shift = np.where(moves, (1 - rows["z"]) * (logit(rows["reference_level"].clip(1e-6, 1 - 1e-6))
-                                                    - logit(rows["composition_level"].clip(1e-6, 1 - 1e-6))), 0.0)
-    rows["pred_rate"] = inverse_logit(logit(rows["tasa_pred"]) + shift)
+    rows["pred_rate"], shift = shifted_rate(rows["tasa_pred"], rows["z"], rows["reference_level"], rows["composition_level"],
+                                            configuration.apply_credibility_shift)
     rows["real_rate"] = rows["real_units"] / rows["due_units"]
     rows["pred_units"] = rows["pred_rate"] * rows["due_units"]
     rows["err_units"] = rows["pred_units"] - rows["real_units"]
@@ -455,11 +450,8 @@
     rows["err_norm"] = rows["err_pp"] / rows["se_binom_pp"].where(rows["se_binom_pp"] > 0)
     # the interval of the test, as the forecast builds its band: the error quantiles of the technique at that
     # horizon (step 14) × the binomial error of the rate with the series' own units due
-    quantiles = bands.set_index(["tecnica", "h"])[["q_low_norm", "q_high_norm"]]
-    rows = rows.join(quantiles, on=["tecnica", "h"])
-    se_rate = np.sqrt(rows["pred_rate"] * (1 - rows["pred_rate"]) / rows["due_units"].clip(lower=1))
-    rows["band_low_rate"] = np.clip(rows["pred_rate"] + np.minimum(rows["q_low_norm"], 0) * se_rate, 0, 1)
-    rows["band_high_rate"] = np.clip(rows["pred_rate"] + np.maximum(rows["q_high_norm"], 0) * se_rate, 0, 1)
+    q_low, q_high = band_quantiles(bands, rows["tecnica"], rows["h"], configuration)
+    rows["band_low_rate"], rows["band_high_rate"] = rate_band(rows["pred_rate"], rows["due_units"], q_low, q_high)
     rows["band_low_units"] = rows["band_low_rate"] * rows["due_units"]
     rows["band_high_units"] = rows["band_high_rate"] * rows["due_units"]
     rows["in_band"] = ((rows["real_rate"] >= rows["band_low_rate"] - 1e-12)
@@ -468,25 +460,13 @@
     rows["is_chosen"] = (pd.Series(list(zip(rows[COMPOSITION_ID_COLUMN], rows["tramo_h"])), index=rows.index).map(chosen)
                          == rows["tecnica"]).astype(int)
     rows["shifted_by_credibility"] = (np.abs(shift) > 0).astype(int)
+    rows["credibility_shift_logit"] = shift
     return rows[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "mes_objetivo", "h", "tramo_h", "origin", "tecnica", "is_chosen",
                  "shifted_by_credibility", "tasa_pred", "pred_rate", "real_rate", "due_units", "pred_units", "real_units",
                  "err_units", "err_pp", "se_binom_pp", "err_norm", "band_low_rate", "band_high_rate", "band_low_units",
                  "band_high_units", "in_band"]]
 
 
-def levels_at_origins(history: pd.DataFrame, key: str, wanted: pd.DataFrame, configuration: Config) -> pd.DataFrame:
-    """The rate (Σ renewed / Σ due) of every key with what was known at every origin it needs."""
-    rows = []
-    history_of_key = dict(tuple(history.groupby(key)))                              # one pass over the history
-    for key_value, origins in wanted.groupby(key)["origin"]:
-        own = history_of_key.get(key_value, history.iloc[0:0])
-        for origin in origins:
-            known = own[own[configuration.period_col] <= origin]
-            due = known["_due"].sum()
-            rows.append({key: key_value, "origin": origin, "level": known["_renewed"].sum() / due if due > 0 else np.nan})
-    return pd.DataFrame(rows, columns=[key, "origin", "level"])
-
-
 def series_technique_summary(series_backtest: pd.DataFrame) -> pd.DataFrame:
     """Forecast series × band × technique: errors, bias, WAPE, rank within the series, chosen, best."""
     grouped = series_backtest.assign(abs_err_pp=series_backtest["err_pp"].abs(), abs_err_units=series_backtest["err_units"].abs(),
@@ -504,23 +484,26 @@
 
 
 def check_series_backtest(series_backtest: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame,
-                          configuration: Config, check_log: list) -> None:
-    """Checks 6 to 8: the series add up to their composition; one chosen technique per series and band."""
-    exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
-    keys = [COMPOSITION_ID_COLUMN, "mes_objetivo", "h", "tecnica"]
-    summed = series_backtest.groupby(keys)[["due_units", "real_units", "pred_units"]].sum()
-    shifted = series_backtest.groupby(keys)["shifted_by_credibility"].max()
-    compared = exam.set_index(keys)[["vencen_real", "tasa_real", "tasa_pred"]].join(summed, how="inner").join(shifted)
-    adds_up = (((compared["due_units"] - compared["vencen_real"]).abs() <= UNITS_TOLERANCE)
-               & ((compared["real_units"] - compared["tasa_real"] * compared["vencen_real"]).abs() <= UNITS_TOLERANCE))
-    configuration.log_check(STEP_LABEL, check_log, "the series of a composition add up to its exam months (units due and renewed)",
-                            bool(adds_up.all()) and len(compared) == len(exam),
-                            failure_detail=f"{int((~adds_up).sum())} composition × month × h × technique differ · "
-                                           f"{len(compared)} of {len(exam)} exam predictions covered",
-                            context=f"{len(compared):,} composition × month × h × technique")
-    unshifted = compared[compared["shifted_by_credibility"] == 0]
-    same_prediction = ((unshifted["pred_units"] - unshifted["tasa_pred"] * unshifted["vencen_real"]).abs() <= UNITS_TOLERANCE)
-    configuration.log_check(STEP_LABEL, check_log, "without credibility, the series' predictions add up to the composition's prediction",
+                          members: pd.DataFrame, rated_units: pd.DataFrame, configuration: Config, check_log: list) -> None:
+    """Checks 6 to 8: every composition is the sum of every series in its rate (those that use it and those
+    that only lend their history); without credibility a series predicts its composition's rate; one chosen
+    technique per series and band."""
+    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
+    exam = predictions[predictions["proposito"] == PURPOSE_EXAM].drop_duplicates([COMPOSITION_ID_COLUMN, "mes_objetivo"])
+    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & (rated_units[due] > 0)]
+    summed = (real.merge(members[[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN]], on=SERIES_ID_COLUMN)
+              .groupby([COMPOSITION_ID_COLUMN, period])[[due, renewed]].sum())
+    compared = exam.set_index([COMPOSITION_ID_COLUMN, "mes_objetivo"])[["vencen_real", "tasa_real"]].join(
+        summed.rename_axis([COMPOSITION_ID_COLUMN, "mes_objetivo"]), how="left")
+    adds_up = (((compared[due] - compared["vencen_real"]).abs() <= UNITS_TOLERANCE)
+               & ((compared[renewed] - compared["tasa_real"] * compared["vencen_real"]).abs() <= UNITS_TOLERANCE))
+    configuration.log_check(STEP_LABEL, check_log, "every composition is the sum of every series in its rate, in every exam month "
+                            "(the ones that use it and the ones that lend their history)", bool(adds_up.all()),
+                            failure_detail=f"{int((~adds_up).sum())} composition × month differ",
+                            context=f"{len(compared):,} composition × exam month")
+    unshifted = series_backtest[series_backtest["shifted_by_credibility"] == 0]
+    same_prediction = (unshifted["pred_rate"] - unshifted["tasa_pred"]).abs() <= 1e-12
+    configuration.log_check(STEP_LABEL, check_log, "without credibility, every series predicts exactly its composition's rate",
                             bool(same_prediction.all()),
                             failure_detail=f"{int((~same_prediction).sum())} differ", context=f"{len(unshifted):,} checked")
     per_series_band = summary.groupby([SERIES_ID_COLUMN, "tramo_h"])["is_chosen"].sum()
@@ -530,45 +513,23 @@
                             context=f"{len(per_series_band):,} series × band")
 
 
-def series_exam_table(series_backtest: pd.DataFrame, series_rate: pd.DataFrame, groups: pd.DataFrame,
-                      pool_reference: pd.DataFrame) -> pd.DataFrame:
-    """Per forecast series, the exam of its CHOSEN technique (every exam month and horizon): how many
-    predictions, how many in their interval, units predicted and real, the error; and, when it has no
-    exam, why (exam_status)."""
-    chosen = series_backtest[series_backtest["is_chosen"] == 1].assign(abs_err_units=lambda frame: frame["err_units"].abs(),
-                                                                         abs_err_pp=lambda frame: frame["err_pp"].abs())
-    grouped = chosen.groupby(SERIES_ID_COLUMN)
-    exam = pd.DataFrame({"exam_predictions": grouped.size(), "exam_in_band": grouped["in_band"].sum(),
-                         "exam_pred_units": grouped["pred_units"].sum(), "exam_real_units": grouped["real_units"].sum(),
-                         "exam_abs_err_units": grouped["abs_err_units"].sum(), "exam_mae_pp": grouped["abs_err_pp"].mean(),
-                         "exam_bias_pp": grouped["err_pp"].mean()})
-    exam["exam_wape"] = exam["exam_abs_err_units"] / exam["exam_real_units"].where(exam["exam_real_units"] > 0)
-    exam["exam_coverage"] = exam["exam_in_band"] / exam["exam_predictions"]
-    table = series_rate[[SERIES_ID_COLUMN]].join(exam, on=SERIES_ID_COLUMN)
-    composition_of = groups.set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN]
-    gate_of = pool_reference.set_index(COMPOSITION_ID_COLUMN)["gate"]
-    gate = table[SERIES_ID_COLUMN].map(composition_of).map(gate_of)
-    table["exam_status"] = np.select(
-        [table["exam_predictions"].notna(), table[SERIES_ID_COLUMN].map(composition_of).isna(), gate != GATE_LEVEL],
-        [TECHNIQUE_TESTED, "not_estimable", TECHNIQUE_COMPOSITION_BELOW_FLOOR], default="no_exam_months")
-    return table
-
-
-def check_series_exam(series_exam: pd.DataFrame, series_backtest: pd.DataFrame, configuration: Config, check_log: list) -> None:
-    """Checks 10 and 11: the summary adds up to the chosen predictions; the interval keeps its promise."""
-    chosen = series_backtest[series_backtest["is_chosen"] == 1]
-    adds_up = (int(series_exam["exam_predictions"].sum()) == len(chosen)
-               and int(series_exam["exam_in_band"].sum()) == int(chosen["in_band"].sum())
-               and abs(series_exam["exam_pred_units"].sum() - chosen["pred_units"].sum()) <= UNITS_TOLERANCE
-               and abs(series_exam["exam_real_units"].sum() - chosen["real_units"].sum()) <= UNITS_TOLERANCE)
-    configuration.log_check(STEP_LABEL, check_log, "the exam of every series adds up to its chosen predictions (counts and units)",
-                            adds_up, failure_detail="the summary per series does not add up to sff_series_backtest",
-                            context=f"{len(chosen):,} chosen predictions")
-    coverage = chosen["in_band"].mean() if len(chosen) else np.nan
-    configuration.log_check(STEP_LABEL, check_log, f"the predictions in their interval reach {COVERAGE_WARNING:.0%} in the exam",
-                            bool(np.isnan(coverage) or coverage >= COVERAGE_WARNING),
-                            failure_detail=f"only {coverage:.0%} of the predictions in their interval",
-                            context=f"{coverage:.0%} in the interval" if np.isfinite(coverage) else "no prediction", blocking=False)
+def check_against_series_exam(series_backtest: pd.DataFrame, series_exam_detail, configuration: Config,
+                               check_log: list) -> None:
+    """Check 10: the backtest of step 14 (the chosen technique) applied to the series gives the framework
+    prediction of step 19, computed apart with prediction.py: one way of predicting, not two."""
+    if series_exam_detail is None:
+        configuration.log_not_evaluated(STEP_LABEL, check_log, "the chosen technique gives the framework prediction of step 19",
+                                        "step 19 has not run")
+        return
+    chosen = series_backtest[series_backtest["is_chosen"] == 1][[SERIES_ID_COLUMN, "mes_objetivo", "h", "pred_rate"]]
+    framework = series_exam_detail[series_exam_detail["method"] == "framework"].rename(columns={configuration.period_col: "mes_objetivo"})
+    compared = chosen.merge(framework[[SERIES_ID_COLUMN, "mes_objetivo", "h", "pred_rate"]], on=[SERIES_ID_COLUMN, "mes_objetivo", "h"],
+                            suffixes=("_audit", "_exam"))
+    differs = (compared["pred_rate_audit"] - compared["pred_rate_exam"]).abs() > 1e-9
+    configuration.log_check(STEP_LABEL, check_log, "the chosen technique, applied to every series, gives the framework prediction of step 19",
+                            len(compared) == len(chosen) and not bool(differs.any()),
+                            failure_detail=f"{int(differs.sum())} of {len(compared)} predictions differ · {len(chosen) - len(compared)} not found",
+                            context=f"{len(compared):,} predictions compared")
 
 
 # ═══════════════════════════════════════════════════════════════════════════════════
```

## step_informe.py

```diff
--- antes/step_informe.py	2026-10-01 08:08:00.898274921 +0000
+++ step_informe.py	2026-10-01 08:23:41.316206193 +0000
@@ -348,9 +348,11 @@
     if portfolio_summary is not None:
         for horizon, block in portfolio_summary.groupby("h"):
             framework = block[block["metodo"] == METHOD_FRAMEWORK].iloc[0]
-            spreadsheet = block[block["metodo"] != METHOD_FRAMEWORK].sort_values("error_total_medio").iloc[0]
-            headline.append((f"error del TOTAL en el examen, h = {horizon}: framework vs mejor hoja de cálculo",
-                             f"{framework['error_total_medio']:.1%} vs {spreadsheet['error_total_medio']:.1%} ({spreadsheet['metodo']})"))
+            spreadsheet = block[block["metodo"].str.startswith("hoja_")].sort_values("error_total_medio").iloc[0]
+            alone = block[block["metodo"] == "raw"]
+            headline.append((f"error del TOTAL en el examen, h = {horizon}: framework vs mejor hoja de cálculo vs serie sola",
+                             f"{framework['error_total_medio']:.1%} vs {spreadsheet['error_total_medio']:.1%} ({spreadsheet['metodo']})"
+                             + (f" vs {alone['error_total_medio'].iloc[0]:.1%}" if len(alone) else "")))
     if len(by_band):
         first = by_band.iloc[0]
         headline.append((f"error medio de la tasa por pool en el examen ({first['tramo']})",
@@ -373,8 +375,9 @@
              "**Precisión en el examen por tramo** (error medio de la tasa en pp, ponderado por dinero; dentro_banda: "
              "proporción de errores dentro de la banda del 90 %):", "", markdown_table(by_band),
              "**Precisión en el examen por nivel de riesgo** (tramo corto):", "", markdown_table(by_level),
-             "**La cartera en el examen, serie a serie: el framework frente a la hoja de cálculo** (paso 19; cada serie "
-             "predicha como la predice el forecast, con solo lo que se sabía h meses antes; la hoja: la tasa de los últimos "
+             "**La cartera en el examen, serie a serie: el framework frente a la serie sola (raw) y a la hoja de cálculo** "
+             "(paso 19; cada serie predicha como la predice el forecast, con solo lo que se sabía h meses antes; raw: la serie "
+             "con su propia historia, sin escalera; la hoja: la tasa de los últimos "
              f"{configuration.baseline_months} meses por grano × la pipeline real; error_total: de la suma de la cartera; "
              "wape_series: serie a serie, sin compensaciones):", "",
              markdown_table(results["portfolio_exam_summary"], 3) if results.get("portfolio_exam_summary") is not None else "",
@@ -382,28 +385,30 @@
     precision_by_type = exam_precision_by_series_type(results)
     if precision_by_type is not None:
         overall = precision_by_type.iloc[0]
-        headline.append(("predicciones del examen dentro de su intervalo · WAPE serie a serie",
-                         f"{overall['en_intervalo']:.0%} de {int(overall['predicciones']):,} · {overall['wape']:.1%}"))
-        lines += ["**Cuánto acertamos, forecast serie a forecast serie** (la técnica elegida de cada serie, en cada mes de "
-                  "examen y horizonte, aplicada a su propia pipeline; cada predicción con su intervalo, construido como la banda "
-                  "del forecast; en_intervalo: proporción de predicciones cuyo valor real cayó dentro). Cruzado por el tipo de "
-                  "serie: volatilidad (φ de su propia tasa), tendencia y estacionalidad, solo donde son medibles:", "",
+        headline.append(("examen serie a serie: dentro del intervalo · WAPE, framework frente a la serie sola (raw)",
+                         f"{overall['en_intervalo']:.0%} vs {overall['en_intervalo_raw']:.0%} · {overall['wape']:.1%} vs {overall['wape_raw']:.1%}"))
+        lines += ["**Cuánto acertamos, forecast serie a forecast serie, y cuánto mejora frente a la serie sola** (paso 19: en cada "
+                  "mes de examen y horizonte, el framework —su composición con credibilidad— y la serie sola con su propia historia "
+                  "(raw), sobre las mismas filas y su propia pipeline; cada predicción con su intervalo, construido como la banda "
+                  "del forecast). Cruzado por el tipo de serie: volatilidad (φ de su propia tasa), tendencia y estacionalidad:", "",
                   markdown_table(precision_by_type.assign(
                       predicciones=precision_by_type["predicciones"].astype(int),
                       en_intervalo=precision_by_type["en_intervalo"].map("{:.0%}".format),
+                      en_intervalo_raw=precision_by_type["en_intervalo_raw"].map("{:.0%}".format),
                       wape=precision_by_type["wape"].map("{:.1%}".format),
+                      wape_raw=precision_by_type["wape_raw"].map("{:.1%}".format),
                       sesgo=precision_by_type["sesgo"].map("{:+.1%}".format)), 0)]
     return "\n".join(lines) + "\n"
 
 
 def exam_precision_by_series_type(results: dict):
-    """The exam of every forecast series (its chosen technique) summed by type of series: all, by
-    volatility, by trend and by seasonality (counts and units added, then the ratios)."""
-    audit = results.get("audit")
-    if not audit:
+    """The exam of every forecast series summed by type of series (all, by volatility, trend and seasonality):
+    the framework against the series alone (raw), counts and units added, then the ratios."""
+    audit, series_exam = results.get("audit"), results.get("series_exam")
+    if not audit or not series_exam:
         return None
-    exam = audit["series_exam"].merge(audit["series_dynamics"], on=SERIES_ID_COLUMN, how="left")
-    exam = exam[exam["exam_status"] == "tested"]
+    exam = series_exam["per_series"].merge(audit["series_dynamics"], on=SERIES_ID_COLUMN, how="left")
+    exam = exam[exam["framework_predictions"] > 0]
     if exam.empty:
         return None
     measured = exam["measurable"] == "yes"
@@ -414,16 +419,18 @@
                 ("sin tendencia (medible)", measured & exam["trend"].fillna(0).eq(0)),
                 ("estacional (medible)", measured & exam["seasonal"].eq(1)),
                 ("no estacional (medible)", measured & exam["seasonal"].eq(0)),
-                ("dinámica no medible (poco soporte o historia)", ~measured)]
+                ("sin dinámica medible / sin composición", ~measured)]
     rows = []
     for name, mask in segments:
         block = exam[mask]
         if block.empty:
             continue
-        rows.append({"segmento": name, "series": len(block), "predicciones": block["exam_predictions"].sum(),
-                     "en_intervalo": block["exam_in_band"].sum() / block["exam_predictions"].sum(),
-                     "wape": block["exam_abs_err_units"].sum() / block["exam_real_units"].sum(),
-                     "sesgo": block["exam_pred_units"].sum() / block["exam_real_units"].sum() - 1})
+        rows.append({"segmento": name, "series": len(block), "predicciones": block["framework_predictions"].sum(),
+                     "en_intervalo": block["framework_in_band"].sum() / block["framework_predictions"].sum(),
+                     "en_intervalo_raw": block["raw_in_band"].sum() / block["raw_predictions"].sum(),
+                     "wape": block["framework_abs_err_units"].sum() / block["framework_real_units"].sum(),
+                     "wape_raw": block["raw_abs_err_units"].sum() / block["raw_real_units"].sum(),
+                     "sesgo": block["framework_pred_units"].sum() / block["framework_real_units"].sum() - 1})
     return pd.DataFrame(rows)
 
 
```

## step_nucleo.py

```diff
--- antes/step_nucleo.py	2026-10-01 08:08:00.762094991 +0000
+++ step_nucleo.py	2026-10-01 08:21:41.103583988 +0000
@@ -31,7 +31,7 @@
   1. the rows of the extract (the fine table) with their raw and step-02 measures and their ids
   2. the gap rows of step 08
   3. the values of the forecast unit (steps 04, 07), if they have run
-  4. the values of the series (steps 06, 08, 11) and of its estimation id (step 14), if they have run
+  4. the forecast rows (step 17), the final block, and the forecast series dimension (one row per series)
   5. the legend: every column, its step, its level and how to aggregate it
   6. check the core against the extract and the legend              checks 1-4
   7. write the core and its legend                                  checks 5-6
@@ -53,14 +53,15 @@
 import pandas as pd
 
 from config import Config, join_columns
-from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, COMPOSITION_ID_COLUMN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN,
                          FINE_ROWS_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_SIMULATED, ROLE_TEST,
                          ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
                          S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN,
                          S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_CORE,
-                         TABLE_CORE_LEGEND, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
-                         TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL, TS_PROJECTED, TS_REAL, TS_REENTRY,
-                         UNIT_ID_COLUMN, UNIVERSE_COLUMN, UPLIFT_CELL_ID_COLUMN, TRUTH_ROLES as TRUTH_ROLES_OF_CORE)
+                         TABLE_CORE_LEGEND, TABLE_FORECAST_SERIES, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED,
+                         TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL, TS_PROJECTED, TS_REAL,
+                         TS_REENTRY, UNIT_ID_COLUMN, UNIVERSE_COLUMN, UPLIFT_CELL_ID_COLUMN,
+                         TRUTH_ROLES as TRUTH_ROLES_OF_CORE)
 
 
 # ─── the step ────────────────────────────────────────────────────────────────────
@@ -72,7 +73,7 @@
 STEP_ACTIONS = ["the rows of the extract (the fine table) with their raw and step-02 measures and their ids",
                 "the gap rows of step 08",
                 "the values of the forecast unit (steps 04, 07), if they have run",
-                "the values of the series (steps 06, 08, 11) and of its estimation id (step 14), if they have run",
+                "the forecast rows (step 17), the final block, and the forecast series dimension (one row per series)",
                 "the legend: every column, its step, its level and how to aggregate it",
                 "check the core against the extract and the legend (checks 1-4)",
                 "write the core and its legend (checks 5-6)",
@@ -162,17 +163,23 @@
                           ("measurable", "s13_series_measurable", "13", "yes · low_support (< 30 a month: not conclusive) · short_history"),
                           ("differs_from_composition", "s13_series_differs", "13", "1 when its trend or season differs from its composition's")]
 # per forecast series: the exam of its chosen technique (every exam month and horizon), as ratios repeated on its rows
-SERIES_VALUES_EXAM = [("exam_status", "s19_exam_status", "19", "tested · composition_below_floor · not_estimable · no_exam_months"),
-                      ("exam_mae_pp", "s19_exam_mae_pp", "19", "mean absolute error of its rate in the exam (pp)"),
-                      ("exam_bias_pp", "s19_exam_bias_pp", "19", "mean error of its rate in the exam (pp): + over-predicts"),
-                      ("exam_wape", "s19_exam_wape", "19", "Σ |predicted − real| / Σ real renewed units in the exam"),
-                      ("exam_coverage", "s19_exam_coverage", "19", "share of its exam predictions whose real rate fell inside the interval")]
-# … and as counts and units on its FIRST row only: a SUM over any filter counts every series once
-SERIES_SUMMABLE_EXAM = [("exam_predictions", "s19_exam_predictions", "19", "exam predictions (first row of the series only: SUM)"),
-                        ("exam_in_band", "s19_exam_in_band", "19", "exam predictions inside their interval (first row only: SUM)"),
-                        ("exam_pred_units", "s19_exam_pred_units", "19", "renewed units predicted in the exam (first row only: SUM)"),
-                        ("exam_real_units", "s19_exam_real_units", "19", "renewed units real in the exam (first row only: SUM)"),
-                        ("exam_abs_err_units", "s19_exam_abs_err_units", "19", "Σ |predicted − real| units in the exam (first row only: SUM)")]
+SERIES_VALUES_EXAM = [("framework_mae_pp", "s19_exam_mae_pp", "19", "mean absolute error of its rate in the exam, framework (pp)"),
+                      ("framework_bias_pp", "s19_exam_bias_pp", "19", "mean error of its rate in the exam, framework (pp): + over-predicts"),
+                      ("framework_wape", "s19_exam_wape", "19", "Σ |predicted − real| / Σ real renewed units in the exam, framework"),
+                      ("framework_coverage", "s19_exam_coverage", "19", "share of its framework predictions whose real rate fell inside the interval"),
+                      ("raw_mae_pp", "s19_raw_mae_pp", "19", "the same error predicting the series alone with its own history (raw)"),
+                      ("raw_coverage", "s19_raw_coverage", "19", "share of its raw predictions inside their interval"),
+                      ("improvement_mae_pp", "s19_improvement_mae_pp", "19", "raw error − framework error (pp): + the framework predicts it better")]
+# … and as counts and units: in the dimension a SUM over any filter counts every forecast series once
+SERIES_SUMMABLE_EXAM = [("framework_predictions", "s19_exam_predictions", "19", "exam predictions (SUM)"),
+                        ("framework_in_band", "s19_exam_in_band", "19", "framework predictions inside their interval (SUM)"),
+                        ("framework_pred_units", "s19_exam_pred_units", "19", "renewed units predicted by the framework (SUM)"),
+                        ("framework_real_units", "s19_exam_real_units", "19", "renewed units real in the exam (SUM)"),
+                        ("framework_abs_err_units", "s19_exam_abs_err_units", "19", "Σ |predicted − real| units, framework (SUM)"),
+                        ("raw_in_band", "s19_raw_in_band", "19", "raw predictions inside their interval (SUM)"),
+                        ("raw_abs_err_units", "s19_raw_abs_err_units", "19", "Σ |predicted − real| units, raw (SUM)"),
+                        ("raw_real_units", "s19_raw_real_units", "19", "renewed units real in the exam, raw rows (SUM)"),
+                        ("raw_predictions", "s19_raw_predictions", "19", "raw exam predictions (SUM)")]
 # step 13, per estimation id (the dynamics of the rate the series takes)
 ESTIMATION_VALUES_STEP_13 = [("phi", "s13_phi", "13", "φ of the estimation id: observed variation of its rate / binomial noise (≈ 1: nothing to model)"),
                              ("tendencia", "s13_tendencia", "13", "+1 / −1 significant trend of the rate, 0 none"),
@@ -239,32 +246,9 @@
             blocks_present.append(block_specs)
     configuration.log_action(STEP_LABEL, 3, f"unit blocks added: {[specs[0][2] for specs in blocks_present] or 'none'}")
 
-    # [4] the values of the series and of its estimation id
-    ladder_stages = stages_of_every_series(ladder_stages, series_rate)
-    if series_dynamics is not None:
-        core = add_block(core, series_dynamics, SERIES_ID_COLUMN, "s03_fs_id", SERIES_VALUES_DYNAMICS)
-        blocks_present.append(SERIES_VALUES_DYNAMICS)
-    if series_exam is not None:
-        core = add_block(core, series_exam, SERIES_ID_COLUMN, "s03_fs_id", SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)
-        # the counts and units only on the first row of the extract of every series: SUM counts each series once
-        first_rows = core[core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW].groupby("s03_fs_id").head(1).index
-        summable_names = [name for _, name, _, _ in SERIES_SUMMABLE_EXAM]
-        core.loc[~core.index.isin(first_rows), summable_names] = np.nan
-        blocks_present.append(SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)
-    series_blocks = [(series_table, SERIES_VALUES_STEP_06), (series_rate, SERIES_VALUES_STEP_08),
-                     (ladder_stages, SERIES_VALUES_STEP_10), (series_estimate, SERIES_VALUES_STEP_11)]
-    for block_frame, block_specs in series_blocks:
-        if block_frame is not None:
-            core = add_block(core, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
-            blocks_present.append(block_specs)
-    if pool_dynamics is not None and len(pool_dynamics) and "s10_stage3_id" in core.columns:
-        dynamics_values = pool_dynamics[[COMPOSITION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
-        dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_13})
-        core = core.merge(dynamics_values, left_on="s10_stage3_id", right_on=COMPOSITION_ID_COLUMN, how="left").drop(columns=COMPOSITION_ID_COLUMN)
-        blocks_present.append(ESTIMATION_VALUES_STEP_13)
-    if technique_decision is not None and "s10_stage3_id" in core.columns:
-        core, backtest_specs = add_backtest_block(core, technique_decision, exam_by_pool, configuration)
-        blocks_present.append(backtest_specs)
+    # [4] the forecast series dimension: one row per forecast series with everything of its level
+    #     (route, own rate, the 4 stages, its composition and credibility, technique, dynamics, exam);
+    #     the core keeps only what belongs to a row (or a unit) and joins it by s03_fs_id
     if forecast is not None:
         forecast_values = forecast.assign(_clave=forecast_keys(forecast), _vencen_unidades=forecast[configuration.pipeline_units_col],
                                           _vencen_usd=forecast[configuration.pipeline_usd_col])
@@ -278,20 +262,26 @@
     if "s04_filas_finas" in core.columns:
         core.loc[core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP, "s04_filas_finas"] = 0     # a gap adds no fine row
     core = core.sort_values(["s03_fs_id", configuration.period_col, ROW_ORIGIN_COLUMN]).reset_index(drop=True)
-    configuration.log_action(STEP_LABEL, 4, f"blocks present: {sorted({specs[0][2] for specs in blocks_present})} · "
-                                            f"the core has {len(core):,} rows × {len(core.columns)} columns")
+    dimension = build_series_dimension(core, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
+                                       technique_decision, exam_by_pool, series_dynamics, series_exam, configuration)
+    configuration.log_action(STEP_LABEL, 4, f"the core has {len(core):,} rows × {len(core.columns)} columns · the forecast series "
+                                            f"dimension {len(dimension):,} rows × {len(dimension.columns)} columns, joined by s03_fs_id")
 
     # [5] the legend
     legend = core_legend(core, dimension_columns, blocks_present, configuration)
+    legend = pd.concat([legend.assign(tabla="sff_" + TABLE_CORE), dimension_legend(dimension)], ignore_index=True)
     configuration.log_action(STEP_LABEL, 5, f"legend of {len(legend)} columns: {legend['nivel'].value_counts().to_dict()}")
 
     # [6] the checks
     configuration.log_action(STEP_LABEL, 6, "checking the core against the extract and the legend")
-    check_core(core, fine_table, gap_rows, legend, blocks_present, forecast_total, configuration, check_log)
+    check_core(core, fine_table, gap_rows, legend[legend["tabla"] == "sff_" + TABLE_CORE], blocks_present, forecast_total,
+               configuration, check_log)
+    check_dimension(core, dimension, series_exam, configuration, check_log)
 
     # [7] the tables, written
     configuration.log_action(STEP_LABEL, 7, "writing the core and its legend")
     configuration.write_table(STEP_LABEL, check_log, core, TABLE_CORE)
+    configuration.write_table(STEP_LABEL, check_log, dimension, TABLE_FORECAST_SERIES)
     configuration.write_table(STEP_LABEL, check_log, legend, TABLE_CORE_LEGEND)
 
     # [8] the count of the checks; stop if anything failed
@@ -299,9 +289,80 @@
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
 
     # [9] one series as it looks in the core, how to read it in Power BI, and the questions answered from it
-    log_core_report(core, configuration)
+    log_core_report(core, dimension, configuration)
     log_core_answers(core, configuration)
-    return core, legend
+    return core, legend, dimension
+
+
+DIMENSION_SUMMARY_VALUES = [("s17_esperado_usd_total", "17", "USD expected to renew in the months to predict (sum of its rows)"),
+                            ("s17_pct_usd_high", "17", "share of its expected USD with confidence high"),
+                            ("s17_pct_usd_medium", "17", "share of its expected USD with confidence medium"),
+                            ("s17_pct_usd_low", "17", "share of its expected USD with confidence low"),
+                            ("s17_confidence", "17", "the confidence of most of its expected USD")]
+
+
+def build_series_dimension(core: pd.DataFrame, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
+                           technique_decision, exam_by_pool, series_dynamics, series_exam, configuration: Config) -> pd.DataFrame:
+    """sff_forecast_series: one row per forecast series of the core, with every value of its level."""
+    dimension = pd.DataFrame({"s03_fs_id": sorted(core["s03_fs_id"].dropna().unique())})
+    ladder_stages = stages_of_every_series(ladder_stages, series_rate)
+    for block_frame, block_specs in ((series_table, SERIES_VALUES_STEP_06), (series_rate, SERIES_VALUES_STEP_08),
+                                     (ladder_stages, SERIES_VALUES_STEP_10), (series_estimate, SERIES_VALUES_STEP_11),
+                                     (series_dynamics, SERIES_VALUES_DYNAMICS),
+                                     (series_exam, SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)):
+        if block_frame is not None:
+            dimension = add_block(dimension, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
+    if pool_dynamics is not None and len(pool_dynamics) and "s10_stage3_id" in dimension.columns:
+        dynamics_values = pool_dynamics[[COMPOSITION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
+        dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_13})
+        dimension = dimension.merge(dynamics_values, left_on="s10_stage3_id", right_on=COMPOSITION_ID_COLUMN,
+                                    how="left").drop(columns=COMPOSITION_ID_COLUMN)
+    if technique_decision is not None and "s10_stage3_id" in dimension.columns:
+        dimension, _ = add_backtest_block(dimension, technique_decision, exam_by_pool, configuration)
+    # the money to predict and its confidence, from the rows of the core
+    if "s17_confidence" in core.columns:
+        future = core[core["s17_esperado_usd"].notna()]
+        by_label = future.pivot_table(index="s03_fs_id", columns="s17_confidence", values="s17_esperado_usd", aggfunc="sum").fillna(0.0)
+        total = by_label.sum(axis=1)
+        summary = pd.DataFrame({"s17_esperado_usd_total": total})
+        for label in ("high", "medium", "low"):
+            summary[f"s17_pct_usd_{label}"] = (by_label[label] / total.where(total > 0)) if label in by_label.columns else 0.0
+        summary["s17_confidence"] = by_label.idxmax(axis=1)
+        dimension = dimension.merge(summary, left_on="s03_fs_id", right_index=True, how="left")
+    return dimension
+
+
+def dimension_legend(dimension: pd.DataFrame) -> pd.DataFrame:
+    """The legend of sff_forecast_series: every column with its step and how to aggregate it."""
+    described = {name: (step, description) for _, name, step, description in
+                 SERIES_VALUES_STEP_06 + SERIES_VALUES_STEP_08 + SERIES_VALUES_STEP_10 + SERIES_VALUES_STEP_11
+                 + SERIES_VALUES_DYNAMICS + SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM + ESTIMATION_VALUES_STEP_13}
+    described.update({name: (step, description) for name, step, description in DIMENSION_SUMMARY_VALUES})
+    summable = {name for _, name, _, _ in SERIES_SUMMABLE_EXAM} | {"s06_usd_por_predecir", "s17_esperado_usd_total"}
+    rows = [("s03_fs_id", "03", LEVEL_SERIES, "clave: relación 1 → n con sff_nucleo[s03_fs_id]", "the forecast series")]
+    for column_name in dimension.columns[1:]:
+        base = next((name for _, name, _, _ in ESTIMATION_VALUES_STEP_14 if column_name.startswith(name + "_")), None)
+        step, description = described.get(column_name, ("14", next((d for _, n, _, d in ESTIMATION_VALUES_STEP_14 if n == base), ""))
+                                          if base else ("", ""))
+        rows.append((column_name, step, LEVEL_SERIES, AGGREGATE_SUM if column_name in summable else AGGREGATE_SLICER, description))
+    legend = pd.DataFrame(rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
+    return legend.assign(tabla="sff_" + TABLE_FORECAST_SERIES)
+
+
+def check_dimension(core: pd.DataFrame, dimension: pd.DataFrame, series_exam, configuration: Config, check_log: list) -> None:
+    """Check 6 and 7: one row per forecast series of the core; its exam sums equal step 19's."""
+    configuration.log_check(STEP_LABEL, check_log, "the forecast series dimension has one row per forecast series of the core",
+                            dimension["s03_fs_id"].is_unique and set(dimension["s03_fs_id"]) == set(core["s03_fs_id"].dropna()),
+                            failure_detail="the dimension and the core do not have the same forecast series",
+                            context=f"{len(dimension):,} forecast series")
+    if series_exam is None or "s19_exam_real_units" not in dimension.columns:
+        configuration.log_not_evaluated(STEP_LABEL, check_log, "the exam sums of the dimension equal step 19's", "no exam")
+        return
+    same = (abs(dimension["s19_exam_real_units"].sum() - series_exam["framework_real_units"].sum()) < 1e-6
+            and int(dimension["s19_exam_predictions"].sum()) == int(series_exam["framework_predictions"].sum()))
+    configuration.log_check(STEP_LABEL, check_log, "the exam sums of the dimension equal step 19's (each series once)", same,
+                            failure_detail="the dimension lost or duplicated exam predictions",
+                            context=f"{int(dimension['s19_exam_predictions'].sum()):,} exam predictions")
 
 
 def add_block(core: pd.DataFrame, block_frame: pd.DataFrame, key_column: str, core_key: str, block_specs: list) -> pd.DataFrame:
@@ -651,25 +712,29 @@
     configuration.show_table(answers)
 
 
-def log_core_report(core: pd.DataFrame, configuration: Config) -> None:
-    """Action 9: one series as it looks in the core, and how to read it in Power BI."""
+def log_core_report(core: pd.DataFrame, dimension: pd.DataFrame, configuration: Config) -> None:
+    """Action 9: one series as it looks in the two tables, and how to read them in Power BI."""
     largest_series = core.groupby("s03_fs_id")["s00_vencen_usd"].sum().idxmax()
-    configuration.log_action(STEP_LABEL, 9, f"the series with the most money, '{largest_series}', as it looks in the "
-                                            f"core (last {MONTHS_SHOWN} closed months; the series values repeat on every row):")
+    configuration.log_action(STEP_LABEL, 9, f"the series with the most money, '{largest_series}': its row of sff_forecast_series "
+                                            f"and its last {MONTHS_SHOWN} closed months in sff_nucleo:")
+    shown = ["s03_fs_id", "s06_ruta", "s08_n_propio", "s08_tasa_propia", "s10_stage3_id", "s10_stage3_support", "s11_tasa_estimada",
+             "s11_nivel_riesgo", "s19_exam_mae_pp", "s19_raw_mae_pp", "s19_exam_coverage", "s17_confidence"]
+    configuration.show_table(dimension[dimension["s03_fs_id"] == largest_series][[column for column in shown if column in dimension.columns]])
     series_rows = core[(core["s03_fs_id"] == largest_series) & core["s02_rol"].isin([ROLE_TRAIN, ROLE_TEST])]
-    shown_columns = [configuration.period_col, ROW_ORIGIN_COLUMN, "s02_rol", "s00_vencen_unidades", "s02_renovadas_unidades",
-                     "s06_ruta", "s08_tasa_propia", "s11_tasa_estimada", "s11_nivel_riesgo", "s14_tecnica_corto",
-                     "s14_examen_err_pp_corto"]
+    shown_columns = [configuration.period_col, ROW_ORIGIN_COLUMN, "s02_rol", "s00_vencen_unidades", "s02_renovadas_unidades"]
     configuration.show_table(series_rows.tail(MONTHS_SHOWN)[[column for column in shown_columns if column in core.columns]])
-    configuration.logger.doc(f"[{STEP_LABEL}] in Power BI: a slicer on s03_fs_id selects one series; the rows are its "
-                             f"months. A rate is a measure (a ratio of sums over the rows selected), never a column; "
-                             f"a value of the series (s06_, s08_, s11_) is read with MAX. Measures to create:")
+    configuration.logger.doc(f"[{STEP_LABEL}] in Power BI: relate sff_forecast_series[s03_fs_id] 1 → n sff_nucleo[s03_fs_id]; a "
+                             f"slicer on the dimension selects forecast series (by id, confidence, risk level, stage, exam…) and "
+                             f"filters their rows. A rate is a measure (a ratio of sums), never a column. Measures to create:")
     configuration.show_table(pd.DataFrame([
         ("Tasa renovación (meses cerrados)",
          "DIVIDE(CALCULATE(SUM(sff_nucleo[s02_renovadas_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}), "
          "CALCULATE(SUM(sff_nucleo[s00_vencen_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}))"),
         ("Vence USD", "SUM(sff_nucleo[s00_vencen_usd])"),
-        ("Tasa estimada de la serie", "MAX(sff_nucleo[s11_tasa_estimada])"),
         ("Renovado (real + previsto)", "SUM(sff_nucleo[fin_renovado_usd])  — segmentar por fin_ano, fin_estado, fin_origen"),
         ("Pipeline", "SUM(sff_nucleo[fin_vence_usd])  — segmentar por fin_ano, fin_origen"),
-        ("Nivel de riesgo de la serie", "MAX(sff_nucleo[s11_nivel_riesgo])")], columns=["medida", "DAX"]))
+        ("Examen: dentro del intervalo", "DIVIDE(SUM(sff_forecast_series[s19_exam_in_band]), SUM(sff_forecast_series[s19_exam_predictions]))"),
+        ("Examen: WAPE framework / raw", "DIVIDE(SUM(sff_forecast_series[s19_exam_abs_err_units]), SUM(sff_forecast_series[s19_exam_real_units]))"
+                                        "  ·  the same with s19_raw_abs_err_units")], columns=["medida", "DAX"]))
+
+
```

## test_step_10_11.py

```diff
--- antes/test_step_10_11.py	2026-10-01 08:08:00.672203577 +0000
+++ test_step_10_11.py	2026-10-01 08:25:41.990739307 +0000
@@ -1,8 +1,9 @@
 """
-test_step_10_11.py — Steps 10 and 11 on the synthetic: every pass of the ladder is a partition
-(the totals add up, the groups get fewer and bigger), a closed group keeps its id, the sign is
-never mixed, every final group lends one rate, and the rate blends with the reference by
-credibility.
+test_step_10_11.py — Steps 10 and 11 on the synthetic: the mechanical rule of the ladder (every
+series grouped by the same pattern at every pass; each uses the first pass whose group reaches the
+floor; a big series keeps its own id but lends its history), every pass is a partition of the ids,
+the sign is never mixed, every merge is recorded with its error alone and merged, stage 3 merges at
+most collapse_passes dims, and the rate blends with the reference by credibility.
 
     python test_step_10_11.py
 """
@@ -40,7 +41,7 @@
                                                                                  decision, configuration)))
     ladder = frames["ladder"]
     steps, summary, groups = ladder["steps"], ladder["summary"], ladder["groups"]
-    check("7 checks: 7 ok" in console, "the 7 checks of step 10 pass")
+    check("9 checks: 9 ok" in console, "the 9 checks of step 10 pass")
     check(list(summary["step_name"][:2]) == ["itself", "sign"] and summary["step_name"].iloc[2].startswith("extra"),
           "the passes in order: itself → sign → extras → mandatory dims")
     check(summary["units_due"].nunique() == 1, "every pass is a partition: the units due add up to the same total")
@@ -60,6 +61,18 @@
     mixed = steps[steps["fs_id"] == "EU|B|1|0|0|1|web"]
     check(mixed["group_id"].nunique() == 1, "a mixed series is never merged")
 
+    tele = groups.set_index("fs_id").loc["NA|A|0|0|0|0|tele"]
+    check(tele["composition_id"] == "NA|A|SIG=neutro|*" and tele["group_series"] == 2 and tele["composition_users"] == 1,
+          "the mechanical rule: NA|A|tele joins its sibling NA|A|web when the channel is removed (2 series in the rate, 1 uses it)")
+    members = ladder["composition_members"]
+    web_role = members[(members["composition_id"] == "NA|A|SIG=neutro|*") & (members["fs_id"] == "NA|A|0|0|0|0|web")]["role"]
+    check(list(web_role) == ["lends"] and groups.set_index("fs_id").loc["NA|A|0|0|0|0|web", "composition_id"] == "NA|A|0|0|0|0|web",
+          "NA|A|web predicts alone (own id) and lends its history to NA|A|tele's composition")
+    merges = ladder["merges"]
+    check(len(merges) > 0 and (merges["error_merged_pp"] < merges["error_alone_pp"]).all()
+          and {"bias_pp", "error_alone_pp", "error_merged_pp", "improves"} <= set(merges.columns),
+          "every merge is recorded with its bias and its error alone and merged; in the synthetic every merge improves")
+
     with_reference = groups.dropna(subset=["credibility_ref_id"])
     check((with_reference["ref_series"] > with_reference["group_series"]).all()
           and (with_reference["group_support"] < configuration.own_rate_floor).all(),
@@ -68,6 +81,21 @@
     check(closed_big["credibility_ref_id"].isna().all(), "a group with own precision takes no reference")
 
 
+def test_the_collapse_limit() -> None:
+    print("A2 · stage 3 merges at most collapse_passes dims; the later passes only give a reference")
+    rated_units, series_rate, series_lookup, decision, configuration = inputs_of_the_ladder()
+    configuration.collapse_passes = 0
+    frames = {}
+    console_of(lambda: frames.setdefault("ladder", build_ladder_groups(rated_units, series_rate, series_lookup, decision, configuration)))
+    steps = frames["ladder"]["steps"]
+    check(not steps["step_name"].str.startswith("without").any(),
+          "with collapse_passes = 0 no mandatory dim is merged: the passes stop at the extras")
+    groups = frames["ladder"]["groups"]
+    references = groups.dropna(subset=["credibility_ref_id"])
+    check(len(references) > 0 and (references["credibility_ref_step"] > references["final_step"]).all(),
+          "a composition below the own-rate floor can still take its reference from a later (non-merging) pass")
+
+
 def test_the_rate() -> None:
     print("B · step 11: the group lends its rate, blended with its reference by credibility")
     rated_units, series_rate, series_lookup, decision, configuration = inputs_of_the_ladder()
@@ -97,5 +125,6 @@
 
 if __name__ == "__main__":
     test_the_passes()
+    test_the_collapse_limit()
     test_the_rate()
     finish()
```

## test_step_12_14.py

```diff
--- antes/test_step_12_14.py	2026-10-01 08:08:00.827042674 +0000
+++ test_step_12_14.py	2026-10-01 08:24:53.068121573 +0000
@@ -78,9 +78,9 @@
     total = backtest["exam_total"]
     check(len(total) == 3 * len(configuration.backtest_horizons) and total["elegida_error_pct"].abs().max() < 0.2,
           "the exam of the total: one row per exam month and horizon")
-    core = results["core"]
-    check({"s14_tecnica_corto", "s14_tecnica_medio_largo", "s14_examen_err_pp_corto"} <= set(core.columns),
-          "the core carries the chosen technique and the exam error of every series' estimation id")
+    dimension = results["forecast_series"]
+    check({"s14_tecnica_corto", "s14_tecnica_medio_largo", "s14_examen_err_pp_corto"} <= set(dimension.columns),
+          "the forecast series dimension carries the chosen technique and the exam error of its composition")
 
 
 if __name__ == "__main__":
```

## test_step_13_informe.py

```diff
--- antes/test_step_13_informe.py	2026-10-01 08:08:00.900755339 +0000
+++ test_step_13_informe.py	2026-10-01 08:24:53.068377846 +0000
@@ -53,7 +53,7 @@
     card = results["card"]
     check(len(card) == len(results["series"]) and {"phi", "tecnica_corto", "elegida_err_pp_medio_corto", "nivel_riesgo"} <= set(card.columns),
           "the card: one row per series with dynamics, technique and exam error")
-    check("s13_phi" in results["core"].columns, "the core carries the dynamics of the estimation id")
+    check("s13_phi" in results["forecast_series"].columns, "the forecast series dimension carries the dynamics of its composition")
     check("3 checks: 3 ok" in console.split("STEP IN")[1], "the checks of the report pass")
 
 
```

## test_step_15_18.py

```diff
--- antes/test_step_15_18.py	2026-10-01 08:08:00.950274924 +0000
+++ test_step_15_18.py	2026-10-01 08:09:01.510215465 +0000
@@ -10,7 +10,7 @@
 import pandas as pd
 
 from main import run
-from step_17_forecast import judged_horizon
+from prediction import judged_horizon
 from test_helpers import check, console_of, finish, synthetic_with
 
 
```

## test_step_19.py

```diff
--- antes/test_step_19.py	2026-10-01 08:08:00.951911714 +0000
+++ test_step_19.py	2026-10-01 08:26:06.742153378 +0000
@@ -1,39 +1,54 @@
 """
-test_step_19.py — The exam of the portfolio: series by series, without peeking, the
-framework against the spreadsheet on the same rows.
+test_step_19.py — The exam of every forecast series: three methods on the same rows (the series
+alone, the framework, the spreadsheet), without peeking, every prediction traced and with its interval.
 
     python test_step_19.py
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
+import numpy as np
 import pandas as pd
 
 from main import run
 from test_helpers import check, console_of, finish, synthetic_with
 
 
-def test_the_portfolio_exam() -> None:
-    print("A · the exam of the portfolio on the synthetic")
+def test_the_series_exam() -> None:
+    print("A · the exam of every forecast series on the synthetic")
     configuration = synthetic_with()
     results = {}
     console = console_of(lambda: results.update(run(configuration)))
-    check("5 checks: 5 ok" in console.split("STEP 19")[1], "the checks of step 19 pass")
-    by_month, summary = results["portfolio_exam"], results["portfolio_exam_summary"]
+    check("8 checks: 8 ok" in console.split("STEP 19")[1], "the 8 checks of step 19 pass")
+    exam = results["series_exam"]
+    detail, per_series, summary = exam["detail"], exam["per_series"], exam["summary"]
     rated = results["rated_units"]
     exam_real = rated[(rated["sintetica"] == 0) & (rated["rol"] == "examen")]
-    check(abs(by_month[by_month["h"] == 1]["renovadas_reales"].sum() - exam_real["total_renewed_units"].sum()) < 1e-6,
-          "the real renewals of the exam are the renewals of every series in the exam months (no pool overlap)")
-    check(set(summary["metodo"]) == {"framework", "hoja_global", "hoja_mandatory"} and set(summary["h"]) == {1, 6}
-          and configuration.baseline_months == 12,
-          "the framework and every spreadsheet grain (12 months, what the business does), at every horizon")
-    framework = summary[summary["metodo"] == "framework"].set_index("h")
-    spreadsheet = summary[summary["metodo"] == "hoja_mandatory"].set_index("h")
-    check((framework["wape_series"] < spreadsheet["wape_series"]).all(),
+    check(abs(exam["by_month"][exam["by_month"]["h"] == 1]["renovadas_reales"].sum() - exam_real["total_renewed_units"].sum()) < 1e-6,
+          "the real renewals of the exam are the renewals of every series in the exam months (nothing counted twice)")
+    check(set(summary["metodo"]) == {"raw", "framework", "hoja_global", "hoja_mandatory"} and set(summary["h"]) == {1, 6},
+          "three ways on the same rows: the series alone (raw), the framework and every spreadsheet grain, at every horizon")
+    check((detail.groupby("method").size().nunique() == 1), "every method predicts exactly the same rows")
+    check((detail["origin"] == detail["period"] - detail["h"]).all(), "every prediction knows only up to its origin, T − h")
+    framework = detail[detail["method"] == "framework"]
+    traced = framework.dropna(subset=["composition_rate"])
+    no_shift = traced[traced["credibility_shift_logit"] == 0]
+    check(len(traced) > 0 and np.allclose(no_shift["pred_rate"], no_shift["composition_rate"]),
+          "every framework prediction is traced: without credibility, the series' rate is its composition's prediction")
+    with_interval = detail[detail["method"].isin(["raw", "framework"])]
+    check(with_interval["band_low"].notna().all() and (with_interval["band_low"] <= with_interval["pred_rate"] + 1e-12).all()
+          and (with_interval["pred_rate"] <= with_interval["band_high"] + 1e-12).all()
+          and detail.loc[detail["method"].str.startswith("hoja_"), "band_low"].isna().all(),
+          "raw and framework predictions carry their interval; the spreadsheet has none")
+    check({"raw_mae_pp", "framework_mae_pp", "improvement_mae_pp", "raw_coverage", "framework_coverage"} <= set(per_series.columns)
+          and per_series["fs_id"].is_unique,
+          "per forecast series, raw and framework side by side: the improvement is one subtraction")
+    by_method = summary.groupby("metodo")["wape_series"].mean()
+    check(by_method["framework"] < by_method["hoja_mandatory"],
           "series by series, the framework beats the spreadsheet by mandatory cell in the synthetic")
     report = open(f"{configuration.output_folder}/informe_sff.md", encoding="utf-8").read()
-    check("framework vs mejor hoja de cálculo" in report, "the report puts the comparison in its headline")
+    check("framework vs mejor hoja de cálculo vs serie sola" in report, "the report puts the three ways in its headline")
 
 
 if __name__ == "__main__":
-    test_the_portfolio_exam()
+    test_the_series_exam()
     finish()
```

## test_step_audit.py

```diff
--- antes/test_step_audit.py	2026-10-01 08:08:00.952398110 +0000
+++ test_step_audit.py	2026-10-01 08:25:41.991403281 +0000
@@ -23,26 +23,28 @@
     results, console = run_synthetic()
     stages, steps = results["ladder"]["stages"], results["ladder"]["steps"]
     tele = steps[steps["fs_id"] == "NA|A|0|0|0|0|tele"].sort_values("ladder_step")
-    check(list(tele["group_id"][:3]) == ["NA|A|0|0|0|0|tele"] * 3 and tele["group_id"].iloc[-1] == "*|*|SIG=neutro|*",
-          "an id only changes when the group really changes: NA|A|tele keeps its own id until it joins others")
+    check(list(tele["group_id"][:2]) == ["NA|A|0|0|0|0|tele"] * 2 and tele["group_id"].iloc[2] == "NA|A|SIG=neutro|*"
+          and tele["group_id"].iloc[-1] == "NA|A|SIG=neutro|*",
+          "an id only changes when the group really changes: NA|A|tele keeps its own id until it joins its sibling web")
     negative = stages[stages["fs_id"] == "EU|A|0|0|1|0|web"].iloc[0]
     check(negative["stage0_id"] == "EU|A|0|0|1|0|web" and negative["stage0_support"] == 10
           and negative["stage1_id"] == "EU|A|SIG=negativo|web" and negative["stage1_support"] == 42
           and negative["stage3_id"] == negative["stage2_id"],
           "a negative series: own support 10 at stage 0, 42 after the sign; nothing more in stages 2 and 3")
-    core = results["core"]
-    raw = core[core["origen_fila"] == "raw"]
+    core, dimension = results["core"], results["forecast_series"]
+    raw = core[core["origen_fila"] == "raw"].merge(dimension, on="s03_fs_id")
     check(all(raw.groupby(f"s10_stage{stage}_id")["s00_vencen_usd"].sum().sum() == raw["s00_vencen_usd"].sum() for stage in range(4)),
-          "grouping the core by the id of any stage, the USD due adds up to the same total")
+          "grouping the core (joined to its dimension) by the id of any stage, the USD due adds up to the same total")
     check(raw["s10_stage3_id"].nunique() <= raw["s10_stage0_id"].nunique(), "fewer groups at stage 3 than at stage 0")
-    check({"s17_confidence", "s11_credibility_effect_pp", "s10_stage3_support"} <= set(core.columns),
-          "the core carries the stages, the credibility effect and the confidence")
+    check({"s17_confidence"} <= set(core.columns) and {"s11_credibility_effect_pp", "s10_stage3_support", "s17_confidence"}
+          <= set(dimension.columns),
+          "the core carries the confidence of every row; the dimension the stages, the credibility effect and the confidence")
 
 
 def test_the_audit_tables() -> None:
     print("B · the audit tables add up to their sources")
     results, console = run_synthetic()
-    check("20 checks: 20 ok" in console.split("STEP AUD")[1], "the 20 checks of the audit step pass")
+    check("18 checks: 18 ok" in console.split("STEP AUD")[1], "the 18 checks of the audit step pass")
     audit = results["audit"]
     techniques = audit["composition_techniques"]
     check(set(techniques["status"]) <= {"tested", "not_enough_history", "composition_below_floor"}
@@ -72,33 +74,31 @@
           "the future rate of every technique, one chosen per composition and horizon")
 
 
-def test_the_exam_in_the_core() -> None:
-    print("C · the exam of every forecast series: its interval, and its sums from the core")
+def test_the_exam_in_the_dimension() -> None:
+    print("C · the exam of every forecast series: its interval, raw against framework, summed from the dimension")
     results, console = run_synthetic()
-    backtest = results["audit"]["series_backtest"]
-    row = backtest.iloc[0]
-    check(row["band_low_rate"] <= row["pred_rate"] <= row["band_high_rate"]
-          and row["in_band"] == int(row["band_low_rate"] - 1e-12 <= row["real_rate"] <= row["band_high_rate"] + 1e-12),
+    detail = results["series_exam"]["detail"]
+    row = detail[detail["method"] == "framework"].iloc[0]
+    check(row["band_low"] <= row["pred_rate"] <= row["band_high"]
+          and row["in_band"] == int(row["band_low"] - 1e-12 <= row["real_rate"] <= row["band_high"] + 1e-12),
           "every test has its interval around the prediction, and says whether the real rate fell inside")
-    core = results["core"]
-    chosen = backtest[backtest["is_chosen"] == 1]
-    check(core["s19_exam_predictions"].sum() == len(chosen) and core["s19_exam_in_band"].sum() == chosen["in_band"].sum(),
-          "SUM over the core counts every exam prediction once (counts on the first row of each series)")
-    check(abs(core["s19_exam_real_units"].sum() - chosen["real_units"].sum()) < 1e-6
-          and abs(core["s19_exam_pred_units"].sum() - chosen["pred_units"].sum()) < 1e-6,
-          "SUM over the core gives the units predicted and real of the exam")
-    one_series = core[core["s03_fs_id"] == "EU|A|0|0|0|0|web"]
-    check(one_series["s19_exam_coverage"].nunique() == 1 and one_series["s19_exam_predictions"].notna().sum() == 1,
-          "a ratio is repeated on every row of the series; a count only on its first row")
-    check({"s13_series_phi", "s13_series_trend", "s13_series_seasonal", "s13_series_measurable"} <= set(core.columns),
-          "the core carries each series' own dynamics, to cross it with the precision")
+    dimension, per_series = results["forecast_series"], results["series_exam"]["per_series"]
+    check(dimension["s19_exam_predictions"].sum() == per_series["framework_predictions"].sum()
+          and dimension["s19_exam_in_band"].sum() == per_series["framework_in_band"].sum()
+          and abs(dimension["s19_exam_real_units"].sum() - per_series["framework_real_units"].sum()) < 1e-6,
+          "SUM over the dimension counts every exam prediction once")
+    check({"s19_raw_mae_pp", "s19_exam_mae_pp", "s19_improvement_mae_pp", "s10_stage0_support", "s10_stage3_support",
+           "s08_error_binomial_pp", "s11_se_prediccion_pp"} <= set(dimension.columns),
+          "the dimension shows the improvement over the raw: support, binomial error and exam error, raw against framework")
+    check({"s13_series_phi", "s13_series_trend", "s13_series_seasonal", "s13_series_measurable"} <= set(dimension.columns),
+          "the dimension carries each series' own dynamics, to cross it with the precision")
     report = open(f"{synthetic_with().output_folder}/informe_sff.md", encoding="utf-8").read()
-    check("Cuánto acertamos" in report and "dentro de su intervalo" in report,
-          "the report says how much we erred and how many predictions were in their interval, by type of series")
+    check("Cuánto acertamos" in report and "frente a la serie sola" in report,
+          "the report says how much we erred, how many predictions were in their interval, and the framework against the raw")
 
 
 if __name__ == "__main__":
     test_the_stages()
     test_the_audit_tables()
-    test_the_exam_in_the_core()
+    test_the_exam_in_the_dimension()
     finish()
```

## test_step_nucleo.py

```diff
--- antes/test_step_nucleo.py	2026-10-01 08:08:00.822274916 +0000
+++ test_step_nucleo.py	2026-10-01 08:24:53.067386387 +0000
@@ -13,12 +13,12 @@
 
 
 def test_the_core() -> None:
-    print("A · the core after steps 00-11")
+    print("A · the core (fact table) and the forecast series dimension")
     configuration = synthetic_with()
     results = {}
     console = console_of(lambda: results.update(run(configuration)))
-    core, legend = results["core"], results["core_legend"]
-    check("7 checks: 7 ok" in console.split("STEP NU")[1], "the 7 checks of the core pass")
+    core, legend, dimension = results["core"], results["core_legend"], results["forecast_series"]
+    check("10 checks: 10 ok" in console.split("STEP NU")[1], "the 10 checks of the core pass")
     extended = int(results["forecast"]["forecast"]["_fila"].isna().sum())
     time_series_rows = len(results["time_series_rows"])
     synthetic_time_series = int(results["time_series"]["origen"].isin(["ts_proyectado", "ts_reentrada"]).sum())
@@ -39,24 +39,31 @@
     expected_2026 = core[(core["fin_ano"] == 2026) & (core["fin_estado"] == "previsto")]["fin_renovado_usd"].sum()
     check(abs(real_2026 + expected_2026 - renewed_2026) < 0.01, "2026 splits into real (done) + previsto (to renew)")
     gaps = core[core["origen_fila"] == "hueco"]
-    check(len(gaps) > 0 and (gaps["s00_vencen_unidades"] == 0).all() and gaps["s08_tasa_propia"].notna().all(),
-          "a gap row has every measure at 0 and carries its series values")
+    check(len(gaps) > 0 and (gaps["s00_vencen_unidades"] == 0).all() and gaps["s03_fs_id"].isin(dimension["s03_fs_id"]).all(),
+          "a gap row has every measure at 0 and its forecast series is in the dimension")
     raw = configuration.read_raw()
     check(abs(core.loc[core["fin_universo"] == "pipeline", "s00_vencen_usd"].sum()
               - raw.loc[raw["flag_time_series"] != 1, "total_tr_usd"].sum()) < 0.01,
           "Σ USD due in the core = Σ in the extract without the time_series universe (it is simulated apart, step 20)")
-    one_series = core[core["s03_fs_id"] == "EU|A|0|0|0|0|web"]
-    check(one_series["s11_tasa_estimada"].nunique() == 1 and one_series["s11_nivel_riesgo"].nunique() == 1,
-          "a value of the series is the same on every row of the series")
+    check(dimension["s03_fs_id"].is_unique and set(dimension["s03_fs_id"]) == set(core["s03_fs_id"].dropna())
+          and not any(column.startswith(("s06_", "s08_", "s10_", "s11_", "s19_")) for column in core.columns),
+          "every value of a forecast series lives once, in sff_forecast_series; the core keeps the rows")
+    one_series = dimension[dimension["s03_fs_id"] == "EU|A|0|0|0|0|web"]
     check(not any(column.endswith("_key") for column in core.columns), "no hash keys in the core")
-    check(set(legend["columna"]) == set(core.columns) and set(legend["nivel"]) == {"fila", "unidad", "serie"},
-          "the legend describes every column with its level")
+    core_legend_rows = legend[legend["tabla"] == "sff_nucleo"]
+    dimension_legend_rows = legend[legend["tabla"] == "sff_forecast_series"]
+    check(set(core_legend_rows["columna"]) == set(core.columns) and set(dimension_legend_rows["columna"]) == set(dimension.columns),
+          "the legend describes every column of both tables")
     closed = core[core["s02_rol"].isin(["entrenamiento", "examen"]) & (core["s03_fs_id"] == "EU|A|0|0|0|0|web")]
     check(abs(closed["s02_renovadas_unidades"].sum() / closed["s00_vencen_unidades"].sum()
               - one_series["s08_tasa_propia"].iloc[0]) < 1e-12,
           "the ratio of sums over the closed rows of a series = its own rate (what the Power BI measure computes)")
     written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_nucleo", configuration.sql_engine)["n"].item()
-    check(written == len(core), "sff_nucleo written")
+    written_dimension = pd.read_sql("SELECT COUNT(*) AS n FROM sff_forecast_series", configuration.sql_engine)["n"].item()
+    check(written == len(core) and written_dimension == len(dimension), "sff_nucleo and sff_forecast_series written")
+    summed = dimension["s19_exam_real_units"].sum()
+    check(abs(summed - results["series_exam"]["per_series"]["framework_real_units"].sum()) < 1e-6,
+          "a SUM over the dimension counts every forecast series' exam once")
 
 
 if __name__ == "__main__":
```

## vocabulario.py

```diff
--- antes/vocabulario.py	2026-10-01 08:08:00.672521693 +0000
+++ vocabulario.py	2026-10-01 08:21:05.217668349 +0000
@@ -195,7 +195,12 @@
 TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
 TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
 TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
-TABLE_SERIES_EXAM = "series_exam"                        # every forecast series: how its chosen technique did in the exam
+TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
+TABLE_DIMENSION_LEVELS = "dimension_levels"              # step 02b: the generated groups of every leveled dimension
+TABLE_LADDER_MERGES = "ladder_merges"                    # step 10: every merge of every series, and whether it improves
+TABLE_COMPOSITION_MEMBERS = "composition_members"        # step 10: every series in the rate of every composition (uses / lends)
+TABLE_SERIES_EXAM = "series_exam"                        # step 19: every forecast series: raw and framework in the exam
+TABLE_SERIES_EXAM_DETAIL = "series_exam_detail"          # step 19: series × exam month × h × method, every prediction traced
 TECHNIQUE_TESTED = "tested"
 TECHNIQUE_NOT_ENOUGH_HISTORY = "not_enough_history"
 TECHNIQUE_COMPOSITION_BELOW_FLOOR = "composition_below_floor"
```
