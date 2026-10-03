# Cambio · el criterio de los niveles generados, su evidencia por pantalla, y el peso en dinero de las señales

## Paso 01b
- **Umbral de fusión derivado** (por defecto): dos valores vecinos se juntan mientras sus tasas estandarizadas difieren
  menos que el ruido binomial de una serie en el suelo de soporte, 100·√(p(1−p)/support_floor), con p la tasa de
  entrenamiento. Con tus datos (p = 0,64, suelo 30): 8,8 pp. `level_merge_max_pp` fija otro valor (por ejemplo 5).
- **Evidencia por pantalla** (acción 7): el criterio de cada dimensión; el ruido según el tamaño de la serie; los grupos
  con su tasa de cada año en columnas; cada valor con su soporte, tasa, ruido y años; y las decisiones de fusión en orden,
  con lo que habría dicho el 5 pp fijo.
- **El JSON guarda** cada valor, cada decisión y el criterio. **Se regenera si cambia el umbral.**
- **Tabla nueva** `sff_dimension_level_values` (una fila por valor). `sff_dimension_levels` lleva la tasa por año en
  columnas.
- **Una dimensión que termina en un solo grupo** se avisa ("ONE group").

## Paso 09
Una mandatory sin variación (por ejemplo, un nivel generado de un solo grupo) no es una pasada de la escalera: orden de
colapso 0; la escalera solo la usa para buscar referencia. Así no se pierde una de las `collapse_passes`.

## Paso 01
Las señales (comprobaciones 12-15) y el descuento desconocido (16) dicen el % de filas **y el % del dinero que vence**.

## Paso 02
Textos al día: tres roles, seis columnas nuevas.

Verificación: 22 ficheros de test en verde.

## config.py

```diff
--- /tmp/before_levels2/config.py	2026-10-01 14:18:36.281634149 +0000
+++ config.py	2026-10-01 14:19:34.777511637 +0000
@@ -238,7 +238,9 @@
                                                       # right after it, when the Config is built
     levels_path: Optional[str] = None                 # the JSON of the generated groups (None: <output_folder>/sff_levels.json);
                                                       # a later run reuses it; delete it to regenerate
-    level_merge_max_pp: float = 5.0                   # two neighbouring values merge while their rates differ by at most this
+    level_merge_max_pp: Optional[float] = None        # two neighbouring values merge while their standardised rates differ
+                                                      # by at most this (pp). None: the binomial noise of a series at the
+                                                      # support floor, 100·√(p(1−p)/support_floor), p = the training rate
     save_checkpoints: bool = True                     # every step saves its tables, so a later run can start from any step
     checkpoint_folder: Optional[str] = None           # where (None: <output_folder>/checkpoints); one .pkl per table + manifest
     structural_timevarying_dims: dict = field(default_factory=dict)   # column → "negative" | "positive"
```

## step_01_validate_values.py

```diff
--- /tmp/before_levels2/step_01_validate_values.py	2026-10-01 14:18:35.849957616 +0000
+++ step_01_validate_values.py	2026-10-01 14:18:36.368072418 +0000
@@ -131,6 +131,13 @@
                             failure_detail=f"empty values in dimensions (rows per column): {empty_by_dimension}",
                             context=f"{len(dimension_columns(configuration))} dimensions")
 
+    # the share of rows AND the share of the USD due: a flag on many small rows weighs little in money
+    usd_due = validated[configuration.pipeline_usd_col]
+    total_usd_due = float(usd_due.sum())
+
+    def share_of_money(rows_mask) -> str:
+        return f"{usd_due[rows_mask].sum() / total_usd_due:.0%} of the USD due" if total_usd_due else "no USD due"
+
     dichotomous_columns = list(configuration.structural_timevarying_dims) + [configuration.flag_time_series_col]
     for dichotomous_column in dichotomous_columns:
         values = validated[dichotomous_column]
@@ -140,7 +147,8 @@
                                 not unexpected_values and not has_nulls,
                                 failure_detail=f"{dichotomous_column} must be 0 / 1: found {unexpected_values[:10]}"
                                                f"{' and nulls' if has_nulls else ''}",
-                                context=f"{int(values.isin(ACTIVE_FLAG_VALUES).mean() * 100)} % of rows at 1")
+                                context=f"{values.isin(ACTIVE_FLAG_VALUES).mean():.0%} of rows at 1 · "
+                                        f"{share_of_money(values.isin(ACTIVE_FLAG_VALUES))}")
 
     # [4] the exact discount: a share between 0 and 1, or null (unknown)
     if configuration.discount_value_column:
@@ -152,7 +160,8 @@
                                 out_of_range == 0,
                                 failure_detail=f"{out_of_range:,} rows of {configuration.discount_value_column} "
                                                f"outside [0, 1] (the discount is a share: 0.25 = 25 %)",
-                                context=f"{discount_values.isna().mean():.1%} unknown")
+                                context=f"{discount_values.isna().mean():.1%} of rows unknown · "
+                                        f"{share_of_money(discount_values.isna())}")
     else:
         configuration.log_action(STEP_LABEL, 4, "no exact discount declared: nothing to check")
 
```

## step_01b_dimension_levels.py

```diff
--- /tmp/before_levels2/step_01b_dimension_levels.py	2026-10-01 14:18:35.850577059 +0000
+++ step_01b_dimension_levels.py	2026-10-01 14:19:34.778247040 +0000
@@ -16,12 +16,23 @@
      RESIDUAL group: too little to tell their rate.
   3. the others are ordered (ordinal: by their value, numbers as numbers; nominal: by their rate) and
      the two NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
-     level_merge_max_pp points (the ±5 pp promise by default).
-  4. the rate of every group, year by year, goes with it as evidence of its stability.
+     the MERGE THRESHOLD.
+  4. the rate of every value and of every group, year by year, goes with them as evidence.
+
+THE MERGE THRESHOLD (the same principle as the ladder: merge while the bias it adds is smaller than the
+noise it removes). The generated level is only used by the series that climb the ladder, the ones below
+the support floor; their monthly rate carries at least the binomial noise of a series AT the floor,
+√(p(1−p)/support_floor). Two values whose rates differ by less than that cannot be told apart by any
+series that will use the merge: merging them adds less bias than the noise it removes. With p = 0.64 and
+a floor of 30, the threshold is 8.8 pp. level_merge_max_pp fixes another value (5 pp = the noise of a
+series of about 90 contracts a month).
+
+A dimension that ends in a single group does not separate the renewal rate: its level_1 is constant and
+the ladder gives it no pass (step 09).
 
 THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
 the ids of the forecast series do not move from one month to the next. A dimension missing from the file
-(or with another type) is generated and added. To regenerate everything, delete the file. A value the file
+(or with another type or another merge threshold) is generated and added. To regenerate everything, delete the file. A value the file
 does not know goes to the residual group, with a warning.
 
 Actions (logged as they are done):
@@ -29,13 +40,13 @@
   2. generate the groups of every dimension that needs them (training months)
   3. fill <column>_level_1
   4. check the values the file knows                                     check 1
-  5. write the levels table                                              check 2
+  5. write the levels tables                                             checks 2-3
   6. count the checks; stop if any failed
-  7. show the groups and the columns by role, as tables
+  7. show the criterion, the groups (rate per year), the values and the merge decisions, as tables
 
 Checks (logged as they are made, numbered, at the level of their status):
    1. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
-   2. table sff_dimension_levels written and read back
+   2-3. tables sff_dimension_levels and sff_dimension_level_values written and read back
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
@@ -48,7 +59,7 @@
 import pandas as pd
 
 from config import GENERATED_LEVEL_SUFFIX, Config, join_columns, roles_overview
-from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVELS
+from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVEL_VALUES, TABLE_DIMENSION_LEVELS
 
 
 STEP_LABEL = "01b"
@@ -60,10 +71,11 @@
                 "generate the groups of every dimension that needs them (training months)",
                 "fill <column>_level_1",
                 "check the values the file knows (check 1)",
-                "write the levels table (check 2)",
+                "write the levels tables (checks 2-3)",
                 "count the checks; stop if any failed",
-                "show the groups and the columns by role, as tables"]
-STEP_OUTPUT = "the raw with <column>_level_1 · the JSON of the groups · table sff_dimension_levels"
+                "show the criterion, the groups (rate per year), the values and the merge decisions, as tables"]
+STEP_OUTPUT = ("the raw with <column>_level_1 · the JSON of the groups · tables sff_dimension_levels (groups) and "
+               "sff_dimension_level_values (values)")
 
 RESIDUAL_GROUP = "residual"
 
@@ -88,7 +100,8 @@
     path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
     stored = read_levels_file(path)
     to_generate = [column_name for column_name, level_type in configuration.leveled_dims.items()
-                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type]
+                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type
+                   or stored["dims"][column_name].get("criterion", {}).get("fixed_pp") != configuration.level_merge_max_pp]
     configuration.log_action(STEP_LABEL, 1, f"{len(configuration.leveled_dims)} dimensions with a generated level "
                                             f"{list(configuration.leveled_dims)} · file {path}: "
                                             + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
@@ -127,16 +140,16 @@
 
     # [5] the table
     configuration.log_action(STEP_LABEL, 5, "writing the levels")
-    levels_table = levels_as_table(stored, configuration)
+    levels_table, values_table, decisions_table = levels_as_tables(stored, configuration)
     configuration.write_table(STEP_LABEL, check_log, levels_table, TABLE_DIMENSION_LEVELS)
+    configuration.write_table(STEP_LABEL, check_log, values_table, TABLE_DIMENSION_LEVEL_VALUES)
 
     # [6] the count of the checks
     configuration.log_action(STEP_LABEL, 6, "counting the checks")
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
 
-    # [7] the groups and the columns by role
-    configuration.log_action(STEP_LABEL, 7, "the groups of every dimension (rate_std: standardised by cell, training months):")
-    configuration.show_table(levels_table.drop(columns=["rate_by_year"], errors="ignore"))
+    # [7] the evidence: the criterion, the groups year by year, the values, the merge decisions
+    show_the_evidence(stored, levels_table, values_table, decisions_table, configuration)
     configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, with the generated levels:")
     configuration.show_table(roles_overview(raw.columns, configuration))
     return raw
@@ -147,9 +160,19 @@
     return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]
 
 
+def merge_threshold_pp(global_rate: float, configuration: Config) -> tuple:
+    """(threshold in pp, how it was set): the binomial noise of a series at the support floor, or the fixed value."""
+    if configuration.level_merge_max_pp is not None:
+        return float(configuration.level_merge_max_pp), f"fixed in the Config (level_merge_max_pp = {configuration.level_merge_max_pp})"
+    noise = 100 * np.sqrt(global_rate * (1 - global_rate) / configuration.support_floor)
+    return float(noise), (f"the binomial noise of a series at the support floor: 100·√(p(1−p)/{configuration.support_floor:.0f}) "
+                          f"with p = {global_rate:.2f}")
+
+
 def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
-    """The groups of one dimension, from the training months: standardised rate per value, residual for
-    the rare ones, neighbours merged while their rates differ by at most level_merge_max_pp."""
+    """The groups of one dimension, from the training months: standardised rate per value, residual for the
+    rare ones, neighbours merged while their rates differ by at most the merge threshold. Keeps the evidence:
+    every value, every group year by year, every merge decision."""
     due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
     training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
     train = raw[training & (raw[due].to_numpy() > 0)].copy()
@@ -159,61 +182,74 @@
     train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
     cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
     train["_expected"] = train["_cell"].map(cell_rate) * train[due]
-    global_rate = train[renewed].sum() / train[due].sum()
+    train["_year"] = train[period].map(lambda month: month.year)
+    global_rate = float(train[renewed].sum() / train[due].sum())
+    threshold_pp, threshold_rule = merge_threshold_pp(global_rate, configuration)
 
     per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
     per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
     rare = per_value["monthly_support"] < configuration.support_floor
+
+    def standardised(values: list, rows: pd.DataFrame = None) -> float:
+        if rows is None:
+            block = per_value.loc[values]
+            return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)
+        expected = rows["_expected"].sum()
+        return float(global_rate * rows[renewed].sum() / expected) if expected > 0 else np.nan
+
+    def by_year(values: list) -> dict:
+        block = train[train["_value"].isin(values)]
+        return {str(year): round(standardised(values, rows), 4) for year, rows in block.groupby("_year")}
+
     groups = [[value] for value in per_value.index[~rare]]
     if level_type == "ordinal":
         groups.sort(key=lambda group: natural_key(group[0]))
     else:
-        groups.sort(key=lambda group: per_value.loc[group[0], "observed"] / max(per_value.loc[group[0], "expected"], 1e-9))
+        groups.sort(key=lambda group: standardised(group))
 
-    def standardised(group: list) -> float:
-        block = per_value.loc[group]
-        return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)
+    def label(group: list) -> str:
+        ordered = sorted(group, key=natural_key)
+        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"
 
-    # merge the closest neighbours while they differ by at most level_merge_max_pp
+    # merge the closest neighbours while they differ by at most the threshold; every decision is kept
+    decisions = []
     while len(groups) > 1:
         differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
         closest = int(np.argmin(differences))
-        if 100 * differences[closest] > configuration.level_merge_max_pp:
+        if 100 * differences[closest] > threshold_pp:
             break
+        decisions.append({"left": label(groups[closest]), "right": label(groups[closest + 1]),
+                          "difference_pp": round(100 * differences[closest], 2), "decision": "merged"})
         groups[closest:closest + 2] = [groups[closest] + groups[closest + 1]]
+    for index in range(len(groups) - 1):                       # the neighbours that stay apart, and by how much
+        decisions.append({"left": label(groups[index]), "right": label(groups[index + 1]),
+                          "difference_pp": round(100 * abs(standardised(groups[index]) - standardised(groups[index + 1])), 2),
+                          "decision": "kept apart"})
 
-    def label(group: list) -> str:
-        ordered = sorted(group, key=natural_key)
-        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"
-
-    mapping, evidence = {}, []
-    for group in groups:
-        group_label = label(group)
-        for value in group:
-            mapping[value] = group_label
-        evidence.append(group_evidence(group_label, group, per_value, train, standardised(group), global_rate, configuration))
+    mapping, group_evidence, value_evidence = {}, [], []
+    final_groups = [(label(group), group) for group in groups]
     if rare.any():
-        rare_values = list(per_value.index[rare])
-        for value in rare_values:
-            mapping[value] = RESIDUAL_GROUP
-        evidence.append(group_evidence(RESIDUAL_GROUP, rare_values, per_value, train, standardised(rare_values), global_rate,
-                                       configuration))
-    return {"type": level_type, "max_merge_pp": configuration.level_merge_max_pp,
-            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping, "groups": evidence}
-
-
-def group_evidence(group_label: str, values: list, per_value: pd.DataFrame, train: pd.DataFrame, rate: float,
-                   global_rate: float, configuration: Config) -> dict:
-    """The evidence of one group: its values, units, support, standardised rate and that rate year by year."""
-    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
-    block = train[train["_value"].isin(values)]
-    by_year = {}
-    for year, rows in block.groupby(block[period].map(lambda month: month.year)):
-        expected = rows["_expected"].sum()
-        by_year[str(year)] = round(float(global_rate * rows[renewed].sum() / expected), 4) if expected > 0 else None
-    return {"group": group_label, "values": sorted(values, key=natural_key), "units_due": float(per_value.loc[values, "units_due"].sum()),
-            "monthly_support": float(block.groupby(period)[due].sum().median()), "rate_std": round(float(rate), 4),
-            "rate_by_year": by_year}
+        final_groups.append((RESIDUAL_GROUP, list(per_value.index[rare])))
+    for group_label, values in final_groups:
+        for value in values:
+            mapping[value] = group_label
+            value_rate = standardised([value])
+            support = float(per_value.loc[value, "monthly_support"])
+            value_evidence.append({"value": value, "group": group_label, "units_due": float(per_value.loc[value, "units_due"]),
+                                   "monthly_support": support, "rate_std": round(value_rate, 4),
+                                   "noise_pp": round(100 * np.sqrt(max(value_rate * (1 - value_rate), 0) / max(support, 1)), 2),
+                                   "rate_by_year": by_year([value])})
+        block = train[train["_value"].isin(values)]
+        group_evidence.append({"group": group_label, "values": sorted(values, key=natural_key),
+                               "units_due": float(per_value.loc[values, "units_due"].sum()),
+                               "monthly_support": float(block.groupby(period)[due].sum().median()),
+                               "rate_std": round(standardised(values), 4), "rate_by_year": by_year(values)})
+    value_evidence.sort(key=lambda row: natural_key(row["value"]))
+    return {"type": level_type,
+            "criterion": {"threshold_pp": round(threshold_pp, 2), "rule": threshold_rule, "fixed_pp": configuration.level_merge_max_pp,
+                          "global_rate": round(global_rate, 4), "support_floor": configuration.support_floor},
+            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping,
+            "groups": group_evidence, "values": value_evidence, "decisions": decisions}
 
 
 def read_levels_file(path: str) -> dict:
@@ -233,13 +269,52 @@
         json.dump(content, handle, ensure_ascii=False, indent=2)
 
 
-def levels_as_table(stored: dict, configuration: Config) -> pd.DataFrame:
-    """One row per dimension × group: the evidence of the JSON as a table (for Power BI and the report)."""
-    rows = []
+def levels_as_tables(stored: dict, configuration: Config) -> tuple:
+    """The evidence of the JSON as tables: groups, values and merge decisions (one row each), with the rate
+    of every year as a column of its own, readable on screen and in Power BI."""
+    group_rows, value_rows, decision_rows = [], [], []
+    for column_name in configuration.leveled_dims:
+        stored_dimension = stored["dims"][column_name]
+        threshold = stored_dimension.get("criterion", {}).get("threshold_pp")
+        for group in stored_dimension["groups"]:
+            group_rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
+                               "n_values": len(group["values"]), "units_due": group["units_due"],
+                               "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
+                               **{f"rate_{year}": rate for year, rate in group["rate_by_year"].items()}})
+        for value in stored_dimension.get("values", []):
+            value_rows.append({"dimension": column_name, "value": value["value"], "group": value["group"],
+                               "units_due": value["units_due"], "monthly_support": value["monthly_support"],
+                               "rate_std": value["rate_std"], "noise_pp": value["noise_pp"],
+                               **{f"rate_{year}": rate for year, rate in value["rate_by_year"].items()}})
+        for decision in stored_dimension.get("decisions", []):
+            decision_rows.append({"dimension": column_name, **decision, "threshold_pp": threshold,
+                                  "within_5_pp": "yes" if decision["difference_pp"] <= 5 else "no"})
+    return pd.DataFrame(group_rows), pd.DataFrame(value_rows), pd.DataFrame(decision_rows)
+
+
+def show_the_evidence(stored: dict, levels_table: pd.DataFrame, values_table: pd.DataFrame,
+                      decisions_table: pd.DataFrame, configuration: Config) -> None:
+    """Action 7: the criterion, then the groups year by year, the values and the merge decisions."""
     for column_name in configuration.leveled_dims:
-        for group in stored["dims"][column_name]["groups"]:
-            rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
-                         "n_values": len(group["values"]), "units_due": group["units_due"],
-                         "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
-                         "rate_by_year": json.dumps(group["rate_by_year"])})
-    return pd.DataFrame(rows)
+        criterion = stored["dims"][column_name].get("criterion", {})
+        groups = [group for group in stored["dims"][column_name]["groups"] if group["group"] != RESIDUAL_GROUP]
+        configuration.log_action(STEP_LABEL, 7, f"{column_name} ({stored['dims'][column_name]['type']}): neighbours merge while their "
+                                                f"standardised rates differ by ≤ {criterion.get('threshold_pp', '?')} pp — "
+                                                f"{criterion.get('rule', '?')} · {len(groups)} groups"
+                                                + (" · ONE group: the dimension does not separate the renewal rate beyond that "
+                                                   "noise; its level_1 is constant and gets no pass of the ladder (step 09)"
+                                                   if len(groups) == 1 else ""))
+    rate = next(iter(stored["dims"].values())).get("criterion", {}).get("global_rate", 0.64)
+    noise_rows = [{"contracts_a_month": support, "binomial_noise_pp": round(100 * np.sqrt(rate * (1 - rate) / support), 1)}
+                  for support in (10, 30, 50, 100, 271, 1000)]
+    configuration.logger.doc(f"[{STEP_LABEL}] what a difference in pp means: the noise of the monthly rate of a series of n contracts "
+                             f"(p = {rate:.2f}); a merge helps the series whose noise is larger than the difference it adds:")
+    configuration.show_table(pd.DataFrame(noise_rows))
+    configuration.logger.doc(f"[{STEP_LABEL}] the groups (rate_std: standardised by cell; rate_<year>: the same, year by year — "
+                             f"a group is stable when its years stay close):")
+    configuration.show_table(levels_table)
+    configuration.logger.doc(f"[{STEP_LABEL}] every value (noise_pp: the binomial noise of its own monthly rate; two values whose "
+                             f"rates differ by less than their noise cannot be told apart month by month):")
+    configuration.show_table(values_table)
+    configuration.logger.doc(f"[{STEP_LABEL}] the merge decisions, in order (within_5_pp: what the old fixed 5 pp would have said):")
+    configuration.show_table(decisions_table)
```

## step_02_apply_calendar.py

```diff
--- /tmp/before_levels2/step_02_apply_calendar.py	2026-10-01 14:18:36.132647203 +0000
+++ step_02_apply_calendar.py	2026-10-01 14:18:36.368806288 +0000
@@ -29,7 +29,7 @@
   8. show the calendar per role, as a table
 
 Checks (logged as they are made, numbered, at the level of their status):
-   1. every row has one of the four roles
+   1. every row has one of the three roles
    2. training has at least a year of months                      (warning only)
    3. every exam month has rows                                   (warning only)
    4. the current month has pipeline                              (warning only)
@@ -40,7 +40,7 @@
    9. the s0_ columns hold the raw's renewals and pipeline, untouched
   10. the table sff_calendario is written and read back
 
-Output: a copy of the raw with four new columns (rol, es_mes_en_curso, s0_renovados_*),
+Output: a copy of the raw with six new columns (rol, es_mes_en_curso, s0_renovados_*, s0_vencen_*),
 and the table sff_calendario (one row per month: its role and its money).
 """
 
@@ -69,7 +69,8 @@
                 "build the calendar table (one row per month) and write it (check 10)",
                 "count the checks; stop if any failed",
                 "show the calendar per role, as a table"]
-STEP_OUTPUT = ("the raw with four new columns (rol, es_mes_en_curso, s0_renovados_unidades, s0_renovados_usd) · "
+STEP_OUTPUT = ("the raw with six new columns (rol, es_mes_en_curso, s0_renovados_unidades, s0_renovados_usd, "
+               "s0_vencen_unidades, s0_vencen_usd) · "
                "table sff_calendario (one row per month)")
 
 # ─── named constants ─────────────────────────────────────────────────────────────
@@ -187,10 +188,10 @@
     """Checks 1 to 4: roles assigned, enough training, exam with data, pipeline this month."""
     period_column = configuration.period_col
 
-    # [1] every row has one of the four roles
+    # [1] every row has one of the three roles
     unknown_roles = sorted(set(calendared[CALENDAR_ROLE_COLUMN]) - set(ROLES_IN_ORDER))
     rows_per_role = calendared[CALENDAR_ROLE_COLUMN].value_counts()
-    configuration.log_check(STEP_LABEL, check_log, "every row has one of the four roles", not unknown_roles,
+    configuration.log_check(STEP_LABEL, check_log, "every row has one of the three roles", not unknown_roles,
                             failure_detail=f"unknown roles: {unknown_roles}",
                             context=" · ".join(f"{role} {int(rows_per_role.get(role, 0)):,}" for role in ROLES_IN_ORDER))
 
```

## step_09_dimensions.py

```diff
--- /tmp/before_levels2/step_09_dimensions.py	2026-10-01 14:18:35.911781723 +0000
+++ step_09_dimensions.py	2026-10-01 14:20:13.597230385 +0000
@@ -36,7 +36,7 @@
 Checks (logged as they are made, numbered, at the level of their status):
    1. the base has at least 3 series (else every figure is 0)       (warning only)
    2. every figure is between 0 and 1
-   3. every mandatory dimension has one collapse position, 1 to M
+   3. every mandatory dimension with variation has one collapse position, 1 to K (one without variation: 0, no pass)
    4. a *_level_N never collapses before its *_level_(N+1)
    5-6. tables sff_decision_eta2 and sff_decision_eta2_pares written and read back
 
@@ -209,7 +209,12 @@
                                             f"{r2_all:.3f}")
 
     # [3] the collapse order of the mandatory dimensions
-    collapse_order = sequential_collapse_order(base, mandatory_dims)
+    # a dimension with a single value (e.g. a generated level that ended in one group) has nothing to
+    # collapse: it is not a pass of the ladder
+    without_variation = [dimension for dimension in mandatory_dims if base[dimension].nunique() <= 1]
+    if without_variation:
+        configuration.log_action(STEP_LABEL, 3, f"no variation, not a pass of the ladder: {without_variation}")
+    collapse_order = sequential_collapse_order(base, [dimension for dimension in mandatory_dims if dimension not in without_variation])
     position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
     loss_of = dict(collapse_order)
     decision["orden_colapso"] = decision["dimension"].map(position_of).fillna(0).astype(int)
@@ -282,20 +287,22 @@
                             failure_detail=f"{len(out_of_range)} dimensions with a figure outside [0, 1]",
                             examples=out_of_range)
 
-    # [3] one position per mandatory dimension, 1 to M
+    # [3] one position per mandatory dimension with variation, 1 to K; the ones without variation: 0 (no pass)
     mandatory_dims = configuration.business_mandatory_dims
-    positions = sorted(decision.loc[decision["grupo"] == "mandatory", "orden_colapso"])
-    configuration.log_check(STEP_LABEL, check_log, "every mandatory dimension has one collapse position, 1 to M",
-                            positions == list(range(1, len(mandatory_dims) + 1)),
+    positions = sorted(position for position in decision.loc[decision["grupo"] == "mandatory", "orden_colapso"] if position > 0)
+    without_variation = int((decision.loc[decision["grupo"] == "mandatory", "orden_colapso"] == 0).sum())
+    configuration.log_check(STEP_LABEL, check_log, "every mandatory dimension with variation has one collapse position, 1 to K",
+                            positions == list(range(1, len(mandatory_dims) - without_variation + 1)),
                             failure_detail=f"positions found: {positions}",
-                            context=f"{len(mandatory_dims)} mandatory dimensions")
+                            context=f"{len(positions)} with a pass · {without_variation} without variation")
 
     # [4] the hierarchy: a coarser level never goes before a finer one of its family
     position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
     broken = []
-    for dimension in mandatory_dims:
-        family, level = family_and_level(dimension, mandatory_dims)
-        finer = [other for other in mandatory_dims if family_and_level(other, mandatory_dims) == (family, level + 1)]
+    ordered_dims = [dimension for dimension in mandatory_dims if dimension in position_of]      # the ones that are a pass
+    for dimension in ordered_dims:
+        family, level = family_and_level(dimension, ordered_dims)
+        finer = [other for other in ordered_dims if family_and_level(other, ordered_dims) == (family, level + 1)]
         for finer_dimension in finer:
             if position_of[dimension] < position_of[finer_dimension]:
                 broken.append(f"{dimension} before {finer_dimension}")
```

## vocabulario.py

```diff
--- /tmp/before_levels2/vocabulario.py	2026-10-01 14:18:35.850760301 +0000
+++ vocabulario.py	2026-10-01 14:19:34.778715375 +0000
@@ -196,6 +196,7 @@
 TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
 TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
 TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
+TABLE_DIMENSION_LEVEL_VALUES = "dimension_level_values"  # step 01b: every value of every leveled dimension, its rate and group
 TABLE_DIMENSION_LEVELS = "dimension_levels"              # step 01b: the generated groups of every leveled dimension
 TABLE_LADDER_MERGES = "ladder_merges"                    # step 10: every merge of every series, and whether it improves
 TABLE_COMPOSITION_MEMBERS = "composition_members"        # step 10: every series in the rate of every composition (uses / lends)
```
