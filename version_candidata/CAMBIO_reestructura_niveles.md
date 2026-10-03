# Cambio · niveles generados: estructura en lugar de parche

## Por qué

Con term en una sola columna, el paso 02 borraba la pipeline de todas las licencias desde 2027-09. El primer arreglo
añadió lógica (una función que resolvía dos nombres de columna, una parada, avisos). La causa era de estructura:

1. el paso de niveles iba **después** del calendario sin necesitarlo;
2. **renombraba** la columna raw (`tr_term` → `tr_term_level_2`), así que su nombre dependía del paso;
3. **modificaba la Config** en ejecución, lo que obligó a guardar y restaurar esos cambios en los checkpoints.

## Qué cambia

| Antes | Ahora |
|---|---|
| paso 02b, después del calendario | **paso 01b**, después de validar el raw y antes del calendario (los meses de entrenamiento salen de la Config) |
| `tr_term` → `tr_term_level_2` (raw) + `tr_term_level_1` | `tr_term` **conserva su nombre** (nivel fino) + `tr_term_level_1` (generado) |
| `leveled_dims={"tr_term": {"source": ..., "type": "ordinal"}}` | `leveled_dims={"tr_term": "ordinal"}` |
| el paso 02b añadía las columnas a las mandatory en ejecución | la **Config** las añade al construirse, justo detrás de su columna; ningún paso la modifica |
| `term_column_in` resolvía dos nombres y se detenía si no encontraba ninguno | `term_column` debe ser una mandatory: se comprueba **al construir la Config**; `is_one_year` vuelve a una línea |
| checkpoints con `config_writes` | checkpoints solo con tablas |
| regla de familias `X_level_N` | la misma regla, ampliada: `X` es el nivel más fino cuando existe `X_level_N` |
| — | `extract_mandatory_dims`: las mandatory que trae el extracto (las validan los pasos 00 y 01) |

Se mantiene, porque no es un parche: la línea del log del paso 02 con las filas borradas por plazo, y la separación en el
paso 04 entre unidades borradas por el calendario y unidades a 0 en el extracto.

**Saldo:** 32 líneas menos; desaparecen `term_column_in`, `source`, el renombrado, `replace_in_mandatory` y
`config_writes`.

## Configuración de producción (`main.py`)

```python
business_mandatory_dims=[..., "tr_term", "tr_band", "tr_master_partner_code"],
leveled_dims={"tr_term": "ordinal", "tr_band": "ordinal"},
term_column="tr_term",
```

**Efectos:** los ids de las forecast series llevan `tr_term` en lugar de `tr_term_level_2`, y el JSON de niveles se
regenera (bórralo si existe). Hay que ejecutar desde el principio.

Verificación: 22 ficheros de test en verde (309 comprobaciones). `test_step_02b.py` pasa a `test_step_01b.py`.

## config.py

```diff
--- /tmp/before_restructure/config.py	2026-10-01 10:05:23.479280247 +0000
+++ config.py	2026-10-01 10:08:04.399564197 +0000
@@ -131,34 +131,12 @@
     return buckets
 
 
-def term_column_in(columns, configuration) -> Optional[str]:
-    """The column of these rows that holds the term. term_column itself if present; when term is a leveled
-    dimension (step 02b), the column that holds its raw value at this point: the source before step 02b,
-    <name>_level_2 after it, so term_column may name either. None when no term_column is configured.
-    A term_column that cannot be found STOPS: treating every licence as 1-year would wipe and project
-    the pipeline of the multi-year licences too."""
-    if not configuration.term_column:
-        return None
-    columns = set(columns)
-    if configuration.term_column in columns:
-        return configuration.term_column
-    for name, spec in (configuration.leveled_dims or {}).items():
-        source = (spec or {}).get("source", name)
-        if configuration.term_column in (name, source, f"{name}_level_2"):
-            for candidate in (f"{name}_level_2", source, name):
-                if candidate in columns:
-                    return candidate
-    raise ValueError(f"term_column '{configuration.term_column}' is not among the columns of the rows, nor the raw or "
-                     f"level_2 column of a leveled dimension: the 1-year licences cannot be told apart")
-
-
 def is_one_year(rows: pd.DataFrame, configuration) -> pd.Series:
-    """The rows of a 1-year licence (the term column = one_year_term_value); every row only when no
-    term_column is configured at all (then every licence is treated as 1-year)."""
-    column = term_column_in(rows.columns, configuration)
-    if column is None:
+    """The rows of a 1-year licence (term_column = one_year_term_value); every row when no term_column is
+    configured. term_column is a mandatory dim (checked when the Config is built): it is in every step."""
+    if not configuration.term_column:
         return pd.Series(True, index=rows.index)
-    return rows[column].astype(str) == str(configuration.one_year_term_value)
+    return rows[configuration.term_column].astype(str) == str(configuration.one_year_term_value)
 
 
 ROLE_PURPOSES = [
@@ -172,7 +150,9 @@
     (COLUMN_ROLE_BOTH_EXTRAS, "both extras"),
     (COLUMN_ROLE_FORMULA_INPUT, "inputs of a formula (exact discount, SKU), not a dimension"),
     (COLUMN_ROLE_IGNORE, "read and not used")]
-LEVELED_ROLE = "niveles generados (02b)"
+LEVELED_ROLE = "niveles generados (01b)"
+GENERATED_LEVEL_SUFFIX = "_level_1"           # the coarse level the library generates for a leveled dim
+LEVEL_TYPES = ("ordinal", "nominal")          # ordinal: only neighbouring values merge · nominal: any two
 
 
 def roles_overview(columns, configuration) -> pd.DataFrame:
@@ -187,17 +167,13 @@
         if role == COLUMN_ROLE_BOTH_EXTRAS and not role_columns:
             continue
         rows.append({"rol": role, "columnas": len(role_columns), "nombres": ", ".join(role_columns), "para_que": purpose})
-    leveled = []
-    for name, spec in (configuration.leveled_dims or {}).items():
-        source = (spec or {}).get("source", name)
-        level_type = (spec or {}).get("type", "nominal")
-        made = [column for column in (f"{name}_level_1", f"{name}_level_2") if column in columns]
-        leveled.append(f"{source} → {name}_level_1 (agrupado) + {name}_level_2 (raw) · {level_type}"
-                       + (" · ya generados" if len(made) == 2 else ""))
+    leveled = [f"{column_name} → {column_name}{GENERATED_LEVEL_SUFFIX} ({level_type})"
+               + (" · generado" if f"{column_name}{GENERATED_LEVEL_SUFFIX}" in present else "")
+               for column_name, level_type in configuration.leveled_dims.items()]
     if leveled:
         rows.append({"rol": LEVELED_ROLE, "columnas": len(leveled), "nombres": "; ".join(leveled),
-                     "para_que": "level_1: values grouped by their standardised rate (JSON in levels_path); collapsed before level_1"
-                                 .replace("collapsed before level_1", "the ladder collapses level_2 first, then level_1")})
+                     "para_que": "the column keeps its raw value (fine level); _level_1 groups its values by their standardised "
+                                 "rate (JSON in levels_path); the ladder collapses the fine level first"})
     return pd.DataFrame(rows)
 
 
@@ -255,10 +231,11 @@
     flag_time_series_col: str = "flag_time_series"          # marks the rows of the time_series universe
 
     business_mandatory_dims: list = field(default_factory=list)       # open the series and the uplift cell
-    leveled_dims: dict = field(default_factory=dict)  # dims given two generated levels (step 02b): {name: {"source": raw
-                                                      # column (default: name), "type": "ordinal" | "nominal"}};
-                                                      # <name>_level_2 = the raw value, <name>_level_1 = values grouped
-                                                      # by their standardised renewal rate. The source must be mandatory
+    leveled_dims: dict = field(default_factory=dict)  # mandatory dims that get a generated coarse level (step 01b):
+                                                      # {column: "ordinal" | "nominal"}. The column keeps its raw value
+                                                      # (the fine level); <column>_level_1 groups its values by their
+                                                      # standardised renewal rate and is added to the mandatory dims
+                                                      # right after it, when the Config is built
     levels_path: Optional[str] = None                 # the JSON of the generated groups (None: <output_folder>/sff_levels.json);
                                                       # a later run reuses it; delete it to regenerate
     level_merge_max_pp: float = 5.0                   # two neighbouring values merge while their rates differ by at most this
@@ -400,7 +377,29 @@
         if invalid_signs:
             raise ValueError(f"timevarying signs must be one of {VALID_TIMEVARYING_SIGNS}: {invalid_signs}")
 
-        # [2] no column declared with two roles (built and checked by column_roles), and
+        # [2] the leveled dims: mandatory, with a valid type; their generated level is a mandatory dim from
+        #     here on (step 01b creates it before any step reads the dims)
+        not_mandatory = [column_name for column_name in self.leveled_dims if column_name not in self.business_mandatory_dims]
+        if not_mandatory:
+            raise ValueError(f"leveled_dims must be mandatory dims: {not_mandatory}")
+        invalid_types = {column_name: level_type for column_name, level_type in self.leveled_dims.items()
+                         if level_type not in LEVEL_TYPES}
+        if invalid_types:
+            raise ValueError(f"leveled_dims types must be one of {LEVEL_TYPES}: {invalid_types}")
+        expanded = []
+        for column_name in self.business_mandatory_dims:
+            if column_name in self.generated_columns:
+                continue                                   # re-inserted right after its column
+            expanded.append(column_name)
+            if column_name in self.leveled_dims:
+                expanded.append(f"{column_name}{GENERATED_LEVEL_SUFFIX}")
+        self.business_mandatory_dims = expanded
+
+        # [3] the term of a licence is a declared mandatory dim (it exists in every step)
+        if self.term_column and self.term_column not in self.business_mandatory_dims:
+            raise ValueError(f"term_column '{self.term_column}' must be one of the mandatory dims")
+
+        # [4] no column declared with two roles (built and checked by column_roles), and
         #     the uplift cell takes its mandatory dims from the declared ones
         self.column_roles()
         unknown_uplift_dims = [column_name for column_name in (self.uplift_mandatory_dims or [])
@@ -408,7 +407,7 @@
         if unknown_uplift_dims:
             raise ValueError(f"uplift_mandatory_dims must be mandatory dims: {unknown_uplift_dims}")
 
-        # [3] the calendar parameters are well formed
+        # [5] the calendar parameters are well formed
         if self.current_month is not None:
             parse_month(self.current_month)
         edges = list(self.discount_bucket_edges)
@@ -554,6 +553,17 @@
             print(shown_table.to_string())
 
     @property
+    def generated_columns(self) -> list:
+        """The columns the library adds to the raw (step 01b): the coarse level of every leveled dim."""
+        return [f"{column_name}{GENERATED_LEVEL_SUFFIX}" for column_name in self.leveled_dims]
+
+    @property
+    def extract_mandatory_dims(self) -> list:
+        """The mandatory dims the extract brings (the steps that validate the extract, 00 and 01, read these;
+        from step 01b on, business_mandatory_dims, with the generated levels)."""
+        return [column_name for column_name in self.business_mandatory_dims if column_name not in self.generated_columns]
+
+    @property
     def rate_series_columns(self) -> list:
         """The columns that define ONE renewal-rate series: mandatory + timevarying +
         extra_renovacion, in this order (the field order of fs_id and fu_id)."""
```

## pipeline.py

```diff
--- /tmp/before_restructure/pipeline.py	2026-10-01 10:05:23.478766540 +0000
+++ pipeline.py	2026-10-01 10:07:03.391119944 +0000
@@ -18,8 +18,8 @@
 
 CHECKPOINTS (save_checkpoints, on by default): every step saves the tables it writes to
 checkpoint_folder (one .pkl per table) and records itself in checkpoint_manifest.json: when it
-finished, the hash of its code, and the Config fields it changed (step 02b changes the mandatory
-dims). run_from(step) loads every table of the steps before it and runs from that step on.
+finished and the hash of its code. No step changes the Config, so a checkpoint is only tables.
+run_from(step) loads every table of the steps before it and runs from that step on.
 YOU decide when a full run is needed; the orchestrator only WARNS (it does not stop) when, since the
 checkpoint was saved:
   - the code of a step that is loaded (not re-run) changed, or a shared module changed
@@ -55,7 +55,7 @@
 from step_00_validate_raw import validate_raw
 from step_01_validate_values import validate_values
 from step_02_apply_calendar import apply_calendar
-from step_02b_dimension_levels import apply_dimension_levels
+from step_01b_dimension_levels import apply_dimension_levels
 from step_03_fine_table import build_fine_table
 from step_04_forecast_units import build_forecast_units
 from step_05_lookups import build_lookups
@@ -144,15 +144,14 @@
 
 @dataclass
 class Step:
-    """A step: the tables it reads and writes, how it runs on the context, the module of its code
-    (its hash goes to the checkpoint) and the Config fields it changes while it runs (saved and restored)."""
+    """A step: the tables it reads and writes, how it runs on the context, and the module of its code
+    (its hash goes to the checkpoint). A step never changes the Config."""
     name: str
     label: str
     reads: tuple
     writes: tuple
     run: Callable
     module: str = None
-    config_writes: tuple = ()
 
 
 @dataclass
@@ -253,8 +252,6 @@
                 with open(os.path.join(self.folder, f"{table}.pkl"), "rb") as handle:
                     self.context.tables[table] = pickle.load(handle)
                 self.context.producer[table] = name
-            for field_name, value in saved.get("config_writes", {}).items():
-                setattr(self.configuration, field_name, value)          # what the step had changed in the Config
         last = manifest["steps"][before[-1]]["finished"] if before else "-"
         self.configuration.logger.doc(f"[checkpoint] {len(before)} steps loaded from {self.folder} "
                                       f"({before[0] if before else '-'} … {before[-1] if before else '-'}, saved {last}) "
@@ -299,8 +296,7 @@
                                             "shared_code": code_hash(SHARED_MODULES), "steps": {}}
         manifest["steps"][step.name] = {
             "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
-            "tables": list(step.writes), "code": code_hash([step.module] if step.module else []),
-            "config_writes": {field_name: getattr(self.configuration, field_name) for field_name in step.config_writes}}
+            "tables": list(step.writes), "code": code_hash([step.module] if step.module else [])}
         self.write_manifest(manifest)
 
     def read_manifest(self) -> dict:
@@ -361,12 +357,10 @@
 
 
 def config_fingerprint(configuration: Config, steps) -> dict:
-    """The Config as text, field by field, without what cannot be compared (the engine, the SQL text)
-    and without what the steps themselves change (restored from the checkpoint instead)."""
-    changed_by_steps = {field_name for step in steps for field_name in step.config_writes}
+    """The Config as text, field by field, without what cannot be compared (the engine, the SQL text)."""
     fingerprint = {}
     for config_field in dataclasses.fields(configuration):
-        if config_field.name in changed_by_steps or config_field.name in ("sql_engine", "raw_extract_sql"):
+        if config_field.name in ("sql_engine", "raw_extract_sql"):
             continue
         value = getattr(configuration, config_field.name)
         try:
@@ -489,10 +483,12 @@
         Step("split_time_series", "20a", ("validated_extract",), ("validated_renewals", "time_series_rows"), split, module="step_20_time_series"),
         Step("validate_values", "01", ("validated_renewals",), ("validated_raw",),
              lambda context: {"validated_raw": validate_values(context["validated_renewals"], context.configuration)}, module="step_01_validate_values"),
-        Step("calendar", "02", ("validated_raw",), ("calendared_extract",),
-             lambda context: {"calendared_extract": apply_calendar(context["validated_raw"], context.configuration)}, module="step_02_apply_calendar"),
-        Step("dimension_levels", "02b", ("calendared_extract",), ("calendared_raw",),
-             lambda context: {"calendared_raw": apply_dimension_levels(context["calendared_extract"], context.configuration)}, module="step_02b_dimension_levels", config_writes=("business_mandatory_dims",)),
+        Step("dimension_levels", "01b", ("validated_raw",), ("leveled_raw",),
+             lambda context: {"leveled_raw": apply_dimension_levels(context["validated_raw"], context.configuration)},
+             module="step_01b_dimension_levels"),
+        Step("calendar", "02", ("leveled_raw",), ("calendared_raw",),
+             lambda context: {"calendared_raw": apply_calendar(context["leveled_raw"], context.configuration)},
+             module="step_02_apply_calendar"),
         Step("fine_table", "03", ("calendared_raw",), ("fine_table",),
              lambda context: {"fine_table": build_fine_table(context["calendared_raw"], context.configuration)}, module="step_03_fine_table"),
         Step("forecast_units", "04", ("fine_table",), ("forecast_units",),
```

## step_00_validate_raw.py

```diff
--- /tmp/before_restructure/step_00_validate_raw.py	2026-10-01 10:05:23.474014884 +0000
+++ step_00_validate_raw.py	2026-10-01 10:08:04.400646299 +0000
@@ -76,8 +76,10 @@
                             context=f"{len(raw.columns)} columns")
 
     missing_columns = [column_name for column_name, role in column_roles.items()
-                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns]
-    configuration.log_check(STEP_LABEL, check_log, "every declared column is in the raw (ignored ones may be absent)",
+                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns
+                       and column_name not in configuration.generated_columns]       # made in step 01b
+    configuration.log_check(STEP_LABEL, check_log, "every declared column is in the raw (ignored ones may be absent; "
+                            "the generated levels are made in step 01b)",
                             not missing_columns,
                             failure_detail=f"columns declared in the Config but missing from the raw: {missing_columns}")
 
@@ -170,7 +172,7 @@
 def log_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
     """Action 6: the columns by role and the calendar, rows per role, as tables."""
     configuration.log_action(STEP_LABEL, 6, "the columns of the raw, by role (every role of the Config, also the empty "
-                                            "ones, and the dimensions that get generated levels in step 02b):")
+                                            "ones, and the dimensions that get a generated level in step 01b):")
     configuration.show_table(roles_overview(validated.columns, configuration))
 
     configuration.logger.doc(f"[{STEP_LABEL}] the calendar of the Config on these months: "
```

## step_01_validate_values.py

```diff
--- /tmp/before_restructure/step_01_validate_values.py	2026-10-01 10:05:23.474605560 +0000
+++ step_01_validate_values.py	2026-10-01 10:08:04.400196463 +0000
@@ -203,11 +203,11 @@
 
 def example_rows(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
     """The columns that identify a row and its money, to show a few rows under a check."""
-    return rows[[configuration.period_col] + configuration.business_mandatory_dims + configuration.core_measures]
+    return rows[[configuration.period_col] + configuration.extract_mandatory_dims + configuration.core_measures]
 
 
 def dimension_columns(configuration: Config) -> list:
     """Every dimension column, each once: mandatory, timevarying and both extra groups."""
-    all_dimensions = (configuration.business_mandatory_dims + list(configuration.structural_timevarying_dims)
+    all_dimensions = (configuration.extract_mandatory_dims + list(configuration.structural_timevarying_dims)
                       + configuration.extra_renovacion + configuration.extra_revalorizacion)
     return list(dict.fromkeys(all_dimensions))
```

## step_02_apply_calendar.py

```diff
--- /tmp/before_restructure/step_02_apply_calendar.py	2026-10-01 10:05:23.477949217 +0000
+++ step_02_apply_calendar.py	2026-10-01 10:07:03.390386261 +0000
@@ -48,7 +48,7 @@
 import numpy as np
 import pandas as pd
 
-from config import Config, is_one_year, term_column_in
+from config import Config, is_one_year
 from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                          ROLE_TEST, ROLE_TRAIN, ROLES_IN_ORDER, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                          TABLE_CALENDAR, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN)
@@ -125,15 +125,12 @@
     wiped_pipeline_units = float(calendared.loc[not_known_yet, configuration.pipeline_units_col].sum())
     wiped_pipeline_usd = float(calendared.loc[not_known_yet, configuration.pipeline_usd_col].sum())
     calendared.loc[not_known_yet, [configuration.pipeline_units_col, configuration.pipeline_usd_col]] = 0.0
-    term_column = term_column_in(calendared.columns, configuration)
-    wiped_terms = (calendared.loc[not_known_yet].groupby(calendared.loc[not_known_yet, term_column].astype(str))[
-                   configuration.pipeline_units_col].size().to_dict() if term_column and not_known_yet.any() else {})
+    wiped_per_term = (calendared.loc[not_known_yet, configuration.term_column].astype(str).value_counts().to_dict()
+                      if configuration.term_column else "no term_column: every licence is treated as 1-year")
     configuration.log_action(STEP_LABEL, "4b", f"{int(not_known_yet.sum()):,} rows of 1-year licences due from "
                                             f"{boundaries['current'] + 12} on (sold or renewed from {boundaries['current']} on): "
                                             f"pipeline wiped {wiped_pipeline_units:,.0f} units · ${wiped_pipeline_usd:,.0f} "
-                                            f"(it will be projected) · "
-                                            + (f"term column '{term_column}', rows wiped per term: {wiped_terms}" if term_column
-                                               else "no term_column: every licence is treated as 1-year"))
+                                            f"(it will be projected) · rows wiped per term: {wiped_per_term}")
 
     # [5] the checks
     configuration.log_action(STEP_LABEL, 5, "checking the calendar and the money")
```

## step_09_dimensions.py

```diff
--- /tmp/before_restructure/step_09_dimensions.py	2026-10-01 10:05:23.475344174 +0000
+++ step_09_dimensions.py	2026-10-01 10:07:03.389588693 +0000
@@ -133,13 +133,17 @@
     return float(max(0.0, 1 - residual / total)) if total > 0 else 0.0
 
 
-def family_and_level(dimension_name: str) -> tuple:
-    """("product", 2) for "product_level_2"; (name, 1) for a dimension outside a hierarchy."""
+def family_and_level(dimension_name: str, dimensions=()) -> tuple:
+    """("product", 2) for "product_level_2". A dimension X without a level of its own is the FINEST level
+    of its family when the dimensions include an X_level_N (a leveled dim: its raw value above the level
+    the library generated): (X, highest N + 1). Otherwise (name, 1), a dimension outside a hierarchy."""
     if LEVEL_MARK in dimension_name:
         family, _, level_text = dimension_name.rpartition(LEVEL_MARK)
         if level_text.isdigit():
             return family, int(level_text)
-    return dimension_name, 1
+    coarser = [family_and_level(other)[1] for other in dimensions
+               if other != dimension_name and family_and_level(other)[0] == dimension_name]
+    return dimension_name, (max(coarser) + 1 if coarser else 1)
 
 
 def sequential_collapse_order(base: pd.DataFrame, mandatory_dims: list) -> list:
@@ -152,7 +156,8 @@
     current_r2 = weighted_r2(base, remaining)
     while remaining:
         droppable = [dimension for dimension in remaining
-                     if not any(family_and_level(other) == (family_and_level(dimension)[0], family_and_level(dimension)[1] + 1)
+                     if not any(family_and_level(other, remaining)
+                                == (family_and_level(dimension, remaining)[0], family_and_level(dimension, remaining)[1] + 1)
                                 for other in remaining)]
         losses = {}
         for dimension in droppable:
@@ -289,8 +294,8 @@
     position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
     broken = []
     for dimension in mandatory_dims:
-        family, level = family_and_level(dimension)
-        finer = [other for other in mandatory_dims if family_and_level(other) == (family, level + 1)]
+        family, level = family_and_level(dimension, mandatory_dims)
+        finer = [other for other in mandatory_dims if family_and_level(other, mandatory_dims) == (family, level + 1)]
         for finer_dimension in finer:
             if position_of[dimension] < position_of[finer_dimension]:
                 broken.append(f"{dimension} before {finer_dimension}")
```

## step_17_forecast.py

```diff
--- /tmp/before_restructure/step_17_forecast.py	2026-10-01 10:05:23.477477733 +0000
+++ step_17_forecast.py	2026-10-01 10:06:40.282914833 +0000
@@ -76,7 +76,7 @@
 import numpy as np
 import pandas as pd
 
-from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month, term_column_in
+from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
 from step_14_backtest import band_of_horizon
 from prediction import band_quantiles, predict_composition, rate_band, shifted_rate
 from techniques import inverse_logit, logit, predict_logit
@@ -274,9 +274,8 @@
     if not window:
         return pd.DataFrame()
 
-    term_column = term_column_in(extract_future.columns, configuration)
     carried = list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion
-                                 + ([term_column] if term_column else [])
+                                 + ([configuration.term_column] if configuration.term_column else [])
                                  + ([configuration.acquisition_column] if configuration.acquisition_column else [])))
     carried = [column for column in carried if column in extract_future.columns]
 
@@ -311,8 +310,7 @@
         return pd.DataFrame()
     period, current = configuration.period_col, configuration.calendar_boundaries()["current"]
     timevarying = list(configuration.structural_timevarying_dims)
-    term_column = term_column_in(fine_table.columns, configuration)
-    group_columns = [column for column in carried if column not in timevarying and column != term_column]
+    group_columns = [column for column in carried if column not in timevarying and column != configuration.term_column]
     acquired = fine_table[is_acquisition(fine_table, configuration) & is_one_year(fine_table, configuration)].copy()
     if acquired.empty:
         return pd.DataFrame()
@@ -358,9 +356,8 @@
         return simulated
     for column_name in timevarying:
         simulated[column_name] = inactive_value(fine_table[column_name])
-    term_column = term_column_in(fine_table.columns, configuration)
-    if term_column:
-        simulated[term_column] = configuration.one_year_term_value
+    if configuration.term_column and configuration.term_column in fine_table.columns:
+        simulated[configuration.term_column] = configuration.one_year_term_value
     if configuration.discount_value_column:
         simulated[configuration.discount_value_column] = configuration.acquisition_discount
     simulated[PIPELINE_ORIGIN_COLUMN] = PIPELINE_SIMULATED
```

## main.py

```diff
--- /tmp/before_restructure/main.py	2026-10-01 10:05:23.474256651 +0000
+++ main.py	2026-10-01 10:07:12.250891404 +0000
@@ -68,9 +68,9 @@
         business_mandatory_dims=["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
                                  "tr_product_level_1", "tr_product_level_2", "tr_purchase_type", "tr_renewal_type",
                                  "tr_term", "tr_band", "tr_master_partner_code"],
-        # term and band come in one column each (the raw value); step 02b gives them tr_term_level_1 /
-        # tr_band_level_1 (grouped by the library) and *_level_2 (the raw value). Groups in salida/sff_levels.json
-        leveled_dims={"tr_term": {"type": "ordinal"}, "tr_band": {"type": "ordinal"}},
+        # term and band keep their raw value; step 01b adds tr_term_level_1 / tr_band_level_1 (their values
+        # grouped by the renewal rate). The groups are kept in salida/sff_levels.json
+        leveled_dims={"tr_term": "ordinal", "tr_band": "ordinal"},
         structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                      "not_installed": "negative"},            # [por confirmar] the signs
         extra_renovacion=["net_new", "prev_OperationGroup"],
@@ -82,7 +82,7 @@
         extra_measure_cols=["total_reacquired_units", "total_reacquired_usd", "TR_AUV", "REN_AUV", "ReAC_AUV"],
         ignore_cols=["dataset_role", "is_current_month", "dummy_field", "row_id", "_filter"],   # [por confirmar]
         # the simulation window (current month → December): what happens in it falls due in 2027
-        term_column="tr_term",              # the raw column; after step 02b it is read from tr_term_level_2
+        term_column="tr_term",
         one_year_term_value="1 year",
         acquisition_column="net_new",
         acquisition_values=["Acquisition_Not-New", "Acquisition_Pure-New"],
```

## step_02b_dimension_levels.py → step_01b_dimension_levels.py

```diff
--- /tmp/before_restructure/step_02b_dimension_levels.py	2026-10-01 10:05:23.477798736 +0000
+++ step_01b_dimension_levels.py	2026-10-01 10:06:34.362945040 +0000
@@ -1,47 +1,41 @@
 """
-step_02b_dimension_levels.py — The dimensions with generated levels.
+step_01b_dimension_levels.py — The generated coarse level of the leveled dimensions.
 
-Some dimensions come with many values (a band, a term, a product-usage bucket). Instead of grouping
-them by hand, the Config declares them in leveled_dims and the library gives each two levels:
+Some mandatory dimensions come with many values (a band, a term, a product-usage bucket). The Config
+declares them in leveled_dims ({column: "ordinal" | "nominal"}); the column keeps its raw value (the
+fine level) and the library adds ONE column, <column>_level_1, with its values grouped by their renewal
+rate. The Config already counts <column>_level_1 as a mandatory dim, right after its column, from the
+moment it is built; this step only fills it, early, so every later step sees the same columns. The
+ladder collapses the fine level first, then the coarse one.
 
-  <name>_level_2   the raw value, untouched (the fine detail)
-  <name>_level_1   the values grouped by their renewal rate (the coarse level, optimised)
-
-The ladder collapses level_2 before level_1, as with any _level_N family: the information moves in
-the direction of greater homogeneity, and the coarse groups are now proposed by a rule.
-
-HOW THE GROUPS ARE MADE (only the TRAINING months: the exam must not inform them)
+HOW THE GROUPS ARE MADE (only the TRAINING months, taken from the calendar of the Config)
   1. the STANDARDISED rate of every value: renewed / expected, where expected = the rate of its cell
-     (every other mandatory dim) × its units due, times the global rate. A value is compared with the
-     others inside the same cell, not in bulk: a band sold mostly where people renew less must not
-     look worse because of where it is sold.
+     (every other mandatory dim of the extract) × its units due, times the global rate. A value is
+     compared with the others inside the same cell, not in bulk.
   2. values whose monthly support (median of the monthly units due) is below support_floor go to a
      RESIDUAL group: too little to tell their rate.
-  3. the others are ordered (ordinal: by their value; nominal: by their standardised rate) and the two
-     NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
-     level_merge_max_pp points (the ±5 pp promise by default): a difference smaller than what we
-     promise to measure is not worth a separate group.
+  3. the others are ordered (ordinal: by their value, numbers as numbers; nominal: by their rate) and
+     the two NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
+     level_merge_max_pp points (the ±5 pp promise by default).
   4. the rate of every group, year by year, goes with it as evidence of its stability.
 
-THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it,
-so the ids of the forecast series do not move from one month to the next. A dimension missing from the
-file (or with another source or type) is generated and added. To regenerate everything, delete the file.
-A value the file does not know goes to the residual group, with a warning.
+THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
+the ids of the forecast series do not move from one month to the next. A dimension missing from the file
+(or with another type) is generated and added. To regenerate everything, delete the file. A value the file
+does not know goes to the residual group, with a warning.
 
 Actions (logged as they are done):
   1. the declared dimensions and the file: use it, or generate
   2. generate the groups of every dimension that needs them (training months)
-  3. apply them: <name>_level_1 and <name>_level_2, and the mandatory dims of the Config updated
-  4. check the levels                                                    checks 1-3
-  5. write the levels table                                              check 4
+  3. fill <column>_level_1
+  4. check the values the file knows                                     check 1
+  5. write the levels table                                              check 2
   6. count the checks; stop if any failed
-  7. show the groups, as a table
+  7. show the groups and the columns by role, as tables
 
 Checks (logged as they are made, numbered, at the level of their status):
-   1. every row has its level_1 and level_2 (no value left without a group)
-   2. the units due add up the same grouped by level_1 and by level_2 (a partition)
-   3. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
-   4. table sff_dimension_levels written and read back
+   1. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
+   2. table sff_dimension_levels written and read back
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
@@ -53,52 +47,49 @@
 import numpy as np
 import pandas as pd
 
-from config import Config, join_columns, roles_overview
-from vocabulario import CALENDAR_ROLE_COLUMN, ROLE_TRAIN, TABLE_DIMENSION_LEVELS
+from config import GENERATED_LEVEL_SUFFIX, Config, join_columns, roles_overview
+from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVELS
 
 
-STEP_LABEL = "02b"
-STEP_NAME = "DIMENSIONS WITH GENERATED LEVELS"
-STEP_PURPOSE = ("give the dimensions declared in leveled_dims two levels: level_2 the raw value and level_1 the values "
-                "grouped by their standardised renewal rate (training months only), persisted in a JSON that later runs "
-                "reuse so the forecast series keep their ids")
+STEP_LABEL = "01b"
+STEP_NAME = "DIMENSIONS WITH A GENERATED LEVEL"
+STEP_PURPOSE = ("give every dimension declared in leveled_dims its coarse level, <column>_level_1: its values grouped by "
+                "their standardised renewal rate in the training months, persisted in a JSON that later runs reuse so the "
+                "forecast series keep their ids; the column itself keeps its raw value")
 STEP_ACTIONS = ["the declared dimensions and the file: use it, or generate",
                 "generate the groups of every dimension that needs them (training months)",
-                "apply them: <name>_level_1 and <name>_level_2, and the mandatory dims of the Config updated",
-                "check the levels (checks 1-3)",
-                "write the levels table (check 4)",
+                "fill <column>_level_1",
+                "check the values the file knows (check 1)",
+                "write the levels table (check 2)",
                 "count the checks; stop if any failed",
-                "show the groups, as a table"]
-STEP_OUTPUT = "the raw with <name>_level_1 and <name>_level_2 · the JSON of the groups · table sff_dimension_levels"
+                "show the groups and the columns by role, as tables"]
+STEP_OUTPUT = "the raw with <column>_level_1 · the JSON of the groups · table sff_dimension_levels"
 
 RESIDUAL_GROUP = "residual"
-LEVEL_TYPES = ("ordinal", "nominal")
-UNITS_TOLERANCE = 1e-6
 
 
-def apply_dimension_levels(calendared_raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
-    """The raw with the generated levels of every dimension in leveled_dims.
+def apply_dimension_levels(validated_raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
+    """The raw with the generated coarse level of every dimension in leveled_dims.
 
-    INPUT:   the raw after step 02 (roles of the calendar) · the Config (leveled_dims, levels_path,
-             level_merge_max_pp, support_floor).
-    OUTPUT:  the raw with <name>_level_1 / <name>_level_2; the Config's business_mandatory_dims updated.
+    INPUT:   the raw validated by steps 00 and 01 · the Config (leveled_dims, levels_path, level_merge_max_pp,
+             support_floor, the calendar).
+    OUTPUT:  the raw with <column>_level_1 for every leveled column.
     RULES:   see the module header.
-    EDGE CASES: no leveled_dims → the raw unchanged, nothing written. A dimension with a single value →
-             one group. All values below the floor → a single residual group.
+    EDGE CASES: no leveled_dims → the raw unchanged, nothing written. A dimension with a single value → one
+             group. All values below the floor → a single residual group.
     """
     if not configuration.leveled_dims:
-        return calendared_raw
+        return validated_raw
     configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
     check_log = []
-    raw = calendared_raw.copy()
+    raw = validated_raw.copy()
 
     # [1] the declared dimensions and the file
     path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
     stored = read_levels_file(path)
-    to_generate = [name for name, spec in configuration.leveled_dims.items()
-                   if name not in stored.get("dims", {}) or stored["dims"][name]["source"] != source_of(name, spec)
-                   or stored["dims"][name]["type"] != type_of(spec)]
-    configuration.log_action(STEP_LABEL, 1, f"{len(configuration.leveled_dims)} dimensions with levels "
+    to_generate = [column_name for column_name, level_type in configuration.leveled_dims.items()
+                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type]
+    configuration.log_action(STEP_LABEL, 1, f"{len(configuration.leveled_dims)} dimensions with a generated level "
                                             f"{list(configuration.leveled_dims)} · file {path}: "
                                             + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
                                             + (f"generating {to_generate}" if to_generate else "every dimension in it, reused"))
@@ -106,8 +97,8 @@
     # [2] generate what the file does not have
     if to_generate:
         stored.setdefault("dims", {})
-        for name in to_generate:
-            stored["dims"][name] = generate_levels(raw, name, configuration.leveled_dims[name], configuration)
+        for column_name in to_generate:
+            stored["dims"][column_name] = generate_levels(raw, column_name, configuration.leveled_dims[column_name], configuration)
         stored["created"] = stored.get("created") or datetime.now().isoformat(timespec="seconds")
         stored["updated"] = datetime.now().isoformat(timespec="seconds")
         stored["current_month"] = str(configuration.current_month)
@@ -116,34 +107,22 @@
     else:
         configuration.log_action(STEP_LABEL, 2, "nothing to generate")
 
-    # [3] apply them
+    # [3] fill the coarse level
     unknown_values = {}
-    for name, spec in configuration.leveled_dims.items():
-        source = source_of(name, spec)
-        mapping = stored["dims"][name]["mapping"]
-        values = raw[source].astype(str)
+    for column_name in configuration.leveled_dims:
+        mapping = stored["dims"][column_name]["mapping"]
+        values = raw[column_name].astype(str)
         unknown = sorted(set(values.unique()) - set(mapping))
         if unknown:
-            unknown_values[name] = unknown
-        level_2, level_1 = f"{name}_level_2", f"{name}_level_1"
-        raw[level_2] = raw[source]
-        raw[level_1] = values.map(mapping).fillna(RESIDUAL_GROUP)
-        replace_in_mandatory(configuration, source, name, level_1, level_2)
-    configuration.log_action(STEP_LABEL, 3, "levels applied · mandatory dims now: " + ", ".join(configuration.business_mandatory_dims))
-
-    # [4] the checks
-    configuration.log_action(STEP_LABEL, 4, "checking the levels")
-    missing = sum(int(raw[f"{name}_level_1"].isna().sum() + raw[f"{name}_level_2"].isna().sum()) for name in configuration.leveled_dims)
-    configuration.log_check(STEP_LABEL, check_log, "every row has its level_1 and level_2", missing == 0,
-                            failure_detail=f"{missing:,} rows without a level")
-    due = configuration.pipeline_units_col
-    partition = all(abs(raw.groupby(f"{name}_level_1")[due].sum().sum() - raw.groupby(f"{name}_level_2")[due].sum().sum())
-                    <= UNITS_TOLERANCE for name in configuration.leveled_dims)
-    configuration.log_check(STEP_LABEL, check_log, "the units due add up the same grouped by level_1 and by level_2", partition,
-                            failure_detail="a level loses or duplicates units")
+            unknown_values[column_name] = unknown
+        raw[f"{column_name}{GENERATED_LEVEL_SUFFIX}"] = values.map(mapping).fillna(RESIDUAL_GROUP)
+    configuration.log_action(STEP_LABEL, 3, "filled: " + ", ".join(configuration.generated_columns))
+
+    # [4] the check
+    configuration.log_action(STEP_LABEL, 4, "checking the values the file knows")
     configuration.log_check(STEP_LABEL, check_log, "no value of the extract is unknown to the file", not unknown_values,
-                            failure_detail="values sent to the residual: " + "; ".join(f"{name}: {values[:10]}"
-                                                                                     for name, values in unknown_values.items()),
+                            failure_detail="values sent to the residual: " + "; ".join(f"{column_name}: {values[:10]}"
+                                                                                     for column_name, values in unknown_values.items()),
                             blocking=False)
 
     # [5] the table
@@ -155,49 +134,35 @@
     configuration.log_action(STEP_LABEL, 6, "counting the checks")
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
 
-    # [7] the groups
+    # [7] the groups and the columns by role
     configuration.log_action(STEP_LABEL, 7, "the groups of every dimension (rate_std: standardised by cell, training months):")
     configuration.show_table(levels_table.drop(columns=["rate_by_year"], errors="ignore"))
-    configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, updated with the generated levels:")
+    configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, with the generated levels:")
     configuration.show_table(roles_overview(raw.columns, configuration))
     return raw
 
 
-def source_of(name: str, spec: dict) -> str:
-    """The raw column of a leveled dimension (its own name unless the spec says otherwise)."""
-    return (spec or {}).get("source", name)
-
-
-def type_of(spec: dict) -> str:
-    """ordinal (only neighbouring values merge) or nominal (any two)."""
-    level_type = (spec or {}).get("type", "nominal")
-    if level_type not in LEVEL_TYPES:
-        raise ValueError(f"leveled_dims type must be one of {LEVEL_TYPES}, not {level_type!r}")
-    return level_type
-
-
 def natural_key(value: str) -> list:
     """'2 devices' before '10 devices': numbers compared as numbers."""
     return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]
 
 
-def generate_levels(raw: pd.DataFrame, name: str, spec: dict, configuration: Config) -> dict:
+def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
     """The groups of one dimension, from the training months: standardised rate per value, residual for
     the rare ones, neighbours merged while their rates differ by at most level_merge_max_pp."""
-    source, level_type = source_of(name, spec), type_of(spec)
     due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
-    train = raw[(raw[CALENDAR_ROLE_COLUMN] == ROLE_TRAIN) & (raw[due] > 0)].copy()
-    train["_value"] = train[source].astype(str)
-    other_dims = [column for column in configuration.business_mandatory_dims
-                  if column not in (source, f"{name}_level_1", f"{name}_level_2", name)]
+    training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
+    train = raw[training & (raw[due].to_numpy() > 0)].copy()
+    train["_value"] = train[column_name].astype(str)
+    other_dims = [other for other in configuration.business_mandatory_dims
+                  if other != column_name and other not in configuration.generated_columns]
     train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
     cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
     train["_expected"] = train["_cell"].map(cell_rate) * train[due]
     global_rate = train[renewed].sum() / train[due].sum()
 
     per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
-    monthly = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
-    per_value["monthly_support"] = monthly
+    per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
     rare = per_value["monthly_support"] < configuration.support_floor
     groups = [[value] for value in per_value.index[~rare]]
     if level_type == "ordinal":
@@ -211,7 +176,7 @@
 
     # merge the closest neighbours while they differ by at most level_merge_max_pp
     while len(groups) > 1:
-        differences = [abs(standardised(groups[i]) - standardised(groups[i + 1])) for i in range(len(groups) - 1)]
+        differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
         closest = int(np.argmin(differences))
         if 100 * differences[closest] > configuration.level_merge_max_pp:
             break
@@ -228,11 +193,12 @@
             mapping[value] = group_label
         evidence.append(group_evidence(group_label, group, per_value, train, standardised(group), global_rate, configuration))
     if rare.any():
-        for value in per_value.index[rare]:
+        rare_values = list(per_value.index[rare])
+        for value in rare_values:
             mapping[value] = RESIDUAL_GROUP
-        evidence.append(group_evidence(RESIDUAL_GROUP, list(per_value.index[rare]), per_value, train,
-                                       standardised(list(per_value.index[rare])), global_rate, configuration))
-    return {"source": source, "type": level_type, "max_merge_pp": configuration.level_merge_max_pp,
+        evidence.append(group_evidence(RESIDUAL_GROUP, rare_values, per_value, train, standardised(rare_values), global_rate,
+                                       configuration))
+    return {"type": level_type, "max_merge_pp": configuration.level_merge_max_pp,
             "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping, "groups": evidence}
 
 
@@ -250,17 +216,6 @@
             "rate_by_year": by_year}
 
 
-def replace_in_mandatory(configuration: Config, source: str, name: str, level_1: str, level_2: str) -> None:
-    """The mandatory dims of the Config with the two levels in the place of the source (or of the name)."""
-    dims = list(configuration.business_mandatory_dims)
-    family = (source, name, level_1, level_2)
-    positions = [index for index, column in enumerate(dims) if column in family]
-    position = min(positions) if positions else len(dims)          # where the family already was
-    dims = [column for column in dims if column not in family]
-    dims[position:position] = [level_1, level_2]
-    configuration.business_mandatory_dims = dims
-
-
 def read_levels_file(path: str) -> dict:
     """The stored groups, or an empty dict when there is no file yet."""
     if not os.path.exists(path):
@@ -281,9 +236,9 @@
 def levels_as_table(stored: dict, configuration: Config) -> pd.DataFrame:
     """One row per dimension × group: the evidence of the JSON as a table (for Power BI and the report)."""
     rows = []
-    for name in configuration.leveled_dims:
-        for group in stored["dims"][name]["groups"]:
-            rows.append({"dimension": name, "group": group["group"], "values": ", ".join(group["values"]),
+    for column_name in configuration.leveled_dims:
+        for group in stored["dims"][column_name]["groups"]:
+            rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
                          "n_values": len(group["values"]), "units_due": group["units_due"],
                          "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                          "rate_by_year": json.dumps(group["rate_by_year"])})
```
