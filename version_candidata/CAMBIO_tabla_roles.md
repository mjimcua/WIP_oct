# Cambio · la tabla de roles completa

La tabla "the columns of the raw, by role" (paso 00) muestra ahora **todos los roles de la Config**, en un orden fijo,
también los vacíos (con 0 columnas), con una columna `para_que` que dice para qué sirve cada rol, y una fila
**`niveles generados (02b)`** con las dimensiones que recibirán niveles (`origen → <name>_level_1 (agrupado) +
<name>_level_2 (raw) · tipo`). El paso 02b la vuelve a mostrar **actualizada**: el rol mandatory ya lista las columnas
`level_1` y `level_2`, en el orden de la Config (el de colapso).

Función nueva: `roles_overview(columns, configuration)` en `config.py`.

Verificación: 21 ficheros de test en verde (`test_step_02b` 11/11).

## config.py

```diff
--- /mnt/user-data/outputs/sff_espejo/config.py	2026-10-01 09:37:35.983829000 +0000
+++ config.py	2026-10-01 09:43:45.346973566 +0000
@@ -138,6 +138,46 @@
     return pd.Series(True, index=rows.index)
 
 
+ROLE_PURPOSES = [
+    (COLUMN_ROLE_PERIOD, "the month of the row"),
+    (COLUMN_ROLE_TIME_SERIES_FLAG, "1 = the time_series universe (retail to subscription), projected apart (step 20)"),
+    (COLUMN_ROLE_MEASURE, "summed: units and USD due, renewed, reacquired"),
+    (COLUMN_ROLE_MANDATORY, "open the forecast series and the uplift cell; collapsed in stage 3 of the ladder"),
+    (COLUMN_ROLE_TIMEVARYING, "signals of the customer, summarised by their sign in stage 1"),
+    (COLUMN_ROLE_EXTRA_RENOVACION, "open the forecast series; annulled in stage 2"),
+    (COLUMN_ROLE_EXTRA_REVALORIZACION, "open the uplift cell (the price), not the rate"),
+    (COLUMN_ROLE_BOTH_EXTRAS, "both extras"),
+    (COLUMN_ROLE_FORMULA_INPUT, "inputs of a formula (exact discount, SKU), not a dimension"),
+    (COLUMN_ROLE_IGNORE, "read and not used")]
+LEVELED_ROLE = "niveles generados (02b)"
+
+
+def roles_overview(columns, configuration) -> pd.DataFrame:
+    """Every role of the Config, in a fixed order, with its columns (0 when the role is empty), what it is for,
+    and the dimensions that get generated levels (level_1 grouped by the library, level_2 the raw value)."""
+    columns = list(columns)
+    present = set(columns)
+    roles = configuration.column_roles()                 # in the order the Config declares them
+    rows = []
+    for role, purpose in ROLE_PURPOSES:
+        role_columns = [column for column, column_role in roles.items() if column_role == role and column in present]
+        if role == COLUMN_ROLE_BOTH_EXTRAS and not role_columns:
+            continue
+        rows.append({"rol": role, "columnas": len(role_columns), "nombres": ", ".join(role_columns), "para_que": purpose})
+    leveled = []
+    for name, spec in (configuration.leveled_dims or {}).items():
+        source = (spec or {}).get("source", name)
+        level_type = (spec or {}).get("type", "nominal")
+        made = [column for column in (f"{name}_level_1", f"{name}_level_2") if column in columns]
+        leveled.append(f"{source} → {name}_level_1 (agrupado) + {name}_level_2 (raw) · {level_type}"
+                       + (" · ya generados" if len(made) == 2 else ""))
+    if leveled:
+        rows.append({"rol": LEVELED_ROLE, "columnas": len(leveled), "nombres": "; ".join(leveled),
+                     "para_que": "level_1: values grouped by their standardised rate (JSON in levels_path); collapsed before level_1"
+                                 .replace("collapsed before level_1", "the ladder collapses level_2 first, then level_1")})
+    return pd.DataFrame(rows)
+
+
 def join_columns(frame: pd.DataFrame, columns: list) -> pd.Series:
     """The "|"-joined id of every row from several columns, in the given order (the order
     is part of the id); a null value is written "null". Built on arrays, not Series, so a
```

## step_00_validate_raw.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_00_validate_raw.py	2026-10-01 09:37:36.178735000 +0000
+++ step_00_validate_raw.py	2026-10-01 09:43:32.518367080 +0000
@@ -30,7 +30,7 @@
 # ─── imports ─────────────────────────────────────────────────────────────────────
 import pandas as pd
 
-from config import COLUMN_ROLE_IGNORE, Config, parse_month
+from config import COLUMN_ROLE_IGNORE, Config, parse_month, roles_overview
 from vocabulario import CALENDAR_ROLE_COLUMN
 
 
@@ -169,13 +169,9 @@
 
 def log_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
     """Action 6: the columns by role and the calendar, rows per role, as tables."""
-    configuration.log_action(STEP_LABEL, 6, "the columns of the raw, by role:")
-    columns_by_role = {}
-    for column_name, role in column_roles.items():
-        if column_name in validated.columns:
-            columns_by_role.setdefault(role, []).append(column_name)
-    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "columnas": len(role_columns), "nombres": ", ".join(role_columns)}
-                                           for role, role_columns in columns_by_role.items()]))
+    configuration.log_action(STEP_LABEL, 6, "the columns of the raw, by role (every role of the Config, also the empty "
+                                            "ones, and the dimensions that get generated levels in step 02b):")
+    configuration.show_table(roles_overview(validated.columns, configuration))
 
     configuration.logger.doc(f"[{STEP_LABEL}] the calendar of the Config on these months: "
                              f"{configuration.calendar_description()}")
```

## step_02b_dimension_levels.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_02b_dimension_levels.py	2026-10-01 09:37:36.382396000 +0000
+++ step_02b_dimension_levels.py	2026-10-01 09:43:25.735301614 +0000
@@ -53,7 +53,7 @@
 import numpy as np
 import pandas as pd
 
-from config import Config, join_columns
+from config import Config, join_columns, roles_overview
 from vocabulario import CALENDAR_ROLE_COLUMN, ROLE_TRAIN, TABLE_DIMENSION_LEVELS
 
 
@@ -158,6 +158,8 @@
     # [7] the groups
     configuration.log_action(STEP_LABEL, 7, "the groups of every dimension (rate_std: standardised by cell, training months):")
     configuration.show_table(levels_table.drop(columns=["rate_by_year"], errors="ignore"))
+    configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, updated with the generated levels:")
+    configuration.show_table(roles_overview(raw.columns, configuration))
     return raw
 
 
```

## test_step_02b.py

```diff
--- /mnt/user-data/outputs/sff_espejo/test_step_02b.py	2026-10-01 09:37:37.495182000 +0000
+++ test_step_02b.py	2026-10-01 09:44:23.794308163 +0000
@@ -39,6 +39,23 @@
           "the forecast series keep their compositions from one run to the next")
 
 
+def test_the_role_table() -> None:
+    print("C · the table of roles: every role of the Config, and the leveled dims, before and after step 02b")
+    from config import roles_overview
+    configuration = synthetic_with(leveled_dims={"product": {"type": "nominal"}}, extra_revalorizacion=[])
+    raw = configuration.read_raw()
+    before = roles_overview(raw.columns, configuration)
+    check({"extra_revalorizacion", "niveles generados (02b)"} <= set(before["rol"])
+          and before.set_index("rol").loc["extra_revalorizacion", "columnas"] == 0,
+          "every role is listed, also an empty one (0 columns), and the dims that will get levels")
+    raw = raw.assign(product_level_1="g", product_level_2=raw["product"])
+    configuration.business_mandatory_dims = ["region", "product_level_1", "product_level_2"]
+    after = roles_overview(raw.columns, configuration).set_index("rol")
+    check(after.loc["mandatory", "nombres"] == "region, product_level_1, product_level_2"
+          and "ya generados" in after.loc["niveles generados (02b)", "nombres"],
+          "after step 02b the mandatory role lists the generated levels, in the order of the Config")
+
+
 def test_the_rules() -> None:
     print("B · the rules: ordinal order, the place in the mandatory dims")
     check(sorted(["10 devices", "2 devices", "1 device"], key=natural_key) == ["1 device", "2 devices", "10 devices"],
@@ -52,5 +69,6 @@
 
 if __name__ == "__main__":
     test_the_levels()
+    test_the_role_table()
     test_the_rules()
     finish()
```
