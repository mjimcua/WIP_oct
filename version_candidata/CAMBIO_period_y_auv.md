# Cambio · el mes de cada fila, leído una vez por valor; la Config sin los AUV

## 1. Convertir `period` (35 s con 1 millón de filas)

El paso 00 interpretaba el mes fila a fila (1.023.291 llamadas), aunque la comprobación 6 ya interpreta cada valor
distinto una vez (61). Ahora `check_months` devuelve también esa correspondencia (valor → mes) y la acción 5 la aplica
a las filas. Medido con 1 millón de filas: 12,5 s → 0,1 s, con el mismo resultado. No hace falta tocar el SQL: cualquier
formato que ya se aceptaba ("2026-09", "2026-09-01", "01/09/2026", una fecha) sigue valiendo.

## 2. Los AUV fuera de la Config de producción

Ningún cálculo usa `TR_AUV`, `REN_AUV` ni `ReAC_AUV`; solo estaban declarados en `extra_measure_cols`. Al quitarlos del
raw hay que quitarlos también de la Config (si no, la comprobación 4 del paso 00 se detiene: la Config debe describir lo
que trae el extracto). El `main.py` ya no los declara. Un AUV, donde haga falta, se calcula como Σ USD / Σ unidades.

Verificación: 22 ficheros de test en verde.

## step_00_validate_raw.py

```diff
--- /tmp/s00_before_period.py	2026-10-01 13:35:16.573050438 +0000
+++ step_00_validate_raw.py	2026-10-01 13:35:28.601180947 +0000
@@ -83,10 +83,10 @@
                             not missing_columns,
                             failure_detail=f"columns declared in the Config but missing from the raw: {missing_columns}")
 
-    # [3] the months, and the calendar on them
-    raw_months = []
+    # [3] the months, and the calendar on them (every distinct value read once)
+    raw_months, months_by_value = [], {}
     if period_column in raw.columns and period_column not in duplicated_columns:
-        raw_months = check_months(raw, configuration, check_log)
+        raw_months, months_by_value = check_months(raw, configuration, check_log)
     else:
         configuration.log_action(STEP_LABEL, 3, f"no usable '{period_column}' column: the months cannot be read")
         configuration.log_not_evaluated(STEP_LABEL, check_log, "the months and the calendar",
@@ -96,9 +96,9 @@
     configuration.log_action(STEP_LABEL, 4, "counting the checks")
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
 
-    # [5] the period as a monthly Period
+    # [5] the period as a monthly Period: the month of every distinct value, already read in [3], mapped onto the rows
     validated = raw.copy()
-    validated[period_column] = pd.PeriodIndex([parse_month(value) for value in validated[period_column]], freq="M")
+    validated[period_column] = pd.PeriodIndex(validated[period_column].map(months_by_value), freq="M")
     configuration.log_action(STEP_LABEL, 5, f"'{period_column}' converted to a monthly Period")
 
     # [6] what the raw is
@@ -106,8 +106,9 @@
     return validated
 
 
-def check_months(raw: pd.DataFrame, configuration: Config, check_log: list) -> list:
-    """Checks 5 to 10: the months can be read and the calendar fits them. Returns the sorted months."""
+def check_months(raw: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
+    """Checks 5 to 10: the months can be read and the calendar fits them. Returns the sorted months and
+    the month of every distinct value of the period column (read once each)."""
     period_column = configuration.period_col
     period_values = raw[period_column]
 
@@ -132,7 +133,7 @@
     if not raw_months:
         configuration.log_not_evaluated(STEP_LABEL, check_log, "the calendar against the months of the raw",
                                         "no month could be read")
-        return raw_months
+        return raw_months, months_by_value
 
     # [7] the current month is declared (the calendar has no default date)
     try:
@@ -141,7 +142,7 @@
         configuration.log_check(STEP_LABEL, check_log, "current_month is declared", False, failure_detail=str(error))
         configuration.log_not_evaluated(STEP_LABEL, check_log, "the calendar against the months of the raw",
                                         "no current month")
-        return raw_months
+        return raw_months, months_by_value
     configuration.log_check(STEP_LABEL, check_log, "current_month is declared", True, context=str(boundaries["current"]))
 
     # [8] the current month is inside the raw
@@ -166,7 +167,7 @@
     configuration.log_check(STEP_LABEL, check_log, "every month between the first and the last has rows",
                             not months_without_rows,
                             failure_detail=f"months with no row at all: {months_without_rows}", blocking=False)
-    return raw_months
+    return raw_months, months_by_value
 
 
 def log_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
```

## main.py

```diff
--- /tmp/main_before_auv.py	2026-10-01 13:35:16.574927879 +0000
+++ main.py	2026-10-01 13:35:28.601541017 +0000
@@ -79,7 +79,7 @@
         # the same as the extract's old discount_interval)
         discount_value_column="discount",
         # the other columns of the extract
-        extra_measure_cols=["total_reacquired_units", "total_reacquired_usd", "TR_AUV", "REN_AUV", "ReAC_AUV"],
+        extra_measure_cols=["total_reacquired_units", "total_reacquired_usd"],   # an AUV is USD / units: computed, not read
         ignore_cols=["dataset_role", "is_current_month", "dummy_field", "row_id", "_filter"],   # [por confirmar]
         # the simulation window (current month → December): what happens in it falls due in 2027
         term_column="tr_term",
```
