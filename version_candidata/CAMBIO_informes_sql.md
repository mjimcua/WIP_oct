# Cambio · los informes agregados imprimen su SQL; el paso 21 es un informe; la pipeline parcial

## 1. El paso 21 es un informe, no una tabla
`sff_filas_nuevas` ya no se escribe en SQL: todo lo que contiene es una consulta sobre `sff_nucleo`. El paso lo muestra
por pantalla, alimenta el capítulo 2 del informe e imprime la consulta que lo reproduce.

## 2. `pipeline_borrada` → `pipeline_parcial_borrada`
Es la pipeline de las licencias de 1 año que vencen 12 meses después del mes en curso o más tarde. La generan ventas y
renovaciones que solo han empezado, así que está a medio crear; el paso 17 la reconstruye entera. Descripción corregida
en el código, el informe y la guía.

## 3. Los informes agregados imprimen la consulta SQL que los reproduce
`report_queries.py` (nuevo) y `Config.show_query`. Cada consulta usa los nombres reales de las tablas (esquema y prefijo
de la Config) y SQL que corren SQL Server y SQLite. `test_report_queries.py` ejecuta cada una sobre las tablas del
sintético y comprueba que da exactamente el mismo informe:

| Paso | Informe | Desde |
|---|---|---|
| 02 | calendario por rol; calendario por mes (`sff_calendario`) | `sff_nucleo` |
| 19 | examen por método y horizonte; total de cada mes de examen | `sff_series_exam_detail` |
| 20 | total por año y origen (`sff_forecast_total`) | `sff_nucleo` |
| 21 | filas que el framework añade o borra | `sff_nucleo` |

Sin consulta: los informes con medianas o cuantiles (resumen de la escalera, bandas de error) y los niveles generados
(salen del JSON).

Verificación: 24 ficheros de test en verde (336 comprobaciones).

## config.py

```diff
--- /tmp/before_queries/config.py	2026-10-01 16:37:36.993118543 +0000
+++ config.py	2026-10-01 16:37:37.119086855 +0000
@@ -542,6 +542,10 @@
                        context=f"{rows_read_back:,} rows × {len(persisted_frame.columns)} columns → "
                                f"{destination} ({elapsed_seconds:.1f}s)")
 
+    def show_query(self, step_label: str, what: str, sql: str) -> None:
+        """The SQL that reproduces an aggregated report from the tables the run writes (report_queries.py)."""
+        self.logger.doc(f"[{step_label}] {what}: the SQL that reproduces it from the tables of the run:\n{sql}")
+
     def show_table(self, table: pd.DataFrame) -> None:
         """Concrete rows (examples) as a pandas table, never through the logger: rendered
         with display() in a notebook, printed aligned (to_string) in a terminal. The log
```

## step_02_apply_calendar.py

```diff
--- /tmp/before_queries/step_02_apply_calendar.py	2026-10-01 16:37:36.772701270 +0000
+++ step_02_apply_calendar.py	2026-10-01 16:38:11.132775511 +0000
@@ -49,6 +49,7 @@
 import pandas as pd
 
 from config import Config, is_one_year
+from report_queries import calendar_query, roles_query
 from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                          ROLE_TEST, ROLE_TRAIN, ROLES_IN_ORDER, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                          TABLE_CALENDAR, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN)
@@ -287,3 +288,5 @@
             "tasa_unidades": renewed_units / pipeline_units if pipeline_units > 0 else np.nan,
         })
     configuration.show_table(pd.DataFrame(summary_rows))
+    configuration.show_query(STEP_LABEL, "the calendar per role", roles_query(configuration))
+    configuration.show_query(STEP_LABEL, "the calendar per month (sff_calendario)", calendar_query(configuration))
```

## step_19_series_exam.py

```diff
--- /tmp/before_queries/step_19_series_exam.py	2026-10-01 16:37:36.772564845 +0000
+++ step_19_series_exam.py	2026-10-01 16:38:11.133231226 +0000
@@ -52,6 +52,7 @@
 
 from config import Config, join_columns
 from prediction import band_quantiles, levels_at_origins, predict_composition, rate_band, shifted_rate
+from report_queries import exam_by_month_query, exam_summary_query
 from step_14_backtest import band_of_horizon
 from techniques import CATALOGUE, logit
 from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST,
@@ -175,6 +176,9 @@
     configuration.log_action(STEP_LABEL, 7, "the methods on the same rows (error_total: of the sum of the portfolio; wape_series: "
                                             "series by series, no compensation):")
     configuration.show_table(summary)
+    configuration.show_query(STEP_LABEL, "the exam per method and horizon", exam_summary_query(configuration))
+    configuration.show_query(STEP_LABEL, "the total of every exam month per method (sff_examen_cartera)",
+                             exam_by_month_query(configuration, methods))
     return dict(detail=detail, per_series=per_series, by_month=by_month, summary=summary)
 
 
```

## step_20_time_series.py

```diff
--- /tmp/before_queries/step_20_time_series.py	2026-10-01 16:37:36.825211862 +0000
+++ step_20_time_series.py	2026-10-01 16:38:11.134321714 +0000
@@ -75,6 +75,7 @@
 import pandas as pd
 
 from config import ACTIVE_FLAG_VALUES, Config, join_columns, parse_month
+from report_queries import forecast_total_query
 from vocabulario import (CALENDAR_ROLE_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL,
                          PIPELINE_SIMULATED, TABLE_FORECAST_TOTAL, TABLE_TIME_SERIES, TOTAL_ORIGIN_EXPECTED,
                          TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL,
@@ -242,6 +243,7 @@
     # [9] the total and the regions
     configuration.log_action(STEP_LABEL, 9, "the total by year × origin (usd_vence: pipeline; usd_renovado: renewals / revenue):")
     configuration.show_table(total)
+    configuration.show_query(STEP_LABEL, "the total by year and origin (sff_forecast_total)", forecast_total_query(configuration))
     if len(projected):
         configuration.logger.doc(f"[{STEP_LABEL}] the {TOP_REGIONS_SHOWN} regions with the most projected value:")
         configuration.show_table(projected.join(levels_of_region, on=region).groupby(region_columns)
```

## step_21_new_rows.py

```diff
--- /tmp/before_queries/step_21_new_rows.py	2026-10-01 16:37:36.993390952 +0000
+++ step_21_new_rows.py	2026-10-01 16:38:35.610650024 +0000
@@ -10,15 +10,19 @@
                                measure is 0: it adds a month, not money.
   resultado_adelantado_borrado step 02 · a renewal already booked from the current month on: the month
                                has not finished (or has not started); the forecast predicts it instead.
-  pipeline_borrada             step 02 · the pipeline of a 1-year licence sold or renewed from the
-                               current month on (due 12 months later): it is not known yet.
+  pipeline_parcial_borrada     step 02 · the PARTIAL pipeline of the 1-year licences due 12 months after
+                               the current month or later: it is generated by the sales and renewals
+                               from the current month on, which have only started (the current month is
+                               half way, the next ones have not begun). Left as it is, the forecast of
+                               those months would run on a pipeline half built; step 17 rebuilds it whole.
   proyectada                   step 17 · the expected renewal of a 1-year licence due in the simulation
                                window, due 12 months later: what replaces the wiped pipeline.
   simulada                     step 17 · an acquisition simulated in the simulation window, due 12 months
                                later: also replaces wiped pipeline.
 
 One row per month × origin: rows, series, units and USD (due, or wiped), and USD expected to renew for
-the rows the forecast predicts.
+the rows the forecast predicts. It is a REPORT, not a table of the run: everything in it is a query over
+sff_nucleo, and the step prints that query (report_queries.py) instead of writing a second copy.
 
 Actions (logged as they are done):
   1. the gaps of step 08
@@ -26,13 +30,11 @@
   3. what step 17 created
   4. the table: month × origin; the totals per origin, side by side
   5. check that the raw = what stays + what was wiped                    check 1
-  6. write the table                                                     check 2
-  7. count the checks; stop if any failed
-  8. show the totals per origin and the months with activity, as tables
+  6. count the checks; stop if any failed
+  7. show the totals per origin and the months with activity, as tables, and the SQL that reproduces them
 
 Checks (logged as they are made, numbered, at the level of their status):
-   1. what the raw had due = what stays due + the pipeline wiped (units and USD)
-   2. table sff_filas_nuevas written and read back
+   1. what the raw had due = what stays due + the partial pipeline wiped (units and USD)
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
@@ -40,9 +42,10 @@
 import pandas as pd
 
 from config import Config
+from report_queries import new_rows_query
 from vocabulario import (CALENDAR_ROLE_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_SIMULATED,
                          S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
-                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_NEW_ROWS)
+                         SERIES_ID_COLUMN, SYNTHETIC_COLUMN)
 
 
 STEP_LABEL = "21"
@@ -55,18 +58,17 @@
                 "what step 17 created",
                 "the table: month × origin; the totals per origin, side by side",
                 "check that the raw = what stays + what was wiped (check 1)",
-                "write the table (check 2)",
                 "count the checks; stop if any failed",
-                "show the totals per origin and the months with activity, as tables"]
-STEP_OUTPUT = "one row per month × origin (rows, series, units, USD, USD expected) · table sff_filas_nuevas"
+                "show the totals per origin and the months with activity, as tables, and the SQL that reproduces them"]
+STEP_OUTPUT = "one row per month × origin (rows, series, units, USD, USD expected): a report, with its SQL over sff_nucleo"
 
 ORIGIN_GAP = "hueco"
 ORIGIN_WIPED_RESULT = "resultado_adelantado_borrado"
-ORIGIN_WIPED_PIPELINE = "pipeline_borrada"
+ORIGIN_WIPED_PIPELINE = "pipeline_parcial_borrada"
 ORIGINS_IN_ORDER = [ORIGIN_GAP, ORIGIN_WIPED_RESULT, ORIGIN_WIPED_PIPELINE, PIPELINE_PROJECTED, PIPELINE_SIMULATED]
 WHAT_IT_IS = {ORIGIN_GAP: "added (step 08): a month with nothing due inside the history of a series, so it has no holes",
               ORIGIN_WIPED_RESULT: "wiped (step 02): a renewal booked from the current month on; the forecast predicts it",
-              ORIGIN_WIPED_PIPELINE: "wiped (step 02): pipeline of a 1-year licence sold or renewed from the current month on",
+              ORIGIN_WIPED_PIPELINE: "wiped (step 02): partial pipeline of 1-year licences, half built by sales and renewals that have only started",
               PIPELINE_PROJECTED: "created (step 17): expected renewal of a 1-year licence due in the window, 12 months later",
               PIPELINE_SIMULATED: "created (step 17): acquisition simulated in the window, due 12 months later"}
 UNITS_TOLERANCE = 1e-6
@@ -78,7 +80,7 @@
 
     INPUT:   the fine table (step 03, with the s0_ columns of step 02) · the forecast units with the gaps (step 08)
              · the future rows of the forecast (step 17).
-    OUTPUT:  month × origin: rows, series, units, USD, USD expected; written as sff_filas_nuevas.
+    OUTPUT:  month × origin: rows, series, units, USD, USD expected (a report: its SQL over sff_nucleo is printed).
     RULES:   counts what the steps did; computes nothing new.
     EDGE CASES: an origin with no rows does not appear; a run without a simulation window has no proyectada / simulada.
     """
@@ -131,24 +133,21 @@
     wiped = table[table["origen"] == ORIGIN_WIPED_PIPELINE]
     reconciles = (abs(raw_units - kept_units - wiped["unidades"].sum()) <= UNITS_TOLERANCE * max(raw_units, 1)
                   and abs(raw_usd - kept_usd - wiped["usd"].sum()) <= UNITS_TOLERANCE * max(raw_usd, 1))
-    configuration.log_check(STEP_LABEL, check_log, "what the raw had due = what stays due + the pipeline wiped (units and USD)",
+    configuration.log_check(STEP_LABEL, check_log, "what the raw had due = what stays due + the partial pipeline wiped (units and USD)",
                             reconciles, failure_detail="the wiped pipeline does not explain the difference with the raw",
                             context=f"raw {raw_units:,.0f} units · ${raw_usd:,.0f} = kept {kept_units:,.0f} · ${kept_usd:,.0f} "
                                     f"+ wiped {wiped['unidades'].sum():,.0f} · ${wiped['usd'].sum():,.0f}")
 
-    # [6] the table
-    configuration.log_action(STEP_LABEL, 6, "writing the table")
-    configuration.write_table(STEP_LABEL, check_log, table, TABLE_NEW_ROWS)
-
-    # [7] the count of the checks
-    configuration.log_action(STEP_LABEL, 7, "counting the checks")
+    # [6] the count of the checks
+    configuration.log_action(STEP_LABEL, 6, "counting the checks")
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
 
-    # [8] the tables
-    configuration.log_action(STEP_LABEL, 8, "per origin (what each one is, and how much):")
+    # [7] the tables and their SQL
+    configuration.log_action(STEP_LABEL, 7, "per origin (what each one is, and how much):")
     configuration.show_table(totals)
     configuration.logger.doc(f"[{STEP_LABEL}] the months with wiped or created pipeline (the gaps are spread over the history):")
     configuration.show_table(table[table["origen"] != ORIGIN_GAP].reset_index(drop=True))
+    configuration.show_query(STEP_LABEL, "the rows the framework adds or wipes, month × origin", new_rows_query(configuration))
     return table
 
 
```

## step_informe.py

```diff
--- /tmp/before_queries/step_informe.py	2026-10-01 16:37:36.772334696 +0000
+++ step_informe.py	2026-10-01 16:38:35.611547011 +0000
@@ -227,8 +227,10 @@
              "estación, una media móvil). Todas sus medidas son 0 y su tasa queda **nula, nunca 0 %**: añade un mes, no dinero.",
              "- **resultado_adelantado_borrado** (paso 02): una renovación ya registrada desde el mes en curso; el mes no "
              "ha terminado, y el forecast la predice.",
-             "- **pipeline_borrada** (paso 02): la pipeline de una licencia de 1 año vendida o renovada desde el mes en "
-             "curso (vence 12 meses después): aún no se conoce.",
+             "- **pipeline_parcial_borrada** (paso 02): la pipeline de las licencias de 1 año que vencen 12 meses después "
+             "del mes en curso o más tarde. La generan las ventas y renovaciones desde el mes en curso, que solo han empezado "
+             "(el mes va por la mitad y los siguientes no han empezado): está a medio crear. Si se dejara, el forecast de "
+             "esos meses se calcularía sobre una pipeline a medias; el paso 17 la reconstruye entera.",
              "- **proyectada** y **simulada** (paso 17): las renovaciones esperadas de las licencias de 1 año que vencen en la "
              "ventana de simulación, y la captación simulada; vencen 12 meses después y sustituyen a la pipeline borrada.", ""]
     new_rows = results.get("new_rows")
```

## vocabulario.py

```diff
--- /tmp/before_queries/vocabulario.py	2026-10-01 16:37:36.560617764 +0000
+++ vocabulario.py	2026-10-01 16:38:35.611067709 +0000
@@ -195,7 +195,6 @@
 TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
 TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
 TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
-TABLE_NEW_ROWS = "filas_nuevas"                          # step 21: the rows added or wiped, month × origin
 TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
 TABLE_DIMENSION_LEVEL_VALUES = "dimension_level_values"  # step 01b: every value of every leveled dimension, its rate and group
 TABLE_DIMENSION_LEVELS = "dimension_levels"              # step 01b: the generated groups of every leveled dimension
```

## report_queries.py (nuevo)

Ver el fichero.
