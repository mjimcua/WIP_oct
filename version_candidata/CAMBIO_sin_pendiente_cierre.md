# Cambio · eliminar `pending_close_months` y el rol `pendiente_cierre`
Calendario de 3 roles (entrenamiento · examen · proyeccion); "cerrado" = antes de `current_month` en todo el framework.
Verificación: `grep -rn "pending\|pendiente_cierre\|ROLE_PENDING"` → 0 resultados; 17 ficheros de test en verde; con la configuración de main.py (sintético) las salidas son idénticas a las de antes del cambio: roles, núcleo, backtest, forecast, total y examen de cartera (misma forma y mismos valores). Única diferencia de forma: `sff_series` ya no tiene la columna `meses_pendiente_cierre` (siempre era 0).

## config.py

```diff
--- /mnt/user-data/outputs/sff_espejo/config.py	2026-09-30 12:09:52.593612000 +0000
+++ config.py	2026-09-30 12:25:28.241994352 +0000
@@ -48,7 +48,7 @@
 import pandas as pd
 
 from logging_helpers import DOC_LEVEL, LoggerManager
-from vocabulario import ROLE_PENDING, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN
+from vocabulario import ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN
 
 
 # ─── named constants ─────────────────────────────────────────────────────────────
@@ -205,9 +205,11 @@
     ignore_cols: list = field(default_factory=list)   # read by nobody; may be absent from the raw
 
     # ─── the calendar (the role of every month is generated from it) ────────────────
-    current_month: Optional[str] = None   # first month of the future; no default on purpose
+    current_month: Optional[str] = None   # first month of the future; no default on purpose. If the last months
+                                          # are not mature yet (late renewals still arriving), set current_month to the
+                                          # first immature one: it becomes proyeccion, is predicted, and its partial
+                                          # renewals are wiped in step 02. "Closed" always means: before current_month
     test_months: int = 3                  # closed months before it that only evaluate
-    pending_close_months: int = 0         # months before the exam not closed yet (0 = all closed)
 
     # ─── statistical parameters ────────────────────────────────────────────────────
     z: float = 1.645                      # 90 % two-sided: every band and every binomial error uses it
@@ -340,8 +342,6 @@
             raise ValueError(f"ts_acquisition_discount must be in [0, 1) (found {self.ts_acquisition_discount})")
         if int(self.test_months) < 1:
             raise ValueError(f"test_months must be at least 1 (found {self.test_months}): without an exam nothing is evaluated")
-        if int(self.pending_close_months) < 0:
-            raise ValueError(f"pending_close_months cannot be negative (found {self.pending_close_months})")
 
     @property
     def logger(self) -> logging.Logger:
@@ -568,35 +568,29 @@
     # ═══════════════════════════════════════════════════════════════════════════════
 
     def calendar_boundaries(self) -> dict:
-        """The three months that cut the calendar: current (first month of projection),
-        pending_start (first pending month; = current when there is none) and test_start
-        (first exam month). Raises ValueError when current_month is not declared."""
+        """The two months that cut the calendar: current (the first month of projection; every
+        month before it is closed) and test_start (the first exam month: current − test_months).
+        Raises ValueError when current_month is not declared."""
         if self.current_month is None:
             raise ValueError("current_month is not declared: set it in your Config "
                              "(e.g. current_month=\"2026-09\" or \"01/09/2026\")")
         current = parse_month(self.current_month)
-        pending_start = current - int(self.pending_close_months)
-        test_start = pending_start - int(self.test_months)
-        return dict(current=current, pending_start=pending_start, test_start=test_start)
+        test_start = current - int(self.test_months)
+        return dict(current=current, test_start=test_start)
 
     def role_of_months(self, periods) -> np.ndarray:
-        """The role of every month: ≥ current → proyeccion · ≥ pending_start →
-        pendiente_cierre · ≥ test_start → examen · earlier → entrenamiento."""
+        """The role of every month: ≥ current → proyeccion · ≥ test_start → examen ·
+        earlier → entrenamiento."""
         boundaries = self.calendar_boundaries()
         month_values = pd.Series(periods).reset_index(drop=True)
         # np.select takes the FIRST condition that holds, so the order is the rule
         conditions = [month_values >= boundaries["current"],
-                      month_values >= boundaries["pending_start"],
                       month_values >= boundaries["test_start"]]
-        return np.select(conditions, [ROLE_PROJECTION, ROLE_PENDING, ROLE_TEST], default=ROLE_TRAIN)
+        return np.select(conditions, [ROLE_PROJECTION, ROLE_TEST], default=ROLE_TRAIN)
 
     def calendar_description(self) -> str:
         """The calendar in one line, for the console."""
         boundaries = self.calendar_boundaries()
-        current, pending_start, test_start = boundaries["current"], boundaries["pending_start"], boundaries["test_start"]
-        if self.pending_close_months > 0:
-            pending_text = f"{ROLE_PENDING} {pending_start}..{current - 1}"
-        else:
-            pending_text = f"{ROLE_PENDING}: none"
-        return (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{pending_start - 1} · "
-                f"{pending_text} · {ROLE_PROJECTION} ≥ {current}")
+        current, test_start = boundaries["current"], boundaries["test_start"]
+        return (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{current - 1} · "
+                f"{ROLE_PROJECTION} ≥ {current}")
```

## vocabulario.py

```diff
--- /mnt/user-data/outputs/sff_espejo/vocabulario.py	2026-09-30 12:09:54.257743000 +0000
+++ vocabulario.py	2026-09-30 12:25:45.642965311 +0000
@@ -6,11 +6,10 @@
 # ─── the role of every month (generated from the calendar of the configuration) ──
 ROLE_TRAIN = "entrenamiento"          # closed months the forecast learns from
 ROLE_TEST = "examen"                  # closed months that only evaluate
-ROLE_PENDING = "pendiente_cierre"     # not closed yet: neither learn nor evaluate
 ROLE_PROJECTION = "proyeccion"        # the current month and the future
 
 # ─── the columns step 02 adds to the raw ─────────────────────────────────────────
-CALENDAR_ROLE_COLUMN = "rol"                          # the role of the row's month (the four above)
+CALENDAR_ROLE_COLUMN = "rol"                          # the role of the row's month (the three above)
 CURRENT_MONTH_COLUMN = "es_mes_en_curso"              # 1 in the current month, 0 elsewhere
 S0_RENEWED_UNITS_COLUMN = "s0_renovados_unidades"     # the raw's renewed units, before step 02 touches them
 S0_RENEWED_USD_COLUMN = "s0_renovados_usd"            # the raw's renewed USD, before step 02 touches them
@@ -31,7 +30,7 @@
 TABLE_FINE = "fact_fine"                     # step 03: every raw row with its ids and keys
 
 # The four roles in time order (coverage patterns and tables follow it).
-ROLES_IN_ORDER = [ROLE_TRAIN, ROLE_TEST, ROLE_PENDING, ROLE_PROJECTION]
+ROLES_IN_ORDER = [ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION]
 
 # ─── the columns steps 04 and 06 add ─────────────────────────────────────────────
 FINE_ROWS_COLUMN = "n_filas_finas"           # step 04: fine rows added into a forecast unit
```

## main.py

```diff
--- /mnt/user-data/outputs/sff_espejo/main.py	2026-09-30 12:09:52.568130000 +0000
+++ main.py	2026-09-30 12:25:45.643329644 +0000
@@ -87,7 +87,6 @@
         # the calendar
         current_month="01/09/2026",
         test_months=3,
-        pending_close_months=0,
         # the dimensions
         business_mandatory_dims=["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
                                  "tr_product_level_1", "tr_product_level_2", "tr_purchase_type", "tr_renewal_type",
@@ -122,7 +121,6 @@
         sql_engine=create_engine(f"sqlite:///{SYNTHETIC_DATABASE}"),
         current_month="2026-09",
         test_months=3,
-        pending_close_months=0,
         business_mandatory_dims=["region", "product"],
         structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                      "no_instalado": "negative", "autorenew": "positive"},
```

## step_00_validate_raw.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_00_validate_raw.py	2026-09-30 12:09:52.705784000 +0000
+++ step_00_validate_raw.py	2026-09-30 12:25:45.643592561 +0000
@@ -154,7 +154,7 @@
     configuration.log_check(STEP_LABEL, check_log, "the calendar leaves months to train",
                             boundaries["test_start"] > first_month,
                             failure_detail=f"the exam starts at {boundaries['test_start']} and the raw starts at "
-                                           f"{first_month}: no month to train (lower test_months or pending_close_months)",
+                                           f"{first_month}: no month to train (lower test_months)",
                             context=f"training {first_month}..{boundaries['test_start'] - 1}")
 
     # [10] a month with no row at all is worth a look, not a stop
```

## step_01_validate_values.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_01_validate_values.py	2026-09-30 12:09:52.717807000 +0000
+++ step_01_validate_values.py	2026-09-30 12:25:45.644081789 +0000
@@ -10,7 +10,8 @@
     they are early results that step 02 wipes, so they are not checked.
 
 Actions (logged as they are done):
-  1. split the closed months (before the current one) from the rest
+  1. scope the renewal checks: renewals are checked only in months before the current one
+     (from the current month on they are partial and step 02 wipes them)
   2. check the money: pipeline, every month; renewals, closed months        checks 1-10
   3. check the dimensions and the flags                                     checks 11-16
   4. check the exact discount, if declared                                  check 17
@@ -47,7 +48,8 @@
 STEP_NAME = "VALIDATE VALUES"
 STEP_PURPOSE = ("check that the values inside the raw can be computed with: money without nulls or negatives, "
                 "renewals coherent with what fell due, dimensions never empty, flags 0 / 1, discount as a share")
-STEP_ACTIONS = ["split the closed months (before the current one) from the rest",
+STEP_ACTIONS = ["scope the renewal checks: renewals are checked only in months before the current one "
+                "(from the current month on they are partial and step 02 wipes them)",
                 "check the money: pipeline every month, renewals in closed months (checks 1-10)",
                 "check the dimensions and the flags (checks 11-16)",
                 "check the exact discount, if declared (check 17)",
@@ -66,13 +68,16 @@
     configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
     check_log = []
 
-    # [1] the closed months: renewals are only checked there
+    # [1] the scope of the renewal checks: the months before the current one. From the current
+    #     month on the renewals are partial (the month is still running) and step 02 wipes them,
+    #     so only the pipeline is checked there
     current_month = configuration.calendar_boundaries()["current"]
     closed_rows = validated[validated[configuration.period_col] < current_month]
     renewed_units = closed_rows[configuration.renewed_units_col]
     renewed_usd = closed_rows[configuration.renewed_usd_col]
-    configuration.log_action(STEP_LABEL, 1, f"{len(closed_rows):,} rows in closed months (before {current_month}) · "
-                                            f"{len(validated) - len(closed_rows):,} rows from {current_month} on")
+    configuration.log_action(STEP_LABEL, 1, f"renewals checked in {len(closed_rows):,} rows before {current_month} · "
+                                            f"{len(validated) - len(closed_rows):,} rows from {current_month} on: pipeline only, "
+                                            f"renewals wiped in step 02")
 
     # [2] the money
     configuration.log_action(STEP_LABEL, 2, "checking the money")
```

## step_02_apply_calendar.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_02_apply_calendar.py	2026-09-30 12:09:56.894364000 +0000
+++ step_02_apply_calendar.py	2026-09-30 12:25:45.644553795 +0000
@@ -3,7 +3,7 @@
 
 Steps 00 and 01 only looked at the raw. This is the first step that changes it:
   · every row gets its role in time, generated from the calendar of the Config
-    (entrenamiento · examen · pendiente_cierre · proyeccion) and the mark of the
+    (entrenamiento · examen · proyeccion) and the mark of the
     current month. The raw's own role columns, if any, are ignored.
   · the raw's renewals are kept, untouched, in two s0_ columns: what the raw said
     before any change (the core table reconciles against them)
@@ -49,7 +49,7 @@
 import pandas as pd
 
 from config import Config, is_one_year
-from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PENDING, ROLE_PROJECTION,
+from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                          ROLE_TEST, ROLE_TRAIN, ROLES_IN_ORDER, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                          TABLE_CALENDAR, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN)
 
@@ -200,7 +200,7 @@
                             context=f"{training_months} months", blocking=False)
 
     # [3] every exam month has rows (a month with no rows is a month the exam cannot score)
-    exam_months = pd.period_range(boundaries["test_start"], boundaries["pending_start"] - 1, freq="M")
+    exam_months = pd.period_range(boundaries["test_start"], boundaries["current"] - 1, freq="M")
     exam_months_without_rows = [str(month) for month in exam_months if month not in set(months_per_role[ROLE_TEST])]
     configuration.log_check(STEP_LABEL, check_log, "every exam month has rows", not exam_months_without_rows,
                             failure_detail=f"exam months with no row: {exam_months_without_rows}",
```

## step_06_series_routes.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_06_series_routes.py	2026-09-30 12:09:52.913779000 +0000
+++ step_06_series_routes.py	2026-09-30 12:27:07.723969598 +0000
@@ -1,13 +1,12 @@
 """
 step_06_series_routes.py — The series: which months each one has, and how it will be treated.
 
-A rate series is not forecast the same way depending on the months it has:
+A rate series is not forecast the same way: it depends on the months it has:
   · COVERAGE: the roles of its months, in time order (e.g. entrenamiento+examen+proyeccion)
   · ROUTE, from the coverage:
       predecible     closed months AND something to predict: its rate is estimated
       solo_historia  nothing to predict: kept, it lends its history to its relatives
       solo_futuro    nothing to learn from: predicted from its relatives
-    (a pending month alone is not history, it is not closed yet, but it is a month to predict)
   · UNIVERSE, from the time_series flag of its units: normal · serie_temporal · mixto
 Nothing is filtered: every series is labelled, none is dropped.
 
@@ -15,7 +14,7 @@
   1. group the forecast units by series
   2. count the months of each series in each role; first and last month
   3. coverage, route and universe of every series
-  4. the money each series has to predict (due from the pending months on)
+  4. the money each series has to predict (due from the current month on)
   5. check the series                                               checks 1-3
   6. write the series table                                         check 4
   7. count the checks; stop if any failed
@@ -35,7 +34,7 @@
 import pandas as pd
 
 from config import ACTIVE_FLAG_VALUES, Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ROLE_PENDING, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN,
+from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN,
                          ROLES_IN_ORDER, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, ROUTE_PREDICTABLE,
                          SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_SERIES, UNIT_ID_COLUMN, UNIVERSE_COLUMN,
                          UNIVERSE_MIXED, UNIVERSE_NORMAL, UNIVERSE_TIME_SERIES)
@@ -49,7 +48,7 @@
 STEP_ACTIONS = ["group the forecast units by series",
                 "count the months of each series in each role; first and last month",
                 "coverage, route and universe of every series",
-                "the money each series has to predict (due from the pending months on)",
+                "the money each series has to predict (due from the current month on)",
                 "check the series (checks 1-3)",
                 "write the series table (check 4)",
                 "count the checks; stop if any failed",
@@ -59,8 +58,7 @@
 # ─── named constants ─────────────────────────────────────────────────────────────
 COVERAGE_SEPARATOR = "+"
 ROUTES_IN_ORDER = [ROUTE_PREDICTABLE, ROUTE_HISTORY_ONLY, ROUTE_FUTURE_ONLY]
-MONTH_COUNT_COLUMN = {ROLE_TRAIN: "meses_entrenamiento", ROLE_TEST: "meses_examen",
-                      ROLE_PENDING: "meses_pendiente_cierre", ROLE_PROJECTION: "meses_proyeccion"}
+MONTH_COUNT_COLUMN = {ROLE_TRAIN: "meses_entrenamiento", ROLE_TEST: "meses_examen", ROLE_PROJECTION: "meses_proyeccion"}
 EXAMPLE_ROWS_SHOWN = 3
 
 
@@ -68,7 +66,7 @@
     """The route of a series from the roles it has: nothing to predict → solo_historia;
     something to predict and a closed month (training or exam) → predecible; something
     to predict and no closed month → solo_futuro."""
-    has_something_to_predict = ROLE_PROJECTION in roles_present or ROLE_PENDING in roles_present
+    has_something_to_predict = ROLE_PROJECTION in roles_present
     has_closed_months = ROLE_TRAIN in roles_present or ROLE_TEST in roles_present
     if not has_something_to_predict:
         return ROUTE_HISTORY_ONLY
@@ -114,8 +112,8 @@
     configuration.log_action(STEP_LABEL, 3, f"routes: {route_census} · universes: "
                                             f"{series_table[UNIVERSE_COLUMN].value_counts().to_dict()}")
 
-    # [4] what each series has to predict: what falls due from the pending months on
-    to_predict = forecast_units[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION])
+    # [4] what each series has to predict: what falls due from the current month on
+    to_predict = forecast_units[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
     predicted_units = forecast_units[to_predict].groupby(SERIES_ID_COLUMN)
     series_table["unidades_por_predecir"] = predicted_units[configuration.pipeline_units_col].sum()
     series_table["usd_por_predecir"] = predicted_units[configuration.pipeline_usd_col].sum()
```

## step_08_rate_series.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_08_rate_series.py	2026-09-30 12:09:52.938966000 +0000
+++ step_08_rate_series.py	2026-09-30 12:26:09.583537665 +0000
@@ -120,7 +120,7 @@
                                         rated_units[configuration.renewed_units_col] / pipeline_units.where(pipeline_units > 0),
                                         np.nan)
     configuration.log_action(STEP_LABEL, 3, f"rate computed in {int(rate_is_truth.sum()):,} units; null in the other "
-                                            f"{int((~rate_is_truth).sum()):,} (future, pending, gaps, nothing due)")
+                                            f"{int((~rate_is_truth).sum()):,} (future, gaps, nothing due)")
 
     # [4] the summary of every series
     series_rate = summarise_series(rated_units, series_table, configuration)
```

## step_14_backtest.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_14_backtest.py	2026-09-30 12:09:53.290783000 +0000
+++ step_14_backtest.py	2026-09-30 12:26:09.583938393 +0000
@@ -169,7 +169,7 @@
 def test_calendar(pool_series: pd.DataFrame, configuration: Config) -> tuple:
     """The selection months (the N closed months before the exam) and the exam months."""
     boundaries = configuration.calendar_boundaries()
-    exam_months = pd.period_range(boundaries["test_start"], boundaries["pending_start"] - 1, freq="M")
+    exam_months = pd.period_range(boundaries["test_start"], boundaries["current"] - 1, freq="M")
     selection_months = pd.period_range(boundaries["test_start"] - configuration.backtest_selection_months,
                                        boundaries["test_start"] - 1, freq="M")
     return list(selection_months), list(exam_months)
```

## step_17_forecast.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_17_forecast.py	2026-09-30 12:09:57.100164000 +0000
+++ step_17_forecast.py	2026-09-30 12:26:09.584559354 +0000
@@ -81,7 +81,7 @@
 from techniques import inverse_logit, logit, predict_logit
 from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, PIPELINE_ORIGIN_COLUMN,
                          PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_FROM_CELL,
-                         RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PENDING, ROLE_PROJECTION, SERIES_ID_COLUMN,
+                         RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
                          TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNIT_ID_COLUMN,
                          UPLIFT_CELL_ID_COLUMN)
 
@@ -117,13 +117,13 @@
     check_log = []
     period_column = configuration.period_col
     boundaries = configuration.calendar_boundaries()
-    last_closed = boundaries["pending_start"] - 1
+    last_closed = boundaries["current"] - 1
     context = dict(series_estimate=series_estimate, pool_series=pool_series, pool_reference=pool_reference, backtest=backtest,
                    uplift_cells=uplift_cells, uplift_verdict=uplift_verdict, rated_units=rated_units,
                    forecast_units=forecast_units, last_closed=last_closed, pool_rate_cache={})
 
     # [1] the future rows of the extract
-    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION])].copy()
+    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION].copy()
     future["_fila"] = future.index
     future[PIPELINE_ORIGIN_COLUMN] = PIPELINE_REAL
     configuration.log_action(STEP_LABEL, 1, f"{len(future):,} future rows in the extract, {future[period_column].min()}.."
@@ -513,7 +513,7 @@
     configuration.log_check(STEP_LABEL, check_log, "every rate and band is inside [0, 1] and the band contains the rate",
                             not bad_band.any(), failure_detail=f"{int(bad_band.sum()):,} rows with a bad rate or band",
                             examples=future.loc[bad_band, ["tasa", "tasa_baja", "tasa_alta"]])
-    future_roles = fine_table[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION])
+    future_roles = fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
     configuration.log_check(STEP_LABEL, check_log, "no closed row is forecast; every future row of the extract is",
                             set(all_future.loc[all_future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, "_fila"])
                             == set(fine_table.index[future_roles]),
```

## step_18_validation.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_18_validation.py	2026-09-30 12:09:58.231314000 +0000
+++ step_18_validation.py	2026-09-30 12:26:09.584845529 +0000
@@ -37,7 +37,7 @@
 import pandas as pd
 
 from config import Config
-from vocabulario import (CALENDAR_ROLE_COLUMN, RATE_FROM_POOL, ROLE_PENDING, ROLE_PROJECTION, TABLE_VALIDATION,
+from vocabulario import (CALENDAR_ROLE_COLUMN, RATE_FROM_POOL, ROLE_PROJECTION, TABLE_VALIDATION,
                          TRUTH_ROLES)
 
 
@@ -73,7 +73,7 @@
     configuration.log_check(STEP_LABEL, check_log, "Σ USD due: extract (without time_series and the pipeline not known yet) = fine table = forecast units",
                             max(totals.values()) - min(totals.values()) <= MONEY_TOLERANCE,
                             failure_detail=f"totals differ: {totals}", context=f"${totals['extracto']:,.0f}")
-    future_due = fine.loc[fine[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION]), due].sum()
+    future_due = fine.loc[fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION, due].sum()
     configuration.log_check(STEP_LABEL, check_log, "the future USD due of the extract = Σ USD due of the forecast's extract rows",
                             abs(future_due - forecast.loc[forecast["origen_pipeline"] == "real", due].sum()) <= MONEY_TOLERANCE,
                             failure_detail=f"${future_due:,.0f} vs ${forecast.loc[forecast['origen_pipeline'] == 'real', due].sum():,.0f}",
```

## step_informe.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_informe.py	2026-09-30 12:09:53.493818000 +0000
+++ step_informe.py	2026-09-30 12:26:09.585242770 +0000
@@ -353,7 +353,7 @@
              "**Cómo se mide.** Cada pool con soporte se predice en meses que ya ocurrieron, sin mirar el futuro: para el mes T "
              "a horizonte h, cada técnica solo ve hasta T − h. Los meses de **selección** (los "
              f"{configuration.backtest_selection_months} anteriores al examen) eligen la técnica; los meses de **examen** "
-             f"({boundaries['test_start']} a {boundaries['pending_start'] - 1}) la miden sin que la haya visto. El error se "
+             f"({boundaries['test_start']} a {boundaries['current'] - 1}) la miden sin que la haya visto. El error se "
              "compara con el ruido binomial del mes (err_norm ≈ 1: tan cerca como permite el azar). Una técnica sustituye al "
              f"retador ({configuration.challenger_technique}) solo si le gana por un margen. Compiten todas las técnicas que "
              "la historia permite, también las de series temporales.", "",
```

## step_nucleo.py

```diff
--- /mnt/user-data/outputs/sff_espejo/step_nucleo.py	2026-09-30 12:09:57.390760000 +0000
+++ step_nucleo.py	2026-09-30 12:26:09.585699646 +0000
@@ -480,7 +480,7 @@
                             LEVEL_ROW, AGGREGATE_SLICER, description))
     for name, step, description in RAW_MEASURES:
         legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
-    legend_rows += [("s02_rol", "02", LEVEL_ROW, AGGREGATE_SLICER, "entrenamiento · examen · pendiente_cierre · proyeccion"),
+    legend_rows += [("s02_rol", "02", LEVEL_ROW, AGGREGATE_SLICER, "entrenamiento · examen · proyeccion"),
                     ("s02_es_mes_en_curso", "02", LEVEL_ROW, AGGREGATE_SLICER, "1 in the current month")]
     for name, step, description in CALENDAR_MEASURES:
         legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
```

## test_step_02.py

```diff
--- /mnt/user-data/outputs/sff_espejo/test_step_02.py	2026-09-30 12:09:53.647731000 +0000
+++ test_step_02.py	2026-09-30 12:26:21.241627943 +0000
@@ -13,7 +13,7 @@
 from step_00_validate_raw import validate_raw
 from step_02_apply_calendar import apply_calendar
 from test_helpers import check, console_of, count_status, finish, synthetic_with
-from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PENDING, ROLE_PROJECTION,
+from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                          ROLE_TEST, ROLE_TRAIN, S0_RENEWED_UNITS_COLUMN)
 
 
@@ -41,11 +41,12 @@
     check(role_of_month["2026-05"] == ROLE_TRAIN and role_of_month["2026-06"] == ROLE_TEST
           and role_of_month["2026-08"] == ROLE_TEST and role_of_month["2026-09"] == ROLE_PROJECTION,
           "entrenamiento ≤ 2026-05 · examen 2026-06..08 · proyeccion ≥ 2026-09")
-    check(ROLE_PENDING not in set(calendared[CALENDAR_ROLE_COLUMN]), "no pending month (pending_close_months = 0)")
+    check(set(calendared[CALENDAR_ROLE_COLUMN]) == {ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION},
+          "three roles: entrenamiento · examen · proyeccion")
     current_rows = calendared[CURRENT_MONTH_COLUMN] == 1
     check(set(calendared.loc[current_rows, "period"].astype(str)) == {"2026-09"}, "only 2026-09 is the current month")
     check(len(calendared) == len(raw) and list(calendared.columns[:len(raw.columns)]) == list(raw.columns),
-          "no row lost, the raw's columns first and in order, four columns added")
+          "no row lost, the raw's columns first and in order")
     check(count_status(console, "ok") == 10 and "10 checks: 10 ok" in console, "the 10 checks pass and are logged")
     calendar_table = pd.read_sql("SELECT * FROM sff_calendario", configuration.sql_engine)
     check(len(calendar_table) == 48 and calendar_table.loc[calendar_table["period"] == "2026-09", "rol"].item() == ROLE_PROJECTION
@@ -57,9 +58,12 @@
     check("actions:" in console and "output:" in console and "▸ 4." in console,
           "the step logs its actions and its output, and each action when it is done")
 
-    with_pending, _ = run_step(*validated_synthetic(pending_close_months=1))
-    pending_months = set(with_pending.loc[with_pending[CALENDAR_ROLE_COLUMN] == ROLE_PENDING, "period"].astype(str))
-    check(pending_months == {"2026-08"}, "with one pending month, 2026-08 is pendiente_cierre")
+    # an immature month (late renewals still arriving) is handled by setting current_month on it
+    immature, _ = run_step(*validated_synthetic(current_month="2026-08"))
+    august = immature[immature["period"].astype(str) == "2026-08"]
+    check((august[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION).all() and august["total_renewed_units"].isna().all()
+          and (august["s0_renovados_unidades"].fillna(0) > 0).any(),
+          "current_month on the first immature month (2026-08): it is proyeccion and its partial renewals are wiped")
 
 
 def test_the_renewals() -> None:
```

## test_step_06.py

```diff
--- /mnt/user-data/outputs/sff_espejo/test_step_06.py	2026-09-30 12:09:53.868298000 +0000
+++ test_step_06.py	2026-09-30 12:26:09.585916414 +0000
@@ -25,8 +25,7 @@
     check(route_from_coverage({"entrenamiento", "examen", "proyeccion"}) == "predecible"
           and route_from_coverage({"entrenamiento"}) == "solo_historia"
           and route_from_coverage({"proyeccion"}) == "solo_futuro"
-          and route_from_coverage({"pendiente_cierre"}) == "solo_futuro"
-          and route_from_coverage({"examen", "pendiente_cierre"}) == "predecible",
+          and route_from_coverage({"examen", "proyeccion"}) == "predecible",
           "the route rule: history + future → predecible · only history → solo_historia · only future → solo_futuro")
     _, units, configuration = units_synthetic()
     series, console = run_step(units, configuration)
@@ -36,8 +35,7 @@
     only_history = series[series["ruta"] == "solo_historia"].iloc[0]
     check(only_history["cobertura"] == "entrenamiento" and only_history["usd_por_predecir"] == 0,
           "the solo_historia series covers only training and has nothing to predict")
-    check((series["meses_entrenamiento"] + series["meses_examen"] + series["meses_pendiente_cierre"]
-           + series["meses_proyeccion"] == series["meses"]).all(), "the months per role add up to the months of the series")
+    check((series["meses_entrenamiento"] + series["meses_examen"] + series["meses_proyeccion"] == series["meses"]).all(), "the months per role add up to the months of the series")
     check(abs(series["usd_por_predecir"].sum() - units.loc[units["rol"] == "proyeccion", "total_tr_usd"].sum()) < 0.01,
           "the USD to predict is what falls due in the projection")
     check(count_status(console, "ok") == 4 and "4 checks: 4 ok" in console, "the 4 checks pass and are logged")
```
