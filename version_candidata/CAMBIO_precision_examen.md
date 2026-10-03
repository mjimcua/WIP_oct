# Cambio · la precisión del examen por forecast serie, con su intervalo, en el núcleo
Verificación: 18 ficheros de test en verde (test_step_audit 19/19), 20 comprobaciones del paso AUD y 7 del núcleo en verde; las cifras del forecast idénticas a las de antes.

## main.py

```diff
--- antes/main.py	2026-10-01 06:30:20.257790152 +0000
+++ main.py	2026-10-01 06:30:48.800629260 +0000
@@ -185,14 +185,16 @@
     results["time_series"], results["forecast_total"] = build_time_series_and_total(
         time_series_rows, fine_table, forecast["forecast"], configuration)                      # step 20
     results["time_series_rows"] = time_series_rows
+    results["audit"] = build_audit_tables(ladder, series_rate, series_estimate, rated_units, pool_series, pool_reference,
+                                          pool_dynamics, backtest, forecast["forecast"], configuration)   # the satellites
     results["core"], results["core_legend"] = build_core_table(
         fine_table, configuration, forecast_units=forecast_units, support_bound=support_bound, rated_units=rated_units,
         series_table=series_table, series_rate=series_rate, series_estimate=series_estimate, pool_dynamics=pool_dynamics,
         technique_decision=backtest["decision"], exam_by_pool=backtest["exam_by_pool"], forecast=forecast["forecast"],
         time_series_rows=time_series_rows, time_series_table=results["time_series"],
-        forecast_total=results["forecast_total"], ladder_stages=ladder["stages"])
-    results["audit"] = build_audit_tables(ladder, series_rate, series_estimate, rated_units, pool_series, pool_reference,
-                                          pool_dynamics, backtest, forecast["forecast"], configuration)   # the satellites                                                  # the core: every row, every decision
+        forecast_total=results["forecast_total"], ladder_stages=ladder["stages"],
+        series_dynamics=results["audit"]["series_dynamics"], series_exam=results["audit"]["series_exam"])
+                                                 # the core: every row, every decision
     results["validation"] = validate_chain(raw, results, configuration)                            # step 18 (after 19: it reads its exam)
     results["card"] = build_report(raw, results, configuration)                                   # the report, last
     return results
```

## step_audit.py

```diff
--- antes/step_audit.py	2026-10-01 06:30:20.402956561 +0000
+++ step_audit.py	2026-10-01 06:30:59.430945558 +0000
@@ -13,6 +13,13 @@
   sff_series_backtest            forecast series × exam month × h × technique s03_fs_id
   sff_series_technique_summary   forecast series × band × technique           s03_fs_id
   sff_composition_forecast_all   composition × future horizon × technique     s10_stage3_id
+  sff_series_exam                forecast series (the chosen technique)       s03_fs_id
+
+EVERY TEST HAS ITS INTERVAL, built as the forecast builds its band: the error quantiles of the
+technique in the backtest (step 14) × the binomial error with the series' own units due. A test
+is IN THE INTERVAL when the real rate of the series falls inside it. The exam of every forecast
+series (its chosen technique, every exam month and horizon) is summarised in sff_series_exam and
+in the core: how much it erred and how many predictions were in the interval.
 
 The 4 stages only give the forecast better conditions to predict: the backtest, the exam and the
 forecast are referred back to every forecast series (its own units due, its own renewals).
@@ -28,8 +35,9 @@
   4. sff_series_dynamics                                              check 5
   5. sff_series_backtest and its summary                              checks 6-8
   6. sff_composition_forecast_all                                     check 9
-  7. write the audit tables                                           checks 10-17
-  8. count the checks; stop if any failed
+  7. sff_series_exam: the exam of the chosen technique per series     checks 10-11
+  8. write the audit tables                                           checks 12-20
+  9. count the checks; stop if any failed
 
 Checks (logged as they are made, numbered, at the level of their status):
    1. every stage is a partition: the units due of its ids add up to the raw's
@@ -41,7 +49,9 @@
    7. without credibility, the series' predictions add up to the composition's prediction
    8. every forecast series and band has exactly one chosen technique in its summary
    9. the chosen technique gives the rate step 17 used (rows with no credibility shift)
-   10-17. the eight tables written and read back
+   10. the exam of every series adds up to its chosen predictions (counts and units)
+   11. the predictions in the interval reach 80 % in the exam                      (warning only)
+   12-20. the nine tables written and read back
 
 Output: dict of the audit tables · the tables listed above.
 """
@@ -58,7 +68,7 @@
 from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM, PURPOSE_SELECTION,
                          RATE_COLUMN, RATE_FROM_POOL, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_COMPOSITION,
                          TABLE_COMPOSITION_FORECAST_ALL, TABLE_COMPOSITION_TECHNIQUES, TABLE_CREDIBILITY,
-                         TABLE_CREDIBILITY_MEMBERS, TABLE_SERIES_BACKTEST, TABLE_SERIES_DYNAMICS,
+                         TABLE_CREDIBILITY_MEMBERS, TABLE_SERIES_BACKTEST, TABLE_SERIES_DYNAMICS, TABLE_SERIES_EXAM,
                          TABLE_SERIES_TECHNIQUE_SUMMARY, TECHNIQUE_COMPOSITION_BELOW_FLOOR,
                          TECHNIQUE_NOT_ENOUGH_HISTORY, TECHNIQUE_TESTED, TRUTH_ROLES)
 
@@ -75,15 +85,17 @@
                 "sff_series_dynamics (check 5)",
                 "sff_series_backtest and its summary (checks 6-8)",
                 "sff_composition_forecast_all (check 9)",
-                "write the audit tables (checks 10-17)",
+                "sff_series_exam: the exam of the chosen technique per series (checks 10-11)",
+                "write the audit tables (checks 12-20)",
                 "count the checks; stop if any failed"]
-STEP_OUTPUT = "eight audit tables joined to the core by fs_id, the stage ids and the credibility reference"
+STEP_OUTPUT = "nine audit tables joined to the core by fs_id, the stage ids and the credibility reference"
 
 UNITS_TOLERANCE = 1e-6
 RATE_TOLERANCE = 1e-9
 PERCENTAGE_POINTS = 100
 TREND_MIN_MONTHS = 12          # a trend is measured with at least a year of history
 SEASONALITY_MIN_MONTHS = 24    # a month effect with at least two of every calendar month
+COVERAGE_WARNING = 0.80        # below this share of tests in their interval, the interval promises more than it gives
 STAGES = (0, 1, 2, 3)
 
 
@@ -138,8 +150,8 @@
                             failure_detail=f"{len(series_dynamics):,} rows for {len(groups):,} series")
 
     # [5] the backtest referred back to every forecast series
-    series_backtest = series_backtest_table(backtest["predictions"], backtest["decision"], groups, series_estimate,
-                                            ladder["reference_members"], pool_series, history, configuration)
+    series_backtest = series_backtest_table(backtest["predictions"], backtest["decision"], backtest["bands"], groups,
+                                            series_estimate, ladder["reference_members"], pool_series, history, configuration)
     summary = series_technique_summary(series_backtest)
     configuration.log_action(STEP_LABEL, 5, f"{len(series_backtest):,} forecast series × exam month × h × technique rows · "
                                             f"{len(summary):,} series × band × technique")
@@ -150,20 +162,29 @@
     configuration.log_action(STEP_LABEL, 6, f"{len(forecast_all):,} composition × horizon × technique rates")
     check_forecast_all(forecast_all, forecast_rows, series_estimate, configuration, check_log)
 
-    # [7] the tables
-    configuration.log_action(STEP_LABEL, 7, "writing the audit tables")
+    # [7] the exam of the chosen technique, per forecast series
+    series_exam = series_exam_table(series_backtest, series_rate, groups, pool_reference)
+    tested = series_exam[series_exam["exam_status"] == TECHNIQUE_TESTED]
+    configuration.log_action(STEP_LABEL, 7, f"exam per forecast series: {series_exam['exam_status'].value_counts().to_dict()} · "
+                                            f"{int(tested['exam_in_band'].sum()):,} of {int(tested['exam_predictions'].sum()):,} "
+                                            f"predictions in their interval · WAPE "
+                                            f"{tested['exam_abs_err_units'].sum() / max(tested['exam_real_units'].sum(), 1e-9):.1%}")
+    check_series_exam(series_exam, series_backtest, configuration, check_log)
+
+    # [8] the tables
+    configuration.log_action(STEP_LABEL, 8, "writing the audit tables")
     tables = dict(composition=composition, composition_techniques=techniques, credibility=credibility,
                   credibility_members=members, series_dynamics=series_dynamics, series_backtest=series_backtest,
-                  series_technique_summary=summary, composition_forecast_all=forecast_all)
+                  series_technique_summary=summary, composition_forecast_all=forecast_all, series_exam=series_exam)
     for name, table_name in (("composition", TABLE_COMPOSITION), ("composition_techniques", TABLE_COMPOSITION_TECHNIQUES),
                              ("credibility", TABLE_CREDIBILITY), ("credibility_members", TABLE_CREDIBILITY_MEMBERS),
                              ("series_dynamics", TABLE_SERIES_DYNAMICS), ("series_backtest", TABLE_SERIES_BACKTEST),
                              ("series_technique_summary", TABLE_SERIES_TECHNIQUE_SUMMARY),
-                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL)):
+                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL), ("series_exam", TABLE_SERIES_EXAM)):
         configuration.write_table(STEP_LABEL, check_log, tables[name], table_name)
 
-    # [8] the count of the checks; stop if anything failed
-    configuration.log_action(STEP_LABEL, 8, "counting the checks")
+    # [9] the count of the checks; stop if anything failed
+    configuration.log_action(STEP_LABEL, 9, "counting the checks")
     configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)
     return tables
 
@@ -395,7 +416,7 @@
 # 5 · THE BACKTEST REFERRED BACK TO EVERY FORECAST SERIES
 # ═══════════════════════════════════════════════════════════════════════════════════
 
-def series_backtest_table(predictions: pd.DataFrame, decision: pd.DataFrame, groups: pd.DataFrame,
+def series_backtest_table(predictions: pd.DataFrame, decision: pd.DataFrame, bands: pd.DataFrame, groups: pd.DataFrame,
                           series_estimate: pd.DataFrame, reference_members: pd.DataFrame, pool_series: pd.DataFrame,
                           history: pd.DataFrame, configuration: Config) -> pd.DataFrame:
     """Every exam prediction of a composition applied to each of its forecast series: the composition's
@@ -432,13 +453,25 @@
     rows["err_pp"] = PERCENTAGE_POINTS * (rows["pred_rate"] - rows["real_rate"])
     rows["se_binom_pp"] = PERCENTAGE_POINTS * np.sqrt(rows["pred_rate"] * (1 - rows["pred_rate"]) / rows["due_units"])
     rows["err_norm"] = rows["err_pp"] / rows["se_binom_pp"].where(rows["se_binom_pp"] > 0)
+    # the interval of the test, as the forecast builds its band: the error quantiles of the technique at that
+    # horizon (step 14) × the binomial error of the rate with the series' own units due
+    quantiles = bands.set_index(["tecnica", "h"])[["q_low_norm", "q_high_norm"]]
+    rows = rows.join(quantiles, on=["tecnica", "h"])
+    se_rate = np.sqrt(rows["pred_rate"] * (1 - rows["pred_rate"]) / rows["due_units"].clip(lower=1))
+    rows["band_low_rate"] = np.clip(rows["pred_rate"] + np.minimum(rows["q_low_norm"], 0) * se_rate, 0, 1)
+    rows["band_high_rate"] = np.clip(rows["pred_rate"] + np.maximum(rows["q_high_norm"], 0) * se_rate, 0, 1)
+    rows["band_low_units"] = rows["band_low_rate"] * rows["due_units"]
+    rows["band_high_units"] = rows["band_high_rate"] * rows["due_units"]
+    rows["in_band"] = ((rows["real_rate"] >= rows["band_low_rate"] - 1e-12)
+                       & (rows["real_rate"] <= rows["band_high_rate"] + 1e-12)).astype(int)
     chosen = decision.set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
     rows["is_chosen"] = (pd.Series(list(zip(rows[COMPOSITION_ID_COLUMN], rows["tramo_h"])), index=rows.index).map(chosen)
                          == rows["tecnica"]).astype(int)
     rows["shifted_by_credibility"] = (np.abs(shift) > 0).astype(int)
     return rows[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "mes_objetivo", "h", "tramo_h", "origin", "tecnica", "is_chosen",
                  "shifted_by_credibility", "tasa_pred", "pred_rate", "real_rate", "due_units", "pred_units", "real_units",
-                 "err_units", "err_pp", "se_binom_pp", "err_norm"]]
+                 "err_units", "err_pp", "se_binom_pp", "err_norm", "band_low_rate", "band_high_rate", "band_low_units",
+                 "band_high_units", "in_band"]]
 
 
 def levels_at_origins(history: pd.DataFrame, key: str, wanted: pd.DataFrame, configuration: Config) -> pd.DataFrame:
@@ -497,6 +530,47 @@
                             context=f"{len(per_series_band):,} series × band")
 
 
+def series_exam_table(series_backtest: pd.DataFrame, series_rate: pd.DataFrame, groups: pd.DataFrame,
+                      pool_reference: pd.DataFrame) -> pd.DataFrame:
+    """Per forecast series, the exam of its CHOSEN technique (every exam month and horizon): how many
+    predictions, how many in their interval, units predicted and real, the error; and, when it has no
+    exam, why (exam_status)."""
+    chosen = series_backtest[series_backtest["is_chosen"] == 1].assign(abs_err_units=lambda frame: frame["err_units"].abs(),
+                                                                         abs_err_pp=lambda frame: frame["err_pp"].abs())
+    grouped = chosen.groupby(SERIES_ID_COLUMN)
+    exam = pd.DataFrame({"exam_predictions": grouped.size(), "exam_in_band": grouped["in_band"].sum(),
+                         "exam_pred_units": grouped["pred_units"].sum(), "exam_real_units": grouped["real_units"].sum(),
+                         "exam_abs_err_units": grouped["abs_err_units"].sum(), "exam_mae_pp": grouped["abs_err_pp"].mean(),
+                         "exam_bias_pp": grouped["err_pp"].mean()})
+    exam["exam_wape"] = exam["exam_abs_err_units"] / exam["exam_real_units"].where(exam["exam_real_units"] > 0)
+    exam["exam_coverage"] = exam["exam_in_band"] / exam["exam_predictions"]
+    table = series_rate[[SERIES_ID_COLUMN]].join(exam, on=SERIES_ID_COLUMN)
+    composition_of = groups.set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN]
+    gate_of = pool_reference.set_index(COMPOSITION_ID_COLUMN)["gate"]
+    gate = table[SERIES_ID_COLUMN].map(composition_of).map(gate_of)
+    table["exam_status"] = np.select(
+        [table["exam_predictions"].notna(), table[SERIES_ID_COLUMN].map(composition_of).isna(), gate != GATE_LEVEL],
+        [TECHNIQUE_TESTED, "not_estimable", TECHNIQUE_COMPOSITION_BELOW_FLOOR], default="no_exam_months")
+    return table
+
+
+def check_series_exam(series_exam: pd.DataFrame, series_backtest: pd.DataFrame, configuration: Config, check_log: list) -> None:
+    """Checks 10 and 11: the summary adds up to the chosen predictions; the interval keeps its promise."""
+    chosen = series_backtest[series_backtest["is_chosen"] == 1]
+    adds_up = (int(series_exam["exam_predictions"].sum()) == len(chosen)
+               and int(series_exam["exam_in_band"].sum()) == int(chosen["in_band"].sum())
+               and abs(series_exam["exam_pred_units"].sum() - chosen["pred_units"].sum()) <= UNITS_TOLERANCE
+               and abs(series_exam["exam_real_units"].sum() - chosen["real_units"].sum()) <= UNITS_TOLERANCE)
+    configuration.log_check(STEP_LABEL, check_log, "the exam of every series adds up to its chosen predictions (counts and units)",
+                            adds_up, failure_detail="the summary per series does not add up to sff_series_backtest",
+                            context=f"{len(chosen):,} chosen predictions")
+    coverage = chosen["in_band"].mean() if len(chosen) else np.nan
+    configuration.log_check(STEP_LABEL, check_log, f"the predictions in their interval reach {COVERAGE_WARNING:.0%} in the exam",
+                            bool(np.isnan(coverage) or coverage >= COVERAGE_WARNING),
+                            failure_detail=f"only {coverage:.0%} of the predictions in their interval",
+                            context=f"{coverage:.0%} in the interval" if np.isfinite(coverage) else "no prediction", blocking=False)
+
+
 # ═══════════════════════════════════════════════════════════════════════════════════
 # 6 · THE FUTURE RATE OF EVERY TECHNIQUE
 # ═══════════════════════════════════════════════════════════════════════════════════
```

## step_informe.py

```diff
--- antes/step_informe.py	2026-10-01 06:30:20.403159412 +0000
+++ step_informe.py	2026-10-01 06:31:59.341228588 +0000
@@ -379,9 +379,54 @@
              "wape_series: serie a serie, sin compensaciones):", "",
              markdown_table(results["portfolio_exam_summary"], 3) if results.get("portfolio_exam_summary") is not None else "",
              markdown_table(results["portfolio_exam"], 3) if results.get("portfolio_exam") is not None else ""]
+    precision_by_type = exam_precision_by_series_type(results)
+    if precision_by_type is not None:
+        overall = precision_by_type.iloc[0]
+        headline.append(("predicciones del examen dentro de su intervalo · WAPE serie a serie",
+                         f"{overall['en_intervalo']:.0%} de {int(overall['predicciones']):,} · {overall['wape']:.1%}"))
+        lines += ["**Cuánto acertamos, forecast serie a forecast serie** (la técnica elegida de cada serie, en cada mes de "
+                  "examen y horizonte, aplicada a su propia pipeline; cada predicción con su intervalo, construido como la banda "
+                  "del forecast; en_intervalo: proporción de predicciones cuyo valor real cayó dentro). Cruzado por el tipo de "
+                  "serie: volatilidad (φ de su propia tasa), tendencia y estacionalidad, solo donde son medibles:", "",
+                  markdown_table(precision_by_type.assign(
+                      predicciones=precision_by_type["predicciones"].astype(int),
+                      en_intervalo=precision_by_type["en_intervalo"].map("{:.0%}".format),
+                      wape=precision_by_type["wape"].map("{:.1%}".format),
+                      sesgo=precision_by_type["sesgo"].map("{:+.1%}".format)), 0)]
     return "\n".join(lines) + "\n"
 
 
+def exam_precision_by_series_type(results: dict):
+    """The exam of every forecast series (its chosen technique) summed by type of series: all, by
+    volatility, by trend and by seasonality (counts and units added, then the ratios)."""
+    audit = results.get("audit")
+    if not audit:
+        return None
+    exam = audit["series_exam"].merge(audit["series_dynamics"], on=SERIES_ID_COLUMN, how="left")
+    exam = exam[exam["exam_status"] == "tested"]
+    if exam.empty:
+        return None
+    measured = exam["measurable"] == "yes"
+    segments = [("todas las series examinadas", exam.index == exam.index),
+                ("volatilidad baja (φ ≤ 1,5)", exam["phi"] <= 1.5),
+                ("volatilidad alta (φ > 1,5)", exam["phi"] > 1.5),
+                ("con tendencia (medible)", measured & exam["trend"].fillna(0).ne(0)),
+                ("sin tendencia (medible)", measured & exam["trend"].fillna(0).eq(0)),
+                ("estacional (medible)", measured & exam["seasonal"].eq(1)),
+                ("no estacional (medible)", measured & exam["seasonal"].eq(0)),
+                ("dinámica no medible (poco soporte o historia)", ~measured)]
+    rows = []
+    for name, mask in segments:
+        block = exam[mask]
+        if block.empty:
+            continue
+        rows.append({"segmento": name, "series": len(block), "predicciones": block["exam_predictions"].sum(),
+                     "en_intervalo": block["exam_in_band"].sum() / block["exam_predictions"].sum(),
+                     "wape": block["exam_abs_err_units"].sum() / block["exam_real_units"].sum(),
+                     "sesgo": block["exam_pred_units"].sum() / block["exam_real_units"].sum() - 1})
+    return pd.DataFrame(rows)
+
+
 def chapter_uplift(results: dict, configuration: Config) -> str:
     cells, check = results.get("uplift_cells"), results.get("contract_check")
     comparison, verdict = results.get("uplift_backtest"), results.get("uplift_verdict")
```

## step_nucleo.py

```diff
--- antes/step_nucleo.py	2026-10-01 06:30:20.338159283 +0000
+++ step_nucleo.py	2026-10-01 06:30:43.664249485 +0000
@@ -153,6 +153,26 @@
                          ("se_estimacion_pp", "s11_se_estimacion_pp", "11", "error of the estimate (pp)"),
                          ("se_prediccion_pp", "s11_se_prediccion_pp", "11", "error of next month's prediction (pp): never below the series' own noise"),
                          ("nivel_riesgo", "s11_nivel_riesgo", "11", "how the rate was obtained: A_propio … N_sin_impacto")]
+# per forecast series, from the audit step: its own dynamics (to cross the precision with it)
+SERIES_VALUES_DYNAMICS = [("phi", "s13_series_phi", "13", "volatility of its own rate: observed variation / binomial noise (≈ 1: only chance)"),
+                          ("trend", "s13_series_trend", "13", "+1 / −1 significant trend of its own rate, 0 none"),
+                          ("trend_pp_year", "s13_series_trend_pp_year", "13", "slope of its own rate, pp per year"),
+                          ("seasonal", "s13_series_seasonal", "13", "1 when its own rate has a month effect (≥ 24 months)"),
+                          ("amplitude_pp", "s13_series_amplitude_pp", "13", "size of its month effect (pp)"),
+                          ("measurable", "s13_series_measurable", "13", "yes · low_support (< 30 a month: not conclusive) · short_history"),
+                          ("differs_from_composition", "s13_series_differs", "13", "1 when its trend or season differs from its composition's")]
+# per forecast series: the exam of its chosen technique (every exam month and horizon), as ratios repeated on its rows
+SERIES_VALUES_EXAM = [("exam_status", "s19_exam_status", "19", "tested · composition_below_floor · not_estimable · no_exam_months"),
+                      ("exam_mae_pp", "s19_exam_mae_pp", "19", "mean absolute error of its rate in the exam (pp)"),
+                      ("exam_bias_pp", "s19_exam_bias_pp", "19", "mean error of its rate in the exam (pp): + over-predicts"),
+                      ("exam_wape", "s19_exam_wape", "19", "Σ |predicted − real| / Σ real renewed units in the exam"),
+                      ("exam_coverage", "s19_exam_coverage", "19", "share of its exam predictions whose real rate fell inside the interval")]
+# … and as counts and units on its FIRST row only: a SUM over any filter counts every series once
+SERIES_SUMMABLE_EXAM = [("exam_predictions", "s19_exam_predictions", "19", "exam predictions (first row of the series only: SUM)"),
+                        ("exam_in_band", "s19_exam_in_band", "19", "exam predictions inside their interval (first row only: SUM)"),
+                        ("exam_pred_units", "s19_exam_pred_units", "19", "renewed units predicted in the exam (first row only: SUM)"),
+                        ("exam_real_units", "s19_exam_real_units", "19", "renewed units real in the exam (first row only: SUM)"),
+                        ("exam_abs_err_units", "s19_exam_abs_err_units", "19", "Σ |predicted − real| units in the exam (first row only: SUM)")]
 # step 13, per estimation id (the dynamics of the rate the series takes)
 ESTIMATION_VALUES_STEP_13 = [("phi", "s13_phi", "13", "φ of the estimation id: observed variation of its rate / binomial noise (≈ 1: nothing to model)"),
                              ("tendencia", "s13_tendencia", "13", "+1 / −1 significant trend of the rate, 0 none"),
@@ -176,7 +196,8 @@
                      technique_decision: pd.DataFrame = None, exam_by_pool: pd.DataFrame = None,
                      forecast: pd.DataFrame = None, time_series_rows: pd.DataFrame = None,
                      time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None,
-                     ladder_stages: pd.DataFrame = None) -> tuple:
+                     ladder_stages: pd.DataFrame = None, series_dynamics: pd.DataFrame = None,
+                     series_exam: pd.DataFrame = None) -> tuple:
     """The core table with every block whose step has run, and its legend; checked and written.
 
     INPUT:   the fine table and the results of every step that has run (None = not run).
@@ -220,6 +241,16 @@
 
     # [4] the values of the series and of its estimation id
     ladder_stages = stages_of_every_series(ladder_stages, series_rate)
+    if series_dynamics is not None:
+        core = add_block(core, series_dynamics, SERIES_ID_COLUMN, "s03_fs_id", SERIES_VALUES_DYNAMICS)
+        blocks_present.append(SERIES_VALUES_DYNAMICS)
+    if series_exam is not None:
+        core = add_block(core, series_exam, SERIES_ID_COLUMN, "s03_fs_id", SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)
+        # the counts and units only on the first row of the extract of every series: SUM counts each series once
+        first_rows = core[core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW].groupby("s03_fs_id").head(1).index
+        summable_names = [name for _, name, _, _ in SERIES_SUMMABLE_EXAM]
+        core.loc[~core.index.isin(first_rows), summable_names] = np.nan
+        blocks_present.append(SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)
     series_blocks = [(series_table, SERIES_VALUES_STEP_06), (series_rate, SERIES_VALUES_STEP_08),
                      (ladder_stages, SERIES_VALUES_STEP_10), (series_estimate, SERIES_VALUES_STEP_11)]
     for block_frame, block_specs in series_blocks:
@@ -535,6 +566,9 @@
                                     if name in ("s17_origen_tasa", "s17_tecnica", "s15_via_uplift", "s17_h")
                                     else "no sumar: valor de la fila (una tasa o un uplift)", description))
                 continue
+            if name in {spec_name for _, spec_name, _, _ in SERIES_SUMMABLE_EXAM}:
+                legend_rows.append((name, step, LEVEL_SERIES, AGGREGATE_SUM, description))
+                continue
             level = LEVEL_UNIT if name in unit_names else LEVEL_SERIES
             legend_rows.append((name, step, level, AGGREGATE_ATTRIBUTE, description))
     legend = pd.DataFrame(legend_rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
@@ -570,7 +604,7 @@
     #     a gap is not a unit of step 04, so the unit blocks are checked on the rows of the extract
     unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_04 + UNIT_VALUES_STEP_07}
     first_columns = [block_specs[0][1] for block_specs in blocks_present if block_specs and block_specs[0][1] in core.columns
-                     and not block_specs[0][1].startswith(("s13_", "s14_", "s17_", "fin_"))]
+                     and not block_specs[0][1].startswith(("s13_", "s14_", "s17_", "s19_", "fin_"))]
     series_columns = [name for name in first_columns if name not in unit_names]
     unit_columns = [name for name in first_columns if name in unit_names]
     raw_rows = core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW
```

## test_step_audit.py

```diff
--- antes/test_step_audit.py	2026-10-01 06:30:20.459133117 +0000
+++ test_step_audit.py	2026-10-01 06:31:59.341629285 +0000
@@ -42,7 +42,7 @@
 def test_the_audit_tables() -> None:
     print("B · the audit tables add up to their sources")
     results, console = run_synthetic()
-    check("17 checks: 17 ok" in console.split("STEP AUD")[1], "the 17 checks of the audit step pass")
+    check("20 checks: 20 ok" in console.split("STEP AUD")[1], "the 20 checks of the audit step pass")
     audit = results["audit"]
     techniques = audit["composition_techniques"]
     check(set(techniques["status"]) <= {"tested", "not_enough_history", "composition_below_floor"}
@@ -72,7 +72,33 @@
           "the future rate of every technique, one chosen per composition and horizon")
 
 
+def test_the_exam_in_the_core() -> None:
+    print("C · the exam of every forecast series: its interval, and its sums from the core")
+    results, console = run_synthetic()
+    backtest = results["audit"]["series_backtest"]
+    row = backtest.iloc[0]
+    check(row["band_low_rate"] <= row["pred_rate"] <= row["band_high_rate"]
+          and row["in_band"] == int(row["band_low_rate"] - 1e-12 <= row["real_rate"] <= row["band_high_rate"] + 1e-12),
+          "every test has its interval around the prediction, and says whether the real rate fell inside")
+    core = results["core"]
+    chosen = backtest[backtest["is_chosen"] == 1]
+    check(core["s19_exam_predictions"].sum() == len(chosen) and core["s19_exam_in_band"].sum() == chosen["in_band"].sum(),
+          "SUM over the core counts every exam prediction once (counts on the first row of each series)")
+    check(abs(core["s19_exam_real_units"].sum() - chosen["real_units"].sum()) < 1e-6
+          and abs(core["s19_exam_pred_units"].sum() - chosen["pred_units"].sum()) < 1e-6,
+          "SUM over the core gives the units predicted and real of the exam")
+    one_series = core[core["s03_fs_id"] == "EU|A|0|0|0|0|web"]
+    check(one_series["s19_exam_coverage"].nunique() == 1 and one_series["s19_exam_predictions"].notna().sum() == 1,
+          "a ratio is repeated on every row of the series; a count only on its first row")
+    check({"s13_series_phi", "s13_series_trend", "s13_series_seasonal", "s13_series_measurable"} <= set(core.columns),
+          "the core carries each series' own dynamics, to cross it with the precision")
+    report = open(f"{synthetic_with().output_folder}/informe_sff.md", encoding="utf-8").read()
+    check("Cuánto acertamos" in report and "dentro de su intervalo" in report,
+          "the report says how much we erred and how many predictions were in their interval, by type of series")
+
+
 if __name__ == "__main__":
     test_the_stages()
     test_the_audit_tables()
+    test_the_exam_in_the_core()
     finish()
```

## vocabulario.py

```diff
--- antes/vocabulario.py	2026-10-01 06:30:20.258495149 +0000
+++ vocabulario.py	2026-10-01 06:30:20.541324892 +0000
@@ -195,6 +195,7 @@
 TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
 TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
 TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
+TABLE_SERIES_EXAM = "series_exam"                        # every forecast series: how its chosen technique did in the exam
 TECHNIQUE_TESTED = "tested"
 TECHNIQUE_NOT_ENOUGH_HISTORY = "not_enough_history"
 TECHNIQUE_COMPOSITION_BELOW_FLOOR = "composition_below_floor"
```
