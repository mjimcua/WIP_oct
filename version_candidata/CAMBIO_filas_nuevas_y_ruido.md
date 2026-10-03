# Cambio · las filas que añade o borra el framework (paso 21) y el error del examen frente al ruido

## Paso 21 (nuevo) · `step_21_new_rows.py`
Pone una al lado de otra, mes a mes y con el mismo formato (`mes · rol · origen · filas · series · unidades · usd ·
esperado_usd`), todas las filas que el framework añade o borra:
- **`hueco`** (paso 08): forecast units sin nada que vencer, añadidas dentro de la historia de una serie para que su
  serie mensual sea continua; medidas a 0, tasa nula;
- **`resultado_adelantado_borrado`** y **`pipeline_borrada`** (paso 02);
- **`proyectada`** y **`simulada`** (paso 17).

Concilia con el raw: lo que vencía = lo que queda + la pipeline borrada. Escribe `sff_filas_nuevas`. El capítulo 2 del
informe pasa de "Huecos rellenados" a "Las filas que añade (y borra) el framework", con la misma tabla.

## Paso 19 · el ruido de cada predicción
`noise_pp` = 100·√(p(1−p)/n) en cada predicción; por serie y método, `rmse_pp`, `noise_pp` y `error_over_noise`; en el
resumen, `error_vs_ruido`. En la dimensión: `s19_exam_rmse_pp`, `s19_exam_noise_pp`, `s19_exam_error_over_noise` y
`s19_raw_error_over_noise`. En el informe (capítulo 5): el error frente al ruido por tamaño de serie y para el total de
la cartera, con la parte del error que es ruido.

Verificación: 23 ficheros de test en verde (`test_step_21` nuevo).

## step_19_series_exam.py

```diff
--- /tmp/before_newrows/step_19_series_exam.py	2026-10-01 16:18:17.019410502 +0000
+++ step_19_series_exam.py	2026-10-01 16:18:35.802733177 +0000
@@ -13,6 +13,11 @@
                of levels known at the origin; a series with no composition: its mandatory cell
   hoja_<grain> the spreadsheet: the rate of the last baseline_months closed months per grain
 
+Every prediction also carries the BINOMIAL NOISE of the real rate it tries to hit, √(p(1−p)/units due)
+with p its predicted rate: the error that even a perfect prediction would make (the noise rule: an error
+of the size of the noise is the limit, not a failure). error_over_noise = RMS error / RMS noise: about 1,
+the prediction is at the limit; well above 1, something knowable is missing; well below 1, suspicious.
+
 Every prediction keeps its steps (composition rate, shift, series rate) and, for raw and framework,
 its INTERVAL, built as the forecast builds its band (prediction.py): the error quantiles of the
 technique × the binomial error with the series' own units due. A prediction is IN THE INTERVAL when
@@ -265,12 +270,15 @@
     predictions["pred_units"] = predictions["pred_rate"] * predictions["due_units"]
     predictions["err_units"] = predictions["pred_units"] - predictions["real_units"]
     predictions["err_pp"] = 100 * (predictions["pred_rate"] - predictions["real_rate"])
+    # the noise of the rate it tries to hit: what even a perfect prediction would miss by
+    predictions["noise_pp"] = 100 * np.sqrt(np.clip(predictions["pred_rate"] * (1 - predictions["pred_rate"]), 0, None)
+                                            / predictions["due_units"].clip(lower=1))
     has_interval = predictions["band_low"].notna()
     predictions["in_band"] = np.where(has_interval, ((predictions["real_rate"] >= predictions["band_low"] - 1e-12)
                                                      & (predictions["real_rate"] <= predictions["band_high"] + 1e-12)).astype(float), np.nan)
     return predictions[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, period, "h", "tramo_h", "origin", "method", "technique",
                         "composition_rate", "credibility_shift_logit", "pred_rate", "band_low", "band_high", "real_rate",
-                        "due_units", "pred_units", "real_units", "err_units", "err_pp", "in_band"]]
+                        "due_units", "pred_units", "real_units", "err_units", "err_pp", "noise_pp", "in_band"]]
 
 
 def errors_per_series(detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
@@ -284,7 +292,10 @@
         summary = pd.DataFrame({"predictions": grouped.size(), "in_band": grouped["in_band"].sum(),
                                 "pred_units": grouped["pred_units"].sum(), "real_units": grouped["real_units"].sum(),
                                 "abs_err_units": grouped["abs_err_units"].sum(), "mae_pp": grouped["abs_err_pp"].mean(),
-                                "bias_pp": grouped["err_pp"].mean()})
+                                "bias_pp": grouped["err_pp"].mean(),
+                                "rmse_pp": np.sqrt(block.assign(_e2=block["err_pp"] ** 2).groupby(SERIES_ID_COLUMN)["_e2"].mean()),
+                                "noise_pp": np.sqrt(block.assign(_n2=block["noise_pp"] ** 2).groupby(SERIES_ID_COLUMN)["_n2"].mean())})
+        summary["error_over_noise"] = summary["rmse_pp"] / summary["noise_pp"].where(summary["noise_pp"] > 0)
         summary["wape"] = summary["abs_err_units"] / summary["real_units"].where(summary["real_units"] > 0)
         summary["coverage"] = summary["in_band"] / summary["predictions"]
         frames.append(summary.add_prefix(f"{method}_"))
@@ -321,5 +332,7 @@
             summary_rows.append({"metodo": method, "h": horizon, "error_total_medio": float(month_errors.abs().mean()),
                                  "sesgo_total_medio": float(month_errors.mean()),
                                  "wape_series": float(rows["err_units"].abs().sum() / rows["real_units"].sum()),
+                                 "error_vs_ruido": float(np.sqrt((rows["err_pp"] ** 2).mean()) / np.sqrt((rows["noise_pp"] ** 2).mean()))
+                                                   if (rows["noise_pp"] > 0).any() else np.nan,
                                  "en_intervalo": float(rows["in_band"].mean()) if rows["in_band"].notna().any() else np.nan})
     return by_month, pd.DataFrame(summary_rows).sort_values(["h", "error_total_medio"]).reset_index(drop=True)
```

## step_nucleo.py

```diff
--- /tmp/before_newrows/step_nucleo.py	2026-10-01 16:18:16.878969351 +0000
+++ step_nucleo.py	2026-10-01 16:18:35.803590637 +0000
@@ -169,7 +169,12 @@
                       ("framework_coverage", "s19_exam_coverage", "19", "share of its framework predictions whose real rate fell inside the interval"),
                       ("raw_mae_pp", "s19_raw_mae_pp", "19", "the same error predicting the series alone with its own history (raw)"),
                       ("raw_coverage", "s19_raw_coverage", "19", "share of its raw predictions inside their interval"),
-                      ("improvement_mae_pp", "s19_improvement_mae_pp", "19", "raw error − framework error (pp): + the framework predicts it better")]
+                      ("improvement_mae_pp", "s19_improvement_mae_pp", "19", "raw error − framework error (pp): + the framework predicts it better"),
+                      ("framework_rmse_pp", "s19_exam_rmse_pp", "19", "root mean square error of its rate in the exam, framework (pp)"),
+                      ("framework_noise_pp", "s19_exam_noise_pp", "19", "binomial noise of its monthly rate: the error of a perfect prediction (pp)"),
+                      ("framework_error_over_noise", "s19_exam_error_over_noise", "19",
+                       "framework error / noise: ≈ 1 at the limit · well above 1, something knowable is missing"),
+                      ("raw_error_over_noise", "s19_raw_error_over_noise", "19", "the same for the series alone (raw)")]
 # … and as counts and units: in the dimension a SUM over any filter counts every forecast series once
 SERIES_SUMMABLE_EXAM = [("framework_predictions", "s19_exam_predictions", "19", "exam predictions (SUM)"),
                         ("framework_in_band", "s19_exam_in_band", "19", "framework predictions inside their interval (SUM)"),
```

## step_informe.py

```diff
--- /tmp/before_newrows/step_informe.py	2026-10-01 16:18:16.948677206 +0000
+++ step_informe.py	2026-10-01 16:19:30.287721312 +0000
@@ -101,7 +101,7 @@
     # [2] the chapters
     headline, chapters = [], []
     chapters.append(chapter_raw(raw, results, configuration))
-    chapters.append(chapter_gaps(results, configuration))
+    chapters.append(chapter_new_rows(results, configuration))
     chapters.append(chapter_support(results, configuration, headline))
     chapters.append(chapter_dynamics(results, configuration))
     chapters.append(chapter_precision(results, configuration, headline))
@@ -218,22 +218,76 @@
     return "\n".join(lines) + "\n"
 
 
-def chapter_gaps(results: dict, configuration: Config) -> str:
-    rated = results["rated_units"]
-    gaps = rated[rated[SYNTHETIC_COLUMN] == 1]
+def chapter_new_rows(results: dict, configuration: Config) -> str:
+    """Chapter 2: every row the framework adds or wipes, in one format (step 21), and the series with most gaps."""
+    lines = ["## 2 · Las filas que añade (y borra) el framework", "",
+             "El forecast no usa el extracto tal cual. Añade o borra filas de cuatro tipos, cada uno por un motivo:", "",
+             "- **hueco** (paso 08): una forecast unit sin nada que vencer, añadida DENTRO de la historia de una serie "
+             "estimable para que su serie mensual no tenga agujeros (las técnicas leen meses consecutivos: una tendencia, una "
+             "estación, una media móvil). Todas sus medidas son 0 y su tasa queda **nula, nunca 0 %**: añade un mes, no dinero.",
+             "- **resultado_adelantado_borrado** (paso 02): una renovación ya registrada desde el mes en curso; el mes no "
+             "ha terminado, y el forecast la predice.",
+             "- **pipeline_borrada** (paso 02): la pipeline de una licencia de 1 año vendida o renovada desde el mes en "
+             "curso (vence 12 meses después): aún no se conoce.",
+             "- **proyectada** y **simulada** (paso 17): las renovaciones esperadas de las licencias de 1 año que vencen en la "
+             "ventana de simulación, y la captación simulada; vencen 12 meses después y sustituyen a la pipeline borrada.", ""]
+    new_rows = results.get("new_rows")
+    if new_rows is not None and len(new_rows):
+        totals = totals_of_new_rows(new_rows)
+        lines += ["**Por origen:**", "", markdown_table(totals, 0), ""]
+        future = new_rows[new_rows["origen"] != "hueco"]
+        if len(future):
+            lines += ["**Mes a mes, lo borrado y lo creado** (las unidades y el dinero que vencen; `esperado_usd`: lo que el "
+                      "forecast espera renovar de las filas creadas):", "", markdown_table(future, 0), ""]
     rate_summary = results["series_rate"]
     with_gaps = rate_summary[rate_summary["huecos"] > 0].sort_values("huecos", ascending=False)
     history_months = rate_summary["meses_historia"].sum()
-    lines = ["## 2 · Huecos rellenados", "",
-             "Un hueco es un mes sin vencimientos DENTRO de la historia de una serie estimable. Un mes sin vencimientos "
-             "no dice nada de la tasa: su tasa queda **nula, nunca 0 %**, y el mes aparece como fila explícita con medidas a 0 "
-             "para que la historia de la serie esté completa y las sumas sigan cuadrando.", "",
-             f"- Huecos añadidos: **{len(gaps):,}** en **{len(with_gaps):,} series**, el "
-             f"{len(gaps) / history_months if history_months else 0:.1%} de los meses de historia.", "",
-             markdown_table(with_gaps.head(TOP_ROWS)[[SERIES_ID_COLUMN, "meses_historia", "huecos", "n_propio", "usd_por_predecir"]], 0)]
+    lines += [f"**Los huecos:** {int(with_gaps['huecos'].sum()):,} en {len(with_gaps):,} series, el "
+              f"{with_gaps['huecos'].sum() / history_months if history_months else 0:.1%} de los meses de historia. "
+              f"Las series con más huecos:", "",
+              markdown_table(with_gaps.head(TOP_ROWS)[[SERIES_ID_COLUMN, "meses_historia", "huecos", "n_propio", "usd_por_predecir"]], 0)]
     return "\n".join(lines) + "\n"
 
 
+def totals_of_new_rows(new_rows: pd.DataFrame) -> pd.DataFrame:
+    """One row per origin: months, rows, units, USD and USD expected (the same format as step 21)."""
+    rows = []
+    for origin, block in new_rows.groupby("origen", sort=False):
+        rows.append({"origen": origin, "meses": f"{block['mes'].min()}..{block['mes'].max()}", "filas": int(block["filas"].sum()),
+                     "unidades": float(block["unidades"].sum()), "usd": float(block["usd"].sum()),
+                     "esperado_usd": float(block["esperado_usd"].sum()) if block["esperado_usd"].notna().any() else np.nan})
+    return pd.DataFrame(rows)
+
+
+def exam_error_against_noise(results: dict):
+    """The exam error against the binomial noise, by size of the series (contracts due in the month) and for
+    the total of the portfolio: what share of the error is noise no prediction can remove."""
+    series_exam = results.get("series_exam")
+    if not series_exam:
+        return None, None
+    detail = series_exam["detail"]
+    detail = detail[detail["method"].isin(["raw", "framework"])].copy()
+    detail["tamano"] = pd.cut(detail["due_units"], [0, 30, 271, np.inf], right=False,
+                              labels=["< 30 al mes", "30-270 al mes", "≥ 271 al mes"])
+    rows = []
+    for (size, method), block in detail.groupby(["tamano", "method"], observed=True):
+        rmse, noise = float(np.sqrt((block["err_pp"] ** 2).mean())), float(np.sqrt((block["noise_pp"] ** 2).mean()))
+        rows.append({"tamano": str(size), "metodo": method, "predicciones": len(block), "error_pp": rmse, "ruido_pp": noise,
+                     "error_vs_ruido": rmse / noise if noise > 0 else np.nan,
+                     "parte_del_error_que_es_ruido": min(1.0, noise ** 2 / rmse ** 2) if rmse > 0 else np.nan})
+    by_size = pd.DataFrame(rows)
+    # the total of the portfolio, every exam month and horizon: its error and its noise, in pp of the rate
+    total_rows = []
+    for (month, horizon, method), block in detail.groupby(["period", "h", "method"]):
+        due = block["due_units"].sum()
+        total_rows.append({"metodo": method, "error_pp": 100 * (block["pred_units"].sum() - block["real_units"].sum()) / due,
+                           "ruido_pp": 100 * np.sqrt((block["pred_rate"] * (1 - block["pred_rate"]) * block["due_units"]).sum()) / due})
+    total = pd.DataFrame(total_rows).groupby("metodo").agg(error_pp=("error_pp", lambda values: float(np.sqrt((values ** 2).mean()))),
+                                                           ruido_pp=("ruido_pp", lambda values: float(np.sqrt((values ** 2).mean()))))
+    total["error_vs_ruido"] = total["error_pp"] / total["ruido_pp"]
+    return by_size, total.reset_index()
+
+
 def chapter_support(results: dict, configuration: Config, headline: list) -> str:
     card = results["series_estimate"]
     total_usd = card["usd_por_predecir"].sum()
@@ -382,6 +436,16 @@
              "wape_series: serie a serie, sin compensaciones):", "",
              markdown_table(results["portfolio_exam_summary"], 3) if results.get("portfolio_exam_summary") is not None else "",
              markdown_table(results["portfolio_exam"], 3) if results.get("portfolio_exam") is not None else ""]
+    by_size, total = exam_error_against_noise(results)
+    if by_size is not None and len(by_size):
+        lines += ["", "**El error frente al ruido** (la regla del ruido: una diferencia menor que el ruido no es una diferencia; "
+                  "un error del tamaño del ruido no es un fallo, es el límite). `ruido_pp`: lo que se equivocaría una predicción "
+                  "perfecta, √(p(1−p)/n). `error_vs_ruido` ≈ 1: al límite; claramente mayor que 1: falta algo que se podía saber. "
+                  "`parte_del_error_que_es_ruido`: la parte del error que ninguna predicción puede quitar. Por tamaño de la serie:", "",
+                  markdown_table(by_size.assign(parte_del_error_que_es_ruido=by_size["parte_del_error_que_es_ruido"].map("{:.0%}".format)), 2),
+                  "", "Y el **total de la cartera** en cada mes de examen, con el mismo cálculo: al juntar todo el volumen, el ruido baja "
+                  "con la raíz del tamaño, y lo que queda por encima del ruido es error del modelo:", "",
+                  markdown_table(total, 3)]
     precision_by_type = exam_precision_by_series_type(results)
     if precision_by_type is not None:
         overall = precision_by_type.iloc[0]
```

## pipeline.py

```diff
--- /tmp/before_newrows/pipeline.py	2026-10-01 16:18:17.316225482 +0000
+++ pipeline.py	2026-10-01 16:18:35.804149155 +0000
@@ -74,6 +74,7 @@
 from step_18_validation import validate_chain
 from step_19_series_exam import examine_series
 from step_20_time_series import build_time_series_and_total, split_time_series_rows
+from step_21_new_rows import count_the_new_rows
 from step_audit import build_audit_tables
 from step_informe import build_report
 from step_nucleo import build_core_table
@@ -472,7 +473,7 @@
             series_exam=context["series_exam"]["per_series"])
         return {"core": core_table, "core_legend": legend, "forecast_series": dimension}
 
-    report_reads = ("raw_extract", "audit", "backtest", "contract_check", "fine_table", "forecast", "forecast_total", "ladder",
+    report_reads = ("raw_extract", "new_rows", "audit", "backtest", "contract_check", "fine_table", "forecast", "forecast_total", "ladder",
                     "money_by_level", "pool_dynamics", "pool_reference", "portfolio_dynamics", "portfolio_profile", "rated_units",
                     "series_table", "series_estimate", "series_exam", "series_rate", "time_series", "uplift_backtest",
                     "uplift_cells", "uplift_verdict", "validation", "core")
@@ -516,6 +517,10 @@
                                 "uplift_cells", "uplift_verdict", "rated_units"), ("forecast",), forecast, module="step_17_forecast"),
         Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("series_exam",), series_exam, module="step_19_series_exam"),
         Step("time_series", "20", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series, module="step_20_time_series"),
+        Step("new_rows", "21", ("fine_table", "rated_units", "forecast"), ("new_rows",),
+             lambda context: {"new_rows": count_the_new_rows(context["fine_table"], context["rated_units"],
+                                                             context["forecast"]["forecast"], context.configuration)},
+             module="step_21_new_rows"),
         Step("audit", "AUD", ("ladder", "series_rate", "series_estimate", "rated_units", "pool_series", "pool_reference",
                               "pool_dynamics", "backtest", "forecast", "series_exam"), ("audit",), audit, module="step_audit"),
         Step("core", "NU", ("fine_table", "forecast_units", "support_bound", "rated_units", "series_table", "series_rate",
```

## vocabulario.py

```diff
--- /tmp/before_newrows/vocabulario.py	2026-10-01 16:18:16.770361424 +0000
+++ vocabulario.py	2026-10-01 16:18:35.804487691 +0000
@@ -195,6 +195,7 @@
 TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
 TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
 TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
+TABLE_NEW_ROWS = "filas_nuevas"                          # step 21: the rows added or wiped, month × origin
 TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
 TABLE_DIMENSION_LEVEL_VALUES = "dimension_level_values"  # step 01b: every value of every leveled dimension, its rate and group
 TABLE_DIMENSION_LEVELS = "dimension_levels"              # step 01b: the generated groups of every leveled dimension
```

## step_21_new_rows.py (nuevo)

Ver el fichero.
