"""
step_informe.py — The report: what the raw was, how it was improved, and how well the
framework predicts the renewal rate. And the card of every series.

It reads the results of every step that has run and writes, in the output folder, the
markdown report informe_sff.md with these chapters:
  1. The raw and how it was improved: the extract, the checks of every step, what was
     corrected (null renewals, early results of the future, the discount), the grain.
  2. The gaps filled: months with nothing due inside the history of a series.
  3. The binomial support, before and after the ladder: how much money had a precise rate
     on its own, and how much has it now; the error of the estimate before and after.
  4. The dynamics of the rate: the portfolio's month profile and the pools' attributes.
  5. How well the rate is predicted: the backtest, the choices, the exam per pool, per
     risk level and for the total.
  6. The revaluation: the uplift of the cells, the contract rule and the backtest's verdict.
  7. The forecast in money: by month and by year, with its bands; the final validation.
  8. The card of every series (sff_ficha_serie) and what is still to come.
It also writes sff_ficha_serie: one row per series with every attribute known about it.

Actions (logged as they are done):
  1. the card of every series
  2. the eight chapters of the report
  3. check the card and the report                                    checks 1-2
  4. write the card and the report file                               check 3
  5. count the checks; stop if any failed
  6. show the headline numbers, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. the card has one row per series
   2. the report has its eight chapters
   3. table sff_ficha_serie written and read back

Output: the card (one row per series) · table sff_ficha_serie · file informe_sff.md.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd

from config import Config, UNKNOWN_DISCOUNT_BUCKET
from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, LEVEL_OWN,
                         METHOD_FRAMEWORK, PURPOSE_SELECTION, REPORT_FILE_NAME, ROLES_IN_ORDER, ROLE_PROJECTION,
                         S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN,
                         SYNTHETIC_COLUMN, TABLE_SERIES_CARD, TOTAL_ORIGIN_TOTAL, TRUTH_ROLES, UPLIFT_CELL_ID_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "IN"
STEP_NAME = "REPORT"
STEP_PURPOSE = ("tell, with the numbers of this run, what the raw was and how it was improved, the gaps filled, how "
                "the binomial support improved, the dynamics of the rate and how well the rate is predicted; and "
                "leave the card of every series")
STEP_ACTIONS = ["the card of every series",
                "the eight chapters of the report",
                "check the card and the report (checks 1-2)",
                "write the card and the report file (check 3)",
                "count the checks; stop if any failed",
                "show the headline numbers, as a table"]
STEP_OUTPUT = "one row per series (the card) · table sff_ficha_serie · file informe_sff.md"

# ─── named constants ─────────────────────────────────────────────────────────────
PROMISE_PP = 5.0                 # the promise to the business: a rate known within ±5 pp (90 %)
CHAPTER_COUNT = 8
TOP_ROWS = 10


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    """A pandas table as a markdown table (no extra library)."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    def cell(value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:,.{decimals}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)
    percent_columns = {column for column in frame.columns if str(column).startswith("pct") or str(column).endswith("_pct")
                       or str(column) == "dentro_banda"}
    frame = frame.copy()
    for column in percent_columns:
        frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator] + rows) + "\n"


def build_report(raw: pd.DataFrame, results: dict, configuration: Config) -> pd.DataFrame:
    """The card of every series and the markdown report; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the card of every series
    card = series_card(results, configuration)
    configuration.log_action(STEP_LABEL, 1, f"card of {len(card):,} series × {len(card.columns)} attributes")

    # [2] the chapters
    headline, chapters = [], []
    chapters.append(chapter_raw(raw, results, configuration))
    chapters.append(chapter_new_rows(results, configuration))
    chapters.append(chapter_support(results, configuration, headline))
    chapters.append(chapter_dynamics(results, configuration))
    chapters.append(chapter_precision(results, configuration, headline))
    chapters.append(chapter_uplift(results, configuration))
    chapters.append(chapter_forecast(results, configuration, headline))
    chapters.append(chapter_card_and_next(card))
    report_text = report_cover(raw, results, configuration, headline) + "\n".join(chapters)
    configuration.log_action(STEP_LABEL, 2, f"{len(chapters)} chapters written ({len(report_text):,} characters)")

    # [3] the checks
    configuration.log_action(STEP_LABEL, 3, "checking the card and the report")
    configuration.log_check(STEP_LABEL, check_log, "the card has one row per series",
                            len(card) == len(results["series"]) and card[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(card):,} card rows for {len(results['series']):,} series",
                            context=f"{len(card):,} series")
    configuration.log_check(STEP_LABEL, check_log, f"the report has its {CHAPTER_COUNT} chapters",
                            report_text.count("\n## ") == CHAPTER_COUNT,
                            failure_detail=f"{report_text.count(chr(10) + '## ')} chapters found")

    # [4] the card and the file
    configuration.log_action(STEP_LABEL, 4, "writing the card and the report file")
    configuration.write_table(STEP_LABEL, check_log, card, TABLE_SERIES_CARD)
    os.makedirs(configuration.output_folder, exist_ok=True)
    report_path = os.path.join(configuration.output_folder, REPORT_FILE_NAME)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    configuration.logger.doc(f"[{STEP_LABEL}] report written: {report_path}")

    # [5] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [6] the headline numbers
    configuration.log_action(STEP_LABEL, 6, "the headline numbers of this run:")
    configuration.show_table(pd.DataFrame(headline, columns=["indicador", "valor"]))
    return card


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CARD
# ═══════════════════════════════════════════════════════════════════════════════════

def series_card(results: dict, configuration: Config) -> pd.DataFrame:
    """One row per series: route, support, own rate, ladder, estimate, level, the dynamics and
    the chosen techniques of its estimation id, and their exam error."""
    card = results["series_estimate"].merge(results["series"][[SERIES_ID_COLUMN, COVERAGE_COLUMN, "primer_mes", "ultimo_mes"]],
                                            on=SERIES_ID_COLUMN, how="left")
    dynamics = results.get("pool_dynamics")
    if dynamics is not None and len(dynamics):
        card = card.merge(dynamics[[COMPOSITION_ID_COLUMN, "phi", "tendencia", "tendencia_pp_ano", "estacional",
                                    "amplitud_pp", "meses_alto", "meses_bajo"]], on=COMPOSITION_ID_COLUMN, how="left")
    backtest = results.get("backtest")
    if backtest is not None:
        per_band = backtest["decision"][[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]].merge(
            backtest["exam_by_pool"][[COMPOSITION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio"]],
            on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
        wide = per_band.pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h")
        wide.columns = [f"{name}_{band}" for name, band in wide.columns]
        card = card.merge(wide, left_on=COMPOSITION_ID_COLUMN, right_index=True, how="left")
    card["error_estimacion_pp"] = configuration.z * card["se_estimacion_pp"]
    return card


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHAPTERS
# ═══════════════════════════════════════════════════════════════════════════════════

def report_cover(raw: pd.DataFrame, results: dict, configuration: Config, headline: list) -> str:
    fine = results["fine_table"]
    series = results["series"]
    headline.insert(0, ("filas del extracto", f"{len(raw):,}"))
    headline.insert(1, ("forecast series", f"{len(series):,}"))
    headline.insert(2, ("USD por predecir", f"${series['usd_por_predecir'].sum():,.0f}"))
    lines = ["# SFF · Informe de la ejecución",
             f"Mes en curso: **{configuration.calendar_boundaries()['current']}** · {configuration.calendar_description()}",
             "", "### Cifras principales", "", markdown_table(pd.DataFrame(headline, columns=["indicador", "valor"])), ""]
    return "\n".join(lines)


def chapter_raw(raw: pd.DataFrame, results: dict, configuration: Config) -> str:
    fine = results["fine_table"]
    boundaries = configuration.calendar_boundaries()
    closed = fine[configuration.period_col] < boundaries["current"]
    null_read_as_zero = int((closed & fine[S0_RENEWED_UNITS_COLUMN].isna()).sum())
    projection = fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    early = fine[projection & fine[S0_RENEWED_UNITS_COLUMN].fillna(0).ne(0)]
    per_role = (fine.groupby(CALENDAR_ROLE_COLUMN).agg(meses=(configuration.period_col, "nunique"), filas=(configuration.period_col, "size"),
                                                       usd_vence=(configuration.pipeline_usd_col, "sum"))
                .reindex(ROLES_IN_ORDER).dropna(how="all").reset_index())
    checks = pd.DataFrame(configuration.check_history)
    lines = ["## 1 · El raw y cómo lo mejoramos", "",
             f"El extracto tiene **{len(raw):,} filas × {len(raw.columns)} columnas**, de "
             f"{fine[configuration.period_col].min()} a {fine[configuration.period_col].max()}. "
             f"Cada columna tiene un rol declarado en la Config y el calendario se genera desde el mes en curso.", "",
             markdown_table(per_role, 0),
             "**Comprobaciones de cada paso** (un fallo detiene la ejecución; un aviso se registra y sigue):", "",
             markdown_table(checks[["paso", "nombre", "comprobaciones", "ok", "avisos", "fallos", "avisos_detalle"]], 0) if len(checks) else "",
             "**Lo que se corrigió o completó en el raw:**", "",
             f"- Renovaciones nulas en meses cerrados leídas como 0 (nadie renovó): **{null_read_as_zero:,} filas**.",
             f"- Resultados adelantados borrados desde el mes en curso (el futuro no ha empezado): **{len(early):,} filas**, "
             f"{early[S0_RENEWED_UNITS_COLUMN].sum():,.0f} unidades y ${early[S0_RENEWED_USD_COLUMN].sum():,.0f}. "
             f"El raw original queda en las columnas s0_.",
             ]
    wiped_pipeline = fine[S0_PIPELINE_USD_COLUMN] - fine[configuration.pipeline_usd_col]
    lines.append(f"- Pipeline de licencias de 1 año vendidas o renovadas desde el mes en curso (vence desde "
                 f"{boundaries['current'] + 12}): **aún no se conoce**, se borra y se proyecta: **{int((wiped_pipeline != 0).sum()):,} "
                 f"filas**, ${wiped_pipeline.sum():,.0f} (el raw la conserva en s0_vencen_*).")
    if configuration.discount_value_column:
        unknown = (fine[configuration.discount_bucket_column] == UNKNOWN_DISCOUNT_BUCKET).mean()
        lines.append(f"- Descuento: tramo derivado del descuento exacto con los cortes {configuration.discount_bucket_edges} %; "
                     f"**{unknown:.1%}** de las filas sin dato (tramo `{UNKNOWN_DISCOUNT_BUCKET}`).")
    lines.append(f"- Grano: {len(fine):,} filas = {len(fine):,} filas finas distintas (unidad + valores de revalorización + "
                 f"descuento exacto); el dinero se conserva al agregar a {len(results['forecast_units']):,} forecast units.")
    return "\n".join(lines) + "\n"


def chapter_new_rows(results: dict, configuration: Config) -> str:
    """Chapter 2: every row the framework adds or wipes, in one format (step 21), and the series with most gaps."""
    lines = ["## 2 · Las filas que añade (y borra) el framework", "",
             "El forecast no usa el extracto tal cual. Añade o borra filas de cuatro tipos, cada uno por un motivo:", "",
             "- **hueco** (paso 08): una forecast unit sin nada que vencer, añadida DENTRO de la historia de una serie "
             "estimable para que su serie mensual no tenga agujeros (las técnicas leen meses consecutivos: una tendencia, una "
             "estación, una media móvil). Todas sus medidas son 0 y su tasa queda **nula, nunca 0 %**: añade un mes, no dinero.",
             "- **resultado_adelantado_borrado** (paso 02): una renovación ya registrada desde el mes en curso; el mes no "
             "ha terminado, y el forecast la predice.",
             "- **pipeline_parcial_borrada** (paso 02): la pipeline de las licencias de 1 año que vencen 12 meses después "
             "del mes en curso o más tarde. La generan las ventas y renovaciones desde el mes en curso, que solo han empezado "
             "(el mes va por la mitad y los siguientes no han empezado): está a medio crear. Si se dejara, el forecast de "
             "esos meses se calcularía sobre una pipeline a medias; el paso 17 la reconstruye entera.",
             "- **proyectada** y **simulada** (paso 17): las renovaciones esperadas de las licencias de 1 año que vencen en la "
             "ventana de simulación, y la captación simulada; vencen 12 meses después y sustituyen a la pipeline borrada.", ""]
    new_rows = results.get("new_rows")
    if new_rows is not None and len(new_rows):
        totals = totals_of_new_rows(new_rows)
        lines += ["**Por origen:**", "", markdown_table(totals, 0), ""]
        future = new_rows[new_rows["origen"] != "hueco"]
        if len(future):
            lines += ["**Mes a mes, lo borrado y lo creado** (las unidades y el dinero que vencen; `esperado_usd`: lo que el "
                      "forecast espera renovar de las filas creadas):", "", markdown_table(future, 0), ""]
    rate_summary = results["series_rate"]
    with_gaps = rate_summary[rate_summary["huecos"] > 0].sort_values("huecos", ascending=False)
    history_months = rate_summary["meses_historia"].sum()
    lines += [f"**Los huecos:** {int(with_gaps['huecos'].sum()):,} en {len(with_gaps):,} series, el "
              f"{with_gaps['huecos'].sum() / history_months if history_months else 0:.1%} de los meses de historia. "
              f"Las series con más huecos:", "",
              markdown_table(with_gaps.head(TOP_ROWS)[[SERIES_ID_COLUMN, "meses_historia", "huecos", "n_propio", "usd_por_predecir"]], 0)]
    return "\n".join(lines) + "\n"


def totals_of_new_rows(new_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per origin: months, rows, units, USD and USD expected (the same format as step 21)."""
    rows = []
    for origin, block in new_rows.groupby("origen", sort=False):
        rows.append({"origen": origin, "meses": f"{block['mes'].min()}..{block['mes'].max()}", "filas": int(block["filas"].sum()),
                     "unidades": float(block["unidades"].sum()), "usd": float(block["usd"].sum()),
                     "esperado_usd": float(block["esperado_usd"].sum()) if block["esperado_usd"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def exam_error_against_noise(results: dict):
    """The exam error against the binomial noise, by size of the series (contracts due in the month) and for
    the total of the portfolio: what share of the error is noise no prediction can remove."""
    series_exam = results.get("series_exam")
    if not series_exam:
        return None, None
    detail = series_exam["detail"]
    detail = detail[detail["method"].isin(["raw", "framework"])].copy()
    detail["tamano"] = pd.cut(detail["due_units"], [0, 30, 271, np.inf], right=False,
                              labels=["< 30 al mes", "30-270 al mes", "≥ 271 al mes"])
    rows = []
    for (size, method), block in detail.groupby(["tamano", "method"], observed=True):
        rmse, noise = float(np.sqrt((block["err_pp"] ** 2).mean())), float(np.sqrt((block["noise_pp"] ** 2).mean()))
        rows.append({"tamano": str(size), "metodo": method, "predicciones": len(block), "error_pp": rmse, "ruido_pp": noise,
                     "error_vs_ruido": rmse / noise if noise > 0 else np.nan,
                     "parte_del_error_que_es_ruido": min(1.0, noise ** 2 / rmse ** 2) if rmse > 0 else np.nan})
    by_size = pd.DataFrame(rows)
    # the total of the portfolio, every exam month and horizon: its error and its noise, in pp of the rate
    total_rows = []
    for (month, horizon, method), block in detail.groupby(["period", "h", "method"]):
        due = block["due_units"].sum()
        total_rows.append({"metodo": method, "error_pp": 100 * (block["pred_units"].sum() - block["real_units"].sum()) / due,
                           "ruido_pp": 100 * np.sqrt((block["pred_rate"] * (1 - block["pred_rate"]) * block["due_units"]).sum()) / due})
    total = pd.DataFrame(total_rows).groupby("metodo").agg(error_pp=("error_pp", lambda values: float(np.sqrt((values ** 2).mean()))),
                                                           ruido_pp=("ruido_pp", lambda values: float(np.sqrt((values ** 2).mean()))))
    total["error_vs_ruido"] = total["error_pp"] / total["ruido_pp"]
    return by_size, total.reset_index()


def chapter_support(results: dict, configuration: Config, headline: list) -> str:
    card = results["series_estimate"]
    total_usd = card["usd_por_predecir"].sum()
    with_history = card["meses_historia"] > 0

    def bucket(n):
        if n <= 0:
            return "0 · sin historia propia"
        if n < configuration.support_floor:
            return f"1 · < {configuration.support_floor:.0f} (no fiable solo)"
        if n < configuration.own_rate_floor:
            return f"2 · {configuration.support_floor:.0f}-{configuration.own_rate_floor:.0f} (evidencia, sin precisión)"
        return f"3 · ≥ {configuration.own_rate_floor:.0f} (precisa sola: ±5 pp)"

    before = card.assign(soporte=card["n_propio"].map(bucket)).groupby("soporte").agg(
        series=(SERIES_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum")).reset_index()
    before["pct_usd"] = before["usd_por_predecir"] / total_usd if total_usd else 0.0

    before_error = card["error_binomial_pp"].where(with_history)
    after_error = configuration.z * card["se_estimacion_pp"]
    within_before = card.loc[before_error <= PROMISE_PP, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0
    within_after = card.loc[after_error <= PROMISE_PP, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0
    headline.append(("USD con la tasa conocida a ±5 pp: antes → después de la escalera", f"{within_before:.0%} → {within_after:.0%}"))
    comparison = pd.DataFrame([
        {"medida": "error de la tasa (90 %), ponderado por USD", "antes_pp": np.average(before_error.fillna(50), weights=card["usd_por_predecir"] + 1e-9),
         "despues_pp": np.average(after_error.fillna(50), weights=card["usd_por_predecir"] + 1e-9)},
        {"medida": f"% del USD con la tasa conocida a ±{PROMISE_PP:.0f} pp", "antes_pp": within_before * 100, "despues_pp": within_after * 100}])
    levels = results["money_by_level"]
    lines = ["## 3 · El soporte binomial, antes y después de la escalera", "",
             "La tasa de un mes es k renovaciones de n contratos: aunque nada cambie, oscila por azar "
             "(error binomial √(p(1−p)/n)). Con **30** contratos al mes una serie tiene evidencia para prestar; con **271** "
             "su tasa se conoce a ±5 pp y puede ir sola. **Antes**: cada serie con su propio soporte. **Después**: la "
             "escalera junta las series pasada a pasada (signo, extras y dimensiones mandatory en el orden de colapso) hasta "
             "que cada grupo llega a 30; el grupo presta su tasa a sus series y, por debajo de 271, la mezcla con la de una "
             "referencia más amplia por credibilidad (etapa 4). Ver `DOC_escalera.md` y `DOC_modelo_datos.md`.", "",
             "**Antes · el dinero por soporte propio (el dial):**", "", markdown_table(before),
             "**Antes y después · el error con el que se CONOCE la tasa de cada serie** (antes: su error binomial con su "
             "propio soporte; después: el error de la estimación de la escalera). La predicción de un mes concreto conserva "
             "además el ruido de su propio tamaño, que ninguna escalera elimina: está en el nivel de riesgo.", "",
             markdown_table(comparison, 1),
             "**Etapa a etapa · cómo mejora el soporte** (0 raw · 1 signo · 2 extras · 3 colapso = la composición con la que "
             "se predice; agrupando por el id de cada etapa, las unidades que vencen suman lo mismo):", "",
             markdown_table(results["ladder"]["stage_summary"], 2) if results.get("ladder") else "",
             "**Pasada a pasada · el detalle dentro de cada etapa** (cada pasada es un reparto: las unidades que vencen suman lo "
             "mismo en todas; los grupos son menos y más grandes; pct_usd_floor / pct_usd_own_rate: dinero por predecir en "
             "grupos que llegan a 30 / a 271):", "",
             markdown_table(results["ladder"]["summary"], 2) if results.get("ladder") else "",
             "**Después · el dinero por nivel de riesgo** (error_pp: error de predicción del mes siguiente, ponderado por dinero):", "",
             markdown_table(levels)]
    return "\n".join(lines) + "\n"


def chapter_dynamics(results: dict, configuration: Config) -> str:
    portfolio = results.get("portfolio_dynamics")
    profile = results.get("portfolio_profile")
    dynamics = results.get("pool_dynamics")
    if portfolio is None:
        return "## 4 · La dinámica de la tasa\n\n_(paso 13 no ejecutado)_\n"
    verdict = "**hay efecto mes** más allá del ruido" if portfolio["estacional"] else "**no hay efecto mes** más allá del ruido"
    lines = ["## 4 · La dinámica de la tasa: ¿hay algo más que ruido?", "",
             "φ compara lo que varía la tasa mes a mes con lo que variaría solo por muestreo: φ ≈ 1, nada que modelar "
             "(la media es la mejor técnica); φ > 1, algo la mueve (tendencia, estación, cambio de nivel o de mezcla). La "
             "estacionalidad se prueba sobre la tasa sin su tendencia (prueba F del mes del año, 5 %). Es descriptivo: no "
             "restringe ninguna técnica; el backtest decide.", "",
             f"**Cartera completa:** {verdict} (p = {portfolio['p_valor_mes']:.3f}, amplitud {portfolio['amplitud_pp']:.1f} pp, "
             f"consistencia entre mitades {portfolio['consistencia']:.2f}); tendencia {portfolio['tendencia_pp_ano']:+.1f} pp/año "
             f"(p = {portfolio['p_valor_tendencia']:.3f}); φ {portfolio['phi']:.1f}. Un φ de cartera alto con tendencia suele "
             f"ser cambio de mezcla, no comportamiento.", "",
             markdown_table(profile)]
    if dynamics is not None and len(dynamics):
        lines += ["**Los pools con soporte, uno a uno:**", "",
                  markdown_table(dynamics.sort_values("usd_por_predecir", ascending=False).head(TOP_ROWS)
                                 [[COMPOSITION_ID_COLUMN, "meses", "phi", "tendencia_pp_ano", "estacional", "amplitud_pp",
                                   "meses_alto", "meses_bajo", "usd_por_predecir"]])]
    return "\n".join(lines) + "\n"


def chapter_precision(results: dict, configuration: Config, headline: list) -> str:
    backtest = results.get("backtest")
    if backtest is None:
        return "## 5 · Qué tal se predice la tasa\n\n_(paso 14 no ejecutado)_\n"
    predictions, decision, exam_by_pool, exam_total = (backtest["predictions"], backtest["decision"],
                                                       backtest["exam_by_pool"], backtest["exam_total"])
    boundaries = configuration.calendar_boundaries()
    reference = results["pool_reference"]
    judged_share = (reference.loc[reference["gate"] == GATE_LEVEL, "usd_por_predecir"].sum()
                    / max(reference["usd_por_predecir"].sum(), 1))

    exam_by_band = exam_by_pool.merge(reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
    band_rows = []
    for band_name, rows in exam_by_band.groupby("tramo_h"):
        weights = rows["usd_por_predecir"] + 1e-9
        band_rows.append({"tramo": band_name, "pools": len(rows),
                          "error_elegida_pp": np.average(rows["elegida_err_pp_medio"], weights=weights),
                          "error_retador_pp": np.average(rows["retador_err_pp_medio"], weights=weights),
                          "sesgo_elegida_pp": np.average(rows["elegida_sesgo_pp"], weights=weights),
                          "dentro_banda": rows["dentro_banda"].mean()})
    by_band = pd.DataFrame(band_rows)

    card = results["series_estimate"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "nivel_riesgo", "usd_por_predecir"]]
    short = exam_by_pool[exam_by_pool["tramo_h"] == list(configuration.horizon_bands)[0]]
    by_level = card.merge(short[[COMPOSITION_ID_COLUMN, "elegida_err_pp_medio", "retador_err_pp_medio"]], on=COMPOSITION_ID_COLUMN, how="left")
    by_level = (by_level.dropna(subset=["elegida_err_pp_medio"]).groupby("nivel_riesgo")
                .apply(lambda rows: pd.Series({"series": int(len(rows)), "usd_por_predecir": rows["usd_por_predecir"].sum(),
                                               "error_elegida_pp": np.average(rows["elegida_err_pp_medio"], weights=rows["usd_por_predecir"] + 1e-9),
                                               "error_retador_pp": np.average(rows["retador_err_pp_medio"], weights=rows["usd_por_predecir"] + 1e-9)}),
                       include_groups=False).reset_index())
    if len(by_level):
        by_level["series"] = by_level["series"].astype(int)

    portfolio_summary = results.get("portfolio_exam_summary")
    if portfolio_summary is not None:
        for horizon, block in portfolio_summary.groupby("h"):
            framework = block[block["metodo"] == METHOD_FRAMEWORK].iloc[0]
            spreadsheet = block[block["metodo"].str.startswith("hoja_")].sort_values("error_total_medio").iloc[0]
            alone = block[block["metodo"] == "raw"]
            headline.append((f"error del TOTAL en el examen, h = {horizon}: framework vs mejor hoja de cálculo vs serie sola",
                             f"{framework['error_total_medio']:.1%} vs {spreadsheet['error_total_medio']:.1%} ({spreadsheet['metodo']})"
                             + (f" vs {alone['error_total_medio'].iloc[0]:.1%}" if len(alone) else "")))
    if len(by_band):
        first = by_band.iloc[0]
        headline.append((f"error medio de la tasa por pool en el examen ({first['tramo']})",
                         f"{first['error_elegida_pp']:.1f} pp (retador {first['error_retador_pp']:.1f} pp)"))
    choices = (decision.groupby(["tramo_h", "tecnica_origen"]).size().rename("ids").reset_index())
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    ranking = (selection.assign(abs_norm=selection["err_norm"].abs()).groupby(["tramo_h", "tecnica"])["abs_norm"].mean()
               .rename("err_norm_medio").reset_index().sort_values(["tramo_h", "err_norm_medio"]))
    lines = ["## 5 · Qué tal se predice la tasa de renovación", "",
             "**Cómo se mide.** Cada pool con soporte se predice en meses que ya ocurrieron, sin mirar el futuro: para el mes T "
             "a horizonte h, cada técnica solo ve hasta T − h. Los meses de **selección** (los "
             f"{configuration.backtest_selection_months} anteriores al examen) eligen la técnica; los meses de **examen** "
             f"({boundaries['test_start']} a {boundaries['current'] - 1}) la miden sin que la haya visto. El error se "
             "compara con el ruido binomial del mes (err_norm ≈ 1: tan cerca como permite el azar). Una técnica sustituye al "
             f"retador ({configuration.challenger_technique}) solo si le gana por un margen. Compiten todas las técnicas que "
             "la historia permite, también las de series temporales.", "",
             f"Los pools juzgados cubren el **{judged_share:.0%}** del dinero por predecir; el resto toma el retador.", "",
             "**Ranking en los meses de selección** (error normalizado medio):", "", markdown_table(ranking),
             "**Elecciones:**", "", markdown_table(choices, 0),
             "**Precisión en el examen por tramo** (error medio de la tasa en pp, ponderado por dinero; dentro_banda: "
             "proporción de errores dentro de la banda del 90 %):", "", markdown_table(by_band),
             "**Precisión en el examen por nivel de riesgo** (tramo corto):", "", markdown_table(by_level),
             "**La cartera en el examen, serie a serie: el framework frente a la serie sola (raw) y a la hoja de cálculo** "
             "(paso 19; cada serie predicha como la predice el forecast, con solo lo que se sabía h meses antes; raw: la serie "
             "con su propia historia, sin escalera; la hoja: la tasa de los últimos "
             f"{configuration.baseline_months} meses por grano × la pipeline real; error_total: de la suma de la cartera; "
             "wape_series: serie a serie, sin compensaciones):", "",
             markdown_table(results["portfolio_exam_summary"], 3) if results.get("portfolio_exam_summary") is not None else "",
             markdown_table(results["portfolio_exam"], 3) if results.get("portfolio_exam") is not None else ""]
    by_size, total = exam_error_against_noise(results)
    if by_size is not None and len(by_size):
        lines += ["", "**El error frente al ruido** (la regla del ruido: una diferencia menor que el ruido no es una diferencia; "
                  "un error del tamaño del ruido no es un fallo, es el límite). `ruido_pp`: lo que se equivocaría una predicción "
                  "perfecta, √(p(1−p)/n). `error_vs_ruido` ≈ 1: al límite; claramente mayor que 1: falta algo que se podía saber. "
                  "`parte_del_error_que_es_ruido`: la parte del error que ninguna predicción puede quitar. Por tamaño de la serie:", "",
                  markdown_table(by_size.assign(parte_del_error_que_es_ruido=by_size["parte_del_error_que_es_ruido"].map("{:.0%}".format)), 2),
                  "", "Y el **total de la cartera** en cada mes de examen, con el mismo cálculo: al juntar todo el volumen, el ruido baja "
                  "con la raíz del tamaño, y lo que queda por encima del ruido es error del modelo:", "",
                  markdown_table(total, 3)]
    precision_by_type = exam_precision_by_series_type(results)
    if precision_by_type is not None:
        overall = precision_by_type.iloc[0]
        headline.append(("examen serie a serie: dentro del intervalo · WAPE, framework frente a la serie sola (raw)",
                         f"{overall['en_intervalo']:.0%} vs {overall['en_intervalo_raw']:.0%} · {overall['wape']:.1%} vs {overall['wape_raw']:.1%}"))
        lines += ["**Cuánto acertamos, forecast serie a forecast serie, y cuánto mejora frente a la serie sola** (paso 19: en cada "
                  "mes de examen y horizonte, el framework —su composición con credibilidad— y la serie sola con su propia historia "
                  "(raw), sobre las mismas filas y su propia pipeline; cada predicción con su intervalo, construido como la banda "
                  "del forecast). Cruzado por el tipo de serie: volatilidad (φ de su propia tasa), tendencia y estacionalidad:", "",
                  markdown_table(precision_by_type.assign(
                      predicciones=precision_by_type["predicciones"].astype(int),
                      en_intervalo=precision_by_type["en_intervalo"].map("{:.0%}".format),
                      en_intervalo_raw=precision_by_type["en_intervalo_raw"].map("{:.0%}".format),
                      wape=precision_by_type["wape"].map("{:.1%}".format),
                      wape_raw=precision_by_type["wape_raw"].map("{:.1%}".format),
                      sesgo=precision_by_type["sesgo"].map("{:+.1%}".format)), 0)]
    return "\n".join(lines) + "\n"


def exam_precision_by_series_type(results: dict):
    """The exam of every forecast series summed by type of series (all, by volatility, trend and seasonality):
    the framework against the series alone (raw), counts and units added, then the ratios."""
    audit, series_exam = results.get("audit"), results.get("series_exam")
    if not audit or not series_exam:
        return None
    exam = series_exam["per_series"].merge(audit["series_dynamics"], on=SERIES_ID_COLUMN, how="left")
    exam = exam[exam["framework_predictions"] > 0]
    if exam.empty:
        return None
    measured = exam["measurable"] == "yes"
    segments = [("todas las series examinadas", exam.index == exam.index),
                ("volatilidad baja (φ ≤ 1,5)", exam["phi"] <= 1.5),
                ("volatilidad alta (φ > 1,5)", exam["phi"] > 1.5),
                ("con tendencia (medible)", measured & exam["trend"].fillna(0).ne(0)),
                ("sin tendencia (medible)", measured & exam["trend"].fillna(0).eq(0)),
                ("estacional (medible)", measured & exam["seasonal"].eq(1)),
                ("no estacional (medible)", measured & exam["seasonal"].eq(0)),
                ("sin dinámica medible / sin composición", ~measured)]
    rows = []
    for name, mask in segments:
        block = exam[mask]
        if block.empty:
            continue
        rows.append({"segmento": name, "series": len(block), "predicciones": block["framework_predictions"].sum(),
                     "en_intervalo": block["framework_in_band"].sum() / block["framework_predictions"].sum(),
                     "en_intervalo_raw": block["raw_in_band"].sum() / block["raw_predictions"].sum(),
                     "wape": block["framework_abs_err_units"].sum() / block["framework_real_units"].sum(),
                     "wape_raw": block["raw_abs_err_units"].sum() / block["raw_real_units"].sum(),
                     "sesgo": block["framework_pred_units"].sum() / block["framework_real_units"].sum() - 1})
    return pd.DataFrame(rows)


def chapter_uplift(results: dict, configuration: Config) -> str:
    cells, check = results.get("uplift_cells"), results.get("contract_check")
    comparison, verdict = results.get("uplift_backtest"), results.get("uplift_verdict")
    if cells is None:
        return "## 6 · La revalorización\n\n_(pasos 15-16 no ejecutados)_\n"
    by_origin = cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), uplift_medio=("uplift", "mean"),
                                                   renovadores=("renovadores", "sum")).reset_index()
    lines = ["## 6 · La revalorización: a qué precio se renueva", "",
             "El uplift es lo que paga quien renueva respecto a lo que vencía (1,00 = mismo precio). Dos vías: la "
             "**estadística** (el uplift observado en las renovaciones pasadas de su celda de uplift: dimensiones mandatory, "
             "extras de revalorización y tramo de descuento; con menos de "
             f"{configuration.uplift_floor:.0f} renovadores toma el de su padre) y la del **contrato** (quien pagó con "
             "descuento d renueva a lista: 1 / (1 − d)).", "",
             "**Las celdas por origen de su uplift:**", "", markdown_table(by_origin),
             "**La regla de contrato frente a las renovaciones pasadas con descuento conocido** (ratio_realizacion 1 = exacta):", "",
             markdown_table(check, 3),
             f"**El backtest del uplift** (meses de examen; precio de las renovaciones reales predicho por cada vía, con el "
             f"uplift estadístico estimado ANTES del examen): la vía usada donde hay descuento es la "
             f"**{verdict['via_usada_con_descuento']}**.", "", markdown_table(comparison, 3)]
    return "\n".join(lines) + "\n"


def chapter_forecast(results: dict, configuration: Config, headline: list) -> str:
    forecast = results.get("forecast")
    if forecast is None:
        return "## 7 · El forecast en dinero\n\n_(paso 17 no ejecutado)_\n"
    rows, by_month, by_year = forecast["forecast"], forecast["by_month"], forecast["by_year"]
    validation = results.get("validation")
    for _, year_row in by_year[by_year["esperado_usd"] > 0].iterrows():
        headline.append((f"renovado {int(year_row['ano'])}: real + esperado (± cuadratura)",
                         f"${year_row['total_usd']:,.0f} ± ${year_row['banda_cuadratura_usd']:,.0f}"))
    origins = rows.groupby("origen_tasa").agg(filas=("esperado_usd", "size"), usd_vence=(configuration.pipeline_usd_col, "sum"),
                                              esperado_usd=("esperado_usd", "sum")).reset_index()
    lines = ["## 7 · El forecast en dinero", "",
             "Cada fila futura: **USD que vence × tasa de su serie × uplift de su celda**. La tasa la predice la técnica "
             "elegida para el id de estimación de la serie a su horizonte (aprendiendo de todos los meses cerrados); una serie "
             "que toma prestado conserva su diferencia de nivel con el pool en proporción a su credibilidad"
             + (" (activado)" if configuration.apply_credibility_shift else " (desactivado)") + ". Sin id de estimación: la "
             "tasa de su celda mandatory. Las bandas: **lineal** (todos los errores en el mismo sentido, el peor caso) y "
             "**cuadratura** (errores independientes); la verdad está entre ambas.", "",
             "**La ventana de simulación** (del mes en curso, incluido, a diciembre): lo que ocurre en ella vence doce "
             "meses después. Las licencias de 1 año que vencen en la ventana renuevan como prevé el forecast y su "
             "renovación vence el mismo mes del año siguiente, con las mismas dimensiones y descuento 0 (**proyectada**). "
             "La captación de la ventana se simula por cada valor de captación (mismo mes del año anterior × nivel, valor "
             f"medio de 12 meses, señales a 0, descuento {configuration.acquisition_discount:.0%}; **simulada**). La pipeline "
             "de las licencias de 1 año vendidas o renovadas desde el mes en curso aún no se conoce: el paso 02 la borra "
             "(el raw la conserva en s0_vencen_*) y se proyecta.", "",
             "**Por mes:**", "", markdown_table(by_month, 0),
             "**Por año** (lo renovado en los meses cerrados + lo esperado en los futuros):", "", markdown_table(by_year, 0),
             "**De dónde sale la tasa de las filas futuras:**", "", markdown_table(origins, 0)]
    total = results.get("forecast_total")
    if total is not None and len(total):
        for _, row in total[total["origen"] == TOTAL_ORIGIN_TOTAL].iterrows():
            headline.append((f"TOTAL {int(row['ano'])} renovado + revenue time_series (pipeline {row['usd_vence']:,.0f} $)",
                             f"${row['usd_renovado']:,.0f}"))
        lines += ["**El total del forecast por año y origen** (paso 20; es la SUMA de `sff_nucleo` por `fin_ano` y `fin_origen`, "
                  "comprobado en el núcleo: en Power BI, SUM(fin_renovado_usd) y SUM(fin_vence_usd)). Orígenes: renovaciones ya contabilizadas y "
                  "esperadas de la pipeline real, reentradas y captación del horizonte extendido, y el universo time_series "
                  "de retail a suscripción (ts_real y ts_proyectado cuentan como revenue del año, sin tasa; ts_reentrada es "
                  "pipeline del año siguiente: comprado con descuento, renueva al 100 % con la tasa de su región). "
                  "Total 2026 = renovaciones de la pipeline + ts_real + ts_proyectado · Total 2027 = forecast extendido + "
                  "ts_reentrada. usd_vence: pipeline; usd_renovado: renovaciones o revenue:", "", markdown_table(total, 0)]
        time_series = results.get("time_series")
        if time_series is not None and len(time_series):
            lines += ["**El universo time_series, región × mes:**", "", markdown_table(time_series, 2)]
    if validation is not None:
        lines += ["**Validación final de la cadena:**", "", markdown_table(validation)]
    return "\n".join(lines) + "\n"


def chapter_card_and_next(card: pd.DataFrame) -> str:
    lines = ["## 8 · La ficha de cada serie y lo que falta", "",
             f"`sff_ficha_serie` tiene una fila por serie ({len(card):,}) con todo lo que el framework sabe de ella: ruta, "
             "soporte y tasa propios, grupo final y su referencia, credibilidad, tasa estimada y sus dos errores, nivel de riesgo, la dinámica y "
             "las técnicas de su id de estimación y su error en el examen. `sff_nucleo` tiene la misma información fila a fila "
             "con los meses (en Power BI: seleccionar `s03_fs_id`).", "",
             "**Lo que falta** (siguientes pasos): el horizonte extendido (reentradas de 2026 y captación simulada de 2027, "
             "más allá de la pipeline que trae el extracto), y los análisis del bloque B (composición y mix, descuento y churn, "
             "maduración de las señales, escenarios de precio, baseline de la hoja de cálculo, top movers)."]
    return "\n".join(lines) + "\n"
