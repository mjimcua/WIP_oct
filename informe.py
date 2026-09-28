"""informe.py — SFF v3 · the report: the story the framework tells, in Spanish, with figures.

`run_analysis` ends by writing `<outdir>/informe/informe.md` and its figures. The console
is the technical log; the report is the product a director opens.

The order is the order of TELLING, not of computing (pyramid: the answer first):

  Portada      three numbers, three decisions
  1 Pipeline   what falls due and how much of it is firm (real / projected / simulated)
  2 Ruido      how much of the money can be predicted well (the dial, the risk levels)
  3 Composición  does the rate move by behaviour or by the mix of the pipeline
  4 Estación   is there a season in the rate (the benchmark's decision)
  5 Precisión  how wrong we are, how much of it is chance, and the band we promise
  6 Riesgo     what will change before expiry (signals maturing, top movers)
  7 Hoja       is it better than the spreadsheet
  Anexo        the sheets of the biggest series, the legend of every metric, the
               product tables, the declared weaknesses

Three rules for every number in the report: a unit, a reference (the floor, the % of
the total, the last year) and a reading. The legend of a metric is printed the first
time it appears. Every figure has a title that STATES the finding.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config, TABLE_KIND
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
FIGURE_WIDTH = 12
PALETTE = ["#2E7D7D", "#E8B84F", "#3B8BC8", "#C8553D", "#7B6DAB", "#5A9E5A", "#999999"]
COLOUR_ORIGIN = {PIPELINE_REAL: "#2E7D7D", PIPELINE_PROJECTED: "#3B8BC8", PIPELINE_SIMULATED: "#E8B84F"}
PROMISED_BAND_PCT = 5.0            # the promise to the business: ±5 % where possible
COMPOSITION_HIGH_PCT = 30.0        # above this the aggregate rate is mostly mix
MONTH_LETTERS = "EFMAMJJASOND"
BAND_Z_TEXT = "1,96 errores binomiales (95 %)"

# ─── the legend of the metrics (printed the first time each one appears) ─────────
LEGEND = {
    "error binomial": dict(unidad="pp", definicion="lo que la tasa de un mes oscila por puro azar de muestreo: 100·√(p(1−p)/n)",
                           referencia="30 contratos → ±15 pp · 271 → ±5 pp · 752 → ±3 pp (al 90 %)",
                           mucho="más de 15 pp: menos de 30 contratos, la tasa es una moneda", poco="menos de 3 pp: más de 752 contratos"),
    "nivel de riesgo": dict(unidad="etiqueta", definicion="de dónde toma cada serie su tasa: A propia (≥ 271 contratos), A3 propia reforzada con su pool (30-271), B/C prestada, S con señal y sin pool, D sin historia",
                            referencia="el dinero en A+A3 es el que se predice con precisión propia", mucho="más del 80 % del dinero en A/A3", poco="menos del 50 %: el forecast es mayoritariamente un préstamo"),
    "composición": dict(unidad="%", definicion="parte del movimiento mensual de la tasa de una celda que se debe a que cambian los pesos de quién vence, no a que la gente renueve distinto (descomposición de Kitagawa)",
                        referencia=f"por encima del {COMPOSITION_HIGH_PCT:.0f} % la tasa agregada es sobre todo mezcla de cartera", mucho="> 30 %: no mirar la tasa agregada, mirar las series", poco="< 10 %: el agregado refleja comportamiento"),
    "φ": dict(unidad="ratio", definicion="varianza observada de la tasa mensual (tras quitar la tendencia) dividida por la varianza que el azar de muestreo produciría",
              referencia="1 = la tasa solo muestrea", mucho="> 3: algo real la mueve (cambios de nivel, campañas)", poco="≈ 1: no hay nada que seguir más allá del nivel"),
    "amplitud estacional": dict(unidad="pp", definicion="efecto del mes más alto menos el del más bajo, en una regresión con dummies de mes ponderada por n",
                                referencia="umbral de materialidad 2 pp; el mes extremo debe salirse del ruido (|z| ≥ 1) y repetirse ≥ 2 de 3 años", mucho="> 5 pp y consistente: estación real", poco="< 2 pp: dentro del ruido"),
    "unidad binomial": dict(unidad="×suelo", definicion="error de una técnica dividido por el error binomial del mes que predice",
                            referencia="1,0 = el suelo que ningún método baja", mucho="> 3: la técnica no sigue el nivel", poco="1-1,5: cerca del suelo"),
    "banda total": dict(unidad="% del esperado", definicion="intervalo del 90 %: la parte idiosincrática (cada pool falla por su cuenta, se compensa en cuadratura) combinada con la común (toda la cartera falla a la vez, medida como el error del total)",
                        referencia=f"la promesa: ±{PROMISED_BAND_PCT:.0f} %", mucho="> ±8 %: hay que decirlo", poco="< ±2 %: revisar la parte común"),
    "sesgo": dict(unidad="pp", definicion="media con signo del error en el examen: + sobreestimamos, − infraestimamos",
                  referencia="0", mucho="|sesgo| mayor que el error binomial: hay un desplazamiento sistemático", poco="dentro del ruido"),
    "maduración pendiente": dict(unidad="pp de la celda", definicion="proporción final de una señal (últimos 12 meses cerrados) menos la que la foto de hoy muestra para un mes futuro",
                                 referencia="0 a un mes vista; crece con la distancia", mucho="> 5 pp: mucho dinero cambiará de serie antes de vencer", poco="0"),
    "tasa estandarizada": dict(unidad="%", definicion="la tasa de un tramo de descuento aplicada al mismo reparto de celdas (regiones, productos…) que el resto de tramos: la diferencia entre tramos es comportamiento, no quién está en cada tramo",
                              referencia="su diferencia con la tasa bruta es el efecto de composición", mucho="huecos entre tramos > 5 pp con z > 3", poco="huecos dentro del ruido (|z| < 2)"),
    "importancia relativa": dict(unidad="R²", definicion="parte de la varianza de las tasas de renovación (entre celdas × estado × descuento × cliente nuevo) que explica un grupo de variables solo, y la que se pierde al quitarlo del modelo completo",
                                 referencia="comparar señales, celda y descuento entre sí", mucho="> 0,3", poco="< 0,05: esa variable apenas separa"),
    "tasa a composición fija": dict(unidad="%", definicion="la tasa de cada celda del mes ponderada con las proporciones del año de referencia: lo que la tasa agregada habría sido si la cartera no hubiera cambiado de mezcla",
                                    referencia="observada − fija = efecto mix", mucho="diferencias > 2 pp: la tasa agregada se mueve por cartera", poco="< 0,5 pp"),
    "efecto composición": dict(unidad="pp y $", definicion="parte del cambio de la tasa agregada entre dos periodos debida a que cambian los pesos de las celdas (Kitagawa): Σ (w1 − w0)·(r0 + r1)/2; el resto es comportamiento",
                               referencia="% del cambio por composición; 4 niveles: región → celda mandatory → + quién es el cliente → + cómo está", mucho="> 50 %: la cartera cambia más que la gente", poco="< 20 %"),
    "punto de equilibrio": dict(unidad="% de tasa", definicion="la tasa por debajo de la cual una subida de precio pierde dinero: r × precio_base / precio_escenario",
                                referencia="comparar con la tasa esperada tras el delta", mucho="margen > 3 pp: la subida es segura", poco="margen < 1 pp: la subida está al límite"),
    "WAPE": dict(unidad="% de $", definicion="Σ|previsto − real| / Σ real sobre las renovaciones reales del examen; el sesgo es la versión con signo",
                 referencia="camino A (todo estadístico) frente a B (regla donde hay descuento)", mucho="> 10 %", poco="< 3 %"),
    "dinero disputable": dict(unidad="$", definicion="lo que las series con señal negativa renuevan por debajo de los neutros de su celda × su pipeline: cota superior de lo que una recuperación puede conseguir",
                              referencia="comparar con el forecast del mismo periodo", mucho="> 5 % del forecast", poco="< 1 %"),
}


def legend_table(configuration: Config) -> pd.DataFrame:
    rows = [dict(metrica=name, **fields) for name, fields in LEGEND.items()]
    table = pd.DataFrame(rows)
    configuration.write(table, "metric_legend")
    return table


class Report:
    """Collects markdown and figures; prints the legend of a metric the first time it is used."""

    def __init__(self, folder: str):
        self.folder = folder
        self.lines = []
        self.figures = []
        self.legends_done = set()
        os.makedirs(folder, exist_ok=True)

    def h(self, level: int, text: str) -> None:
        self.lines.append(f"\n{'#' * level} {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(text + "\n")

    def legend(self, *names: str) -> None:
        for name in names:
            if name in self.legends_done or name not in LEGEND:
                continue
            self.legends_done.add(name)
            f = LEGEND[name]
            self.lines.append(f"> **{name}** ({f['unidad']}): {f['definicion']}. Referencia: {f['referencia']}. Mucho: {f['mucho']}. Poco: {f['poco']}.\n")

    def figure(self, figure, name: str, caption: str) -> None:
        path = os.path.join(self.folder, f"{name}.png")
        figure.tight_layout()
        figure.savefig(path, dpi=110)
        plt.close(figure)
        self.figures.append(path)
        self.lines.append(f"![{caption}]({os.path.basename(path)})\n\n*{caption}*\n")

    def table(self, frame: pd.DataFrame, columns: list = None, floatfmt: dict = None, max_rows: int = 12) -> None:
        if frame is None or len(frame) == 0:
            self.lines.append("*(sin filas)*\n")
            return
        show = frame[columns] if columns else frame
        show = show.head(max_rows).copy()
        for column in show.columns:
            if show[column].dtype.kind == "f":
                fmt = (floatfmt or {}).get(column, "{:,.2f}")
                show[column] = show[column].map(lambda v: fmt.format(v) if pd.notna(v) else "")
        header = "| " + " | ".join(str(c) for c in show.columns) + " |"
        rule = "|" + "|".join("---" for _ in show.columns) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False))
        self.lines.append(f"{header}\n{rule}\n{body}\n")

    def write(self) -> str:
        path = os.path.join(self.folder, "informe.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines))
        return path


def money(value: float) -> str:
    """$ in millions with one decimal, or thousands when small."""
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.0f}k"
    return f"${value:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{decimals}f} %"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHAPTERS
# ═══════════════════════════════════════════════════════════════════════════════════

def chapter_cover(report: Report, results: dict, configuration: Config) -> None:
    """Three numbers, three decisions, one figure: the year month by month."""
    business = results["forecast"]["business_summary"]
    summary = results["forecast"]["pipeline_summary"]
    movers = results.get("top_movers", pd.DataFrame())
    alerts = results.get("signal_maturation", {}).get("signal_alerts", pd.DataFrame())
    if business is None or business.empty:
        report.h(1, "Informe del forecast de renovaciones")
        report.p("No hay meses futuros en el extracto: no hay forecast que contar.")
        return
    this_year = business.iloc[0]
    next_year = business.iloc[1] if len(business) > 1 else None
    total_block = summary[summary["bloque"] == "total"].iloc[0] if len(summary) else None
    report.h(1, f"Forecast de renovaciones · {int(this_year['anio'])}-{int(business['anio'].max())}")
    report.p("Este informe responde, en este orden: cuánto vamos a renovar, con qué confianza, dónde está el riesgo y qué hacer. "
             "La evidencia de cada afirmación va después de ella, y cada métrica se explica la primera vez que aparece.")
    report.h(2, "Los tres números")
    booked, forecast_rest, total = this_year["renovado_real_usd"], this_year["forecast_usd"], this_year["total_esperado_usd"]
    adjusted = this_year.get("total_esperado_ajustado_usd", total)
    band_low, band_high = this_year["banda_low_usd"], this_year["banda_high_usd"]
    report.p(f"1. **{int(this_year['anio'])} cierra en {money(total)}** si nada cambia: {money(booked)} ya renovados en {int(this_year['meses_reales'])} meses cerrados "
             f"y {money(forecast_rest)} previstos para los {int(this_year['meses_forecast'])} meses que quedan (incluido el mes pendiente de cierre). "
             f"Con la maduración de las señales que aún no han aparecido, {money(adjusted)}.")
    report.p(f"2. **La banda es {money(band_low)} / +{money(band_high)}** ({pct(100 * band_low / max(total, 1e-9))} / +{pct(100 * band_high / max(total, 1e-9))} del total): "
             f"nueve de cada diez veces el cierre cae dentro si el negocio sigue como está. La promesa era ±{PROMISED_BAND_PCT:.0f} %.")
    if next_year is not None:
        real_share = 100 * next_year["pipeline_real_usd"] / max(next_year["pipeline_total_usd"], 1e-9)
        report.p(f"3. **{int(next_year['anio'])}: {money(next_year['forecast_usd'])} previstos** sobre una pipeline de {money(next_year['pipeline_total_usd'])}, "
                 f"de la que el {pct(real_share, 0)} son contratos que existen hoy; el resto son renovaciones que aún tienen que ocurrir ({money(next_year['pipeline_proyectada_usd'])}) "
                 f"y adquisición al ritmo histórico ({money(next_year['pipeline_simulada_usd'])}).")
    report.legend("banda total")
    report.h(2, "Las tres decisiones del mes")
    decisions = []
    if len(movers):
        worst = movers[movers["tipo"] == "deterioro"].head(1)
        if len(worst):
            row = worst.iloc[0]
            decisions.append(f"**Defender {row['fs_id']}**: la tasa prevista cae {abs(row['delta_pp']):.1f} pp frente a los últimos tres meses cerrados; son {money(abs(row['usd_impacto']))} sobre 12 meses de pipeline.")
        recover = movers[movers["tipo"] == "senal_negativa"].head(1)
        if len(recover):
            row = recover.iloc[0]
            decisions.append(f"**Recuperar en {row['fs_id']}**: sus clientes con señal renuevan {row['delta_pp']:.0f} pp por debajo de los neutros de su celda; {money(row['usd_impacto'])} en disputa (cota superior).")
    if len(alerts):
        level_alerts = alerts[alerts["tipo"] == "nivel_anomalo"].sort_values("z", ascending=False).head(1)
        if len(level_alerts):
            row = level_alerts.iloc[0]
            decisions.append(f"**Vigilar {row['celda_comp']}**: {row['mensaje']} (z = {row['z']}).")
    if total_block is not None:
        decisions.append(f"**Comunicar la banda total, no la idiosincrática**: la parte común (toda la cartera fallando a la vez) es "
                         f"{pct(abs(total_block['pct_banda_comun_low']))} / +{pct(total_block['pct_banda_comun_high'])} y no se compensa entre pools.")
    for decision in decisions[:3] or ["No hay movimientos que destacar este mes."]:
        report.p(f"- {decision}")
    # figure: the year month by month
    horizon = results["forecast"]["horizon_report"]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 5))
    months = horizon["period"].astype(str).tolist() if "period" in horizon.columns else horizon[configuration.period_col].astype(str).tolist()
    expected = horizon["esperado_usd"].to_numpy()
    adjusted_line = horizon["forecast_ajustado_usd"].to_numpy() if "forecast_ajustado_usd" in horizon.columns else expected
    low = -horizon["banda_low_usd"].to_numpy() if "banda_low_usd" in horizon.columns else np.zeros(len(months))
    high = horizon["banda_high_usd"].to_numpy() if "banda_high_usd" in horizon.columns else np.zeros(len(months))
    x = np.arange(len(months))
    axis.bar(x, expected, color=PALETTE[0], alpha=0.85, label="previsto (la foto de hoy)")
    axis.errorbar(x, expected, yerr=[low, high], fmt="none", ecolor="black", capsize=3, label="banda 90 %")
    axis.plot(x, adjusted_line, "o--", color=PALETTE[3], markersize=4, label="previsto con la maduración de las señales")
    axis.set_xticks(x)
    axis.set_xticklabels([MONTH_LETTERS[int(m[5:7]) - 1] + ("\n" + m[:4] if m[5:7] == "01" or m == months[0] else "") for m in months], fontsize=9)
    axis.set_ylabel("renovado esperado, $")
    axis.set_title(f"Mes a mes: {money(expected.sum())} en {len(months)} meses; la banda crece con la distancia y con la pipeline simulada", loc="left", fontsize=12)
    axis.legend(fontsize=9)
    report.figure(figure, "00_portada_mes_a_mes", "Cada barra es un mes futuro (real, proyectada y simulada juntas); la línea punteada es el forecast una vez que las señales de churn que faltan por aparecer hayan aparecido.")


def chapter_panorama(report: Report, results: dict, configuration: Config) -> None:
    """0 · The whole portfolio before any model."""
    panorama = results.get("panorama", {})
    if not panorama:
        return
    monthly, markets, diagnosis, volume_ts = panorama["panorama_mensual"], panorama["panorama_mercado"], panorama["panorama_diagnostico"], panorama["panorama_volumen_ts"]
    period = configuration.period_col
    report.h(1, "0 · La cartera entera, antes de cualquier modelo")
    report.legend("tasa a composición fija")
    total_rate = diagnosis[(diagnosis["ambito"] == "total") & (diagnosis["serie"] == "tasa")].iloc[0]
    total_pipe = diagnosis[(diagnosis["ambito"] == "total") & (diagnosis["serie"] == "pipeline")].iloc[0]
    mix = monthly["efecto_mix_pp"].dropna()
    report.p(f"La **pipeline** (cuánto vence cada mes) es {total_pipe['veredicto']}: amplitud estacional del {total_pipe['amplitud']} % de su media, tendencia {total_pipe['tendencia_anual']:+.1f} %/año. "
             f"La **tasa de renovación**, que es lo que predecimos, es {total_rate['veredicto']}: amplitud {total_rate['amplitud']} pp, tendencia {total_rate['tendencia_anual']:+.1f} pp/año, "
             f"{int(total_rate['meses_fuera_banda']) if pd.notna(total_rate['meses_fuera_banda']) else 0} meses fuera de su banda binomial de ±{BAND_Z_TEXT}. "
             f"La mezcla de cartera mueve la tasa observada hasta {mix.abs().max():.1f} pp respecto a la tasa a composición fija (año de referencia {panorama['anio_referencia']}).")
    if len(markets):
        moved = markets[markets["cartera_movida"] == 1]["mercado"].tolist()
        report.p(f"Mercados: {len(markets)}; con cartera movida (picos de vencimientos sin pico de entradas un año antes): {', '.join(moved) if moved else 'ninguno'}.")
    # figure A: pipeline with shifted entries
    closed = monthly[monthly["rol"].isin(("entrenamiento", "examen"))].sort_values(period)
    if len(closed):
        figure, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 8), sharex=True)
        x = np.arange(len(closed))
        axes[0].bar(x, closed["unidades"], color=PALETTE[2], alpha=0.8, label="pipeline (unidades que vencen)")
        axes[0].plot(x, closed["entradas_hace_12m"], "-o", color=PALETTE[3], markersize=3, label="entradas de hace 12 meses (renovaciones desplazadas)")
        axes[0].set_title(f"La pipeline es {total_pipe['veredicto']}: cuánto vence cada mes viene de lo que entró un año antes", loc="left", fontsize=11)
        axes[0].legend(fontsize=9)
        axes[1].plot(x, 100 * closed["tasa"], "-o", color="black", markersize=3, label="tasa observada")
        axes[1].fill_between(x, 100 * closed["banda_low"], 100 * closed["banda_high"], color="grey", alpha=0.25, label="banda binomial del agregado")
        axes[1].plot(x, 100 * closed["tasa_composicion_fija"], "--", color=PALETTE[0], linewidth=2, label="tasa a composición fija")
        axes[1].set_title(f"La tasa es {total_rate['veredicto']}: lo que se mueve por encima de la línea punteada es cartera, no comportamiento", loc="left", fontsize=11)
        axes[1].set_xticks(x[::3]); axes[1].set_xticklabels([str(p)[2:] for p in closed[period].iloc[::3]], fontsize=8, rotation=45)
        axes[1].set_ylabel("%"); axes[1].legend(fontsize=9)
        report.figure(figure, "000_panorama", "Arriba: la pipeline mensual y las entradas de doce meses antes. Abajo: la tasa observada, su banda binomial (muy estrecha con el soporte agregado) y la tasa a composición fija.")
    if len(volume_ts):
        pivot = volume_ts.pivot_table(index="ambito", columns="tecnica", values="error_medio_pct")
        report.p("**Dónde sí funcionan las series temporales:** el volumen. Error medio a 3 meses vista sobre los últimos 12 meses cerrados: " +
                 "; ".join(f"{a}: " + ", ".join(f"{t} {v:.1f} %" for t, v in row.dropna().items()) for a, row in pivot.iterrows()) +
                 ". Sobre la tasa, la decisión es la del capítulo 4.")
    report.table(markets, [c for c in markets.columns if c in ("mercado", "pct_usd_cartera", "variacion_ultimo_ano_pp", "amplitud_estacional_tasa_pp", "meses_fuera_banda", "cartera_movida") or c.startswith("tasa_")])
    report.p("**Y entonces:** la estacionalidad que negocio ve es de cuántos contratos vencen; predecir la tasa no necesita una onda anual salvo que el capítulo 4 la demuestre; y la parte del movimiento de la tasa que es mezcla ya la absorbe un forecast por segmento.")


def chapter_composition_effect(report: Report, results: dict, configuration: Config) -> None:
    """The composition effect at four levels, historical and future."""
    history, future = results.get("composition_history", {}), results.get("composition_future", {})
    effect = pd.concat([history.get("efecto_composicion", pd.DataFrame()).assign(momento="histórico"), future.get("efecto_composicion", pd.DataFrame()).assign(momento="futuro")], ignore_index=True)
    if effect.empty:
        return
    report.h(1, "3b · Efecto composición: cuánto es cartera y cuánto es gente")
    report.legend("efecto composición")
    total = effect[effect["ambito"] == "total"]
    headlines = pd.concat([history.get("titulares", pd.DataFrame()), future.get("titulares", pd.DataFrame())], ignore_index=True)
    for label, block in total.groupby("comparacion", sort=False):
        first = block.iloc[0]
        levels = " → ".join(f"{r['nivel'].replace('mandatory', 'celda')}: {r['pct_composicion']:.0f} %" for _, r in block.iterrows() if pd.notna(r["pct_composicion"]))
        report.p(f"**{label}**: la tasa pasa de {100 * first['tasa_0']:.1f} % a {100 * first['tasa_1']:.1f} % ({first['cambio_total_pp']:+.1f} pp). Parte que es composición según el nivel de segmentación: {levels}.")
    if len(headlines):
        for _, h in headlines.iterrows():
            report.p(f"- Cifra titular {h['comparacion']}: conocer quién es el cliente y cómo está mueve **{money(abs(h['aporta_conocer_cliente_usd']))}** de 'comportamiento' a 'composición': lo que un forecast por celda mandatory habría errado solo por tipología y estado del cliente.")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    comparisons = list(total["comparacion"].unique())
    levels = ["regional", "mandatory", "mandatory+quien", "mandatory+quien+como"]
    width = 0.8 / max(len(comparisons), 1)
    for index, label in enumerate(comparisons):
        block = total[total["comparacion"] == label].set_index("nivel").reindex(levels)
        axis.bar(np.arange(len(levels)) + (index - (len(comparisons) - 1) / 2) * width, block["pct_composicion"], width=width, color=PALETTE[index % len(PALETTE)], alpha=0.85, label=label)
    axis.set_xticks(np.arange(len(levels))); axis.set_xticklabels(["región", "celda mandatory", "+ quién es", "+ cómo está"], fontsize=10)
    axis.set_ylabel("% del cambio que es composición")
    axis.set_title("Cuanto mejor se conoce al cliente, más del cambio resulta ser cartera y no comportamiento", loc="left", fontsize=11)
    axis.legend(fontsize=9)
    report.figure(figure, "03b_efecto_composicion", "Por comparación de años, el % del cambio de la tasa explicado por composición a cada nivel de segmentación.")
    report.p("**Y entonces:** un forecast por segmento (Σ pipeline_segmento × tasa_segmento) absorbe la composición automáticamente porque la pipeline de cada segmento se conoce por adelantado; la tasa única por celda no.")


def chapter_scenarios(report: Report, results: dict, configuration: Config) -> None:
    """Price and scenarios: the contract rule's backtest and the scenario layer."""
    backtest = results.get("backtest_uplift", pd.DataFrame())
    scenarios = results.get("scenarios", {})
    if (backtest is None or backtest.empty) and not scenarios.get("escenarios_margen", pd.DataFrame()).shape[0]:
        return
    report.h(1, "9 · Precio: la regla de contrato, su examen y los escenarios")
    report.legend("WAPE", "punto de equilibrio")
    if backtest is not None and len(backtest):
        total = backtest[(backtest["ambito"] == "total") & (backtest["autorenew"] == "*")].iloc[0]
        report.p(f"Sobre las renovaciones reales de los meses de examen ({total['pct_usd_descuento_conocido']:.0f} % del dinero con descuento conocido): la revalorización estadística falla {total['wape_estadistico_pct']:.1f} % (sesgo {total['sesgo_estadistico_pct']:+.1f} %); "
                 f"en las filas con descuento, la regla de contrato falla {total['wape_regla_pct']:.1f} % frente a {total['wape_estadistico_conocido_pct']:.1f} % del estadístico. "
                 f"El número final por el camino A (todo estadístico) falla {total['wape_final_A_pct']:.1f} % y por el B (regla donde hay descuento) {total['wape_final_B_pct']:.1f} %: **{total['veredicto'].upper()}** ({total['aportacion_descuento_pct']:+.1f} puntos).")
        report.p("Criterio: si B mejora, el cambio está justificado; si empata, se adopta igual por los escenarios; si empeora, se investiga antes de adoptar (ofertas de renovación, subidas pasadas que la regla no conoce).")
    margin = scenarios.get("escenarios_margen", pd.DataFrame())
    if len(margin):
        without = margin[margin["margen"] == "sin_margen"]
        report.p("Cartera futura **sin margen** de crecimiento por descuento (el descuento ya es 0, o lo será tras la renovación proyectada): " + " · ".join(f"{int(y)}: {p:.0f} %" for y, p in zip(without["anio"], without["pct_usd_anio"])) + ".")
    table = scenarios.get("escenarios_precio", pd.DataFrame())
    if len(table):
        for scenario, block in table.groupby("escenario"):
            report.p(f"Escenario **{scenario}**: {money(block['usd_base'].sum())} → {money(block['usd_escenario'].sum())}: por precio {money(block['usd_por_precio'].sum())}, por retención {money(block['usd_por_retencion'].sum())}, neto {money(block['neto_usd'].sum())}; margen medio hasta el punto de equilibrio {block['margen_pp_hasta_equilibrio'].mean():.1f} pp.")
        report.table(table.groupby(["escenario", "sku", "margen"], as_index=False)[["usd_base", "usd_escenario", "usd_por_precio", "usd_por_retencion", "neto_usd"]].sum(), floatfmt={c: "{:,.0f}" for c in ("usd_base", "usd_escenario", "usd_por_precio", "usd_por_retencion", "neto_usd")})
    report.p("**Y entonces:** la capa de escenarios no recalcula el motor; cambia una tabla de precios por SKU y muestra cuánto mueve el número, con la bajada de tasa que lo compensa y la tasa de equilibrio que no se puede perder.")


def chapter_pipeline(report: Report, results: dict, configuration: Config) -> None:
    horizon = results["forecast"]["horizon_report"]
    business = results["forecast"]["business_summary"]
    report.h(1, "1 · La pipeline: qué vence y cuánto de ello es firme")
    if horizon is None or horizon.empty:
        report.p("Sin meses futuros.")
        return
    period = "period" if "period" in horizon.columns else configuration.period_col
    origins = [c for c in ("pipeline_real_usd", "pipeline_proyectada_usd", "pipeline_simulada_usd") if c in horizon.columns]
    totals = {c: float(horizon[c].sum()) for c in origins}
    grand = sum(totals.values()) or 1.0
    report.p(f"La pipeline del horizonte suma {money(grand)}: **{pct(100 * totals.get('pipeline_real_usd', 0) / grand, 0)} son contratos que existen hoy** (real), "
             f"{pct(100 * totals.get('pipeline_proyectada_usd', 0) / grand, 0)} son renovaciones que estamos prediciendo y que reentrarán un plazo después (proyectada), "
             f"y {pct(100 * totals.get('pipeline_simulada_usd', 0) / grand, 0)} es adquisición futura al ritmo histórico (simulada). "
             "El forecast sobre las dos últimas es un forecast sobre un forecast: cuanto más lejos el mes, menos firme.")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    x = np.arange(len(horizon))
    bottom = np.zeros(len(horizon))
    for column in origins:
        values = horizon[column].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, color=COLOUR_ORIGIN[column.split("_")[1]], alpha=0.85, label=column.split("_")[1])
        bottom = bottom + values
    axis.set_xticks(x)
    axis.set_xticklabels([m[2:] for m in horizon[period].astype(str)], fontsize=8, rotation=45)
    axis.set_ylabel("pipeline $")
    first_simulated = horizon.loc[horizon.get("pipeline_simulada_usd", pd.Series(0, index=horizon.index)) > 0, period]
    axis.set_title("La pipeline real se agota con la distancia: a partir de " + (str(first_simulated.iloc[0]) if len(first_simulated) else "—") + " el forecast se apoya en renovaciones y adquisiciones que aún no han ocurrido", loc="left", fontsize=11)
    axis.legend(fontsize=9)
    report.figure(figure, "01_pipeline_por_origen", "Pipeline por mes y origen. Real = contratos que existen; proyectada = reentradas de renovaciones previstas; simulada = adquisición al ritmo histórico.")
    report.p("**Y entonces:** la pregunta \"¿cómo acaba el año que viene?\" tiene dos respuestas de distinta calidad: la parte sobre pipeline real se predice como este año; la parte proyectada y simulada hereda además la incertidumbre del volumen, que no se mide. "
             "En el informe a negocio se separan.")
    report.table(business, [c for c in ("anio", "renovado_real_usd", "forecast_usd", "forecast_ajustado_usd", "total_esperado_usd", "banda_low_usd", "banda_high_usd",
                                        "pipeline_real_usd", "pipeline_proyectada_usd", "pipeline_simulada_usd") if c in business.columns], floatfmt={c: "{:,.0f}" for c in business.columns})


def chapter_noise(report: Report, results: dict, configuration: Config) -> None:
    profiles = results.get("profiles", {})
    dial = profiles.get("dial_buckets", pd.DataFrame())
    card = results["series_card"]
    levels = results.get("risk_levels", pd.DataFrame())
    report.h(1, "2 · El ruido: qué parte del dinero se puede predecir bien")
    report.legend("error binomial", "nivel de riesgo")
    if len(dial):
        precise = dial[dial["tramo_dial"].isin(["2_271_a_752", "3_desde_752"])]["pct_usd"].sum() if "pct_usd" in dial.columns else np.nan
        under = dial[dial["tramo_dial"] == "0_menos_de_30"]["pct_usd"].sum() if "pct_usd" in dial.columns else np.nan
        report.p(f"Antes de prestar nada, **el {pct(precise, 0)} del dinero proyectado está en series con más de 271 contratos al mes** (±5 pp o mejor por puro muestreo) "
                 f"y el {pct(under, 0)} en series con menos de 30 (±15 pp o peor: una tasa que es una moneda). El resto está entre las dos.")
    trainable = card[card["ruta"] == TREATMENT_PREDICTABLE] if "ruta" in card.columns else card
    by_level = trainable.groupby("nivel_riesgo").agg(series=("fs_id", "count"), usd=("usd_proyectado", "sum"),
                                                     error=("se_prediccion_pp", lambda s: float(np.average(s.fillna(s.mean()), weights=None)))).reset_index()
    by_level["pct"] = 100 * by_level["usd"] / max(by_level["usd"].sum(), 1e-9)
    by_level = by_level.sort_values("usd", ascending=False)
    own = by_level[by_level["nivel_riesgo"].str.startswith("A")]["pct"].sum()
    report.p(f"Después de la escalera, **el {pct(own, 0)} del dinero se predice con precisión propia** (niveles A: la serie tiene ≥ 271 contratos o entre 30 y 271 reforzados con su pool); "
             f"el resto toma prestada la tasa de un pariente (B, C), tiene señal y no encuentra pool (S) o no tiene historia (D).")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    x = np.arange(len(by_level))
    axis.bar(x, by_level["usd"], color=[PALETTE[0] if n.startswith("A") else PALETTE[1] if n[0] in "BC" else PALETTE[3] for n in by_level["nivel_riesgo"]], alpha=0.85)
    for i, (usd, err, share) in enumerate(zip(by_level["usd"], by_level["error"], by_level["pct"])):
        axis.text(i, usd, f"{share:.0f}% · ±{err:.0f} pp", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x)
    axis.set_xticklabels(by_level["nivel_riesgo"], fontsize=9, rotation=20)
    axis.set_ylabel("$ proyectado")
    axis.set_title(f"El {own:.0f} % del dinero se predice con precisión propia; el error de cada nivel es el muestreo de un mes típico", loc="left", fontsize=11)
    report.figure(figure, "02_dinero_por_nivel", "Dinero proyectado por nivel de riesgo, con su % del total y el error de predicción típico (puntos porcentuales).")
    report.p("**Y entonces:** donde el dinero está en B, C o S, invertir en datos (más historia, agrupar mejor) estrecha la banda; donde está en A, el forecast es un compromiso y se gestiona por desviación.")


def chapter_composition(report: Report, results: dict, configuration: Config) -> None:
    composition = results.get("composition", {})
    mix = composition.get("mix_shift_decomposition", pd.DataFrame())
    cost = composition.get("mandatory_only_cost", pd.DataFrame())
    report.h(1, "3 · La composición: ¿se mueve la tasa por comportamiento o por cartera?")
    report.legend("composición")
    if mix is None or mix.empty:
        report.p("Sin descomposición disponible.")
        return
    share = mix["delta_composicion_pp"].abs().sum() / max((mix["delta_composicion_pp"].abs() + mix["delta_comportamiento_pp"].abs()).sum(), 1e-9)
    per_cell = mix.groupby("celda_id").agg(comp=("delta_composicion_pp", lambda s: s.abs().mean()), beh=("delta_comportamiento_pp", lambda s: s.abs().mean())).reset_index()
    per_cell["share"] = per_cell["comp"] / (per_cell["comp"] + per_cell["beh"]).replace(0, np.nan)
    saving = float(cost["ahorro_usd"].sum()) if len(cost) else np.nan
    report.p(f"**El {pct(100 * share, 0)} del movimiento mensual de las tasas de las celdas es composición**, no comportamiento (referencia: por encima del {COMPOSITION_HIGH_PCT:.0f} % la tasa agregada engaña). "
             f"Predecir con las mandatory solas (una tasa por celda) habría costado {money(abs(saving)) if np.isfinite(saving) else '—'} en los últimos meses frente a predecir por series y sumar" + (" (a favor de segmentar)." if np.isfinite(saving) and saving > 0 else "."))
    top = per_cell.sort_values("comp", ascending=False).head(6)
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    x = np.arange(len(top))
    axis.bar(x, top["beh"], color=PALETTE[0], alpha=0.85, label="comportamiento (pp/mes)")
    axis.bar(x, top["comp"], bottom=top["beh"], color=PALETTE[1], alpha=0.85, label="composición (pp/mes)")
    axis.set_xticks(x)
    axis.set_xticklabels([str(c)[:22] for c in top["celda_id"]], fontsize=8, rotation=20)
    axis.set_ylabel("movimiento medio mensual de la tasa, pp")
    axis.set_title(f"En las celdas que más se mueven, la composición explica hasta el {100 * top['share'].max():.0f} % del movimiento", loc="left", fontsize=11)
    axis.legend(fontsize=9)
    report.figure(figure, "03_composicion_por_celda", "Descomposición de Kitagawa del cambio mensual de la tasa: la parte amarilla no es gente cambiando de comportamiento, es que vencen clientes distintos.")
    report.p("**Y entonces:** en las celdas con mucha composición el KPI \"tasa de renovación de la región\" sube y baja sin que nadie haya cambiado; el forecast lo trata bien porque predice por serie y suma con los pesos reales, pero el reporting a negocio debería mirar las series.")


def chapter_season(report: Report, results: dict, configuration: Config) -> None:
    bench = results.get("benchmark", {})
    decision = bench.get("decision_estacionalidad", pd.DataFrame())
    summary = bench.get("summary", {})
    report.h(1, "4 · La estación: ¿hay forma anual en la tasa?")
    report.legend("φ", "amplitud estacional")
    if decision is None or decision.empty:
        report.p("El benchmark no encontró series con soporte suficiente para decidir.")
        return
    seasonal = decision[decision["veredicto_estacional"] == 1]
    report.p(f"Se decidió una vez, donde la prueba tiene potencia: {len(decision)} series grandes y neutras (≥ {configuration.benchmark_min_support:.0f} contratos todos los meses) que llevan el "
             f"{pct(summary.get('cobertura_pct', np.nan), 0)} del pipeline. **Veredicto: {summary.get('veredicto', '—')}.** "
             f"{len(seasonal)} series estacionales ({pct(summary.get('pct_estacional', 0), 0)} del dinero de la muestra); {len(decision[decision['veredicto_tendencia'] != 0])} con tendencia.")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    for index, (_, row) in enumerate(decision.sort_values("usd_proyectado", ascending=False).head(5).iterrows()):
        effects = {int(k): float(v) for k, v in (item.split(":") for item in str(row.get("efectos_mes", "")).split("|") if ":" in item)}
        if not effects:
            continue
        months = sorted(effects)
        axis.plot(months, [effects[m] for m in months], "-o", color=PALETTE[index % len(PALETTE)], linewidth=2,
                  label=f"{row['fs_id'][:34]} · amp {row['amplitud_pp']:.1f} pp · {'ESTACIONAL' if row['veredicto_estacional'] else 'no material'}")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhspan(-1, 1, color="grey", alpha=0.12, label="±1 pp")
    axis.set_xticks(range(1, 13)); axis.set_xticklabels(list(MONTH_LETTERS))
    axis.set_ylabel("efecto del mes, pp")
    axis.set_title(("La estación es real en " + str(len(seasonal)) + " series: efectos de mes que se repiten y baten al nivel reciente") if len(seasonal) else "Los efectos de mes no son materiales: el nivel reciente predice mejor que la forma anual", loc="left", fontsize=11)
    axis.legend(fontsize=8)
    report.figure(figure, "04_efectos_de_mes", "Efectos de mes de las cinco series más grandes del benchmark, tras quitar la tendencia. Solo cuentan si son grandes, consistentes entre años y mejoran la predicción.")
    report.table(decision.sort_values("usd_proyectado", ascending=False), ["fs_id", "n_mediana", "phi", "amplitud_pp", "consistencia_alto", "consistencia_bajo", "mejora_h1_pct", "mejora_h6_pct", "usd_impacto", "motivo"],
                 floatfmt={"usd_impacto": "{:,.0f}", "n_mediana": "{:,.0f}"})
    report.p("**Y entonces:** " + ("las técnicas con efecto de mes compiten solo en esas series; en el resto, el nivel reciente." if len(seasonal) else "la tasa se predice desde su nivel reciente; la estación que negocio ve es de cuántos contratos vencen, no de cuántos renuevan."))


def chapter_precision(report: Report, results: dict, configuration: Config) -> None:
    backtest = results["backtest"]
    holdout, aggregate = backtest["backtest_holdout"], backtest.get("backtest_holdout_aggregate", pd.DataFrame())
    summary = results["forecast"]["pipeline_summary"]
    report.h(1, "5 · La precisión: cuánto nos equivocamos y cuánto de ello es azar")
    report.legend("unidad binomial", "banda total", "sesgo")
    total = summary[summary["bloque"] == "total"].iloc[0] if len(summary) else None
    if len(holdout):
        at_h1 = holdout[holdout["h"] == 1]
        weighted = float(np.average(at_h1["err_pp"].abs(), weights=at_h1["n_real"])) if len(at_h1) else np.nan
        bias = float(np.average(at_h1["err_pp"], weights=at_h1["n_real"])) if len(at_h1) else np.nan
        inside = float(at_h1["dentro_banda"].mean()) if len(at_h1) else np.nan
        rates = at_h1["tasa_real"].clip(0.01, 0.99) if "tasa_real" in at_h1.columns else pd.Series(0.5, index=at_h1.index)
        floor = float(np.average(100 * np.sqrt(rates * (1 - rates) / at_h1["n_real"].clip(lower=1)), weights=at_h1["n_real"])) if len(at_h1) else np.nan
        report.p(f"En el examen (meses que no se usaron para decidir nada), **a un mes vista nos equivocamos {weighted:.1f} pp por pool** (ponderado por contratos); "
                 f"el azar de muestreo por sí solo explica {floor:.1f} pp. Sesgo {bias:+.1f} pp. El {pct(100 * inside, 0)} de las predicciones cayó dentro de la banda del 90 %.")
    if len(aggregate):
        agg_h1 = aggregate[aggregate["h"] == 1]
        report.p(f"**El error del total** (todos los pools sumados) fue de {agg_h1['err_agg_pp'].abs().mean():.2f} pp de media y {agg_h1['err_agg_pp'].abs().max():.2f} pp en el peor mes a un mes vista"
                 + (f"; a seis meses, {aggregate[aggregate['h'] == 6]['err_agg_pp'].abs().mean():.2f} pp." if (aggregate['h'] == 6).any() else "."))
    if total is not None:
        report.p(f"La banda que se promete para el horizonte es **{pct(total['pct_banda_total_low'])} / +{pct(total['pct_banda_total_high'])}** del esperado: "
                 f"la parte idiosincrática (cada pool por su cuenta) es {pct(abs(total['pct_banda_low']))} y la común (toda la cartera a la vez) {pct(abs(total['pct_banda_comun_low']))} / +{pct(total['pct_banda_comun_high'])}. "
                 f"El suelo binomial del total es ±{pct(total['pct_cota_min'])}; el peor caso absoluto, ±{pct(total['pct_cota_max'])}.")
    if len(aggregate):
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
        for index, (h, block) in enumerate(aggregate.groupby("h")):
            block = block.sort_values("mes_objetivo")
            axis.plot(block["mes_objetivo"], 100 * block["tasa_real_agg"], "-o", color="black", linewidth=1.5, markersize=4, label="real" if index == 0 else None)
            axis.plot(block["mes_objetivo"], 100 * block["tasa_pred_agg"], "--s", color=PALETTE[index % len(PALETTE)], markersize=4, label=f"previsto a h={int(h)}")
        axis.set_ylabel("tasa del total, %")
        axis.set_title("El examen del total: previsto contra real, mes a mes, a uno y a seis meses vista", loc="left", fontsize=11)
        axis.legend(fontsize=9)
        report.figure(figure, "05_examen_del_total", "Tasa agregada real (negro) frente a la prevista con la técnica elegida, en los meses de examen. Es la prueba que la banda común resume.")
    report.p("**Y entonces:** la banda idiosincrática sola sería engañosamente estrecha; la común es la que protege. Cuando el error realizado del total cae dentro de la banda total, la banda es honesta; cuando no, hay que decirlo.")


def chapter_risk(report: Report, results: dict, configuration: Config) -> None:
    maturation = results.get("signal_maturation", {})
    adjustment = maturation.get("signal_adjustment", pd.DataFrame())
    alerts = maturation.get("signal_alerts", pd.DataFrame())
    movers = results.get("top_movers", pd.DataFrame())
    region = results["forecast"].get("forecast_by_region", pd.DataFrame())
    business = results["forecast"]["business_summary"]
    report.h(1, "6 · El riesgo que entra: qué cambiará antes del vencimiento")
    report.legend("maduración pendiente", "dinero disputable")
    total_adjust = float(adjustment["ajuste_usd"].sum()) if len(adjustment) else 0.0
    forecast_total = float(business["forecast_usd"].sum()) if len(business) else 1.0
    report.p(f"Las señales de churn maduran: `softcancel` aparece justo después de una renovación y otra vez cerca del vencimiento; `dormant` aparece en cualquier momento y solo crece. "
             f"Si la composición final de cada celda es la de los últimos doce meses cerrados, **la maduración pendiente vale {money(total_adjust)}** ({pct(100 * total_adjust / max(forecast_total, 1e-9))} del forecast). "
             f"Alertas: {int((alerts['tipo'] == 'nivel_anomalo').sum()) if len(alerts) else 0} meses ya por encima de su proporción final; {int((alerts['tipo'] == 'tendencia').sum()) if len(alerts) else 0} señales cuya proporción final crece mes a mes.")
    if len(region):
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
        x = np.arange(len(region))
        axis.bar(x - 0.2, region["pct_nivel_A"], width=0.4, color=PALETTE[0], alpha=0.85, label="% del dinero con precisión propia (A)")
        axis.bar(x + 0.2, region["pct_senal"], width=0.4, color=PALETTE[3], alpha=0.85, label="% del dinero en series con señal")
        for i, (usd, band) in enumerate(zip(region["esperado_usd"], region["pct_banda"])):
            axis.text(i, max(region["pct_nivel_A"].iloc[i], region["pct_senal"].iloc[i]) + 2, f"{money(usd)} · ±{band:.1f}%", ha="center", fontsize=8)
        axis.set_xticks(x); axis.set_xticklabels(region["region"].astype(str), fontsize=9)
        axis.set_ylabel("%")
        axis.set_title("Por región: cuánto del dinero es firme y cuánto está en disputa por las señales", loc="left", fontsize=11)
        axis.legend(fontsize=9)
        report.figure(figure, "06_regiones", "Cada región con su forecast, su banda, el % del dinero que se predice con precisión propia y el % que está en series con señal de churn.")
        report.table(region, ["region", "esperado_usd", "pct_del_total", "pct_banda", "pct_nivel_A", "pct_senal", "pct_nivel_S", "uplift_medio", "composicion_pct"], floatfmt={"esperado_usd": "{:,.0f}"})
    if len(movers):
        top = movers[movers["rank"] <= 3].copy()
        top["abs"] = top["usd_impacto"].abs()
        top = top.sort_values("abs", ascending=False).head(12)
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 5))
        colours = {"deterioro": PALETTE[3], "mejora": PALETTE[5], "senal_negativa": PALETTE[1], "banda_ancha": PALETTE[2], "sesgo_examen": PALETTE[4], "precio": PALETTE[6]}
        y = np.arange(len(top))
        axis.barh(y, top["usd_impacto"], color=[colours.get(t, "#999999") for t in top["tipo"]], alpha=0.85)
        axis.set_yticks(y); axis.set_yticklabels([f"{t} · {str(f)[:32]}" for t, f in zip(top["tipo"], top["fs_id"])], fontsize=8)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("$ en juego sobre 12 meses de pipeline")
        axis.set_title("Los movimientos que más dinero mueven: dónde defender, dónde recuperar, dónde vigilar", loc="left", fontsize=11)
        axis.invert_yaxis()
        report.figure(figure, "07_top_movers", "Los tres primeros de cada tipo de movimiento, ordenados por dinero. Negativo = riesgo; positivo = oportunidad (o presión de precio).")
    report.p("**Y entonces:** la lista de acciones del mes sale de aquí: defender donde la tasa prevista cae, recuperar donde hay señal negativa con dinero, vigilar donde la foto ya va peor de lo habitual, y revisar precio donde el uplift se aleja de la media.")


def chapter_baseline(report: Report, results: dict, configuration: Config) -> None:
    baseline = results.get("baseline", {}).get("baseline_summary", pd.DataFrame())
    report.h(1, "7 · Frente a la hoja de cálculo")
    if baseline is None or baseline.empty:
        report.p("Sin baseline calculada.")
        return
    year_columns = [c for c in baseline.columns if c.startswith("diferencia_")]
    first_year = year_columns[0].split("_")[1] if year_columns else ""
    best = baseline.sort_values("holdout_lag1_abs_pct").iloc[0]
    report.p(f"La previsión de hoja de cálculo (tasa en dólares de los últimos N meses por grupo × pipeline) da entre {baseline[year_columns[0]].min():+.1f} % y {baseline[year_columns[0]].max():+.1f} % respecto al framework para {first_year}. "
             f"Su mejor variante ({best['grano']}, {int(best['ventana_meses'])} meses) se equivocó {best['holdout_lag1_abs_pct']:.1f} % a un mes vista en los últimos 12 meses "
             f"(sesgo {best['holdout_lag1_sesgo_pct']:+.1f} %) y {best['holdout_lag4_abs_pct']:.1f} % a cuatro.")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
    labels = [f"{g}·{int(w)}m" for g, w in zip(baseline["grano"], baseline["ventana_meses"])]
    x = np.arange(len(labels))
    axis.bar(x - 0.2, baseline["holdout_lag1_abs_pct"], width=0.4, color=PALETTE[0], alpha=0.85, label="|error| a 1 mes, %")
    axis.bar(x + 0.2, baseline["holdout_lag4_abs_pct"], width=0.4, color=PALETTE[1], alpha=0.85, label="|error| a 4 meses, %")
    axis.set_xticks(x); axis.set_xticklabels(labels, fontsize=8, rotation=20)
    axis.set_ylabel("% del renovado")
    axis.set_title("La hoja acierta a un mes casi como cualquier método; a cuatro meses se abre la diferencia", loc="left", fontsize=11)
    axis.legend(fontsize=9)
    report.figure(figure, "08_baseline", "Error walk-forward de la previsión de hoja por grano y ventana. Compárese con el error del total del framework (capítulo 5).")
    report.p("**Y entonces:** si la hoja acierta igual, el framework aporta la banda y la trazabilidad; si acierta peor, aporta también el número. Y explica de dónde sale la diferencia con la cifra de negocio cuando la hay.")


def chapter_discount(report: Report, results: dict, configuration: Config) -> None:
    """8 · Price and churn: is the discount a driver?"""
    analysis = results.get("discount_churn", {})
    if not analysis:
        return
    by_bucket, by_state, effects, importance = analysis["by_bucket"], analysis["by_state"], analysis["effects"], analysis["importance"]
    report.h(1, "8 · Precio y churn: ¿el descuento es un driver?")
    report.legend("tasa estandarizada", "importancia relativa")
    reference = str(by_state["tramo_referencia"].iloc[0]) if len(by_state) else "—"
    spread = 100 * (by_bucket["tasa_estandarizada"].max() - by_bucket["tasa_estandarizada"].min()) if len(by_bucket) else np.nan
    report.p(f"La creencia: quien tiene descuento renueva a precio completo y se va. La prueba, sobre {by_bucket['contratos'].sum():,.0f} contratos de meses cerrados, con la composición de celdas fijada: "
             f"entre el tramo que mejor renueva y el que peor hay **{spread:.1f} pp de diferencia estandarizada** (referencia: {reference} = sin descuento).")
    neutral = by_state[(by_state["estado"] == "neutro") & (by_state["tramo"] != reference)]
    flagged = by_state[(by_state["estado"] == "no_instalado") & (by_state["tramo"] != reference)] if "no_instalado" in set(by_state["estado"]) else pd.DataFrame()
    if len(neutral):
        row = neutral.reindex(neutral["hueco_pp"].abs().sort_values(ascending=False).index).iloc[0]
        report.p(f"Entre los clientes **sin señal**, el tramo {row['tramo']} renueva {row['hueco_pp']:+.1f} pp respecto a {reference} (z = {row['z']:+.1f}: "
                 + ("una diferencia real." if abs(row["z"]) >= 2 else "dentro del ruido.") + ")")
    if len(flagged) and flagged["hueco_pp"].notna().any():
        row = flagged.reindex(flagged["hueco_pp"].abs().sort_values(ascending=False).index).iloc[0]
        report.p(f"Entre los clientes que **no instalaron el producto**, el mismo contraste da {row['hueco_pp']:+.1f} pp (z = {row['z']:+.1f}): "
                 + ("el descuento también separa ahí." if abs(row["z"]) >= 2 else "el descuento no separa: manda la señal."))
    if len(importance):
        ranked = importance.sort_values("r2_perdido_al_quitarlo", ascending=False)
        top = ranked.iloc[0]
        discount_row = importance[importance["grupo"] == "tramo"]
        report.p(f"**Qué explica las tasas:** {top['grupo']} es lo que más varianza pierde el modelo al quitarlo ({top['r2_perdido_al_quitarlo']:.2f} de R²); "
                 f"el descuento, {float(discount_row['r2_perdido_al_quitarlo'].iloc[0]) if len(discount_row) else np.nan:.2f}. El modelo completo explica el {100 * top['r2_modelo_completo']:.0f} % de la varianza entre grupos.")
    if len(effects):
        report.table(effects, ["variable", "valor", "referencia", "efecto_pp", "se_pp", "z"], floatfmt={"efecto_pp": "{:+.1f}", "se_pp": "{:.1f}", "z": "{:+.1f}"})
    # figure: rate by bucket inside each state, with the reference marked
    plotted = by_state[by_state["contratos"] >= 30]
    if len(plotted):
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 4.5))
        buckets = sorted(plotted["tramo"].unique())
        x = np.arange(len(buckets))
        states = [s for s in ("neutro", "no_instalado", "dormant", "softcancel", "autorenew") if s in set(plotted["estado"])]
        width = 0.8 / max(len(states), 1)
        for index, state in enumerate(states):
            block = plotted[plotted["estado"] == state].set_index("tramo").reindex(buckets)
            axis.bar(x + (index - (len(states) - 1) / 2) * width, 100 * block["tasa_comparada"] if "tasa_comparada" in block.columns else 100 * block["tasa_estandarizada"].fillna(block["tasa_bruta"]),
                     width=width, color=PALETTE[index % len(PALETTE)], alpha=0.85, label=state)
        axis.set_xticks(x); axis.set_xticklabels(buckets, fontsize=9)
        axis.set_ylabel("tasa de renovación, %")
        axis.set_title("La señal separa mucho más que el descuento: dentro de cada estado, los tramos apenas se distinguen" if (len(importance) and importance.set_index("grupo").loc["tramo", "r2_perdido_al_quitarlo"] < 0.05) else "El descuento separa incluso dentro de cada estado", loc="left", fontsize=11)
        axis.legend(fontsize=9)
        report.figure(figure, "09_descuento_por_estado", "Tasa de renovación por tramo de descuento dentro de cada estado de señal (composición de celdas fijada donde hay soporte). Si las barras de un mismo color son parecidas, el descuento no es el driver en ese estado.")
    report.p("**Y entonces:** es evidencia descriptiva sobre datos agregados, no un experimento: quien acepta un descuento puede diferir en cosas que las dimensiones no ven. Pero si las señales explican diez veces más varianza que el descuento, la política de retención debería empezar por la instalación y el uso, no por el precio.")


def chapter_annex(report: Report, results: dict, configuration: Config, legend: pd.DataFrame) -> None:
    report.h(1, "Anexo")
    report.h(2, "A · Las fichas de las series más grandes")
    for path in results.get("sheets", []):
        report.p(f"- `{path}`")
    report.h(2, "B · Leyenda de todas las métricas")
    report.table(legend, ["metrica", "unidad", "definicion", "referencia", "mucho", "poco"], max_rows=50)
    report.h(2, "C · Tablas producto (las que lee el informe y negocio)")
    report.p(", ".join(f"`sff_{configuration._resolve_physical_table_name(t).replace(configuration.table_prefix, '')}`" if hasattr(configuration, "table_prefix") else f"`{t}`" for t in TABLE_KIND["producto"]))
    core_result = results.get("nucleo") or {}
    core = core_result.get("nucleo")
    if core is not None and len(core):
        report.h(2, "E · La tabla núcleo: toda la historia en una tabla")
        by_origin = core.groupby("paso_alta").agg(filas=("anio", "size"), pipeline_raw_usd=("s1_pipeline_usd", "sum"),
                                                  pipeline_forecast_usd=("s6_pipeline_usd", "sum") if "s6_pipeline_usd" in core.columns else ("anio", "size")).reset_index()
        report.p(f"`sff_nucleo`: {len(core):,} filas × {core.shape[1]} columnas, una por combinación fina × mes, con los pasos de izquierda a derecha "
                 "(s0 el raw, s1 el raw acondicionado con los huecos explícitos, s2 series, s3 escalera, s4 técnica, s5 revalorización, s6 forecast, s7 maduración). "
                 "Cada estudio de este informe se puede rehacer agregándola: una tasa es siempre el cociente de sumas de una agregación.")
        report.table(by_origin, floatfmt={"pipeline_raw_usd": "{:,.0f}", "pipeline_forecast_usd": "{:,.0f}"})
        reconciliation = core_result.get("reconciliacion")
        if reconciliation is not None and len(reconciliation):
            report.p("Reconciliación con el raw: " + ("todas las medidas cuadran, licencia a licencia y dólar a dólar." if bool(reconciliation["cuadra"].all()) else "NO cuadra: revisar `sff_nucleo_reconciliacion`."))
    report.h(2, "D · Debilidades declaradas y su remedio")
    for line in (
        f"El examen son {configuration.test_months} meses: las bandas por pool son cuantiles de pocos puntos y casi todas vienen de la familia (misma técnica en todos los pools). Remedio: más meses de examen cuando el extracto lo permita; bandas propias cuando un pool acumule 20 predicciones.",
        "La banda común se mide con los meses de decisión: pocos puntos por horizonte. Remedio: se recalcula cada mes con uno más.",
        "La maduración de las señales es una aproximación (proporción final de 12 meses − actual) hasta que haya seis fotos mensuales: entonces la curva se mide.",
        "Más allá de seis meses se predice como a seis, con la banda de seis: el error real será mayor.",
        "La adquisición futura se simula al ritmo histórico con las dimensiones del renovador; el mapa de transición (precio tope, primera renovación) está pendiente de los valores del raw.",
    ):
        report.p(f"- {line}")


def build_report(results: dict, configuration: Config) -> str:
    """The whole report. Returns the path of informe.md."""
    folder = os.path.join(configuration.outdir, "informe")
    report = Report(folder)
    legend = legend_table(configuration)
    chapter_cover(report, results, configuration)
    chapter_panorama(report, results, configuration)
    chapter_pipeline(report, results, configuration)
    chapter_noise(report, results, configuration)
    chapter_composition(report, results, configuration)
    chapter_composition_effect(report, results, configuration)
    chapter_season(report, results, configuration)
    chapter_precision(report, results, configuration)
    chapter_risk(report, results, configuration)
    chapter_baseline(report, results, configuration)
    chapter_discount(report, results, configuration)
    chapter_scenarios(report, results, configuration)
    chapter_annex(report, results, configuration, legend)
    path = report.write()
    print(f"[informe] {path} · {len(report.figures)} figuras")
    return path
