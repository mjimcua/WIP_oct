"""diagnostics_plots.py — SFF v3 · look at the series the numbers are built on.

    from diagnostics_plots import run_series_diagnostics
    paths = run_series_diagnostics(configuration, top=10, by="usd")            # reads the sff_* tables
    paths = run_series_diagnostics(results=results, top=10, by="history")     # from run_analysis' dict

Picks the top series (by projected money or by history length, with a minimum history),
prints a table with their dynamics (φ, seasonal, trend, technique, hold-out error) and
saves FOUR compact figures under <outdir>/diagnostics/:

  1_rates.png      the monthly renewal rate of every top series on ONE axis (one colour
                   per series, legend with n · φ · season · slope), hold-out months shaded
  2_pipeline.png   the monthly pipeline units of the same series (log scale): portfolio
                   moves, spikes and holes show here, not in the rates
  3_season.png     the seasonal profile by calendar month (index = month / overall) of the
                   series declared seasonal, with the 2× binomial bound
  4_holdout.png    real vs predicted (h=1, chosen technique) on the hold-out months

Style: palette GREEN #2E7D7D / BLUE #3B8BC8 / YELLOW #E8B84F first, then tab10; vertical
single-letter month labels; big fonts. Everything from code, nothing by hand.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The project is a flat folder imported from notebooks and scripts alike: make sure the
# folder of this file is importable BEFORE importing the sibling modules below (that is
# why those imports come after this block, not at the top).
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if PROJECT_FOLDER not in sys.path:
    sys.path.insert(0, PROJECT_FOLDER)

from audit_series import read_table
from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
PALETTE = ["#2E7D7D", "#3B8BC8", "#E8B84F"] + [matplotlib.colormaps["tab10"](i) for i in range(10)]
MONTH_LETTERS = "JFMAMJJASOND"
FIGURE_WIDTH = 16
PERIOD_COLUMN_CANDIDATES = ("period", "periodo")


def top_series(series_card: pd.DataFrame, top: int, by: str, min_history_months: int) -> pd.DataFrame:
    """The top series by projected money ('usd') or by history length ('history'),
    among trainable series with at least `min_history_months` of history."""
    eligible = series_card[(series_card["ruta"] == TREATMENT_PREDICTABLE) & (series_card["meses_historia"] >= min_history_months)]
    key = "usd_proyectado" if by == "usd" else "meses_historia"
    return eligible.sort_values([key, "usd_proyectado"], ascending=False).head(top)


def monthly_of(units: pd.DataFrame, series_id: str, configuration: Config) -> pd.DataFrame:
    """The real history months of one series: rate and pipeline units, indexed by period."""
    period = configuration.period_col
    rows = units[(units["fs_id"] == series_id) & (units["sintetica"] == 0) & (units[configuration.dataset_role_col].isin((ROLE_TRAIN, ROLE_TEST)))]
    monthly = rows.groupby(period).agg(pipe=(configuration.pipeline_units_col, "sum"), ren=(configuration.renewed_units_col, "sum")).sort_index()
    monthly["rate"] = np.where(monthly["pipe"] > 0, monthly["ren"] / monthly["pipe"].replace(0, np.nan), np.nan)
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly


def month_ticks(axis, periods: pd.PeriodIndex) -> None:
    """Vertical single-letter month labels, the year under every January."""
    positions = np.arange(len(periods))
    axis.set_xticks(positions)
    axis.set_xticklabels([MONTH_LETTERS[p.month - 1] + (f"\n{p.year}" if p.month == 1 else "") for p in periods],
                         fontsize=9, rotation=0)
    axis.grid(axis="x", alpha=0.15)


def label_of(row: pd.Series, dynamics: pd.Series, technique: pd.Series) -> str:
    """One legend line: id · n · φ · season · slope · technique."""
    season = "month effects" if int(dynamics.get("estacional", 0) or 0) else "level"
    return (f"{row['fs_id'][:52]} · n={row['n_propio']:.0f} · pool n={dynamics.get('n_pool', float('nan'))} · {season} · "
            f"{technique.get('tecnica', '-')} {technique.get('tecnica_origen', '')}")


def plot_rates_and_pipeline(monthlies: dict, labels: dict, holdout_start, output_folder: str) -> list:
    """Figures 1 and 2: rates on one axis; pipeline units on one axis (log)."""
    all_periods = sorted(set().union(*[set(m.index) for m in monthlies.values()]))
    all_periods = pd.PeriodIndex(all_periods, freq="M")
    position = {p: i for i, p in enumerate(all_periods)}
    paths = []
    for name, column, ylabel, log in (("1_rates", "rate", "renewal rate", False), ("2_pipeline", "pipe", "pipeline units (log)", True)):
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 7))
        for index, (series_id, monthly) in enumerate(monthlies.items()):
            x = [position[p] for p in monthly.index]
            axis.plot(x, monthly[column], color=PALETTE[index % len(PALETTE)], linewidth=2.2, marker="o", markersize=3.5,
                      label=labels[series_id] if column == "rate" else series_id[:52])
        if holdout_start is not None and holdout_start in position:
            axis.axvspan(position[holdout_start] - 0.5, len(all_periods) - 0.5, color="#E8B84F", alpha=0.12, label="hold-out")
        if log:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0, 1.0)
        month_ticks(axis, all_periods)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_title(f"top series · {ylabel}", fontsize=14, loc="left")
        axis.legend(fontsize=8, loc="lower left" if column == "rate" else "best", ncol=1, framealpha=0.9)
        figure.tight_layout()
        path = os.path.join(output_folder, f"{name}.png")
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_seasonal_profiles(top: pd.DataFrame, dynamics_by_id: dict, output_folder: str) -> str:
    """Figure 3: the benchmark view for the top series — month effects only where the
    benchmark found them; otherwise a note that the rate has no calendar shape."""
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 5.5))
    drawn = 0
    for index, (_, row) in enumerate(top.iterrows()):
        effects_text = row.get("efectos_mes", "") if isinstance(row, pd.Series) else ""
        if not isinstance(effects_text, str) or not effects_text:
            continue
        effects = {int(k): float(v) for k, v in (item.split(":") for item in effects_text.split("|"))}
        months = sorted(effects)
        axis.plot(months, [effects[m] for m in months], "-o", color=PALETTE[index % len(PALETTE)], linewidth=2.2,
                  label=f"{row['fs_id'][:40]} · amplitude {row.get('amplitud_pp', float('nan'))} pp · {'SEASONAL' if row.get('veredicto_estacional', 0) else 'not material'}")
        drawn += 1
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(1, 13))
    axis.set_xticklabels(list(MONTH_LETTERS), fontsize=11)
    axis.set_ylabel("month effect on the rate, pp", fontsize=12)
    axis.set_title("month effects from the seasonality benchmark (regression with month dummies, weighted by n)", fontsize=12, loc="left")
    if drawn:
        axis.legend(fontsize=8, framealpha=0.9)
    else:
        axis.text(0.5, 0.5, "no benchmark rows for these series: the rate is predicted from its level", ha="center", fontsize=11)
    figure.tight_layout()
    path = os.path.join(output_folder, "3_season.png")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_holdout(top: pd.DataFrame, holdout: pd.DataFrame, output_folder: str) -> str:
    """Figure 4: real vs predicted (h=1) on the hold-out months for the top series' estimation ids."""
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6))
    drawn = 0
    if len(holdout):
        at_h1 = holdout[holdout["h"] == 1]
        for index, (_, row) in enumerate(top.iterrows()):
            block = at_h1[at_h1["id_estimacion"] == row["id_estimacion"]].sort_values("mes_objetivo")
            if block.empty:
                continue
            x = np.arange(len(block))
            colour = PALETTE[index % len(PALETTE)]
            axis.plot(x, block["tasa_real"], "-o", color=colour, linewidth=2, markersize=4, label=f"{row['fs_id'][:40]} real")
            axis.plot(x, block["tasa_pred"], "--", color=colour, linewidth=1.6, label=f"   predicted h=1 ({block['tecnica'].iloc[0]}), "
                      f"|err| {block['err_pp'].abs().mean():.1f}pp, {block['dentro_banda'].mean():.0%} in band")
            axis.fill_between(x, block["tasa_pred"] + block["banda_low_pp"] / 100, block["tasa_pred"] + block["banda_high_pp"] / 100,
                              color=colour, alpha=0.08)
            axis.set_xticks(x)
            axis.set_xticklabels(block["mes_objetivo"], rotation=45, fontsize=9)
            drawn += 1
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("renewal rate", fontsize=12)
    axis.set_title("hold-out · real (solid) vs predicted at h=1 (dashed) with its band", fontsize=14, loc="left")
    if drawn:
        axis.legend(fontsize=8, ncol=2, framealpha=0.9)
    figure.tight_layout()
    path = os.path.join(output_folder, "4_holdout.png")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def run_series_diagnostics(configuration: Config = None, results: dict = None, top: int = 10, by: str = "usd",
                           min_history_months: int = 24, output_folder: str = None) -> list:
    """Pick the top series, print their diagnostics, save the four figures. Returns the paths."""
    card = read_table("series_card", configuration, results)
    dynamics = read_table("pool_reference", configuration, results)
    technique = read_table("decision_technique", configuration, results)
    holdout = read_table("backtest_holdout", configuration, results)
    units = results["units"] if results is not None and "units" in results else None
    if units is None:
        fact = read_table("fact_fu", configuration, results)
        if fact.empty:
            raise RuntimeError("series history not available: pass results=run_analysis(...) or an engine with sff_fact_fu")
        fact["fs_id"] = fact["fu_id"].str.rsplit("|", n=1).str[0]
        fact["sintetica"] = 0
        fact[configuration.period_col] = pd.PeriodIndex(fact[configuration.period_col].astype(str), freq="M")
        units = fact
    chosen = top_series(card, top, by, min_history_months)
    dynamics_by_id = {row["id_estimacion"]: row for _, row in dynamics.iterrows()} if len(dynamics) else {}
    technique_by_id = {row["id_estimacion"]: row for _, row in technique.iterrows()} if len(technique) else {}
    folder = output_folder or os.path.join(configuration.outdir if configuration else ".", "diagnostics")
    os.makedirs(folder, exist_ok=True)
    print(f"[diag] top {len(chosen)} series by {by} (≥ {min_history_months} history months)")
    print("   series · $ projected · months · n · id_estimacion · pool n · month effects · technique · hold-out |err| h=1")
    monthlies, labels = {}, {}
    for _, row in chosen.iterrows():
        d, t = dynamics_by_id.get(row["id_estimacion"], {}), technique_by_id.get(row["id_estimacion"], {})
        block = holdout[(holdout["id_estimacion"] == row["id_estimacion"]) & (holdout["h"] == 1)] if len(holdout) else pd.DataFrame()
        err = f"{block['err_pp'].abs().mean():.1f}pp" if len(block) else "-"
        print(f"   {row['fs_id'][:58]:<58} ${row['usd_proyectado']:>11,.0f} {int(row['meses_historia']):>3}m n={row['n_propio']:>6.0f} "
              f"→ {str(row['id_estimacion'])[:36]:<36} pool n={d.get('n_pool', '-')} effects={d.get('estacional', '-')} {t.get('tecnica', '-')}/{t.get('tecnica_origen', '')} {err}")
        monthlies[row["fs_id"]] = monthly_of(units, row["fs_id"], configuration)
        labels[row["fs_id"]] = label_of(row, d, t)
    holdout_start = pd.Period(holdout["mes_objetivo"].min(), freq="M") if len(holdout) else None
    paths = plot_rates_and_pipeline(monthlies, labels, holdout_start, folder)
    benchmark = read_table("decision_estacionalidad", configuration, results)
    top_with_benchmark = chosen.merge(benchmark, on="fs_id", how="left", suffixes=("", "_bench")) if len(benchmark) else chosen
    paths.append(plot_seasonal_profiles(top_with_benchmark, dynamics_by_id, folder))
    paths.append(plot_holdout(chosen, holdout, folder))
    print(f"[diag] figures in {folder}: " + ", ".join(os.path.basename(p) for p in paths))
    return paths


# ═══════════════════════════════════════════════════════════════════════════════════
# THE SHEET OF ONE SERIES · the showcase trio · the guess game
# ═══════════════════════════════════════════════════════════════════════════════════

def binomial_band(rates: np.ndarray, units: np.ndarray, z: float = 1.645) -> tuple:
    """The z·binomial band of each month around its own rate (pure sampling)."""
    se = np.sqrt(np.maximum(rates * (1 - rates), 0.0475) / np.maximum(units, 1.0))
    return np.clip(rates - z * se, 0, 1), np.clip(rates + z * se, 0, 1)


def series_sheet(series_id: str, configuration: Config = None, results: dict = None, output_folder: str = None,
                 verbose: bool = True) -> str:
    """ONE figure with everything the numbers say about one series, next to the series itself.

    Six panels: (1) monthly rate with its binomial band, the pooled mean and the
    last-12-month trend, hold-out shaded; (2) pipeline units by month (the season of the
    volume); (3) month × year panel of standardized residuals (the benchmark's view);
    (4) hold-out real vs predicted with band; (5) the composition of the series' cell,
    month by month (Kitagawa: behaviour vs composition); (6) the forecast of the series,
    expected $ by month with its band, coloured by pipeline origin. A text box carries
    the story in words.
    Returns the PNG path. Prints the summary too.
    """
    card = read_table("series_card", configuration, results)
    row = card[card["fs_id"] == series_id]
    if row.empty:
        raise KeyError(f"series '{series_id}' not in series_card")
    row = row.iloc[0]
    dynamics = read_table("pool_reference", configuration, results)
    dynamics = dynamics[dynamics["id_estimacion"] == row["id_estimacion"]]
    d = dynamics.iloc[0] if len(dynamics) else pd.Series(dtype=object)
    technique = read_table("decision_technique", configuration, results)
    technique = technique[technique["id_estimacion"] == row["id_estimacion"]]
    if len(technique) and "tramo_h" in technique.columns:
        technique = technique.sort_values("h_min")
    t = technique.iloc[0] if len(technique) else pd.Series(dtype=object)
    holdout = read_table("backtest_holdout", configuration, results)
    holdout = holdout[(holdout["id_estimacion"] == row["id_estimacion"]) & (holdout["h"] == 1)].sort_values("mes_objetivo") if len(holdout) else holdout
    units = _units_frame(configuration, results)
    monthly = monthly_of(units, series_id, configuration)
    valid = monthly[monthly["rate"].notna()]
    periods = pd.PeriodIndex(valid.index, freq="M")
    x = np.arange(len(valid))
    rates, pipe = valid["rate"].to_numpy(dtype=float), valid["pipe"].to_numpy(dtype=float)

    figure, axes = plt.subplots(3, 2, figsize=(FIGURE_WIDTH, 15), gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    ax_rate, ax_units, ax_season, ax_holdout = axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]
    ax_mix, ax_forecast = axes[2, 0], axes[2, 1]
    # (1) rate with binomial band, mean, recent trend, level change
    low, high = binomial_band(rates, pipe, configuration.z if configuration else 1.645)
    ax_rate.fill_between(x, low, high, color="#3B8BC8", alpha=0.12, label="binomial band of each month (sampling only)")
    ax_rate.plot(x, rates, "-o", color="#2E7D7D", linewidth=2.4, markersize=4, label="monthly rate")
    pooled = valid["ren"].sum() / max(valid["pipe"].sum(), 1)
    ax_rate.axhline(pooled, color="#E8B84F", linewidth=1.8, linestyle="--", label=f"pooled mean {pooled:.3f}")
    if len(valid) >= 12:
        recent_x = x[-12:]
        coefficients = np.polyfit(recent_x, rates[-12:], 1)
        ax_rate.plot(recent_x, np.polyval(coefficients, recent_x), color="#C0392B", linewidth=2,
                     label=f"last 12 months trend {100 * 12 * coefficients[0]:+.1f} pp/yr")
    if len(holdout):
        first = pd.Period(holdout["mes_objetivo"].min(), freq="M")
        if first in set(periods):
            ax_rate.axvspan(list(periods).index(first) - 0.5, len(x) - 0.5, color="#E8B84F", alpha=0.12, label="hold-out")
    month_ticks(ax_rate, periods)
    ax_rate.set_ylim(0, 1.0)
    ax_rate.set_title(f"{series_id[:70]}  ·  n≈{row['n_propio']:.0f}/month  ·  level {row['nivel_riesgo']}", fontsize=13, loc="left")
    ax_rate.legend(fontsize=8, loc="lower left", framealpha=0.9)
    # (2) units
    ax_units.bar(x, pipe, color="#3B8BC8", alpha=0.8)
    month_ticks(ax_units, periods)
    ax_units.set_ylabel("pipeline units", fontsize=11)
    ax_units.set_title("pipeline units by month (the season of the VOLUME lives here)", fontsize=11, loc="left")
    # (3) month × year panel of standardized residuals after trend (the benchmark's view)
    if len(valid) >= 13:
        from analysis_seasonality_benchmark import month_year_panel
        panel, consistency = month_year_panel(valid)
        for year, block in panel.groupby("anio"):
            ax_season.plot(block["mes"], block["z"], "-o", linewidth=1.8, markersize=4, label=str(year))
        ax_season.axhspan(-2, 2, color="grey", alpha=0.12, label="±2 sampling errors")
        ax_season.set_ylabel("z = (rate − trend) / binomial se", fontsize=11)
    ax_season.axhline(0, color="black", linewidth=0.8)
    ax_season.set_xticks(range(1, 13)); ax_season.set_xticklabels(list(MONTH_LETTERS), fontsize=11)
    ax_season.set_title("month × year: is the same month high (or low) every year? (real season = same sign every year)", fontsize=11, loc="left")
    ax_season.legend(fontsize=8, framealpha=0.9)
    # (4) hold-out
    if len(holdout):
        hx = np.arange(len(holdout))
        ax_holdout.plot(hx, holdout["tasa_real"], "-o", color="#2E7D7D", linewidth=2.4, label="real")
        ax_holdout.plot(hx, holdout["tasa_pred"], "--o", color="#C0392B", linewidth=2, label=f"predicted h=1 · {holdout['tecnica'].iloc[0]}")
        ax_holdout.fill_between(hx, holdout["tasa_pred"] + holdout["banda_low_pp"] / 100, holdout["tasa_pred"] + holdout["banda_high_pp"] / 100,
                                color="#C0392B", alpha=0.10, label=f"band · {holdout['dentro_banda'].mean():.0%} inside")
        ax_holdout.set_xticks(hx); ax_holdout.set_xticklabels(holdout["mes_objetivo"], rotation=45, fontsize=9)
        ax_holdout.set_ylim(0, 1.0)
        ax_holdout.set_title(f"hold-out at h=1 · |error| {holdout['err_pp'].abs().mean():.1f} pp · bias {holdout['err_pp'].mean():+.1f} pp", fontsize=12, loc="left")
        ax_holdout.legend(fontsize=8, framealpha=0.9)
    else:
        ax_holdout.text(0.5, 0.5, "no hold-out (no backtest for this id: under the floor or not run)", ha="center", fontsize=11)
        ax_holdout.set_axis_off()
    # (5) the composition of the series' cell: behaviour vs composition, month by month (Kitagawa)
    mix = read_table("mix_shift_decomposition", configuration, results)
    cell_mix = mix[mix["celda_id"] == row["celda_id"]].sort_values("mes") if len(mix) else mix
    if len(cell_mix):
        mx = np.arange(len(cell_mix))
        ax_mix.bar(mx, cell_mix["delta_comportamiento_pp"], color="#2E7D7D", alpha=0.85, label="behaviour (people renewing differently)")
        ax_mix.bar(mx, cell_mix["delta_composicion_pp"], bottom=np.where(np.sign(cell_mix["delta_composicion_pp"]) == np.sign(cell_mix["delta_comportamiento_pp"]), cell_mix["delta_comportamiento_pp"], 0),
                   color="#E8B84F", alpha=0.85, label="composition (the mix of who falls due)")
        ax_mix.plot(mx, cell_mix["delta_agregado_pp"], "-o", color="black", linewidth=1.6, markersize=4, label="change of the cell's rate")
        ax_mix.set_xticks(mx)
        ax_mix.set_xticklabels([MONTH_LETTERS[int(m[5:7]) - 1] for m in cell_mix["mes"]], fontsize=10)
        share = cell_mix["delta_composicion_pp"].abs().sum() / max((cell_mix["delta_composicion_pp"].abs() + cell_mix["delta_comportamiento_pp"].abs()).sum(), 1e-9)
        ax_mix.set_title(f"cell {row['celda_id'][:45]} · month-to-month change of its rate split by Kitagawa · composition {share:.0%}", fontsize=11, loc="left")
        ax_mix.axhline(0, color="black", linewidth=0.8)
        ax_mix.set_ylabel("pp", fontsize=11)
        ax_mix.legend(fontsize=8, framealpha=0.9)
    else:
        ax_mix.text(0.5, 0.5, "no composition rows for this cell", ha="center", fontsize=11)
        ax_mix.set_axis_off()
    # (6) the forecast of the series: expected $ by month with its band, coloured by pipeline origin
    detail = read_table("forecast_detail", configuration, results)
    detail = detail[detail["fs_id"] == series_id] if len(detail) else detail
    fbands = read_table("forecast_bands", configuration, results)
    if len(detail):
        joined = detail.merge(fbands[["fu_comb_key", "banda_low_usd", "banda_high_usd"]], on="fu_comb_key", how="left") if len(fbands) else detail.assign(banda_low_usd=0.0, banda_high_usd=0.0)
        period_column = [c for c in ("period", "periodo") if c in joined.columns][0]
        monthly = joined.groupby([period_column, "origen_pipeline"], as_index=False).agg(esperado=("esperado_usd", "sum"), low=("banda_low_usd", "sum"), high=("banda_high_usd", "sum"))
        months_f = sorted(monthly[period_column].astype(str).unique())
        colour_of = {"real": "#2E7D7D", "proyectada": "#3B8BC8", "simulada": "#E8B84F"}
        bottom = np.zeros(len(months_f))
        for origin in ("real", "proyectada", "simulada"):
            block = monthly[monthly["origen_pipeline"] == origin].set_index(monthly[monthly["origen_pipeline"] == origin][period_column].astype(str))
            values = np.array([float(block["esperado"].get(m, 0.0)) for m in months_f])
            if values.sum() > 0:
                ax_forecast.bar(np.arange(len(months_f)), values, bottom=bottom, color=colour_of[origin], alpha=0.85, label=f"pipeline {origin}")
                bottom = bottom + values
        totals = monthly.groupby(monthly[period_column].astype(str)).agg(esperado=("esperado", "sum"), low=("low", "sum"), high=("high", "sum")).reindex(months_f)
        ax_forecast.errorbar(np.arange(len(months_f)), totals["esperado"], yerr=[-totals["low"].fillna(0), totals["high"].fillna(0)], fmt="none", ecolor="black", capsize=3, label="band")
        ax_forecast.set_xticks(np.arange(len(months_f)))
        ax_forecast.set_xticklabels([MONTH_LETTERS[int(m[5:7]) - 1] + ("\n" + m[:4] if m[5:7] == "01" or m == months_f[0] else "") for m in months_f], fontsize=9)
        ax_forecast.set_ylabel("expected renewed $", fontsize=11)
        ax_forecast.set_title(f"forecast of the series · ${totals['esperado'].sum():,.0f} over {len(months_f)} months · technique {detail['tecnica'].iloc[0]}", fontsize=11, loc="left")
        ax_forecast.legend(fontsize=8, framealpha=0.9)
    else:
        ax_forecast.text(0.5, 0.5, "no forecast rows (nothing to predict for this series)", ha="center", fontsize=11)
        ax_forecast.set_axis_off()
    summary = sheet_summary(row, d, t, holdout)
    figure.text(0.01, 0.005, summary, fontsize=9, family="monospace", va="bottom")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    folder = output_folder or os.path.join(configuration.outdir if configuration else ".", "diagnostics")
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in series_id)[:80]
    path = os.path.join(folder, f"sheet_{safe}.png")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    if verbose:
        print(summary)
    return path


def sheet_summary(row: pd.Series, d: pd.Series, t: pd.Series, holdout: pd.DataFrame) -> str:
    """The diagnostics of one series in plain words (the text box of the sheet)."""
    lines = [
        f"SERIES {row['fs_id']} · estimated with {row['id_estimacion']} (rung {int(row['peldano'])}, n_efectivo {row['n_efectivo']:.0f}) · level {row['nivel_riesgo']}",
        f"RATE   {row['tasa_estimada']:.3f} · se_estimacion {row['se_estimacion_pp']:.1f} pp · se_prediccion {row['se_prediccion_pp']:.1f} pp",
    ]
    if len(d):
        lines.append(f"POOL   {int(d['meses'])} months · n_pool {d['n_pool']:.0f} · rate {d['tasa_pool']} · "
                     f"{'with support' if d['gate'] == 'nivel' else 'under the floor'} · month effects {'yes' if d['estacional'] else 'no (level only)'}")
    if len(t):
        lines.append(f"TECHN. {t['tecnica']} ({t['tecnica_origen']}, band {t.get('tramo_h', '-')}) · mean |error| {t['err_pp_medio']} pp = {t['err_norm_medio']} binomial units · challenger at {t['retador_err_norm']}"
                     + (f" · hold-out h=1 |error| {holdout['err_pp'].abs().mean():.1f} pp, bias {holdout['err_pp'].mean():+.1f} pp, {holdout['dentro_banda'].mean():.0%} in band" if len(holdout) else ""))
    return "\n".join(lines)


def _units_frame(configuration, results):
    units = results["units"] if results is not None and "units" in results else None
    if units is None:
        fact = read_table("fact_fu", configuration, results)
        if fact.empty:
            raise RuntimeError("series history not available: pass results=run_analysis(...) or an engine with sff_fact_fu")
        fact["fs_id"] = fact["fu_id"].str.rsplit("|", n=1).str[0]
        fact["sintetica"] = 0
        fact[configuration.period_col] = pd.PeriodIndex(fact[configuration.period_col].astype(str), freq="M")
        units = fact
    return units


def pick_showcase_series(configuration: Config = None, results: dict = None, min_history_months: int = 24) -> dict:
    """Series to explain with: the benchmark's seasonal series (if any), the benchmark's
    biggest series, the most trending one, and the biggest series of the cell with the
    most mix-shift."""
    card = read_table("series_card", configuration, results)
    benchmark = read_table("decision_estacionalidad", configuration, results)
    picks = {}
    if len(benchmark):
        seasonal = benchmark[benchmark["veredicto_estacional"] == 1].sort_values("usd_proyectado", ascending=False)
        if len(seasonal):
            picks["estacional"] = seasonal.iloc[0]["fs_id"]
        biggest = benchmark.sort_values("usd_proyectado", ascending=False)
        for _, candidate in biggest.iterrows():
            if candidate["fs_id"] not in picks.values():
                picks["mayor_del_benchmark"] = candidate["fs_id"]
                break
        trending = benchmark[benchmark["veredicto_tendencia"] != 0].sort_values("usd_proyectado", ascending=False)
        for _, candidate in trending.iterrows():
            if candidate["fs_id"] not in picks.values():
                picks["tendencia"] = candidate["fs_id"]
                break
    mix = read_table("mix_shift_decomposition", configuration, results)
    if len(mix):
        worst_cell = mix.groupby("celda_id")["delta_composicion_pp"].apply(lambda s: s.abs().mean()).idxmax()
        in_cell = card[(card["celda_id"] == worst_cell) & (card["ruta"] == TREATMENT_PREDICTABLE)].sort_values("usd_proyectado", ascending=False)
        if len(in_cell):
            picks["mix_shift"] = in_cell.iloc[0]["fs_id"]
    return picks


def technique_error_by_horizon(key, configuration: Config = None, results: dict = None, output_folder: str = None) -> str:
    """ONE figure: mean |error| (binomial units) by horizon, one line per technique, for
    the pool a key points at. The champion of each band is marked. This is where "simple
    wins near, shape wins far" (or not) becomes visible."""
    from sheet import resolve_key
    keys = resolve_key(key, configuration, results)
    estimation_id = keys["id_estimacion"]
    predictions = read_table("backtest_predictions", configuration, results)
    block = predictions[predictions["id_estimacion"] == estimation_id]
    if block.empty:
        raise KeyError(f"no backtest predictions for {estimation_id} (under the floor, or backtest not run)")
    decision = read_table("decision_technique", configuration, results)
    decision = decision[decision["id_estimacion"] == estimation_id]
    table = block.assign(abs_norm=block["err_norm"].abs()).groupby(["tecnica_id", "h"])["abs_norm"].mean().unstack("h")
    counts = block.groupby(["tecnica_id", "h"]).size().unstack("h")
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.5))
    order = table.mean(axis=1).sort_values().index
    for index, technique_id in enumerate(order):
        row = table.loc[technique_id].dropna()
        judged_everywhere = counts.loc[technique_id].reindex(row.index).fillna(0) > 0
        style = "-" if len(row) >= 3 else "--"
        axis.plot(row.index, row.values, style, marker="o", markersize=5, linewidth=2.2 if index < 4 else 1.4,
                  color=PALETTE[index % len(PALETTE)], alpha=1.0 if index < 6 else 0.6, label=technique_id)
    for _, band in decision.iterrows():
        h_lo, h_hi = int(band["h_min"]), int(min(band["h_max"], table.columns.max()))
        axis.axvspan(h_lo - 0.4, h_hi + 0.4, alpha=0.06, color="grey")
        axis.text((h_lo + h_hi) / 2, axis.get_ylim()[1] * 0.97 if axis.get_ylim()[1] > 0 else 1, f"{band['tramo_h']}: {band['tecnica']}\n({band['tecnica_origen']})",
                  ha="center", va="top", fontsize=9)
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle=":", label="1.0 = one sampling error (the floor)")
    axis.set_xlabel("horizon h (months ahead)", fontsize=12)
    axis.set_ylabel("mean |error| in binomial units", fontsize=12)
    axis.set_xticks(sorted(table.columns))
    axis.set_title(f"{estimation_id[:70]} · backtest error by horizon and technique (n≈{block['n_real'].median():.0f}/month)", fontsize=13, loc="left")
    axis.legend(fontsize=8, ncol=2, framealpha=0.9)
    figure.tight_layout()
    folder = output_folder or os.path.join(configuration.outdir if configuration else ".", "diagnostics")
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in estimation_id)[:70]
    path = os.path.join(folder, f"horizon_{safe}.png")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"[diag] error by horizon of {estimation_id}: {path}")
    return path
