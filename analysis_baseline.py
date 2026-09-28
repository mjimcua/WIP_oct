"""analysis_baseline.py — SFF v3 · ANALYSIS · the Excel baseline.

The forecast someone would build in a spreadsheet: take the DOLLAR renewal rate of the
last few months (renewed $ / expiring $, so the price uplift is inside it) at a coarse
grain (the whole portfolio, or a few mandatory dimensions), and multiply the future
pipeline $ by it. No series, no ladder, no techniques.

It exists to answer two questions the business will ask:
  1. "Our number is X, yours is Y — why?"  The baseline reproduces X (or shows what X
     assumes), and the difference to Y is decomposed by month.
  2. "Which one should we trust?"  The same rule is walked forward over the last months
     with truth (predict month t with data up to t−lag) and its error is compared with
     the framework's hold-out of the total (`backtest_holdout_agg`) at the same lag.

Tables:
  baseline_forecast   grain × window × month: pipeline $, dollar rate applied, forecast $
  baseline_summary    grain × window: forecast of the rest of this year and of next year,
                      the framework's, the difference, and the walk-forward error of the
                      baseline at lag 1 and lag 4 (mean |error| % and bias)
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
GRAIN_GLOBAL = "global"
WINDOWS_MONTHS = (1, 3, 12)
HOLDOUT_LAGS = (1, 4)
HOLDOUT_MONTHS = 12


def grain_columns(name: str, configuration: Config) -> list:
    """The columns of a baseline grain: 'global' = none, 'mandatory' = every mandatory
    dim, else a '+'-joined list of column names."""
    if name == GRAIN_GLOBAL:
        return []
    if name == "mandatory":
        return list(configuration.business_mandatory_dims)
    return [c.strip() for c in name.split("+")]


def dollar_rate(history: pd.DataFrame, columns: list, months: list, configuration: Config) -> pd.Series:
    """Σ renewed $ / Σ pipeline $ over the given months, per grain group (empty columns →
    one global value under key 'global')."""
    block = history[history[configuration.period_col].isin(months)]
    if not columns:
        pipe = block[configuration.pipeline_usd_col].sum()
        return pd.Series({GRAIN_GLOBAL: block[configuration.renewed_usd_col].sum() / pipe if pipe > 0 else np.nan})
    grouped = block.groupby(join_columns(block, columns)).agg(ren=(configuration.renewed_usd_col, "sum"), pipe=(configuration.pipeline_usd_col, "sum"))
    return (grouped["ren"] / grouped["pipe"].replace(0, np.nan))


def baseline_forecast(fine_table: pd.DataFrame, future: pd.DataFrame, grains: list, configuration: Config) -> pd.DataFrame:
    """The baseline applied to the future pipeline, per grain × window × month."""
    period, role = configuration.period_col, configuration.dataset_role_col
    history = fine_table[fine_table[role].isin(TRUTH_ROLES)]
    truth_months = sorted(history[period].unique())
    rows = []
    for grain in grains:
        columns = grain_columns(grain, configuration)
        future_keys = join_columns(future, columns) if columns else pd.Series(GRAIN_GLOBAL, index=future.index)
        global_fallback = dollar_rate(history, [], truth_months[-3:], configuration)[GRAIN_GLOBAL]
        for window in WINDOWS_MONTHS:
            rates = dollar_rate(history, columns, truth_months[-window:], configuration)
            applied = future_keys.map(rates).fillna(global_fallback)
            frame = pd.DataFrame({period: future[period].astype(str), "pipeline_usd": future[configuration.pipeline_usd_col],
                                  "forecast_usd": future[configuration.pipeline_usd_col] * applied})
            monthly = frame.groupby(period, as_index=False).sum()
            monthly["tasa_usd_aplicada"] = (monthly["forecast_usd"] / monthly["pipeline_usd"].replace(0, np.nan)).round(4)
            monthly.insert(0, "ventana_meses", window)
            monthly.insert(0, "grano", grain)
            rows.append(monthly)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def baseline_holdout(fine_table: pd.DataFrame, grains: list, configuration: Config) -> pd.DataFrame:
    """Walk the same rule forward: predict the renewed $ of month t with the dollar rate of
    the `window` months up to t−lag, per grain; compare with what happened.
    OUTPUT: grain, window, lag, mes, real_usd, pred_usd, err_pct."""
    period, role = configuration.period_col, configuration.dataset_role_col
    history = fine_table[fine_table[role].isin(TRUTH_ROLES)]
    truth_months = sorted(history[period].unique())
    targets = truth_months[-HOLDOUT_MONTHS:]
    rows = []
    for grain in grains:
        columns = grain_columns(grain, configuration)
        for window in WINDOWS_MONTHS:
            for lag in HOLDOUT_LAGS:
                for target in targets:
                    position = truth_months.index(target)
                    if position - lag - window + 1 < 0:
                        continue
                    train_months = truth_months[position - lag - window + 1: position - lag + 1]
                    rates = dollar_rate(history, columns, train_months, configuration)
                    month_rows = history[history[period] == target]
                    keys = join_columns(month_rows, columns) if columns else pd.Series(GRAIN_GLOBAL, index=month_rows.index)
                    fallback = dollar_rate(history, [], train_months, configuration)[GRAIN_GLOBAL]
                    predicted = float((month_rows[configuration.pipeline_usd_col] * keys.map(rates).fillna(fallback)).sum())
                    real = float(month_rows[configuration.renewed_usd_col].sum())
                    rows.append(dict(grano=grain, ventana_meses=window, lag=lag, mes=str(target), real_usd=round(real, 2),
                                     pred_usd=round(predicted, 2), err_pct=round(100 * (predicted - real) / real, 3) if real > 0 else np.nan))
    return pd.DataFrame(rows)


def run_baseline(fine_table: pd.DataFrame, forecast_units_extended: pd.DataFrame, business_summary: pd.DataFrame,
                 configuration: Config) -> dict:
    """Phase 5.9: the baseline against the framework, per year, with its own hold-out.
    The future pipeline = the projection rows of the fine table + the simulated rows
    (both carry the dimensions). Persists baseline_forecast and baseline_summary."""
    period, role = configuration.period_col, configuration.dataset_role_col
    grains = list(configuration.baseline_grains)
    known_future = fine_table[fine_table[role].isin((ROLE_PROJECTION, ROLE_PENDING))]
    future = pd.concat([known_future, forecast_units_extended], ignore_index=True) if forecast_units_extended is not None and len(forecast_units_extended) else known_future.copy()
    future[period] = pd.PeriodIndex(future[period].astype(str), freq="M")
    if future.empty or business_summary is None or business_summary.empty:
        print("[5] baseline: nothing to compare (no future rows)")
        return dict(baseline_forecast=pd.DataFrame(), baseline_summary=pd.DataFrame(), baseline_holdout=pd.DataFrame())
    forecast = baseline_forecast(fine_table, future, grains, configuration)
    holdout = baseline_holdout(fine_table, grains, configuration)
    forecast["anio"] = pd.PeriodIndex(forecast[period].astype(str), freq="M").year
    framework = business_summary.set_index("anio")
    rows = []
    for (grain, window), block in forecast.groupby(["grano", "ventana_meses"]):
        by_year = block.groupby("anio")["forecast_usd"].sum()
        errors = holdout[(holdout["grano"] == grain) & (holdout["ventana_meses"] == window)]
        row = dict(grano=grain, ventana_meses=window)
        for year in framework.index:
            row[f"baseline_{year}_usd"] = round(float(by_year.get(year, 0.0)), 2)
            row[f"framework_{year}_usd"] = round(float(framework.loc[year, "forecast_usd"]), 2)
            row[f"diferencia_{year}_pct"] = round(100 * (by_year.get(year, 0.0) - framework.loc[year, "forecast_usd"]) / max(framework.loc[year, "forecast_usd"], 1e-9), 2)
        for lag in HOLDOUT_LAGS:
            at_lag = errors[errors["lag"] == lag]["err_pct"]
            row[f"holdout_lag{lag}_abs_pct"] = round(float(at_lag.abs().mean()), 2) if len(at_lag) else np.nan
            row[f"holdout_lag{lag}_sesgo_pct"] = round(float(at_lag.mean()), 2) if len(at_lag) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    configuration.write(forecast.drop(columns="anio"), "baseline_forecast")
    configuration.write(summary, "baseline_summary")
    print("[5] BASELINE (the spreadsheet forecast: dollar renewal rate of the last N months × future pipeline) vs the framework")
    print("   grain · window   " + "  ".join(f"{y}: baseline / framework / diff" for y in framework.index) + "   walk-forward |err| lag1 (bias) · lag4 (bias)")
    for _, row in summary.iterrows():
        years = "  ".join(f"{y}: ${row[f'baseline_{y}_usd']:,.0f} / ${row[f'framework_{y}_usd']:,.0f} / {row[f'diferencia_{y}_pct']:+.1f}%" for y in framework.index)
        print(f"   {row['grano'][:28]:<28} {int(row['ventana_meses']):>2}m  {years}   {row['holdout_lag1_abs_pct']:.1f}% ({row['holdout_lag1_sesgo_pct']:+.1f})"
              f" · {row['holdout_lag4_abs_pct']:.1f}% ({row['holdout_lag4_sesgo_pct']:+.1f})")
    return dict(baseline_forecast=forecast, baseline_summary=summary, baseline_holdout=holdout)
