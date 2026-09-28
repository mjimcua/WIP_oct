"""analysis_portfolio_overview.py — SFF v3 · the panorama of the whole portfolio (chapter 0).

Before any model, show the whole portfolio and look at the RIGHT seasonality:
  · the PIPELINE (how much falls due each month) is seasonal because it comes from
    acquisition (Black Friday, a weak summer);
  · the RENEWAL RATE, which is what we forecast, need not be: almost everything is
    autorenewal;
  · the aggregate rate can move with nobody changing behaviour, only because the
    proportions change (composition) or because the portfolio grows or shrinks.
This is the aggregate view BEFORE the study of seasonality on the big series
(decision_estacionalidad): first it is seen in the aggregate, then confirmed series by
series.

Tables:
  panorama_resumen      per year × role: pipeline units and $, projected $ — how much
                        there is to train on, to examine, to project, and how much the
                        projection is
  panorama_mensual      the aggregate monthly series: pipeline units and $; the
                        ENTRIES of 12 months earlier (renewals of one-year licences,
                        shifted 12 months: if a pipeline peak has no entries peak a year
                        before, it is not acquisition seasonality but a moved portfolio);
                        observed rate; rate at FIXED composition (the rate of every
                        mandatory cell of the month weighted with the proportions of a
                        reference year: observed − fixed = the mix effect); binomial band
                        of the observed rate ±1.96·√(p(1−p)/n)
  panorama_mercado      the same by market (regional_level_1): mean rate per year, change
                        between years, seasonal amplitude, months outside the binomial
                        band, % of the portfolio's $
  panorama_diagnostico  one row per aggregate series (total and every market; pipeline
                        and rate): trend (pp/year or %/year), seasonality (amplitude of
                        the mean monthly profile and whether the pattern repeats: same
                        sign in ≥ 2 years; the thresholds of decision_estacionalidad),
                        months outside the band with no repeating pattern, verdict:
                        estacional / tendencia / estable / fluctua_sin_patron
  panorama_volumen_ts   WHERE TIME SERIES DO WORK: the pipeline units of each market
                        forecast with Holt-Winters (additive, on log units) and with
                        the seasonal index on the recent level, backtested on the last
                        12 closed months against the flat recent level. The season of
                        the volume is real; the report says by how much a seasonal
                        method beats a flat one.
Figures of chapter 0 (titles that state the finding): monthly pipeline with the shifted
entries; pipeline with and without the markets flagged as moved; observed vs fixed-
composition rate with the binomial band; rate by market (small multiples); the volume
forecast against the truth.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from analysis_composition import decompose_rate_change
from techniques import CATALOGUE, month_numbers_of

# ─── named constants ─────────────────────────────────────────────────────────────
BAND_Z = 1.96                           # the band of the aggregate rate: 95 %
SEASON_AMPLITUDE_PP = 2.0               # the same threshold as decision_estacionalidad (rate)
SEASON_AMPLITUDE_VOLUME_PCT = 15.0      # the pipeline is seasonal beyond ±15 % of its mean
MOVED_PORTFOLIO_RATIO = 1.5             # a pipeline peak with entries(t−12) × this or less is "moved"
MIN_MONTHS_FOR_DIAGNOSIS = 18
VOLUME_BACKTEST_MONTHS = 12


def monthly_aggregate(units: pd.DataFrame, configuration: Config, scope_column: str = None) -> pd.DataFrame:
    """The aggregate monthly series (total, or per value of scope_column): units, $,
    renewed, rate, binomial band, entries of 12 months earlier and the role of the month."""
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren, usd = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.pipeline_usd_col
    frame = units[units.get("sintetica", 0) == 0]
    keys = [period] + ([scope_column] if scope_column else [])
    monthly = frame.groupby(keys).agg(unidades=(pipe, "sum"), usd=(usd, "sum"), renovados=(ren, "sum"), rol=(role, "first")).reset_index()
    monthly["tasa"] = np.where(monthly["rol"].isin(TRUTH_ROLES) & (monthly["unidades"] > 0), monthly["renovados"] / monthly["unidades"].replace(0, np.nan), np.nan)
    se = np.sqrt(np.maximum(monthly["tasa"] * (1 - monthly["tasa"]), 0.0475) / monthly["unidades"].clip(lower=1))
    monthly["banda_low"], monthly["banda_high"] = monthly["tasa"] - BAND_Z * se, monthly["tasa"] + BAND_Z * se
    # entries of 12 months earlier: the renewals of the month 12 months before (one-year licences)
    shifted = monthly[keys + ["renovados"]].copy()
    shifted[period] = shifted[period] + 12
    shifted = shifted.rename(columns={"renovados": "entradas_hace_12m"})
    monthly = monthly.merge(shifted, on=keys, how="left")
    monthly["ratio_pipeline_entradas"] = monthly["unidades"] / monthly["entradas_hace_12m"].replace(0, np.nan)
    return monthly


def fixed_composition_rate(units: pd.DataFrame, configuration: Config, reference_year: int) -> pd.Series:
    """The rate of every closed month at the composition of the reference year: the rate of
    each mandatory cell of the month, weighted with the cell's share of the pipeline in the
    reference year. Uses the one decomposition function (observed − composition effect of
    the month against the reference-year composition)."""
    period, role = configuration.period_col, configuration.dataset_role_col
    frame = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES)].copy()
    # every month is compared with the reference year AS A WHOLE: the reference rows are the
    # year's rows relabelled "ref" (a month of the reference year appears in both sides of
    # its own comparison, which is what "at the composition of the year" means)
    reference_rows = frame[pd.PeriodIndex(frame[period], freq="M").year == reference_year].assign(_ref="ref")
    result = {}
    cells = list(configuration.business_mandatory_dims)
    for month in sorted(frame[period].unique()):
        month_rows = frame[frame[period] == month].assign(_ref=str(month))
        pair = pd.concat([reference_rows, month_rows], ignore_index=True)
        summary, _ = decompose_rate_change(pair, cells, "ref", str(month), configuration, period_column="_ref")
        result[month] = summary["tasa_1"] - summary["composicion_pp"] / 100 if pd.notna(summary["composicion_pp"]) else np.nan
    return pd.Series(result, name="tasa_composicion_fija")


def diagnose_series(values: pd.Series, months: pd.PeriodIndex, kind: str, band_low: pd.Series = None, band_high: pd.Series = None) -> dict:
    """Trend, seasonality and out-of-band fluctuations of one aggregate series.
    kind = 'tasa' (pp) or 'pipeline' (% of the mean)."""
    valid = values.notna()
    y, m = values[valid].to_numpy(dtype=float), months[valid]
    if len(y) < MIN_MONTHS_FOR_DIAGNOSIS:
        return dict(meses=int(len(y)), tendencia_anual=np.nan, amplitud=np.nan, patron_repetido=0, meses_fuera_banda=np.nan, veredicto="historia_corta")
    scale = 100.0 if kind == "tasa" else 100.0 / max(np.mean(y), 1e-9)          # pp, or % of the mean
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    detrended = y - (intercept + slope * x)
    by_month = pd.DataFrame(dict(mes=m.month, anio=m.year, dev=detrended))
    profile = by_month.groupby("mes")["dev"].mean()
    amplitude = scale * float(profile.max() - profile.min())
    # the pattern repeats when the extreme months keep their sign in ≥ 2 years
    high, low = int(profile.idxmax()), int(profile.idxmin())
    repeats = int((by_month.loc[by_month["mes"] == high, "dev"] > 0).sum() >= 2 and (by_month.loc[by_month["mes"] == low, "dev"] < 0).sum() >= 2)
    outside = np.nan
    if band_low is not None and band_high is not None:
        inside = (values >= band_low.rolling(3, center=True, min_periods=1).mean()) & (values <= band_high.rolling(3, center=True, min_periods=1).mean())
        outside = int((~inside[valid]).sum())
    trend_per_year = scale * slope * 12
    threshold = SEASON_AMPLITUDE_PP if kind == "tasa" else SEASON_AMPLITUDE_VOLUME_PCT
    if amplitude >= threshold and repeats:
        verdict = "estacional"
    elif abs(trend_per_year) >= (1.0 if kind == "tasa" else 5.0) and abs(trend_per_year) >= 2 * scale * float(np.std(detrended, ddof=1)) / np.sqrt(len(y)) * 12 / max(len(y), 1):
        verdict = "tendencia"
    elif np.isfinite(outside) and outside > len(y) * 0.3:
        verdict = "fluctua_sin_patron"
    else:
        verdict = "estable"
    return dict(meses=int(len(y)), tendencia_anual=round(trend_per_year, 2), amplitud=round(amplitude, 2), patron_repetido=repeats,
                mes_alto=high, mes_bajo=low, meses_fuera_banda=outside, veredicto=verdict)


def volume_time_series_check(monthly: pd.DataFrame, configuration: Config, scope: str) -> list:
    """WHERE TIME SERIES DO WORK: forecast the pipeline units of the last 12 closed months,
    each from an origin 3 months before, with (a) the flat recent level (T3_ma3 on log
    units), (b) the seasonal index on the recent level (T15), (c) Holt-Winters (T11); the
    error in % of the units. Rows for panorama_volumen_ts."""
    closed = monthly[monthly["rol"].isin(TRUTH_ROLES) & (monthly["unidades"] > 0)].sort_values(configuration.period_col)
    if len(closed) < 30:
        return []
    log_units = np.log(closed["unidades"].to_numpy(dtype=float))
    months = month_numbers_of(pd.PeriodIndex(closed[configuration.period_col], freq="M"))
    rows = []
    for technique in ("T3_ma3", "T15_level_seasonal", "T11_holt_winters"):
        errors = []
        for target in range(len(closed) - VOLUME_BACKTEST_MONTHS, len(closed)):
            origin = target - 3
            if origin + 1 < 24:
                continue
            # the techniques work on any real-valued series (they are written for the logit of a rate);
            # here they receive the LOG of the units, so they are called directly, not through predict()
            technique_function = CATALOGUE[technique][4]
            predicted = float(technique_function(log_units[:origin + 1], months[:origin + 1], 3, dict(estacional=1)))
            errors.append(abs(np.exp(predicted) - closed["unidades"].iloc[target]) / closed["unidades"].iloc[target])
        if errors:
            rows.append(dict(ambito=scope, tecnica=technique, meses=len(errors), error_medio_pct=round(100 * float(np.mean(errors)), 2), horizonte=3))
    return rows


def run_portfolio_overview(units: pd.DataFrame, configuration: Config) -> dict:
    """The panorama end to end. Persists the five tables; returns them with the monthly
    series per market for the figures."""
    period, role = configuration.period_col, configuration.dataset_role_col
    region_dim = configuration.business_mandatory_dims[0]
    frame = units[units.get("sintetica", 0) == 0]
    years = pd.PeriodIndex(frame[period], freq="M").year
    # 1 · summary per year × role
    summary = frame.assign(anio=years).groupby(["anio", role]).agg(unidades=(configuration.pipeline_units_col, "sum"), usd=(configuration.pipeline_usd_col, "sum")).reset_index()
    summary["usd_proyectado"] = np.where(summary[role].isin(FUTURE_ROLES), summary["usd"], 0.0)
    configuration.write(summary, "panorama_resumen")
    # 2 · monthly aggregate + fixed composition
    monthly = monthly_aggregate(frame, configuration)
    closed_years = sorted(set(pd.PeriodIndex(frame.loc[frame[role].isin(TRUTH_ROLES), period], freq="M").year))
    full_years = [y for y in closed_years if (pd.PeriodIndex(frame.loc[frame[role].isin(TRUTH_ROLES), period], freq="M").year == y).sum() >= 12 * 0 and
                  frame.loc[frame[role].isin(TRUTH_ROLES) & (pd.PeriodIndex(frame[period], freq="M").year == y), period].nunique() == 12]
    reference_year = configuration.panorama_reference_year or (full_years[-1] if full_years else closed_years[-1])
    fixed = fixed_composition_rate(frame, configuration, reference_year)
    monthly["tasa_composicion_fija"] = monthly[period].map(fixed)
    monthly["efecto_mix_pp"] = 100 * (monthly["tasa"] - monthly["tasa_composicion_fija"])
    monthly["anio_referencia"] = reference_year
    monthly_out = monthly.copy(); monthly_out[period] = monthly_out[period].astype(str)
    configuration.write(monthly_out, "panorama_mensual")
    # 3 · markets
    by_market = monthly_aggregate(frame, configuration, region_dim)
    market_rows, diagnostics = [], []
    total_usd = float(by_market.loc[by_market["rol"].isin(FUTURE_ROLES), "usd"].sum())
    for market, block in by_market.groupby(region_dim):
        block = block.sort_values(period)
        closed = block[block["rol"].isin(TRUTH_ROLES)]
        idx = pd.PeriodIndex(closed[period], freq="M")
        rate_diag = diagnose_series(closed["tasa"].reset_index(drop=True), idx, "tasa", closed["banda_low"].reset_index(drop=True), closed["banda_high"].reset_index(drop=True))
        volume_diag = diagnose_series(closed["unidades"].astype(float).reset_index(drop=True), idx, "pipeline")
        per_year = closed.assign(anio=idx.year).groupby("anio").apply(lambda g: g["renovados"].sum() / max(g["unidades"].sum(), 1), include_groups=False)
        peaks = closed[closed["unidades"] >= closed["unidades"].quantile(0.8)]
        moved = int((peaks["ratio_pipeline_entradas"] > MOVED_PORTFOLIO_RATIO).mean() >= 0.5) if peaks["ratio_pipeline_entradas"].notna().any() else 0
        market_rows.append(dict(mercado=str(market), pct_usd_cartera=round(100 * float(block.loc[block["rol"].isin(FUTURE_ROLES), "usd"].sum()) / max(total_usd, 1e-9), 1),
                                **{f"tasa_{y}": round(float(r), 4) for y, r in per_year.items()},
                                variacion_ultimo_ano_pp=round(100 * (per_year.iloc[-1] - per_year.iloc[-2]), 2) if len(per_year) > 1 else np.nan,
                                amplitud_estacional_tasa_pp=rate_diag.get("amplitud"), meses_fuera_banda=rate_diag.get("meses_fuera_banda"),
                                cartera_movida=moved))
        diagnostics.append(dict(ambito=str(market), serie="tasa", **rate_diag))
        diagnostics.append(dict(ambito=str(market), serie="pipeline", **volume_diag, cartera_movida=moved))
    closed_total = monthly[monthly["rol"].isin(TRUTH_ROLES)].sort_values(period)
    idx = pd.PeriodIndex(closed_total[period], freq="M")
    diagnostics.insert(0, dict(ambito="total", serie="pipeline", **diagnose_series(closed_total["unidades"].astype(float).reset_index(drop=True), idx, "pipeline")))
    diagnostics.insert(0, dict(ambito="total", serie="tasa", **diagnose_series(closed_total["tasa"].reset_index(drop=True), idx, "tasa", closed_total["banda_low"].reset_index(drop=True), closed_total["banda_high"].reset_index(drop=True))))
    markets = pd.DataFrame(market_rows)
    diagnosis = pd.DataFrame(diagnostics)
    configuration.write(markets, "panorama_mercado")
    configuration.write(diagnosis, "panorama_diagnostico")
    # 5 · time series on the volume
    ts_rows = volume_time_series_check(monthly, configuration, "total")
    for market, block in by_market.groupby(region_dim):
        ts_rows.extend(volume_time_series_check(block, configuration, str(market)))
    volume_ts = pd.DataFrame(ts_rows, columns=["ambito", "tecnica", "meses", "error_medio_pct", "horizonte"])
    configuration.write(volume_ts, "panorama_volumen_ts")
    print_overview(summary, monthly, markets, diagnosis, volume_ts, reference_year, configuration)
    by_market_out = by_market.copy(); by_market_out[period] = by_market_out[period].astype(str)
    return dict(panorama_resumen=summary, panorama_mensual=monthly, panorama_mercado=markets, panorama_diagnostico=diagnosis,
                panorama_volumen_ts=volume_ts, por_mercado=by_market_out, anio_referencia=reference_year)


def print_overview(summary, monthly, markets, diagnosis, volume_ts, reference_year, configuration) -> None:
    print("[0.5] PANORAMA · the whole portfolio before any model")
    for (year, role), block in summary.groupby(["anio", configuration.dataset_role_col]):
        print(f"   {year} {role:<18} {block['unidades'].sum():>10,.0f} units  ${block['usd'].sum():>14,.0f}")
    total_rate, total_pipe = diagnosis.iloc[0], diagnosis.iloc[1]
    print(f"   TOTAL rate: {total_rate['veredicto']} (trend {total_rate['tendencia_anual']} pp/yr, amplitude {total_rate['amplitud']} pp, {total_rate['meses_fuera_banda']} months outside the binomial band) · "
          f"TOTAL pipeline: {total_pipe['veredicto']} (trend {total_pipe['tendencia_anual']} %/yr, amplitude {total_pipe['amplitud']} % of the mean)")
    mix = monthly["efecto_mix_pp"].dropna()
    if len(mix):
        print(f"   rate at fixed composition (reference year {reference_year}): the mix moves the observed rate by up to {mix.abs().max():.1f} pp (mean |mix| {mix.abs().mean():.1f} pp)")
    moved = markets[markets["cartera_movida"] == 1]["mercado"].tolist() if len(markets) else []
    print(f"   markets: {len(markets)} · flagged as moved portfolio (pipeline peaks without an entries peak a year before): {moved if moved else 'none'}")
    if len(volume_ts):
        pivot = volume_ts[volume_ts["ambito"] == "total"].set_index("tecnica")["error_medio_pct"] if (volume_ts["ambito"] == "total").any() else volume_ts.groupby("tecnica")["error_medio_pct"].mean()
        print("   VOLUME (units due), 3 months ahead over the last 12 closed months, mean |error| %: " + " · ".join(f"{t}: {e:.1f}%" for t, e in pivot.items()))
    explain(configuration,
            "The pipeline is seasonal because acquisition is (Black Friday, a weak summer); the RATE need not be: almost everything is autorenewal.",
            "Rate at fixed composition = each month's cells weighted as in the reference year: the difference to the observed rate is the mix effect, not behaviour.",
            "The binomial band of the aggregate is very narrow (n = 50,000 → ±0.4 pp): a rate that leaves it without a repeating pattern is telling that something moves underneath (mix, signals, portfolio moves).",
            "Time series techniques are judged HERE, on the volume, where the season is real; on the rate the benchmark of chapter 4 decides.")
