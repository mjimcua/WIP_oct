"""analysis_seasonality_benchmark.py — SFF v3 · ANALYSIS · the seasonality benchmark on big series.

The hypothesis to test: the month-to-month variation of the aggregate renewal rate comes
from COMPOSITION (which contracts fall due each month), not from BEHAVIOUR (how a given
cohort renews). If that holds, the calendar dynamics of the RATE are not material and the
rate branch simplifies to "recent level (with a damped trend where there is one)"; the
time-series requirement moves whole to VOLUME (acquisition, portfolio peaks).

The test is run where it has power and where composition is controlled: the biggest
NEUTRAL series (no flag active), fully segmented by every dimension, with a monthly
support of at least 271 (a binomial floor of ±5 pp or better) and 36 months of history
(24 admitted, marked `historia_corta`). Whatever moves there is behaviour. If no material
seasonality shows there, it is assumed not material in the rest, where there is no
support to estimate it anyway.

ANALYSIS only. Nothing in RUN, uplift or volume changes until the decision table exists.

Deliveries (one function each, reviewed before the next):
  1. select_benchmark_series     the sample and its share of the projected pipeline   [this file, now]
  2. floor_and_phi               binomial floor and φ after removing a linear trend
  3. month_effect_regression     logit(rate) ~ trend + month dummies, weighted by n: amplitude, LRT, slope ± se
  4. month_year_panel            standardized residuals z after trend, month × year; consistency of sign
  5. predictive_backtest         T15 (same-month shape on the recent level) vs T3_ma3 at h=1 and h=6, money-weighted
  6. materiality_and_verdict     amplitude × pipeline of the extreme months in $; the decision table
  7. flag_composition_check      share of signed units by month-of-year inside the same cells
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
NEUTRAL_SIGN = SIGN_NEUTRAL
TOP_N_PER_GROUP = 5
MIN_MONTHLY_SUPPORT = 271.0          # the dial at ±5 pp (90 %, p = 0.5)
MIN_HISTORY_MONTHS = 36
SHORT_HISTORY_MONTHS = 24


def select_benchmark_series(series_card: pd.DataFrame, units: pd.DataFrame, configuration: Config,
                            group_dims: list = None, top_n: int = TOP_N_PER_GROUP,
                            min_support: float = MIN_MONTHLY_SUPPORT, min_months: int = MIN_HISTORY_MONTHS,
                            short_months: int = SHORT_HISTORY_MONTHS) -> pd.DataFrame:
    """Delivery 1: the sample of big neutral series, top N by projected money inside each group.

    INPUT:   series_card (fase 1.3: fs_id, signo, ruta, n_propio, meses_historia, usd_proyectado)
             · units (fase 1.1: one row per unit with the dimensions, roles, pipeline units)
             · configuration · group_dims — the dimensions that define a group (default: the
             first two mandatory dims, e.g. regional_level_1 × product_level_1) · top_n ·
             min_support (median AND minimum monthly pipeline units over the closed months)
             · min_months · short_months.
    OUTPUT:  one row per selected series: fs_id, grupo, usd_proyectado, n_mediana, n_minimo,
             meses_historia, historia_corta (0/1), rango_desde, rango_hasta, orden_en_grupo,
             pct_pipeline_grupo (share of the group's projected money it carries).
             The frame carries `attrs["cobertura"]`: the share of the whole projected
             pipeline covered by the sample, and the console prints it.
    RULES:   neutral series only (signed ones go to the flag check, delivery 7); route
             trainable; the minimum monthly support must hold in EVERY closed month used
             (not only the median), so the binomial floor is ±5 pp or better throughout;
             months of history counted over the closed months (train + test); series with
             short_months ≤ months < min_months are admitted with historia_corta = 1;
             below short_months they are out. Top N by usd_proyectado inside each group so
             no single market concentrates the sample.
    EDGE CASES: a group with fewer than top_n qualifying series contributes what it has;
             a group with none contributes nothing and is listed in the console.
    CONSOLE: series selected, groups covered, % of projected pipeline covered, groups with
             no qualifying series.
    STEPS:
      [1] The candidates: neutral, trainable, enough months.
      [2] The support of every candidate month by month (median and minimum).
      [3] Top N by money inside each group.
      [4] The coverage of the sample.
    """
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe_units = configuration.pipeline_units_col
    dims = group_dims or configuration.business_mandatory_dims[:2]

    # [1] candidates
    candidates = series_card[(series_card["signo"] == NEUTRAL_SIGN) & (series_card["ruta"] == TREATMENT_PREDICTABLE)
                             & (series_card["meses_historia"] >= short_months)].copy()

    # [2] monthly support over the closed months: median and minimum per series
    closed = units[(units[role].isin(TRUTH_ROLES)) & (units.get("sintetica", 0) == 0) & units["fs_id"].isin(candidates["fs_id"])]
    support = closed.groupby("fs_id")[pipe_units].agg(n_mediana="median", n_minimo="min")
    span = closed.groupby("fs_id")[period].agg(rango_desde="min", rango_hasta="max")
    candidates = candidates.merge(support, on="fs_id", how="left").merge(span, on="fs_id", how="left")
    candidates = candidates[(candidates["n_mediana"] >= min_support) & (candidates["n_minimo"] >= min_support)]
    candidates["historia_corta"] = (candidates["meses_historia"] < min_months).astype(int)

    # [3] the group of every series and the top N inside it
    one_per_series = units.drop_duplicates("fs_id").set_index("fs_id")
    candidates["grupo"] = join_columns(one_per_series.loc[candidates["fs_id"], dims].reset_index(drop=True), dims).to_numpy()
    candidates = candidates.sort_values(["grupo", "usd_proyectado"], ascending=[True, False])
    candidates["orden_en_grupo"] = candidates.groupby("grupo").cumcount() + 1
    selected = candidates[candidates["orden_en_grupo"] <= top_n].copy()

    # [4] coverage
    all_groups = join_columns(one_per_series[dims], dims)
    group_money = series_card.assign(grupo=series_card["fs_id"].map(all_groups)).groupby("grupo")["usd_proyectado"].sum()
    selected["pct_pipeline_grupo"] = (100 * selected["usd_proyectado"] / selected["grupo"].map(group_money).replace(0, np.nan)).round(1)
    total_projected = float(series_card["usd_proyectado"].sum())
    coverage = 100 * float(selected["usd_proyectado"].sum()) / total_projected if total_projected > 0 else 0.0
    empty_groups = sorted(set(group_money.index) - set(selected["grupo"]))
    selected = selected[["fs_id", "grupo", "usd_proyectado", "n_mediana", "n_minimo", "meses_historia", "historia_corta",
                         "rango_desde", "rango_hasta", "orden_en_grupo", "pct_pipeline_grupo"]].reset_index(drop=True)
    selected["rango_desde"], selected["rango_hasta"] = selected["rango_desde"].astype(str), selected["rango_hasta"].astype(str)
    selected.attrs["cobertura"] = round(coverage, 1)
    print(f"[bench 1] {len(selected)} series selected (top {top_n} by projected money inside {selected['grupo'].nunique()} groups of "
          f"{' × '.join(dims)}; neutral, support ≥ {min_support:.0f} in every closed month, ≥ {short_months} months) · "
          f"they carry {coverage:.1f}% of the projected pipeline · {int(selected['historia_corta'].sum())} with short history")
    if empty_groups:
        print(f"[bench 1] groups with no qualifying series: {len(empty_groups)} ({', '.join(empty_groups[:8])}{'…' if len(empty_groups) > 8 else ''})")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════════
# 2 · FLOOR AND φ
# ═══════════════════════════════════════════════════════════════════════════════════

def monthly_series_of(units: pd.DataFrame, series_id: str, configuration: Config) -> pd.DataFrame:
    """The closed months of one series: index Period, columns ren, pipe, rate."""
    closed = units[(units["fs_id"] == series_id) & (units[configuration.dataset_role_col].isin(TRUTH_ROLES))
                   & (units.get("sintetica", 0) == 0)]
    monthly = closed.groupby(configuration.period_col).agg(ren=(configuration.renewed_units_col, "sum"),
                                                          pipe=(configuration.pipeline_units_col, "sum")).sort_index()
    monthly = monthly[monthly["pipe"] > 0]
    monthly["rate"] = monthly["ren"] / monthly["pipe"]
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly


def linear_trend(monthly: pd.DataFrame) -> tuple:
    """Weighted (by n) linear fit of the rate on time. Returns (fitted values, slope per month, intercept)."""
    x = np.arange(len(monthly), dtype=float)
    weights = np.sqrt(monthly["pipe"].to_numpy(dtype=float))
    design = np.vstack([np.ones_like(x), x]).T
    coefficients = np.linalg.lstsq(design * weights[:, None], monthly["rate"].to_numpy(dtype=float) * weights, rcond=None)[0]
    return design @ coefficients, float(coefficients[1]), float(coefficients[0])


def floor_and_phi(monthly: pd.DataFrame) -> dict:
    """Delivery 2: the binomial floor of the series and its overdispersion after a linear trend.

    INPUT:   monthly — from `monthly_series_of` (closed months, pipe > 0).
    OUTPUT:  dict(meses, n_mediana, tasa_media, se_binomial_pp (typical month, at the pooled
             rate), sd_observada_pp (of the rate around its linear trend), phi (observed
             variance around the trend / binomial variance, using each month's own n)).
    RULES:   the trend is removed first so a slope is not counted as dispersion; the
             binomial variance is p̄(1−p̄)·mean(1/n_t). φ ≈ 1: the rate only samples;
             φ ≫ 1: something moves it (season, level changes, campaigns) — φ says HOW
             MUCH beyond sampling, not WHAT. Below 6 months φ is NaN.
    EDGE CASES: constant pipeline and constant rate → variance 0 → φ = 0.
    """
    supports = monthly["pipe"].to_numpy(dtype=float)
    rates = monthly["rate"].to_numpy(dtype=float)
    pooled = float(monthly["ren"].sum() / monthly["pipe"].sum())
    typical_n = float(np.median(supports))
    se_typical = float(np.sqrt(max(pooled * (1 - pooled), 0.0475) / max(typical_n, 1.0)))
    if len(monthly) < 6:
        return dict(meses=len(monthly), n_mediana=typical_n, tasa_media=pooled, se_binomial_pp=100 * se_typical,
                    sd_observada_pp=np.nan, phi=np.nan)
    fitted, _, _ = linear_trend(monthly)
    residuals = rates - fitted
    observed_variance = float(np.var(residuals, ddof=2))
    binomial_variance = pooled * (1 - pooled) * float(np.mean(1.0 / supports))
    phi = observed_variance / binomial_variance if binomial_variance > 0 else np.nan
    return dict(meses=len(monthly), n_mediana=typical_n, tasa_media=pooled, se_binomial_pp=100 * se_typical,
                sd_observada_pp=100 * float(np.sqrt(observed_variance)), phi=phi)


# ═══════════════════════════════════════════════════════════════════════════════════
# 3 · MONTH-EFFECT REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════════

def month_effect_regression(monthly: pd.DataFrame) -> dict:
    """Delivery 3: logit(rate) ~ trend + month-of-year dummies, weighted by n.

    INPUT:   monthly — closed months with pipe > 0 (≥ 24 months for the dummies to mean anything).
    OUTPUT:  dict(amplitud_pp (effect of the highest month − the lowest, in pp at the series'
             mean rate), mes_alto, mes_bajo, lrt (Gaussian likelihood-ratio statistic of the
             dummies: T·log(RSS_sin/RSS_con), 11 df; reported, never decides),
             lrt_pvalue, pendiente_pp_anio (trend slope in pp per year at the mean rate),
             pendiente_se_pp_anio (its standard error), efectos_mes (dict month → pp)).
    RULES:   weights = n (a month with 5,000 contracts weighs more than one with 500).
             Everything is DESCRIPTIVE: with thousands of contracts the LRT is always
             significant; what decides is amplitude in pp and dollars, consistency across
             years (delivery 4) and the predictive backtest (delivery 5).
    EDGE CASES: a calendar month never observed has no dummy (effect 0); fewer than 14
             months → amplitude NaN (the model is saturated).
    """
    from binomial_reference import logit
    rates = np.clip(monthly["rate"].to_numpy(dtype=float), 1e-4, 1 - 1e-4)
    y = logit(rates)
    n = monthly["pipe"].to_numpy(dtype=float)
    weights = np.sqrt(n)
    x = np.arange(len(monthly), dtype=float)
    months = monthly.index.month
    present = sorted(set(months))
    if len(monthly) < 14:
        return dict(amplitud_pp=np.nan, mes_alto=np.nan, mes_bajo=np.nan, lrt=np.nan, lrt_pvalue=np.nan,
                    pendiente_pp_anio=np.nan, pendiente_se_pp_anio=np.nan, efectos_mes={})
    base = np.vstack([np.ones_like(x), x]).T
    dummies = np.column_stack([(months == m).astype(float) for m in present[1:]])      # first present month = reference
    full = np.hstack([base, dummies])

    def weighted_fit(design):
        coefficients, _, _, _ = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)
        rss = float(np.sum((weights * (y - design @ coefficients)) ** 2))
        return coefficients, rss
    coefficients_base, rss_base = weighted_fit(base)
    coefficients_full, rss_full = weighted_fit(full)
    lrt = len(y) * np.log(rss_base / rss_full) if rss_full > 0 else np.inf
    try:
        from scipy import stats
        lrt_pvalue = float(stats.chi2.sf(lrt, df=max(len(present) - 1, 1)))
    except Exception:
        lrt_pvalue = np.nan
    mean_rate = float(np.average(rates, weights=n))
    to_pp = 100 * mean_rate * (1 - mean_rate)                     # d rate / d logit at the mean
    effects = {present[0]: 0.0}
    for month, coefficient in zip(present[1:], coefficients_full[2:]):
        effects[month] = float(coefficient)
    centre = float(np.mean(list(effects.values())))
    effects_pp = {m: to_pp * (e - centre) for m, e in effects.items()}
    slope_per_month = float(coefficients_full[1])
    residual_variance = rss_full / max(len(y) - full.shape[1], 1)
    covariance = residual_variance * np.linalg.pinv((full * weights[:, None]).T @ (full * weights[:, None]))
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return dict(amplitud_pp=max(effects_pp.values()) - min(effects_pp.values()),
                mes_alto=max(effects_pp, key=effects_pp.get), mes_bajo=min(effects_pp, key=effects_pp.get),
                lrt=float(lrt), lrt_pvalue=lrt_pvalue,
                pendiente_pp_anio=12 * to_pp * slope_per_month, pendiente_se_pp_anio=12 * to_pp * slope_se,
                efectos_mes={int(m): round(v, 2) for m, v in effects_pp.items()})


# ═══════════════════════════════════════════════════════════════════════════════════
# 4 · MONTH × YEAR PANEL
# ═══════════════════════════════════════════════════════════════════════════════════

def month_year_panel(monthly: pd.DataFrame) -> tuple:
    """Delivery 4: standardized residuals after the linear trend (no dummies), month × year.

    OUTPUT:  (panel: DataFrame(mes, anio, z, tasa, n), consistency: dict month → fraction
             of years in which z has the majority sign, and the sign).
    RULES:   z = (rate − trend) / binomial se of that month with its own n. A month that
             is "high" is high in most years if seasonality is real; a month that is high
             one year and low the next is noise, whatever the aggregate ANOVA says.
    """
    fitted, _, _ = linear_trend(monthly)
    rates = monthly["rate"].to_numpy(dtype=float)
    n = monthly["pipe"].to_numpy(dtype=float)
    pooled = float(monthly["ren"].sum() / monthly["pipe"].sum())
    se = np.sqrt(np.maximum(pooled * (1 - pooled), 0.0475) / np.maximum(n, 1.0))
    panel = pd.DataFrame(dict(mes=monthly.index.month, anio=monthly.index.year, z=(rates - fitted) / se, tasa=rates, n=n))
    consistency = {}
    for month, block in panel.groupby("mes"):
        signs = np.sign(block["z"].to_numpy())
        signs = signs[signs != 0]
        if len(signs) == 0:
            consistency[int(month)] = (np.nan, 0)
            continue
        positive = float(np.mean(signs > 0))
        majority = 1 if positive >= 0.5 else -1
        consistency[int(month)] = (max(positive, 1 - positive), majority)
    return panel, consistency


# ═══════════════════════════════════════════════════════════════════════════════════
# 5 · PREDICTIVE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════════

def predictive_backtest(monthly: pd.DataFrame, targets: int = 6, horizons: tuple = (1, 6)) -> dict:
    """Delivery 5: the seasonal shape on the recent level (T15) against the recent level
    alone (T3_ma3), on the last `targets` closed months, at h = 1 and h = 6, weighted by n.

    OUTPUT:  dict(err_ma3_h1_pp, err_t15_h1_pp, mejora_h1_pct, err_ma3_h6_pp, err_t15_h6_pp,
             mejora_h6_pct). mejora = (err_ma3 − err_t15) / err_ma3 × 100: positive when
             the shape helps. This is the test that DECIDES.
    RULES:   each target month is predicted from an origin h months before with only the
             history up to the origin; both techniques see exactly the same data.
    EDGE CASES: fewer than 13 months before an origin → the seasonal technique cannot be
             built for that target and it is skipped (both techniques skip it).
    """
    from techniques import predict, month_numbers_of
    rates = monthly["rate"].to_numpy(dtype=float)
    n = monthly["pipe"].to_numpy(dtype=float)
    months = month_numbers_of(monthly.index)
    result = {}
    for h in horizons:
        errors = {"T3_ma3": [], "T15_level_seasonal": []}
        weights = []
        for target in range(max(len(rates) - targets, 0), len(rates)):
            origin = target - h
            if origin + 1 < 13:
                continue
            history_rates, history_months = rates[:origin + 1], months[:origin + 1]
            for technique in errors:
                predicted = predict(technique, history_rates, history_months, h, dict(estacional=1))
                errors[technique].append(abs(predicted - rates[target]))
            weights.append(n[target])
        if not weights:
            result.update({f"err_ma3_h{h}_pp": np.nan, f"err_t15_h{h}_pp": np.nan, f"mejora_h{h}_pct": np.nan})
            continue
        err_ma3 = 100 * float(np.average(errors["T3_ma3"], weights=weights))
        err_t15 = 100 * float(np.average(errors["T15_level_seasonal"], weights=weights))
        result.update({f"err_ma3_h{h}_pp": err_ma3, f"err_t15_h{h}_pp": err_t15,
                       f"mejora_h{h}_pct": 100 * (err_ma3 - err_t15) / err_ma3 if err_ma3 > 0 else np.nan})
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# 6 · MATERIALITY AND VERDICT
# ═══════════════════════════════════════════════════════════════════════════════════

def materiality_and_verdict(series_id: str, monthly: pd.DataFrame, future_units: pd.DataFrame, configuration: Config,
                            amplitude_threshold_pp: float, consistency_threshold: float, improvement_threshold_pct: float,
                            min_years: int, min_extreme_z: float = 1.0) -> dict:
    """Delivery 6: everything of one series in one row, with the verdict.

    INPUT:   series_id · monthly (closed months) · future_units (the series' future rows:
             projection + pending, with period and pipeline $) · configuration · thresholds.
    OUTPUT:  the row of `decision_estacionalidad`: fs_id, meses, anios, n_mediana, tasa_media,
             se_binomial_pp, sd_observada_pp, phi, amplitud_pp, mes_alto, mes_bajo,
             consistencia_alto, consistencia_bajo, lrt, lrt_pvalue, err_ma3_h1_pp, err_t15_h1_pp,
             mejora_h1_pct, err_ma3_h6_pp, err_t15_h6_pp, mejora_h6_pct, usd_impacto,
             veredicto_estacional (0/1), pendiente_pp_anio, pendiente_se_pp_anio,
             veredicto_tendencia (−1/0/1), motivo.
    RULES:   seasonal if amplitude ≥ threshold AND both extreme months are consistent
             (≥ consistency_threshold of the years with the same sign) AND both stand out
             of the noise (mean |z| ≥ min_extreme_z sampling errors) AND the shape improves
             the recent level by ≥ improvement_threshold at h = 6 without worsening it at
             h = 1 AND at least min_years of history. Trend if
             |slope| ≥ 2 × its standard error and ≥ 1 pp/year. Significance never decides.
             usd_impacto = amplitude × the future pipeline $ of the extreme months: what
             ignoring the season would cost if it were real.
    """
    floor = floor_and_phi(monthly)
    regression = month_effect_regression(monthly)
    panel, consistency = month_year_panel(monthly)
    backtest = predictive_backtest(monthly)
    years = len(monthly) / 12.0
    high, low = regression["mes_alto"], regression["mes_bajo"]
    consistency_high = consistency.get(high, (np.nan, 0))[0] if pd.notna(high) else np.nan
    consistency_low = consistency.get(low, (np.nan, 0))[0] if pd.notna(low) else np.nan
    z_high = float(panel.loc[panel["mes"] == high, "z"].mean()) if pd.notna(high) else np.nan
    z_low = float(panel.loc[panel["mes"] == low, "z"].mean()) if pd.notna(low) else np.nan
    future_months = pd.PeriodIndex(future_units[configuration.period_col].astype(str), freq="M") if len(future_units) else pd.PeriodIndex([], freq="M")
    extreme_money = float(future_units.loc[future_months.month.isin([m for m in (high, low) if pd.notna(m)]), configuration.pipeline_usd_col].sum()) if len(future_units) else 0.0
    impact = (regression["amplitud_pp"] / 100) * extreme_money if pd.notna(regression["amplitud_pp"]) else 0.0
    reasons = []
    if pd.isna(regression["amplitud_pp"]) or regression["amplitud_pp"] < amplitude_threshold_pp:
        reasons.append(f"amplitud < {amplitude_threshold_pp} pp")
    if pd.isna(consistency_high) or consistency_high < consistency_threshold or pd.isna(consistency_low) or consistency_low < consistency_threshold:
        reasons.append("meses extremos no consistentes entre años")
    if pd.isna(z_high) or pd.isna(z_low) or abs(z_high) < min_extreme_z or abs(z_low) < min_extreme_z:
        reasons.append(f"meses extremos dentro del ruido (|z| < {min_extreme_z})")
    if pd.isna(backtest["mejora_h6_pct"]) or backtest["mejora_h6_pct"] < improvement_threshold_pct:
        reasons.append(f"la forma no mejora ≥ {improvement_threshold_pct:.0f}% a h=6")
    if pd.notna(backtest["mejora_h1_pct"]) and backtest["mejora_h1_pct"] < 0:
        reasons.append("la forma empeora a h=1")
    if years < min_years:
        reasons.append(f"< {min_years} años")
    seasonal = int(len(reasons) == 0)
    slope, slope_se = regression["pendiente_pp_anio"], regression["pendiente_se_pp_anio"]
    trend = int(np.sign(slope)) if (pd.notna(slope) and pd.notna(slope_se) and abs(slope) >= 2 * slope_se and abs(slope) >= 1.0) else 0
    row = dict(fs_id=series_id, meses=floor["meses"], anios=round(years, 1), n_mediana=floor["n_mediana"],
               tasa_media=round(floor["tasa_media"], 4), se_binomial_pp=round(floor["se_binomial_pp"], 2),
               sd_observada_pp=round(floor["sd_observada_pp"], 2) if pd.notna(floor["sd_observada_pp"]) else np.nan,
               phi=round(floor["phi"], 2) if pd.notna(floor["phi"]) else np.nan,
               amplitud_pp=round(regression["amplitud_pp"], 2) if pd.notna(regression["amplitud_pp"]) else np.nan,
               mes_alto=high, mes_bajo=low,
               consistencia_alto=round(consistency_high, 2) if pd.notna(consistency_high) else np.nan,
               consistencia_bajo=round(consistency_low, 2) if pd.notna(consistency_low) else np.nan,
               z_alto=round(z_high, 2) if pd.notna(z_high) else np.nan, z_bajo=round(z_low, 2) if pd.notna(z_low) else np.nan,
               lrt=round(regression["lrt"], 2) if pd.notna(regression["lrt"]) else np.nan,
               lrt_pvalue=round(regression["lrt_pvalue"], 4) if pd.notna(regression["lrt_pvalue"]) else np.nan,
               usd_impacto=round(impact, 2), veredicto_estacional=seasonal,
               pendiente_pp_anio=round(slope, 2) if pd.notna(slope) else np.nan,
               pendiente_se_pp_anio=round(slope_se, 2) if pd.notna(slope_se) else np.nan,
               veredicto_tendencia=trend, motivo="; ".join(reasons) if reasons else "estacional",
               efectos_mes="|".join(f"{m}:{v:+.2f}" for m, v in sorted(regression["efectos_mes"].items())))
    row.update({k: round(v, 2) if pd.notna(v) else np.nan for k, v in backtest.items()})
    return row


# ═══════════════════════════════════════════════════════════════════════════════════
# 7 · FLAG COMPOSITION CHECK
# ═══════════════════════════════════════════════════════════════════════════════════

def flag_composition_check(units: pd.DataFrame, selected: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Delivery 7: inside the mandatory cells of the benchmark series, the monthly share of
    units with an active flag, by month-of-year, with the same consistency test.

    OUTPUT:  DataFrame(celda_id, flag, mes, anios, share_medio, z_medio, consistencia):
             for every cell × flag × month-of-year, the mean share of flagged units, the
             mean standardized deviation from the cell's linear trend, and the fraction of
             years with the majority sign. If a flag's share is seasonal, the seasonality
             the neutral series do not show may live in the flags.
    PRECONDITION (declared, not checked here): the flag must be as-of the same distance
             to expiry that is used when predicting; if it is as-of the extract date, the
             benchmark rate of a flagged series already contains warnings the month to
             predict did not have.
    """
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe_units = configuration.pipeline_units_col
    cells = set(units.loc[units["fs_id"].isin(selected["fs_id"]), "celda_id"]) if "celda_id" in units.columns else set()
    if not cells:
        one = units.drop_duplicates("fs_id")
        cell_of = dict(zip(one["fs_id"], join_columns(one, configuration.business_mandatory_dims)))
        units = units.assign(celda_id=units["fs_id"].map(cell_of))
        cells = set(units.loc[units["fs_id"].isin(selected["fs_id"]), "celda_id"])
    closed = units[(units[role].isin(TRUTH_ROLES)) & (units.get("sintetica", 0) == 0) & units["celda_id"].isin(cells)]
    rows = []
    for flag in configuration.structural_timevarying_dims:
        flagged = closed[flag].isin(configuration.timevarying_positive_values)
        monthly = closed.assign(_flagged_units=closed[pipe_units].where(flagged, 0.0)).groupby(["celda_id", period]).agg(
            flagged=("_flagged_units", "sum"), total=(pipe_units, "sum")).reset_index()
        monthly["share"] = monthly["flagged"] / monthly["total"].replace(0, np.nan)
        for cell, block in monthly.groupby("celda_id"):
            block = block.sort_values(period)
            if len(block) < 14 or block["share"].isna().all():
                continue
            x = np.arange(len(block), dtype=float)
            share = block["share"].fillna(0).to_numpy()
            fitted = np.polyval(np.polyfit(x, share, 1), x)
            se = np.sqrt(np.maximum(share * (1 - share), 0.0475) / np.maximum(block["total"].to_numpy(dtype=float), 1.0))
            z = (share - fitted) / se
            panel = pd.DataFrame(dict(mes=pd.PeriodIndex(block[period], freq="M").month, z=z, share=share))
            for month, group in panel.groupby("mes"):
                signs = np.sign(group["z"].to_numpy()); signs = signs[signs != 0]
                positive = float(np.mean(signs > 0)) if len(signs) else np.nan
                rows.append(dict(celda_id=cell, flag=flag, mes=int(month), anios=int(len(group)),
                                 share_medio=round(float(group["share"].mean()), 4), z_medio=round(float(group["z"].mean()), 2),
                                 consistencia=round(max(positive, 1 - positive), 2) if pd.notna(positive) else np.nan))
    return pd.DataFrame(rows, columns=["celda_id", "flag", "mes", "anios", "share_medio", "z_medio", "consistencia"])


# ═══════════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════

def run_seasonality_benchmark(series_card: pd.DataFrame, units: pd.DataFrame, fine_table: pd.DataFrame,
                              configuration: Config) -> dict:
    """Phase 2: the benchmark end to end. Persists decision_estacionalidad, bench_panel_mes_anio
    and bench_flags; prints the one-page summary; returns the verdict for the portfolio."""
    period, role = configuration.period_col, configuration.dataset_role_col
    selected = select_benchmark_series(series_card, units, configuration, configuration.benchmark_group_dims or None,
                                       configuration.benchmark_top_n, configuration.benchmark_min_support,
                                       configuration.benchmark_min_months, configuration.benchmark_short_months)
    future = fine_table[fine_table[role].isin((ROLE_PROJECTION, ROLE_PENDING))].copy()
    future["fs_id"] = join_columns(future, configuration.rate_series_columns)
    rows, panels = [], []
    for _, series in selected.iterrows():
        monthly = monthly_series_of(units, series["fs_id"], configuration)
        row = materiality_and_verdict(series["fs_id"], monthly, future[future["fs_id"] == series["fs_id"]], configuration,
                                      configuration.benchmark_amplitude_pp, configuration.benchmark_consistency,
                                      configuration.benchmark_improvement_pct, configuration.benchmark_min_years,
                                      configuration.benchmark_min_extreme_z)
        row.update(dict(grupo=series["grupo"], usd_proyectado=series["usd_proyectado"], historia_corta=series["historia_corta"]))
        rows.append(row)
        panel, _ = month_year_panel(monthly)
        panels.append(panel.assign(fs_id=series["fs_id"]))
    decision = pd.DataFrame(rows)
    panel_table = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame(columns=["fs_id", "mes", "anio", "z", "tasa", "n"])
    flags = flag_composition_check(units, selected, configuration) if len(selected) else pd.DataFrame()
    configuration.write(decision, "decision_estacionalidad")
    configuration.write(panel_table, "bench_panel_mes_anio")
    configuration.write(flags, "bench_flags")
    summary = benchmark_summary(decision, flags, selected, configuration)
    return dict(decision_estacionalidad=decision, bench_panel=panel_table, bench_flags=flags, selected=selected, summary=summary)


def benchmark_summary(decision: pd.DataFrame, flags: pd.DataFrame, selected: pd.DataFrame, configuration: Config) -> dict:
    """The one page: % of money covered, % seasonal, % trending, flags, the decision."""
    coverage = selected.attrs.get("cobertura", np.nan) if hasattr(selected, "attrs") else np.nan
    money = decision["usd_proyectado"].sum() if len(decision) else 0.0
    seasonal = decision[decision["veredicto_estacional"] == 1] if len(decision) else decision
    trending = decision[decision["veredicto_tendencia"] != 0] if len(decision) else decision
    pct_seasonal = 100 * seasonal["usd_proyectado"].sum() / money if money else 0.0
    pct_trending = 100 * trending["usd_proyectado"].sum() / money if money else 0.0
    seasonal_flags = flags[(flags["consistencia"] >= configuration.benchmark_consistency) & (flags["z_medio"].abs() >= 1.0) & (flags["anios"] >= 2)] if len(flags) else flags
    verdict = ("sin estacionalidad material en la tasa: la tasa se predice desde su nivel"
               if pct_seasonal < configuration.benchmark_material_share_pct
               else f"estacionalidad en {len(seasonal)} series ({pct_seasonal:.0f} % del dinero de la muestra): efectos de mes solo en esas series")
    print("[2] SEASONALITY BENCHMARK · big neutral series, fully segmented (behaviour, not composition)")
    print(f"   sample: {len(decision)} series · {coverage}% of the projected pipeline · thresholds: amplitude ≥ {configuration.benchmark_amplitude_pp} pp, "
          f"consistency ≥ {configuration.benchmark_consistency:.0%}, shape must improve the recent level ≥ {configuration.benchmark_improvement_pct:.0f}% at h=6 and not worsen it at h=1")
    print(f"   seasonal: {len(seasonal)} series = {pct_seasonal:.1f}% of the sample's money · trending: {len(trending)} series = {pct_trending:.1f}% · "
          f"flags with a consistent month pattern: {len(seasonal_flags)} cell×flag×month rows")
    if len(decision):
        print("   series · φ · amplitude pp (high/low month) · consistency · shape vs level h1 / h6 · $ impact · verdict")
        for _, row in decision.sort_values("usd_proyectado", ascending=False).head(configuration.console_top_rows).iterrows():
            print(f"   {row['fs_id'][:56]:<56} φ={row['phi']:>6} amp={row['amplitud_pp']:>5} ({row['mes_alto']}/{row['mes_bajo']}) "
                  f"cons={row['consistencia_alto']}/{row['consistencia_bajo']} z={row['z_alto']}/{row['z_bajo']} mejora {row['mejora_h1_pct']:+.0f}%/{row['mejora_h6_pct']:+.0f}% "
                  f"${row['usd_impacto']:,.0f} → {'ESTACIONAL' if row['veredicto_estacional'] else row['motivo']}"
                  f"{' · tendencia ' + str(row['pendiente_pp_anio']) + ' pp/año' if row['veredicto_tendencia'] else ''}")
    print(f"   DECISION: {verdict}")
    explain(configuration,
            "φ = variance of the monthly rate around its trend ÷ the variance sampling alone would produce: 1 = the rate only samples; 5 = something real moves it.",
            "amplitude = highest month effect − lowest, in pp, from a regression with month dummies weighted by n; consistency = share of years in which that month keeps its sign.",
            "The decisive test is predictive: does 'recent level + month effect' beat 'recent level' on the last 6 closed months? Significance (LRT) is reported but never decides:",
            "with thousands of contracts everything is significant; what matters is money (amplitude × the pipeline of the extreme months) and consistency.")
    return dict(cobertura_pct=coverage, pct_estacional=round(pct_seasonal, 1), pct_tendencia=round(pct_trending, 1),
                series_estacionales=list(seasonal["fs_id"]) if len(seasonal) else [], series_con_tendencia=list(trending["fs_id"]) if len(trending) else [],
                veredicto=verdict, estacionalidad_material=bool(pct_seasonal >= configuration.benchmark_material_share_pct))
