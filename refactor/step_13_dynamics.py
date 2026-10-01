"""
step_13_dynamics.py — The dynamics of the renewal rate: is there anything beyond noise?

The rate of a month is k renewals out of n contracts: even if nothing changes, it moves by
chance (the binomial error). This step asks, for the portfolio and for every estimation id
with support, what moves the rate BEYOND that noise. It is DESCRIPTIVE: it labels, it does
not restrict any technique (the backtest of step 14 lets every technique compete as far
as its history allows, and decides).

For every estimation id with support and at least dynamics_min_months (24) of history:
  · φ (phi) = observed variance of the monthly rate / the variance chance alone would give
    with a constant rate (mean of p(1 − p)/n). φ ≈ 1: the rate moves what the coin
    predicts, nothing to model, the average is the best technique. φ > 1: something moves
    it (a trend, a season, a change of level, a changing mix): there is room for a
    time-series technique.
  · TREND: the slope of the rate over time (weighted by the units of each month), in pp per
    year, and its p-value. tendencia = +1 / −1 when significant, 0 otherwise.
  · SEASONALITY: on the rate WITHOUT its trend, is the month of the year significant? An
    F test: variance between the 12 calendar months vs within them. estacional = 1 when
    significant. amplitud_pp: the highest month minus the lowest. meses_alto / meses_bajo:
    the calendar months above / below the others beyond twice their error.
  · CONSISTENCY: the correlation of the month profile of the first half of the years with
    the second half: a real season repeats.
The same for the PORTFOLIO as a whole (every estimable series, summed): its month profile.
The significance level is dynamics_significance (5 %): no threshold in pp or in % of
improvement; every test is relative to the noise of the series itself.

Actions (logged as they are done):
  1. the pools measured: support and history
  2. φ, trend and seasonality of every pool
  3. the month profile of the whole portfolio
  4. check the attributes                                            checks 1-2
  5. write the pools' dynamics and the portfolio's profile           checks 3-4
  6. count the checks; stop if any failed
  7. show the general verdict: the portfolio's profile and how many pools (and how much money)
     show a trend, a season, φ > 1

Checks (logged as they are made, numbered, at the level of their status):
   1. every pool measured has its attributes (φ ≥ 0, p-values between 0 and 1)
   2. the portfolio profile covers the 12 calendar months
   3-4. tables sff_composition_dynamics and sff_estacionalidad_cartera written and read back

Output: (the dynamics of every pool, the month profile of the portfolio) · tables
sff_composition_dynamics, sff_estacionalidad_cartera.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

from config import Config
from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, TABLE_POOL_DYNAMICS,
                         TABLE_PORTFOLIO_SEASONALITY, TRUTH_ROLES)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "13"
STEP_NAME = "DYNAMICS OF THE RATE"
STEP_PURPOSE = ("measure what moves the renewal rate beyond the binomial noise, for the portfolio and for every pool "
                "with support: φ (observed variation / noise), trend, seasonality and its months; descriptive: it "
                "does not restrict any technique, the backtest decides")
STEP_ACTIONS = ["the pools measured: support and history",
                "φ, trend and seasonality of every pool",
                "the month profile of the whole portfolio",
                "check the attributes (checks 1-2)",
                "write the pools' dynamics and the portfolio's profile (checks 3-4)",
                "count the checks; stop if any failed",
                "show the general verdict: the portfolio's profile and how many pools show a trend, a season, φ > 1"]
STEP_OUTPUT = "one row per pool (φ, trend, season, months high/low) · the portfolio's month profile · tables sff_composition_dynamics, sff_estacionalidad_cartera"

# ─── named constants ─────────────────────────────────────────────────────────────
PERCENTAGE_POINTS = 100
MONTHS_PER_YEAR = 12
MONTH_SIGNIFICANCE_SIGMAS = 2.0      # a month is 'high' or 'low' when it deviates by more than 2 errors
PHI_WITH_ENGINE = 1.5                # for the summary: φ above this = clearly more than noise


def measure_dynamics(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> tuple:
    """φ, trend and seasonality of every pool with support, and the portfolio's month profile."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    truth_months = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (pool_series["vencen"] > 0)]

    # [1] the pools measured
    months_per_pool = truth_months.groupby(COMPOSITION_ID_COLUMN).size()
    measured_ids = [estimation_id for estimation_id in pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN]
                    if months_per_pool.get(estimation_id, 0) >= configuration.dynamics_min_months]
    configuration.log_action(STEP_LABEL, 1, f"{len(measured_ids):,} pools measured (support ≥ {configuration.support_floor:.0f} "
                                            f"and ≥ {configuration.dynamics_min_months} months) of {len(pool_reference):,}")

    # [2] the attributes of every pool
    dynamics_rows = []
    for estimation_id in measured_ids:
        monthly = truth_months[truth_months[COMPOSITION_ID_COLUMN] == estimation_id].sort_values(period_column)
        dynamics_rows.append({COMPOSITION_ID_COLUMN: estimation_id,
                              **dynamics_of_one_series(monthly, period_column, configuration)})
    pool_dynamics = pd.DataFrame(dynamics_rows)
    if len(pool_dynamics):
        pool_dynamics = pool_dynamics.merge(pool_reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
    configuration.log_action(STEP_LABEL, 2, f"φ median {pool_dynamics['phi'].median():.2f} · "
                                            f"{int(pool_dynamics['estacional'].sum())} seasonal · "
                                            f"{int((pool_dynamics['tendencia'] != 0).sum())} with a trend"
                             if len(pool_dynamics) else "no pool to measure")

    # [3] the portfolio: every estimable series summed, its month profile
    portfolio_monthly = (truth_months.groupby(period_column)[["renovadas", "vencen"]].sum().reset_index())
    portfolio = dynamics_of_one_series(portfolio_monthly, period_column, configuration)
    portfolio_profile = pd.DataFrame(portfolio.pop("_profile"))
    configuration.log_action(STEP_LABEL, 3, f"portfolio: φ {portfolio['phi']:.2f} · month effect p = {portfolio['p_valor_mes']:.3f} · "
                                            f"amplitude {portfolio['amplitud_pp']:.1f} pp · trend {portfolio['tendencia_pp_ano']:+.1f} pp/year "
                                            f"(p = {portfolio['p_valor_tendencia']:.3f})")
    if len(pool_dynamics):
        pool_dynamics = pool_dynamics.drop(columns="_profile")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the attributes")
    invalid = pool_dynamics[(pool_dynamics["phi"] < 0) | ~pool_dynamics["p_valor_mes"].between(0, 1)
                            | ~pool_dynamics["p_valor_tendencia"].between(0, 1)] if len(pool_dynamics) else pool_dynamics
    configuration.log_check(STEP_LABEL, check_log, "every pool measured has its attributes (φ ≥ 0, p-values in [0, 1])",
                            len(pool_dynamics) == len(measured_ids) and invalid.empty,
                            failure_detail=f"{len(invalid)} pools with invalid attributes", context=f"{len(pool_dynamics)} pools")
    configuration.log_check(STEP_LABEL, check_log, "the portfolio profile covers the 12 calendar months",
                            len(portfolio_profile) == MONTHS_PER_YEAR,
                            failure_detail=f"only {len(portfolio_profile)} calendar months")

    # [5] the tables
    configuration.log_action(STEP_LABEL, 5, "writing the pools' dynamics and the portfolio's profile")
    configuration.write_table(STEP_LABEL, check_log, pool_dynamics, TABLE_POOL_DYNAMICS)
    configuration.write_table(STEP_LABEL, check_log, portfolio_profile, TABLE_PORTFOLIO_SEASONALITY)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the general verdict
    log_dynamics_report(pool_dynamics, portfolio, portfolio_profile, configuration)
    return pool_dynamics, portfolio_profile, portfolio


def dynamics_of_one_series(monthly: pd.DataFrame, period_column: str, configuration: Config) -> dict:
    """φ, trend (pp/year and p-value), seasonality on the detrended rate (F test), amplitude,
    months high and low, consistency between halves, and the 12-month profile."""
    renewed = monthly["renovadas"].to_numpy(dtype=float)
    due = monthly["vencen"].to_numpy(dtype=float)
    rates = renewed / due
    pooled_rate = renewed.sum() / due.sum()
    calendar_months = np.array([month.month for month in monthly[period_column]])
    years = np.array([month.year for month in monthly[period_column]])
    time_in_years = np.array([(month.ordinal - monthly[period_column].iloc[0].ordinal) / MONTHS_PER_YEAR
                              for month in monthly[period_column]])

    # φ: observed variance of the monthly rate / the binomial variance with a constant rate
    binomial_variance = np.mean(pooled_rate * (1 - pooled_rate) / due)
    phi = float(np.var(rates, ddof=1) / binomial_variance) if binomial_variance > 0 else np.nan

    # the trend: weighted least squares of the rate on time (weights = units due)
    weights = due / due.mean()
    design = np.column_stack([np.ones_like(time_in_years), time_in_years])
    weighted_design = design * np.sqrt(weights)[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, rates * np.sqrt(weights), rcond=None)
    residuals = rates - design @ coefficients
    degrees_of_freedom = max(len(rates) - 2, 1)
    residual_variance = float(np.sum(weights * residuals ** 2) / degrees_of_freedom)
    slope_se = float(np.sqrt(residual_variance * np.linalg.inv(weighted_design.T @ weighted_design)[1, 1]))
    slope_t = coefficients[1] / slope_se if slope_se > 0 else 0.0
    trend_p_value = float(2 * stats.t.sf(abs(slope_t), degrees_of_freedom))
    trend_sign = int(np.sign(coefficients[1])) if trend_p_value < configuration.dynamics_significance else 0

    # the season: on the detrended rate, the 12 calendar months (weighted), F test
    detrended = pd.DataFrame({"mes": calendar_months, "residuo": residuals, "peso": weights})
    month_groups = detrended.groupby("mes")
    month_weight = month_groups["peso"].sum()
    month_effect = (detrended.assign(weighted=detrended["residuo"] * detrended["peso"]).groupby("mes")["weighted"].sum()
                    / month_weight)
    overall = float(np.sum(weights * residuals) / np.sum(weights))
    between = float(np.sum(month_weight * (month_effect - overall) ** 2))
    within = float(np.sum(weights * (residuals - detrended["mes"].map(month_effect).to_numpy()) ** 2))
    months_present = len(month_effect)
    between_df, within_df = max(months_present - 1, 1), max(len(rates) - months_present - 1, 1)
    f_statistic = (between / between_df) / (within / within_df) if within > 0 else 0.0
    month_p_value = float(stats.f.sf(f_statistic, between_df, within_df))
    seasonal = int(month_p_value < configuration.dynamics_significance)

    # the months high and low: beyond twice their own error
    month_count = month_groups.size()
    month_error = np.sqrt(within / within_df / month_count)
    deviation_pp = PERCENTAGE_POINTS * (month_effect - overall)
    high_months = [int(month) for month in month_effect.index
                   if month_effect[month] - overall > MONTH_SIGNIFICANCE_SIGMAS * month_error[month]]
    low_months = [int(month) for month in month_effect.index
                  if overall - month_effect[month] > MONTH_SIGNIFICANCE_SIGMAS * month_error[month]]

    # consistency: the month profile of the first half of the years vs the second half
    middle_year = np.median(np.unique(years))
    first_half = detrended[years <= middle_year].groupby("mes")["residuo"].mean()
    second_half = detrended[years > middle_year].groupby("mes")["residuo"].mean()
    shared = first_half.index.intersection(second_half.index)
    # a profile that does not move in one of the halves has no correlation (a series with the same rate every month)
    both_move = len(shared) >= 3 and first_half[shared].std() > 0 and second_half[shared].std() > 0
    consistency = float(np.corrcoef(first_half[shared], second_half[shared])[0, 1]) if both_move else np.nan

    profile = pd.DataFrame({"mes": month_effect.index.astype(int), "efecto_pp": deviation_pp.to_numpy(),
                            "error_pp": PERCENTAGE_POINTS * month_error.to_numpy(), "meses_observados": month_count.to_numpy(),
                            "tasa_media": (pooled_rate + month_effect.to_numpy() - overall)})
    return {"meses": int(len(rates)), "tasa_pool": float(pooled_rate), "phi": phi,
            "tendencia": trend_sign, "tendencia_pp_ano": float(PERCENTAGE_POINTS * coefficients[1]),
            "p_valor_tendencia": trend_p_value, "estacional": seasonal, "p_valor_mes": month_p_value,
            "amplitud_pp": float(deviation_pp.max() - deviation_pp.min()),
            "meses_alto": ",".join(str(month) for month in high_months), "meses_bajo": ",".join(str(month) for month in low_months),
            "consistencia": consistency, "_profile": profile}


def log_dynamics_report(pool_dynamics: pd.DataFrame, portfolio: dict, portfolio_profile: pd.DataFrame,
                        configuration: Config) -> None:
    """Action 7: the general verdict, as tables."""
    verdict = ("the portfolio rate HAS a month effect beyond noise" if portfolio["estacional"]
               else "the portfolio rate shows NO month effect beyond noise")
    configuration.log_action(STEP_LABEL, 7, f"general verdict: {verdict} (p = {portfolio['p_valor_mes']:.3f}, amplitude "
                                            f"{portfolio['amplitud_pp']:.1f} pp, consistency between halves "
                                            f"{portfolio['consistencia']:.2f}). Month profile of the portfolio (efecto_pp: "
                                            f"the month minus the year, without trend):")
    configuration.show_table(portfolio_profile)
    if len(pool_dynamics):
        total_usd = pool_dynamics["usd_por_predecir"].sum()
        def share(mask):
            return {"pools": int(mask.sum()), "usd_por_predecir": pool_dynamics.loc[mask, "usd_por_predecir"].sum(),
                    "pct_usd": pool_dynamics.loc[mask, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0}
        summary = pd.DataFrame([
            {"atributo": f"φ > {PHI_WITH_ENGINE} (more than noise)", **share(pool_dynamics["phi"] > PHI_WITH_ENGINE)},
            {"atributo": "trend (+1 or −1)", **share(pool_dynamics["tendencia"] != 0)},
            {"atributo": "seasonal (month effect)", **share(pool_dynamics["estacional"] == 1)},
            {"atributo": "all pools measured", **share(pool_dynamics["phi"].notna())}])
        configuration.logger.doc(f"[{STEP_LABEL}] the pools with support, by attribute (the money their series have to predict):")
        configuration.show_table(summary)
