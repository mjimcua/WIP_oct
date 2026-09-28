"""analysis_top_movers.py — SFF v3 · ANALYSIS · the top movers: where the money moves, where it is at risk.

One table, `top_movers`, with one row per (series or cell, kind of movement), ranked by
money, so the business reads opportunities and risks without opening the forecast:

  deterioro / mejora   the forecast rate of the next months against the realized rate
                       of the last closed months: who is getting worse, who better, and
                       how many dollars that is on the coming pipeline
  senal_negativa       series with a negative signal: the gap between their rate and the
                       neutral rate of their cell × their pipeline = the money a recovery
                       action can dispute (an upper bound: not every warned customer
                       can be recovered)
  banda_ancha          the series whose band is widest in dollars: the risk of a surprise,
                       whatever the point forecast says
  sesgo_examen         pools that the exam showed we systematically over- or under-
                       forecast at one month ahead: model risk, in dollars
  precio               uplift cells far below (or above) the portfolio's revaluation:
                       price opportunity (or price pressure)

Every row carries `tipo`, the metric in pp, the money at stake, the region, and the
rank inside its kind. Read with ESTRATEGIA_POR_REGION.md: each kind is a lever.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
RECENT_MONTHS = 3            # the realized rate of the last closed months
AHEAD_MONTHS = 3             # the forecast rate of the next months
IMPACT_MONTHS = 12           # the pipeline the movement is priced on
KINDS = ("deterioro", "mejora", "senal_negativa", "banda_ancha", "sesgo_examen", "precio")


def recent_rate_by_series(units: pd.DataFrame, configuration: Config, months: int = RECENT_MONTHS) -> pd.Series:
    """Σ renewed / Σ pipeline over the last `months` closed months, per series."""
    period, role = configuration.period_col, configuration.dataset_role_col
    closed = units[(units[role].isin(TRUTH_ROLES)) & (units.get("sintetica", 0) == 0) & (units[configuration.pipeline_units_col] > 0)]
    last_months = sorted(closed[period].unique())[-months:]
    block = closed[closed[period].isin(last_months)]
    grouped = block.groupby("fs_id").agg(ren=(configuration.renewed_units_col, "sum"), pipe=(configuration.pipeline_units_col, "sum"))
    return (grouped["ren"] / grouped["pipe"].replace(0, np.nan)).rename("tasa_reciente")


def forecast_rate_by_series(detail: pd.DataFrame, configuration: Config, months: int = AHEAD_MONTHS) -> tuple:
    """Money-weighted forecast rate over the next `months` of the horizon, and the pipeline $
    of the next IMPACT_MONTHS, per series."""
    period = configuration.period_col
    frame = detail.assign(_period=pd.PeriodIndex(detail[period].astype(str), freq="M"))
    horizon = sorted(frame["_period"].unique())
    ahead = frame[frame["_period"].isin(horizon[:months])]
    weight = ahead[configuration.pipeline_usd_col]
    rate = (ahead["tasa"] * weight).groupby(ahead["fs_id"]).sum() / weight.groupby(ahead["fs_id"]).sum().replace(0, np.nan)
    impact = frame[frame["_period"].isin(horizon[:IMPACT_MONTHS])].groupby("fs_id")[configuration.pipeline_usd_col].sum()
    return rate.rename("tasa_prevista"), impact.rename("pipeline_12m_usd")


def rate_movers(units: pd.DataFrame, detail: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """deterioro / mejora: forecast rate of the next months vs realized rate of the last ones."""
    recent = recent_rate_by_series(units, configuration)
    ahead, impact = forecast_rate_by_series(detail, configuration)
    frame = pd.concat([recent, ahead, impact], axis=1).dropna(subset=["tasa_reciente", "tasa_prevista"])
    frame["delta_pp"] = 100 * (frame["tasa_prevista"] - frame["tasa_reciente"])
    frame["usd_impacto"] = frame["delta_pp"] / 100 * frame["pipeline_12m_usd"].fillna(0)
    frame = frame.reset_index().rename(columns={"index": "fs_id"})
    frame["tipo"] = np.where(frame["delta_pp"] < 0, "deterioro", "mejora")
    frame["metrica"] = "tasa prevista (3 m) − tasa realizada (últimos 3 m cerrados), pp"
    return frame[["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "tasa_reciente", "tasa_prevista", "pipeline_12m_usd"]]


def negative_signal_movers(series_card: pd.DataFrame, detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """senal_negativa: the gap to the neutral rate of the cell, priced on the series' coming pipeline."""
    neutral = series_card[(series_card["signo"] == SIGN_NEUTRAL) & series_card["tasa_estimada"].notna() & (series_card["n_propio"] > 0)]
    cell_neutral_rate = (neutral["tasa_estimada"] * neutral["n_propio"]).groupby(neutral["celda_id"]).sum() / neutral.groupby("celda_id")["n_propio"].sum()
    _, impact = forecast_rate_by_series(detail, configuration)
    signed = series_card[(series_card["signo"] == SIGN_NEGATIVE) & series_card["tasa_estimada"].notna()].copy()
    signed["tasa_neutra_celda"] = signed["celda_id"].map(cell_neutral_rate)
    signed = signed.dropna(subset=["tasa_neutra_celda"])
    signed["delta_pp"] = 100 * (signed["tasa_neutra_celda"] - signed["tasa_estimada"])
    signed["pipeline_12m_usd"] = signed["fs_id"].map(impact).fillna(0)
    signed["usd_impacto"] = signed["delta_pp"] / 100 * signed["pipeline_12m_usd"]
    signed["tipo"] = "senal_negativa"
    signed["metrica"] = "tasa neutra de la celda − tasa de la serie con señal, pp (dinero recuperable como cota superior)"
    return signed[["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "tasa_estimada", "tasa_neutra_celda", "pipeline_12m_usd"]].rename(columns={"tasa_estimada": "tasa_reciente", "tasa_neutra_celda": "tasa_prevista"})


def wide_band_movers(detail: pd.DataFrame, bands: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """banda_ancha: half the band in dollars over the next IMPACT_MONTHS, per series."""
    period = configuration.period_col
    joined = detail.merge(bands[["fu_comb_key", "banda_low_usd", "banda_high_usd"]], on="fu_comb_key", how="left")
    joined["_period"] = pd.PeriodIndex(joined[period].astype(str), freq="M")
    horizon = sorted(joined["_period"].unique())[:IMPACT_MONTHS]
    joined = joined[joined["_period"].isin(horizon)]
    per_series = joined.groupby("fs_id").agg(low=("banda_low_usd", "sum"), high=("banda_high_usd", "sum"),
                                             esperado=("esperado_usd", "sum"), pipe=(configuration.pipeline_usd_col, "sum")).reset_index()
    per_series["usd_impacto"] = (per_series["high"] - per_series["low"]) / 2
    per_series["delta_pp"] = 100 * per_series["usd_impacto"] / per_series["esperado"].replace(0, np.nan)
    per_series["tipo"] = "banda_ancha"
    per_series["metrica"] = "media banda en $ sobre 12 meses (y en % del esperado)"
    return per_series.rename(columns={"pipe": "pipeline_12m_usd"})[["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "pipeline_12m_usd"]]


def exam_bias_movers(backtest_holdout: pd.DataFrame, decision_support: pd.DataFrame, detail: pd.DataFrame,
                     configuration: Config) -> pd.DataFrame:
    """sesgo_examen: pools with a systematic miss at h=1 in the exam, priced on the series that use them."""
    if backtest_holdout is None or backtest_holdout.empty:
        return pd.DataFrame(columns=["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "pipeline_12m_usd"])
    at_h1 = backtest_holdout[backtest_holdout["h"] == 1]
    bias = at_h1.groupby("id_estimacion").apply(lambda g: float(np.average(g["err_pp"], weights=g["n_real"])), include_groups=False).rename("sesgo_pp")
    _, impact = forecast_rate_by_series(detail, configuration)
    frame = decision_support[["fs_id", "id_estimacion"]].merge(bias, left_on="id_estimacion", right_index=True)
    frame["pipeline_12m_usd"] = frame["fs_id"].map(impact).fillna(0)
    frame["delta_pp"] = frame["sesgo_pp"]
    frame["usd_impacto"] = frame["delta_pp"] / 100 * frame["pipeline_12m_usd"]
    frame["tipo"] = "sesgo_examen"
    frame["metrica"] = "sesgo del examen a h=1 del pool (+ sobreestimamos, − infraestimamos), pp"
    return frame[["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "pipeline_12m_usd", "id_estimacion"]]


def price_movers(detail: pd.DataFrame, decision_uplift: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """precio: uplift cells below / above the portfolio's money-weighted uplift, priced on their pipeline."""
    period = configuration.period_col
    frame = detail.assign(_period=pd.PeriodIndex(detail[period].astype(str), freq="M"))
    horizon = sorted(frame["_period"].unique())[:IMPACT_MONTHS]
    frame = frame[frame["_period"].isin(horizon)]
    portfolio_uplift = float(np.average(frame["uplift"], weights=np.maximum(frame[configuration.pipeline_usd_col], 1e-9)))
    per_cell = frame.groupby("uplift_cell_id").agg(uplift=("uplift", "first"), pipe=(configuration.pipeline_usd_col, "sum"),
                                                   tasa=("tasa", "mean")).reset_index()
    per_cell["delta_pp"] = 100 * (per_cell["uplift"] - portfolio_uplift)
    per_cell["usd_impacto"] = (per_cell["uplift"] - portfolio_uplift) * per_cell["pipe"] * per_cell["tasa"]
    per_cell["tipo"] = "precio"
    per_cell["metrica"] = f"uplift de la celda − uplift medio de la cartera ({portfolio_uplift:.3f}), en % del precio"
    return per_cell.rename(columns={"uplift_cell_id": "fs_id", "pipe": "pipeline_12m_usd"})[["fs_id", "tipo", "metrica", "delta_pp", "usd_impacto", "pipeline_12m_usd", "uplift"]]


def run_top_movers(units: pd.DataFrame, series_card: pd.DataFrame, forecast: dict, decisions: dict,
                   backtest_holdout: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Phase 5.10: the top movers table, persisted and printed by kind."""
    detail, bands = forecast["forecast_detail"], forecast["forecast_bands"]
    if detail is None or detail.empty:
        return pd.DataFrame()
    parts = [rate_movers(units, detail, series_card, configuration),
             negative_signal_movers(series_card, detail, configuration),
             wide_band_movers(detail, bands, configuration),
             exam_bias_movers(backtest_holdout, decisions["decision_support"], detail, configuration),
             price_movers(detail, decisions["decision_uplift"], configuration)]
    movers = pd.concat(parts, ignore_index=True)
    region_of = series_card.set_index("fs_id")["celda_id"].astype(str).str.split("|").str[0]
    movers["region"] = movers["fs_id"].map(region_of).fillna(movers["fs_id"].astype(str).str.split("|").str[0])
    movers["abs_usd"] = movers["usd_impacto"].abs()
    movers["rank"] = movers.groupby("tipo")["abs_usd"].rank(ascending=False, method="first").astype(int)
    movers = movers.sort_values(["tipo", "rank"]).drop(columns="abs_usd").reset_index(drop=True)
    configuration.write(movers, "top_movers")
    print("[5] TOP MOVERS · the biggest movements and risks in money, by kind (full table: top_movers)")
    for kind in KINDS:
        block = movers[movers["tipo"] == kind].head(configuration.console_top_rows if kind in ("deterioro", "mejora") else 5)
        if block.empty:
            continue
        print(f"   {kind} · {block['metrica'].iloc[0]}")
        for _, row in block.iterrows():
            print(f"      {str(row['fs_id'])[:60]:<60} {row['delta_pp']:+7.1f} pp  ${row['usd_impacto']:>12,.0f}  (pipeline 12m ${row['pipeline_12m_usd']:,.0f})")
    explain(configuration,
            "deterioro/mejora: the rate we forecast for the next 3 months against the rate realized in the last 3 closed months, priced on 12 months of pipeline.",
            "senal_negativa: how much the series with a churn signal renews below the neutral customers of its cell × its pipeline = the money a recovery action disputes (upper bound).",
            "banda_ancha: half the band in dollars over 12 months: where a surprise costs most, whatever the point forecast. sesgo_examen: pools the exam showed we miss systematically.",
            "precio: cells revalued well below (or above) the portfolio: room for price, or price pressure on renewals.")
    return movers
