"""answers.py — SFF v3 · the answers: the tables and prose the business reads.

Everything here is derived from the forecast rows (`forecast_bands`) and the closed
months; nothing here changes a number of the forecast. Four views:

  pipeline_summary   per block (rest of the year, next year, total): what we predict,
                     the TOTAL band to promise (idiosyncratic ⊕ common), the binomial
                     floor, the worst case, and what really happened in the exam
  business_summary   per year: how the year ends (booked + forecast), the pipeline of
                     the next year by origin (real / projected / simulated), the forecast
                     on each, and the adjusted forecast once the signals mature
  regional_summary   per region: money, band, share at own precision, on signals, signed
                     under the floor, uplift, composition — the strategy table
  carry_signal_adjustment  adds forecast_ajustado_usd to the two summaries above
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from binomial_reference import MIN_PROPORTION_VARIANCE


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PIPELINE SUMMARY · what we predict, how well, and the best and worst it can get
# ═══════════════════════════════════════════════════════════════════════════════════

def common_band_per_row(bands: pd.DataFrame, aggregate_bands: pd.DataFrame, configuration: Config) -> tuple:
    """The COMMON error of every forecast row, in dollars: the p5/p95 of the aggregate
    error at its horizon (decision_aggregate_bands) × its pipeline$ × uplift. Horizons
    beyond the last judged one take the last. Returns (low, high) arrays; zeros when
    there are no aggregate bands."""
    if aggregate_bands is None or aggregate_bands.empty:
        zeros = np.zeros(len(bands))
        return zeros, zeros
    table = aggregate_bands.sort_values("h").set_index("h")
    h_values = bands["h"].fillna(table.index.max()).astype(int).clip(lower=int(table.index.min()))
    nearest = [table.index[table.index >= h].min() if (table.index >= h).any() else table.index.max() for h in h_values]
    q_low = table.loc[nearest, "q_low_pp"].to_numpy(dtype=float) / 100
    q_high = table.loc[nearest, "q_high_pp"].to_numpy(dtype=float) / 100
    money = (bands[configuration.pipeline_usd_col] * bands["uplift"]).to_numpy(dtype=float)
    return q_low * money, q_high * money


def pipeline_summary(bands: pd.DataFrame, holdout: pd.DataFrame, configuration: Config,
                     holdout_aggregate: pd.DataFrame = None, aggregate_bands: pd.DataFrame = None) -> pd.DataFrame:
    """One table that answers "how good is this pipeline": per block of months and in total,
    the money to predict, the expected money, THREE margins and the realized error.

    Blocks: the rest of the current year, the next year, everything (`total`).
    Columns:
      pipeline_usd        what falls due (known + simulated)
      esperado_usd        the forecast
      banda_low/high_usd  the IDIOSYNCRATIC band: backtest quantiles per id and horizon,
                          aggregated linearly inside (id, month) and in quadrature across.
                          It assumes pools miss independently — they do not entirely.
      banda_comun_low/high_usd  the COMMON band: the p5/p95 of the error of the whole
                          portfolio per horizon (decision_aggregate_bands), applied to each
                          month's money and summed LINEARLY across months (a common shock
                          persists). This is what pools missing together costs.
      banda_total_low/high_usd  the band to PROMISE: idiosyncratic ⊕ common (quadrature).
      cota_min_usd        the binomial FLOOR in quadrature: the sampling of every unit
                          with its own rate and its own n, independent units. No method
                          beats it: with these n, next month will move this much by chance
      cota_max_usd        the ABSOLUTE WORST CASE: the same sampling margin of every
                          unit summed linearly, as if every unit missed in the same
                          direction by its full margin. It cannot really happen; it is
                          the ceiling of "how wrong could a month be"
      pct_*               the three margins as % of the expected money
      pct_simulado        share built on simulated pipeline
      pct_nivel_A         share of money at levels A/A2/A3 (own precision or reinforced)
      error_realizado_pct realized hold-out error at h ≤ 4, weighted by units, as % of
                          the rate (≈ % of money): what actually happened when we predicted
      error_total_realizado_pct  the same but of the TOTAL: all pools summed per month and
                          compared with the real total; the honest check of the calibrated
                          band (pools that miss together do not cancel). error_total_peor_mes_pct
                          is the worst month.
    Improvements show as banda_* moving toward cota_min; cota_min only moves with more
    customers per unit (segmenting less, or a bigger portfolio).
    """
    period = configuration.period_col
    frame = bands.copy()
    frame["year"] = pd.PeriodIndex(frame[period].astype(str), freq="M").year
    current_year = int(frame["year"].min())
    frame["bloque"] = np.where(frame["year"] == current_year, f"resto_{current_year}", np.where(frame["year"] == current_year + 1, f"ano_{current_year + 1}", "mas_adelante"))
    units = frame[configuration.pipeline_units_col].clip(lower=1.0).astype(float)
    se_row = np.sqrt(np.maximum(frame["tasa"] * (1 - frame["tasa"]), MIN_PROPORTION_VARIANCE) / units)
    frame["cota_fila_usd"] = configuration.z * se_row * frame[configuration.pipeline_usd_col] * frame["uplift"]
    frame["nivel_A"] = frame["nivel_riesgo"].astype(str).str.startswith("A")
    frame["comun_low_usd"], frame["comun_high_usd"] = common_band_per_row(frame, aggregate_bands, configuration)
    realized, realized_total, realized_total_worst = np.nan, np.nan, np.nan
    if holdout is not None and len(holdout):
        near = holdout[holdout["h"] <= 4]
        realized = float(np.average(near["err_pp"].abs(), weights=near["n_real"])) if len(near) else np.nan
    if holdout_aggregate is not None and len(holdout_aggregate):
        near_total = holdout_aggregate[holdout_aggregate["h"] <= 4]
        realized_total = float(near_total["err_agg_pp"].abs().mean()) if len(near_total) else np.nan
        realized_total_worst = float(near_total["err_agg_pp"].abs().max()) if len(near_total) else np.nan
    rows = []
    for block, group in list(frame.groupby("bloque")) + [("total", frame)]:
        shared = group.groupby(["id_estimacion", period]).agg(low=("banda_low_usd", "sum"), high=("banda_high_usd", "sum"))
        expected = float(group["esperado_usd"].sum())
        idio_low, idio_high = -float(np.sqrt((shared["low"] ** 2).sum())), float(np.sqrt((shared["high"] ** 2).sum()))
        common_low, common_high = float(group["comun_low_usd"].sum()), float(group["comun_high_usd"].sum())
        rows.append(dict(
            bloque=block, meses=int(group[period].nunique()),
            pipeline_usd=round(float(group[configuration.pipeline_usd_col].sum()), 2), esperado_usd=round(expected, 2),
            banda_low_usd=round(idio_low, 2), banda_high_usd=round(idio_high, 2),
            banda_comun_low_usd=round(common_low, 2), banda_comun_high_usd=round(common_high, 2),
            banda_total_low_usd=round(-float(np.sqrt(idio_low ** 2 + common_low ** 2)), 2),
            banda_total_high_usd=round(float(np.sqrt(idio_high ** 2 + common_high ** 2)), 2),
            cota_min_usd=round(float(np.sqrt((group["cota_fila_usd"] ** 2).sum())), 2), cota_max_usd=round(float(group["cota_fila_usd"].sum()), 2),
            pct_simulado=round(100 * float(group.loc[group["simulada"] == 1, "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            pct_uplift_contrato=round(100 * float(group.loc[group["uplift_via"] == UPLIFT_VIA_CONTRACT, configuration.pipeline_usd_col].sum()) / max(float(group[configuration.pipeline_usd_col].sum()), 1e-9), 1) if "uplift_via" in group.columns else 0.0,
            pct_nivel_A=round(100 * float(group.loc[group["nivel_A"], "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            error_realizado_pct=round(realized, 2) if np.isfinite(realized) else np.nan,
            error_total_realizado_pct=round(realized_total, 2) if np.isfinite(realized_total) else np.nan,
            error_total_peor_mes_pct=round(realized_total_worst, 2) if np.isfinite(realized_total_worst) else np.nan))
    summary = pd.DataFrame(rows)
    for column in ("banda_low_usd", "banda_high_usd", "banda_comun_low_usd", "banda_comun_high_usd", "banda_total_low_usd",
                   "banda_total_high_usd", "cota_min_usd", "cota_max_usd"):
        summary["pct_" + column.replace("_usd", "")] = (100 * summary[column] / summary["esperado_usd"].replace(0, np.nan)).round(2)
    order = {f"resto_{current_year}": 0, f"ano_{current_year + 1}": 1, "mas_adelante": 2, "total": 3}
    return summary.sort_values("bloque", key=lambda s: s.map(order)).reset_index(drop=True)


def print_pipeline_summary(summary: pd.DataFrame) -> None:
    print("[5] PIPELINE SUMMARY · what we predict; the band to promise = idiosyncratic (pools independent) ⊕ common (pools missing together)")
    print("   block · months · pipeline $ · expected $ · TOTAL band · idiosyncratic · common · binomial floor · worst case · simulated · level A · realized |err| h<=4 (pool · total · worst month)")
    for _, row in summary.iterrows():
        def pct(low, high):
            return f"{row[low]:+.1f}%/{row[high]:+.1f}%"
        print(f"   {row['bloque']:<12} {int(row['meses']):>3}  ${row['pipeline_usd']:>12,.0f}  ${row['esperado_usd']:>12,.0f}  "
              f"{pct('pct_banda_total_low', 'pct_banda_total_high')} (${row['banda_total_high_usd']:,.0f})  "
              f"idio {pct('pct_banda_low', 'pct_banda_high')}  common {pct('pct_banda_comun_low', 'pct_banda_comun_high')}  "
              f"floor ±{row['pct_cota_min']:.1f}%  worst ±{row['pct_cota_max']:.1f}%  sim {row['pct_simulado']:.0f}%  A {row['pct_nivel_A']:.0f}%  "
              f"{row['error_realizado_pct'] if pd.notna(row['error_realizado_pct']) else float('nan'):.2f} · "
              f"{row['error_total_realizado_pct'] if pd.notna(row['error_total_realizado_pct']) else float('nan'):.2f} · "
              f"{row['error_total_peor_mes_pct'] if pd.notna(row['error_total_peor_mes_pct']) else float('nan'):.2f} pp")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE BUSINESS ANSWERS · how does this year end · what is next year's pipeline · how does it end
# ═══════════════════════════════════════════════════════════════════════════════════

def carry_signal_adjustment(forecast: dict, per_month: pd.Series, configuration: Config) -> dict:
    """Add `ajuste_senales_usd` and `forecast_ajustado_usd` to horizon_report_total and
    business_summary from the per-month adjustment of the signal maturation, rewrite the
    two tables, and return the forecast dict."""
    period = configuration.period_col
    horizon = forecast.get("horizon_report")
    if horizon is not None and len(horizon):
        horizon["ajuste_senales_usd"] = horizon[period].astype(str).map(per_month.rename(index=str)).fillna(0.0).round(2)
        horizon["forecast_ajustado_usd"] = (horizon["esperado_usd"] + horizon["ajuste_senales_usd"]).round(2)
        configuration.write(horizon, "horizon_report_total")
    answers = forecast.get("business_summary")
    if answers is not None and len(answers):
        by_year = per_month.groupby(pd.PeriodIndex(per_month.index.astype(str), freq="M").year).sum() if len(per_month) else pd.Series(dtype=float)
        answers["ajuste_senales_usd"] = answers["anio"].map(by_year).fillna(0.0).round(2)
        answers["forecast_ajustado_usd"] = (answers["forecast_usd"] + answers["ajuste_senales_usd"]).round(2)
        answers["total_esperado_ajustado_usd"] = (answers["total_esperado_usd"] + answers["ajuste_senales_usd"]).round(2)
        configuration.write(answers, "business_summary")
        for _, row in answers.iterrows():
            print(f"   {int(row['anio'])}  adjusted for the maturation of the signals: forecast {row['forecast_ajustado_usd']:,.0f} $ "
                  f"({row['ajuste_senales_usd']:+,.0f} $) → total {row['total_esperado_ajustado_usd']:,.0f} $")
    return forecast


def business_summary(bands: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config,
                     aggregate_bands: pd.DataFrame = None) -> pd.DataFrame:
    """One table, one row per year, answering the three questions the business asks:
      1. How does this year end?  renovado_real (months with truth) + forecast of the rest.
      2. What is next year's pipeline?  pipeline_real + pipeline_proyectada + pipeline_simulada.
      3. How does next year end?  the forecast on that pipeline, with its band.
    Columns: anio, meses_reales, meses_forecast, pipeline_real_usd, pipeline_proyectada_usd,
    pipeline_simulada_usd, pipeline_total_usd, renovado_real_usd, forecast_usd,
    forecast_sobre_real_usd, forecast_sobre_proyectada_usd, forecast_sobre_simulada_usd,
    total_esperado_usd (= renovado_real + forecast), banda_low_usd, banda_high_usd (the
    TOTAL band: idiosyncratic quadrature ⊕ the common error of the portfolio).
    """
    period, role = configuration.period_col, configuration.dataset_role_col
    truth = fine_table[fine_table[role].isin(TRUTH_ROLES)]
    pending = fine_table[fine_table[role] == ROLE_PENDING]
    pending_year = pd.PeriodIndex(pending[period].astype(str), freq="M").year
    observed_pending = pending.groupby(pending_year)[configuration.renewed_usd_col].sum() if len(pending) else pd.Series(dtype=float)
    truth_year = pd.PeriodIndex(truth[period].astype(str), freq="M").year
    realized = truth.groupby(truth_year).agg(renovado_real_usd=(configuration.renewed_usd_col, "sum"),
                                              pipeline_vencida_usd=(configuration.pipeline_usd_col, "sum"), meses_reales=(period, "nunique"))
    future = bands.assign(anio=pd.PeriodIndex(bands[period].astype(str), freq="M").year)
    future["comun_low_usd"], future["comun_high_usd"] = common_band_per_row(future, aggregate_bands, configuration)
    by_origin = future.pivot_table(index="anio", columns="origen_pipeline", values=configuration.pipeline_usd_col, aggfunc="sum").fillna(0.0)
    expected_by_origin = future.pivot_table(index="anio", columns="origen_pipeline", values="esperado_usd", aggfunc="sum").fillna(0.0)
    years = sorted(set(by_origin.index) | {int(y) for y in realized.index if y >= min(by_origin.index)}) if len(by_origin) else sorted(realized.index)
    rows = []
    for year in years:
        year_rows = future[future["anio"] == year]
        shared = year_rows.groupby(["id_estimacion", period]).agg(low=("banda_low_usd", "sum"), high=("banda_high_usd", "sum"))
        forecast = float(year_rows["esperado_usd"].sum())
        real_money = float(realized["renovado_real_usd"].get(year, 0.0)) if year in realized.index else 0.0
        rows.append(dict(
            anio=int(year),
            meses_reales=int(realized["meses_reales"].get(year, 0)) if year in realized.index else 0,
            meses_forecast=int(year_rows[period].nunique()),
            pipeline_real_usd=round(float(by_origin.get(PIPELINE_REAL, pd.Series(dtype=float)).get(year, 0.0)), 2),
            pipeline_proyectada_usd=round(float(by_origin.get(PIPELINE_PROJECTED, pd.Series(dtype=float)).get(year, 0.0)), 2),
            pipeline_simulada_usd=round(float(by_origin.get(PIPELINE_SIMULATED, pd.Series(dtype=float)).get(year, 0.0)), 2),
            pipeline_total_usd=round(float(year_rows[configuration.pipeline_usd_col].sum()), 2),
            renovado_real_usd=round(real_money, 2),
            renovado_parcial_mes_pendiente_usd=round(float(observed_pending.get(year, 0.0)), 2),
            forecast_usd=round(forecast, 2),
            forecast_mes_pendiente_usd=round(float(year_rows.loc[year_rows.get("mes_pendiente_cierre", 0) == 1, "esperado_usd"].sum()), 2),
            forecast_sobre_real_usd=round(float(expected_by_origin.get(PIPELINE_REAL, pd.Series(dtype=float)).get(year, 0.0)), 2),
            forecast_sobre_proyectada_usd=round(float(expected_by_origin.get(PIPELINE_PROJECTED, pd.Series(dtype=float)).get(year, 0.0)), 2),
            forecast_sobre_simulada_usd=round(float(expected_by_origin.get(PIPELINE_SIMULATED, pd.Series(dtype=float)).get(year, 0.0)), 2),
            total_esperado_usd=round(real_money + forecast, 2),
            banda_low_usd=round(-float(np.sqrt((shared["low"] ** 2).sum() + year_rows["comun_low_usd"].sum() ** 2)), 2) if len(shared) else 0.0,
            banda_high_usd=round(float(np.sqrt((shared["high"] ** 2).sum() + year_rows["comun_high_usd"].sum() ** 2)), 2) if len(shared) else 0.0))
    return pd.DataFrame(rows)


def print_business_summary(summary: pd.DataFrame) -> None:
    print("[5] BUSINESS ANSWERS · per year: what is already real, what we forecast, and on which pipeline")
    for _, row in summary.iterrows():
        print(f"   {int(row['anio'])}  renewed (closed months) ${row['renovado_real_usd']:>13,.0f} ({int(row['meses_reales'])} months)  "
              f"+ forecast ${row['forecast_usd']:>13,.0f} ({int(row['meses_forecast'])} months, of which the pending month "
              f"${row['forecast_mes_pendiente_usd']:,.0f} vs ${row['renovado_parcial_mes_pendiente_usd']:,.0f} already booked)  "
              f"= ${row['total_esperado_usd']:>13,.0f}  band {row['banda_low_usd']:+,.0f} / {row['banda_high_usd']:+,.0f}")
        print(f"         pipeline to renew: real ${row['pipeline_real_usd']:,.0f} · projected (our own forecast re-entering) ${row['pipeline_proyectada_usd']:,.0f} · "
              f"simulated (acquisition at the historical pace) ${row['pipeline_simulada_usd']:,.0f} → forecast on each: "
              f"${row['forecast_sobre_real_usd']:,.0f} / ${row['forecast_sobre_proyectada_usd']:,.0f} / ${row['forecast_sobre_simulada_usd']:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE REGIONAL VIEW · one row per region (the first mandatory dim): where to act
# ═══════════════════════════════════════════════════════════════════════════════════

def regional_summary(bands: pd.DataFrame, series_card: pd.DataFrame, composition: pd.DataFrame, configuration: Config,
                     region_dim: str = None) -> pd.DataFrame:
    """One row per value of the region dimension (the first mandatory dim unless given):
    what falls due, what we expect, the band, how much of the money predicts with its own
    precision, how much rides on signals, the average uplift, the share of composition in
    the movement of its cells, and the realized rate of the last closed months.

    Columns: region, series, pipeline_usd, esperado_usd, tasa_media, uplift_medio,
    banda_low_usd, banda_high_usd, pct_banda, pct_nivel_A, pct_nivel_S, pct_senal (money
    of series with a sign), pct_simulado, composicion_pct (share of the month-to-month
    movement of the region's cells that is mix), tasa_reciente (last 3 closed months).
    The strategy by region reads this table (ESTRATEGIA_POR_REGION.md).
    """
    period = configuration.period_col
    dim = region_dim or configuration.business_mandatory_dims[0]
    frame = bands.copy()
    frame["region"] = frame["fs_id"].str.split("|").str[0] if dim == configuration.business_mandatory_dims[0] else frame[dim].astype(str)
    signed = series_card.set_index("fs_id")["signo"].reindex(frame["fs_id"]).fillna(SIGN_NEUTRAL).to_numpy()
    frame["con_senal"] = signed != SIGN_NEUTRAL
    frame["nivel_A"] = frame["nivel_riesgo"].astype(str).str.startswith("A")
    frame["nivel_S"] = frame["nivel_riesgo"].astype(str).str.startswith("S")
    rows = []
    for region, group in frame.groupby("region"):
        shared = group.groupby(["id_estimacion", period]).agg(low=("banda_low_usd", "sum"), high=("banda_high_usd", "sum"))
        expected = float(group["esperado_usd"].sum())
        pipeline = float(group[configuration.pipeline_usd_col].sum())
        cells = composition[composition["celda_id"].astype(str).str.split("|").str[0] == region] if len(composition) else composition
        comp_share = (cells["delta_composicion_pp"].abs().sum() / max((cells["delta_composicion_pp"].abs() + cells["delta_comportamiento_pp"].abs()).sum(), 1e-9)) if len(cells) else np.nan
        rows.append(dict(
            region=region, series=int(group["fs_id"].nunique()),
            pipeline_usd=round(pipeline, 2), esperado_usd=round(expected, 2),
            tasa_media=round(float(np.average(group["tasa"], weights=np.maximum(group[configuration.pipeline_usd_col], 1e-9))), 4),
            uplift_medio=round(float(np.average(group["uplift"], weights=np.maximum(group[configuration.pipeline_usd_col], 1e-9))), 4),
            banda_low_usd=round(-float(np.sqrt((shared["low"] ** 2).sum())), 2), banda_high_usd=round(float(np.sqrt((shared["high"] ** 2).sum())), 2),
            pct_nivel_A=round(100 * float(group.loc[group["nivel_A"], "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            pct_nivel_S=round(100 * float(group.loc[group["nivel_S"], "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            pct_senal=round(100 * float(group.loc[group["con_senal"], "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            pct_simulado=round(100 * float(group.loc[group["simulada"] == 1, "esperado_usd"].sum()) / max(expected, 1e-9), 1),
            pct_uplift_contrato=round(100 * float(group.loc[group["uplift_via"] == UPLIFT_VIA_CONTRACT, configuration.pipeline_usd_col].sum()) / max(float(group[configuration.pipeline_usd_col].sum()), 1e-9), 1) if "uplift_via" in group.columns else 0.0,
            composicion_pct=round(100 * comp_share, 1) if pd.notna(comp_share) else np.nan))
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["pct_banda"] = (100 * (summary["banda_high_usd"] - summary["banda_low_usd"]) / 2 / summary["esperado_usd"].replace(0, np.nan)).round(2)
        summary["pct_del_total"] = (100 * summary["esperado_usd"] / summary["esperado_usd"].sum()).round(1)
    return summary.sort_values("esperado_usd", ascending=False).reset_index(drop=True) if len(summary) else summary


def print_regional_summary(summary: pd.DataFrame, dim: str) -> None:
    print(f"[5] BY REGION ({dim}) · expected $ · % of total · band ±% · % money at own precision (A) · % on signals · % signed under floor (S) · uplift · composition")
    for _, row in summary.iterrows():
        print(f"   {str(row['region'])[:28]:<28} ${row['esperado_usd']:>13,.0f} {row['pct_del_total']:5.1f}%  ±{row['pct_banda']:.1f}%  "
              f"A {row['pct_nivel_A']:5.1f}%  signals {row['pct_senal']:5.1f}%  S {row['pct_nivel_S']:4.1f}%  uplift {row['uplift_medio']:.3f}  "
              f"composition {row['composicion_pct'] if pd.notna(row['composicion_pct']) else float('nan'):.0f}%")
