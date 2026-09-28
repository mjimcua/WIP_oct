"""analysis_signal_maturation.py — SFF v3 · RUN · the maturation of the timevarying signals.

The forecast is a photo: every future contract carries its signals AS OF TODAY. But the
signals mature. `softcancel` appears at two moments — right after a renewal (eleven months
before the next expiry, for annual contracts) and close to the expiry (the customers who
do not want to autorenew); `dormant` appears at any time and only grows. So for a month
h months ahead, a share of today's neutral units will have moved to a signed series by
the time it expires, and signed series renew far below the neutral ones.

Aggregating by each customer's own signal date disperses the data. This module works
instead with the COMPOSITION of every cell (mandatory dims + extras de renovación,
WITHOUT the timevarying dims): what share of its units is neutral, softcancel, dormant…

  · in the closed months, the composition is FINAL (the raw carries the last known
    state of every past contract): what the mix looks like when a month expires;
  · in the future months, the composition is the photo of today, at distance h.

Pending maturation of a signal in a future month = final share (last 12 closed months of
the cell) − today's share. The money at risk = the units that will migrate × their price ×
(neutral rate − signed rate). It is reported as `forecast_ajustado_usd` next to the photo
forecast, never replacing it, until the retrospective says the approximation holds.

Every run also appends the photo (`signal_snapshot`): cell × future month × distance ×
state. After enough months the curve of maturation by distance is MEASURED from those
photos and replaces the approximation (mode "medida").

Tables: signal_composition (cell × month × state, final and current), signal_snapshot
(appended), signal_adjustment (cell × future month × signal), signal_alerts.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
NEUTRAL_STATE = "neutro"
MIXED_STATE = "mixto"
MIN_VARIANCE = 0.0475            # p(1−p) floor for the share's binomial error
Z_ALERT = 2.0                    # an alert needs the share to sit 2 sampling errors away
TREND_MONTHS = 12                # window of the final-share trend


def state_of_rows(units: pd.DataFrame, configuration: Config) -> pd.Series:
    """The signal state of every unit: 'neutro', the name of the single active flag, or
    'mixto' when more than one is active."""
    flags = list(configuration.structural_timevarying_dims)
    active = pd.DataFrame({flag: units[flag].isin(configuration.timevarying_positive_values) for flag in flags})
    count = active.sum(axis=1)
    single = active.idxmax(axis=1).where(count == 1, "")
    return pd.Series(np.where(count == 0, NEUTRAL_STATE, np.where(count == 1, single, MIXED_STATE)), index=units.index)


def cell_without_signals(units: pd.DataFrame, configuration: Config) -> pd.Series:
    """The composition cell: mandatory dims + extras de renovación, no timevarying."""
    columns = list(configuration.business_mandatory_dims) + list(configuration.extra_renovacion)
    return join_columns(units, columns)


def composition_table(units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """cell × month × state: units, dollars, share of units, plus `rol` (cerrado / futuro)
    and the distance h for the future months."""
    period, role = configuration.period_col, configuration.dataset_role_col
    frame = units[(units.get("sintetica", 0) == 0) & (units[configuration.pipeline_units_col] > 0)].copy()
    frame["celda_comp"] = cell_without_signals(frame, configuration)
    frame["estado"] = state_of_rows(frame, configuration)
    frame["rol_comp"] = np.where(frame[role].isin(TRUTH_ROLES), "cerrado", np.where(frame[role].isin(FUTURE_ROLES), "futuro", "otro"))
    frame = frame[frame["rol_comp"] != "otro"]
    grouped = frame.groupby(["celda_comp", period, "rol_comp", "estado"], as_index=False).agg(
        unidades=(configuration.pipeline_units_col, "sum"), usd=(configuration.pipeline_usd_col, "sum"))
    totals = grouped.groupby(["celda_comp", period])["unidades"].transform("sum")
    grouped["share"] = grouped["unidades"] / totals.replace(0, np.nan)
    grouped["unidades_celda"] = totals
    current = frame.loc[frame[role].isin(FUTURE_ROLES), period].min()
    grouped["h"] = np.where(grouped["rol_comp"] == "futuro", [int((p - current).n) if pd.notna(current) else np.nan for p in grouped[period]], np.nan)
    return grouped


def final_composition(composition: pd.DataFrame, configuration: Config, window: int) -> pd.DataFrame:
    """Per cell × state: the FINAL share over the last `window` closed months (units-weighted),
    the same-month-last-year share, and the trend of the share (pp per month ± se)."""
    period = configuration.period_col
    closed = composition[composition["rol_comp"] == "cerrado"]
    rows = []
    for (cell, state), block in closed.groupby(["celda_comp", "estado"]):
        block = block.sort_values(period)
        recent = block.tail(window)
        final_share = recent["unidades"].sum() / max(recent["unidades_celda"].sum(), 1e-9)
        slope, slope_se = np.nan, np.nan
        if len(block) >= 6:
            x = np.arange(len(block.tail(TREND_MONTHS)), dtype=float)
            y = block.tail(TREND_MONTHS)["share"].to_numpy(dtype=float)
            if len(x) >= 6 and np.isfinite(y).all():
                design = np.vstack([np.ones_like(x), x]).T
                coefficients, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
                slope = float(coefficients[1])
                residual_variance = float(residuals[0]) / max(len(x) - 2, 1) if len(residuals) else float(np.var(y - design @ coefficients, ddof=2))
                slope_se = float(np.sqrt(residual_variance / max(np.sum((x - x.mean()) ** 2), 1e-9)))
        rows.append(dict(celda_comp=cell, estado=state, share_final=final_share, meses=int(len(recent)),
                         unidades_final_mes=float(recent["unidades_celda"].mean()),
                         tendencia_pp_mes=100 * slope if np.isfinite(slope) else np.nan,
                         tendencia_se_pp_mes=100 * slope_se if np.isfinite(slope_se) else np.nan,
                         por_mes={str(p): float(s) for p, s in zip(block[period], block["share"])}))
    return pd.DataFrame(rows)


def rates_by_cell_and_state(series_card: pd.DataFrame, units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """cell × state → the estimated rate (support-weighted over the series of that cell and state)."""
    one_per_series = units.drop_duplicates("fs_id").set_index("fs_id")
    cell_of = cell_without_signals(one_per_series, configuration)
    state_of = state_of_rows(one_per_series, configuration)
    card = series_card[series_card["tasa_estimada"].notna() & (series_card["n_efectivo"].fillna(0) > 0)].copy()
    card["celda_comp"] = card["fs_id"].map(cell_of)
    card["estado"] = card["fs_id"].map(state_of)
    weight = card["n_efectivo"].clip(lower=1.0)
    rates = (card["tasa_estimada"] * weight).groupby([card["celda_comp"], card["estado"]]).sum() / weight.groupby([card["celda_comp"], card["estado"]]).sum()
    return rates.rename("tasa").reset_index()


def signal_adjustment(composition: pd.DataFrame, final: pd.DataFrame, rates: pd.DataFrame, detail: pd.DataFrame,
                      configuration: Config) -> pd.DataFrame:
    """cell × future month × signal: today's share, the final share expected, the pending
    maturation, the units that will migrate from the neutral state, and the adjustment in
    dollars = − migrating units × neutral AUV × neutral uplift × (neutral rate − signed rate).
    Positive signals (autorenew) adjust upwards. At h ≤ 1 nothing is pending."""
    period = configuration.period_col
    future = composition[composition["rol_comp"] == "futuro"]
    final_share = final.set_index(["celda_comp", "estado"])["share_final"]
    rate_of = rates.set_index(["celda_comp", "estado"])["tasa"]
    rows = []
    for (cell, month), block in future.groupby(["celda_comp", period]):
        neutral_units = float(block.loc[block["estado"] == NEUTRAL_STATE, "unidades"].sum())
        neutral_usd = float(block.loc[block["estado"] == NEUTRAL_STATE, "usd"].sum())
        neutral_auv = neutral_usd / neutral_units if neutral_units > 0 else np.nan
        h = int(block["h"].iloc[0]) if pd.notna(block["h"].iloc[0]) else np.nan
        neutral_rate = rate_of.get((cell, NEUTRAL_STATE), np.nan)
        for flag, polarity in configuration.structural_timevarying_dims.items():
            share_now = float(block.loc[block["estado"] == flag, "share"].sum())
            has_history = any(key[0] == cell for key in final_share.index)
            if not has_history:
                continue
            share_end = float(final_share.get((cell, flag), 0.0))      # a signal never seen in the cell: final share 0
            pending = max(0.0, share_end - share_now) if (np.isfinite(h) and h > 1) else 0.0
            migrating = pending * neutral_units
            signed_rate = rate_of.get((cell, flag), np.nan)
            gap = (neutral_rate - signed_rate) if np.isfinite(neutral_rate) and np.isfinite(signed_rate) else np.nan
            adjustment = -migrating * neutral_auv * gap if np.isfinite(gap) and np.isfinite(neutral_auv) else np.nan
            rows.append(dict(celda_comp=cell, **{period: month}, h=h, senal=flag, polaridad=polarity,
                             share_actual=round(share_now, 4), share_final_esperado=round(share_end, 4), pendiente=round(pending, 4),
                             unidades_neutras=round(neutral_units, 1), unidades_migran=round(migrating, 1),
                             tasa_neutra=round(neutral_rate, 4) if np.isfinite(neutral_rate) else np.nan,
                             tasa_senal=round(signed_rate, 4) if np.isfinite(signed_rate) else np.nan,
                             auv_neutro=round(neutral_auv, 2) if np.isfinite(neutral_auv) else np.nan,
                             ajuste_usd=round(adjustment, 2) if np.isfinite(adjustment) else np.nan))
    return pd.DataFrame(rows)


def signal_alerts(composition: pd.DataFrame, final: pd.DataFrame, adjustment: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Two alerts that need no photo history:
      nivel_anomalo   a future month whose share of a signal, at distance h, is already ABOVE
                      the final share of the cell by ≥ Z_ALERT sampling errors — it will end
                      higher than usual;
      tendencia       the final share of a signal grows month over month (slope ≥ 2 se and
                      ≥ 0.1 pp/month) over the last 12 closed months.
    Money at stake = the excess (or the slope × horizon) × neutral units × AUV × rate gap."""
    period = configuration.period_col
    rows = []
    for _, row in adjustment.iterrows():
        n = row["unidades_neutras"] + row["unidades_migran"]
        share_end, share_now = row["share_final_esperado"], row["share_actual"]
        se = np.sqrt(max(share_end * (1 - share_end), MIN_VARIANCE) / max(n, 1.0))
        z = (share_now - share_end) / se if se > 0 else np.nan
        if np.isfinite(z) and z >= Z_ALERT:
            gap = (row["tasa_neutra"] - row["tasa_senal"]) if pd.notna(row["tasa_senal"]) else np.nan
            money = (share_now - share_end) * n * row["auv_neutro"] * gap if np.isfinite(gap) and pd.notna(row["auv_neutro"]) else np.nan
            rows.append(dict(celda_comp=row["celda_comp"], senal=row["senal"], **{period: row[period]}, h=row["h"], tipo="nivel_anomalo",
                             share_actual=share_now, share_referencia=share_end, z=round(float(z), 2), usd_en_juego=round(-money, 2) if np.isfinite(money) else np.nan,
                             mensaje=f"a {int(row['h'])} meses ya tiene {100 * share_now:.1f}% de {row['senal']}; la proporción final habitual es {100 * share_end:.1f}%"))
    for _, row in final.iterrows():
        if row["estado"] in (NEUTRAL_STATE, MIXED_STATE) or not np.isfinite(row["tendencia_pp_mes"]) or not np.isfinite(row["tendencia_se_pp_mes"]):
            continue
        if row["tendencia_pp_mes"] >= 0.1 and row["tendencia_pp_mes"] >= 2 * row["tendencia_se_pp_mes"]:
            rows.append(dict(celda_comp=row["celda_comp"], senal=row["estado"], **{period: ""}, h=np.nan, tipo="tendencia",
                             share_actual=round(row["share_final"], 4), share_referencia=np.nan, z=round(row["tendencia_pp_mes"] / row["tendencia_se_pp_mes"], 2),
                             usd_en_juego=np.nan, mensaje=f"la proporción final de {row['estado']} crece {row['tendencia_pp_mes']:.2f} pp/mes en los últimos 12 meses cerrados"))
    return pd.DataFrame(rows, columns=["celda_comp", "senal", period, "h", "tipo", "share_actual", "share_referencia", "z", "usd_en_juego", "mensaje"])


def run_signal_maturation(units: pd.DataFrame, series_card: pd.DataFrame, forecast: dict, configuration: Config) -> dict:
    """Phase 5.9-5.11: composition, photo, adjustment, alerts. Returns the tables and the
    adjustment per month (to be carried into horizon_report and business_summary)."""
    period = configuration.period_col
    detail = forecast["forecast_detail"]
    composition = composition_table(units, configuration)
    if composition.empty or detail is None or detail.empty:
        return dict(signal_composition=composition, signal_adjustment=pd.DataFrame(), signal_alerts=pd.DataFrame(), ajuste_por_mes=pd.Series(dtype=float))
    final = final_composition(composition, configuration, configuration.signal_final_window_months)
    rates = rates_by_cell_and_state(series_card, units, configuration)
    adjustment = signal_adjustment(composition, final, rates, detail, configuration)
    alerts = signal_alerts(composition, final, adjustment, configuration)
    photo = composition[composition["rol_comp"] == "futuro"].drop(columns=["rol_comp"]).assign(mes_foto=str(units.loc[units[configuration.dataset_role_col] == ROLE_PROJECTION, period].min()))
    configuration.write(composition, "signal_composition")
    configuration.write(photo, "signal_snapshot", append=True)
    configuration.write(final.drop(columns=["por_mes"]), "signal_final_composition")
    configuration.write(adjustment, "signal_adjustment")
    configuration.write(alerts, "signal_alerts")
    per_month = adjustment.groupby(period)["ajuste_usd"].sum() if len(adjustment) else pd.Series(dtype=float)
    print_signal_maturation(composition, final, adjustment, alerts, per_month, configuration)
    return dict(signal_composition=composition, signal_final_composition=final, signal_adjustment=adjustment,
                signal_alerts=alerts, ajuste_por_mes=per_month)


def print_signal_maturation(composition: pd.DataFrame, final: pd.DataFrame, adjustment: pd.DataFrame,
                            alerts: pd.DataFrame, per_month: pd.Series, configuration: Config) -> None:
    period = configuration.period_col
    closed = composition[composition["rol_comp"] == "cerrado"]
    future = composition[composition["rol_comp"] == "futuro"]
    by_state_closed = closed.groupby("estado")["unidades"].sum() / max(closed["unidades"].sum(), 1e-9)
    by_state_future = future.groupby("estado")["unidades"].sum() / max(future["unidades"].sum(), 1e-9)
    print(f"[5] SIGNAL MATURATION · composition of the cells (mandatory + extras, no timevarying), closed months (final) vs the photo of the future")
    for state in sorted(set(by_state_closed.index) | set(by_state_future.index)):
        print(f"   {state:<14} final {100 * by_state_closed.get(state, 0):5.1f}%   today in the future months {100 * by_state_future.get(state, 0):5.1f}%")
    total = float(per_month.sum()) if len(per_month) else 0.0
    print(f"   pending maturation priced: {total:+,.0f} $ over the horizon "
          f"({', '.join(f'{m}: {v:+,.0f}' for m, v in per_month.head(6).items())}{' …' if len(per_month) > 6 else ''})")
    if len(alerts):
        print(f"   alerts: {(alerts['tipo'] == 'nivel_anomalo').sum()} months already above their final share · {(alerts['tipo'] == 'tendencia').sum()} signals with a rising final share")
        for _, row in alerts.sort_values("z", ascending=False).head(configuration.console_top_rows).iterrows():
            print(f"      {str(row['celda_comp'])[:48]:<48} {row['senal']:<12} {row['tipo']:<13} z={row['z']:>5} {row['mensaje']}")
    explain(configuration,
            "The forecast carries today's signals; softcancel appears right after a renewal and again near the expiry, dormant appears any time and only grows.",
            "So a month far ahead has fewer signed units today than it will have when it expires. Pending maturation = final share of the cell (last 12 closed months) − today's share;",
            "priced as the units that will migrate × their price × (neutral rate − signed rate). It is shown as forecast_ajustado next to the photo, not instead of it.",
            "Each run appends the photo (signal_snapshot); after enough months the curve of maturation by distance is measured from the photos and replaces this approximation.")
