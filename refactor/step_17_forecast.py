"""
step_17_forecast.py — The forecast in money: every future row, its rate, its uplift, its
expected renewals and its band; the total by month and by year.

For every future fine row (roles pendiente_cierre and proyeccion):
  · h = months from the last closed month to the row's month (1 = the first month to predict)
  · THE RATE of its series:
      pool    the series has an estimation id (step 11): the technique chosen for that id
              and horizon band (step 14) predicts the id's rate at h, learning from EVERY
              closed month (entrenamiento and examen). With apply_credibility_shift, a
              series that borrows keeps its own difference of level with the pool in
              proportion to its credibility: logit(rate) = logit(pool prediction) +
              z · (logit(own rate) − logit(pool rate)).
      celda   no estimation id (solo_futuro, time_series universe): the rate of its mandatory
              cell over the closed months
      global  not even that: the rate of the whole portfolio
    Its band: the quantiles of the normalised error of its technique at the judged horizon
    (h = 1 → 1; any other h → the judged horizon above it, or the last), times the binomial
    error of the rate with the units due of its forecast unit; clipped to [0, 1].
  · THE UPLIFT of its cell: the CONTRACT rule 1 / (1 − discount) where the discount is known
    and the uplift backtest (step 16) chose it; the STATISTICAL uplift of its cell otherwise
    (with its bootstrap band).
  · EXPECTED: renewed units = units due × rate; renewed USD = USD due × rate × uplift;
    the band of the USD from the bands of the rate and the uplift.
THE TOTALS by month and by year, with two bands: the LINEAR one (the sum of the row bands:
every error in the same direction, the worst case) and the QUADRATURE one (√Σ of the row
half-widths²: independent errors). The truth is between them.

Actions (logged as they are done):
  1. the future rows and their horizon
  2. the rate of every estimation id at every horizon, with its technique
  3. the rate of every future row (pool, cell or global) and its band
  4. the uplift of every future row (contract or statistical) and its band
  5. the expected renewals and their band, row by row
  6. the totals by month and by year
  7. check the forecast                                              checks 1-5
  8. write the forecast, the months and the summary                  checks 6-8
  9. count the checks; stop if any failed
 10. show the forecast by month and the answers by year, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every future row has a rate, an uplift and an expected value
   2. every rate and every band is inside [0, 1] and the band contains the rate
   3. no closed row is forecast; every future row is
   4. the pipeline of the future rows is conserved (Σ USD due)
   5. the totals are the sum of the rows
   6-8. tables sff_forecast, sff_forecast_mes, sff_resumen_negocio written and read back

Output: (the forecast row by row, by month, by year) · tables sff_forecast, sff_forecast_mes,
sff_resumen_negocio.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from step_14_backtest import band_of_horizon
from techniques import inverse_logit, logit, predict_logit
from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, RATE_FROM_CELL,
                         RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PENDING, ROLE_PROJECTION, SERIES_ID_COLUMN,
                         TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNIT_ID_COLUMN,
                         UPLIFT_CELL_ID_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "17"
STEP_NAME = "FORECAST"
STEP_PURPOSE = ("predict, row by row, the renewals of every future month: units due × rate of its series (the chosen "
                "technique at its horizon) and USD due × rate × uplift of its cell, each with its band; and the totals "
                "by month and by year with a worst-case and an independent-errors band")
STEP_ACTIONS = ["the future rows and their horizon",
                "the rate of every estimation id at every horizon, with its technique",
                "the rate of every future row (pool, cell or global) and its band",
                "the uplift of every future row (contract or statistical) and its band",
                "the expected renewals and their band, row by row",
                "the totals by month and by year",
                "check the forecast (checks 1-5)",
                "write the forecast, the months and the summary (checks 6-8)",
                "count the checks; stop if any failed",
                "show the forecast by month and the answers by year, as tables"]
STEP_OUTPUT = "every future row with rate, uplift, expected USD and bands · totals by month and year · three tables"

PERCENTAGE_POINTS = 100
MONEY_TOLERANCE = 0.01


def assemble_forecast(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, series_estimate: pd.DataFrame,
                      pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict, uplift_cells: pd.DataFrame,
                      uplift_verdict: dict, rated_units: pd.DataFrame, configuration: Config) -> dict:
    """The forecast row by row, by month and by year; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    boundaries = configuration.calendar_boundaries()
    last_closed = boundaries["pending_start"] - 1

    # [1] the future rows
    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION])].copy()
    future["_fila"] = future.index
    future["h"] = future[period_column].map(lambda month: (month - last_closed).n)
    configuration.log_action(STEP_LABEL, 1, f"{len(future):,} future rows, {future[period_column].min()}..{future[period_column].max()}, "
                                            f"h = 1..{future['h'].max()} from the last closed month {last_closed}; "
                                            f"${future[configuration.pipeline_usd_col].sum():,.0f} due")

    # [2] the rate of every estimation id at every horizon
    pool_rates = pool_predictions(pool_series, backtest, sorted(future["h"].unique()), configuration)
    configuration.log_action(STEP_LABEL, 2, f"{len(pool_rates):,} (estimation id, h) predictions")

    # [3] the rate of every row
    future = rate_of_rows(future, series_estimate, pool_rates, pool_reference, rated_units, backtest, forecast_units,
                          configuration)
    configuration.log_action(STEP_LABEL, 3, f"rate origins: {future['origen_tasa'].value_counts().to_dict()} · "
                                            f"mean rate {np.average(future['tasa'], weights=future[configuration.pipeline_units_col] + 1e-9):.1%}")

    # [4] the uplift of every row
    future = uplift_of_rows(future, uplift_cells, uplift_verdict, configuration)
    configuration.log_action(STEP_LABEL, 4, f"uplift paths: {future['via_uplift'].value_counts().to_dict()}")

    # [5] the expected renewals, row by row
    future["esperado_unidades"] = future[configuration.pipeline_units_col] * future["tasa"]
    future["esperado_usd"] = future[configuration.pipeline_usd_col] * future["tasa"] * future["uplift"]
    future["esperado_usd_bajo"] = future[configuration.pipeline_usd_col] * future["tasa_baja"] * future["uplift_bajo"]
    future["esperado_usd_alto"] = future[configuration.pipeline_usd_col] * future["tasa_alta"] * future["uplift_alto"]
    configuration.log_action(STEP_LABEL, 5, f"expected ${future['esperado_usd'].sum():,.0f} renewed of "
                                            f"${future[configuration.pipeline_usd_col].sum():,.0f} due")

    # [6] the totals
    by_month, by_year = totals(future, fine_table, configuration)
    configuration.log_action(STEP_LABEL, 6, f"{len(by_month)} months · {len(by_year)} years")

    # [7] the checks
    configuration.log_action(STEP_LABEL, 7, "checking the forecast")
    check_forecast(future, fine_table, by_month, configuration, check_log)

    # [8] the tables
    configuration.log_action(STEP_LABEL, 8, "writing the forecast, the months and the summary")
    forecast_columns = ([period_column, "h", SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN, ESTIMATION_ID_COLUMN,
                         configuration.pipeline_units_col, configuration.pipeline_usd_col, "origen_tasa", "tecnica",
                         "tasa", "tasa_baja", "tasa_alta", "via_uplift", "uplift", "uplift_bajo", "uplift_alto",
                         "esperado_unidades", "esperado_usd", "esperado_usd_bajo", "esperado_usd_alto"]
                        + ([configuration.discount_value_column] if configuration.discount_value_column else []))
    configuration.write_table(STEP_LABEL, check_log, future[forecast_columns], TABLE_FORECAST)
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_FORECAST_MONTH)
    configuration.write_table(STEP_LABEL, check_log, by_year, TABLE_BUSINESS_SUMMARY)

    # [9] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 9, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [10] the forecast by month and the answers by year
    configuration.log_action(STEP_LABEL, 10, "the forecast by month (banda lineal: every error in the same direction, the "
                                             "worst case; banda cuadratura: independent errors):")
    configuration.show_table(by_month)
    configuration.logger.doc(f"[{STEP_LABEL}] by year: what was renewed in the closed months + what is expected in the future ones:")
    configuration.show_table(by_year)
    return dict(forecast=future, by_month=by_month, by_year=by_year)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RATE
# ═══════════════════════════════════════════════════════════════════════════════════

def judged_horizon(horizon: int, judged: list) -> int:
    """The judged horizon whose band a horizon takes: itself, the next one above, or the last."""
    above = [judged_h for judged_h in sorted(judged) if judged_h >= horizon]
    return above[0] if above else max(judged)


def pool_predictions(pool_series: pd.DataFrame, backtest: dict, horizons: list, configuration: Config) -> pd.DataFrame:
    """The rate of every estimation id at every future horizon, with the technique of its band,
    learning from every closed month of the id."""
    decision = backtest["decision"].set_index([ESTIMATION_ID_COLUMN, "tramo_h"])["tecnica"]
    truth = pool_series[pool_series["rol"].isin(TRUTH_ROLES) & pool_series["tasa"].notna() & (pool_series["vencen"] > 0)]
    rows = []
    for estimation_id, monthly in truth.groupby(ESTIMATION_ID_COLUMN):
        monthly = monthly.sort_values(configuration.period_col)
        history = logit(monthly["tasa"].to_numpy(dtype=float))
        calendar_months = np.array([month.month for month in monthly[configuration.period_col]])
        for horizon in horizons:
            band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
            technique = decision.get((estimation_id, band_name), configuration.challenger_technique)
            predicted = predict_logit(technique, history, calendar_months, int(horizon))
            if not np.isfinite(predicted):
                technique = configuration.challenger_technique
                predicted = predict_logit(technique, history, calendar_months, int(horizon))
            rows.append((estimation_id, int(horizon), technique, float(inverse_logit(predicted))))
    return pd.DataFrame(rows, columns=[ESTIMATION_ID_COLUMN, "h", "tecnica", "tasa_pool_h"])


def rate_of_rows(future: pd.DataFrame, series_estimate: pd.DataFrame, pool_rates: pd.DataFrame, pool_reference: pd.DataFrame,
                 rated_units: pd.DataFrame, backtest: dict, forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The rate of every future row: pool (with the credibility shift), mandatory cell, or global; and its band."""
    estimate = series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "tasa_propia", "peldano"]]
    future = future.merge(estimate, on=SERIES_ID_COLUMN, how="left")
    future = future.merge(pool_rates, on=[ESTIMATION_ID_COLUMN, "h"], how="left")
    future = future.merge(pool_reference[[ESTIMATION_ID_COLUMN, "tasa_pool"]], on=ESTIMATION_ID_COLUMN, how="left")

    # the pool rate, shifted by the series' own level in proportion to its credibility
    pool_rate = future["tasa_pool_h"]
    shift = np.zeros(len(future))
    if configuration.apply_credibility_shift:
        borrows = (future["peldano"] > 0) & future["tasa_propia"].notna() & future["tasa_pool"].notna()
        shift = np.where(borrows, future["z"].fillna(0) * (logit(future["tasa_propia"].fillna(0.5)) - logit(future["tasa_pool"].fillna(0.5))), 0.0)
    future["tasa"] = np.where(pool_rate.notna(), inverse_logit(logit(pool_rate.fillna(0.5)) + shift), np.nan)
    future["origen_tasa"] = np.where(pool_rate.notna(), RATE_FROM_POOL, None)

    # no pool: the rate of the mandatory cell, then the global rate (closed months)
    truth = rated_units[rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & rated_units["tasa"].notna()].copy()
    truth["_celda"] = join_columns(truth, configuration.business_mandatory_dims)
    cell_rates = truth.groupby("_celda")[configuration.renewed_units_col].sum() / truth.groupby("_celda")[configuration.pipeline_units_col].sum()
    global_rate = truth[configuration.renewed_units_col].sum() / truth[configuration.pipeline_units_col].sum()
    future["_celda"] = join_columns(future, configuration.business_mandatory_dims)
    without_pool = future["tasa"].isna()
    cell_rate = future["_celda"].map(cell_rates)
    future.loc[without_pool & cell_rate.notna(), "origen_tasa"] = RATE_FROM_CELL
    future.loc[without_pool & cell_rate.notna(), "tasa"] = cell_rate
    future.loc[future["tasa"].isna(), "origen_tasa"] = RATE_FROM_GLOBAL
    future["tasa"] = future["tasa"].fillna(global_rate)
    future["tecnica"] = future["tecnica"].fillna(configuration.challenger_technique)

    # the band: quantiles of the normalised error × binomial error with the unit's units due
    bands = backtest["bands"].set_index(["tecnica", "h"])
    judged = list(configuration.backtest_horizons)
    unit_units = future[UNIT_ID_COLUMN].map(forecast_units.set_index(UNIT_ID_COLUMN)[configuration.pipeline_units_col])
    se_rate = np.sqrt(future["tasa"] * (1 - future["tasa"]) / unit_units.clip(lower=1))
    keys = list(zip(future["tecnica"], future["h"].map(lambda horizon: judged_horizon(int(horizon), judged))))
    challenger_keys = [(configuration.challenger_technique, judged_h) for _, judged_h in keys]
    q_low = np.array([bands["q_low_norm"].get(key, bands["q_low_norm"].get(fallback, -configuration.z))
                      for key, fallback in zip(keys, challenger_keys)])
    q_high = np.array([bands["q_high_norm"].get(key, bands["q_high_norm"].get(fallback, configuration.z))
                       for key, fallback in zip(keys, challenger_keys)])
    future["tasa_baja"] = np.clip(future["tasa"] + np.minimum(q_low, 0) * se_rate, 0, 1)
    future["tasa_alta"] = np.clip(future["tasa"] + np.maximum(q_high, 0) * se_rate, 0, 1)
    return future.drop(columns=["_celda"])


def uplift_of_rows(future: pd.DataFrame, uplift_cells: pd.DataFrame, uplift_verdict: dict, configuration: Config) -> pd.DataFrame:
    """Contract where the discount is known and the backtest chose it; statistical otherwise."""
    cells = uplift_cells.set_index(UPLIFT_CELL_ID_COLUMN)
    statistical = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift"])
    future["uplift"] = statistical
    future["uplift_bajo"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_bajo"]).fillna(statistical)
    future["uplift_alto"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_alto"]).fillna(statistical)
    future["via_uplift"] = PATH_STATISTICAL
    discount = configuration.discount_value_column
    if discount and uplift_verdict.get("via_usada_con_descuento") == PATH_CONTRACT:
        known = future[discount].notna()
        contract = 1 / (1 - future[discount].clip(upper=0.99))
        future.loc[known, "uplift"] = contract
        future.loc[known, "uplift_bajo"] = contract
        future.loc[known, "uplift_alto"] = contract
        future.loc[known, "via_uplift"] = PATH_CONTRACT
    future["uplift_bajo"] = np.minimum(future["uplift_bajo"], future["uplift"])
    future["uplift_alto"] = np.maximum(future["uplift_alto"], future["uplift"])
    return future


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TOTALS AND THE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════

def totals(future: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """By month (expected, linear band, quadrature band) and by year (closed + future)."""
    period_column = configuration.period_col
    future = future.assign(_media_baja=(future["esperado_usd"] - future["esperado_usd_bajo"]) ** 2,
                           _media_alta=(future["esperado_usd_alto"] - future["esperado_usd"]) ** 2)
    grouped = future.groupby(period_column)
    by_month = pd.DataFrame({"usd_vence": grouped[configuration.pipeline_usd_col].sum(),
                             "esperado_usd": grouped["esperado_usd"].sum(),
                             "banda_lineal_baja": grouped["esperado_usd_bajo"].sum(),
                             "banda_lineal_alta": grouped["esperado_usd_alto"].sum(),
                             "banda_cuadratura_baja": grouped["esperado_usd"].sum() - np.sqrt(grouped["_media_baja"].sum()),
                             "banda_cuadratura_alta": grouped["esperado_usd"].sum() + np.sqrt(grouped["_media_alta"].sum())}).reset_index()
    by_month["tasa_usd"] = by_month["esperado_usd"] / by_month["usd_vence"]

    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    closed_by_year = closed.groupby(closed[period_column].map(lambda month: month.year))[configuration.renewed_usd_col].sum()
    future_by_year = future.groupby(future[period_column].map(lambda month: month.year))
    years = sorted(set(closed_by_year.index) | set(future_by_year.groups))
    by_year = pd.DataFrame({"ano": years})
    by_year["renovado_real_usd"] = by_year["ano"].map(closed_by_year).fillna(0.0)
    by_year["esperado_usd"] = by_year["ano"].map(future_by_year["esperado_usd"].sum()).fillna(0.0)
    by_year["total_usd"] = by_year["renovado_real_usd"] + by_year["esperado_usd"]
    by_year["banda_cuadratura_usd"] = by_year["ano"].map(np.sqrt(future_by_year["_media_alta"].sum())).fillna(0.0)
    by_year["banda_lineal_baja"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_bajo"].sum()).fillna(0.0)
    by_year["banda_lineal_alta"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_alto"].sum()).fillna(0.0)
    return by_month, by_year


def check_forecast(future: pd.DataFrame, fine_table: pd.DataFrame, by_month: pd.DataFrame, configuration: Config,
                   check_log: list) -> None:
    """Checks 1 to 5."""
    missing = future[["tasa", "uplift", "esperado_usd"]].isna().any(axis=1)
    configuration.log_check(STEP_LABEL, check_log, "every future row has a rate, an uplift and an expected value",
                            not missing.any(), failure_detail=f"{int(missing.sum()):,} rows incomplete",
                            context=f"{len(future):,} rows")
    bad_band = ~(future["tasa"].between(0, 1) & (future["tasa_baja"] <= future["tasa"] + 1e-12)
                 & (future["tasa"] <= future["tasa_alta"] + 1e-12) & future["tasa_baja"].between(0, 1) & future["tasa_alta"].between(0, 1))
    configuration.log_check(STEP_LABEL, check_log, "every rate and band is inside [0, 1] and the band contains the rate",
                            not bad_band.any(), failure_detail=f"{int(bad_band.sum()):,} rows with a bad rate or band",
                            examples=future.loc[bad_band, ["tasa", "tasa_baja", "tasa_alta"]])
    future_roles = fine_table[CALENDAR_ROLE_COLUMN].isin([ROLE_PENDING, ROLE_PROJECTION])
    configuration.log_check(STEP_LABEL, check_log, "no closed row is forecast; every future row is",
                            set(future["_fila"]) == set(fine_table.index[future_roles]),
                            failure_detail="the forecast rows are not the future rows")
    due_difference = future[configuration.pipeline_usd_col].sum() - fine_table.loc[future_roles, configuration.pipeline_usd_col].sum()
    configuration.log_check(STEP_LABEL, check_log, "the pipeline of the future rows is conserved (Σ USD due)",
                            abs(due_difference) <= MONEY_TOLERANCE, failure_detail=f"difference ${due_difference:,.2f}",
                            context=f"${future[configuration.pipeline_usd_col].sum():,.0f} due")
    total_difference = by_month["esperado_usd"].sum() - future["esperado_usd"].sum()
    configuration.log_check(STEP_LABEL, check_log, "the totals are the sum of the rows", abs(total_difference) <= MONEY_TOLERANCE,
                            failure_detail=f"difference ${total_difference:,.2f}", context=f"${future['esperado_usd'].sum():,.0f} expected")
