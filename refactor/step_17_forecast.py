"""
step_17_forecast.py — The forecast in money: every future row, its rate, its uplift, its
expected renewals and its band; the total by month and by year.

THE FUTURE ROWS: the fine rows of the extract from the current month on (pipeline 'real'),
and the rows that the SIMULATION WINDOW creates. The window goes from the current month
(included: it is simulated whole) to simulation_end (December). What happens in it falls due
12 months later, and the extract cannot have it yet:
  · proyectada  every 1-year licence (term_column = one_year_term_value) due in the window
                renews as the forecast expects; its renewal falls due the same month next year,
                with the same dims, units = expected renewed units,
                value = expected renewed USD, discount renewal_reentry_discount (0)
  · simulada    the acquisition of the window, simulated for every value of acquisition_column
                in acquisition_values apart (they have different proportions): per group of
                dims, units = units acquired the same month a year before × level (last 3
                closed months vs the same months a year before), value = units × value per unit
                (last 12 closed months); a group that cannot be projected takes its share of
                the projected total of its acquisition value. The timevarying dims are 0 (no
                signal on a new licence) and the discount is acquisition_discount (0.4). An
                acquisition of month m is read in the extract as the pipeline it created: the
                acquisition rows due in m + 12
  The extract's own 1-year rows due 12 months after the window were created by events of the
  window, not known yet: step 02 wiped their pipeline (the raw keeps it in s0_vencen_*).
The time_series universe re-enters in step 20.
Every future row, of the extract or extended, is then predicted the same way:

For every future row:
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

from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
from step_14_backtest import band_of_horizon
from techniques import inverse_logit, logit, predict_logit
from vocabulario import (CALENDAR_ROLE_COLUMN, ESTIMATION_ID_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, PIPELINE_ORIGIN_COLUMN,
                         PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_FROM_CELL,
                         RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
                         TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNIT_ID_COLUMN,
                         UPLIFT_CELL_ID_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "17"
STEP_NAME = "FORECAST"
STEP_PURPOSE = ("predict, row by row, the renewals of every future month: units due × rate of its series (the chosen "
                "technique at its horizon) and USD due × rate × uplift of its cell, each with its band; and the totals "
                "by month and by year with a worst-case and an independent-errors band")
STEP_ACTIONS = ["the future rows of the extract",
                "the rate of every estimation id at every horizon, with its technique",
                "the rate of every future row (pool, cell or global) and its band",
                "the uplift of every future row (contract or statistical) and its band",
                "the expected renewals and their band, row by row",
                "the simulation window: renewals of 1-year licences and acquisitions, due 12 months later",
                "the totals by month and by year, by origin of the pipeline",
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
    """The forecast row by row (extract and extended horizon), by month and by year; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    boundaries = configuration.calendar_boundaries()
    last_closed = boundaries["current"] - 1
    context = dict(series_estimate=series_estimate, pool_series=pool_series, pool_reference=pool_reference, backtest=backtest,
                   uplift_cells=uplift_cells, uplift_verdict=uplift_verdict, rated_units=rated_units,
                   forecast_units=forecast_units, last_closed=last_closed, pool_rate_cache={})

    # [1] the future rows of the extract
    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION].copy()
    future["_fila"] = future.index
    future[PIPELINE_ORIGIN_COLUMN] = PIPELINE_REAL
    configuration.log_action(STEP_LABEL, 1, f"{len(future):,} future rows in the extract, {future[period_column].min()}.."
                                            f"{future[period_column].max()}; ${future[configuration.pipeline_usd_col].sum():,.0f} due; "
                                            f"last closed month {last_closed}")

    # [2]-[5] rate, uplift and expected renewals of the extract's future rows
    future = predict_rows(future, context, configuration)
    configuration.log_action(STEP_LABEL, 2, f"rates of the estimation ids at h = 1..{future['h'].max()} (technique of each id and band)")
    configuration.log_action(STEP_LABEL, 3, f"rate origins: {future['origen_tasa'].value_counts().to_dict()}")
    configuration.log_action(STEP_LABEL, 4, f"uplift paths: {future['via_uplift'].value_counts().to_dict()}")
    configuration.log_action(STEP_LABEL, 5, f"expected ${future['esperado_usd'].sum():,.0f} renewed of "
                                            f"${future[configuration.pipeline_usd_col].sum():,.0f} due in the extract")

    # [6] the simulation window: renewals and acquisitions that fall due 12 months later; the
    #     extract rows of the target months that the simulation replaces
    extension = extend_horizon(fine_table, future, context, configuration)
    if len(extension):
        future = pd.concat([future, extension], ignore_index=True)
    window = configuration.simulation_window
    configuration.log_action(STEP_LABEL, 6, f"simulation window {window[0] if window else '—'}..{window[-1] if window else '—'} → "
                                            f"due {window[0] + 12 if window else '—'}..{window[-1] + 12 if window else '—'}: "
                                            f"{extension[PIPELINE_ORIGIN_COLUMN].value_counts().to_dict() if len(extension) else {}} rows · "
                                            f"${extension[configuration.pipeline_usd_col].sum() if len(extension) else 0:,.0f} due · "
                                            f"${extension['esperado_usd'].sum() if len(extension) else 0:,.0f} expected")

    # [7] the totals
    by_month, by_year = totals(future, fine_table, configuration)
    configuration.log_action(STEP_LABEL, 7, f"{len(by_month)} months · {len(by_year)} years")

    # [8] the checks
    configuration.log_action(STEP_LABEL, 8, "checking the forecast")
    check_forecast(future, future, fine_table, by_month, configuration, check_log)

    # [9] the tables
    configuration.log_action(STEP_LABEL, 9, "writing the forecast, the months and the summary")
    forecast_columns = ([period_column, "h", PIPELINE_ORIGIN_COLUMN, SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
                         ESTIMATION_ID_COLUMN, configuration.pipeline_units_col, configuration.pipeline_usd_col, "origen_tasa",
                         "tecnica", "tasa", "tasa_baja", "tasa_alta", "via_uplift", "uplift", "uplift_bajo", "uplift_alto",
                         "esperado_unidades", "esperado_usd", "esperado_usd_bajo", "esperado_usd_alto"]
                        + configuration.rate_series_columns + configuration.extra_revalorizacion
                        + ([configuration.discount_value_column, configuration.discount_bucket_column]
                           if configuration.discount_value_column else []))
    forecast_columns = list(dict.fromkeys(forecast_columns))
    configuration.write_table(STEP_LABEL, check_log, future[forecast_columns], TABLE_FORECAST)
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_FORECAST_MONTH)
    configuration.write_table(STEP_LABEL, check_log, by_year, TABLE_BUSINESS_SUMMARY)

    # [10] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 10, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [11] the forecast by month and the answers by year
    configuration.log_action(STEP_LABEL, 11, "the forecast by month (banda lineal: every error in the same direction, the "
                                             "worst case; banda cuadratura: independent errors):")
    configuration.show_table(by_month)
    configuration.logger.doc(f"[{STEP_LABEL}] by year: renewed in the closed months + expected in the future ones (by origin "
                             f"of the pipeline):")
    configuration.show_table(by_year)
    return dict(forecast=future, by_month=by_month, by_year=by_year)


def predict_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> pd.DataFrame:
    """Rate, uplift and expected renewals (with bands) of future rows, of the extract or extended."""
    rows = rows.copy()
    rows["h"] = rows[configuration.period_col].map(lambda month: (month - context["last_closed"]).n)
    missing_horizons = [int(horizon) for horizon in sorted(rows["h"].unique()) if int(horizon) not in context["pool_rate_cache"]]
    if missing_horizons:
        new_rates = pool_predictions(context["pool_series"], context["backtest"], missing_horizons, configuration)
        for horizon, block in new_rates.groupby("h"):
            context["pool_rate_cache"][int(horizon)] = block
    pool_rates = pd.concat([context["pool_rate_cache"][int(horizon)] for horizon in sorted(rows["h"].unique())
                            if int(horizon) in context["pool_rate_cache"]], ignore_index=True)
    rows = rate_of_rows(rows, context["series_estimate"], pool_rates, context["pool_reference"], context["rated_units"],
                        context["backtest"], context["forecast_units"], configuration)
    rows = uplift_of_rows(rows, context["uplift_cells"], context["uplift_verdict"], configuration)
    rows["esperado_unidades"] = rows[configuration.pipeline_units_col] * rows["tasa"]
    rows["esperado_usd"] = rows[configuration.pipeline_usd_col] * rows["tasa"] * rows["uplift"]
    rows["esperado_usd_bajo"] = rows[configuration.pipeline_usd_col] * rows["tasa_baja"] * rows["uplift_bajo"]
    rows["esperado_usd_alto"] = rows[configuration.pipeline_usd_col] * rows["tasa_alta"] * rows["uplift_alto"]
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# THE EXTENDED HORIZON
# ═══════════════════════════════════════════════════════════════════════════════════

def is_acquisition(rows: pd.DataFrame, configuration: Config) -> pd.Series:
    """The acquisition rows (acquisition_column in acquisition_values); none when it is not declared."""
    if not configuration.acquisition_column or not configuration.acquisition_values:
        return pd.Series(False, index=rows.index)
    return rows[configuration.acquisition_column].isin(configuration.acquisition_values)


def inactive_value(values: pd.Series):
    """The value that means 'flag off' in a timevarying column, as the data writes it (False, 0, "0")."""
    inactive = values[~values.isin(ACTIVE_FLAG_VALUES)].dropna()
    return inactive.mode().iloc[0] if len(inactive) else 0


def extend_horizon(fine_table: pd.DataFrame, extract_future: pd.DataFrame, context: dict, configuration: Config) -> pd.DataFrame:
    """The rows the simulation window creates, predicted like any future row.

    INPUT:   the fine table (normal universe) · the extract's future rows, already predicted · the context.
    OUTPUT:  the new rows (proyectada and simulada), predicted like any future row.
    RULES:   window = current month … simulation_end; a row of the window falls due in month + 12.
             proyectada: 1-year rows of the window, expected renewals, same dims, discount 0.
             simulada: acquisitions of the window, per acquisition value and group, level × same
             month a year before, value per unit of 12 months, timevarying off, acquisition_discount.
             The group of an acquisition keeps the grain of the pipeline: every mandatory dim, every
             extra_renovacion and every extra_revalorizacion (built from the Config: a new dim joins by
             itself); the discount is one value (acquisition_discount), the timevarying dims are off.
    EDGE CASES: an empty window → nothing. No acquisition_column → no simulada. A group with no
             history for the level → its share of the projected total of its acquisition value.
    """
    period = configuration.period_col
    window = configuration.simulation_window
    if not window:
        return pd.DataFrame()

    carried = list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion
                                 + ([configuration.term_column] if configuration.term_column else [])
                                 + ([configuration.acquisition_column] if configuration.acquisition_column else [])))
    carried = [column for column in carried if column in extract_future.columns]

    # proyectada: the expected renewals of the 1-year licences due in the window
    renewing = extract_future[extract_future[period].isin(window) & is_one_year(extract_future, configuration)
                              & (extract_future["esperado_unidades"] > 0)]
    projected = renewing[carried].copy()
    projected[period] = renewing[period].to_numpy() + 12
    projected[configuration.pipeline_units_col] = renewing["esperado_unidades"].to_numpy()
    projected[configuration.pipeline_usd_col] = renewing["esperado_usd"].to_numpy()
    if configuration.discount_value_column:
        projected[configuration.discount_value_column] = configuration.renewal_reentry_discount
    projected[PIPELINE_ORIGIN_COLUMN] = PIPELINE_PROJECTED

    # simulada: the acquisitions of the window
    simulated = simulate_acquisitions(fine_table, window, carried, configuration)

    new_rows = pd.concat([frame for frame in (projected, simulated) if len(frame)], ignore_index=True) \
        if len(projected) or len(simulated) else pd.DataFrame()
    if new_rows.empty:
        return new_rows
    new_rows = with_ids(new_rows, configuration)
    new_rows = predict_rows(new_rows, context, configuration)
    new_rows["_fila"] = np.nan
    return new_rows


def simulate_acquisitions(fine_table: pd.DataFrame, window: list, carried: list, configuration: Config) -> pd.DataFrame:
    """The acquisitions of the window, one line per group of dims and acquisition value, due 12 months later.
    An acquisition of month m is read as the pipeline it created: the 1-year acquisition rows due in m + 12."""
    if not configuration.acquisition_column or not configuration.acquisition_values:
        return pd.DataFrame()
    period, current = configuration.period_col, configuration.calendar_boundaries()["current"]
    timevarying = list(configuration.structural_timevarying_dims)
    group_columns = [column for column in carried if column not in timevarying and column != configuration.term_column]
    acquired = fine_table[is_acquisition(fine_table, configuration) & is_one_year(fine_table, configuration)].copy()
    if acquired.empty:
        return pd.DataFrame()
    acquired["_mes_evento"] = acquired[period] - 12
    acquired["_grupo"] = join_columns(acquired, group_columns)
    events = (acquired[acquired["_mes_evento"] < current]
              .groupby(["_grupo", configuration.acquisition_column, "_mes_evento"])
              [[configuration.pipeline_units_col, configuration.pipeline_usd_col]].sum().reset_index()
              .rename(columns={configuration.pipeline_units_col: "unidades", configuration.pipeline_usd_col: "valor"}))
    dims_of_group = acquired.drop_duplicates("_grupo").set_index("_grupo")[group_columns]
    level_months = list(pd.period_range(current - configuration.acquisition_level_window_months, current - 1, freq="M"))
    auv_start = current - configuration.acquisition_auv_window_months

    rows = []
    for acquisition_value, block in events.groupby(configuration.acquisition_column):
        by_group = block.set_index(["_grupo", "_mes_evento"])["unidades"]
        recent = block[block["_mes_evento"].isin(level_months)].groupby("_grupo")["unidades"].sum()
        before = block[block["_mes_evento"].isin([month - 12 for month in level_months])].groupby("_grupo")["unidades"].sum()
        level = (recent / before.replace(0, np.nan)).dropna()
        total_level = recent.sum() / before.sum() if before.sum() > 0 else np.nan
        auv_window = block[block["_mes_evento"] >= auv_start]
        auv = auv_window.groupby("_grupo")["valor"].sum() / auv_window.groupby("_grupo")["unidades"].sum().replace(0, np.nan)
        total_auv = auv_window["valor"].sum() / auv_window["unidades"].sum() if auv_window["unidades"].sum() > 0 else np.nan
        share = auv_window.groupby("_grupo")["unidades"].sum() / auv_window["unidades"].sum() if auv_window["unidades"].sum() > 0 else pd.Series(dtype=float)
        total_by_month = block.groupby("_mes_evento")["unidades"].sum()
        for month in window:
            for group in share.index:
                same_month = by_group.get((group, month - 12), np.nan)
                group_level = level.get(group, np.nan)
                by_share = not (np.isfinite(same_month) and np.isfinite(group_level))
                units = (total_by_month.get(month - 12, np.nan) * total_level * share[group]) if by_share else same_month * group_level
                if not np.isfinite(units) or units <= 0:
                    continue
                unit_value = auv.get(group, np.nan)
                unit_value = unit_value if np.isfinite(unit_value) else total_auv
                row = dims_of_group.loc[group].to_dict()
                row.update({configuration.acquisition_column: acquisition_value, period: month + 12,
                            configuration.pipeline_units_col: float(units),
                            configuration.pipeline_usd_col: float(units * unit_value), "_por_cuota": int(by_share)})
                rows.append(row)
    simulated = pd.DataFrame(rows)
    if simulated.empty:
        return simulated
    for column_name in timevarying:
        simulated[column_name] = inactive_value(fine_table[column_name])
    if configuration.term_column and configuration.term_column in fine_table.columns:
        simulated[configuration.term_column] = configuration.one_year_term_value
    if configuration.discount_value_column:
        simulated[configuration.discount_value_column] = configuration.acquisition_discount
    simulated[PIPELINE_ORIGIN_COLUMN] = PIPELINE_SIMULATED
    return simulated.drop(columns="_por_cuota")


def with_ids(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The ids of a simulated row: its series, its forecast unit, its discount bucket and its uplift cell."""
    rows = rows.copy()
    rows[SERIES_ID_COLUMN] = join_columns(rows, configuration.rate_series_columns)
    rows[UNIT_ID_COLUMN] = rows[SERIES_ID_COLUMN] + "|" + rows[configuration.period_col].astype(str)
    if configuration.discount_value_column:
        rows[configuration.discount_bucket_column] = discount_bucket_labels(rows[configuration.discount_value_column],
                                                                           configuration.discount_bucket_edges)
    rows[UPLIFT_CELL_ID_COLUMN] = join_columns(rows, configuration.uplift_cell_columns)
    rows[CALENDAR_ROLE_COLUMN] = ROLE_PROJECTION
    return rows


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
    """The rate of every future row: its group's prediction (moved toward its credibility reference),
    mandatory cell, or global; and its band."""
    estimate = series_estimate[[SERIES_ID_COLUMN, ESTIMATION_ID_COLUMN, "z", "group_rate", "ref_rate"]]
    future = future.drop(columns=[column for column in (ESTIMATION_ID_COLUMN, "z", "group_rate", "ref_rate", "tecnica",
                                                        "tasa_pool_h", "tasa_pool", "tasa", "origen_tasa")
                                  if column in future.columns])
    future = future.merge(estimate, on=SERIES_ID_COLUMN, how="left")
    future = future.merge(pool_rates, on=[ESTIMATION_ID_COLUMN, "h"], how="left")
    future = future.merge(pool_reference[[ESTIMATION_ID_COLUMN, "tasa_pool"]], on=ESTIMATION_ID_COLUMN, how="left")

    # the group's predicted rate, moved toward its credibility reference by (1 − z) of the difference
    # of levels (logit scale): the same blend as step 11, applied to the prediction
    pool_rate = future["tasa_pool_h"]
    shift = np.zeros(len(future))
    if configuration.apply_credibility_shift:
        blended = (future["z"] < 1) & future["ref_rate"].notna() & future["group_rate"].notna()
        shift = np.where(blended, (1 - future["z"].fillna(1)) * (logit(future["ref_rate"].fillna(0.5))
                                                                  - logit(future["group_rate"].fillna(0.5))), 0.0)
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
    unit_units = unit_units.fillna(future[configuration.pipeline_units_col])     # an extended row: its own units
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
    # a cell never seen (an extended row with a new discount bucket): its mandatory cell, then the global uplift
    if statistical.isna().any():
        uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
        weights = uplift_cells["renovadores"].clip(lower=1e-9)
        by_mandatory = (uplift_cells.assign(_w=weights, _wu=weights * uplift_cells["uplift"])
                        .groupby("uplift_celda_mandatory_id")[["_w", "_wu"]].sum())
        mandatory_uplift = by_mandatory["_wu"] / by_mandatory["_w"]
        global_uplift = float((weights * uplift_cells["uplift"]).sum() / weights.sum())
        fallback = join_columns(future, list(uplift_mandatory)).map(mandatory_uplift).fillna(global_uplift)
        statistical = statistical.fillna(fallback)
    future = future.drop(columns=[column for column in ("uplift", "uplift_bajo", "uplift_alto", "via_uplift") if column in future.columns])
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
    """By month (expected, linear band, quadrature band, split by origin of the pipeline) and by
    year (real renewals of the closed months + expected by origin)."""
    period_column = configuration.period_col
    future = future.assign(_media_baja=(future["esperado_usd"] - future["esperado_usd_bajo"]) ** 2,
                           _media_alta=(future["esperado_usd_alto"] - future["esperado_usd"]) ** 2)
    grouped = future.groupby(period_column)
    by_month = pd.DataFrame({"usd_vence": grouped[configuration.pipeline_usd_col].sum(),
                             "esperado_usd": grouped["esperado_usd"].sum(),
                             "banda_lineal_baja": grouped["esperado_usd_bajo"].sum(),
                             "banda_lineal_alta": grouped["esperado_usd_alto"].sum(),
                             "banda_cuadratura_baja": grouped["esperado_usd"].sum() - np.sqrt(grouped["_media_baja"].sum()),
                             "banda_cuadratura_alta": grouped["esperado_usd"].sum() + np.sqrt(grouped["_media_alta"].sum())})
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_month[f"esperado_{origin}"] = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(period_column)["esperado_usd"].sum()
    by_month = by_month.fillna(0.0).reset_index()
    by_month["tasa_usd"] = by_month["esperado_usd"] / by_month["usd_vence"]

    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    closed_by_year = closed.groupby(closed[period_column].map(lambda month: month.year))[configuration.renewed_usd_col].sum()
    year_of = future[period_column].map(lambda month: month.year)
    future_by_year = future.groupby(year_of)
    years = sorted(set(closed_by_year.index) | set(future_by_year.groups))
    by_year = pd.DataFrame({"ano": years})
    by_year["renovado_real_usd"] = by_year["ano"].map(closed_by_year).fillna(0.0)
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_origin = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(year_of[future[PIPELINE_ORIGIN_COLUMN] == origin])["esperado_usd"].sum()
        by_year[f"esperado_{origin}_usd"] = by_year["ano"].map(by_origin).fillna(0.0)
    by_year["esperado_usd"] = by_year["ano"].map(future_by_year["esperado_usd"].sum()).fillna(0.0)
    by_year["total_usd"] = by_year["renovado_real_usd"] + by_year["esperado_usd"]
    by_year["banda_cuadratura_usd"] = by_year["ano"].map(np.sqrt(future_by_year["_media_alta"].sum())).fillna(0.0)
    by_year["banda_lineal_baja"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_bajo"].sum()).fillna(0.0)
    by_year["banda_lineal_alta"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_alto"].sum()).fillna(0.0)
    return by_month, by_year


def check_forecast(future: pd.DataFrame, all_future: pd.DataFrame, fine_table: pd.DataFrame, by_month: pd.DataFrame,
                   configuration: Config, check_log: list) -> None:
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
    future_roles = fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    configuration.log_check(STEP_LABEL, check_log, "no closed row is forecast; every future row of the extract is",
                            set(all_future.loc[all_future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, "_fila"])
                            == set(fine_table.index[future_roles]),
                            failure_detail="the forecast rows are not the future rows")
    real_rows = all_future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL
    due_difference = all_future.loc[real_rows, configuration.pipeline_usd_col].sum() - fine_table.loc[future_roles, configuration.pipeline_usd_col].sum()
    configuration.log_check(STEP_LABEL, check_log, "the pipeline of the extract's future rows is conserved (Σ USD due)",
                            abs(due_difference) <= MONEY_TOLERANCE, failure_detail=f"difference ${due_difference:,.2f}",
                            context=f"${all_future.loc[real_rows, configuration.pipeline_usd_col].sum():,.0f} due in the extract")
    total_difference = by_month["esperado_usd"].sum() - future["esperado_usd"].sum()
    configuration.log_check(STEP_LABEL, check_log, "the totals are the sum of the rows", abs(total_difference) <= MONEY_TOLERANCE,
                            failure_detail=f"difference ${total_difference:,.2f}", context=f"${future['esperado_usd'].sum():,.0f} expected")
