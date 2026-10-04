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
                value = expected renewed USD, discount renewal_reentry_discount (0); an acquisition
                that renews is no longer one: acquisition_column becomes renewed_acquisition_value
                (an acquisition of the window must not be counted again as one next year); the dims
                of dims_after_renewal take their declared value; the timevarying marks are neutral
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
THE DISTRIBUTION of the timevarying marks (the negative structural_timevarying_dims: softcancel,
dormant, not_installed…). The future rows carry TODAY's marks: a licence due in 10 months has not had the
time to stop using the product or to be softcancelled, and a projected or simulated row is neutral because
its marks are unknown. For every group of the future (every dim but the timevarying ones) and month, the mix
of mark combinations moves toward the mix of the same calendar month in the history (where the marks are
final): neutral units move to the marked combinations (a mark only grows: the extract records its last
state) and renew at their historical rate. The forecast IS the forecast with the marks distributed; the base
(today's marks) is kept next to it (esperado_*_base) and the table by month shows both
(sff_forecast_maduracion). It predicts, besides the rate and the uplift, the proportion of the marks.
THE TOTALS by month and by year, with two bands: the LINEAR one (the sum of the row bands:
every error in the same direction, the worst case) and the QUADRATURE one (√Σ of the row
half-widths²: independent errors). The truth is between them.

Actions (logged as they are done):
  1. the future rows and their horizon
  2. the rate of every estimation id at every horizon, with its technique
  3. the rate of every future row (pool, cell or global) and its band
  4. the uplift of every future row (contract or statistical) and its band
  5. the expected renewals and their band, row by row
  6. the simulation window
  7. the distribution of the timevarying marks still to come: the forecast with them
  8. the totals by month and by year
  9. check the forecast                                              checks 1-6
 10. write the forecast, the months, the summary and the maturation  checks 7-10
 11. count the checks; stop if any failed
 12. show the forecast by month, the answers by year and the maturation, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every future row has a rate, an uplift and an expected value
   2. every rate and every band is inside [0, 1] and the band contains the rate
   3. no closed row is forecast; every future row is
   4. the pipeline of the future rows is conserved (Σ USD due)
   5. the totals are the sum of the rows
   6. the distribution moves units, it does not create them (moved ≤ units due, l in [0, 1])
   7-10. tables sff_forecast, sff_forecast_mes, sff_resumen_negocio, sff_forecast_maduracion written
        and read back

Output: (the forecast row by row, by month, by year) · tables sff_forecast, sff_forecast_mes,
sff_resumen_negocio.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ACTIVE_FLAG_VALUES, Config, discount_bucket_labels, is_one_year, join_columns, parse_month
from step_14_backtest import band_of_horizon
from prediction import band_quantiles, index_history, predict_composition, rate_band, shifted_rate
from techniques import inverse_logit, logit, predict_logit
from vocabulario import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, GATE_LEVEL, PATH_CONTRACT, PATH_STATISTICAL,
                         PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_COLUMN,
                         RATE_FROM_CELL, RATE_FROM_GLOBAL, RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN,
                         TABLE_BUSINESS_SUMMARY, TABLE_FORECAST, TABLE_FORECAST_MATURATION, TABLE_FORECAST_MONTH,
                         TRUTH_ROLES, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN)


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
                "the distribution of the timevarying marks still to come: the forecast with them",
                "the totals by month and by year, by origin of the pipeline",
                "check the forecast (checks 1-6)",
                "write the forecast, the months, the summary and the maturation (checks 7-10)",
                "count the checks; stop if any failed",
                "show the forecast by month, the answers by year and the maturation, as tables"]
STEP_OUTPUT = "every future row with rate, uplift, expected USD and bands · totals by month and year · three tables"

PERCENTAGE_POINTS = 100


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

    # the marks still to come of the extract rows are distributed first (action 7): what is expected to renew
    # in the window, the pipeline of the projected rows, is the forecast with them
    future = distribute_marks(future, fine_table, configuration)

    # [6] the simulation window: renewals and acquisitions that fall due 12 months later; the
    #     extract rows of the target months that the simulation replaces
    extension = extend_horizon(fine_table, future, context, configuration)
    if len(extension):
        extension = distribute_marks(extension, fine_table, configuration)      # their marks are unknown, not absent
        future = pd.concat([future, extension], ignore_index=True)
    window = configuration.simulation_window
    configuration.log_action(STEP_LABEL, 6, f"simulation window {window[0] if window else '—'}..{window[-1] if window else '—'} → "
                                            f"due {window[0] + 12 if window else '—'}..{window[-1] + 12 if window else '—'}: "
                                            f"{extension[PIPELINE_ORIGIN_COLUMN].value_counts().to_dict() if len(extension) else {}} rows · "
                                            f"${extension[configuration.pipeline_usd_col].sum() if len(extension) else 0:,.0f} due · "
                                            f"${extension['esperado_usd'].sum() if len(extension) else 0:,.0f} expected")

    # the confidence of every row: how sure the forecast is of its rate
    future["confidence"] = confidence_of_rows(future, context, configuration)
    configuration.log_action(STEP_LABEL, 6, "confidence (USD expected): " + " · ".join(
        f"{label} {share:.0%}" for label, share in
        (future.groupby("confidence")["esperado_usd"].sum() / max(future["esperado_usd"].sum(), 1e-9)).items()))

    # [7] the distribution of the timevarying marks (done above: the extract before the projection, the projected
    #     and simulated rows after it): the table by month of the whole future
    maturation = distribution_table(future, configuration)
    if maturation is None:
        configuration.log_action(STEP_LABEL, 7, "the marks are not distributed (distribute_marks off, or no negative mark)")
    else:
        configuration.log_action(STEP_LABEL, 7, f"marks distributed ({', '.join(negative_marks(configuration))}): "
                                                f"{future['maduracion_unidades_migran'].sum():,.0f} units move to a marked "
                                                f"combination before falling due · {future['maduracion_delta_usd'].sum():+,.0f} "
                                                f"USD in the forecast")

    # [8] the totals
    by_month, by_year = totals(future, fine_table, configuration)
    configuration.log_action(STEP_LABEL, 8, f"{len(by_month)} months · {len(by_year)} years")

    # [9] the checks
    configuration.log_action(STEP_LABEL, 9, "checking the forecast")
    check_forecast(future, future, fine_table, by_month, configuration, check_log)
    after_renewal = dict(configuration.dims_after_renewal)
    if configuration.acquisition_column and configuration.renewed_acquisition_value is not None:
        after_renewal[configuration.acquisition_column] = configuration.renewed_acquisition_value
    history_values = {column_name: set(fine_table[column_name].dropna().astype(str).unique())
                      for column_name in after_renewal if column_name in fine_table.columns}
    unseen = {column_name: value for column_name, value in after_renewal.items()
              if column_name in history_values and str(value) not in history_values[column_name]}
    configuration.log_check(STEP_LABEL, check_log,
                            "every value a projected renewal takes after renewing exists in the history "
                            "(its retention series can be found)",
                            not unseen,
                            failure_detail="; ".join(f"{column_name} = {value!r} is not in the data (values: "
                                                     f"{sorted(history_values[column_name])[:8]})"
                                                     for column_name, value in unseen.items()),
                            blocking=False, context=f"{after_renewal}" if after_renewal else "nothing declared")
    if maturation is not None:
        moved_too_much = future[future["maduracion_unidades_migran"] > future[configuration.pipeline_units_col] + 1e-9]
        configuration.log_check(STEP_LABEL, check_log,
                                "the distribution moves units, it does not create them (moved ≤ units due, l in [0, 1])",
                                moved_too_much.empty and future["maduracion_l"].between(0, 1).all(),
                                failure_detail=f"{len(moved_too_much)} rows move more units than they have")

    # [10] the tables
    configuration.log_action(STEP_LABEL, 10, "writing the forecast, the months, the summary and the maturation")
    forecast_columns = ([period_column, "h", PIPELINE_ORIGIN_COLUMN, SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
                         COMPOSITION_ID_COLUMN, configuration.pipeline_units_col, configuration.pipeline_usd_col, "origen_tasa",
                         "tecnica", "tasa", "tasa_baja", "tasa_alta", "confidence", "via_uplift", "uplift", "uplift_bajo", "uplift_alto",
                         "esperado_unidades", "esperado_usd", "esperado_usd_bajo", "esperado_usd_alto",
                         "esperado_unidades_base", "esperado_usd_base",
                         "maduracion_l", "maduracion_unidades_migran", "maduracion_delta_unidades", "maduracion_delta_usd"]
                        + configuration.rate_series_columns + configuration.extra_revalorizacion
                        + ([configuration.discount_value_column, configuration.discount_bucket_column]
                           if configuration.discount_value_column else []))
    forecast_columns = list(dict.fromkeys(forecast_columns))
    configuration.write_table(STEP_LABEL, check_log, future[forecast_columns], TABLE_FORECAST)
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_FORECAST_MONTH)
    configuration.write_table(STEP_LABEL, check_log, by_year, TABLE_BUSINESS_SUMMARY)
    if maturation is not None:
        configuration.write_table(STEP_LABEL, check_log, maturation, TABLE_FORECAST_MATURATION)

    # [11] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 11, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [12] the forecast by month and the answers by year; the maturation next to it
    configuration.log_action(STEP_LABEL, 12, "the forecast by month (banda lineal: every error in the same direction, the "
                                             "worst case; banda cuadratura: independent errors):")
    configuration.show_table(by_month)
    configuration.logger.doc(f"[{STEP_LABEL}] by year: renewed in the closed months + expected in the future ones (by origin "
                             f"of the pipeline):")
    configuration.show_table(by_year)
    if maturation is not None:
        show_distribution(maturation, configuration)
    timevarying_diagnostic(future, fine_table, configuration)
    result = dict(forecast=future, by_month=by_month, by_year=by_year)
    if maturation is not None:
        result["maturation"] = maturation
    return result


# ─── the maturation of a timevarying mark (action 7) ─────────────────────────────


CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW = "high", "medium", "low"


def negative_marks(configuration: Config) -> list:
    """The timevarying marks that lower the rate (structural_timevarying_dims with sign "negative")."""
    return [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "negative"]


def mark_combination(frame: pd.DataFrame, marks: list) -> pd.Series:
    """The combination of negative marks of every row: "softcancel+dormant", … or "neutral" when none is on."""
    labels = pd.Series("", index=frame.index)
    for mark in marks:
        active = frame[mark].isin(ACTIVE_FLAG_VALUES) if mark in frame.columns else pd.Series(False, index=frame.index)
        labels = labels.where(~active, labels + np.where(labels == "", "", "+") + mark)
    return labels.where(labels != "", "neutral")


def distribute_marks(future: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config):
    """The distribution of the timevarying marks (action 7): the future rows carry today's marks, and the marks
    still to come are distributed in the proportion seen in the history. The result IS the forecast.

    INPUT:   the future rows, already predicted (tasa, uplift, esperado_*), and the fine table (closed months).
    OUTPUT:  the future rows with esperado_* adjusted (the base kept in esperado_unidades_base and
             esperado_usd_base) and the maduracion_* columns; the table by month. (future, None) when
             distribute_marks is off or there is no negative mark.
    RULES:   the marks are the negative structural_timevarying_dims (softcancel, dormant, not_installed…),
             treated as COMBINATIONS (neutral, dormant, softcancel, dormant+softcancel…): they overlap and every
             combination renews its own way. The extract records the LAST state of a mark, so a mark only
             grows: units move from neutral to marked, never back.
             For every group of the future (every dim but the timevarying ones, the forecast series without
             its marks) and month:
               the final mix  the share of every combination in the same calendar month of the last
                              maturation_history_months closed months (there the marks are final): of the
                              group; with fewer than maturation_min_units, of its mandatory cell; with fewer
                              still, of the whole portfolio
               the gap        final share − today's share of every marked combination, when positive
               the move       the neutral units of the group move to every marked combination in proportion
                              to its gap (never more than the neutral units there are)
             The units moved renew at the rate of their destination combination in the history (same levels;
             units and USD): delta units = moved × (rate of the combination − row rate); delta USD = USD moved ×
             (USD rate of the combination − row rate × row uplift). The bands move by the same delta.
             The projected and simulated rows (all neutral) are distributed too: their marks are unknown,
             not absent. No backtest: the technique is the one of the acquisition simulation, the same
             calendar month of the history.
    """
    period = configuration.period_col
    units_column, usd_column = configuration.pipeline_units_col, configuration.pipeline_usd_col
    future["esperado_unidades_base"] = future["esperado_unidades"]
    future["esperado_usd_base"] = future["esperado_usd"]
    for column_name, empty in (("maduracion_l", 0.0), ("maduracion_unidades_migran", 0.0), ("maduracion_usd_migran", 0.0),
                               ("maduracion_usd_sale", 0.0), ("maduracion_usd_entra", 0.0),
                               ("maduracion_delta_unidades", 0.0), ("maduracion_delta_usd", 0.0)):
        future[column_name] = empty
    marks = negative_marks(configuration)
    for mark in marks:
        future[f"maduracion_migran_{mark}"] = 0.0
    if not configuration.distribute_marks or not marks or future.empty:
        return future
    minimum = configuration.maturation_min_units

    # the history: the last closed months, where the marks are final
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (fine_table[units_column] > 0)]
    history_months = sorted(closed[period].unique())[-configuration.maturation_history_months:]
    history = closed[closed[period].isin(history_months)].copy()
    group_dims = [column_name for column_name in configuration.rate_series_columns
                  if column_name not in configuration.structural_timevarying_dims]
    history["_combinacion"] = mark_combination(history, marks)
    history["_mes"] = history[period].map(lambda month: month.month)
    history["_grupo"] = join_columns(history, group_dims)
    history["_celda"] = join_columns(history, configuration.business_mandatory_dims)
    history["_cartera"] = "cartera"
    history["_renovadas"] = history[configuration.renewed_units_col].fillna(0.0)
    history["_renovado_usd"] = history[configuration.renewed_usd_col].fillna(0.0)
    levels = ["_grupo", "_celda", "_cartera"]
    mix_units, mix_total, combination_sums = {}, {}, {}
    for level in levels:
        mix_units[level] = history.groupby([level, "_mes", "_combinacion"])[units_column].sum().to_dict()
        mix_total[level] = history.groupby([level, "_mes"])[units_column].sum().to_dict()
        combination_sums[level] = (history.groupby([level, "_combinacion"])
                                   [[units_column, "_renovadas", usd_column, "_renovado_usd"]].sum()
                                   .apply(tuple, axis=1).to_dict())
    marked_combinations = sorted(set(history["_combinacion"]) - {"neutral"})

    def combination_rates(keys: dict, combination: str) -> tuple:
        """The units and USD rates of a combination: of the group, its cell or the portfolio (≥ minimum units)."""
        for level in levels:
            sums = combination_sums[level].get((keys[level], combination))
            if sums and sums[0] >= minimum or (level == "_cartera" and sums and sums[0] > 0):
                return sums[1] / sums[0], sums[3] / sums[2] if sums[2] > 0 else np.nan
        return np.nan, np.nan

    # today: the mix of every group and month of the future
    rows = future[future[units_column] > 0].copy()
    rows["_combinacion"] = mark_combination(rows, marks)
    rows["_grupo"] = join_columns(rows, group_dims)
    rows["_celda"] = join_columns(rows, configuration.business_mandatory_dims)
    today_units = rows.groupby(["_grupo", period, "_combinacion"])[units_column].sum()
    today_total = rows.groupby(["_grupo", period])[units_column].sum()
    cell_of_group = rows.drop_duplicates("_grupo").set_index("_grupo")["_celda"].to_dict()
    today_by_pair = {}
    for (group, month, combination), units in today_units.items():
        today_by_pair.setdefault((group, month), {})[combination] = units / today_total[(group, month)]

    # the move of every group and month: the share of its neutral units going to every marked combination
    move_of_pair, moved_by_mark = {}, {}
    for (group, month), today in today_by_pair.items():
        neutral_today = today.get("neutral", 0.0)
        if neutral_today <= 0:
            continue
        keys = {"_grupo": group, "_celda": cell_of_group[group], "_cartera": "cartera"}
        level = next((candidate for candidate in levels
                      if mix_total[candidate].get((keys[candidate], month.month), 0.0) >= minimum), "_cartera")
        total = mix_total[level].get((keys[level], month.month), 0.0)
        if total <= 0:
            continue
        gaps = {combination: max(mix_units[level].get((keys[level], month.month, combination), 0.0) / total
                                 - today.get(combination, 0.0), 0.0) for combination in marked_combinations}
        gap_total = sum(gaps.values())
        if gap_total <= 0:
            continue
        scale = min(1.0, neutral_today / gap_total)
        fractions = {combination: gap * scale / neutral_today for combination, gap in gaps.items() if gap > 0}
        moved_share = sum(fractions.values())
        rate_units = rate_usd = 0.0
        for combination, fraction in fractions.items():
            combination_units_rate, combination_usd_rate = combination_rates(keys, combination)
            rate_units += fraction * combination_units_rate
            rate_usd += fraction * combination_usd_rate
        move_of_pair[(group, month)] = (moved_share, rate_units / moved_share, rate_usd / moved_share)
        moved_by_mark[(group, month)] = {mark: sum(fraction for combination, fraction in fractions.items()
                                                   if mark in combination.split("+")) for mark in marks}

    # the neutral rows move their share: the forecast becomes the forecast with the marks distributed
    neutral_rows = rows[rows["_combinacion"] == "neutral"]
    pair_keys = list(zip(neutral_rows["_grupo"], neutral_rows[period]))
    moves = [move_of_pair.get(key, (0.0, np.nan, np.nan)) for key in pair_keys]
    share = np.array([move[0] for move in moves], dtype=float)
    destination_rate_units = np.nan_to_num(np.array([move[1] for move in moves], dtype=float))
    destination_rate_usd = np.nan_to_num(np.array([move[2] for move in moves], dtype=float))
    units = neutral_rows[units_column].to_numpy(dtype=float)
    usd = neutral_rows[usd_column].to_numpy(dtype=float)
    row_rate = neutral_rows["tasa"].to_numpy(dtype=float)
    row_uplift = neutral_rows["uplift"].to_numpy(dtype=float)
    index = neutral_rows.index
    future.loc[index, "maduracion_l"] = share
    future.loc[index, "maduracion_unidades_migran"] = units * share
    future.loc[index, "maduracion_usd_migran"] = usd * share
    future.loc[index, "maduracion_usd_sale"] = usd * share * row_rate * row_uplift
    future.loc[index, "maduracion_usd_entra"] = usd * share * destination_rate_usd
    future.loc[index, "maduracion_delta_unidades"] = units * share * (destination_rate_units - row_rate)
    future.loc[index, "maduracion_delta_usd"] = usd * share * destination_rate_usd - usd * share * row_rate * row_uplift
    for mark in marks:
        future.loc[index, f"maduracion_migran_{mark}"] = units * np.array(
            [moved_by_mark.get(key, {}).get(mark, 0.0) for key in pair_keys], dtype=float)
    future["esperado_unidades"] = future["esperado_unidades_base"] + future["maduracion_delta_unidades"]
    future["esperado_usd"] = future["esperado_usd_base"] + future["maduracion_delta_usd"]
    future["esperado_usd_bajo"] = (future["esperado_usd_bajo"] + future["maduracion_delta_usd"]).clip(lower=0.0)
    future["esperado_usd_alto"] = (future["esperado_usd_alto"] + future["maduracion_delta_usd"]).clip(lower=0.0)
    return future


def distribution_table(future: pd.DataFrame, configuration: Config):
    """The table by month of the distribution of the marks, over the whole future (extract and extension):
    every mark today and expected at due date, the base and the final forecast, the two vessels.
    None when the marks are not distributed."""
    marks = negative_marks(configuration)
    if not configuration.distribute_marks or not marks:
        return None
    period = configuration.period_col
    units_column = configuration.pipeline_units_col
    future = future.copy()
    future["_combinacion"] = mark_combination(future, marks)
    grouped = future.groupby(period)
    table = pd.DataFrame({"meses_vista": grouped["h"].min(), "vencen_unidades": grouped[units_column].sum()})
    for mark in marks:
        marked_today = future[mark].isin(ACTIVE_FLAG_VALUES) if mark in future.columns else pd.Series(False, index=future.index)
        today_units_mark = future[units_column].where(marked_today, 0.0).groupby(future[period]).sum()
        table[f"{mark}_hoy"] = today_units_mark / table["vencen_unidades"]
        table[f"{mark}_esperado"] = (today_units_mark + grouped[f"maduracion_migran_{mark}"].sum()) / table["vencen_unidades"]
    neutral_today_units = future[units_column].where(future["_combinacion"] == "neutral", 0.0).groupby(future[period]).sum()
    table["neutro_hoy"] = neutral_today_units / table["vencen_unidades"]
    table["neutro_esperado"] = (neutral_today_units - grouped["maduracion_unidades_migran"].sum()) / table["vencen_unidades"]
    table["unidades_migran"] = grouped["maduracion_unidades_migran"].sum()
    table["tasa_base"] = grouped["esperado_unidades_base"].sum() / table["vencen_unidades"]
    table["tasa_final"] = grouped["esperado_unidades"].sum() / table["vencen_unidades"]
    table["renovado_usd_base"] = grouped["esperado_usd_base"].sum()
    table["ajuste_usd"] = grouped["maduracion_delta_usd"].sum()
    table["renovado_usd_final"] = grouped["esperado_usd"].sum()
    table["ajuste_pct"] = table["ajuste_usd"] / table["renovado_usd_base"].where(table["renovado_usd_base"] > 0)
    is_neutral = future["_combinacion"] == "neutral"
    table["sin_marca_usd_base"] = future["esperado_usd_base"].where(is_neutral, 0.0).groupby(future[period]).sum()
    table["sin_marca_usd_final"] = table["sin_marca_usd_base"] - grouped["maduracion_usd_sale"].sum()
    table["marcada_usd_base"] = future["esperado_usd_base"].where(~is_neutral, 0.0).groupby(future[period]).sum()
    table["marcada_usd_final"] = table["marcada_usd_base"] + grouped["maduracion_usd_entra"].sum()
    return table.reset_index()


def show_distribution(table: pd.DataFrame, configuration: Config) -> None:
    """The distribution of the marks on screen: month by month and by year, with the two vessels."""
    doc = configuration.logger.doc
    marks = negative_marks(configuration)
    share_columns = [f"{mark}_{moment}" for mark in marks for moment in ("hoy", "esperado")] + ["neutro_hoy", "neutro_esperado"]
    shown = table[[configuration.period_col, "meses_vista"] + share_columns
                  + ["unidades_migran", "tasa_base", "tasa_final", "renovado_usd_base", "ajuste_usd", "renovado_usd_final",
                     "ajuste_pct"]].copy()
    for share_column in share_columns + ["tasa_base", "tasa_final", "ajuste_pct"]:
        shown[share_column] = (100 * shown[share_column]).round(1)
    for money_column in ["unidades_migran", "renovado_usd_base", "ajuste_usd", "renovado_usd_final"]:
        shown[money_column] = shown[money_column].round(0)
    doc(f"[{STEP_LABEL}] ══════════ REPARTO DE MARCAS ({', '.join(marks)}): EL FORECAST FINAL ══════════")
    doc(f"[{STEP_LABEL}] mes a mes, en %: cada marca HOY (foto del extracto; extendido neutro) y ESPERADA al vencer "
        f"(su proporción histórica del mismo mes); tasa_base = con las marcas de hoy, tasa_final = el forecast")
    configuration.show_table(shown)
    year_of = table[configuration.period_col].map(lambda month: month.year)
    by_year = table.groupby(year_of)[["renovado_usd_base", "ajuste_usd", "renovado_usd_final", "sin_marca_usd_base",
                                      "sin_marca_usd_final", "marcada_usd_base", "marcada_usd_final"]].sum()
    doc(f"[{STEP_LABEL}] por año: el forecast con las marcas de hoy, el ajuste y el forecast final, y los dos vasos "
        f"(sin marca pierde, con marca gana, la pipeline es la misma):")
    configuration.show_table(by_year.round(0).reset_index().rename(columns={configuration.period_col: "ano"}))
    for year, row in by_year.iterrows():
        if row["renovado_usd_base"] <= 0:
            continue
        doc(f"[{STEP_LABEL}] VALORACIÓN {year}: con las marcas de hoy ${row['renovado_usd_base']:,.0f}; con las marcas "
            f"repartidas en su proporción histórica (el forecast final) ${row['renovado_usd_final']:,.0f} "
            f"({row['ajuste_usd'] / row['renovado_usd_base']:+.1%}): sin marca ${row['sin_marca_usd_base'] - row['sin_marca_usd_final']:,.0f} "
            f"menos, con marca ${row['marcada_usd_final'] - row['marcada_usd_base']:,.0f} más")
    doc(f"[{STEP_LABEL}] ══════════ FIN DEL REPARTO DE MARCAS ══════════")


def timevarying_diagnostic(future: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> None:
    """On screen only, to decide how the forecast should carry the marks that are not final yet (softcancel,
    dormant, not_installed): for every future month, each mark today against the same calendar month in the
    closed history (where the marks are final); what every combination of marks renews in the history; and
    what the rate of every future month would be with today's mix of marks and with the historical final mix.
    Every figure is in units due; the rates of the last table are the HISTORICAL rates of each combination,
    not the forecast's: they measure the size of the effect, not the adjusted forecast."""
    doc = configuration.logger.doc
    marks = negative_marks(configuration)
    if not marks:
        return
    period = configuration.period_col
    units_column, usd_column = configuration.pipeline_units_col, configuration.pipeline_usd_col
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (fine_table[units_column] > 0)]
    history_months = sorted(closed[period].unique())[-configuration.maturation_history_months:]
    history = closed[closed[period].isin(history_months)].copy()
    history["_combinacion"] = mark_combination(history, marks)
    history["_mes"] = history[period].map(lambda month: month.month)
    future_rows = future[future[units_column] > 0].copy()
    future_rows["_combinacion"] = mark_combination(future_rows, marks)
    future_rows["_mes"] = future_rows[period].map(lambda month: month.month)

    doc(f"[{STEP_LABEL}] ══════════ MARCAS TIMEVARYING: DIAGNÓSTICO ({', '.join(marks)}) ══════════")

    # A · every mark, month by month: today (future rows) against final (the same calendar month, closed)
    rows = []
    for month, block in future_rows.groupby(period):
        same_month = history[history["_mes"] == month.month]
        row = {period: month, "meses_vista": int(block["h"].min()),
               "vencen_usd": round(float(block[usd_column].sum())),
               "%_del_extracto": round(100 * float(block.loc[block[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, units_column].sum()
                                                   / block[units_column].sum()), 1)}
        for mark in marks:
            today = block[mark].isin(ACTIVE_FLAG_VALUES) if mark in block.columns else pd.Series(False, index=block.index)
            final = same_month[mark].isin(ACTIVE_FLAG_VALUES)
            row[f"%{mark}_hoy"] = round(100 * float(block.loc[today, units_column].sum() / block[units_column].sum()), 1)
            row[f"%{mark}_final"] = (round(100 * float(same_month.loc[final, units_column].sum() / same_month[units_column].sum()), 1)
                                     if len(same_month) else np.nan)
        neutral_today = block["_combinacion"] == "neutral"
        row["%neutro_hoy"] = round(100 * float(block.loc[neutral_today, units_column].sum() / block[units_column].sum()), 1)
        row["%neutro_final"] = (round(100 * float(same_month.loc[same_month["_combinacion"] == "neutral", units_column].sum()
                                                  / same_month[units_column].sum()), 1) if len(same_month) else np.nan)
        rows.append(row)
    doc(f"[{STEP_LABEL}] A · cada marca, mes a mes: % de las unidades que vencen marcadas HOY (filas futuras: extracto con "
        f"la foto de hoy, horizonte extendido neutro) frente al % FINAL del mismo mes de calendario en los últimos "
        f"{len(history_months)} meses cerrados (la marca ya definitiva). %_del_extracto: el resto es proyectada/simulada")
    configuration.show_table(pd.DataFrame(rows))

    # B · what every combination of marks renews in the history: the cost of a mark
    combinations = (history.groupby("_combinacion")
                    .agg(vencen_unidades=(units_column, "sum"), renovadas=(configuration.renewed_units_col, "sum"),
                         vencen_usd=(usd_column, "sum"), renovado_usd=(configuration.renewed_usd_col, "sum")))
    combinations["%_unidades"] = (100 * combinations["vencen_unidades"] / combinations["vencen_unidades"].sum()).round(1)
    combinations["tasa_unidades"] = (100 * combinations["renovadas"] / combinations["vencen_unidades"]).round(1)
    combinations["tasa_usd"] = (100 * combinations["renovado_usd"] / combinations["vencen_usd"]).round(1)
    combinations = combinations.sort_values("vencen_unidades", ascending=False)
    doc(f"[{STEP_LABEL}] B · lo que renueva cada combinación de marcas en los últimos {len(history_months)} meses cerrados "
        f"(marca final): el coste de cada marca frente a neutral")
    configuration.show_table(combinations[["%_unidades", "tasa_unidades", "tasa_usd"]].reset_index()
                             .rename(columns={"_combinacion": "combinacion"}))

    # C · the size of the effect: the rate of every future month with today's mix and with the final mix
    rate_units = (combinations["renovadas"] / combinations["vencen_unidades"]).to_dict()
    rate_usd = (combinations["renovado_usd"] / combinations["vencen_usd"]).to_dict()
    neutral_units, neutral_usd = rate_units.get("neutral", np.nan), rate_usd.get("neutral", np.nan)
    effect_rows = []
    for month, block in future_rows.groupby(period):
        same_month = history[history["_mes"] == month.month]
        today_mix = block.groupby("_combinacion")[units_column].sum() / block[units_column].sum()
        final_mix = (same_month.groupby("_combinacion")[units_column].sum() / same_month[units_column].sum()
                     if len(same_month) else today_mix)
        with_today_units = sum(share * rate_units.get(label, neutral_units) for label, share in today_mix.items())
        with_final_units = sum(share * rate_units.get(label, neutral_units) for label, share in final_mix.items())
        with_today_usd = sum(share * rate_usd.get(label, neutral_usd) for label, share in today_mix.items())
        with_final_usd = sum(share * rate_usd.get(label, neutral_usd) for label, share in final_mix.items())
        effect_rows.append({period: month, "año": month.year, "vencen_usd": float(block[usd_column].sum()),
                            "tasa_u_mezcla_hoy": 100 * with_today_units, "tasa_u_mezcla_final": 100 * with_final_units,
                            "tasa_usd_mezcla_hoy": 100 * with_today_usd, "tasa_usd_mezcla_final": 100 * with_final_usd})
    effect = pd.DataFrame(effect_rows)
    effect["dif_pp_usd"] = effect["tasa_usd_mezcla_final"] - effect["tasa_usd_mezcla_hoy"]
    effect["efecto_usd"] = effect["vencen_usd"] * effect["dif_pp_usd"] / 100
    doc(f"[{STEP_LABEL}] C · el tamaño del efecto, mes a mes: la tasa con la mezcla de marcas de HOY y con la mezcla FINAL "
        f"del mismo mes histórico, ambas con las tasas históricas de cada combinación (tabla B). efecto_usd = lo que "
        f"cambiaría lo renovado del mes si las marcas llegan a su proporción histórica")
    configuration.show_table(effect.drop(columns=["año"]).round({"vencen_usd": 0, "tasa_u_mezcla_hoy": 1,
                                                                 "tasa_u_mezcla_final": 1, "tasa_usd_mezcla_hoy": 1,
                                                                 "tasa_usd_mezcla_final": 1, "dif_pp_usd": 1, "efecto_usd": 0}))
    by_year = effect.groupby("año")[["vencen_usd", "efecto_usd"]].sum()
    for year, row in by_year.iterrows():
        doc(f"[{STEP_LABEL}] VALORACIÓN {year}: si las marcas llegan a su proporción histórica, lo renovado cambia del orden de "
            f"${row['efecto_usd']:,.0f} sobre ${row['vencen_usd']:,.0f} que vencen ({row['efecto_usd'] / row['vencen_usd']:+.1%} "
            f"de lo que vence)")
    doc(f"[{STEP_LABEL}] ══════════ FIN DEL DIAGNÓSTICO DE MARCAS ══════════")


def confidence_of_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> np.ndarray:
    """The confidence of every future row:
      high    the half-width of its rate band ≤ confidence_high_band_pp (5), its composition judged by
              the backtest, and the exam error of the chosen technique for its horizon band ≤
              confidence_high_exam_pp (5)
      medium  its composition judged and the half-width of its band ≤ confidence_medium_band_pp (10)
      low     the rest: a wider band, a composition below the support floor (not judged), or a rate
              from the mandatory cell or the global rate (no composition)"""
    band_half_width_pp = 100 * (rows["tasa_alta"] - rows["tasa_baja"]) / 2
    judged_ids = set(context["pool_reference"].loc[context["pool_reference"]["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN])
    judged = rows[COMPOSITION_ID_COLUMN].isin(judged_ids) & (rows["origen_tasa"] == RATE_FROM_POOL)
    exam_of_key = context["backtest"]["exam_by_pool"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["elegida_err_pp_medio"].to_dict()
    horizon_band = rows["h"].map(lambda horizon: band_of_horizon(int(horizon), configuration.horizon_bands))
    exam_error = pd.Series([exam_of_key.get(pair, np.nan) for pair in zip(rows[COMPOSITION_ID_COLUMN], horizon_band)],
                           index=rows.index, dtype=float)
    high = (judged & (band_half_width_pp <= configuration.confidence_high_band_pp)
            & (exam_error <= configuration.confidence_high_exam_pp))
    medium = judged & (band_half_width_pp <= configuration.confidence_medium_band_pp)
    return np.select([high, medium], [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM], default=CONFIDENCE_LOW)


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
             proyectada: 1-year rows of the window, expected renewals, same dims, discount 0; an
             acquisition becomes renewed_acquisition_value (it renewed: it is not an acquisition any more);
             the dims of dims_after_renewal take their declared value (prev_OperationGroup → the renewal);
             the timevarying marks are neutral (nothing is known yet of the renewal's marks).
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
    projected[period] = renewing[period].to_numpy() + configuration.one_year_term_months
    projected[configuration.pipeline_units_col] = renewing["esperado_unidades"].to_numpy()
    projected[configuration.pipeline_usd_col] = renewing["esperado_usd"].to_numpy()
    if configuration.discount_value_column:
        projected[configuration.discount_value_column] = configuration.renewal_reentry_discount
    # an acquisition that renews is no longer an acquisition: its next due date is a retention (and is predicted
    # with the retention series, not with the first-renewal rate of the acquisitions)
    acquisition = configuration.acquisition_column
    if acquisition and acquisition in projected.columns and configuration.renewed_acquisition_value is not None:
        was_acquisition = projected[acquisition].isin(configuration.acquisition_values)
        projected[acquisition] = projected[acquisition].astype(object)
        projected.loc[was_acquisition, acquisition] = configuration.renewed_acquisition_value
    # the other dims a renewal changes (the previous operation is now a renewal), as the Config declares them
    for column_name, value_after_renewal in configuration.dims_after_renewal.items():
        if column_name in projected.columns:
            projected[column_name] = projected[column_name].astype(object)
            projected[column_name] = value_after_renewal
    # the timevarying marks of a renewal are neutral: nothing is known yet of its dormant, softcancel…
    for column_name in configuration.structural_timevarying_dims:
        if column_name in projected.columns:
            projected[column_name] = inactive_value(fine_table[column_name])
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
    acquired["_mes_evento"] = acquired[period] - configuration.one_year_term_months
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
        before = block[block["_mes_evento"].isin([month - configuration.one_year_term_months for month in level_months])].groupby("_grupo")["unidades"].sum()
        level = (recent / before.replace(0, np.nan)).dropna()
        total_level = recent.sum() / before.sum() if before.sum() > 0 else np.nan
        auv_window = block[block["_mes_evento"] >= auv_start]
        auv = auv_window.groupby("_grupo")["valor"].sum() / auv_window.groupby("_grupo")["unidades"].sum().replace(0, np.nan)
        total_auv = auv_window["valor"].sum() / auv_window["unidades"].sum() if auv_window["unidades"].sum() > 0 else np.nan
        share = auv_window.groupby("_grupo")["unidades"].sum() / auv_window["unidades"].sum() if auv_window["unidades"].sum() > 0 else pd.Series(dtype=float)
        total_by_month = block.groupby("_mes_evento")["unidades"].sum()
        for month in window:
            for group in share.index:
                same_month = by_group.get((group, month - configuration.one_year_term_months), np.nan)
                group_level = level.get(group, np.nan)
                by_share = not (np.isfinite(same_month) and np.isfinite(group_level))
                units = (total_by_month.get(month - configuration.one_year_term_months, np.nan) * total_level * share[group]) if by_share else same_month * group_level
                if not np.isfinite(units) or units <= 0:
                    continue
                unit_value = auv.get(group, np.nan)
                unit_value = unit_value if np.isfinite(unit_value) else total_auv
                row = dims_of_group.loc[group].to_dict()
                row.update({configuration.acquisition_column: acquisition_value, period: month + configuration.one_year_term_months,
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

def pool_predictions(pool_series: pd.DataFrame, backtest: dict, horizons: list, configuration: Config) -> pd.DataFrame:
    """The rate of every estimation id at every future horizon, with the technique of its band,
    learning from every closed month of the id."""
    decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna() & (pool_series["vencen"] > 0)]
    rows = []
    history_of_composition = index_history(truth, COMPOSITION_ID_COLUMN, configuration.period_col, [RATE_COLUMN])
    for composition_id, entry in history_of_composition.items():          # in the order of groupby (sorted ids)
        history = logit(entry[RATE_COLUMN])
        calendar_months = entry["month"]
        for horizon in horizons:
            band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
            technique = decision.get((composition_id, band_name), configuration.challenger_technique)
            technique, rate = predict_composition(history, calendar_months, technique, int(horizon),
                                                  configuration.challenger_technique)
            rows.append((composition_id, int(horizon), technique, rate))
    return pd.DataFrame(rows, columns=[COMPOSITION_ID_COLUMN, "h", "tecnica", "tasa_pool_h"])


def rate_of_rows(future: pd.DataFrame, series_estimate: pd.DataFrame, pool_rates: pd.DataFrame, pool_reference: pd.DataFrame,
                 rated_units: pd.DataFrame, backtest: dict, forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The rate of every future row: its group's prediction (moved toward its credibility reference),
    mandatory cell, or global; and its band."""
    estimate = series_estimate[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "z", "group_rate", "ref_rate"]]
    future = future.drop(columns=[column for column in (COMPOSITION_ID_COLUMN, "z", "group_rate", "ref_rate", "tecnica",
                                                        "tasa_pool_h", "tasa_pool", "tasa", "origen_tasa")
                                  if column in future.columns])
    future = future.merge(estimate, on=SERIES_ID_COLUMN, how="left")
    future = future.merge(pool_rates, on=[COMPOSITION_ID_COLUMN, "h"], how="left")
    future = future.merge(pool_reference[[COMPOSITION_ID_COLUMN, "tasa_pool"]], on=COMPOSITION_ID_COLUMN, how="left")

    # the composition's predicted rate, moved toward its credibility reference (prediction.py: the same
    # computation as the exam and the audit)
    pool_rate = future["tasa_pool_h"]
    rate, shift = shifted_rate(pool_rate, future["z"], future["ref_rate"], future["group_rate"],
                               configuration.apply_credibility_shift)
    future["credibility_shift_logit"] = shift
    future["tasa"] = np.where(pool_rate.notna(), rate, np.nan)
    future["origen_tasa"] = np.where(pool_rate.notna(), RATE_FROM_POOL, None)

    # no pool: the rate of the mandatory cell, then the global rate (closed months)
    truth = rated_units[rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & rated_units[RATE_COLUMN].notna()].copy()
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

    # the band: quantiles of the normalised error × binomial error with the unit's units due (prediction.py)
    unit_units = future[UNIT_ID_COLUMN].map(forecast_units.set_index(UNIT_ID_COLUMN)[configuration.pipeline_units_col])
    unit_units = unit_units.fillna(future[configuration.pipeline_units_col])     # an extended row: its own units
    q_low, q_high = band_quantiles(backtest["bands"], future["tecnica"], future["h"], configuration)
    future["tasa_baja"], future["tasa_alta"] = rate_band(future["tasa"], unit_units, q_low, q_high)
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
        contract = 1 / (1 - future[discount].clip(upper=configuration.contract_discount_cap))
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
                            abs(due_difference) <= configuration.money_tolerance, failure_detail=f"difference ${due_difference:,.2f}",
                            context=f"${all_future.loc[real_rows, configuration.pipeline_usd_col].sum():,.0f} due in the extract")
    total_difference = by_month["esperado_usd"].sum() - future["esperado_usd"].sum()
    configuration.log_check(STEP_LABEL, check_log, "the totals are the sum of the rows", abs(total_difference) <= configuration.money_tolerance,
                            failure_detail=f"difference ${total_difference:,.2f}", context=f"${future['esperado_usd'].sum():,.0f} expected")
