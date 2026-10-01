"""
step_nucleo.py — The core table: the whole story of every row in ONE wide table.

Decided on 27-sep-2026: the prediction lives in one wide table at the fine grain, history
and future, WITHOUT hash keys (in Power BI they cost dictionary and space and say nothing
to a reader). The columns go from left to right in the order of the steps, each with the
prefix of the step that creates it (s00_, s02_, s03_…), so a row reads as its own story.
The dimensions keep their names: they are the slicers.

It can be built after ANY step from 03 on: every block of columns is added only when the
step that produces it has run (its argument is not None), and the legend describes the
columns present. main.py builds it last, with everything. It grows with every step of the
mirror: a new step adds its block here. Today: steps 00-14. The values of steps 13 and
14 belong to the ESTIMATION ID of the series (the pool it takes its rate from); a series
whose pool was not measured (no support, short history) has them empty.

Three levels live in the same table, and the legend says which is which:
  · fila    a value of the row: it adds up (SUM)
  · unidad  a value of the forecast unit (fu_id), repeated in its fine rows: do NOT add it up
  · serie   a value of the rate series (fs_id), repeated in all its rows: do NOT add it up;
            with one series selected, MAX (or MIN) returns it
A RATE is never stored per row: a rate belongs to an aggregation and is the ratio of sums
of the rows selected (in Power BI too). The estimates of the ladder are stored, as
attributes of the series.

GAPS are rows: a month with no expirations inside the history of a series (step 08) gets
a row with every measure at 0 (origen_fila = hueco), so the months of a series are
complete and the sums still reconcile with the extract.

Actions (logged as they are done):
  1. the rows of the extract (the fine table) with their raw and step-02 measures and their ids
  2. the gap rows of step 08
  3. the values of the forecast unit (steps 04, 07), if they have run
  4. the values of the series (steps 06, 08, 11) and of its estimation id (step 14), if they have run
  5. the legend: every column, its step, its level and how to aggregate it
  6. check the core against the extract and the legend              checks 1-4
  7. write the core and its legend                                  checks 5-6
  8. count the checks; stop if any failed
  9. show one series as it looks in the core, and how to read it in Power BI

Checks (logged as they are made, numbered, at the level of their status):
   1. one row per fine row of the extract plus one per gap
   2. the money reconciles with the extract (units and USD due, renewed as the raw had it)
   3. every row has the values of every block present
   4. the legend describes every column of the core
   5-6. tables sff_nucleo and sff_nucleo_leyenda written and read back

Output: the core table and its legend · tables sff_nucleo, sff_nucleo_leyenda.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, ESTIMATION_ID_COLUMN,
                         FINE_ROWS_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_SIMULATED, ROLE_TEST,
                         ROLE_TRAIN, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
                         S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN,
                         S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_CORE,
                         TABLE_CORE_LEGEND, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                         TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL, TS_PROJECTED, TS_REAL, TS_REENTRY,
                         UNIT_ID_COLUMN, UNIVERSE_COLUMN, UPLIFT_CELL_ID_COLUMN, TRUTH_ROLES as TRUTH_ROLES_OF_CORE)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "NU"
STEP_NAME = "CORE TABLE"
STEP_PURPOSE = ("put the whole story of every row in one wide table, left to right in the order of the steps, "
                "with the gaps as explicit rows, so that selecting one forecast series shows everything the "
                "framework knows about it; a legend says how every column is aggregated")
STEP_ACTIONS = ["the rows of the extract (the fine table) with their raw and step-02 measures and their ids",
                "the gap rows of step 08",
                "the values of the forecast unit (steps 04, 07), if they have run",
                "the values of the series (steps 06, 08, 11) and of its estimation id (step 14), if they have run",
                "the legend: every column, its step, its level and how to aggregate it",
                "check the core against the extract and the legend (checks 1-4)",
                "write the core and its legend (checks 5-6)",
                "count the checks; stop if any failed",
                "show one series as it looks in the core, and how to read it in Power BI"]
STEP_OUTPUT = "one row per fine row of the extract plus one per gap · tables sff_nucleo, sff_nucleo_leyenda"

# ─── named constants ─────────────────────────────────────────────────────────────
LEVEL_ROW, LEVEL_UNIT, LEVEL_SERIES = "fila", "unidad", "serie"
AGGREGATE_SUM = "SUM"
AGGREGATE_ATTRIBUTE = "no sumar: atributo repetido; con una serie o unidad seleccionada, MAX"
AGGREGATE_SLICER = "segmentar / filtrar"
MONEY_TOLERANCE = 0.01
MONTHS_SHOWN = 6

# The measure columns of the core: (core name, step, description). The raw measures are as
# the extract had them; the step-02 renewals are the ones the framework learns from.
RAW_MEASURES = [("s00_vencen_unidades", "00", "units falling due, as in the extract"),
                ("s00_vencen_usd", "00", "USD falling due, as in the extract"),
                ("s00_renovadas_unidades", "00", "renewed units as the extract had them (early results of the future included)"),
                ("s00_renovado_usd", "00", "renewed USD as the extract had them (early results of the future included)")]
CALENDAR_MEASURES = [("s02_vencen_unidades", "02", "units due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s02_vencen_usd", "02", "USD due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s02_renovadas_unidades", "02", "renewed units after the calendar: null → 0 in closed months, wiped from the current month on"),
                     ("s02_renovado_usd", "02", "renewed USD after the calendar: null → 0 in closed months, wiped from the current month on")]

# steps 15-17, per FUTURE ROW (a value of the row: the money columns add up)
ROW_VALUES_STEP_17 = [(PIPELINE_ORIGIN_COLUMN, "s17_origen_pipeline", "17", "real (extract) · proyectada · simulada"),
                      ("_vencen_unidades", "s17_vencen_unidades", "17", "units due of the future row: the extract's or the extended horizon's (SUM)"),
                      ("_vencen_usd", "s17_vencen_usd", "17", "USD due of the future row: the extract's or the extended horizon's (SUM)"),
                      ("h", "s17_h", "17", "months from the last closed month"),
                      ("origen_tasa", "s17_origen_tasa", "17", "pool (technique of its estimation id) · celda_mandatory · global"),
                      ("tecnica", "s17_tecnica", "17", "the technique that predicted the rate of this row"),
                      ("tasa", "s17_tasa", "17", "the predicted rate of the row (units)"),
                      ("tasa_baja", "s17_tasa_baja", "17", "low end of the rate band"),
                      ("tasa_alta", "s17_tasa_alta", "17", "high end of the rate band"),
                      ("via_uplift", "s15_via_uplift", "15", "contrato (1/(1 − d)) · estadistica (its cell)"),
                      ("uplift", "s15_uplift", "15", "the uplift applied to the row"),
                      ("esperado_unidades", "s17_esperado_unidades", "17", "expected renewed units (SUM)"),
                      ("esperado_usd", "s17_esperado_usd", "17", "expected renewed USD (SUM)"),
                      ("esperado_usd_bajo", "s17_esperado_usd_bajo", "17", "low end of the row's USD band (SUM = the worst case of a total)"),
                      ("esperado_usd_alto", "s17_esperado_usd_alto", "17", "high end of the row's USD band (SUM = the worst case of a total)")]

# The blocks of values of the forecast unit, of the series and of its estimation id:
# (source column, core name, step, description). A block enters when its step has run.
UNIT_VALUES_STEP_04 = [(FINE_ROWS_COLUMN, "s04_filas_finas", "04", "fine rows added into the unit")]
UNIT_VALUES_STEP_07 = [("moe_pp_max", "s07_moe_pp_max", "07", "worst-case binomial margin of the unit's rate (pp, at z): the yardstick")]
SERIES_VALUES_STEP_06 = [(ROUTE_COLUMN, "s06_ruta", "06", "predecible · solo_historia · solo_futuro"),
                         (COVERAGE_COLUMN, "s06_cobertura", "06", "the roles the series has, in time order"),
                         (UNIVERSE_COLUMN, "s06_universo", "06", "normal · serie_temporal · mixto"),
                         ("usd_por_predecir", "s06_usd_por_predecir", "06", "USD due in the months to predict, whole series")]
SERIES_VALUES_STEP_08 = [("n_propio", "s08_n_propio", "08", "units due in a typical closed month of the series"),
                         ("tasa_propia", "s08_tasa_propia", "08", "own rate of the series: Σ renewed / Σ due over its closed months"),
                         ("error_binomial_pp", "s08_error_binomial_pp", "08", "Wilson half-width of one month at n_propio (pp)"),
                         (SIGN_COLUMN, "s08_signo", "08", "neutro · negativo · positivo · mixto"),
                         ("bajo_suelo", "s08_bajo_suelo", "08", "1 when n_propio is below support_floor")]
SERIES_VALUES_STEP_11 = [(ESTIMATION_ID_COLUMN, "s11_final_group_id", "11", "the final group of the series in the ladder: it lends its rate ('*' = collapsed)"),
                         ("final_step", "s11_final_step", "11", "the pass of the ladder where its group was formed (0 = itself)"),
                         ("group_series", "s11_group_series", "11", "series in its final group"),
                         ("group_support", "s11_group_support", "11", "support of its final group (units due in a typical month)"),
                         ("group_rate", "s11_group_rate", "11", "rate of its final group"),
                         ("credibility_ref_id", "s11_credibility_ref_id", "11", "the wider group its rate is blended with (below own_rate_floor)"),
                         ("ref_support", "s11_ref_support", "11", "support of the credibility reference"),
                         ("ref_rate", "s11_ref_rate", "11", "rate of the credibility reference"),
                         ("alcanzo_suelo", "s11_alcanzo_suelo", "11", "1 when its final group reaches the support floor"),
                         ("k", "s11_k", "11", "Bühlmann k of the reference"),
                         ("z", "s11_z", "11", "credibility of the group's own rate: n / (n + k); 1 without a reference"),
                         ("tasa_estimada", "s11_tasa_estimada", "11", "estimated rate: z·group + (1 − z)·reference (the same for every series of a group)"),
                         ("se_estimacion_pp", "s11_se_estimacion_pp", "11", "error of the estimate (pp)"),
                         ("se_prediccion_pp", "s11_se_prediccion_pp", "11", "error of next month's prediction (pp): never below the series' own noise"),
                         ("nivel_riesgo", "s11_nivel_riesgo", "11", "how the rate was obtained: A_propio … N_sin_impacto")]
# step 13, per estimation id (the dynamics of the rate the series takes)
ESTIMATION_VALUES_STEP_13 = [("phi", "s13_phi", "13", "φ of the estimation id: observed variation of its rate / binomial noise (≈ 1: nothing to model)"),
                             ("tendencia", "s13_tendencia", "13", "+1 / −1 significant trend of the rate, 0 none"),
                             ("tendencia_pp_ano", "s13_tendencia_pp_ano", "13", "slope of the rate, pp per year"),
                             ("estacional", "s13_estacional", "13", "1 when the month of the year is significant (on the rate without trend)"),
                             ("amplitud_pp", "s13_amplitud_pp", "13", "highest month minus lowest month (pp)"),
                             ("meses_alto", "s13_meses_alto", "13", "calendar months above the others beyond twice their error"),
                             ("meses_bajo", "s13_meses_bajo", "13", "calendar months below the others beyond twice their error")]
# step 14, per estimation id and horizon band (pivoted: one column per band)
ESTIMATION_VALUES_STEP_14 = [("tecnica", "s14_tecnica", "14", "the technique chosen for the estimation id in this horizon band"),
                             ("tecnica_origen", "s14_origen", "14", "campeon (beat the challenger) · retador · sin_soporte (not judged)"),
                             ("elegida_err_pp_medio", "s14_examen_err_pp", "14", "mean |error| of the chosen technique in the exam months (pp of rate)"),
                             ("retador_err_pp_medio", "s14_examen_retador_err_pp", "14", "the same for the challenger: the one to beat"),
                             ("dentro_banda", "s14_examen_dentro_banda", "14", "share of the exam errors inside the band (nominal 90 %)")]


def build_core_table(fine_table: pd.DataFrame, configuration: Config, forecast_units: pd.DataFrame = None,
                     support_bound: pd.DataFrame = None, rated_units: pd.DataFrame = None,
                     series_table: pd.DataFrame = None, series_rate: pd.DataFrame = None,
                     series_estimate: pd.DataFrame = None, pool_dynamics: pd.DataFrame = None,
                     technique_decision: pd.DataFrame = None, exam_by_pool: pd.DataFrame = None,
                     forecast: pd.DataFrame = None, time_series_rows: pd.DataFrame = None,
                     time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None) -> tuple:
    """The core table with every block whose step has run, and its legend; checked and written.

    INPUT:   the fine table and the results of every step that has run (None = not run).
    OUTPUT:  (core, legend). The core has EVERY row: the original records of the pipeline and of the
             time_series universe, and the synthetic ones (gaps, extended horizon, ts_proyectado,
             ts_reentrada); every decision as a column; and the FINAL block (fin_*) that answers the
             business questions by summing it.
    RULES:   the fin_* columns are the same for every row: pipeline due (fin_vence_*) and renewed or
             revenue (fin_renovad*), real or expected, with its origin. A total of any year and origin
             is a SUM of the core; sff_forecast_total must equal it (check 5).
    EDGE CASES: a step not run leaves its block out; no time_series universe → no ts rows."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    dimension_columns = core_dimension_columns(configuration)
    blocks_present = []

    # [1] the rows of the extract
    raw_rows = rows_of_the_extract(fine_table, dimension_columns, configuration)
    configuration.log_action(STEP_LABEL, 1, f"{len(raw_rows):,} rows of the extract with {len(dimension_columns)} "
                                            f"dimension columns, their measures and their ids")

    # [2] the gap rows
    gap_rows = rows_of_the_gaps(rated_units, raw_rows.columns, configuration) if rated_units is not None else raw_rows.iloc[0:0]
    core = pd.concat([raw_rows, gap_rows], ignore_index=True) if len(gap_rows) else raw_rows
    extension_rows = rows_of_the_extension(forecast, core.columns, configuration) if forecast is not None else core.iloc[0:0]
    if len(extension_rows):
        core = pd.concat([core, extension_rows], ignore_index=True)
    time_series_core_rows = rows_of_the_time_series(time_series_rows, time_series_table, core.columns, configuration)
    if len(time_series_core_rows):
        core = pd.concat([core, time_series_core_rows], ignore_index=True)
    configuration.log_action(STEP_LABEL, 2, f"{len(gap_rows):,} gap rows · {len(extension_rows):,} extended-horizon rows · "
                                            f"{len(time_series_core_rows):,} time_series rows "
                                            f"({time_series_core_rows[ROW_ORIGIN_COLUMN].value_counts().to_dict() if len(time_series_core_rows) else {}})")

    # [3] the values of the forecast unit
    for block_frame, block_specs in ((forecast_units, UNIT_VALUES_STEP_04), (support_bound, UNIT_VALUES_STEP_07)):
        if block_frame is not None:
            core = add_block(core, block_frame, UNIT_ID_COLUMN, "s03_fu_id", block_specs)
            blocks_present.append(block_specs)
    configuration.log_action(STEP_LABEL, 3, f"unit blocks added: {[specs[0][2] for specs in blocks_present] or 'none'}")

    # [4] the values of the series and of its estimation id
    series_blocks = [(series_table, SERIES_VALUES_STEP_06), (series_rate, SERIES_VALUES_STEP_08),
                     (series_estimate, SERIES_VALUES_STEP_11)]
    for block_frame, block_specs in series_blocks:
        if block_frame is not None:
            core = add_block(core, block_frame, SERIES_ID_COLUMN, "s03_fs_id", block_specs)
            blocks_present.append(block_specs)
    if pool_dynamics is not None and len(pool_dynamics) and "s11_final_group_id" in core.columns:
        dynamics_values = pool_dynamics[[ESTIMATION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_13]]
        dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_13})
        core = core.merge(dynamics_values, left_on="s11_final_group_id", right_on=ESTIMATION_ID_COLUMN, how="left").drop(columns=ESTIMATION_ID_COLUMN)
        blocks_present.append(ESTIMATION_VALUES_STEP_13)
    if technique_decision is not None and "s11_final_group_id" in core.columns:
        core, backtest_specs = add_backtest_block(core, technique_decision, exam_by_pool, configuration)
        blocks_present.append(backtest_specs)
    if forecast is not None:
        forecast_values = forecast.assign(_clave=forecast_keys(forecast), _vencen_unidades=forecast[configuration.pipeline_units_col],
                                          _vencen_usd=forecast[configuration.pipeline_usd_col])
        forecast_values = forecast_values[["_clave"] + [source for source, _, _, _ in ROW_VALUES_STEP_17]]
        forecast_values = forecast_values.rename(columns={source: name for source, name, _, _ in ROW_VALUES_STEP_17})
        core = core.merge(forecast_values, on="_clave", how="left")
        blocks_present.append(ROW_VALUES_STEP_17)
    core = core.drop(columns=["_fila", "_clave"], errors="ignore")
    core = add_final_block(core, configuration)
    blocks_present.append(FINAL_VALUES)
    if "s04_filas_finas" in core.columns:
        core.loc[core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP, "s04_filas_finas"] = 0     # a gap adds no fine row
    core = core.sort_values(["s03_fs_id", configuration.period_col, ROW_ORIGIN_COLUMN]).reset_index(drop=True)
    configuration.log_action(STEP_LABEL, 4, f"blocks present: {sorted({specs[0][2] for specs in blocks_present})} · "
                                            f"the core has {len(core):,} rows × {len(core.columns)} columns")

    # [5] the legend
    legend = core_legend(core, dimension_columns, blocks_present, configuration)
    configuration.log_action(STEP_LABEL, 5, f"legend of {len(legend)} columns: {legend['nivel'].value_counts().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the core against the extract and the legend")
    check_core(core, fine_table, gap_rows, legend, blocks_present, forecast_total, configuration, check_log)

    # [7] the tables, written
    configuration.log_action(STEP_LABEL, 7, "writing the core and its legend")
    configuration.write_table(STEP_LABEL, check_log, core, TABLE_CORE)
    configuration.write_table(STEP_LABEL, check_log, legend, TABLE_CORE_LEGEND)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] one series as it looks in the core, how to read it in Power BI, and the questions answered from it
    log_core_report(core, configuration)
    log_core_answers(core, configuration)
    return core, legend


def add_block(core: pd.DataFrame, block_frame: pd.DataFrame, key_column: str, core_key: str, block_specs: list) -> pd.DataFrame:
    """The values of one block, joined to every row of the core by its unit or its series."""
    source_columns = [source for source, _, _, _ in block_specs]
    values = block_frame[[key_column] + source_columns].drop_duplicates(key_column)
    values = values.rename(columns={source: name for source, name, _, _ in block_specs})
    return core.merge(values, left_on=core_key, right_on=key_column, how="left").drop(columns=key_column)


def add_backtest_block(core: pd.DataFrame, technique_decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                       configuration: Config) -> tuple:
    """The chosen technique and the exam error of the row's estimation id, one column per horizon band."""
    per_band = technique_decision[[ESTIMATION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
    if exam_by_pool is not None:
        per_band = per_band.merge(exam_by_pool[[ESTIMATION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "retador_err_pp_medio",
                                                "dentro_banda"]], on=[ESTIMATION_ID_COLUMN, "tramo_h"], how="left")
    wide = per_band.pivot(index=ESTIMATION_ID_COLUMN, columns="tramo_h")
    specs = []
    renamed = {}
    for source, name, step, description in ESTIMATION_VALUES_STEP_14:
        for band_name in configuration.horizon_bands:
            if (source, band_name) in wide.columns:
                renamed[(source, band_name)] = f"{name}_{band_name}"
                specs.append((source, f"{name}_{band_name}", step, f"{description} (band {band_name})"))
    wide = wide[list(renamed)]
    wide.columns = [renamed[column] for column in wide.columns]
    core = core.merge(wide, left_on="s11_final_group_id", right_index=True, how="left")
    return core, specs


# ─── the final block: the same columns for every row, the ones the business questions sum ───
FINAL_UNIVERSE_COLUMN = "fin_universo"
FINAL_ORIGIN_COLUMN = "fin_origen"
FINAL_STATE_COLUMN = "fin_estado"
FINAL_YEAR_COLUMN = "fin_ano"
FINAL_UNIVERSE_PIPELINE, FINAL_UNIVERSE_TIME_SERIES = "pipeline", "time_series"
STATE_REAL, STATE_EXPECTED = "real", "previsto"
TS_WITHOUT_RESULT = "ts_sin_resultado"       # an original time_series row of a month not closed yet (its result is projected)
FINAL_VALUES = [
    ("fin_universo", "fin_universo", "FIN", "pipeline (renewals) · time_series (retail to subscription)"),
    ("fin_origen", "fin_origen", "FIN", "pipeline_renovado_real · pipeline_real_esperado · pipeline_proyectada · pipeline_simulada · "
                                        "ts_real · ts_proyectado · ts_reentrada · hueco · ts_sin_resultado"),
    ("fin_estado", "fin_estado", "FIN", "real (already happened) · previsto (forecast)"),
    ("fin_ano", "fin_ano", "FIN", "the year of the month"),
    ("fin_vence_unidades", "fin_vence_unidades", "FIN", "PIPELINE: units falling due (extract, extended horizon, ts re-entry) · SUM"),
    ("fin_vence_usd", "fin_vence_usd", "FIN", "PIPELINE: USD falling due (acquisition and ts re-entry: at their discount) · SUM"),
    ("fin_renovadas_unidades", "fin_renovadas_unidades", "FIN", "RENEWED units: real in closed months, expected in the future; ts: converted units · SUM"),
    ("fin_renovado_usd", "fin_renovado_usd", "FIN", "RENEWED USD (or time_series revenue): real in closed months, expected in the future · SUM"),
    ("fin_renovado_usd_bajo", "fin_renovado_usd_bajo", "FIN", "low end of the row's band (SUM = the worst case of a total; real rows: the real value)"),
    ("fin_renovado_usd_alto", "fin_renovado_usd_alto", "FIN", "high end of the row's band (SUM = the worst case of a total; real rows: the real value)")]
FINAL_SUMMABLE = {"fin_vence_unidades", "fin_vence_usd", "fin_renovadas_unidades", "fin_renovado_usd", "fin_renovado_usd_bajo",
                  "fin_renovado_usd_alto"}


def rows_of_the_time_series(time_series_rows, time_series_table, core_columns, configuration: Config) -> pd.DataFrame:
    """The time_series universe in the core: its ORIGINAL records (ts_real in the closed months,
    ts_sin_resultado in the months not closed yet) and its SYNTHETIC rows (ts_proyectado, ts_reentrada).
    Helper columns _ts_* carry their final values until the final block reads them."""
    frames = []
    helper = ["_ts_vence_unidades", "_ts_vence_usd", "_ts_renovadas", "_ts_usd"]
    current = configuration.calendar_boundaries()["current"]
    if time_series_rows is not None and len(time_series_rows):
        original = pd.DataFrame({configuration.period_col: time_series_rows[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in time_series_rows.columns and column_name not in original.columns:
                original[column_name] = time_series_rows[column_name].to_numpy()
        closed = original[configuration.period_col] < current
        original[ROW_ORIGIN_COLUMN] = np.where(closed, TS_REAL, TS_WITHOUT_RESULT)
        original["s00_vencen_unidades"] = time_series_rows[configuration.pipeline_units_col].fillna(0).to_numpy()
        original["s00_vencen_usd"] = time_series_rows[configuration.pipeline_usd_col].fillna(0).to_numpy()
        original["s00_renovadas_unidades"] = time_series_rows[configuration.renewed_units_col].fillna(0).to_numpy()
        original["s00_renovado_usd"] = time_series_rows[configuration.renewed_usd_col].fillna(0).to_numpy()
        original["s02_rol"] = configuration.role_of_months(original[configuration.period_col])
        original["_ts_vence_unidades"], original["_ts_vence_usd"] = 0.0, 0.0
        original["_ts_renovadas"] = np.where(closed, original["s00_renovadas_unidades"], 0.0)
        original["_ts_usd"] = np.where(closed, original["s00_renovado_usd"], 0.0)
        frames.append(original)
    if time_series_table is not None and len(time_series_table):
        synthetic = time_series_table[time_series_table["origen"].isin([TS_PROJECTED, TS_REENTRY])]
        rows = pd.DataFrame({ROW_ORIGIN_COLUMN: synthetic["origen"].to_numpy(),
                             configuration.period_col: synthetic[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in synthetic.columns and column_name not in rows.columns:
                rows[column_name] = synthetic[column_name].to_numpy()
        for measure_name, _, _ in RAW_MEASURES:
            rows[measure_name] = 0.0
        rows["s02_rol"] = configuration.role_of_months(rows[configuration.period_col])
        reentry = (synthetic["origen"] == TS_REENTRY).to_numpy()
        rows["_ts_vence_unidades"] = np.where(reentry, synthetic["unidades"].to_numpy(), 0.0)
        rows["_ts_vence_usd"] = np.where(reentry, synthetic["valor"].to_numpy(), 0.0)      # the pipeline: value at 40 % off
        rows["_ts_renovadas"] = np.where(reentry, synthetic.get("unidades_renovadas", pd.Series(0.0, index=synthetic.index)).fillna(0).to_numpy(),
                                         synthetic["unidades"].to_numpy())
        rows["_ts_usd"] = synthetic["revenue"].to_numpy()
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=list(core_columns))
    return pd.concat(frames, ignore_index=True).reindex(columns=list(core_columns) + helper)


def add_final_block(core: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The final block: universe, origin, state, year, pipeline due and renewed (or revenue), row by row."""
    origin = core[ROW_ORIGIN_COLUMN]
    closed = core["s02_rol"].isin(TRUTH_ROLES_OF_CORE)
    is_time_series = origin.astype(str).str.startswith("ts_")
    zeros = pd.Series(0.0, index=core.index)
    column = lambda name: core[name].fillna(0.0) if name in core.columns else zeros

    raw_closed, raw_future = (origin == ROW_FROM_RAW) & closed, (origin == ROW_FROM_RAW) & ~closed
    extended = origin.isin([PIPELINE_PROJECTED, PIPELINE_SIMULATED])
    core[FINAL_UNIVERSE_COLUMN] = np.where(is_time_series, FINAL_UNIVERSE_TIME_SERIES, FINAL_UNIVERSE_PIPELINE)
    core[FINAL_ORIGIN_COLUMN] = np.select(
        [raw_closed, raw_future, origin == PIPELINE_PROJECTED, origin == PIPELINE_SIMULATED, origin == ROW_FROM_GAP],
        [TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED, ROW_FROM_GAP],
        default=origin.astype(str))
    core[FINAL_STATE_COLUMN] = np.where(raw_closed | (origin == ROW_FROM_GAP) | origin.isin([TS_REAL, TS_WITHOUT_RESULT]),
                                        STATE_REAL, STATE_EXPECTED)
    core[FINAL_YEAR_COLUMN] = core[configuration.period_col].map(lambda month: month.year)

    extract_rows = origin.isin([ROW_FROM_RAW, ROW_FROM_GAP])            # the pipeline after the calendar (s02): not what is not known yet
    core["fin_vence_unidades"] = np.select([extract_rows, extended, is_time_series],
                                           [column("s02_vencen_unidades"), column("s17_vencen_unidades"), column("_ts_vence_unidades")], 0.0)
    core["fin_vence_usd"] = np.select([extract_rows, extended, is_time_series],
                                      [column("s02_vencen_usd"), column("s17_vencen_usd"), column("_ts_vence_usd")], 0.0)
    core["fin_renovadas_unidades"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                               [column("s02_renovadas_unidades"), column("s17_esperado_unidades"), column("_ts_renovadas")], 0.0)
    core["fin_renovado_usd"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                         [column("s02_renovado_usd"), column("s17_esperado_usd"), column("_ts_usd")], 0.0)
    with_band = raw_future | extended
    core["fin_renovado_usd_bajo"] = np.where(with_band, column("s17_esperado_usd_bajo"), core["fin_renovado_usd"])
    core["fin_renovado_usd_alto"] = np.where(with_band, column("s17_esperado_usd_alto"), core["fin_renovado_usd"])
    return core.drop(columns=[name for name in core.columns if name.startswith("_ts_")])


def core_dimension_columns(configuration: Config) -> list:
    """The dimension columns of the core, as the extract names them (they are the slicers):
    the rate series columns, the revaluation extras, the exact discount and its bucket."""
    discount = [configuration.discount_value_column, configuration.discount_bucket_column] if configuration.discount_value_column else []
    return list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion + discount))


def rows_of_the_extract(fine_table: pd.DataFrame, dimension_columns: list, configuration: Config) -> pd.DataFrame:
    """One row per fine row: its month, its dimensions, its measures (raw and after the
    calendar), its role and its ids in text (no hash keys)."""
    raw_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_RAW, configuration.period_col: fine_table[configuration.period_col]})
    for column_name in dimension_columns:
        raw_rows[column_name] = fine_table[column_name]
    raw_rows["s00_vencen_unidades"] = fine_table[S0_PIPELINE_UNITS_COLUMN]
    raw_rows["s00_vencen_usd"] = fine_table[S0_PIPELINE_USD_COLUMN]
    raw_rows["s02_vencen_unidades"] = fine_table[configuration.pipeline_units_col]
    raw_rows["s02_vencen_usd"] = fine_table[configuration.pipeline_usd_col]
    raw_rows["s00_renovadas_unidades"] = fine_table[S0_RENEWED_UNITS_COLUMN]
    raw_rows["s00_renovado_usd"] = fine_table[S0_RENEWED_USD_COLUMN]
    raw_rows["s02_rol"] = fine_table[CALENDAR_ROLE_COLUMN]
    raw_rows["s02_es_mes_en_curso"] = fine_table[CURRENT_MONTH_COLUMN]
    raw_rows["s02_renovadas_unidades"] = fine_table[configuration.renewed_units_col]
    raw_rows["s02_renovado_usd"] = fine_table[configuration.renewed_usd_col]
    raw_rows["s03_fs_id"] = fine_table[SERIES_ID_COLUMN]
    raw_rows["s03_fu_id"] = fine_table[UNIT_ID_COLUMN]
    raw_rows["s03_uplift_cell_id"] = fine_table[UPLIFT_CELL_ID_COLUMN]
    raw_rows["_fila"] = fine_table.index      # the position of the fine row: the forecast joins on it, then it is dropped
    raw_rows["_clave"] = [f"f{position}" for position in fine_table.index]
    return raw_rows.reset_index(drop=True)


def forecast_keys(forecast: pd.DataFrame) -> list:
    """The join key of every forecast row: 'f<position>' for a row of the extract, 'e<n>' for an extended row."""
    return [f"f{int(position)}" if pd.notna(position) else f"e{number}"
            for number, position in enumerate(forecast["_fila"])]


def rows_of_the_extension(forecast: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per row of the extended horizon (proyectada, simulada): its month, dims and ids;
    the measures of the extract at 0 (it is not in the extract: its pipeline is in s17_vencen_*)."""
    keys = forecast_keys(forecast)
    extension = forecast[forecast["_fila"].isna()]
    if extension.empty:
        return pd.DataFrame(columns=core_columns)
    rows = pd.DataFrame({ROW_ORIGIN_COLUMN: extension[PIPELINE_ORIGIN_COLUMN].to_numpy(),
                         configuration.period_col: extension[configuration.period_col].to_numpy()})
    for column_name in core_columns:
        if column_name in extension.columns and column_name not in rows.columns and not column_name.startswith("s"):
            rows[column_name] = extension[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES:
        rows[measure_name] = 0.0
    rows["s02_rol"] = extension[CALENDAR_ROLE_COLUMN].to_numpy()
    rows["s02_es_mes_en_curso"] = 0
    rows["s03_fs_id"] = extension[SERIES_ID_COLUMN].to_numpy()
    rows["s03_fu_id"] = extension[UNIT_ID_COLUMN].to_numpy()
    rows["s03_uplift_cell_id"] = extension[UPLIFT_CELL_ID_COLUMN].to_numpy()
    rows["_clave"] = [key for key, position in zip(keys, forecast["_fila"]) if pd.isna(position)]
    return rows.reindex(columns=list(core_columns))


def rows_of_the_gaps(rated_units: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per gap of step 08: its month, its series columns, every measure at 0, its
    role; the revaluation columns are empty (a gap has no contract)."""
    gaps = rated_units[rated_units[SYNTHETIC_COLUMN] == 1]
    if gaps.empty:
        return pd.DataFrame(columns=core_columns)
    gap_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_GAP, configuration.period_col: gaps[configuration.period_col].to_numpy()})
    for column_name in configuration.rate_series_columns:
        gap_rows[column_name] = gaps[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES + CALENDAR_MEASURES:
        gap_rows[measure_name] = 0.0
    gap_rows["s02_rol"] = gaps[CALENDAR_ROLE_COLUMN].to_numpy()
    gap_rows["s02_es_mes_en_curso"] = 0
    gap_rows["s03_fs_id"] = gaps[SERIES_ID_COLUMN].to_numpy()
    gap_rows["s03_fu_id"] = gaps[UNIT_ID_COLUMN].to_numpy()
    return gap_rows.reindex(columns=core_columns)


def core_legend(core: pd.DataFrame, dimension_columns: list, blocks_present: list, configuration: Config) -> pd.DataFrame:
    """Every column of the core with its step, its level and how to aggregate it."""
    legend_rows = [(ROW_ORIGIN_COLUMN, "NU", LEVEL_ROW, AGGREGATE_SLICER, "raw = a row of the extract · hueco = a month with nothing "
                    "due inside a history · proyectada / simulada = a row of the extended horizon (renewal falling due again / "
                    "last year's acquisition) · ts_real / ts_sin_resultado = an original row of the time_series universe · "
                    "ts_proyectado / ts_reentrada = its synthetic rows"),
                   (configuration.period_col, "00", LEVEL_ROW, AGGREGATE_SLICER, "the month")]
    for column_name in dimension_columns:
        description = ("the exact discount (share)" if column_name == configuration.discount_value_column
                       else "the discount bucket, derived in step 03" if column_name == configuration.discount_bucket_column
                       else "a dimension of the extract")
        legend_rows.append((column_name, "03" if column_name == configuration.discount_bucket_column else "00",
                            LEVEL_ROW, AGGREGATE_SLICER, description))
    for name, step, description in RAW_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s02_rol", "02", LEVEL_ROW, AGGREGATE_SLICER, "entrenamiento · examen · proyeccion"),
                    ("s02_es_mes_en_curso", "02", LEVEL_ROW, AGGREGATE_SLICER, "1 in the current month")]
    for name, step, description in CALENDAR_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s03_fs_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the rate series: select one to see its story"),
                    ("s03_fu_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the forecast unit: the series in its month"),
                    ("s03_uplift_cell_id", "03", LEVEL_ROW, AGGREGATE_SLICER, "the price context (uplift cell)")]
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_04 + UNIT_VALUES_STEP_07}
    row_names = {name for _, name, _, _ in ROW_VALUES_STEP_17}
    summable = {"s17_esperado_unidades", "s17_esperado_usd", "s17_esperado_usd_bajo", "s17_esperado_usd_alto",
                "s17_vencen_unidades", "s17_vencen_usd"}
    for block_specs in blocks_present:
        for _, name, step, description in block_specs:
            if name in {spec_name for _, spec_name, _, _ in FINAL_VALUES}:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in FINAL_SUMMABLE else AGGREGATE_SLICER,
                                    description))
                continue
            if name in row_names:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in summable else AGGREGATE_SLICER
                                    if name in ("s17_origen_tasa", "s17_tecnica", "s15_via_uplift", "s17_h")
                                    else "no sumar: valor de la fila (una tasa o un uplift)", description))
                continue
            level = LEVEL_UNIT if name in unit_names else LEVEL_SERIES
            legend_rows.append((name, step, level, AGGREGATE_ATTRIBUTE, description))
    legend = pd.DataFrame(legend_rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
    return legend[legend["columna"].isin(core.columns)].drop_duplicates("columna").reset_index(drop=True)


def check_core(core: pd.DataFrame, fine_table: pd.DataFrame, gap_rows: pd.DataFrame, legend: pd.DataFrame,
               blocks_present: list, forecast_total, configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    # [1] rows: the extract, the gaps, the extended horizon and the time_series universe
    origin_counts = core[ROW_ORIGIN_COLUMN].value_counts().to_dict()
    configuration.log_check(STEP_LABEL, check_log, "every original record and every synthetic row, once",
                            int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW).sum()) == len(fine_table)
                            and int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP).sum()) == len(gap_rows),
                            failure_detail=f"rows by origin {origin_counts} for {len(fine_table):,} fine rows and {len(gap_rows):,} gaps",
                            context=" · ".join(f"{origin} {count:,}" for origin, count in origin_counts.items()))

    # [2] the money of the pipeline reconciles with the extract (the time_series rows are apart)
    pipeline_rows = core[FINAL_UNIVERSE_COLUMN] == FINAL_UNIVERSE_PIPELINE
    reconciliation = {"s00_vencen_unidades": fine_table[S0_PIPELINE_UNITS_COLUMN].sum(),
                      "s00_vencen_usd": fine_table[S0_PIPELINE_USD_COLUMN].sum(),
                      "s02_vencen_unidades": fine_table[configuration.pipeline_units_col].sum(),
                      "s02_vencen_usd": fine_table[configuration.pipeline_usd_col].sum(),
                      "s00_renovadas_unidades": fine_table[S0_RENEWED_UNITS_COLUMN].sum(),
                      "s00_renovado_usd": fine_table[S0_RENEWED_USD_COLUMN].sum()}
    differences = {name: float(core.loc[pipeline_rows, name].sum() - total) for name, total in reconciliation.items()
                   if abs(core.loc[pipeline_rows, name].sum() - total) > MONEY_TOLERANCE}
    configuration.log_check(STEP_LABEL, check_log, "the money reconciles with the extract", not differences,
                            failure_detail=f"differences core − extract: {differences}",
                            context=f"${reconciliation['s00_vencen_usd']:,.0f} due · ${reconciliation['s00_renovado_usd']:,.0f} renewed")

    # [3] every row has the values of every block present (the first column of each block);
    #     a gap is not a unit of step 04, so the unit blocks are checked on the rows of the extract
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_04 + UNIT_VALUES_STEP_07}
    first_columns = [block_specs[0][1] for block_specs in blocks_present if block_specs and block_specs[0][1] in core.columns
                     and not block_specs[0][1].startswith(("s13_", "s14_", "s17_", "fin_"))]
    series_columns = [name for name in first_columns if name not in unit_names]
    unit_columns = [name for name in first_columns if name in unit_names]
    raw_rows = core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW
    history_rows = core[ROW_ORIGIN_COLUMN].isin([ROW_FROM_RAW, ROW_FROM_GAP])       # an extended row may be a new series
    without_values = (int(core.loc[history_rows, series_columns].isna().any(axis=1).sum()) if series_columns else 0) + \
                     (int(core.loc[raw_rows, unit_columns].isna().any(axis=1).sum()) if unit_columns else 0)
    configuration.log_check(STEP_LABEL, check_log, "every row has the values of every block present", without_values == 0,
                            failure_detail=f"{without_values:,} rows without some block's values",
                            context=f"{len(blocks_present)} blocks")

    # [4] the legend covers the core
    undescribed = [column_name for column_name in core.columns if column_name not in set(legend["columna"])]
    configuration.log_check(STEP_LABEL, check_log, "the legend describes every column of the core", not undescribed,
                            failure_detail=f"columns without legend: {undescribed}",
                            context=f"{len(core.columns)} columns")

    # [5] the core answers the totals: summed by year and origin it equals sff_forecast_total
    if forecast_total is None:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the core summed by year and origin = sff_forecast_total",
                                        "step 20 has not run")
        return
    parts = forecast_total[forecast_total["origen"] != TOTAL_ORIGIN_TOTAL].set_index(["ano", "origen"])
    summed = core.groupby([FINAL_YEAR_COLUMN, FINAL_ORIGIN_COLUMN])[["fin_vence_usd", "fin_renovado_usd"]].sum()
    summed.index = summed.index.set_names(["ano", "origen"])
    compared = parts.join(summed, how="left").fillna(0.0)
    mismatched = compared[((compared["usd_vence"] - compared["fin_vence_usd"]).abs() > MONEY_TOLERANCE)
                          | ((compared["usd_renovado"] - compared["fin_renovado_usd"]).abs() > MONEY_TOLERANCE)]
    configuration.log_check(STEP_LABEL, check_log, "the core summed by year and origin = sff_forecast_total (the total comes from the core)",
                            mismatched.empty, failure_detail=f"{len(mismatched)} year × origin differ",
                            context=f"{len(compared)} year × origin compared",
                            examples=mismatched.reset_index()[["ano", "origen", "usd_vence", "fin_vence_usd", "usd_renovado",
                                                               "fin_renovado_usd"]])


def log_core_answers(core: pd.DataFrame, configuration: Config) -> None:
    """The business questions answered by SUMMING the core: renewed and pipeline by year, state and origin."""
    current_year = configuration.calendar_boundaries()["current"].year
    answers = (core[core[FINAL_YEAR_COLUMN] >= current_year]
               .groupby([FINAL_YEAR_COLUMN, FINAL_STATE_COLUMN, FINAL_ORIGIN_COLUMN])[["fin_vence_usd", "fin_renovado_usd"]].sum()
               .reset_index())
    answers = answers[(answers["fin_vence_usd"] != 0) | (answers["fin_renovado_usd"] != 0)]
    configuration.logger.doc(f"[{STEP_LABEL}] the questions answered by SUMMING the core (fin_renovado_usd: renewed or revenue, "
                             f"real or expected · fin_vence_usd: pipeline), by year, state and origin:")
    configuration.show_table(answers)


def log_core_report(core: pd.DataFrame, configuration: Config) -> None:
    """Action 9: one series as it looks in the core, and how to read it in Power BI."""
    largest_series = core.groupby("s03_fs_id")["s00_vencen_usd"].sum().idxmax()
    configuration.log_action(STEP_LABEL, 9, f"the series with the most money, '{largest_series}', as it looks in the "
                                            f"core (last {MONTHS_SHOWN} closed months; the series values repeat on every row):")
    series_rows = core[(core["s03_fs_id"] == largest_series) & core["s02_rol"].isin([ROLE_TRAIN, ROLE_TEST])]
    shown_columns = [configuration.period_col, ROW_ORIGIN_COLUMN, "s02_rol", "s00_vencen_unidades", "s02_renovadas_unidades",
                     "s06_ruta", "s08_tasa_propia", "s11_tasa_estimada", "s11_nivel_riesgo", "s14_tecnica_corto",
                     "s14_examen_err_pp_corto"]
    configuration.show_table(series_rows.tail(MONTHS_SHOWN)[[column for column in shown_columns if column in core.columns]])
    configuration.logger.doc(f"[{STEP_LABEL}] in Power BI: a slicer on s03_fs_id selects one series; the rows are its "
                             f"months. A rate is a measure (a ratio of sums over the rows selected), never a column; "
                             f"a value of the series (s06_, s08_, s11_) is read with MAX. Measures to create:")
    configuration.show_table(pd.DataFrame([
        ("Tasa renovación (meses cerrados)",
         "DIVIDE(CALCULATE(SUM(sff_nucleo[s02_renovadas_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}), "
         "CALCULATE(SUM(sff_nucleo[s00_vencen_unidades]), sff_nucleo[s02_rol] IN {\"entrenamiento\", \"examen\"}))"),
        ("Vence USD", "SUM(sff_nucleo[s00_vencen_usd])"),
        ("Tasa estimada de la serie", "MAX(sff_nucleo[s11_tasa_estimada])"),
        ("Renovado (real + previsto)", "SUM(sff_nucleo[fin_renovado_usd])  — segmentar por fin_ano, fin_estado, fin_origen"),
        ("Pipeline", "SUM(sff_nucleo[fin_vence_usd])  — segmentar por fin_ano, fin_origen"),
        ("Nivel de riesgo de la serie", "MAX(sff_nucleo[s11_nivel_riesgo])")], columns=["medida", "DAX"]))
