"""nucleo.py — SFF v3 · the core table: the whole story in one table, left to right.

WHY ONE TABLE. Every number of the forecast is a product of factors that can be carried
down to the finest row: expected renewed $ = pipeline $ × rate × uplift. So the core of
the prediction lives in ONE table at the fine grain (combination × month), history and
future together, with no hashed keys: the dimensions themselves are columns. Every study
is an aggregation of it: a rate always belongs to an aggregation, and an aggregation is a
set of dimensions with a purpose. Rates are never stored to be summed: the table stores
numerators and denominators, and every rate is computed on the fly as a ratio of sums
(`agregar`), in Python or in Power BI.

THE ROWS (column `paso_alta`: which step created the row)
  s0_raw                   a row of the extract, as it came
  s1_hueco                 a month of a series with nothing due, made EXPLICIT: it is
                           information ("nothing was due, and that is correct"), stated in
                           the same format as the rest. Measures 0: no licence and no dollar
                           is added, the sums still match the reporting the raw came from
  s6_reentrada_proyectada  a renewal we forecast in a projection month, falling due again
                           one term later (at list price: its discount is 0 after renewing)
  s6_adquisicion_simulada  the acquisition of a projection month at the historical pace,
                           falling due for the first time one term later

THE COLUMNS, one block per step, in the order the framework adds them
  (identity)  paso_alta, s0_id_fila (the raw row key; null on added rows), period, and
              every declared dimension (mandatory, timevarying, extras, discount, sku)
  s0_  what the RAW said: role, current-month flag, pipeline and renewed as extracted
  s1_  the conditioned raw: calendar role from the current month (entrenamiento, examen,
       pendiente_cierre, proyeccion), renewed wiped in the future, treatment, universe,
       the binomial bound of the unit — raw rows + gap rows; Σ s1 pipeline = Σ raw
  s2_  the series and its support: id, sign, median monthly support, months, dial
       bucket, own rate
  s3_  the ladder: the pattern at every rung (s3_id_peldano_0 … _N: grouping by one
       reproduces the rung-N pools EXACTLY, members included), the chosen pool
       (s3_id_estimacion), rung, effective n, credibility z, estimated rate, prediction
       error, risk level, mandatory cell
  s4_  dynamics and technique: the benchmark's seasonal verdict, the technique of each
       horizon band
  s5_  revaluation: uplift cell, uplift, path (contrato / estadistico), origin
  s6_  the forecast: pipeline origin, h, the pipeline the forecast renews (raw future +
       re-entries + acquisition), pool rate at h, own deviation, predicted rate, expected
       units, expected $ at pipeline price and at renewal price, bands (per row and the
       common part), and on the exam months the prediction at h=1 and h=6
  s7_  maturation of the signals: the adjustment distributed to the neutral rows of its
       cell, and the adjusted expected $

HOW EACH COLUMN AGGREGATES (the contract Power BI follows)
  additive:        every *_unidades, *_usd, s6_unidades_esperadas, s6_usd_*, s6_banda_comun_*,
                   s7_* → SUM
  ratios, built at any aggregation from additive columns:
      realized rate  = Σ s1_renovados_unidades / Σ s1_pipeline_unidades   (closed months)
      predicted rate = Σ s6_unidades_esperadas / Σ s6_pipeline_unidades
      uplift         = Σ s6_usd_esperado / Σ s6_usd_esperado_precio_pipeline
      exam rate h    = Σ s6_examen_unidades_h{h} / Σ s1_pipeline_unidades  (exam months)
  per-row attributes (rate, z, level, technique…): never summed; averaged or filtered
  bands: linear inside (s3_id_estimacion, period), quadrature across, plus the common part
         linearly → `banda_agregada`
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, hash_key, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
DIAL_BUCKETS = ((0, 30, "0_menos_de_30"), (30, 271, "1_30_a_271"), (271, 752, "2_271_a_752"), (752, np.inf, "3_desde_752"))
EXAM_HORIZONS = (1, 6)


# ═══════════════════════════════════════════════════════════════════════════════════
# BUILDING THE ROWS
# ═══════════════════════════════════════════════════════════════════════════════════

def dimension_columns(configuration: Config) -> list:
    """Every declared dimension, in the configuration's order, without repetition."""
    ordered = (list(configuration.business_mandatory_dims) + list(configuration.structural_timevarying_dims)
               + list(configuration.extra_renovacion) + list(configuration.extra_revalorizacion)
               + [c for c in (configuration.discount_value_column, configuration.sku_column) if c])
    seen = []
    for column in ordered:
        if column not in seen:
            seen.append(column)
    return seen


def raw_rows(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 0 + step 1 of the raw rows: the extract as it came and as it is used."""
    period, role = configuration.period_col, configuration.dataset_role_col
    dims = [c for c in dimension_columns(configuration) if c in fine_table.columns]
    rows = pd.DataFrame({"paso_alta": ROW_FROM_RAW, "s0_id_fila": fine_table["fu_comb_key"].to_numpy(),
                         "fu_comb_key": fine_table["fu_comb_key"].to_numpy(), "fu_id": fine_table["fu_id"].to_numpy(),
                         period: fine_table[period].to_numpy()})
    for column in dims:
        rows[column] = fine_table[column].to_numpy()
    rows["s0_rol"] = fine_table["s0_rol"].to_numpy() if "s0_rol" in fine_table.columns else fine_table[role].astype(str).to_numpy()
    rows["s0_mes_en_curso"] = fine_table[configuration.current_month_col].to_numpy()
    rows["s0_pipeline_unidades"] = fine_table[configuration.pipeline_units_col].to_numpy(dtype=float)
    rows["s0_pipeline_usd"] = fine_table[configuration.pipeline_usd_col].to_numpy(dtype=float)
    rows["s0_renovados_unidades"] = (fine_table["s0_renovados_unidades"] if "s0_renovados_unidades" in fine_table.columns else fine_table[configuration.renewed_units_col]).to_numpy(dtype=float)
    rows["s0_renovados_usd"] = (fine_table["s0_renovados_usd"] if "s0_renovados_usd" in fine_table.columns else fine_table[configuration.renewed_usd_col]).to_numpy(dtype=float)
    rows["s1_rol"] = fine_table[role].to_numpy()
    rows["s1_pipeline_unidades"] = rows["s0_pipeline_unidades"]
    rows["s1_pipeline_usd"] = rows["s0_pipeline_usd"]
    rows["s1_renovados_unidades"] = fine_table[configuration.renewed_units_col].to_numpy(dtype=float)
    rows["s1_renovados_usd"] = fine_table[configuration.renewed_usd_col].to_numpy(dtype=float)
    return rows


def gap_rows(units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 1: the gap months of every series made explicit, one row each, measures 0.
    They carry the series' dimensions and the role of the previous real month; the
    revaluation extras do not exist for them (comb 'hueco')."""
    period, role = configuration.period_col, configuration.dataset_role_col
    gaps = units[units.get("sintetica", 0) == 1]
    if gaps.empty:
        return pd.DataFrame()
    rows = pd.DataFrame({"paso_alta": ROW_GAP, "s0_id_fila": np.nan, "fu_id": gaps["fu_id"].to_numpy(), period: gaps[period].to_numpy()})
    rows["fu_comb_key"] = [hash_key(f"{fu}||{GAP_COMBINATION}") for fu in rows["fu_id"]]
    for column in dimension_columns(configuration):
        rows[column] = gaps[column].to_numpy() if column in gaps.columns else np.nan
    for column in ("s0_rol", "s0_mes_en_curso", "s0_pipeline_unidades", "s0_pipeline_usd", "s0_renovados_unidades", "s0_renovados_usd"):
        rows[column] = np.nan
    rows["s1_rol"] = gaps[role].to_numpy()
    for column in ("s1_pipeline_unidades", "s1_pipeline_usd", "s1_renovados_unidades", "s1_renovados_usd"):
        rows[column] = 0.0
    return rows


def extended_rows(units_extended: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 6: the re-entries and the simulated acquisition. Not in the raw: s0_ and s1_
    stay null; their pipeline lives in s6_pipeline_* (filled from the forecast)."""
    period = configuration.period_col
    if units_extended is None or units_extended.empty:
        return pd.DataFrame()
    origin = units_extended.get("origen_pipeline", pd.Series(PIPELINE_SIMULATED, index=units_extended.index))
    rows = pd.DataFrame({"paso_alta": np.where(origin == PIPELINE_PROJECTED, ROW_PROJECTED_REENTRY, ROW_SIMULATED_ACQUISITION),
                         "s0_id_fila": np.nan, "fu_comb_key": units_extended["fu_comb_key"].to_numpy(),
                         "fu_id": units_extended["fu_id"].to_numpy(), period: units_extended[period].to_numpy()})
    for column in dimension_columns(configuration):
        rows[column] = units_extended[column].to_numpy() if column in units_extended.columns else np.nan
    for column in ("s0_rol", "s0_mes_en_curso", "s0_pipeline_unidades", "s0_pipeline_usd", "s0_renovados_unidades", "s0_renovados_usd",
                   "s1_rol", "s1_pipeline_unidades", "s1_pipeline_usd", "s1_renovados_unidades", "s1_renovados_usd"):
        rows[column] = np.nan
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# ADDING THE COLUMNS OF EVERY STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def add_step1_unit_columns(core: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """s1_: universe, treatment and the binomial bound of the unit (broadcast to its rows)."""
    per_unit = units.drop_duplicates("fu_id").set_index("fu_id")
    core["s1_universo"] = core["fu_id"].map(per_unit.get("universo")) if "universo" in per_unit.columns else np.nan
    core["s1_tratamiento"] = core["fu_id"].map(per_unit.get("ruta")) if "ruta" in per_unit.columns else np.nan
    core["s1_se_pp_max"] = core["fu_id"].map(per_unit.get("se_pp_max")) if "se_pp_max" in per_unit.columns else np.nan
    core["s1_moe_usd_max"] = core["fu_id"].map(per_unit.get("moe_usd_max")) if "moe_usd_max" in per_unit.columns else np.nan
    added = core["paso_alta"].isin((ROW_PROJECTED_REENTRY, ROW_SIMULATED_ACQUISITION))
    core.loc[added, ["s1_universo", "s1_tratamiento", "s1_se_pp_max", "s1_moe_usd_max"]] = np.nan
    return core


def dial_bucket(n: float) -> str:
    for low, high, label in DIAL_BUCKETS:
        if low <= n < high:
            return label
    return DIAL_BUCKETS[-1][2]


def add_step2_series(core: pd.DataFrame, configuration: Config, series_card: pd.DataFrame) -> pd.DataFrame:
    """s2_: the series of every row (rebuilt from its dimensions, so added rows get it too)."""
    core["s2_id_serie"] = join_columns(core, configuration.rate_series_columns)
    card = series_card.set_index("fs_id")
    core["s2_signo"] = core["s2_id_serie"].map(card["signo"])
    core["s2_n_serie"] = core["s2_id_serie"].map(card["n_propio"])
    core["s2_meses_historia"] = core["s2_id_serie"].map(card["meses_historia"])
    core["s2_tramo_dial"] = core["s2_n_serie"].map(lambda n: dial_bucket(float(n)) if pd.notna(n) else np.nan)
    core["s2_tasa_propia"] = core["s2_id_serie"].map(card["tasa_propia"]) if "tasa_propia" in card.columns else np.nan
    return core


def add_step3_ladder(core: pd.DataFrame, series_card: pd.DataFrame, parent_ladder: pd.DataFrame) -> pd.DataFrame:
    """s3_: the pattern at every rung, the chosen pool and the estimation of the series."""
    rungs = parent_ladder.pivot_table(index="fs_id", columns="peldano", values="padre_id", aggfunc="first") if len(parent_ladder) else pd.DataFrame()
    for rung in sorted(rungs.columns) if len(rungs) else []:
        core[f"s3_id_peldano_{int(rung)}"] = core["s2_id_serie"].map(rungs[rung])
    card = series_card.set_index("fs_id")
    for target, source in (("s3_id_estimacion", "id_estimacion"), ("s3_peldano", "peldano"), ("s3_n_efectivo", "n_efectivo"), ("s3_z", "z"),
                           ("s3_tasa_estimada", "tasa_estimada"), ("s3_se_prediccion_pp", "se_prediccion_pp"),
                           ("s3_nivel_riesgo", "nivel_riesgo"), ("s3_celda_mandatory", "celda_id")):
        core[target] = core["s2_id_serie"].map(card[source]) if source in card.columns else np.nan
    return core


def add_step4_technique(core: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """s4_: the benchmark's seasonal verdict of the pool and the technique of each horizon band."""
    reference = decisions.get("pool_reference", pd.DataFrame())
    if len(reference):
        core["s4_estacional"] = core["s3_id_estimacion"].map(reference.set_index("id_estimacion")["estacional"])
    technique = decisions.get("decision_technique", pd.DataFrame())
    if len(technique):
        for band, block in technique.groupby("tramo_h"):
            core[f"s4_tecnica_{band}"] = core["s3_id_estimacion"].map(block.drop_duplicates("id_estimacion").set_index("id_estimacion")["tecnica"])
    return core


def add_steps5_6_forecast(core: pd.DataFrame, detail: pd.DataFrame, bands: pd.DataFrame, common_low: np.ndarray,
                          common_high: np.ndarray, configuration: Config) -> pd.DataFrame:
    """s5_ and s6_: uplift and forecast of every future row (raw future + added rows)."""
    if detail is None or detail.empty:
        return core
    forecast = detail.set_index("fu_comb_key")
    band = bands.set_index("fu_comb_key") if bands is not None and len(bands) else pd.DataFrame()
    common = pd.DataFrame({"low": common_low, "high": common_high}, index=detail["fu_comb_key"].to_numpy())
    key = core["fu_comb_key"]
    core["s5_celda_uplift"] = key.map(forecast["uplift_cell_id"]) if "uplift_cell_id" in forecast.columns else np.nan
    core["s5_uplift"] = key.map(forecast["uplift"])
    core["s5_uplift_via"] = key.map(forecast["uplift_via"]) if "uplift_via" in forecast.columns else np.nan
    core["s5_uplift_origen"] = key.map(forecast["uplift_origen"])
    core["s6_origen_pipeline"] = key.map(forecast["origen_pipeline"])
    core["s6_h"] = key.map(forecast["h"])
    core["s6_mes_pendiente_cierre"] = key.map(forecast["mes_pendiente_cierre"]) if "mes_pendiente_cierre" in forecast.columns else np.nan
    core["s6_tecnica"] = key.map(forecast["tecnica"])
    core["s6_pipeline_unidades"] = key.map(forecast[configuration.pipeline_units_col])
    core["s6_pipeline_usd"] = key.map(forecast[configuration.pipeline_usd_col])
    core["s6_tasa_pool_h"] = key.map(forecast["tasa_pool_h"]) if "tasa_pool_h" in forecast.columns else np.nan
    core["s6_desviacion_propia_pp"] = key.map(forecast["desviacion_propia_pp"]) if "desviacion_propia_pp" in forecast.columns else np.nan
    core["s6_tasa_prevista"] = key.map(forecast["tasa"])
    core["s6_unidades_esperadas"] = core["s6_pipeline_unidades"] * core["s6_tasa_prevista"]
    core["s6_usd_esperado_precio_pipeline"] = core["s6_pipeline_usd"] * core["s6_tasa_prevista"]
    core["s6_usd_esperado"] = key.map(forecast["esperado_usd"])
    core["s6_banda_low_usd"] = key.map(band["banda_low_usd"]) if len(band) else np.nan
    core["s6_banda_high_usd"] = key.map(band["banda_high_usd"]) if len(band) else np.nan
    core["s6_banda_comun_low_usd"] = key.map(common["low"])
    core["s6_banda_comun_high_usd"] = key.map(common["high"])
    return core


def add_step6_exam(core: pd.DataFrame, backtest_holdout: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """s6_examen_*: on the exam months, the rate the chosen technique predicted for the pool
    at h=1 and h=6, and the expected units — 'predicted vs real' from the same table."""
    if backtest_holdout is None or backtest_holdout.empty:
        return core
    period = configuration.period_col
    exam = core["s1_rol"] == ROLE_TEST
    months = core[period].astype(str)
    for h in EXAM_HORIZONS:
        at_h = backtest_holdout[backtest_holdout["h"] == h]
        lookup = dict(zip(zip(at_h["id_estimacion"], at_h["mes_objetivo"].astype(str)), at_h["tasa_pred"]))
        rate = pd.Series([lookup.get((pool, month), np.nan) for pool, month in zip(core["s3_id_estimacion"], months)], index=core.index)
        core[f"s6_examen_tasa_h{h}"] = rate.where(exam)
        core[f"s6_examen_unidades_h{h}"] = (core["s1_pipeline_unidades"] * rate).where(exam)
    return core


def add_step7_maturation(core: pd.DataFrame, signal_adjustment: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """s7_: the adjustment of a cell × month (all signals) spread over the neutral real rows
    of that cell and month in proportion to their expected $."""
    period = configuration.period_col
    core["s7_ajuste_senales_usd"] = 0.0
    if "s6_usd_esperado" not in core.columns:                 # no future rows: nothing to adjust
        core["s7_usd_esperado_ajustado"] = np.nan
        return core
    if signal_adjustment is not None and len(signal_adjustment):
        cell = join_columns(core, list(configuration.business_mandatory_dims) + list(configuration.extra_renovacion))
        flags = [f for f in configuration.structural_timevarying_dims if f in core.columns]
        neutral = ~pd.concat([core[f].isin(configuration.timevarying_positive_values) for f in flags], axis=1).any(axis=1) if flags else pd.Series(True, index=core.index)
        eligible = neutral & (core["s6_origen_pipeline"] == PIPELINE_REAL) & core["s6_usd_esperado"].notna()
        per_cell_month = signal_adjustment.groupby(["celda_comp", signal_adjustment[period].astype(str)])["ajuste_usd"].sum()
        keys = pd.Series(list(zip(cell, core[period].astype(str))), index=core.index)
        weight = core["s6_usd_esperado"].where(eligible, 0.0)
        weight_total = weight.groupby(keys).transform("sum")
        adjustment = keys.map(per_cell_month).fillna(0.0)
        core["s7_ajuste_senales_usd"] = np.where(weight_total > 0, adjustment * weight / weight_total.replace(0, np.nan), 0.0)
        core["s7_ajuste_senales_usd"] = core["s7_ajuste_senales_usd"].fillna(0.0)
    core["s7_usd_esperado_ajustado"] = core["s6_usd_esperado"] + core["s7_ajuste_senales_usd"].where(core["s6_usd_esperado"].notna())
    return core


def build_core_table(results: dict, configuration: Config) -> pd.DataFrame:
    """The core table, rows and columns, in the order of the steps."""
    from answers import common_band_per_row
    period = configuration.period_col
    fine_table, units = results["fine_table"], results["units"]
    forecast, decisions = results["forecast"], results["decisions"]
    parts = [raw_rows(fine_table, configuration), gap_rows(units, configuration), extended_rows(forecast.get("forecast_units_extended"), configuration)]
    core = pd.concat([p for p in parts if len(p)], ignore_index=True)
    core = add_step1_unit_columns(core, units)
    core = add_step2_series(core, configuration, results["series_card"])
    core = add_step3_ladder(core, results["series_card"], results["parent_ladder"])
    core = add_step4_technique(core, decisions)
    detail = forecast["forecast_detail"]
    common_low, common_high = common_band_per_row(detail, decisions.get("decision_aggregate_bands"), configuration) if len(detail) else (np.array([]), np.array([]))
    core = add_steps5_6_forecast(core, detail, forecast["forecast_bands"], common_low, common_high, configuration)
    core = add_step6_exam(core, (results.get("backtest") or {}).get("backtest_holdout"), configuration)
    maturation = results.get("signal_maturation", {}) or {}
    core = add_step7_maturation(core, maturation.get("signal_adjustment"), configuration)
    core[period] = core[period].astype(str)
    core["anio"] = core[period].str[:4].astype(int)
    helper = [c for c in ("fu_comb_key", "fu_id") if c in core.columns]
    leading = ["paso_alta", "s0_id_fila", period, "anio"] + [c for c in dimension_columns(configuration) if c in core.columns]
    ordered = leading + [c for step in ("s0_", "s1_", "s2_", "s3_", "s4_", "s5_", "s6_", "s7_") for c in core.columns
                         if c.startswith(step) and c not in leading]
    return core[ordered + helper].drop(columns=helper)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE ONE AGGREGATION · rates on the fly
# ═══════════════════════════════════════════════════════════════════════════════════

ADDITIVE = ["s0_pipeline_unidades", "s0_pipeline_usd", "s0_renovados_unidades", "s0_renovados_usd",
            "s1_pipeline_unidades", "s1_pipeline_usd", "s1_renovados_unidades", "s1_renovados_usd",
            "s6_pipeline_unidades", "s6_pipeline_usd", "s6_unidades_esperadas", "s6_usd_esperado_precio_pipeline", "s6_usd_esperado",
            "s6_banda_comun_low_usd", "s6_banda_comun_high_usd", "s6_examen_unidades_h1", "s6_examen_unidades_h6",
            "s7_ajuste_senales_usd", "s7_usd_esperado_ajustado"]


def agregar(core: pd.DataFrame, dims: list, filtro: pd.Series = None) -> pd.DataFrame:
    """Every study is this call with a different set of dimensions.

    INPUT:   core · dims — the columns that define the aggregation (empty = the total) ·
             filtro — an optional boolean mask over the rows.
    OUTPUT:  one row per group: the additive columns summed, plus the rates of the
             aggregation computed as ratios of sums:
               tasa_real      Σ renovados / Σ pipeline over the CLOSED months (entrenamiento, examen)
               tasa_prevista  Σ unidades esperadas / Σ pipeline of the forecast
               uplift         Σ $ esperado / Σ $ esperado a precio de pipeline
               tasa_examen_h1 / _h6  Σ unidades previstas / Σ pipeline over the EXAM months
               tasa_real_examen      Σ renovados / Σ pipeline over the exam months
    RULES:   a rate is NaN when its denominator is 0 in the group. Nothing here is a
             model: it is arithmetic over the core, identical in Power BI.
    """
    frame = core if filtro is None else core[filtro]
    frame = frame.copy()
    closed = frame["s1_rol"].isin(TRUTH_ROLES)
    exam = frame["s1_rol"] == ROLE_TEST
    frame["_pipe_cerrado"] = frame["s1_pipeline_unidades"].where(closed, 0.0)
    frame["_ren_cerrado"] = frame["s1_renovados_unidades"].where(closed, 0.0)
    frame["_pipe_examen"] = frame["s1_pipeline_unidades"].where(exam, 0.0)
    frame["_ren_examen"] = frame["s1_renovados_unidades"].where(exam, 0.0)
    for h in EXAM_HORIZONS:
        frame[f"_pipe_examen_h{h}"] = frame["s1_pipeline_unidades"].where(exam & frame.get(f"s6_examen_tasa_h{h}", pd.Series(np.nan, index=frame.index)).notna(), 0.0)
    additive = [c for c in ADDITIVE if c in frame.columns] + ["_pipe_cerrado", "_ren_cerrado", "_pipe_examen", "_ren_examen"] + [f"_pipe_examen_h{h}" for h in EXAM_HORIZONS]
    if dims:
        grouped = frame.groupby(dims, dropna=False)[additive].sum(min_count=1).reset_index()
    else:
        grouped = frame[additive].sum(min_count=1).to_frame().T
    def ratio(numerator, denominator):
        return grouped[numerator] / grouped[denominator].where(grouped[denominator] > 0)
    grouped["tasa_real"] = ratio("_ren_cerrado", "_pipe_cerrado")
    grouped["tasa_real_examen"] = ratio("_ren_examen", "_pipe_examen")
    if "s6_unidades_esperadas" in grouped.columns:
        grouped["tasa_prevista"] = ratio("s6_unidades_esperadas", "s6_pipeline_unidades")
        grouped["uplift"] = ratio("s6_usd_esperado", "s6_usd_esperado_precio_pipeline")
    for h in EXAM_HORIZONS:
        if f"s6_examen_unidades_h{h}" in grouped.columns:
            grouped[f"tasa_examen_h{h}"] = ratio(f"s6_examen_unidades_h{h}", f"_pipe_examen_h{h}")
    return grouped.rename(columns={"_pipe_cerrado": "pipeline_cerrado_unidades", "_ren_cerrado": "renovados_cerrado_unidades",
                                   "_pipe_examen": "pipeline_examen_unidades", "_ren_examen": "renovados_examen_unidades"}).drop(
        columns=[f"_pipe_examen_h{h}" for h in EXAM_HORIZONS])


def banda_agregada(core: pd.DataFrame, dims: list, period_column: str = "period") -> pd.DataFrame:
    """The band of any aggregation of the forecast, with the rules of the framework: the
    rows of one pool and month miss together (their bands add linearly); pools and months
    miss independently (quadrature); the common part adds linearly. total = √(idio² + common²).
    OUTPUT: dims + banda_idio_low/high_usd, banda_comun_low/high_usd, banda_total_low/high_usd."""
    if "s6_usd_esperado" not in core.columns:
        return pd.DataFrame(columns=list(dims) + ["banda_idio_low_usd", "banda_idio_high_usd", "banda_comun_low_usd", "banda_comun_high_usd", "banda_total_low_usd", "banda_total_high_usd"])
    future = core[core["s6_usd_esperado"].notna()]
    keys = list(dims) + ["s3_id_estimacion", period_column]
    per_pool_month = future.groupby(keys, dropna=False)[["s6_banda_low_usd", "s6_banda_high_usd"]].sum().reset_index()
    per_pool_month["low2"], per_pool_month["high2"] = per_pool_month["s6_banda_low_usd"] ** 2, per_pool_month["s6_banda_high_usd"] ** 2
    if dims:
        idio = per_pool_month.groupby(list(dims), dropna=False)[["low2", "high2"]].sum()
        common = future.groupby(list(dims), dropna=False)[["s6_banda_comun_low_usd", "s6_banda_comun_high_usd"]].sum()
        out = idio.join(common).reset_index()
    else:
        out = pd.DataFrame([dict(low2=per_pool_month["low2"].sum(), high2=per_pool_month["high2"].sum(),
                                 s6_banda_comun_low_usd=future["s6_banda_comun_low_usd"].sum(), s6_banda_comun_high_usd=future["s6_banda_comun_high_usd"].sum())])
    out["banda_idio_low_usd"], out["banda_idio_high_usd"] = -np.sqrt(out["low2"]), np.sqrt(out["high2"])
    out["banda_comun_low_usd"], out["banda_comun_high_usd"] = out["s6_banda_comun_low_usd"], out["s6_banda_comun_high_usd"]
    out["banda_total_low_usd"] = -np.sqrt(out["low2"] + out["banda_comun_low_usd"] ** 2)
    out["banda_total_high_usd"] = np.sqrt(out["high2"] + out["banda_comun_high_usd"] ** 2)
    return out.drop(columns=["low2", "high2", "s6_banda_comun_low_usd", "s6_banda_comun_high_usd"])


# ═══════════════════════════════════════════════════════════════════════════════════
# RECONCILIATION AND THE STORY BY STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def reconciliar(core: pd.DataFrame, raw_totals: dict) -> pd.DataFrame:
    """Licence by licence and dollar by dollar against the raw: Σ s0 and Σ s1 (raw rows +
    explicit gaps) must equal the raw totals exactly; the gaps add 0."""
    checks = []
    for measure, total in raw_totals.items():
        s0 = float(core[f"s0_{measure}"].sum()) if f"s0_{measure}" in core.columns else np.nan
        gaps = float(core.loc[core["paso_alta"] == ROW_GAP, f"s1_{measure}"].sum()) if f"s1_{measure}" in core.columns else 0.0
        checks.append(dict(medida=measure, raw=total, s0=s0, huecos=gaps, cuadra=bool(abs(s0 - total) < 1e-6 and abs(gaps) < 1e-12)))
    s1 = float(core["s1_pipeline_usd"].sum())
    checks.append(dict(medida="pipeline_usd (s1, raw + huecos)", raw=raw_totals.get("pipeline_usd", np.nan), s0=s1, huecos=0.0,
                       cuadra=bool(abs(s1 - raw_totals.get("pipeline_usd", np.nan)) < 1e-6)))
    return pd.DataFrame(checks)


def print_core_story(core: pd.DataFrame, reconciliation: pd.DataFrame, configuration: Config) -> None:
    """The story of the core table, one line per step."""
    period = configuration.period_col
    if "s6_pipeline_usd" not in core.columns:
        core = core.assign(s6_pipeline_usd=np.nan)
    by_origin = core.groupby("paso_alta").agg(filas=(period, "size"), pipeline_usd_s1=("s1_pipeline_usd", "sum"), pipeline_usd_s6=("s6_pipeline_usd", "sum"))
    print(f"[núcleo] one table, {len(core):,} rows × {core.shape[1]} columns (steps s0 → s7, left to right)")
    for origin, row in by_origin.iterrows():
        print(f"   {origin:<26} {int(row['filas']):>9,} rows · pipeline in the raw ${row['pipeline_usd_s1']:>14,.0f} · pipeline the forecast renews ${row['pipeline_usd_s6']:>14,.0f}")
    ok = bool(reconciliation["cuadra"].all())
    print(f"   reconciliation with the raw: {'every measure matches, licence by licence and dollar by dollar' if ok else 'MISMATCH: ' + str(reconciliation[~reconciliation['cuadra']]['medida'].tolist())}")
    by_role = agregar(core, ["s1_rol"])
    print("   by role (s1): " + " · ".join(f"{r['s1_rol']}: ${r['s1_pipeline_usd']:,.0f}" + (f", tasa real {100 * r['tasa_real']:.1f}%" if pd.notna(r['tasa_real']) else "")
                                            for _, r in by_role.dropna(subset=["s1_rol"]).iterrows()))
    rungs = [c for c in core.columns if c.startswith("s3_id_peldano_")]
    print(f"   ladder in the table: {len(rungs)} rung columns ({', '.join(rungs)}) + s3_id_estimacion · "
          f"{core['s3_id_estimacion'].nunique():,} pools chosen")
    explain(configuration,
            "The core is the whole story: a step adds columns (s0 the raw, s1 the conditioned raw with explicit gaps, s2 series, s3 ladder, s4 technique, s5 uplift, s6 forecast, s7 maturation) and, sometimes, rows (paso_alta).",
            "Every study is an aggregation: choose the columns, sum the additive ones, divide. Grouping by s3_id_peldano_k reproduces the rung-k pools exactly.",
            "Gap rows state 'nothing was due' explicitly with measures 0: the table still matches the raw licence by licence and dollar by dollar.")


def run_core_table(results: dict, configuration: Config, raw_totals: dict) -> dict:
    """Build, reconcile, persist and print. Returns the core and its reconciliation."""
    core = build_core_table(results, configuration)
    reconciliation = reconciliar(core, raw_totals)
    configuration.write(core, "nucleo")
    configuration.write(reconciliation, "nucleo_reconciliacion")
    print_core_story(core, reconciliation, configuration)
    return dict(nucleo=core, reconciliacion=reconciliation)
