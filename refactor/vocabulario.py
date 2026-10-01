"""
vocabulario.py — The labels the framework persists (Spanish: they are the contract of
the tables, of the report and of the BI). Grows with the steps that need a new label.
"""

# ─── the role of every month (generated from the calendar of the configuration) ──
ROLE_TRAIN = "entrenamiento"          # closed months the forecast learns from
ROLE_TEST = "examen"                  # closed months that only evaluate
ROLE_PROJECTION = "proyeccion"        # the current month and the future

# ─── the columns step 02 adds to the raw ─────────────────────────────────────────
CALENDAR_ROLE_COLUMN = "rol"                          # the role of the row's month (the three above)
CURRENT_MONTH_COLUMN = "es_mes_en_curso"              # 1 in the current month, 0 elsewhere
S0_RENEWED_UNITS_COLUMN = "s0_renovados_unidades"     # the raw's renewed units, before step 02 touches them
S0_RENEWED_USD_COLUMN = "s0_renovados_usd"            # the raw's renewed USD, before step 02 touches them
S0_PIPELINE_UNITS_COLUMN = "s0_vencen_unidades"   # the pipeline as the raw had it (step 02 wipes what is not known yet)
S0_PIPELINE_USD_COLUMN = "s0_vencen_usd"

# ─── the ids and keys step 03 adds (kept in English: the BI joins on these names) ─
SERIES_ID_COLUMN = "fs_id"                   # the rate series: mandatory | timevarying | extra_renovacion
UNIT_ID_COLUMN = "fu_id"                     # the forecast unit: fs_id | month (where the RATE is predicted)
UPLIFT_CELL_ID_COLUMN = "uplift_cell_id"     # the price context: uplift mandatory | extra_revalorizacion
SERIES_KEY_COLUMN = "fs_key"
UNIT_KEY_COLUMN = "fu_key"
UPLIFT_CELL_KEY_COLUMN = "uplift_cell_key"
ROW_KEY_COLUMN = "fila_key"                  # one key per fine row: fu_id || extra_revalorizacion values

# ─── the tables (logical names; the Config adds the prefix) ──────────────────────
TABLE_CALENDAR = "calendario"                # step 02: one row per month, its role and its money
TABLE_FINE = "fact_fine"                     # step 03: every raw row with its ids and keys

# The four roles in time order (coverage patterns and tables follow it).
ROLES_IN_ORDER = [ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION]

# ─── the columns steps 04 and 06 add ─────────────────────────────────────────────
FINE_ROWS_COLUMN = "n_filas_finas"           # step 04: fine rows added into a forecast unit
UNIVERSE_COLUMN = "universo"                 # step 06: normal · serie_temporal · mixto
COVERAGE_COLUMN = "cobertura"                # step 06: the roles a series has, in time order, joined with "+"
ROUTE_COLUMN = "ruta"                        # step 06: how the series will be treated

# ─── the universe of a series (from the time_series flag of its units) ───────────
UNIVERSE_NORMAL = "normal"                   # no unit flagged
UNIVERSE_TIME_SERIES = "serie_temporal"      # every unit flagged
UNIVERSE_MIXED = "mixto"                     # some units flagged and some not (worth a look)

# ─── the route of a series (from its coverage) ───────────────────────────────────
ROUTE_PREDICTABLE = "predecible"             # closed months AND something to predict: its rate is estimated
ROUTE_HISTORY_ONLY = "solo_historia"         # nothing to predict: kept, it lends history to its relatives
ROUTE_FUTURE_ONLY = "solo_futuro"            # nothing to learn from: predicted from its relatives

# ─── more tables ─────────────────────────────────────────────────────────────────
TABLE_UNITS = "fact_fu"                      # step 04: one row per forecast unit
TABLE_LOOKUP_SERIES = "lookup_fs"            # step 05: fs_id ↔ fs_key and the series columns
TABLE_LOOKUP_UNITS = "lookup_fu"             # step 05: fu_id ↔ fu_key, its series and its month
TABLE_LOOKUP_UPLIFT_CELLS = "lookup_uplift_cell"   # step 05: uplift_cell_id ↔ uplift_cell_key and its columns
TABLE_SERIES = "series"                      # step 06: one row per series, its coverage and route

# The months that are truth: closed and used to learn and to evaluate.
TRUTH_ROLES = (ROLE_TRAIN, ROLE_TEST)

# ─── the sign of a series (from its ACTIVE timevarying flags) ─────────────────────
SIGN_NEUTRAL = "neutro"                      # no flag active
SIGN_NEGATIVE = "negativo"                   # only flags that rotate towards churn
SIGN_POSITIVE = "positivo"                   # only flags that rotate towards renewal
SIGN_MIXED = "mixto"                         # both kinds active: never pooled with the others

# ─── the columns steps 07 and 08 add ─────────────────────────────────────────────
RATE_COLUMN = "tasa"                         # step 08: renewed / due units of a unit; null = no information
SYNTHETIC_COLUMN = "sintetica"               # step 08: 1 on a gap row (a month with no unit inside the history)
SIGN_COLUMN = "signo"                        # step 08: the sign of the series

# ─── the tables of steps 07 and 08 ───────────────────────────────────────────────
TABLE_UNIT_SUPPORT = "fu_soporte"            # step 07: worst-case binomial bound per forecast unit
TABLE_GAPS = "fu_huecos"                     # step 08: the gap rows added inside the history of a series
TABLE_SERIES_RATE = "series_tasa"            # step 08: one row per series: support, own rate, error, sign

# ─── the tables of step 09 ───────────────────────────────────────────────────────
TABLE_DIMENSIONS = "decision_eta2"           # step 09: how much each dimension separates the rate; collapse order
TABLE_DIMENSION_PAIRS = "decision_eta2_pares"   # step 09: the pairs of dimensions with the most interaction

# ─── the ladder (steps 10 and 11) ────────────────────────────────────────────────
SIGN_TOKEN = "SIG="                          # inside a relative's pattern: the timevarying block summarised as its sign
WILDCARD = "*"                               # inside a relative's pattern: a dimension collapsed or annulled
COMPOSITION_ID_COLUMN = "composition_id"     # the composition of a forecast series: its group after the 3 merging stages of
                                             # the ladder (step 10); it is what is predicted, and it lends its rate

# The risk level of a series: how its rate is estimated, from best to worst.
LEVEL_OWN = "A_propio"                        # its own support is precise (≥ own_rate_floor) and it has a full year
LEVEL_OWN_SHORT = "A2_propio_corto"           # precise, but less than a year of history
LEVEL_OWN_REINFORCED = "A3_propio_reforzado"  # evidence but not precision: its rate blended with its first pool
LEVEL_BORROWED = "B_prestado"                 # below the floor; a close relative (same mandatory dims)
LEVEL_FAR = "C_lejano"                        # below the floor; the mandatory cell or a collapsed dimension
LEVEL_SIGNED_UNDER_FLOOR = "S_signo_bajo_suelo"   # signed, and no relative of its sign reaches the floor
LEVEL_MIXED = "M_signo_mixto"                 # flags of both signs: never pooled
LEVEL_NO_HISTORY = "D_sin_historia"           # nothing to learn from (solo_futuro)
LEVEL_NO_IMPACT = "N_sin_impacto"             # nothing to predict (solo_historia)
LEVEL_TIME_SERIES = "T_universo_ts"           # the time_series universe: treated apart

# ─── the tables of steps 10 and 11 ───────────────────────────────────────────────
TABLE_LADDER_STEPS = "ladder_steps"           # step 10: every series × pass: its group id and the group's support
TABLE_LADDER_SUMMARY = "ladder_summary"       # step 10: every pass: groups, support, money in groups that reach the floor
TABLE_LADDER_GROUPS = "ladder_groups"         # step 10: every series: its final group and its credibility reference
TABLE_SERIES_ESTIMATE = "series_estimacion"   # step 11: every series: its chosen relative, rate, errors, level
TABLE_RISK_LEVELS = "niveles_riesgo"          # step 11: money by risk level

# ─── the core table (built last, grows with every step) ──────────────────────────
TABLE_CORE = "nucleo"                         # ONE wide table at the fine grain: history and gaps (the future later)
TABLE_CORE_LEGEND = "nucleo_leyenda"          # every column of the core: its step, its level and how to aggregate it
ROW_ORIGIN_COLUMN = "origen_fila"             # where the row comes from
ROW_FROM_RAW = "raw"                          # a row of the extract
ROW_FROM_GAP = "hueco"                        # a month with no expirations inside the history of a series (step 08)

# ─── steps 12 and 14: the pool series and the backtest ───────────────────────────
GATE_SUPPORT = "soporte"                      # a pool below the support floor: not judged (it takes the challenger)
GATE_LEVEL = "nivel"                          # a pool with support: judged by the backtest
PURPOSE_SELECTION = "seleccion"               # a target month used to CHOOSE the technique
PURPOSE_EXAM = "examen"                       # a target month used only to MEASURE the chosen technique
CHAMPION_ORIGIN = "campeon"                   # the technique beat the challenger by the margin
CHALLENGER_ORIGIN = "retador"                 # nobody beat it: the challenger stays

TABLE_POOL_SERIES = "pool_serie"              # step 12: the monthly series of every estimation id
TABLE_POOL_REFERENCE = "pool_referencia"      # step 12: one row per estimation id: months, support, rate, gate
TABLE_TECHNIQUES = "dim_tecnica"              # step 14: the catalogue of techniques
TABLE_BACKTEST_PREDICTIONS = "backtest_predicciones"   # step 14: one row per id × target × horizon × technique
TABLE_TECHNIQUE_DECISION = "decision_tecnica"          # step 14: the chosen technique per id and horizon band
TABLE_ERROR_BANDS = "decision_bandas"                  # step 14: error quantiles per technique and horizon
TABLE_EXAM_BY_POOL = "backtest_examen"                 # step 14: the chosen technique vs the challenger in the exam, per id
TABLE_EXAM_TOTAL = "backtest_examen_total"             # step 14: the error of the TOTAL renewals in every exam month

# ─── step 13: the dynamics of the rate ───────────────────────────────────────────
TABLE_PORTFOLIO_SEASONALITY = "estacionalidad_cartera"   # step 13: the month effect of the whole portfolio
TABLE_POOL_DYNAMICS = "composition_dynamics"   # step 13: φ, trend and seasonality of every composition

# ─── the report ──────────────────────────────────────────────────────────────────
TABLE_SERIES_CARD = "ficha_serie"            # one row per series: every attribute the framework knows about it
REPORT_FILE_NAME = "informe_sff.md"          # the report, in the output folder

# ─── steps 15-18: uplift, forecast, validation ───────────────────────────────────
UPLIFT_OWN, UPLIFT_PARENT, UPLIFT_CELL, UPLIFT_GLOBAL = "propia", "padre", "celda", "global"   # where a cell's uplift comes from
PATH_STATISTICAL = "estadistica"             # uplift observed in the past renewals of the cell
PATH_CONTRACT = "contrato"                   # uplift of the contract: 1 / (1 − discount)
RATE_FROM_POOL = "pool"                      # the rate comes from the technique chosen for the series' estimation id
RATE_FROM_CELL = "celda_mandatory"           # no pool: the rate of its mandatory cell
RATE_FROM_GLOBAL = "global"                  # nothing else: the rate of the whole portfolio

TABLE_UPLIFT_CELLS = "uplift_celda"          # step 15: the uplift of every uplift cell
TABLE_CONTRACT_CHECK = "uplift_contrato_check"   # step 15: the contract rule against the past renewals
TABLE_UPLIFT_BACKTEST = "backtest_uplift"    # step 16: statistical vs contract in the exam months
TABLE_FORECAST = "forecast"                  # step 17: every future fine row with its forecast and bands
TABLE_FORECAST_MONTH = "forecast_mes"        # step 17: the total by month with its bands
TABLE_BUSINESS_SUMMARY = "resumen_negocio"   # step 17: the answers by year
TABLE_VALIDATION = "validacion"              # step 18: the final checks across steps

# ─── step 19: the exam of the portfolio (framework vs the spreadsheet) ───────────
TABLE_PORTFOLIO_EXAM = "examen_cartera"               # per exam month and horizon: real vs framework vs spreadsheet
TABLE_PORTFOLIO_EXAM_SUMMARY = "examen_cartera_resumen"   # per method and horizon: the mean error of the total and by series
METHOD_FRAMEWORK = "framework"

# ─── the extended horizon (step 17) ──────────────────────────────────────────────
PIPELINE_ORIGIN_COLUMN = "origen_pipeline"   # where the pipeline of a future row comes from
PIPELINE_REAL = "real"                       # in the extract (known)
PIPELINE_PROJECTED = "proyectada"            # an expected renewal of the simulation window, due 12 months later
PIPELINE_SIMULATED = "simulada"              # an acquisition simulated in the simulation window, due 12 months later

# ─── step 20: the time_series universe (retail to subscription) and the total ────
TS_REAL = "ts_real"                          # closed months of the year: real conversions (revenue of the year)
TS_PROJECTED = "ts_proyectado"               # current month → ts_projection_end: simulated conversions (revenue of the year)
TS_REENTRY = "ts_reentrada"                  # a projected month falling due 12 months later, renewed at the region's rate
ORIGIN_PIPELINE_RENEWED = "pipeline_renovado_real"     # renewals already booked in the closed months (normal universe)
ORIGIN_PIPELINE_EXPECTED = "pipeline_esperado"         # expected renewals of the extract's future pipeline
ORIGIN_EXTENDED_PROJECTED = "extendido_reentrada"      # expected renewals of the extended horizon's re-entries
ORIGIN_EXTENDED_SIMULATED = "extendido_captacion"      # expected renewals of the extended horizon's simulated acquisition
ORIGIN_TOTAL = "TOTAL"
TABLE_TIME_SERIES = "time_series"            # step 20: one row per region × month of the time_series universe
TABLE_FORECAST_TOTAL = "forecast_total"      # step 20: year × origin, units and USD, and the total of every year

# ─── step 20: the time_series universe (retail to subscription) and the forecast total ──
TS_REAL = "ts_real"                          # revenue of the time_series universe in the closed months of the year
TS_PROJECTED = "ts_proyectado"               # simulated revenue from the current month to ts_projection_end
TS_REENTRY = "ts_reentrada"                  # the projected months falling due again a year later, as pipeline
TOTAL_ORIGIN_RENEWED = "pipeline_renovado_real"      # renewals already booked in the closed months (normal universe)
TOTAL_ORIGIN_EXPECTED = "pipeline_real_esperado"     # expected renewals of the extract's future pipeline
TOTAL_ORIGIN_PROJECTED = "pipeline_proyectada"       # expected renewals of the extended re-entries
TOTAL_ORIGIN_SIMULATED = "pipeline_simulada"         # expected renewals of the simulated acquisition
TOTAL_ORIGIN_TOTAL = "TOTAL"
TABLE_TIME_SERIES = "time_series"            # step 20: region × month: origin, units, value, level, AUV, discount, rate
TABLE_FORECAST_TOTAL = "forecast_total"      # step 20: year × origin and the total of every year

# ─── step AUD: the audit tables (satellites of the core, for the drill-down) ───────
TABLE_COMPOSITION = "composition"                        # every id of every stage: who forms it, support, rate, prediction
TABLE_COMPOSITION_TECHNIQUES = "composition_techniques"  # composition × band × technique: status, errors, rank, chosen
TABLE_CREDIBILITY = "credibility"                        # every credibility reference: support, rate, k and its parts
TABLE_CREDIBILITY_MEMBERS = "credibility_members"        # reference × forecast series: what goes into its rate
TABLE_SERIES_DYNAMICS = "series_dynamics"                # every forecast series: φ, trend, seasonality vs its composition
TABLE_SERIES_BACKTEST = "series_backtest"                # forecast series × exam month × horizon × technique
TABLE_SERIES_TECHNIQUE_SUMMARY = "series_technique_summary"   # forecast series × band × technique: errors, rank, chosen
TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
TABLE_DIMENSION_LEVELS = "dimension_levels"              # step 02b: the generated groups of every leveled dimension
TABLE_LADDER_MERGES = "ladder_merges"                    # step 10: every merge of every series, and whether it improves
TABLE_COMPOSITION_MEMBERS = "composition_members"        # step 10: every series in the rate of every composition (uses / lends)
TABLE_SERIES_EXAM = "series_exam"                        # step 19: every forecast series: raw and framework in the exam
TABLE_SERIES_EXAM_DETAIL = "series_exam_detail"          # step 19: series × exam month × h × method, every prediction traced
TECHNIQUE_TESTED = "tested"
TECHNIQUE_NOT_ENOUGH_HISTORY = "not_enough_history"
TECHNIQUE_COMPOSITION_BELOW_FLOOR = "composition_below_floor"
