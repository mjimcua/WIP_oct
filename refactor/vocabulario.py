"""
vocabulario.py — The labels the framework persists (Spanish: they are the contract of
the tables, of the report and of the BI). Grows with the steps that need a new label.
"""

# ─── the role of every month (generated from the calendar of the configuration) ──
ROLE_TRAIN = "entrenamiento"          # closed months the forecast learns from
ROLE_TEST = "examen"                  # closed months that only evaluate
ROLE_PENDING = "pendiente_cierre"     # not closed yet: neither learn nor evaluate
ROLE_PROJECTION = "proyeccion"        # the current month and the future

# ─── the columns step 02 adds to the raw ─────────────────────────────────────────
CALENDAR_ROLE_COLUMN = "rol"                          # the role of the row's month (the four above)
CURRENT_MONTH_COLUMN = "es_mes_en_curso"              # 1 in the current month, 0 elsewhere
S0_RENEWED_UNITS_COLUMN = "s0_renovados_unidades"     # the raw's renewed units, before step 02 touches them
S0_RENEWED_USD_COLUMN = "s0_renovados_usd"            # the raw's renewed USD, before step 02 touches them

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
ROLES_IN_ORDER = [ROLE_TRAIN, ROLE_TEST, ROLE_PENDING, ROLE_PROJECTION]

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
RUNG_COLUMN = "peldano"                      # 0 = the series itself; the higher, the farther the relative
PATTERN_COLUMN = "patron"                    # the id of a relative: the pool of every series that matches it
ESTIMATION_ID_COLUMN = "id_estimacion"       # the pattern of the relative a series takes its rate from

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
TABLE_RELATIVES = "parientes"                 # step 10: every series × rung: its relative's pattern
TABLE_POOLS = "pools"                         # step 10: every pattern: its support and its rate
TABLE_SERIES_ESTIMATE = "series_estimacion"   # step 11: every series: its chosen relative, rate, errors, level
TABLE_RISK_LEVELS = "niveles_riesgo"          # step 11: money by risk level

# ─── the core table (built last, grows with every step) ──────────────────────────
TABLE_CORE = "nucleo"                         # ONE wide table at the fine grain: history and gaps (the future later)
TABLE_CORE_LEGEND = "nucleo_leyenda"          # every column of the core: its step, its level and how to aggregate it
ROW_ORIGIN_COLUMN = "origen_fila"             # where the row comes from
ROW_FROM_RAW = "raw"                          # a row of the extract
ROW_FROM_GAP = "hueco"                        # a month with no expirations inside the history of a series (step 08)
