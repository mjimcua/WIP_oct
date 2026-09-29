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
