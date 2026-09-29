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
UNIT_ID_COLUMN = "fu_id"                     # the forecast unit: fs_id | month
COMBINATION_ID_COLUMN = "comb_id"            # the revaluation combination: extra_revalorizacion joined
SERIES_KEY_COLUMN = "fs_key"
UNIT_KEY_COLUMN = "fu_key"
COMBINATION_KEY_COLUMN = "comb_key"
UNIT_COMBINATION_KEY_COLUMN = "fu_comb_key"  # one key per fine row: the BI joins on it

# ─── the tables (logical names; the Config adds the prefix) ──────────────────────
TABLE_CALENDAR = "calendario"                # step 02: one row per month, its role and its money
TABLE_FINE = "fact_fine"                     # step 03: every raw row with its ids and keys
