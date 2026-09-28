"""vocabulario.py — SFF v3 · the ONE place where the persisted labels live.

Every value that reaches a table (a role, a sign, a treatment, a pipeline origin, a
risk level) is defined here, in Spanish, and imported by every module. The raw's own
contract values (`train`, `test`, `projection` in `dataset_role`) are mapped to this
vocabulary in phase 0 and never appear downstream.

Why one module: the same label was defined in six files, and a mixed vocabulary
(`trainable` next to `peldano`) crept into the tables. A label changed here changes
everywhere, tests included.
"""

# ─── roles of a month (the calendar decided from the current month) ──────────────
ROLE_TRAIN = "entrenamiento"        # closed month, used to learn and to decide
ROLE_TEST = "examen"                # closed month, used only to evaluate (and to learn)
ROLE_PENDING = "pendiente_cierre"   # the month before the current one: not closed, not truth
ROLE_PROJECTION = "proyeccion"      # the current month and later: the future
TRUTH_ROLES = (ROLE_TRAIN, ROLE_TEST)              # the only months that are truth
FUTURE_ROLES = (ROLE_PROJECTION, ROLE_PENDING)     # the months that get a forecast

# the raw's own contract values → our vocabulary (used only when no current-month flag exists)
RAW_ROLE_MAP = {"train": ROLE_TRAIN, "test": ROLE_TEST, "projection": ROLE_PROJECTION, "pending_close": ROLE_PENDING,
                ROLE_TRAIN: ROLE_TRAIN, ROLE_TEST: ROLE_TEST, ROLE_PROJECTION: ROLE_PROJECTION, ROLE_PENDING: ROLE_PENDING}

# ─── universes ───────────────────────────────────────────────────────────────────
UNIVERSE_NORMAL = "normal"
UNIVERSE_TIME_SERIES = "serie_temporal"

# ─── treatment of a series (what the framework does with it) ─────────────────────
TREATMENT_PREDICTABLE = "predecible"      # closed months AND something to predict: a rate is estimated
TREATMENT_HISTORY_ONLY = "solo_historia"  # nothing to predict: kept for the pools and the backtest
TREATMENT_FUTURE_ONLY = "solo_futuro"     # nothing to learn from: predicted by the cascade

# ─── sign of a series (from its timevarying flags) ───────────────────────────────
SIGN_NEUTRAL = "neutro"
SIGN_NEGATIVE = "negativo"
SIGN_POSITIVE = "positivo"
SIGN_MIXED = "mixto"
SIGN_TOKEN = "SIG="                       # the token that carries the sign inside an estimation id

# ─── origin of a pipeline row of the forecast ────────────────────────────────────
PIPELINE_REAL = "real"
PIPELINE_PROJECTED = "proyectada"
PIPELINE_SIMULATED = "simulada"

# ─── origin of a rate / technique / band / uplift ────────────────────────────────
ORIGIN_SERIES, ORIGIN_CELL, ORIGIN_GLOBAL = "serie", "celda", "global"
CHAMPION_ORIGIN, CHALLENGER_ORIGIN, DEFAULT_ORIGIN = "campeon", "retador", "defecto"
BAND_OWN, BAND_FAMILY, BAND_BINOMIAL = "propia", "familia", "binomial"
UPLIFT_VIA_CONTRACT, UPLIFT_VIA_STATISTICAL = "contrato", "estadistico"    # how a row's uplift was set

# ─── which step created a row of the core table (column `paso_alta`) ─────────────
ROW_FROM_RAW = "s0_raw"                          # a row of the extract, as it came
ROW_GAP = "s1_hueco"                             # a month of a series with nothing due: made explicit, measures 0
ROW_PROJECTED_REENTRY = "s6_reentrada_proyectada" # a renewal we forecast, falling due again one term later
ROW_SIMULATED_ACQUISITION = "s6_adquisicion_simulada"  # acquisition of a future month at the historical pace
GAP_COMBINATION = "hueco"                        # comb_id of a gap row (it has no revaluation extras)

# ─── risk levels ─────────────────────────────────────────────────────────────────
LEVEL_OWN = "A_propio"
LEVEL_OWN_SHORT = "A2_propio_corto"
LEVEL_OWN_REINFORCED = "A3_propio_reforzado"
LEVEL_BORROWED = "B_prestado"
LEVEL_FAR = "C_lejano"
LEVEL_SIGNED_UNDER_FLOOR = "S_signo_bajo_suelo"
LEVEL_MIXED = "M_signo_mixto"
LEVEL_NO_HISTORY = "D_sin_historia"
LEVEL_NO_IMPACT = "N_sin_impacto"
LEVEL_TIME_SERIES = "T_universo_ts"
