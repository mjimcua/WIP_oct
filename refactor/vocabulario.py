"""
vocabulario.py — The labels the framework persists (Spanish: they are the contract of
the tables, of the report and of the BI). Grows with the steps that need a new label.
"""

# ─── the role of every month (generated from the calendar of the configuration) ──
ROLE_TRAIN = "entrenamiento"          # closed months the forecast learns from
ROLE_TEST = "examen"                  # closed months that only evaluate
ROLE_PENDING = "pendiente_cierre"     # not closed yet: neither learn nor evaluate
ROLE_PROJECTION = "proyeccion"        # the current month and the future
