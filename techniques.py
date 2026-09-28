"""techniques.py — SFF v3 · the catalogue of forecasting techniques (shared by ANALYSIS and RUN).

REDUCED ON PURPOSE. The seasonality benchmark (phase 2) decides ONCE, on the big series,
whether the rate has a material calendar shape. The catalogue is level techniques (recent
level, smoothed level, level with a damped slope), two classic time-series methods that
never invent a season (Theta, Holt damped), and two seasonal ones (T15: recent level +
month effect; T11: Holt-Winters) that compete only in the series the benchmark declared
seasonal. No technique "finds" a season on its own.

Every technique predicts the rate of a monthly series at horizon h. They work on the
LOGIT of the rate: no technique can predict above 100 % or below 0 %, intervals become
asymmetric when transformed back (narrower near the ceiling), and a trend damps by
itself as it approaches 1. Point forecasts are transformed back to rate units.

Every technique declares its ELIGIBILITY: the minimum months of history and, for T15,
the `estacional` label that only the benchmark grants. A series the benchmark did not
declare seasonal never competes with the month-effect technique.

Families (for tie-breaking, richer first): time_series > smoothing > average.

MIXED techniques (T15, T16): a seasonal SHAPE on a RECENT LEVEL. When the level of a
series moves (a regime, a portfolio move) the whole-history seasonal techniques miss the
level and the short-window ones miss the season; these deseasonalize the history, take
the recent level, and put the target month's season back on top.

Numpy only. Additional techniques (SARIMA, ETS auto...) plug in by adding an entry to
CATALOGUE with the same signature: f(logit_series: np.ndarray, month_numbers: np.ndarray
of calendar months 1..12, horizon: int, dynamics: dict) → float (logit). Calendar months
are plain integers (not a PeriodIndex) so a technique costs microseconds: the backtest
calls them millions of times.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

from binomial_reference import inverse_logit, logit

# ─── named constants ─────────────────────────────────────────────────────────────
MONTHS_PER_CYCLE = 12
EWMA_HALFLIFE_MONTHS = 3.0
SES_ALPHA = 0.3
HOLT_ALPHA, HOLT_BETA, HOLT_DAMPING = 0.3, 0.1, 0.9
HW_ALPHA, HW_BETA, HW_GAMMA = 0.3, 0.05, 0.2
THETA_WEIGHT = 0.5
CROSTON_ZERO_SHARE = 0.30
TEMPORAL_CREDIBILITY_K = 6.0
RECENT_WINDOW_MONTHS = 6
FAMILY_RANK = {UNIVERSE_TIME_SERIES: 3, "smoothing": 2, "average": 1}


# ─── helpers ─────────────────────────────────────────────────────────────────────

def damping_weight(horizon: int, damping: float) -> float:
    """Σ_{i=1..h} φ^i: how much of the trend survives at horizon h (damped Holt)."""
    return float(sum(damping ** i for i in range(1, horizon + 1)))


def target_calendar_month(month_numbers: np.ndarray, horizon: int) -> int:
    """The calendar month h months after the last observed one."""
    return int((month_numbers[-1] - 1 + horizon) % MONTHS_PER_CYCLE + 1)


def seasonal_profile_logit(month_numbers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The 12 additive indices at once (index 0 = January)."""
    overall = float(np.mean(values))
    profile = np.zeros(MONTHS_PER_CYCLE)
    for month in range(1, MONTHS_PER_CYCLE + 1):
        same = values[month_numbers == month]
        profile[month - 1] = float(np.mean(same) - overall) if len(same) else 0.0
    return profile


# ─── techniques (all: logit series, months, horizon, dynamics → logit prediction) ──

def t2_mean(y, months, h, dyn):
    return float(np.mean(y))


def t3_moving_average_3(y, months, h, dyn):
    return float(np.mean(y[-3:]))


def t3_moving_average_6(y, months, h, dyn):
    return float(np.mean(y[-6:]))


def t4_ewma(y, months, h, dyn):
    weights = 0.5 ** (np.arange(len(y))[::-1] / EWMA_HALFLIFE_MONTHS)
    return float(np.sum(weights * y) / np.sum(weights))


def t9_ses(y, months, h, dyn):
    level = y[0]
    for value in y[1:]:
        level = SES_ALPHA * value + (1 - SES_ALPHA) * level
    return float(level)


def t10_holt_damped(y, months, h, dyn):
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    for value in y[1:]:
        previous_level = level
        level = HOLT_ALPHA * value + (1 - HOLT_ALPHA) * (level + HOLT_DAMPING * trend)
        trend = HOLT_BETA * (level - previous_level) + (1 - HOLT_BETA) * HOLT_DAMPING * trend
    return float(level + trend * damping_weight(h, HOLT_DAMPING))


def deseasonalized(y, months):
    """The series minus its additive seasonal index (logit): the LEVEL without the season."""
    profile = seasonal_profile_logit(months, y)
    return y - profile[months - 1], profile


def t15_recent_level_seasonal(y, months, h, dyn):
    """Mixed: the level of the last 3 months (deseasonalized) + the seasonal index of the
    target month. Keeps the SHAPE of the season while following a level that moved."""
    level_series, profile = deseasonalized(y, months)
    return float(np.mean(level_series[-3:]) + profile[target_calendar_month(months, h) - 1])


def t11_holt_winters_additive(y, months, h, dyn):
    """Additive Holt-Winters on the logit, seasonal period 12, damped trend: the full
    seasonal time-series model. Competes only where the benchmark declared month effects."""
    season = seasonal_profile_logit(months[:MONTHS_PER_CYCLE], y[:MONTHS_PER_CYCLE])
    level, trend = float(np.mean(y[:MONTHS_PER_CYCLE])), 0.0
    for index in range(MONTHS_PER_CYCLE, len(y)):
        month = int(months[index])
        previous_level = level
        level = HW_ALPHA * (y[index] - season[month - 1]) + (1 - HW_ALPHA) * (level + HOLT_DAMPING * trend)
        trend = HW_BETA * (level - previous_level) + (1 - HW_BETA) * HOLT_DAMPING * trend
        season[month - 1] = HW_GAMMA * (y[index] - level) + (1 - HW_GAMMA) * season[month - 1]
    return float(level + trend * damping_weight(h, HOLT_DAMPING) + season[target_calendar_month(months, h) - 1])


def t8_damped_trend(y, months, h, dyn):
    """Linear trend fitted on the whole history, extrapolated with damping (used inside Theta)."""
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (len(y) - 1) + slope * damping_weight(h, HOLT_DAMPING))


def t12_theta(y, months, h, dyn):
    """Theta (Assimakopoulos & Nikolopoulos 2000), simple form: the average of the damped
    linear trend and simple exponential smoothing. A classic, non-seasonal time-series
    method: it follows level and trend, it never invents a season."""
    return float(THETA_WEIGHT * t8_damped_trend(y, months, h, dyn) + (1 - THETA_WEIGHT) * t9_ses(y, months, h, dyn))


def t14_temporal_credibility(y, months, h, dyn):
    """Recent window vs whole history, blended with z = n/(n+k) on the number of recent months."""
    recent = y[-RECENT_WINDOW_MONTHS:]
    z = len(recent) / (len(recent) + TEMPORAL_CREDIBILITY_K)
    return float(z * np.mean(recent) + (1 - z) * np.mean(y))


# ─── catalogue ───────────────────────────────────────────────────────────────────
# id → (family, description, min_history_months, required label, function)
CATALOGUE = {
    "T2_mean":          ("average",     "mean of the whole history",                  1,  None,         t2_mean),
    "T3_ma3":           ("average",     "moving average, 3 months (challenger)",      3,  None,         t3_moving_average_3),
    "T3_ma6":           ("average",     "moving average, 6 months",                   6,  None,         t3_moving_average_6),
    "T4_ewma":          ("smoothing",   "exponentially weighted mean, half-life 3",   4,  None,         t4_ewma),
    "T9_ses":           ("smoothing",   "simple exponential smoothing",               4,  None,         t9_ses),
    "T10_holt_damped":  ("smoothing",   "Holt damped trend (level with a damped slope)", 12, None,      t10_holt_damped),
    "T12_theta":        (UNIVERSE_TIME_SERIES, "Theta: damped trend + exponential smoothing",  12, None,         t12_theta),
    "T11_holt_winters": (UNIVERSE_TIME_SERIES, "Holt-Winters additive on logit, damped (only where the benchmark found seasonality)", 24, "estacional", t11_holt_winters_additive),
    "T14_temporal_cred":("average",     "recent window with temporal credibility",    6,  None,         t14_temporal_credibility),
    "T15_level_seasonal":(UNIVERSE_TIME_SERIES, "recent level + month effect (only where the benchmark found seasonality)", 13, "estacional", t15_recent_level_seasonal),
}


# How many months of history each technique actually uses to build its forecast
# (None = the whole history). Far horizons prefer memory: within the margin, the
# technique that has seen more of the past wins.
MEMORY_MONTHS = {"T2_mean": None, "T3_ma3": 3, "T3_ma6": 6, "T4_ewma": 9, "T9_ses": 9, "T10_holt_damped": None,
                 "T12_theta": None, "T11_holt_winters": None, "T14_temporal_cred": 6, "T15_level_seasonal": None}


def memory_rank(technique_id: str) -> int:
    """Months of memory, with the whole history ranked above any window."""
    memory = MEMORY_MONTHS.get(technique_id)
    return 10_000 if memory is None else int(memory)


def technique_dimension() -> pd.DataFrame:
    """The master table of techniques (`dim_tecnica`)."""
    return pd.DataFrame([dict(tecnica_id=tid, familia=fam, descripcion=desc, historia_minima=mh, requiere=req or "",
                              memoria_meses=MEMORY_MONTHS.get(tid) if MEMORY_MONTHS.get(tid) is not None else 0)
                         for tid, (fam, desc, mh, req, _) in CATALOGUE.items()])


def eligible_techniques(history_months: int, dynamics: dict) -> list:
    """The technique ids a series may compete with, given its history and its labels.

    INPUT:   history_months · dynamics — dict with estacional (0/1, the benchmark's verdict
             for the series), tendencia (−1/0/1), intermitente (0/1); missing keys count as 0.
    RULES:   min history; the `estacional` technique (T15) needs estacional = 1. No technique
             of the current catalogue needs `tendencia` or `intermitente` (kept in the
             contract for future techniques). The challenger T3_ma3 is eligible with 3 months.
    """
    labels = {"estacional": int(dynamics.get("estacional", 0) or 0) >= 1,
              "tendencia": int(dynamics.get("tendencia", 0) or 0) != 0,
              "intermitente": int(dynamics.get("intermitente", 0) or 0) == 1}
    eligible = []
    for tid, (_, _, min_history, required, _) in CATALOGUE.items():
        if history_months < min_history:
            continue
        if required is not None and not labels[required]:
            continue
        eligible.append(tid)
    return eligible


def month_numbers_of(months: pd.PeriodIndex) -> np.ndarray:
    """PeriodIndex → integer calendar months (1..12), the form the techniques take."""
    return np.asarray(pd.PeriodIndex(months).month, dtype=int)


def predict(technique_id: str, rates: np.ndarray, months: pd.PeriodIndex, horizon: int, dynamics: dict) -> float:
    """Point forecast of the RATE at horizon h with one technique. Returns NaN if it fails.
    `months` may be a PeriodIndex or an integer array of calendar months."""
    clean = np.isfinite(rates)
    month_numbers = months if isinstance(months, np.ndarray) else month_numbers_of(months)
    y, m = logit(np.asarray(rates, dtype=float)[clean]), month_numbers[clean]
    if len(y) == 0:
        return float("nan")
    try:
        value = CATALOGUE[technique_id][4](y, m, int(horizon), dynamics or {})
    except Exception:
        return float("nan")
    return float(inverse_logit(value)) if np.isfinite(value) else float("nan")


def family_rank(technique_id: str) -> int:
    return FAMILY_RANK[CATALOGUE[technique_id][0]]
