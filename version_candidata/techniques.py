"""
techniques.py — The catalogue of techniques that predict a renewal rate (used by step 14).

Every technique takes the monthly history of a rate (as logits: the rate lives in (0, 1),
the logit is unbounded, so an average or a trend never predicts a rate above 100 %),
the calendar months of that history and a horizon h, and returns the predicted rate h
months after the last month it saw.

What limits a technique is only its HISTORY: it competes when the months it has seen reach
its minimum (a seasonal technique needs to have seen every month of the year). No
seasonality verdict restricts it: the backtest decides.

    id                  family       what it does                                       memory
    T2_mean             average      mean of the whole history                          all
    T3_ma3              average      mean of the last 3 months (THE CHALLENGER)          3
    T3_ma6              average      mean of the last 6 months                          6
    T4_ewma             smoothing    weighted mean, weights halve every 3 months        ~6
    T9_ses              smoothing    simple exponential smoothing (α = 0.3)             ~6
    T10_holt_damped     smoothing    level + a trend that fades (Holt, damped 0.9)      12
    T12_theta           time series  Theta: half damped linear trend, half SES          all
    T14_temporal_cred   average      recent 6 months blended with the whole history     all
    T15_level_seasonal  time series  recent level + the effect of the target month      13 months of history
    T11_holt_winters    time series  Holt-Winters additive, damped trend, period 12      24 months of history
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ─── named constants ─────────────────────────────────────────────────────────────
LOGIT_CLIP = 0.001                 # a rate of exactly 0 or 1 has no logit: it is clipped to [0.001, 0.999]
EWMA_HALFLIFE_MONTHS = 3.0
SES_ALPHA = 0.3
HOLT_ALPHA, HOLT_BETA, HOLT_DAMPING = 0.3, 0.1, 0.9
THETA_WEIGHT = 0.5
TEMPORAL_CREDIBILITY_K = 6.0
RECENT_WINDOW_MONTHS = 6
MONTHS_PER_CYCLE = 12
HW_ALPHA, HW_BETA, HW_GAMMA = 0.3, 0.05, 0.2


def logit(rates: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(rates, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(clipped / (1 - clipped))


def inverse_logit(values) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(values, dtype=float)))


def damping_weight(horizon: int, damping: float) -> float:
    """Σ_{i=1..h} φ^i: how much of a trend survives h months ahead when it fades by φ a month."""
    return float(sum(damping ** step for step in range(1, horizon + 1)))


# ─── the techniques: logit history y, horizon h → logit prediction ────────────────
def mean_of_history(y, months, h):
    return float(np.mean(y))


def moving_average_3(y, months, h):
    return float(np.mean(y[-3:]))


def moving_average_6(y, months, h):
    return float(np.mean(y[-6:]))


def exponentially_weighted_mean(y, months, h):
    weights = 0.5 ** (np.arange(len(y))[::-1] / EWMA_HALFLIFE_MONTHS)
    return float(np.sum(weights * y) / np.sum(weights))


def simple_exponential_smoothing(y, months, h):
    level = y[0]
    for value in y[1:]:
        level = SES_ALPHA * value + (1 - SES_ALPHA) * level
    return float(level)


def holt_damped(y, months, h):
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    for value in y[1:]:
        previous_level = level
        level = HOLT_ALPHA * value + (1 - HOLT_ALPHA) * (level + HOLT_DAMPING * trend)
        trend = HOLT_BETA * (level - previous_level) + (1 - HOLT_BETA) * HOLT_DAMPING * trend
    return float(level + trend * damping_weight(h, HOLT_DAMPING))


def damped_linear_trend(y, months, h):
    positions = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(positions, y, 1)
    return float(intercept + slope * (len(y) - 1) + slope * damping_weight(h, HOLT_DAMPING))


def theta(y, months, h):
    """Theta (Assimakopoulos & Nikolopoulos 2000), simple form: half damped linear trend,
    half simple exponential smoothing. Follows level and trend, never invents a season."""
    return float(THETA_WEIGHT * damped_linear_trend(y, months, h) + (1 - THETA_WEIGHT) * simple_exponential_smoothing(y, months, h))


def temporal_credibility(y, months, h):
    """The recent window blended with the whole history, z = n / (n + k) on the recent months."""
    recent = y[-RECENT_WINDOW_MONTHS:]
    z = len(recent) / (len(recent) + TEMPORAL_CREDIBILITY_K)
    return float(z * np.mean(recent) + (1 - z) * np.mean(y))


def target_calendar_month(months, horizon: int) -> int:
    """The calendar month (1-12) h months after the last month seen."""
    return int((months[-1] - 1 + horizon) % MONTHS_PER_CYCLE + 1)


def seasonal_profile(y, months) -> np.ndarray:
    """The 12 additive month effects (logit), index 0 = January; 0 for a month not seen."""
    overall = float(np.mean(y))
    profile = np.zeros(MONTHS_PER_CYCLE)
    for month in range(1, MONTHS_PER_CYCLE + 1):
        same_month = y[months == month]
        profile[month - 1] = float(np.mean(same_month) - overall) if len(same_month) else 0.0
    return profile


def level_plus_month_effect(y, months, h):
    """The level of the last 3 months without their month effect + the effect of the target
    month: keeps the SHAPE of the year while following a level that moved."""
    profile = seasonal_profile(y, months)
    level = y - profile[months - 1]
    return float(np.mean(level[-3:]) + profile[target_calendar_month(months, h) - 1])


def holt_winters(y, months, h):
    """Additive Holt-Winters on the logit, seasonal period 12, damped trend."""
    season = seasonal_profile(y[:MONTHS_PER_CYCLE], months[:MONTHS_PER_CYCLE])
    level, trend = float(np.mean(y[:MONTHS_PER_CYCLE])), 0.0
    for index in range(MONTHS_PER_CYCLE, len(y)):
        month = int(months[index])
        previous_level = level
        level = HW_ALPHA * (y[index] - season[month - 1]) + (1 - HW_ALPHA) * (level + HOLT_DAMPING * trend)
        trend = HW_BETA * (level - previous_level) + (1 - HW_BETA) * HOLT_DAMPING * trend
        season[month - 1] = HW_GAMMA * (y[index] - level) + (1 - HW_GAMMA) * season[month - 1]
    return float(level + trend * damping_weight(h, HOLT_DAMPING) + season[target_calendar_month(months, h) - 1])


# id → (family, description, months of memory, minimum months of history, function)
CATALOGUE = {
    "T2_mean":            ("media", "media de toda la historia", "toda", 1, mean_of_history),
    "T3_ma3":             ("media", "media de los últimos 3 meses (el retador)", "3", 3, moving_average_3),
    "T3_ma6":             ("media", "media de los últimos 6 meses", "6", 6, moving_average_6),
    "T4_ewma":            ("suavizado", "media ponderada, pesos que se reducen a la mitad cada 3 meses", "~6", 4, exponentially_weighted_mean),
    "T9_ses":             ("suavizado", "suavizado exponencial simple (α = 0,3)", "~6", 4, simple_exponential_smoothing),
    "T10_holt_damped":    ("suavizado", "nivel + tendencia que se amortigua (Holt, φ = 0,9)", "12", 12, holt_damped),
    "T12_theta":          ("serie_temporal", "Theta: mitad tendencia lineal amortiguada, mitad suavizado simple", "toda", 12, theta),
    "T14_temporal_cred":  ("media", "últimos 6 meses mezclados con toda la historia por credibilidad", "toda", 6, temporal_credibility),
    "T15_level_seasonal": ("serie_temporal", "nivel reciente + efecto del mes objetivo", "toda", 13, level_plus_month_effect),
    "T11_holt_winters":   ("serie_temporal", "Holt-Winters aditivo, tendencia amortiguada, periodo 12", "toda", 24, holt_winters),
}


def eligible_techniques(history_months: int) -> list:
    """The techniques whose minimum history the series has reached: only the history limits."""
    return [technique_id for technique_id, entry in CATALOGUE.items() if history_months >= entry[3]]


def predict_logit(technique_id: str, history_logit: np.ndarray, month_numbers: np.ndarray, horizon: int) -> float:
    """The prediction of one technique in logit; NaN if it fails."""
    try:
        value = CATALOGUE[technique_id][4](history_logit, np.asarray(month_numbers, dtype=int), int(horizon))
    except Exception:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def technique_table() -> pd.DataFrame:
    return pd.DataFrame([{"tecnica": technique_id, "familia": family, "descripcion": description, "memoria_meses": memory,
                          "historia_minima_meses": minimum}
                         for technique_id, (family, description, memory, minimum, _) in CATALOGUE.items()])
