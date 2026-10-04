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

from config import Config


# ─── the parameters of the techniques: they live in the Config ────────────────────
# The orchestrator hands them over before any step runs (apply_configuration). Until then they hold the
# Config's defaults, so a step or a test that calls a technique alone sees the same numbers.
TECHNIQUE_PARAMETERS = ["logit_clip", "ewma_halflife_months", "ses_alpha", "holt_alpha", "holt_beta", "holt_damping",
                        "theta_weight", "temporal_credibility_k", "recent_window_months", "holt_winters_alpha",
                        "holt_winters_beta", "holt_winters_gamma"]
parameters = {name: Config.__dataclass_fields__[name].default for name in TECHNIQUE_PARAMETERS}
MONTHS_PER_CYCLE = 12                                     # a definition, not a parameter: the months of a year


def apply_configuration(configuration: Config) -> None:
    """The technique parameters of this run, from its Config."""
    for name in TECHNIQUE_PARAMETERS:
        parameters[name] = getattr(configuration, name)


def logit(rates: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(rates, dtype=float), parameters["logit_clip"], 1 - parameters["logit_clip"])
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
    weights = 0.5 ** (np.arange(len(y))[::-1] / parameters["ewma_halflife_months"])
    return float(np.sum(weights * y) / np.sum(weights))


def simple_exponential_smoothing(y, months, h):
    level = y[0]
    for value in y[1:]:
        level = parameters["ses_alpha"] * value + (1 - parameters["ses_alpha"]) * level
    return float(level)


def holt_damped(y, months, h):
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    for value in y[1:]:
        previous_level = level
        level = parameters["holt_alpha"] * value + (1 - parameters["holt_alpha"]) * (level + parameters["holt_damping"] * trend)
        trend = parameters["holt_beta"] * (level - previous_level) + (1 - parameters["holt_beta"]) * parameters["holt_damping"] * trend
    return float(level + trend * damping_weight(h, parameters["holt_damping"]))


def damped_linear_trend(y, months, h):
    positions = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(positions, y, 1)
    return float(intercept + slope * (len(y) - 1) + slope * damping_weight(h, parameters["holt_damping"]))


def theta(y, months, h):
    """Theta (Assimakopoulos & Nikolopoulos 2000), simple form: half damped linear trend,
    half simple exponential smoothing. Follows level and trend, never invents a season."""
    return float(parameters["theta_weight"] * damped_linear_trend(y, months, h) + (1 - parameters["theta_weight"]) * simple_exponential_smoothing(y, months, h))


def temporal_credibility(y, months, h):
    """The recent window blended with the whole history, z = n / (n + k) on the recent months."""
    recent = y[-parameters["recent_window_months"]:]
    z = len(recent) / (len(recent) + parameters["temporal_credibility_k"])
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
        level = parameters["holt_winters_alpha"] * (y[index] - season[month - 1]) + (1 - parameters["holt_winters_alpha"]) * (level + parameters["holt_damping"] * trend)
        trend = parameters["holt_winters_beta"] * (level - previous_level) + (1 - parameters["holt_winters_beta"]) * parameters["holt_damping"] * trend
        season[month - 1] = parameters["holt_winters_gamma"] * (y[index] - level) + (1 - parameters["holt_winters_gamma"]) * season[month - 1]
    return float(level + trend * damping_weight(h, parameters["holt_damping"]) + season[target_calendar_month(months, h) - 1])


# id → (family, description, months of memory, minimum months of history, function)
CATALOGUE = {
    "T2_mean":            ("media", "media de toda la historia", "toda", 1, mean_of_history),
    "T3_ma3":             ("media", "media de los últimos 3 meses (el retador)", "3", 3, moving_average_3),
    "T3_ma6":             ("media", "media de los últimos 6 meses", "6", 6, moving_average_6),
    "T4_ewma":            ("suavizado", "media ponderada, pesos que se reducen a la mitad cada 3 meses", "~6", 4, exponentially_weighted_mean),
    "T9_ses":             ("suavizado", "suavizado exponencial simple (α de la Config: ses_alpha)", "~6", 4, simple_exponential_smoothing),
    "T10_holt_damped":    ("suavizado", "nivel + tendencia que se amortigua (Holt, φ de la Config: holt_damping)", "12", 12, holt_damped),
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
