"""
prediction.py — How the rate of a forecast series is predicted, in ONE place.

The forecast (step 17), the exam of every forecast series (step 19) and the audit tables (AUD)
predict the rate of a forecast series the same way, so they all call these functions:

  1. predict_composition   the technique of the composition predicts its rate h months ahead
                           (falls back to the challenger when the technique cannot predict)
  2. credibility_shift     a composition below own_rate_floor moves its prediction toward its
                           credibility reference: (1 − z) × (logit reference level − logit
                           composition level); 0 when z = 1 or a level is missing
  3. shifted_rate          the prediction with the shift applied: the rate of the forecast series
  4. band_quantiles        the error quantiles of the technique at the horizon (step 14; the
                           nearest judged horizon beyond them; the challenger's, then ±z, if missing)
  5. rate_band             the interval of the rate: rate + quantile × binomial error with the
                           units due of the forecast series, clipped to [0, 1]
  6. levels_at_origins     the level (Σ renewed / Σ due) of every key with only what was known at
                           every origin (for the exam: nothing after the origin is used)

Every function returns its intermediate values too, so a prediction can be traced step by step
(composition rate → shift → series rate → band).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from techniques import inverse_logit, logit, predict_logit

LEVEL_CLIP = 1e-6          # a level of exactly 0 or 1 has no logit: it is clipped before the shift


def predict_composition(history_logit: np.ndarray, calendar_months: np.ndarray, technique: str, horizon: int,
                        challenger: str) -> tuple:
    """(technique used, predicted rate) of a composition h months ahead; the challenger when the
    technique cannot predict (too short a history, a failed fit)."""
    predicted = predict_logit(technique, history_logit, calendar_months, int(horizon))
    if not np.isfinite(predicted):
        technique = challenger
        predicted = predict_logit(technique, history_logit, calendar_months, int(horizon))
    return technique, (float(inverse_logit(predicted)) if np.isfinite(predicted) else np.nan)


def credibility_shift(z, reference_level, composition_level, apply: bool = True) -> np.ndarray:
    """(1 − z) × (logit reference − logit composition) where z < 1 and both levels are in (0, 1); else 0."""
    z = np.asarray(pd.Series(z).fillna(1.0), dtype=float)
    reference = np.asarray(pd.Series(reference_level), dtype=float)
    composition = np.asarray(pd.Series(composition_level), dtype=float)
    valid = (z < 1) & np.isfinite(reference) & np.isfinite(composition) \
        & (reference > 0) & (reference < 1) & (composition > 0) & (composition < 1)
    if not apply:
        return np.zeros(len(z))
    reference = np.clip(np.where(valid, reference, 0.5), LEVEL_CLIP, 1 - LEVEL_CLIP)
    composition = np.clip(np.where(valid, composition, 0.5), LEVEL_CLIP, 1 - LEVEL_CLIP)
    return np.where(valid, (1 - z) * (logit(reference) - logit(composition)), 0.0)


def shifted_rate(composition_rate, z, reference_level, composition_level, apply: bool = True) -> tuple:
    """(rate of the forecast series, shift in logit): the composition's prediction moved toward its reference."""
    shift = credibility_shift(z, reference_level, composition_level, apply)
    composition_rate = np.asarray(pd.Series(composition_rate), dtype=float)
    rate = np.where(np.isfinite(composition_rate),
                    inverse_logit(logit(np.clip(np.nan_to_num(composition_rate, nan=0.5), LEVEL_CLIP, 1 - LEVEL_CLIP)) + shift),
                    np.nan)
    return rate, shift


def judged_horizon(horizon: int, judged: list) -> int:
    """The judged horizon whose error quantiles a horizon uses: itself, the next one above, or the last."""
    above = [judged_h for judged_h in sorted(judged) if judged_h >= horizon]
    return above[0] if above else max(judged)


def band_quantiles(bands: pd.DataFrame, techniques, horizons, configuration) -> tuple:
    """(q_low, q_high) of every (technique, horizon): the technique's quantiles at its judged horizon
    (itself, the next one above, or the last); the challenger's if the technique has none; ±z if neither has."""
    table = bands.set_index(["tecnica", "h"])
    judged = list(configuration.backtest_horizons)
    keys = [(technique, judged_horizon(int(horizon), judged)) for technique, horizon in zip(techniques, horizons)]
    fallbacks = [(configuration.challenger_technique, judged_h) for _, judged_h in keys]
    q_low = np.array([table["q_low_norm"].get(key, table["q_low_norm"].get(fallback, -configuration.z))
                      for key, fallback in zip(keys, fallbacks)], dtype=float)
    q_high = np.array([table["q_high_norm"].get(key, table["q_high_norm"].get(fallback, configuration.z))
                       for key, fallback in zip(keys, fallbacks)], dtype=float)
    return q_low, q_high


def rate_band(rate, units_due, q_low, q_high) -> tuple:
    """(low, high) of the rate: rate + quantile × √(rate (1 − rate) / units due), clipped to [0, 1]."""
    rate = np.asarray(pd.Series(rate), dtype=float)
    units = np.maximum(np.asarray(pd.Series(units_due), dtype=float), 1.0)
    binomial_error = np.sqrt(rate * (1 - rate) / units)
    low = np.clip(rate + np.minimum(q_low, 0) * binomial_error, 0, 1)
    high = np.clip(rate + np.maximum(q_high, 0) * binomial_error, 0, 1)
    return low, high


def levels_at_origins(history: pd.DataFrame, key: str, wanted: pd.DataFrame, period_column: str,
                      due_column: str, renewed_column: str) -> pd.DataFrame:
    """The level (Σ renewed / Σ due) of every key with only the months up to every origin it needs.
    wanted: rows of (key, origin). Returns (key, origin, level)."""
    rows = []
    history_of_key = dict(tuple(history.groupby(key))) if len(history) else {}
    for key_value, origins in wanted.dropna().groupby(key)["origin"]:
        own = history_of_key.get(key_value)
        for origin in origins:
            known = own[own[period_column] <= origin] if own is not None else None
            due = known[due_column].sum() if known is not None else 0.0
            rows.append({key: key_value, "origin": origin,
                         "level": known[renewed_column].sum() / due if due > 0 else np.nan})
    return pd.DataFrame(rows, columns=[key, "origin", "level"])
