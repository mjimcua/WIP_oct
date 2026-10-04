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

from config import Config

from techniques import inverse_logit, logit, predict_logit

# the level clip lives in the Config (level_clip); the orchestrator hands it over before any step runs,
# until then it holds the Config's default
parameters = {"level_clip": Config.__dataclass_fields__["level_clip"].default}


def apply_configuration(configuration: Config) -> None:
    """The prediction parameters of this run, from its Config."""
    parameters["level_clip"] = configuration.level_clip


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
    reference = np.clip(np.where(valid, reference, 0.5), parameters["level_clip"], 1 - parameters["level_clip"])
    composition = np.clip(np.where(valid, composition, 0.5), parameters["level_clip"], 1 - parameters["level_clip"])
    return np.where(valid, (1 - z) * (logit(reference) - logit(composition)), 0.0)


def shifted_rate(composition_rate, z, reference_level, composition_level, apply: bool = True) -> tuple:
    """(rate of the forecast series, shift in logit): the composition's prediction moved toward its reference.
    Without a shift the rate IS the composition's prediction, untouched (also when it is exactly 0 or 1);
    only a shifted rate goes through the logit scale."""
    shift = credibility_shift(z, reference_level, composition_level, apply)
    composition_rate = np.asarray(pd.Series(composition_rate), dtype=float)
    moved = inverse_logit(logit(np.clip(np.nan_to_num(composition_rate, nan=0.5), parameters["level_clip"], 1 - parameters["level_clip"])) + shift)
    rate = np.where(shift != 0, moved, composition_rate)
    return rate, shift


def judged_horizon(horizon: int, judged: list) -> int:
    """The judged horizon whose error quantiles a horizon uses: itself, the next one above, or the last."""
    above = [judged_h for judged_h in sorted(judged) if judged_h >= horizon]
    return above[0] if above else max(judged)


def band_quantiles(bands: pd.DataFrame, techniques, horizons, configuration) -> tuple:
    """(q_low, q_high) of every (technique, horizon): the technique's quantiles at its judged horizon
    (itself, the next one above, or the last); the challenger's if the technique has none; ±z if neither has."""
    table = bands.set_index(["tecnica", "h"])
    # plain dicts, built once: a lookup per row in a dict costs ~0.1 µs; on the MultiIndex of a Series it
    # costs tens of µs, and close to a millisecond when the key is missing (it goes through an exception)
    low_of_key = table["q_low_norm"].to_dict()
    high_of_key = table["q_high_norm"].to_dict()
    judged = list(configuration.backtest_horizons)
    keys = [(technique, judged_horizon(int(horizon), judged)) for technique, horizon in zip(techniques, horizons)]
    fallbacks = [(configuration.challenger_technique, judged_h) for _, judged_h in keys]
    q_low = np.array([low_of_key.get(key, low_of_key.get(fallback, -configuration.z))
                      for key, fallback in zip(keys, fallbacks)], dtype=float)
    q_high = np.array([high_of_key.get(key, high_of_key.get(fallback, configuration.z))
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


def index_history(frame: pd.DataFrame, key_column: str, period_column: str, value_columns: list) -> dict:
    """The monthly history of every key as plain numpy arrays, sorted by period, built ONCE.

    One entry per key (in sorted order, like groupby; keys that are null are left out, like groupby):
    'ordinal' (the period as an integer, for searchsorted), 'month' (1..12) and one float array per value
    column. The hot loops slice these arrays instead of filtering, grouping and sorting a DataFrame per key
    per origin: the data and its order are exactly what groupby(key) + sort_values(period) gave.
    The arrays are read-only: a technique that wrote into its history would change the next origin's.
    """
    index = {}
    narrow = frame.loc[frame[key_column].notna(), [key_column, period_column] + list(value_columns)]
    if not len(narrow):
        return index
    narrow = narrow.sort_values([key_column, period_column])
    ordinals = np.array([period.ordinal for period in narrow[period_column]], dtype=np.int64)
    months = np.array([period.month for period in narrow[period_column]], dtype=np.int64)
    values = {column: narrow[column].to_numpy(dtype=float) for column in value_columns}
    keys = narrow[key_column].to_numpy()
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    bounds = np.append(starts, len(keys))
    for start, end in zip(bounds[:-1], bounds[1:]):
        entry = {"ordinal": ordinals[start:end], "month": months[start:end]}
        for column in value_columns:
            entry[column] = values[column][start:end]
        index[keys[start]] = entry
    for array in [ordinals, months] + list(values.values()):
        array.flags.writeable = False
    return index


def months_known(entry: dict, origin) -> int:
    """How many months of this key's history are at or before the origin (period <= origin)."""
    return int(np.searchsorted(entry["ordinal"], origin.ordinal, side="right"))


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
