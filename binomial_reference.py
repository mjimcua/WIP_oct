"""binomial_reference.py — SFF v3 · the binomial reference and the logit scale, shared by everything.

GUIDING PRINCIPLE. The renewal rate is a proportion: k renewals out of n expirations.
That gives an exact reference that depends on no external threshold — the sampling
error by sample size, √(p(1−p)/n). With p = 0.8 and n = 100 a month, the monthly rate
moves ±4 pp even if nothing changes. Everything else in the framework is measured
against it: the support floor, the normalization of error bands, the seasonality and
trend tests, and φ (observed variance / binomial variance) that says WHERE a complex
technique is worth it.

This module holds the pure functions; no frame, no config, no console.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np

# ─── named constants ─────────────────────────────────────────────────────────────
# p(1−p) can be 0 when p is exactly 0 or 1 in a tiny sample; a floor keeps the error
# finite (0.05·0.95 ≈ 0.0475: the variance of a 5 % rate).
MIN_PROPORTION_VARIANCE = 0.0475
# The logit is undefined at 0 and 1: rates are clipped to this open interval first.
LOGIT_CLIP = 1e-4
PERCENTAGE_POINTS = 100.0


def binomial_se(rate: float, support: float) -> float:
    """Standard error of a proportion, in proportion units: √(p(1−p)/n).

    INPUT:   rate in [0, 1] · support n (clipped to ≥ 1).
    OUTPUT:  the sampling error of ONE month with that n; ×100 for percentage points.
    RULES:   p(1−p) is floored at MIN_PROPORTION_VARIANCE so a 0 % or 100 % sample does
             not claim zero error.
    EDGE CASES: n ≤ 0 → treated as 1 (the error of "no evidence").
    """
    variance = max(rate * (1.0 - rate), MIN_PROPORTION_VARIANCE)
    return float(np.sqrt(variance / max(support, 1.0)))


def binomial_se_pp(rate: float, support: float) -> float:
    """`binomial_se` in percentage points."""
    return PERCENTAGE_POINTS * binomial_se(rate, support)


def wilson_interval(rate: float, support: float, z: float) -> tuple:
    """Wilson score interval for a proportion: honest for small n and extreme p.

    INPUT:   rate p̂ · support n · z (1.645 for 90 %).
    OUTPUT:  (low, high) in proportion units, always inside [0, 1] and asymmetric near
             the edges — unlike Wald (p̂ ± z·se), which can cross 0 or 1 and is too
             narrow when p̂ is extreme (Brown, Cai & DasGupta 2001).
    EDGE CASES: n ≤ 0 → (0, 1).
    """
    if support <= 0:
        return 0.0, 1.0
    n = float(support)
    z2 = z * z
    centre = (rate + z2 / (2 * n)) / (1 + z2 / n)
    half_width = z * np.sqrt(rate * (1 - rate) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def wilson_half_width_pp(rate: float, support: float, z: float) -> float:
    """Half the Wilson interval, in pp: the binomial error by sample size of one month."""
    low, high = wilson_interval(rate, support, z)
    return PERCENTAGE_POINTS * (high - low) / 2.0


def support_for_half_width(half_width_pp: float, z: float, rate: float = 0.5) -> float:
    """The n needed for a given 90 % half-width (the "dial"): (z·100·√(p(1−p)) / τ)².

    With p = 0.5: ±15 pp ↔ 30, ±5 pp ↔ 271, ±3 pp ↔ 752.
    """
    return (z * PERCENTAGE_POINTS * np.sqrt(rate * (1 - rate)) / half_width_pp) ** 2


def logit(rate):
    """log(p/(1−p)), with p clipped to (LOGIT_CLIP, 1−LOGIT_CLIP). Works on arrays."""
    clipped = np.clip(np.asarray(rate, dtype=float), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(clipped / (1.0 - clipped))


def inverse_logit(value):
    """1/(1+e^{−x}). Works on arrays. Always inside (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def overdispersion_phi(monthly_rates: np.ndarray, monthly_supports: np.ndarray) -> tuple:
    """φ = observed variance of the monthly rate / variance the binomial predicts.

    INPUT:   monthly_rates, monthly_supports — aligned arrays over the months with truth.
    OUTPUT:  (phi, sd_observed_pp, sd_binomial_pp). NaN when fewer than 6 months.
    RULES:   the binomial variance uses EACH month's own n (a month with 400 units is a
             steadier witness than one with 40): p̄(1−p̄)·mean(1/n_t), with p̄ the pooled
             rate. φ ≈ 1: the rate does not move, it samples. φ > 1: an engine exists.
    """
    rates = np.asarray(monthly_rates, dtype=float)
    supports = np.asarray(monthly_supports, dtype=float)
    keep = np.isfinite(rates) & (supports > 0)
    rates, supports = rates[keep], supports[keep]
    if len(rates) < 6:
        return float("nan"), float("nan"), float("nan")
    observed_variance = float(np.var(rates, ddof=1))
    pooled_rate = float(np.sum(rates * supports) / np.sum(supports))
    binomial_variance = pooled_rate * (1 - pooled_rate) * float(np.mean(1.0 / supports))
    phi = observed_variance / binomial_variance if binomial_variance > 0 else float("nan")
    return phi, PERCENTAGE_POINTS * np.sqrt(observed_variance), PERCENTAGE_POINTS * np.sqrt(binomial_variance)


def weighted_quantile(values: np.ndarray, quantile: float) -> float:
    """Plain empirical quantile that tolerates NaN and empty input (returns NaN)."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(clean, quantile))
