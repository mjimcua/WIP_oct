"""synthetic.py — synthetic dataset v3 (test tooling).

Calendar (designed for "train on all history, test on the 2026 months that already
happened, forecast the rest of 2026, extend into 2027"):

  2023-01 .. 2025-12   train      (36 months)
  2026-01 .. 2026-08   test       (8 months with truth: the 2026 months already happened)
  2026-09              CURRENT month (labeled test on purpose: the doctrine must reassign it)
  2026-09 .. 2026-12   projection (known pipeline, no results)

FEATURE-COVERAGE MATRIX (feature → scenario), kept from v2 and extended:
  exhaustive config   → optional leftover column
  conservation        → discount combinations inside one forecast unit
  universes / routes  → flag_time_series, train_only (no_impact), projection_only (heuristic)
  gaps                → NA|B tele with months missing inside its history
  sign grouping       → small EU negatives (15/12/10: only together they cross 30) + a minority positive
  extra annulment     → channel mute (same rates on web and tele; tele small)
  mandatory ladder    → NA|A tele neutral small, cell NA has support
  mixed sign          → ONE series with dormant=1 and autorenew=1 (must stay alone)
  mix-shift → NA: product A (.90, weight falling) vs B (.50, weight rising)
  seasonality         → EU|A yearly sine, amplitude 5 pp, 3 full cycles
  trend               → EU|B declining .86 → .64 over 44 months (must saturate)
  uplift              → newcust+d40 at 1.5 vs veterans 1.03 / d40 1.10
  current month       → 2026-09 labeled test, carrying early results to be wiped
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── named constants ─────────────────────────────────────────────────────────────
HISTORY_START = "2023-01"
PROJECTION_END = "2026-12"
TEST_START = "2026-01"
CURRENT_MONTH = "2026-09"
PROJECTION_START = "2026-09"
MONTHS = pd.period_range(HISTORY_START, PROJECTION_END, freq="M")
BASE_AUV = 30.0
DISCOUNT_AUV_FACTOR = 0.6


def role_of(month: pd.Period) -> str:
    """The dataset role the SQL extract would label the month with."""
    month_text = str(month)
    if month_text > CURRENT_MONTH:
        return "projection"
    if month_text >= TEST_START:
        return "test"          # includes the current month, on purpose
    return "train"


def build_raw(seed: int = 7, with_discount_pct: bool = False) -> pd.DataFrame:
    """Build the synthetic raw. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    collected_rows = []

    def series(region, product, channel, dormant, softcancel, no_instalado, autorenew,
               base_units, rate_of, combinations, months=None, time_series=0):
        for index, month in enumerate(months if months is not None else MONTHS):
            role = role_of(month)
            base_rate = rate_of(index, month)
            for discount, newcust, share in combinations:
                units = max(1, int(round(base_units * share)))
                auv = BASE_AUV * (DISCOUNT_AUV_FACTOR if discount == "d40" else 1.0)
                pipeline_units, pipeline_usd = units, units * auv
                if role == "projection":
                    renewed_units = renewed_usd = np.nan
                else:
                    probability = min(0.99, max(0.01, base_rate + rng.normal(0, 0.015)))
                    renewed_units = rng.binomial(units, probability)
                    uplift = 1.5 if (newcust == 1 and discount == "d40") else (1.10 if discount == "d40" else 1.03)
                    renewed_usd = renewed_units * auv * (uplift + rng.normal(0, 0.02))
                    if str(month) == CURRENT_MONTH:          # early, partial results
                        renewed_units = int(renewed_units * 0.3)
                        renewed_usd = renewed_usd * 0.3
                collected_rows.append(dict(
                    period=str(month), dataset_role=role,
                    total_tr_units=pipeline_units, total_tr_usd=pipeline_usd,
                    total_renewed_units=renewed_units, total_renewed_usd=renewed_usd,
                    is_current_month=int(str(month) == CURRENT_MONTH), flag_time_series=time_series,
                    region=region, product=product, channel=channel,
                    dormant=dormant, softcancel=softcancel, no_instalado=no_instalado, autorenew=autorenew,
                    discount=discount, newcust=newcust))

    mix = [("d0", 0, .7), ("d40", 1, .3)]
    flat = [("d0", 0, 1.0)]
    seasonal = lambda i, m: .80 + .05 * np.sin(2 * np.pi * (m.month - 1) / 12)
    declining = lambda i, m: .86 - .005 * i

    # EU|A big seasonal · EU|B declining trend
    series("EU", "A", "web", 0, 0, 0, 0, 600, seasonal, mix)
    series("EU", "B", "web", 0, 0, 0, 0, 300, declining, mix)
    # channel mute: same rates on tele, small → extra annulment
    series("EU", "A", "tele", 0, 0, 0, 0, 12, seasonal, flat)
    series("EU", "B", "tele", 0, 0, 0, 0, 10, declining, flat)
    # NA mix-shift: A .90 weight falling, B .50 weight rising
    for index, month in enumerate(MONTHS):
        weight_a = max(.4, 1.0 - .014 * index)
        series("NA", "A", "web", 0, 0, 0, 0, int(400 * weight_a), lambda i, m: .90, flat, months=[month])
        series("NA", "B", "web", 0, 0, 0, 0, int(400 * (1.6 - weight_a)), lambda i, m: .50, flat, months=[month])
    # small EU negatives (sign grouping) + minority positive + one mixed sign
    series("EU", "A", "web", 1, 0, 0, 0, 15, lambda i, m: .45, flat)
    series("EU", "A", "web", 0, 1, 0, 0, 12, lambda i, m: .35, flat)
    series("EU", "A", "web", 0, 0, 1, 0, 10, lambda i, m: .40, flat)
    series("EU", "A", "web", 1, 1, 0, 0, 5, lambda i, m: .30, flat)
    series("EU", "A", "web", 0, 0, 0, 1, 6, lambda i, m: .93, flat)
    series("EU", "B", "web", 1, 0, 0, 1, 4, lambda i, m: .60, flat)          # mixed sign
    # gaps: NA|B tele with months missing
    with_holes = [m for j, m in enumerate(MONTHS) if j % 4 != 2]
    series("NA", "B", "tele", 0, 0, 0, 0, 40, lambda i, m: .55, flat, months=with_holes)
    # neutral small in NA: the mandatory ladder must find the NA cell
    series("NA", "A", "tele", 0, 0, 0, 0, 8, lambda i, m: .88, flat)
    # routes: train_only (no_impact) and projection_only (heuristic)
    series("NA", "A", "kiosk", 0, 0, 0, 0, 25, lambda i, m: .70, flat,
           months=[m for m in MONTHS if role_of(m) == "train"])
    series("EU", "B", "tienda", 0, 0, 0, 0, 30, lambda i, m: .60, flat,
           months=[m for m in MONTHS if role_of(m) == "projection"])
    # time_series universe
    series("EU", "A", "kiosk", 0, 0, 0, 0, 50, lambda i, m: .65, flat, time_series=1)
    raw = pd.DataFrame(collected_rows)
    if with_discount_pct:
        # the exact discount in tanto por 1 (0.4 for the d40 bucket, 0.0 for d0), UNKNOWN (null) in the kiosk channel
        raw["discount_pct"] = np.where(raw["channel"] == "kiosk", np.nan, np.where(raw["discount"] == "d40", 0.4, 0.0))
    return raw
