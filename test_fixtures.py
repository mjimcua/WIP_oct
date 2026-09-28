"""test_fixtures.py — shared fixtures for the v3 tests.

`build_ladder_raw` builds a raw of SIX SERIES designed so the ladder can be checked by eye
(mandatory = [region], timevarying = {softcancel: negative, autorenew: positive},
extra_renovacion = [channel]; 24 months of history 2024-01..2025-12, current month
2026-01, projection 2026-01..2026-03; floor 30):

  S1  EU | soft=0 auto=0 | web    n=200  rate .80   → A: its own (rung 0)
  S2  EU | soft=0 auto=0 | tele   n=10   rate .80   → B: rung 2 (channel annulled) pools with S1: n=210
  S3  EU | soft=1 auto=0 | web    n=12   rate .35   → B: rung 1 'EU|SIG=negativo|web' is itself (12) → rung 2 'EU|SIG=negativo|*' with S4: n=32
  S4  EU | soft=1 auto=0 | tele   n=20   rate .35   → B: rung 1 'EU|SIG=negativo|tele' is itself (20) → rung 2 'EU|SIG=negativo|*' with S3
  S5  EU | soft=0 auto=1 | web    n=6    rate .95   → S: SIG=pos alone, top of its ladder still < 30
  S6  EU | soft=1 auto=1 | web    n=4    rate .60   → M: mixed sign, never pooled
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# The project is a flat folder imported from notebooks and scripts alike: make sure the
# folder of this file is importable BEFORE importing the sibling modules below (that is
# why those imports come after this block, not at the top).
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if PROJECT_FOLDER not in sys.path:
    sys.path.insert(0, PROJECT_FOLDER)

from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
import raw_data_validation

LADDER_TAXONOMY = dict(business_mandatory_dims=["region"],
                       structural_timevarying_dims={"softcancel": "negative", "autorenew": "positive"},
                       extra_renovacion=["channel"],
                       extra_revalorizacion=["discount"],
                       ignore_cols=[])
LADDER_MONTHS = pd.period_range("2024-01", "2026-03", freq="M")
LADDER_SERIES = [  # (region, softcancel, autorenew, channel, units, rate)
    ("EU", 0, 0, "web", 200, .80),
    ("EU", 0, 0, "tele", 10, .80),
    ("EU", 1, 0, "web", 12, .35),
    ("EU", 1, 0, "tele", 20, .35),
    ("EU", 0, 1, "web", 6, .95),
    ("EU", 1, 1, "web", 4, .60),
]


def ladder_role(month: pd.Period) -> str:
    return ROLE_PROJECTION if str(month) >= "2026-01" else ROLE_TRAIN


def build_ladder_raw(seed: int = 3, seasonal_series: bool = False, trend_series: bool = False) -> pd.DataFrame:
    """The six-series raw. Optional: a 7th seasonal series and an 8th trending one (n=300)."""
    rng = np.random.default_rng(seed)
    rows = []
    series = list(LADDER_SERIES)
    if seasonal_series:
        series.append(("NA", 0, 0, "web", 300, "seasonal"))
    if trend_series:
        series.append(("NA", 0, 0, "tele", 300, "trend"))
    for region, softcancel, autorenew, channel, units, rate in series:
        for index, month in enumerate(LADDER_MONTHS):
            role = ladder_role(month)
            if rate == "seasonal":
                p = .70 + .10 * np.sin(2 * np.pi * (month.month - 1) / 12)
            elif rate == "trend":
                p = .85 - .008 * index
            else:
                p = rate
            renewed = np.nan if role == ROLE_PROJECTION else rng.binomial(units, min(.99, max(.01, p)))
            rows.append(dict(period=str(month), dataset_role=role, is_current_month=int(str(month) == "2026-01"),
                             flag_time_series=0, total_tr_units=float(units), total_tr_usd=float(units * 20),
                             total_renewed_units=renewed, total_renewed_usd=(renewed * 20 * 1.05) if role != ROLE_PROJECTION else np.nan,
                             total_reacquired_units=0.0, total_reacquired_usd=0.0, TR_AUV=20.0, REN_AUV=21.0, ReAC_AUV=0.0,
                             region=region, softcancel=softcancel, autorenew=autorenew, channel=channel, discount="d0"))
    return pd.DataFrame(rows)


@dataclass
class LadderConfig(Config):
    """Config on the ladder taxonomy whose raw is `build_ladder_raw`."""
    seasonal_series: bool = False
    trend_series: bool = False

    def read_raw(self) -> pd.DataFrame:
        return build_ladder_raw(seasonal_series=self.seasonal_series, trend_series=self.trend_series)


def ladder_config(temporary_directory: str, **overrides) -> Config:
    """A LadderConfig writing to a sqlite in the temporary directory."""
    engine = create_engine(f"sqlite:///{os.path.join(temporary_directory, 'test.db')}")
    arguments = dict(LADDER_TAXONOMY)
    arguments.update(dict(sql_engine=engine, sql_schema=None, outdir=temporary_directory, backtest_test_start="2025-10",
                          pending_close_months=0,      # the fixture's last month is closed
                          own_rate_floor=30.0,         # the six-series ladder was designed with one floor
                          signed_ladder_max_loss=0.0,  # and with the strict sign rule
                          challenger_technique="T2_mean",   # and the mean as challenger
                          backtest_max_targets=24, backtest_screen_horizons=[1, 2, 3, 6, 12], backtest_horizon_cap=12,
                          backtest_horizon_bands={"h1": [1, 1], "corto": [2, 3], "medio": [4, 6], "largo": [7, 12]},
                          challenger_margin_by_band={"h1": 0.10, "corto": 0.10, "medio": 0.05, "largo": 0.0}))
    arguments.update(overrides)
    return LadderConfig(**arguments)


@contextlib.contextmanager
def quiet():
    """Swallow the console of the setup steps so the test output shows only the checks."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


def phase_0_units(configuration: Config) -> tuple:
    """Run phase 0 quietly and return (fine_table, labeled_units)."""
    with quiet():
        conditioned = raw_data_validation.apply_current_month_doctrine(
            raw_data_validation.validate_raw(configuration.read_raw(), configuration), configuration)
        fine = raw_data_validation.build_fine_table(conditioned, configuration)
        units = raw_data_validation.aggregate_to_forecast_units(fine, configuration)
        labeled = raw_data_validation.label_universe_and_routes(units, configuration)
    return fine, labeled
