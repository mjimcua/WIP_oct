"""
main.py — Runs SFF on Kamelot, or on the synthetic raw.

    python main.py              # Kamelot (SQL Server)
    python main.py sintetico    # the synthetic raw of the tests

The Config is declared here and only here: its columns, its calendar and where the raw
comes from. Each step adds one line to run().
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import time

import pandas as pd
from sqlalchemy import create_engine

from config import Config
from step_00_validate_raw import validate_raw
from step_01_validate_values import validate_values
from step_02_apply_calendar import apply_calendar
from step_03_fine_table import build_fine_table
from step_04_forecast_units import build_forecast_units
from step_05_lookups import build_lookups
from step_06_series_routes import build_series_routes
from step_07_support_bound import build_support_bound
from step_08_rate_series import build_rate_series
from step_09_dimensions import analyse_dimensions
from step_10_relatives import build_relatives
from step_11_ladder import climb_the_ladder


# ─── named constants ─────────────────────────────────────────────────────────────
KAMELOT_SERVER = "..."                      # [por completar]
KAMELOT_DATABASE = "Kamelot"
KAMELOT_DRIVER = "ODBC Driver 17 for SQL Server"
RAW_EXTRACT_QUERY = "SELECT * FROM ..."     # [por completar] the query of the raw extract
SYNTHETIC_KEYWORD = "sintetico"
SYNTHETIC_OUTPUT_FOLDER = "salida"
SYNTHETIC_DATABASE = os.path.join(SYNTHETIC_OUTPUT_FOLDER, "sff_sintetico.db")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TWO CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════════

def kamelot_engine():
    """The SQL Server of Kamelot: the raw is read from it and the tables are written to it."""
    connection_url = (f"mssql+pyodbc://@{KAMELOT_SERVER}/{KAMELOT_DATABASE}"
                      f"?driver={KAMELOT_DRIVER.replace(' ', '+')}&trusted_connection=yes")
    return create_engine(connection_url, fast_executemany=True)


class KamelotConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        return pd.read_sql(self.raw_extract_sql, self.sql_engine)


class SyntheticConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        from synthetic_v3 import build_raw
        return build_raw(7, with_discount_pct=True)


def kamelot_configuration() -> Config:
    """The Kamelot extract, as seen in the runs of 29-sep. Names marked [por confirmar]
    must be checked against it: step 00 names every column that is missing or has no role."""
    return KamelotConfig(
        # where the raw is read from and the tables are written to
        sql_engine=kamelot_engine(),
        sql_schema="dbo",
        raw_extract_sql=RAW_EXTRACT_QUERY,
        # the calendar
        current_month="01/09/2026",
        test_months=3,
        pending_close_months=0,
        # the dimensions
        business_mandatory_dims=["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
                                 "tr_product_level_1", "tr_product_level_2", "tr_purchase_type", "tr_renewal_type",
                                 "tr_term_level_1", "tr_term_level_2", "tr_band_level_1", "tr_band_level_2",
                                 "tr_master_partner_code"],
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "not_installed": "negative"},            # [por confirmar] the signs
        extra_renovacion=["net_new", "prev_OperationGroup"],
        extra_revalorizacion=["price_cap", "msrp_increased"],
        # the discount: ONE column, both sides (the bucket is derived with the default edges,
        # the same as the extract's old discount_interval)
        discount_value_column="discount",
        # the other columns of the extract
        extra_measure_cols=["total_reacquired_units", "total_reacquired_usd", "TR_AUV", "REN_AUV", "ReAC_AUV"],
        ignore_cols=["dataset_role", "is_current_month", "dummy_field", "row_id", "_filter"],   # [por confirmar]
    )


def synthetic_configuration() -> Config:
    """The synthetic raw of the tests: 2023-01..2026-12, current month 2026-09. Its tables
    go to a SQLite file in salida/."""
    os.makedirs(SYNTHETIC_OUTPUT_FOLDER, exist_ok=True)
    return SyntheticConfig(
        sql_engine=create_engine(f"sqlite:///{SYNTHETIC_DATABASE}"),
        current_month="2026-09",
        test_months=3,
        pending_close_months=0,
        business_mandatory_dims=["region", "product"],
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "no_instalado": "negative", "autorenew": "positive"},
        extra_renovacion=["channel"],
        extra_revalorizacion=["newcust"],
        # the synthetic carries the exact discount as discount_pct and its own bucket as
        # discount: the framework derives the bucket, so the synthetic's is ignored
        discount_value_column="discount_pct",
        ignore_cols=["dataset_role", "is_current_month", "discount"],
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RUN
# ═══════════════════════════════════════════════════════════════════════════════════

def run(configuration: Config) -> dict:
    """The steps, in order. Each step adds its line here when it is built."""
    configuration.logger.doc("[main] ═══ SFF · reading the raw ═══")
    read_start_time = time.time()
    raw = configuration.read_raw()
    configuration.logger.doc(f"[main] raw read: {len(raw):,} rows × {len(raw.columns)} columns "
                             f"({time.time() - read_start_time:.1f}s)")

    validated_raw = validate_raw(raw, configuration)                     # step 00
    validated_raw = validate_values(validated_raw, configuration)        # step 01
    calendared_raw = apply_calendar(validated_raw, configuration)        # step 02
    fine_table = build_fine_table(calendared_raw, configuration)         # step 03
    forecast_units = build_forecast_units(fine_table, configuration)     # step 04
    lookups = build_lookups(fine_table, forecast_units, configuration)   # step 05
    series_table = build_series_routes(forecast_units, configuration)    # step 06
    support_bound = build_support_bound(forecast_units, configuration)   # step 07
    rated_units, series_rate = build_rate_series(forecast_units, series_table, configuration)   # step 08
    dimension_decision, dimension_pairs = analyse_dimensions(series_rate, lookups["lookup_fs"], configuration)   # step 09
    relatives, pools = build_relatives(rated_units, series_rate, lookups["lookup_fs"], dimension_decision,
                                       configuration)                                            # step 10
    series_estimate, money_by_level = climb_the_ladder(series_rate, relatives, pools, configuration)   # step 11
    return dict(relatives=relatives, pools=pools, series_estimate=series_estimate, money_by_level=money_by_level,
                dimension_decision=dimension_decision, dimension_pairs=dimension_pairs, raw=calendared_raw, fine_table=fine_table, forecast_units=forecast_units, lookups=lookups,
                series=series_table, support_bound=support_bound, rated_units=rated_units, series_rate=series_rate)


if __name__ == "__main__":
    if SYNTHETIC_KEYWORD in sys.argv[1:]:
        run(synthetic_configuration())
    else:
        run(kamelot_configuration())
