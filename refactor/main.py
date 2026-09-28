"""
main.py — Runs SFF on Kamelot, or on the synthetic raw.

    python main.py              # Kamelot (SQL Server)
    python main.py sintetico    # the synthetic raw of the tests

The Config is declared here and only here: its columns, its calendar and where the raw
comes from. Today it runs step 00 (validate the raw); each new step adds one line to run().
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import sys

import pandas as pd
from sqlalchemy import create_engine

from config import Config
from step_00_validate_raw import validate_raw


# ─── named constants ─────────────────────────────────────────────────────────────
KAMELOT_SERVER = "..."                      # [por completar]
KAMELOT_DATABASE = "Kamelot"
KAMELOT_DRIVER = "ODBC Driver 17 for SQL Server"
RAW_EXTRACT_QUERY = "SELECT * FROM ..."     # [por completar] the query of the raw extract
SYNTHETIC_KEYWORD = "sintetico"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TWO CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════════

class KamelotConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        connection_url = (f"mssql+pyodbc://@{KAMELOT_SERVER}/{KAMELOT_DATABASE}"
                          f"?driver={KAMELOT_DRIVER.replace(' ', '+')}&trusted_connection=yes")
        return pd.read_sql(RAW_EXTRACT_QUERY, create_engine(connection_url))


class SyntheticConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        from synthetic_v3 import build_raw
        return build_raw(7, with_discount_pct=True)


def kamelot_configuration() -> Config:
    """The Kamelot extract. Names marked [por confirmar] must be checked against it:
    step 00 names every column that is missing or has no role."""
    return KamelotConfig(
        # the calendar
        current_month="01/09/2026",
        test_months=3,
        pending_close_months=0,
        # the dimensions
        business_mandatory_dims=["regional_level_1", "regional_level_2", "regional_level_3",
                                 "product_level_1", "product_level_2", "purchase_type",
                                 "term_level_1", "term_level_2", "band_level_1", "band_level_2"],
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "no_instalado": "negative", "autorenew": "positive"},
        extra_renovacion=["net_new"],                   # [por confirmar] + the channel column, if any
        extra_revalorizacion=["discount_interval"],     # [por confirmar] the discount bucket
        # the other columns of the extract
        extra_measure_cols=["total_reacquired_units", "total_reacquired_usd", "TR_AUV", "REN_AUV", "ReAC_AUV"],
        discount_value_column="discount",               # [por confirmar]
        sku_column="sku",                               # [por confirmar]
        ignore_cols=["dataset_role", "is_current_month", "dummy_field", "row_id", "_filter"],
    )


def synthetic_configuration() -> Config:
    """The synthetic raw of the tests: 2023-01..2026-12, current month 2026-09."""
    return SyntheticConfig(
        current_month="2026-09",
        test_months=3,
        pending_close_months=0,
        business_mandatory_dims=["region", "product"],
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "no_instalado": "negative", "autorenew": "positive"},
        extra_renovacion=["channel"],
        extra_revalorizacion=["discount", "newcust"],
        discount_value_column="discount_pct",
        ignore_cols=["dataset_role", "is_current_month"],
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RUN
# ═══════════════════════════════════════════════════════════════════════════════════

def run(configuration: Config) -> dict:
    """The steps, in order. Each step adds its line here when it is built."""
    raw = configuration.read_raw()
    validated_raw = validate_raw(raw, configuration)                     # step 00
    return dict(raw=validated_raw)


if __name__ == "__main__":
    if SYNTHETIC_KEYWORD in sys.argv[1:]:
        run(synthetic_configuration())
    else:
        run(kamelot_configuration())
