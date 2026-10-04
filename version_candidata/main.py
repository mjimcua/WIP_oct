"""
main.py — Runs SFF on the production extract, or on the synthetic raw.

    python main.py              # the production extract (SQL Server)
    python main.py sintetico    # the synthetic raw of the tests

The Config is declared here and only here: its columns, its calendar and where the raw
comes from. Each step adds one line to run().
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

from pipeline import Orchestrator
from config import Config


# ─── named constants ─────────────────────────────────────────────────────────────
SQL_SERVER = "..."                          # [por completar]
SQL_DATABASE = "..."                        # [por completar]
SQL_DRIVER = "ODBC Driver 17 for SQL Server"
RAW_EXTRACT_QUERY = "SELECT * FROM ..."     # [por completar] the query of the raw extract
SYNTHETIC_KEYWORD = "sintetico"
SYNTHETIC_OUTPUT_FOLDER = "salida"
SYNTHETIC_DATABASE = os.path.join(SYNTHETIC_OUTPUT_FOLDER, "sff_sintetico.db")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TWO CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════════

def production_engine():
    """The production SQL Server: the raw is read from it and the tables are written to it.
    fast_executemany: pyodbc sends every chunk of to_sql in one round trip (without it, one
    INSERT per row: a table of a million rows takes minutes instead of seconds)."""
    connection_url = (f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DATABASE}"
                      f"?driver={SQL_DRIVER.replace(' ', '+')}&trusted_connection=yes")
    return create_engine(connection_url, fast_executemany=True)


class ProductionConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        return pd.read_sql(self.raw_extract_sql, self.sql_engine)


class SyntheticConfig(Config):
    def read_raw(self) -> pd.DataFrame:
        from synthetic_v3 import build_raw
        return build_raw(7, with_discount_pct=True)


def production_configuration() -> Config:
    """The production extract, as seen in the runs of 29-sep. Names marked [por confirmar]
    must be checked against it: step 00 names every column that is missing or has no role."""
    return ProductionConfig(
        # where the raw is read from and the tables are written to
        sql_engine=production_engine(),
        sql_schema="dbo",
        raw_extract_sql=RAW_EXTRACT_QUERY,
        # the calendar
        current_month="01/09/2026",
        test_months=3,
        # the dimensions
        business_mandatory_dims=["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
                                 "tr_product_level_1", "tr_product_level_2", "tr_purchase_type", "tr_renewal_type",
                                 "tr_term", "tr_band", "tr_master_partner_code"],
        # term and band keep their raw value; step 01 adds tr_term_level_1 / tr_band_level_1 (their values
        # grouped by the renewal rate). The groups are kept in salida/sff_levels.json
        leveled_dims={"tr_term": "ordinal", "tr_band": "ordinal"},
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "not_installed": "negative"},            # [por confirmar] the signs
        # softcancel is not final until the due date (payment attempts, grace period): step 17 adjusts next to the
        # forecast the units expected to be marked before falling due
        maturation_flag_col="softcancel",
        extra_renovacion=["net_new", "prev_OperationGroup"],
        extra_revalorizacion=["price_cap", "msrp_increased"],
        # the discount: ONE column, both sides (the bucket is derived with the default edges,
        # the same as the extract's old discount_interval)
        discount_value_column="discount",
        # the other columns of the extract
        extra_measure_cols=["total_reacquired_units", "total_reacquired_usd"],   # an AUV is USD / units: computed, not read
        # the exact base of the uplift: the USD that was falling due of the contracts that RENEWED (built per
        # licence and summed). With it, the uplift is pure revaluation, conditional on renewing
        renewed_pipeline_usd_col="total_tr_usd_renewed",
        # the ISOLATED renewals: here, the contracts that never had a softcancel (before, during or after the
        # renewal): the reference to see a price increase without the retention discounts
        isolated_pipeline_units_col="total_tr_units_without_softcancel",
        isolated_renewed_units_col="total_renewed_units_without_softcancel",
        isolated_renewed_usd_col="total_renewed_usd_without_softcancel",
        isolated_renewed_pipeline_usd_col="total_tr_usd_renewed_without_softcancel",
        # the dispersion of the isolated renewals (built per licence, summed): the second moment and six ratio bands
        isolated_renewed_usd_sq_over_tr_col="total_renewed_usd_sq_over_tr_isolated",
        isolated_tr_usd_renewed_lt095_col="total_tr_usd_renewed_isolated_lt095",
        isolated_tr_usd_renewed_095_100_col="total_tr_usd_renewed_isolated_095_100",
        isolated_tr_usd_renewed_100_105_col="total_tr_usd_renewed_isolated_100_105",
        isolated_tr_usd_renewed_105_110_col="total_tr_usd_renewed_isolated_105_110",
        isolated_tr_usd_renewed_110_120_col="total_tr_usd_renewed_isolated_110_120",
        isolated_tr_usd_renewed_ge120_col="total_tr_usd_renewed_isolated_ge120",
        ignore_cols=["dataset_role", "is_current_month", "dummy_field", "row_id", "_filter"],   # [por confirmar]
        # the simulation window (current month → December): what happens in it falls due in 2027
        term_column="tr_term",
        one_year_term_value="1 year",
        acquisition_column="net_new",
        acquisition_values=["Acquisition_Not-New", "Acquisition_Pure-New"],
        acquisition_discount=0.4,
        # the time_series universe (retail to subscription): its region and its projection
        ts_region_columns=["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3"],   # projected by country
    )


def synthetic_configuration() -> Config:
    """The synthetic raw of the tests: 2023-01..2026-12, current month 2026-09. Its tables
    go to a SQLite file in salida/."""
    os.makedirs(SYNTHETIC_OUTPUT_FOLDER, exist_ok=True)
    return SyntheticConfig(
        sql_engine=create_engine(f"sqlite:///{SYNTHETIC_DATABASE}"),
        current_month="2026-09",
        test_months=3,
        business_mandatory_dims=["region", "product"],
        structural_timevarying_dims={"dormant": "negative", "softcancel": "negative",
                                     "no_instalado": "negative", "autorenew": "positive"},
        maturation_flag_col="softcancel",
        extra_renovacion=["channel"],
        extra_revalorizacion=["newcust"],
        # the synthetic carries the exact discount as discount_pct and its own bucket as
        # discount: the framework derives the bucket, so the synthetic's is ignored
        discount_value_column="discount_pct",
        ignore_cols=["dataset_role", "is_current_month", "discount"],
        renewed_pipeline_usd_col="total_tr_usd_renewed",
        isolated_pipeline_units_col="total_tr_units_without_softcancel",
        isolated_renewed_units_col="total_renewed_units_without_softcancel",
        isolated_renewed_usd_col="total_renewed_usd_without_softcancel",
        isolated_renewed_pipeline_usd_col="total_tr_usd_renewed_without_softcancel",
        # the dispersion of the isolated renewals (built per licence, summed): the second moment and six ratio bands
        isolated_renewed_usd_sq_over_tr_col="total_renewed_usd_sq_over_tr_isolated",
        isolated_tr_usd_renewed_lt095_col="total_tr_usd_renewed_isolated_lt095",
        isolated_tr_usd_renewed_095_100_col="total_tr_usd_renewed_isolated_095_100",
        isolated_tr_usd_renewed_100_105_col="total_tr_usd_renewed_isolated_100_105",
        isolated_tr_usd_renewed_105_110_col="total_tr_usd_renewed_isolated_105_110",
        isolated_tr_usd_renewed_110_120_col="total_tr_usd_renewed_isolated_110_120",
        isolated_tr_usd_renewed_ge120_col="total_tr_usd_renewed_isolated_ge120",
        # the simulation window: acquisitions are newcust = 1
        acquisition_column="newcust",
        acquisition_values=[1],
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RUN
# ═══════════════════════════════════════════════════════════════════════════════════

def run(configuration: Config, from_step: str = None) -> dict:
    """Every step, in the order of their dependencies (pipeline.py: contracts, step interface, orchestrator).
    With from_step, the steps before it are loaded from the last checkpoint and it and the rest run."""
    orchestrator = Orchestrator(configuration)
    return orchestrator.run_from(from_step) if from_step else orchestrator.run()


if __name__ == "__main__":
    # python main.py [sintetico] [--desde <step>]    e.g. python main.py --desde forecast
    arguments = sys.argv[1:]
    from_step = arguments[arguments.index("--desde") + 1] if "--desde" in arguments else None
    configuration = synthetic_configuration() if SYNTHETIC_KEYWORD in arguments else production_configuration()
    run(configuration, from_step=from_step)
