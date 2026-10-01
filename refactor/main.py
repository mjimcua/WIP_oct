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
from step_10_ladder_groups import build_ladder_groups
from step_11_ladder import climb_the_ladder
from step_12_pool_series import build_pool_series
from step_13_dynamics import measure_dynamics
from step_14_backtest import run_backtest
from step_15_uplift import estimate_uplift
from step_16_uplift_backtest import backtest_uplift
from step_17_forecast import assemble_forecast
from step_18_validation import validate_chain
from step_19_portfolio_exam import examine_portfolio
from step_20_time_series import build_time_series_and_total, split_time_series_rows
from step_audit import build_audit_tables
from step_nucleo import build_core_table
from step_informe import build_report


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
        # the simulation window (current month → December): what happens in it falls due in 2027
        term_column="tr_term_level_2",
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
        extra_renovacion=["channel"],
        extra_revalorizacion=["newcust"],
        # the synthetic carries the exact discount as discount_pct and its own bucket as
        # discount: the framework derives the bucket, so the synthetic's is ignored
        discount_value_column="discount_pct",
        ignore_cols=["dataset_role", "is_current_month", "discount"],
        # the simulation window: acquisitions are newcust = 1
        acquisition_column="newcust",
        acquisition_values=[1],
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
    validated_raw, time_series_rows = split_time_series_rows(validated_raw, configuration)   # the time_series universe waits for step 20
    validated_raw = validate_values(validated_raw, configuration)        # step 01
    calendared_raw = apply_calendar(validated_raw, configuration)        # step 02
    fine_table = build_fine_table(calendared_raw, configuration)         # step 03
    forecast_units = build_forecast_units(fine_table, configuration)     # step 04
    lookups = build_lookups(fine_table, forecast_units, configuration)   # step 05
    series_table = build_series_routes(forecast_units, configuration)    # step 06
    support_bound = build_support_bound(forecast_units, configuration)   # step 07
    rated_units, series_rate = build_rate_series(forecast_units, series_table, configuration)   # step 08
    dimension_decision, dimension_pairs = analyse_dimensions(series_rate, lookups["lookup_fs"], configuration)   # step 09
    ladder = build_ladder_groups(rated_units, series_rate, lookups["lookup_fs"], dimension_decision,
                                 configuration)                                                   # step 10
    series_estimate, money_by_level = climb_the_ladder(series_rate, ladder, configuration)   # step 11
    pool_series, pool_reference = build_pool_series(rated_units, ladder, series_estimate, configuration)   # step 12
    pool_dynamics, portfolio_profile, portfolio_dynamics = measure_dynamics(pool_series, pool_reference, configuration)   # step 13
    backtest = run_backtest(pool_series, pool_reference, configuration)                            # step 14
    uplift_cells, contract_check = estimate_uplift(fine_table, configuration)                       # step 15
    uplift_backtest, uplift_verdict = backtest_uplift(fine_table, configuration)                     # step 16
    forecast = assemble_forecast(fine_table, forecast_units, series_estimate, pool_series, pool_reference, backtest,
                                 uplift_cells, uplift_verdict, rated_units, configuration)          # step 17
    results = dict(pool_series=pool_series, pool_reference=pool_reference,
                   backtest=backtest, pool_dynamics=pool_dynamics, portfolio_profile=portfolio_profile,
                   portfolio_dynamics=portfolio_dynamics, ladder=ladder,
                   series_estimate=series_estimate, money_by_level=money_by_level,
                   dimension_decision=dimension_decision, dimension_pairs=dimension_pairs,
                   raw=calendared_raw, fine_table=fine_table, forecast_units=forecast_units, lookups=lookups,
                   series=series_table, support_bound=support_bound, rated_units=rated_units, series_rate=series_rate)
    results.update(uplift_cells=uplift_cells, contract_check=contract_check, uplift_backtest=uplift_backtest,
                   uplift_verdict=uplift_verdict, forecast=forecast)
    results["portfolio_exam"], results["portfolio_exam_summary"] = examine_portfolio(
        rated_units, series_estimate, pool_series, backtest, configuration,
        reference_members=ladder["reference_members"])                                               # step 19
    results["time_series"], results["forecast_total"] = build_time_series_and_total(
        time_series_rows, fine_table, forecast["forecast"], configuration)                      # step 20
    results["time_series_rows"] = time_series_rows
    results["audit"] = build_audit_tables(ladder, series_rate, series_estimate, rated_units, pool_series, pool_reference,
                                          pool_dynamics, backtest, forecast["forecast"], configuration)   # the satellites
    results["core"], results["core_legend"] = build_core_table(
        fine_table, configuration, forecast_units=forecast_units, support_bound=support_bound, rated_units=rated_units,
        series_table=series_table, series_rate=series_rate, series_estimate=series_estimate, pool_dynamics=pool_dynamics,
        technique_decision=backtest["decision"], exam_by_pool=backtest["exam_by_pool"], forecast=forecast["forecast"],
        time_series_rows=time_series_rows, time_series_table=results["time_series"],
        forecast_total=results["forecast_total"], ladder_stages=ladder["stages"],
        series_dynamics=results["audit"]["series_dynamics"], series_exam=results["audit"]["series_exam"])
                                                 # the core: every row, every decision
    results["validation"] = validate_chain(raw, results, configuration)                            # step 18 (after 19: it reads its exam)
    results["card"] = build_report(raw, results, configuration)                                   # the report, last
    return results


if __name__ == "__main__":
    if SYNTHETIC_KEYWORD in sys.argv[1:]:
        run(synthetic_configuration())
    else:
        run(production_configuration())
