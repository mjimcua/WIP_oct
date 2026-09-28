"""main.py — SFF v3 · production entry point.

Three things, nothing else: a Config whose `read_raw` returns the extract, the engine
the tables go to, and one of the two runners. No parameters, no files.

    python main.py             # the monthly RUN (reads the decisions of the last analysis)
    python main.py analysis    # the ANALYSIS (decides, writes decision_*, produces the forecast)

The raw comes from your own query; write it in `KamelotConfig.read_raw` below. The
framework assumes only that the method returns a DataFrame with every column of the
extract: the column contract (phase 0) checks the rest.

The test entry point is `test_pipeline.py`: the same lines with `read_raw` returning
the synthetic dataset and a throw-away sqlite engine.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys

import pandas as pd

# every module lives in this folder; make sure it is importable.
# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from pipeline import run_analysis, run_pipeline


# ─── named constants ─────────────────────────────────────────────────────────────
# Fill in with the query that produces the raw extract (every declared column).
RAW_EXTRACT_QUERY = "SELECT * FROM ..."
ANALYSIS_KEYWORD = "analysis"


class KamelotConfig(Config):
    """Production Config: `Config()` defaults (the production taxonomy) + the raw query."""

    def read_raw(self) -> pd.DataFrame:
        """The raw extract, read with the configured engine."""
        return pd.read_sql(RAW_EXTRACT_QUERY, self.engine)


if __name__ == "__main__":
    configuration = KamelotConfig(
        sql_server="...", sql_database="Kamelot",
        # ── horizon: the whole of 2027; the re-entries of projected renewals and the
        #    acquisition of the projection months are simulated (labelled) up to here
        extended_horizon_end="2027-12",
        # ── only 1-year contracts re-enter within the horizon: 2- and 3-year ones renewed
        #    now fall due in 2028-2029, and the ones due in 2026-2027 already exist
        term_column="term_level_2",
        term_months_by_value={"1 year": 12, "2 year": 24, "3 year": 36},
        extension_row_filter={"term_level_2": ["1 year"]},
        # ── uplift cells on the dims that move the price (10 mandatory dims made 26,090
        #    cells, 70 % of them under the floor)
        uplift_mandatory_dims=["regional_level_1", "product_level_1", "purchase_type", "term_level_2"],
        uplift_parent_keep_columns=["net_new"],
        # ── the calendar from the current month: the month before it is not closed
        #    (pending), the 6 before those are the exam, everything earlier trains
        pending_close_months=1, test_months=6,
        # ── the seasonality benchmark: top 5 big neutral series inside each region × product family
        benchmark_group_dims=["regional_level_1", "product_level_1"],
        # ── the spreadsheet baseline the business compares against
        baseline_grains=["global", "regional_level_1+product_level_1+purchase_type", "mandatory"],
        # ── experiment: let the techniques learn from the last 24 months only (pools keep all)
        technique_history_months=None,
    )
    if ANALYSIS_KEYWORD in sys.argv[1:]:
        run_analysis(configuration)
    else:
        run_pipeline(configuration)
