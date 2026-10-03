"""
step_22_power_bi.py — The adaptation to Power BI: the auxiliary tables the report needs, so that
Power BI only relates and sums.

The core (sff_nucleo) and the forecast series dimension (sff_forecast_series) carry the numbers. What
Power BI still needs is the context to show them to business: a name for every code, an order, a block.
That context is built here, once, in Python, instead of as calculated tables or DAX inside the report.

Tables written (one per auxiliary dimension; this step will grow with the report):

  sff_forecast_pipeline_source   one row per value of forecast_pipeline_source (the origin of every
                                 row of the core): its business label, the block it belongs to and its
                                 order in the report. Relation in Power BI:
                                     sff_forecast_pipeline_source[forecast_pipeline_source]
                                         1 → n  sff_nucleo[forecast_pipeline_source]

Actions (logged as they are done):
  1. the table of pipeline sources: code, label, block, order
  2. check it against the core                                           checks 1-2
  3. write it                                                            check 3
  4. count the checks; stop if any failed
  5. show the table

Checks (logged as they are made, numbered, at the level of their status):
   1. every forecast_pipeline_source of the core has its row (no row of the core left without a label)
   2. one row per source (the key is unique: the relation in Power BI is 1 → n)
   3. table sff_forecast_pipeline_source written and read back
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import Config
from step_nucleo import FINAL_ORIGIN_COLUMN, TS_WITHOUT_RESULT
from vocabulario import (ROW_FROM_GAP, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                         TOTAL_ORIGIN_SIMULATED, TS_PROJECTED, TS_REAL, TS_REENTRY)


STEP_LABEL = "22"
STEP_NAME = "ADAPTATION TO POWER BI"
STEP_PURPOSE = ("build the auxiliary tables of the Power BI report (names, orders and blocks of the codes of the core), "
                "so that Power BI only relates and sums: no logic inside the report")
STEP_ACTIONS = ["the table of pipeline sources: code, label, block, order",
                "check it against the core (checks 1-2)",
                "write it (check 3)",
                "count the checks; stop if any failed",
                "show the table"]
STEP_OUTPUT = "table sff_forecast_pipeline_source (one row per forecast_pipeline_source)"

PIPELINE_SOURCE_TABLE = "forecast_pipeline_source"

# the blocks of the report: where each source sits when the pipeline of a year is broken down
BLOCK_EXTRACT_PIPELINE = "Extract pipeline"        # due dates already in the extract
BLOCK_EXTENDED_HORIZON = "Extended horizon"        # pipeline built by the framework (simulation window, +12 months)
BLOCK_TIME_SERIES = "Time series"                  # the retail to subscription universe
BLOCK_CONTROL = "Control"                          # rows with no money: they keep the sums equal to the reporting

# one row per source: (code in the core, business label, block, order in the report)
PIPELINE_SOURCES = [
    (TOTAL_ORIGIN_RENEWED,   "Pipeline already due (actual renewal)",    BLOCK_EXTRACT_PIPELINE, 1),
    (TOTAL_ORIGIN_EXPECTED,  "Known pipeline still to fall due",         BLOCK_EXTRACT_PIPELINE, 2),
    (TOTAL_ORIGIN_PROJECTED, "Re-renewal of 1-year licences",            BLOCK_EXTENDED_HORIZON, 3),
    (TOTAL_ORIGIN_SIMULATED, "Simulated acquisition",                    BLOCK_EXTENDED_HORIZON, 4),
    (TS_REAL,                "Retail to subscription (actual)",          BLOCK_TIME_SERIES,      5),
    (TS_PROJECTED,           "Retail to subscription (projected)",       BLOCK_TIME_SERIES,      6),
    (TS_REENTRY,             "Retail to subscription (re-entry)",        BLOCK_TIME_SERIES,      7),
    (ROW_FROM_GAP,           "Months with nothing due",                  BLOCK_CONTROL,          8),
    (TS_WITHOUT_RESULT,      "Time series month not closed yet",         BLOCK_CONTROL,          9),
]


def build_power_bi_tables(core_table: pd.DataFrame, configuration: Config) -> dict:
    """The auxiliary tables of the Power BI report.

    INPUT:   the core (step NU), with its column forecast_pipeline_source.
    OUTPUT:  {"pipeline_source": the table of pipeline sources}.
    RULES:   every code of the core has its row; the table lists every source the framework can produce,
             also those absent in this run (a run without acquisition has no simulated acquisition, and the
             report keeps the same rows).
    EDGE CASES: a code in the core that the table does not know stops the step (check 1): a new origin
             was added to the core and this table was not updated.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the table of pipeline sources
    pipeline_source_table = build_pipeline_source_table()
    configuration.log_action(STEP_LABEL, 1, f"{len(pipeline_source_table)} pipeline sources in "
                                            f"{pipeline_source_table['source_block'].nunique()} blocks")

    # [2] the checks against the core
    configuration.log_action(STEP_LABEL, 2, "checking the table against the core")
    sources_in_core = set(core_table[FINAL_ORIGIN_COLUMN].dropna().unique())
    sources_in_table = set(pipeline_source_table[FINAL_ORIGIN_COLUMN])
    sources_without_row = sorted(sources_in_core - sources_in_table)
    configuration.log_check(STEP_LABEL, check_log,
                            "every forecast_pipeline_source of the core has its row",
                            not sources_without_row,
                            failure_detail=f"sources of the core without a row: {sources_without_row}",
                            context=f"{len(sources_in_core)} sources in the core, "
                                    f"{len(sources_in_table - sources_in_core)} with no rows in this run")

    repeated_sources = pipeline_source_table[pipeline_source_table[FINAL_ORIGIN_COLUMN].duplicated()][FINAL_ORIGIN_COLUMN].tolist()
    configuration.log_check(STEP_LABEL, check_log,
                            "one row per source (the relation in Power BI is 1 → n)",
                            not repeated_sources,
                            failure_detail=f"repeated sources: {repeated_sources}")

    # [3] the write
    configuration.log_action(STEP_LABEL, 3, "writing the table")
    configuration.write_table(STEP_LABEL, check_log, pipeline_source_table, PIPELINE_SOURCE_TABLE)

    # [4] the count
    configuration.log_action(STEP_LABEL, 4, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [5] the table on screen
    configuration.log_action(STEP_LABEL, 5, "the pipeline sources, as Power BI will show them:")
    configuration.show_table(pipeline_source_table)

    return {"pipeline_source": pipeline_source_table}


def build_pipeline_source_table() -> pd.DataFrame:
    """One row per pipeline source: code, business label, block and order."""
    pipeline_source_table = pd.DataFrame(PIPELINE_SOURCES,
                                         columns=[FINAL_ORIGIN_COLUMN, "source_label", "source_block", "source_order"])
    return pipeline_source_table
