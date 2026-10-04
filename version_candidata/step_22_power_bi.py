"""
step_22_power_bi.py — The adaptation to Power BI: the auxiliary tables the report needs, so that
Power BI only relates and sums.

The core (sff_nucleo) and the forecast series dimension (sff_forecast_series) carry the numbers. What
Power BI still needs is the context to show them to business: a name for every code, an order, a block.
That context is built here, once, in Python, instead of as calculated tables or DAX inside the report.

Tables written (one per auxiliary table; this step will grow with the report):

  sff_price_increase_monitor     one row per closed pipeline month: the uplift of everyone, the uplift
                                 of the ISOLATED renewals (the detector: the normal renewal process, no
                                 retention discounts in it), the step against the average of the previous 12
                                 months, and the increase cycle (a price increase affects the renewals
                                 of ONE cycle: who renews pays the new price, but a year later everyone
                                 already bought at it, so the step lasts CYCLE_MONTHS and then fades)
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
import numpy as np
import pandas as pd

from config import (CORE_ISOLATED_BAND_PREFIX, CORE_ISOLATED_PIPELINE_UNITS, CORE_ISOLATED_RENEWED_PIPELINE_USD,
                    CORE_ISOLATED_RENEWED_UNITS, CORE_ISOLATED_RENEWED_USD, CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR,
                    CORE_RENEWED_PIPELINE_USD, ISOLATED_RATIO_BANDS, Config, weighted_ratio_spread)
from step_nucleo import FINAL_ORIGIN_COLUMN, TS_WITHOUT_RESULT
from vocabulario import (ROW_FROM_GAP, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                         TOTAL_ORIGIN_SIMULATED, TS_PROJECTED, TS_REAL, TS_REENTRY)


STEP_LABEL = "22"
STEP_NAME = "ADAPTATION TO POWER BI"
STEP_PURPOSE = ("build the auxiliary tables of the Power BI report (names, orders and blocks of the codes of the core), "
                "so that Power BI only relates and sums: no logic inside the report")
STEP_ACTIONS = ["the table of pipeline sources: code, label, block, order",
                "the price increase monitor: the uplift of the closed months and its steps",
                "check the tables against the core (checks 1-4)",
                "write them (checks 5-6)",
                "count the checks; stop if any failed",
                "show the tables"]
STEP_OUTPUT = ("tables sff_forecast_pipeline_source (one row per forecast_pipeline_source) and "
               "sff_price_increase_monitor (one row per closed pipeline month)")

PIPELINE_SOURCE_TABLE = "forecast_pipeline_source"
PRICE_MONITOR_TABLE = "price_increase_monitor"

# the detector of price increases, over the uplift of the isolated renewals (no retention event):
PRICE_STEP_THRESHOLD = 0.03    # a month this far above the average of its previous 12 is above the threshold
PERSISTENCE_MONTHS = 3         # an increase is a step that STAYS: this many consecutive months above the threshold
                               # (one noisy month is not an increase; the price of it: a step is confirmed 2 months late)
CYCLE_MONTHS = 12              # the effect of an increase lasts one renewal cycle: 12 months of due dates
MIN_REFERENCE_MONTHS = 6       # fewer previous months than this: no reference yet, no flag

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

    # [2] the price increase monitor
    price_monitor = price_increase_monitor(core_table, configuration)
    flagged_months = price_monitor[price_monitor["price_increase_flag"] == 1]["period"].tolist()
    configuration.log_action(STEP_LABEL, 2, f"{len(price_monitor):,} closed pipeline months in the monitor · detector on "
                                            f"{price_monitor.attrs['detector_series']} · increases flagged: "
                                            f"{flagged_months if flagged_months else 'none'}")

    # [3] the checks against the core
    configuration.log_action(STEP_LABEL, 3, "checking the tables against the core")
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

    with_renewals = price_monitor[price_monitor["renewed_units"] > 0]
    configuration.log_check(STEP_LABEL, check_log,
                            "every closed pipeline month with renewals has its uplift in the monitor",
                            with_renewals["uplift"].notna().all(),
                            failure_detail=f"{int(with_renewals['uplift'].isna().sum())} months with renewals and no uplift",
                            context=f"{len(price_monitor):,} months")
    flags_without_reference = price_monitor[(price_monitor["price_increase_flag"] == 1)
                                            & price_monitor["uplift_previous_12"].isna()]
    configuration.log_check(STEP_LABEL, check_log,
                            f"an increase is only flagged against a reference of ≥ {MIN_REFERENCE_MONTHS} previous months",
                            flags_without_reference.empty,
                            failure_detail=f"{len(flags_without_reference)} months flagged without a reference")

    # [4] the writes
    configuration.log_action(STEP_LABEL, 4, "writing the tables")
    configuration.write_table(STEP_LABEL, check_log, pipeline_source_table, PIPELINE_SOURCE_TABLE)
    configuration.write_table(STEP_LABEL, check_log, price_monitor.drop(columns=["renewed_units"]), PRICE_MONITOR_TABLE)

    # [5] the count
    configuration.log_action(STEP_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [6] the tables on screen
    configuration.log_action(STEP_LABEL, 6, "the pipeline sources, as Power BI will show them, and the last 13 months "
                                            "of the price increase monitor:")
    configuration.show_table(pipeline_source_table)
    configuration.show_table(price_monitor.drop(columns=["renewed_units"]).tail(13))

    return {"pipeline_source": pipeline_source_table, "price_monitor": price_monitor.drop(columns=["renewed_units"])}



def price_increase_monitor(core_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per closed pipeline month: the uplift of everyone, the uplift of the ISOLATED renewals
    (the normal renewal process, no retention event), and the detection of price increases over it.

    INPUT:   the core. The closed-month measures are read IF PRESENT, by their fixed core names
             (CORE_* in config.py, whatever the extract calls them): the exact uplift base and the four
             isolated measures. Each metric needs only its own columns, so a missing one blanks only the
             metrics that use it. Each uplift uses its exact base when it is there and the approximation
             renewed units × the row's due price when it is not.
    OUTPUT:  one row per period: uplift, uplift_isolated, share_due_isolated, rate_units_isolated,
             desv_isolated (the std deviation of the ratio between isolated licences),
             share_isolated_near_100 (their value in [0.95, 1.05)), share_isolated_ge110 (at or above 1.10),
             uplift_previous_12 (average of the previous 12 months, NaN with fewer than
             MIN_REFERENCE_MONTHS), uplift_step (the month over that average, − 1), above_threshold (step
             ≥ PRICE_STEP_THRESHOLD), price_increase_flag (1 in the FIRST month of a run of ≥
             PERSISTENCE_MONTHS months above the threshold: an increase is a step that stays, one noisy
             month is not), in_increase_cycle (a flagged month in this or the previous CYCLE_MONTHS − 1
             months) and months_since_increase. A run still too short at the end of the history is not
             flagged yet: it is confirmed when its months close.
    RULES:   the detector runs on uplift_isolated when it can be computed (no retention discounts in it),
             on the plain uplift otherwise; attrs["detector_series"] says which.
             A price increase raises the uplift of the renewals of ONE cycle: who renews pays the new
             tariff against the old one. A cycle later everyone due already bought at the new tariff and
             the step fades, so in_increase_cycle marks exactly CYCLE_MONTHS months per detected step.
    """
    closed = core_table[(core_table["forecast_status"] == "actual") & (core_table["forecast_universe"] == "pipeline")
                        & (core_table["forecast_to_renew_units"] > 0)].copy()
    closed["_precio_fila"] = closed["forecast_to_renew_USD"] / closed["forecast_to_renew_units"]
    closed["_base_aproximada"] = closed["forecast_renewed_units"] * closed["_precio_fila"]
    present = set(closed.columns)
    if CORE_ISOLATED_RENEWED_UNITS in present:
        closed["_base_aproximada_isolated"] = closed[CORE_ISOLATED_RENEWED_UNITS] * closed["_precio_fila"]

    summed = closed.groupby("period").sum(numeric_only=True)
    monitor = pd.DataFrame(index=summed.index)
    monitor["renewed_units"] = summed["forecast_renewed_units"]
    uplift_base = summed[CORE_RENEWED_PIPELINE_USD] if CORE_RENEWED_PIPELINE_USD in present else summed["_base_aproximada"]
    monitor["uplift"] = summed["forecast_renewed_USD"] / uplift_base.where(uplift_base > 0)

    # the isolated renewals: each metric with the columns it needs, and nothing else
    monitor["share_due_isolated"] = np.nan
    monitor["rate_units_isolated"] = np.nan
    monitor["uplift_isolated"] = np.nan
    if CORE_ISOLATED_PIPELINE_UNITS in present:
        monitor["share_due_isolated"] = summed[CORE_ISOLATED_PIPELINE_UNITS] / summed["forecast_to_renew_units"]
        if CORE_ISOLATED_RENEWED_UNITS in present:
            due_isolated = summed[CORE_ISOLATED_PIPELINE_UNITS]
            monitor["rate_units_isolated"] = summed[CORE_ISOLATED_RENEWED_UNITS] / due_isolated.where(due_isolated > 0)
    isolated_base = (summed[CORE_ISOLATED_RENEWED_PIPELINE_USD] if CORE_ISOLATED_RENEWED_PIPELINE_USD in present
                     else summed["_base_aproximada_isolated"] if CORE_ISOLATED_RENEWED_UNITS in present else None)
    if CORE_ISOLATED_RENEWED_USD in present and isolated_base is not None:
        monitor["uplift_isolated"] = summed[CORE_ISOLATED_RENEWED_USD] / isolated_base.where(isolated_base > 0)

    # the dispersion of the isolated renewals, licence by licence (from the sums the extract builds per licence)
    monitor["desv_isolated"] = np.nan
    monitor["share_isolated_near_100"] = np.nan
    monitor["share_isolated_ge110"] = np.nan
    if CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR in present and CORE_ISOLATED_RENEWED_PIPELINE_USD in present:
        monitor["desv_isolated"] = weighted_ratio_spread(summed[CORE_ISOLATED_RENEWED_PIPELINE_USD],
                                                         summed[CORE_ISOLATED_RENEWED_USD],
                                                         summed[CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR])
    band_columns = {short_name: CORE_ISOLATED_BAND_PREFIX + short_name for _, short_name, _, _ in ISOLATED_RATIO_BANDS}
    if all(column in present for column in band_columns.values()):
        band_total = summed[list(band_columns.values())].sum(axis=1)
        near = summed[band_columns["095_100"]] + summed[band_columns["100_105"]]
        high = summed[band_columns["110_120"]] + summed[band_columns["ge120"]]
        monitor["share_isolated_near_100"] = near / band_total.where(band_total > 0)
        monitor["share_isolated_ge110"] = high / band_total.where(band_total > 0)
    monitor = monitor.sort_index().reset_index()

    # the detector: the chosen series against the average of its previous 12 months
    detector_series = "uplift_isolated" if monitor["uplift_isolated"].notna().any() else "uplift"
    detector = monitor[detector_series]
    monitor["uplift_previous_12"] = detector.shift(1).rolling(window=12, min_periods=MIN_REFERENCE_MONTHS).mean()
    monitor["uplift_step"] = detector / monitor["uplift_previous_12"] - 1
    above = ((monitor["uplift_step"] >= PRICE_STEP_THRESHOLD) & monitor["uplift_previous_12"].notna()).to_numpy()
    monitor["above_threshold"] = above.astype(int)
    flags = np.zeros(len(monitor), dtype=int)
    position = 0
    while position < len(above):
        if not above[position]:
            position += 1
            continue
        run_end = position
        while run_end < len(above) and above[run_end]:
            run_end += 1
        if run_end - position >= PERSISTENCE_MONTHS:
            flags[position] = 1                      # the increase starts where the run starts
        position = run_end
    monitor["price_increase_flag"] = flags
    monitor["in_increase_cycle"] = (monitor["price_increase_flag"].rolling(window=CYCLE_MONTHS, min_periods=1)
                                    .max().astype(int))
    months_since = []
    last_flagged_position = None
    for position, flagged in enumerate(monitor["price_increase_flag"]):
        if flagged:
            last_flagged_position = position
        months_since.append(position - last_flagged_position if last_flagged_position is not None else np.nan)
    monitor["months_since_increase"] = months_since
    monitor.attrs["detector_series"] = detector_series
    return monitor

def build_pipeline_source_table() -> pd.DataFrame:
    """One row per pipeline source: code, business label, block and order."""
    pipeline_source_table = pd.DataFrame(PIPELINE_SOURCES,
                                         columns=[FINAL_ORIGIN_COLUMN, "source_label", "source_block", "source_order"])
    return pipeline_source_table
