"""
step_22_power_bi.py — The adaptation to Power BI: the auxiliary tables the report needs, so that
Power BI only relates and sums.

The core (sff_nucleo) and the forecast series dimension (sff_forecast_series) carry the numbers. What
Power BI still needs is the context to show them to business: a name for every code, an order, a block.
That context is built here, once, in Python, instead of as calculated tables or DAX inside the report.

Tables written (one per auxiliary table; this step will grow with the report):

  sff_price_increase_calendar    one row per increase detected: price group, start month, size of the
                                 step, uplift before and during the cycle, the renewed USD it touched
  sff_price_increase_monitor     one row per PRICE GROUP (price_group_dims: a tariff changes by region and
                                 product) and closed pipeline month: the uplift of everyone, the uplift
                                 of the ISOLATED renewals (the detector: the normal renewal process, no
                                 retention discounts in it), the step against the average of the previous 12
                                 months, and the increase cycle (a price increase affects the renewals
                                 of ONE cycle: who renews pays the new price, but a year later everyone
                                 already bought at it, so the step lasts configuration.price_cycle_months and then fades)
  sff_parametros                 one row per field of the Config: its value in this run, its default,
                                 whether it was changed, the steps whose code reads it and the steps
                                 its comment documents, and the comment (what it decides)
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

from config import (CORE_ISOLATED_BAND_PREFIX, join_columns, CORE_ISOLATED_PIPELINE_UNITS, CORE_ISOLATED_RENEWED_PIPELINE_USD,
                    CORE_ISOLATED_RENEWED_UNITS, CORE_ISOLATED_RENEWED_USD, CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR,
                    CORE_RENEWED_PIPELINE_USD, ISOLATED_RATIO_BANDS, Config, parameter_table, weighted_ratio_spread)
from step_nucleo import FINAL_ORIGIN_COLUMN, TS_WITHOUT_RESULT
from vocabulario import (ROW_FROM_GAP, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED,
                         TOTAL_ORIGIN_SIMULATED, TS_PROJECTED, TS_REAL, TS_REENTRY)


STEP_LABEL = "22"
STEP_NAME = "ADAPTATION TO POWER BI"
STEP_PURPOSE = ("build the auxiliary tables of the Power BI report (names, orders and blocks of the codes of the core), "
                "so that Power BI only relates and sums: no logic inside the report")
STEP_ACTIONS = ["the table of pipeline sources: code, label, block, order",
                "the price increase monitor: the uplift of the closed months and its steps",
                "the parameters of this run: every field of the Config, its value and the steps that read it",
                "check the tables against the core (checks 1-4)",
                "write them (checks 5-8)",
                "count the checks; stop if any failed",
                "show the tables"]
STEP_OUTPUT = ("tables sff_forecast_pipeline_source (one row per forecast_pipeline_source) and "
               "sff_price_increase_monitor (one row per closed pipeline month)")

PIPELINE_SOURCE_TABLE = "forecast_pipeline_source"
PRICE_MONITOR_TABLE = "price_increase_monitor"
PARAMETERS_TABLE = "parametros"
PRICE_CALENDAR_TABLE = "price_increase_calendar"

# the detector of price increases, over the uplift of the isolated renewals (no retention event):
                               # (one noisy month is not an increase; the price of it: a step is confirmed 2 months late)

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

    # [2] the price increase monitor, per price group, and the calendar of the increases it detects
    price_monitor = price_increase_monitor(core_table, configuration)
    calendar = increase_calendar(price_monitor, configuration)
    groups = price_monitor["price_group"].nunique()
    configuration.log_action(STEP_LABEL, 2, f"{groups:,} price groups ({', '.join(configuration.price_group_dims) or 'the portfolio'}) "
                                            f"× {price_monitor['period'].nunique()} closed months · detector on "
                                            f"{price_monitor.attrs['detector_series']} · {len(calendar)} increases detected in "
                                            f"{calendar['price_group'].nunique() if len(calendar) else 0} groups")

    # [3] the parameters of this run
    parameters = parameter_table(configuration)
    changed = parameters[parameters["cambiado"] == 1]
    configuration.log_action(STEP_LABEL, 3, f"{len(parameters)} parameters in the Config · {len(changed)} with a value of "
                                            f"this run (not its default) · "
                                            f"{int((parameters['pasos'] == '').sum())} read by no step")

    # [4] the checks against the core
    configuration.log_action(STEP_LABEL, 4, "checking the tables against the core")
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
                            f"an increase is only flagged against a reference of ≥ {configuration.price_min_reference_months} previous months",
                            flags_without_reference.empty,
                            failure_detail=f"{len(flags_without_reference)} months flagged without a reference")

    # [5] the writes
    configuration.log_action(STEP_LABEL, 5, "writing the tables")
    configuration.write_table(STEP_LABEL, check_log, pipeline_source_table, PIPELINE_SOURCE_TABLE)
    configuration.write_table(STEP_LABEL, check_log, price_monitor.drop(columns=["renewed_units"]), PRICE_MONITOR_TABLE)
    configuration.write_table(STEP_LABEL, check_log, calendar, PRICE_CALENDAR_TABLE)
    configuration.write_table(STEP_LABEL, check_log, parameters, PARAMETERS_TABLE)

    # [6] the count
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the tables on screen
    configuration.log_action(STEP_LABEL, 7, "the pipeline sources, as Power BI will show them, the last 13 months of the "
                                            "price increase monitor, and the parameters of this run by step:")
    configuration.show_table(pipeline_source_table)
    configuration.logger.doc(f"[{STEP_LABEL}] ══════════ CALENDARIO DE SUBIDAS DETECTADAS ══════════")
    configuration.logger.doc(f"[{STEP_LABEL}] por grupo de precio ({', '.join(configuration.price_group_dims) or 'cartera'}): un "
                             f"escalón ≥ {configuration.price_step_threshold:.0%} frente a sus 12 meses previos, mantenido "
                             f"{configuration.price_persistence_months} meses, en meses con ≥ "
                             f"{configuration.price_group_min_renewed_units:,.0f} renovaciones; las 30 de más valor:")
    if len(calendar):
        shown_calendar = calendar.head(30).copy()
        for column_name in ["escalon", "uplift_antes", "uplift_ciclo"]:
            shown_calendar[column_name] = shown_calendar[column_name].round(3)
        shown_calendar["usd_renovado_ciclo"] = shown_calendar["usd_renovado_ciclo"].round(0)
        configuration.show_table(shown_calendar.drop(columns=["price_group"]))
    else:
        configuration.logger.doc(f"[{STEP_LABEL}] no increase detected in any group")
    configuration.logger.doc(f"[{STEP_LABEL}] ══════════ FIN DEL CALENDARIO DE SUBIDAS ══════════")
    configuration.show_table(parameters[["parametro", "valor", "por_defecto", "cambiado", "pasos"]])

    return {"pipeline_source": pipeline_source_table, "price_monitor": price_monitor.drop(columns=["renewed_units"]),
            "price_calendar": calendar, "parameters": parameters}



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
             configuration.price_min_reference_months), uplift_step (the month over that average, − 1), above_threshold (step
             ≥ configuration.price_step_threshold), price_increase_flag (1 in the FIRST month of a run of ≥
             configuration.price_persistence_months months above the threshold: an increase is a step that stays, one noisy
             month is not), in_increase_cycle (a flagged month in this or the previous configuration.price_cycle_months − 1
             months) and months_since_increase. A run still too short at the end of the history is not
             flagged yet: it is confirmed when its months close.
    RULES:   the detector runs on uplift_isolated when it can be computed (no retention discounts in it),
             on the plain uplift otherwise; attrs["detector_series"] says which.
             A price increase raises the uplift of the renewals of ONE cycle: who renews pays the new
             tariff against the old one. A cycle later everyone due already bought at the new tariff and
             the step fades, so in_increase_cycle marks exactly configuration.price_cycle_months months per detected step.
    """
    closed = core_table[(core_table["forecast_status"] == "actual") & (core_table["forecast_universe"] == "pipeline")
                        & (core_table["forecast_to_renew_units"] > 0)].copy()
    closed["_precio_fila"] = closed["forecast_to_renew_USD"] / closed["forecast_to_renew_units"]
    closed["_base_aproximada"] = closed["forecast_renewed_units"] * closed["_precio_fila"]
    present = set(closed.columns)
    if CORE_ISOLATED_RENEWED_UNITS in present:
        closed["_base_aproximada_isolated"] = closed[CORE_ISOLATED_RENEWED_UNITS] * closed["_precio_fila"]

    # the price groups: a tariff changes by region and product, so every group is watched on its own
    group_dims = [column_name for column_name in configuration.price_group_dims if column_name in closed.columns]
    closed["price_group"] = join_columns(closed, group_dims) if group_dims else "cartera"
    dims_of_group = (closed.drop_duplicates("price_group").set_index("price_group")[group_dims]
                     if group_dims else pd.DataFrame(index=pd.Index(["cartera"], name="price_group")))
    summed = closed.groupby(["price_group", "period"]).sum(numeric_only=True)
    monitor = pd.DataFrame(index=summed.index)
    monitor["renewed_units"] = summed["forecast_renewed_units"]
    uplift_base = summed[CORE_RENEWED_PIPELINE_USD] if CORE_RENEWED_PIPELINE_USD in present else summed["_base_aproximada"]
    monitor["renewed_USD"] = summed["forecast_renewed_USD"]          # the two sums, so Power BI can add groups up
    monitor["uplift_base_USD"] = uplift_base
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

    # the detector: per price group, the chosen series against the average of its previous 12 months; a month
    # with too few renewals in its group is noise, not a price: it is left out of the detection
    detector_series = "uplift_isolated" if monitor["uplift_isolated"].notna().any() else "uplift"
    # the volume of the detector: the isolated renewals when it reads them and they are there, every renewal otherwise
    volume = (summed[CORE_ISOLATED_RENEWED_UNITS] if detector_series == "uplift_isolated" and CORE_ISOLATED_RENEWED_UNITS in present
              else summed["forecast_renewed_units"])
    monitor["volumen_detector"] = volume.to_numpy()
    monitor["detector"] = monitor[detector_series].where(monitor["volumen_detector"] >= configuration.price_group_min_renewed_units)
    monitor = monitor.reset_index().sort_values(["price_group", "period"]).reset_index(drop=True)
    detected = [detect_increases(group_rows, configuration) for _, group_rows in monitor.groupby("price_group", sort=False)]
    monitor = pd.concat(detected).sort_index()
    monitor = monitor.merge(dims_of_group.reset_index(), on="price_group", how="left")
    monitor.attrs["detector_series"] = detector_series
    return monitor


def detect_increases(group_rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The detection of one price group, month by month (its rows sorted by period): the step against the
    average of its previous 12 months, the months above the threshold, the increases (a run of
    price_persistence_months above it starts one), the cycle and the months since the increase."""
    rows = group_rows.copy()
    detector = rows["detector"]
    rows["uplift_previous_12"] = detector.shift(1).rolling(window=12, min_periods=configuration.price_min_reference_months).mean()
    rows["uplift_step"] = detector / rows["uplift_previous_12"] - 1
    above = ((rows["uplift_step"] >= configuration.price_step_threshold) & rows["uplift_previous_12"].notna()).to_numpy()
    rows["above_threshold"] = above.astype(int)
    flags = np.zeros(len(rows), dtype=int)
    position = 0
    while position < len(above):
        if not above[position]:
            position += 1
            continue
        run_end = position
        while run_end < len(above) and above[run_end]:
            run_end += 1
        if run_end - position >= configuration.price_persistence_months:
            flags[position] = 1                      # the increase starts where the run starts
        position = run_end
    rows["price_increase_flag"] = flags
    rows["in_increase_cycle"] = (rows["price_increase_flag"].rolling(window=configuration.price_cycle_months, min_periods=1)
                                 .max().astype(int).to_numpy())
    months_since, last_flagged_position = [], None
    for position, flagged in enumerate(rows["price_increase_flag"]):
        if flagged:
            last_flagged_position = position
        months_since.append(position - last_flagged_position if last_flagged_position is not None else np.nan)
    rows["months_since_increase"] = months_since
    return rows


def increase_calendar(monitor: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The increases detected, one row per price group and start month: its size (the step), the uplift
    before (the previous 12 months) and during the cycle, the months of the cycle already seen and the
    renewed USD in them (what the increase touched), biggest first."""
    rows = []
    for _, start in monitor[monitor["price_increase_flag"] == 1].iterrows():
        same_group = monitor[monitor["price_group"] == start["price_group"]]
        in_cycle = same_group[(same_group["period"] >= start["period"])
                              & (same_group["period"] < start["period"] + configuration.price_cycle_months)]
        row = {"price_group": start["price_group"], "inicio": start["period"],
               "escalon": start["uplift_step"], "uplift_antes": start["uplift_previous_12"],
               "uplift_ciclo": in_cycle["detector"].mean(), "meses_vistos": int(in_cycle["detector"].notna().sum()),
               "usd_renovado_ciclo": in_cycle["renewed_USD"].sum()}
        for column_name in configuration.price_group_dims:
            if column_name in start.index:
                row[column_name] = start[column_name]
        rows.append(row)
    columns = (["price_group", "inicio"] + [c for c in configuration.price_group_dims if c in monitor.columns]
               + ["escalon", "uplift_antes", "uplift_ciclo", "meses_vistos", "usd_renovado_ciclo"])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("usd_renovado_ciclo", ascending=False).reset_index(drop=True)

def build_pipeline_source_table() -> pd.DataFrame:
    """One row per pipeline source: code, business label, block and order."""
    pipeline_source_table = pd.DataFrame(PIPELINE_SOURCES,
                                         columns=[FINAL_ORIGIN_COLUMN, "source_label", "source_block", "source_order"])
    return pipeline_source_table
