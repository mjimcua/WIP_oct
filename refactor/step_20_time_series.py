"""
step_20_time_series.py — The time_series universe (retail to subscription) and the total of
the forecast.

REGION: the rows carry the region levels of ts_region_columns (the real extract: region, subregion,
country). The projection is made at the FINEST level (the joined levels are the key of a
region); the output keeps every level, so any of them can be aggregated. The renewal rate of
a re-entry climbs the levels: finest → … → coarsest → global.

THE UNIVERSE: the rows flagged time_series are the revenue of the retail-to-subscription
campaigns: licences bought in retail that move to an online subscription with an offer.
There is no signal proportional to the people who move and the campaigns are not modelled:
it is the best effort with the series itself. These rows come with the REGION and the
RESULT only (renewed units and renewed USD, the equivalent of what converted); the other
dimensions and the pipeline are empty. That is why they are split from the raw right after
step 00 (split_time_series_rows): the normal pipeline never sees them, and this step
receives them at the end.

It adds rows and the total; it changes nothing that came before.

2026 (the current year), per region and month:
  · ts_real        the closed months of the year, as they are
  · ts_proyectado  the simulation window (the current month, included, to simulation_end):
        units = units of the same month last year × level
        level = Σ units of the last ts_level_window_months (3) closed months of this year /
                Σ units of the same months last year
        value = units × value per unit of the region
        value per unit = Σ value / Σ units of the last ts_auv_window_months (12) closed months
    A region with no same month last year, or a level denominator of 0: the TOTAL of all
    regions is projected with the same rule and shared by the region's share of units of
    the last 12 months (warning).
  These rows are REVENUE of 2026, not renewals: no rate is applied to them.
2027: only what was PROJECTED falls due again (the real converts are already in the real
pipeline: counting them again would duplicate them):
  · ts_reentrada  every projected month falls due the same month of next year, in its
    region (a 1-year licence). Acquired at ts_acquisition_discount (0.4) off and renewed at
    100 %: value to renew = projected value / (1 − 0.4), the pure rule. Renewal rate = the
    rate of the region in the normal pipeline (finest level that has one): Σ renewed / Σ due units of the last
    ts_rate_window_months (12) closed months (the global rate if the region has none,
    warning). Renewed units = units × rate; revenue 2027 = value to renew × rate.

THE TOTAL, by year and origin (units and USD due and renewed), and the total of every year:
  pipeline_renovado_real   renewals booked in the closed months (normal universe)
  pipeline_real_esperado   expected renewals of the extract's future pipeline
  pipeline_proyectada      expected renewals of the extended re-entries
  pipeline_simulada        expected renewals of the simulated acquisition
  ts_real, ts_proyectado   revenue of the time_series universe (2026)
  ts_reentrada             expected revenue of its re-entry (2027)
  Total 2026 = pipeline + ts_real + ts_proyectado · Total 2027 = extended forecast + ts_reentrada

Actions (logged as they are done):
  1. the time_series rows by region and month, and the calendar of the simulation
  2. the level and the value per unit of every region
  3. the projected months of the year (fallback by share where a region cannot be projected)
  4. the re-entry of the projected months as pipeline of next year, with the region's rate
  5. the total by year and origin
  6. check the simulation and the total                               checks 1-4
  7. write the time_series rows and the total                          checks 5-6
  8. count the checks; stop if any failed
  9. show the total by year × origin and the 10 regions with the most projected value

Checks (logged as they are made, numbered, at the level of their status):
   1. the origins of every year add up to its total
   2. no ts_reentrada row comes from a ts_real month
   3. the implied value per unit of the projected months is inside the region's history (warning)
   4. every region with a re-entry has a rate of some region level (warning if it takes the global one)
   5-6. tables sff_time_series and sff_forecast_total written and read back

Output: the time_series rows (region × month) and the total (year × origin) · tables
sff_time_series, sff_forecast_total.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ACTIVE_FLAG_VALUES, Config, join_columns, parse_month
from vocabulario import (CALENDAR_ROLE_COLUMN, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL,
                         PIPELINE_SIMULATED, TABLE_FORECAST_TOTAL, TABLE_TIME_SERIES, TOTAL_ORIGIN_EXPECTED,
                         TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL,
                         TRUTH_ROLES, TS_PROJECTED, TS_REAL, TS_REENTRY)


STEP_LABEL = "20"
STEP_NAME = "TIME SERIES UNIVERSE AND TOTAL"
STEP_PURPOSE = ("simulate the rest of the year of the retail-to-subscription revenue (the time_series universe) with "
                "its own series, let what is projected fall due again next year as pipeline (40 % off, renewed at 100 %, "
                "at the region's rate), and give the total of every year separable by origin")
STEP_ACTIONS = ["the time_series rows by region and month, and the calendar of the simulation",
                "the level and the value per unit of every region",
                "the projected months of the year (fallback by share where a region cannot be projected)",
                "the re-entry of the projected months as pipeline of next year, with the region's rate",
                "the total by year and origin",
                "check the simulation and the total (checks 1-4)",
                "write the time_series rows and the total (checks 5-6)",
                "count the checks; stop if any failed",
                "show the total by year × origin and the 10 regions with the most projected value"]
STEP_OUTPUT = "region × month rows (ts_real, ts_proyectado, ts_reentrada) · year × origin total · tables sff_time_series, sff_forecast_total"

SPLIT_LABEL = "00b"
MONEY_TOLERANCE = 0.01
MONTHS_PER_YEAR = 12
TOP_REGIONS_SHOWN = 10


REGION_KEY = "region_ts"         # the key of a region: its levels joined, coarse to fine


def region_columns_of(configuration: Config) -> list:
    """The region levels of the universe, coarse to fine (the first mandatory dim if none is declared)."""
    return list(configuration.ts_region_columns) or [configuration.business_mandatory_dims[0]]


def with_region_key(frame: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The frame with the key of its finest region (the levels joined)."""
    return frame.assign(**{REGION_KEY: join_columns(frame, region_columns_of(configuration))})


# ═══════════════════════════════════════════════════════════════════════════════════
# THE SPLIT (right after step 00)
# ═══════════════════════════════════════════════════════════════════════════════════

def split_time_series_rows(validated_raw: pd.DataFrame, configuration: Config) -> tuple:
    """Split the time_series rows (flag on) from the raw right after step 00.

    INPUT:   the raw validated by step 00 · the Config (flag_time_series_col, region levels).
    OUTPUT:  (the raw of the normal universe, the time_series rows).
    RULES:   a row is time_series when its flag is active. Its region and its renewed units
             and USD must be there; the rest of its dimensions and its pipeline may be empty.
    EDGE CASES: no time_series row → an empty frame, and step 20 only builds the total.
    """
    configuration.logger.doc(f"[{SPLIT_LABEL}] ═══ SPLIT · the time_series universe (retail to subscription) ═══")
    configuration.logger.doc(f"[{SPLIT_LABEL}] purpose: take the time_series rows out of the raw before step 01: they carry "
                             f"only the region and the result, and step 20 simulates them at the end")
    check_log = []
    flagged = validated_raw[configuration.flag_time_series_col].isin(ACTIVE_FLAG_VALUES)
    time_series_rows = validated_raw[flagged].copy()
    normal_raw = validated_raw[~flagged].copy()
    region_columns = region_columns_of(configuration)
    configuration.log_action(SPLIT_LABEL, 1, f"{int(flagged.sum()):,} time_series rows out of {len(validated_raw):,}; "
                                             f"{len(normal_raw):,} rows go on through steps 01-19 · region levels {region_columns}")
    missing_region = int(time_series_rows[region_columns].isna().any(axis=1).sum()) if len(time_series_rows) else 0
    configuration.log_check(SPLIT_LABEL, check_log, "every time_series row has every region level", missing_region == 0,
                            failure_detail=f"{missing_region:,} time_series rows without some region level",
                            context=f"{time_series_rows[region_columns].drop_duplicates().shape[0] if len(time_series_rows) else 0} "
                                    f"finest regions")
    missing_result = int(time_series_rows[configuration.renewed_units_col].isna().sum()
                         + time_series_rows[configuration.renewed_usd_col].isna().sum()) if len(time_series_rows) else 0
    closed = time_series_rows[configuration.period_col] < configuration.calendar_boundaries()["current"] if len(time_series_rows) else []
    missing_closed = int(time_series_rows.loc[closed, [configuration.renewed_units_col, configuration.renewed_usd_col]]
                         .isna().any(axis=1).sum()) if len(time_series_rows) else 0
    configuration.log_check(SPLIT_LABEL, check_log, "every closed time_series row has its units and value", missing_closed == 0,
                            failure_detail=f"{missing_closed:,} closed rows without units or value (read as 0)",
                            blocking=False)
    configuration.log_check_summary(SPLIT_LABEL, "SPLIT", check_log)
    return normal_raw, time_series_rows


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def build_time_series_and_total(time_series_rows: pd.DataFrame, fine_table: pd.DataFrame, forecast_rows: pd.DataFrame,
                                configuration: Config) -> tuple:
    """Simulate the time_series universe and build the total by year and origin.

    INPUT:   the time_series rows (split after step 00) · the fine table (normal universe, for the
             rates of the regions and the booked renewals) · the forecast rows of step 17
             (extract and extended horizon) · the Config.
    OUTPUT:  (time_series table: region × month; total table: year × origin).
    RULES:   see the module header: level × same month last year; value per unit of 12 months;
             re-entry of the projected months only, 1 / (1 − discount), region rate of 12 months.
    EDGE CASES: no time_series row → only the total of the normal universe. A region that cannot
             be projected takes its share of the projected total (warning). A region with no rate
             in the normal pipeline takes the global rate (warning).
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period, region = configuration.period_col, REGION_KEY
    region_columns = region_columns_of(configuration)
    current = configuration.calendar_boundaries()["current"]

    # [1] the time_series rows by region and month
    keyed_rows = with_region_key(time_series_rows, configuration)
    monthly = (keyed_rows.groupby([region, period])[[configuration.renewed_units_col, configuration.renewed_usd_col]]
               .sum(min_count=1).fillna(0).reset_index()
               .rename(columns={configuration.renewed_units_col: "unidades", configuration.renewed_usd_col: "valor"}))
    levels_of_region = keyed_rows[[region] + region_columns].drop_duplicates(region).set_index(region)
    closed_monthly = monthly[monthly[period] < current]
    projected_months = configuration.simulation_window
    configuration.log_action(STEP_LABEL, 1, f"{len(monthly):,} region × month rows, {monthly[region].nunique()} regions at the "
                                            f"finest level ({region_columns[-1]}) · "
                                            f"real {current.year}-01..{current - 1} · projected "
                                            f"{projected_months[0] if projected_months else '—'}..{projected_months[-1] if projected_months else '—'}")

    # [2] the level and the value per unit of every region
    parameters = region_parameters(closed_monthly, current, configuration)
    configuration.log_action(STEP_LABEL, 2, f"level (last {configuration.ts_level_window_months} closed months vs a year before) "
                                            f"and value per unit (last {configuration.ts_auv_window_months} months) of "
                                            f"{len(parameters)} regions")

    # [3] the projected months
    projected, fallback_regions = project_months(closed_monthly, parameters, projected_months, current, configuration)
    real_rows = closed_monthly[closed_monthly[period].map(lambda month: month.year) == current.year].assign(origen=TS_REAL)
    configuration.log_action(STEP_LABEL, 3, f"{len(projected):,} projected region × month rows: {projected['unidades'].sum():,.0f} "
                                            f"units · ${projected['valor'].sum():,.0f}; {len(fallback_regions)} regions by share "
                                            f"of the total" if len(projected) else "nothing to project")

    # [4] the re-entry, next year, at the region's rate
    rates, global_rate = region_rates(fine_table, current, configuration)
    reentry = reentry_rows(projected.join(levels_of_region, on=region) if len(projected) else projected,
                           rates, global_rate, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{len(reentry):,} re-entry rows: ${reentry['valor'].sum():,.0f} of pipeline at "
                                            f"{configuration.ts_acquisition_discount:.0%} off, renewed at 100 % "
                                            f"(× 1 / (1 − {configuration.ts_acquisition_discount})) · expected "
                                            f"${reentry['revenue'].sum():,.0f}" if len(reentry) else "no re-entry")

    time_series_table = pd.concat([real_rows.assign(revenue=real_rows["valor"]),
                                   projected.assign(revenue=projected["valor"]), reentry], ignore_index=True)
    time_series_table = time_series_table.drop(columns=region_columns, errors="ignore").join(levels_of_region, on=region)
    time_series_table = time_series_table[region_columns + [column for column in time_series_table.columns
                                                            if column not in region_columns]]

    # [5] the total by year and origin
    total = forecast_total(fine_table, forecast_rows, time_series_table, current, configuration)
    configuration.log_action(STEP_LABEL, 5, "total by year and origin: " + " · ".join(
        f"{int(row['ano'])} ${row['usd_renovado']:,.0f}" for _, row in total[total["origen"] == TOTAL_ORIGIN_TOTAL].iterrows()))

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the simulation and the total")
    check_time_series(total, time_series_table, projected, parameters, fallback_regions, reentry, configuration, check_log)

    # [7] the tables
    configuration.log_action(STEP_LABEL, 7, "writing the time_series rows and the total")
    configuration.write_table(STEP_LABEL, check_log, time_series_table, TABLE_TIME_SERIES)
    configuration.write_table(STEP_LABEL, check_log, total, TABLE_FORECAST_TOTAL)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] the total and the regions
    configuration.log_action(STEP_LABEL, 9, "the total by year × origin (usd_vence: pipeline; usd_renovado: renewals / revenue):")
    configuration.show_table(total)
    if len(projected):
        configuration.logger.doc(f"[{STEP_LABEL}] the {TOP_REGIONS_SHOWN} regions with the most projected value:")
        configuration.show_table(projected.join(levels_of_region, on=region).groupby(region_columns)
                                 .agg(meses=(period, "size"), unidades=("unidades", "sum"), valor=("valor", "sum"),
                                      nivel=("nivel", "first"), valor_medio=("valor_medio", "first"),
                                      por_cuota=("por_cuota", "max"))
                                 .reset_index().sort_values("valor", ascending=False).head(TOP_REGIONS_SHOWN))
    return time_series_table, total


def region_parameters(closed_monthly: pd.DataFrame, current, configuration: Config) -> pd.DataFrame:
    """Per region: level (last N closed months vs the same months a year before), value per unit
    (last 12 months), share of units (last 12 months) and the range of its monthly value per unit."""
    period, region = configuration.period_col, REGION_KEY
    level_months = pd.period_range(current - configuration.ts_level_window_months, current - 1, freq="M")
    last_year_months = [month - MONTHS_PER_YEAR for month in level_months]
    auv_window = closed_monthly[closed_monthly[period] >= current - configuration.ts_auv_window_months]
    recent = closed_monthly[closed_monthly[period].isin(level_months)].groupby(region)["unidades"].sum()
    before = closed_monthly[closed_monthly[period].isin(last_year_months)].groupby(region)["unidades"].sum()
    grouped = auv_window.groupby(region)
    parameters = pd.DataFrame({"unidades_recientes": recent, "unidades_ano_anterior": before,
                               "valor_medio": grouped["valor"].sum() / grouped["unidades"].sum().replace(0, np.nan),
                               "cuota_unidades": grouped["unidades"].sum() / auv_window["unidades"].sum()})
    monthly_auv = auv_window.assign(_auv=auv_window["valor"] / auv_window["unidades"].replace(0, np.nan)).groupby(region)["_auv"]
    parameters["valor_medio_min"], parameters["valor_medio_max"] = monthly_auv.min(), monthly_auv.max()
    parameters["nivel"] = parameters["unidades_recientes"] / parameters["unidades_ano_anterior"].replace(0, np.nan)
    parameters.attrs["total_nivel"] = float(recent.sum() / before.sum()) if before.sum() > 0 else np.nan
    parameters.attrs["total_valor_medio"] = float(auv_window["valor"].sum() / auv_window["unidades"].sum()) if auv_window["unidades"].sum() > 0 else np.nan
    return parameters


def project_months(closed_monthly: pd.DataFrame, parameters: pd.DataFrame, projected_months: list, current,
                   configuration: Config) -> tuple:
    """Units = units of the same month last year × level; value = units × value per unit. A region
    with no same month last year or no level: its share of the projected TOTAL (all regions)."""
    period, region = configuration.period_col, REGION_KEY
    last_year = closed_monthly.set_index([region, period])["unidades"]
    total_last_year = closed_monthly.groupby(period)["unidades"].sum()
    rows, fallback_regions = [], set()
    for month in projected_months:
        total_units = total_last_year.get(month - MONTHS_PER_YEAR, np.nan) * parameters.attrs["total_nivel"]
        for region_value, region_parameters_row in parameters.iterrows():
            same_month = last_year.get((region_value, month - MONTHS_PER_YEAR), np.nan)
            level, auv = region_parameters_row["nivel"], region_parameters_row["valor_medio"]
            by_share = not (np.isfinite(same_month) and np.isfinite(level))
            if by_share:
                fallback_regions.add(region_value)
                units = total_units * region_parameters_row["cuota_unidades"]
                auv = auv if np.isfinite(auv) else parameters.attrs["total_valor_medio"]
            else:
                units = same_month * level
            if not np.isfinite(units):
                continue
            rows.append({region: region_value, period: month, "origen": TS_PROJECTED, "unidades": float(units),
                         "valor": float(units * auv), "nivel": level if not by_share else parameters.attrs["total_nivel"],
                         "valor_medio": auv, "por_cuota": int(by_share)})
    return pd.DataFrame(rows, columns=[region, period, "origen", "unidades", "valor", "nivel", "valor_medio", "por_cuota"]), fallback_regions


def region_rates(fine_table: pd.DataFrame, current, configuration: Config) -> tuple:
    """The renewal rate of the normal pipeline at every region level (Σ renewed / Σ due units of the
    last 12 closed months): {depth: Series by the levels joined up to that depth}; and the global rate."""
    region_columns = region_columns_of(configuration)
    window = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.period_col] >= current - configuration.ts_rate_window_months)]
    rates = {}
    for depth in range(1, len(region_columns) + 1):
        keys = join_columns(window, region_columns[:depth])
        grouped = window.groupby(keys)
        rates[depth] = (grouped[configuration.renewed_units_col].sum()
                        / grouped[configuration.pipeline_units_col].sum().replace(0, np.nan)).dropna()
    global_rate = float(window[configuration.renewed_units_col].sum() / window[configuration.pipeline_units_col].sum())
    return rates, global_rate


def reentry_rows(projected: pd.DataFrame, rates: pd.Series, global_rate: float, configuration: Config) -> pd.DataFrame:
    """Every projected month falls due the same month next year: value to renew = value / (1 − discount),
    renewed units = units × rate, revenue = value to renew × rate."""
    region, region_columns = REGION_KEY, region_columns_of(configuration)
    if projected.empty:
        return pd.DataFrame(columns=[region, configuration.period_col, "origen"])
    reentry = projected[[region] + region_columns + [configuration.period_col, "unidades", "valor"]].copy()
    reentry["mes_origen"] = reentry[configuration.period_col]
    reentry[configuration.period_col] = reentry[configuration.period_col] + MONTHS_PER_YEAR
    reentry["origen"] = TS_REENTRY
    reentry["descuento"] = configuration.ts_acquisition_discount           # the pipeline: units and value, 40 % off
    reentry["valor_a_renovar"] = reentry["valor"] / (1 - configuration.ts_acquisition_discount)   # renewed at 100 %
    # the rate climbs the levels: the finest level that has one, then coarser, then the global rate
    reentry["tasa"] = np.nan
    reentry["nivel_tasa"] = "global"
    for depth in range(len(region_columns), 0, -1):
        level_rate = join_columns(reentry, region_columns[:depth]).map(rates[depth])
        take = reentry["tasa"].isna() & level_rate.notna()
        reentry.loc[take, "tasa"] = level_rate[take]
        reentry.loc[take, "nivel_tasa"] = region_columns[depth - 1]
    reentry["tasa_global"] = reentry["tasa"].isna().astype(int)
    reentry["tasa"] = reentry["tasa"].fillna(global_rate)
    reentry["unidades_renovadas"] = reentry["unidades"] * reentry["tasa"]
    reentry["revenue"] = reentry["valor_a_renovar"] * reentry["tasa"]
    return reentry


def forecast_total(fine_table: pd.DataFrame, forecast_rows: pd.DataFrame, time_series_table: pd.DataFrame, current,
                   configuration: Config) -> pd.DataFrame:
    """By year × origin: units and USD due (pipeline) and renewed (renewals or revenue); plus TOTAL per year."""
    period = configuration.period_col
    year_of = lambda frame: frame[period].map(lambda month: month.year)
    rows = []
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    for year, block in closed.groupby(year_of(closed)):
        rows.append({"ano": year, "origen": TOTAL_ORIGIN_RENEWED, "unidades_vencen": block[configuration.pipeline_units_col].sum(),
                     "usd_vence": block[configuration.pipeline_usd_col].sum(),
                     "unidades_renovadas": block[configuration.renewed_units_col].sum(),
                     "usd_renovado": block[configuration.renewed_usd_col].sum()})
    origin_of = {PIPELINE_REAL: TOTAL_ORIGIN_EXPECTED, PIPELINE_PROJECTED: TOTAL_ORIGIN_PROJECTED,
                 PIPELINE_SIMULATED: TOTAL_ORIGIN_SIMULATED}
    for (year, origin), block in forecast_rows.groupby([year_of(forecast_rows), PIPELINE_ORIGIN_COLUMN]):
        rows.append({"ano": year, "origen": origin_of.get(origin, origin), "unidades_vencen": block[configuration.pipeline_units_col].sum(),
                     "usd_vence": block[configuration.pipeline_usd_col].sum(), "unidades_renovadas": block["esperado_unidades"].sum(),
                     "usd_renovado": block["esperado_usd"].sum()})
    for (year, origin), block in time_series_table.groupby([year_of(time_series_table), "origen"]):
        is_reentry = origin == TS_REENTRY
        rows.append({"ano": year, "origen": origin,
                     "unidades_vencen": block["unidades"].sum() if is_reentry else 0.0,
                     "usd_vence": block["valor"].sum() if is_reentry else 0.0,
                     "unidades_renovadas": block["unidades_renovadas"].sum() if is_reentry else block["unidades"].sum(),
                     "usd_renovado": block["revenue"].sum()})
    by_origin = pd.DataFrame(rows)
    # the years answered: the current one and the ones after it
    by_origin = by_origin[by_origin["ano"] >= current.year]
    totals = by_origin.groupby("ano")[["unidades_vencen", "usd_vence", "unidades_renovadas", "usd_renovado"]].sum().reset_index()
    totals["origen"] = TOTAL_ORIGIN_TOTAL
    return pd.concat([by_origin, totals], ignore_index=True).sort_values(["ano", "origen"]).reset_index(drop=True)


def check_time_series(total: pd.DataFrame, time_series_table: pd.DataFrame, projected: pd.DataFrame,
                      parameters: pd.DataFrame, fallback_regions: set, reentry: pd.DataFrame, configuration: Config,
                      check_log: list) -> None:
    """Checks 1 to 4."""
    region = REGION_KEY
    # [1] the origins add up to the total of every year
    parts = total[total["origen"] != TOTAL_ORIGIN_TOTAL].groupby("ano")["usd_renovado"].sum()
    totals = total[total["origen"] == TOTAL_ORIGIN_TOTAL].set_index("ano")["usd_renovado"]
    differences = (parts - totals).abs()
    configuration.log_check(STEP_LABEL, check_log, "the origins of every year add up to its total",
                            bool((differences <= MONEY_TOLERANCE).all()),
                            failure_detail=f"differences: {differences.to_dict()}", context=f"{len(totals)} years")
    # [2] only projected months re-enter
    real_months = set(zip(time_series_table.loc[time_series_table["origen"] == TS_REAL, region],
                          time_series_table.loc[time_series_table["origen"] == TS_REAL, configuration.period_col]))
    from_real = [key for key in zip(reentry.get(region, []), reentry.get("mes_origen", [])) if key in real_months]
    configuration.log_check(STEP_LABEL, check_log, "no ts_reentrada row comes from a ts_real month", not from_real,
                            failure_detail=f"{len(from_real)} re-entries from real months (they would be counted twice)",
                            context=f"{len(reentry)} re-entry rows")
    # [3] the implied value per unit inside the region's history
    if len(projected):
        implied = projected.groupby(region)["valor"].sum() / projected.groupby(region)["unidades"].sum()
        low, high = parameters["valor_medio_min"].reindex(implied.index), parameters["valor_medio_max"].reindex(implied.index)
        outside = implied[(implied < low - 1e-9) | (implied > high + 1e-9)]
        configuration.log_check(STEP_LABEL, check_log, "the implied value per unit of the projected months is inside the region's history",
                                outside.empty, failure_detail=f"{len(outside)} regions outside their monthly range: {list(outside.index)[:5]}",
                                blocking=False)
    else:
        configuration.log_not_evaluated(STEP_LABEL, check_log, "the implied value per unit is inside the region's history",
                                        "nothing projected")
    # [4] every region with a re-entry has its own rate
    global_regions = sorted(set(reentry.loc[reentry.get("tasa_global", pd.Series(dtype=int)) == 1, region])) if len(reentry) else []
    rate_levels = reentry["nivel_tasa"].value_counts().to_dict() if len(reentry) else {}
    configuration.log_check(STEP_LABEL, check_log, "every region with a re-entry has a rate of some region level",
                            not global_regions,
                            failure_detail=f"{len(global_regions)} regions take the global rate: {global_regions[:5]}",
                            context=f"re-entry rows by the level of their rate: {rate_levels}", blocking=False)
    if fallback_regions:
        configuration.logger.warning(f"[{STEP_LABEL}] projected by share of the total (no same month last year or no level): "
                                     f"{sorted(fallback_regions)[:10]}")
