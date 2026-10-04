"""
step_06_series_routes.py — The series: which months each one has, and how it will be treated.

A rate series is not forecast the same way: it depends on the months it has:
  · COVERAGE: the roles of its months, in time order (e.g. entrenamiento+examen+proyeccion)
  · ROUTE, from the coverage:
      predecible     closed months AND something to predict: its rate is estimated
      solo_historia  nothing to predict: kept, it lends its history to its relatives
      solo_futuro    nothing to learn from: predicted from its relatives
  · UNIVERSE, from the time_series flag of its units: normal · serie_temporal · mixto
Nothing is filtered: every series is labelled, none is dropped.

Actions (logged as they are done):
  1. group the forecast units by series
  2. count the months of each series in each role; first and last month
  3. coverage, route and universe of every series
  4. the money each series has to predict (due from the current month on)
  5. check the series                                               checks 1-3
  6. write the series table                                         check 4
  7. count the checks; stop if any failed
  8. show the series and the money per route, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. every forecast unit belongs to a series of the table, and every series has a route
   2. the time_series flag is the same in every unit of a series     (warning only)
   3. every series with something to predict has pipeline to predict (warning only)
   4. table sff_series written and read back

Output: one row per series (ids, series columns, months per role, first and last month,
coverage, route, universe, units and USD to predict) · table sff_series.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import ACTIVE_FLAG_VALUES, Config
from vocabulario import (CALENDAR_ROLE_COLUMN, COVERAGE_COLUMN, ROLES_IN_ORDER, ROLE_PROJECTION, ROLE_TEST,
                         ROLE_TRAIN, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, ROUTE_PREDICTABLE,
                         SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_SERIES, UNIT_ID_COLUMN, UNIVERSE_COLUMN,
                         UNIVERSE_MIXED, UNIVERSE_NORMAL, UNIVERSE_TIME_SERIES)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "06"
STEP_NAME = "SERIES AND ROUTES"
STEP_PURPOSE = ("describe every rate series by the months it has (coverage) and decide how it will be treated "
                "(route): estimated, used only as history, or predicted from its relatives; nothing is dropped")
STEP_ACTIONS = ["group the forecast units by series",
                "count the months of each series in each role; first and last month",
                "coverage, route and universe of every series",
                "the money each series has to predict (due from the current month on)",
                "check the series (checks 1-3)",
                "write the series table (check 4)",
                "count the checks; stop if any failed",
                "show the series and the money per route, as a table"]
STEP_OUTPUT = "one row per series with its coverage, route and universe · table sff_series"

# ─── named constants ─────────────────────────────────────────────────────────────
COVERAGE_SEPARATOR = "+"
ROUTES_IN_ORDER = [ROUTE_PREDICTABLE, ROUTE_HISTORY_ONLY, ROUTE_FUTURE_ONLY]
MONTH_COUNT_COLUMN = {ROLE_TRAIN: "meses_entrenamiento", ROLE_TEST: "meses_examen", ROLE_PROJECTION: "meses_proyeccion"}


def route_from_coverage(roles_present: set) -> str:
    """The route of a series from the roles it has: nothing to predict → solo_historia;
    something to predict and a closed month (training or exam) → predecible; something
    to predict and no closed month → solo_futuro."""
    has_something_to_predict = ROLE_PROJECTION in roles_present
    has_closed_months = ROLE_TRAIN in roles_present or ROLE_TEST in roles_present
    if not has_something_to_predict:
        return ROUTE_HISTORY_ONLY
    if has_closed_months:
        return ROUTE_PREDICTABLE
    return ROUTE_FUTURE_ONLY


def build_series_routes(forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per series: its months per role, coverage, route and universe; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    flag_column = configuration.flag_time_series_col

    # [1] the units of each series
    grouped_by_series = forecast_units.groupby(SERIES_ID_COLUMN, sort=True)
    series_table = grouped_by_series[[SERIES_KEY_COLUMN] + configuration.rate_series_columns].first()
    configuration.log_action(STEP_LABEL, 1, f"{len(forecast_units):,} forecast units grouped into "
                                            f"{len(series_table):,} series")

    # [2] months per role, first and last month
    months_per_role = (forecast_units.groupby([SERIES_ID_COLUMN, CALENDAR_ROLE_COLUMN]).size()
                       .unstack(fill_value=0).reindex(columns=ROLES_IN_ORDER, fill_value=0))
    for role in ROLES_IN_ORDER:
        series_table[MONTH_COUNT_COLUMN[role]] = months_per_role[role]
    series_table["primer_mes"] = grouped_by_series[period_column].min()
    series_table["ultimo_mes"] = grouped_by_series[period_column].max()
    series_table["meses"] = grouped_by_series.size()
    configuration.log_action(STEP_LABEL, 2, f"months counted per role; a series has "
                                            f"{series_table['meses'].median():.0f} months (median)")

    # [3] coverage, route and universe
    roles_of_series = months_per_role.apply(lambda month_counts: {role for role in ROLES_IN_ORDER if month_counts[role] > 0},
                                            axis=1)
    series_table[COVERAGE_COLUMN] = roles_of_series.map(
        lambda roles: COVERAGE_SEPARATOR.join(role for role in ROLES_IN_ORDER if role in roles))
    series_table[ROUTE_COLUMN] = roles_of_series.map(route_from_coverage)
    flagged_share = grouped_by_series[flag_column].apply(lambda flags: flags.isin(ACTIVE_FLAG_VALUES).mean())
    series_table[UNIVERSE_COLUMN] = flagged_share.map(
        lambda share: UNIVERSE_TIME_SERIES if share == 1 else (UNIVERSE_NORMAL if share == 0 else UNIVERSE_MIXED))
    route_census = series_table[ROUTE_COLUMN].value_counts().to_dict()
    configuration.log_action(STEP_LABEL, 3, f"routes: {route_census} · universes: "
                                            f"{series_table[UNIVERSE_COLUMN].value_counts().to_dict()}")

    # [4] what each series has to predict: what falls due from the current month on
    to_predict = forecast_units[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    predicted_units = forecast_units[to_predict].groupby(SERIES_ID_COLUMN)
    series_table["unidades_por_predecir"] = predicted_units[configuration.pipeline_units_col].sum()
    series_table["usd_por_predecir"] = predicted_units[configuration.pipeline_usd_col].sum()
    series_table[["unidades_por_predecir", "usd_por_predecir"]] = series_table[["unidades_por_predecir", "usd_por_predecir"]].fillna(0)
    series_table = series_table.reset_index()
    configuration.log_action(STEP_LABEL, 4, f"${series_table['usd_por_predecir'].sum():,.0f} to predict across "
                                            f"{int((series_table['usd_por_predecir'] > 0).sum()):,} series")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the series")
    check_series(forecast_units, series_table, configuration, check_log)

    # [6] the series table, written
    configuration.log_action(STEP_LABEL, 6, "writing the series table")
    configuration.write_table(STEP_LABEL, check_log, series_table, TABLE_SERIES)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] series and money per route
    log_routes_report(series_table, configuration)
    return series_table


def check_series(forecast_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config,
                 check_log: list) -> None:
    """Checks 1 to 3."""
    # [1] every unit has its series in the table, and every series a known route
    units_without_series = int((~forecast_units[SERIES_ID_COLUMN].isin(set(series_table[SERIES_ID_COLUMN]))).sum())
    unknown_routes = sorted(set(series_table[ROUTE_COLUMN]) - set(ROUTES_IN_ORDER))
    configuration.log_check(STEP_LABEL, check_log,
                            "every forecast unit belongs to a series of the table, and every series has a route",
                            units_without_series == 0 and not unknown_routes,
                            failure_detail=f"{units_without_series:,} units without series · unknown routes {unknown_routes}",
                            context=f"{len(series_table):,} series · {len(forecast_units):,} units")

    # [2] a series belongs to one universe
    mixed_series = series_table[series_table[UNIVERSE_COLUMN] == UNIVERSE_MIXED]
    configuration.log_check(STEP_LABEL, check_log,
                            f"{configuration.flag_time_series_col} is the same in every unit of a series",
                            mixed_series.empty,
                            failure_detail=f"{len(mixed_series):,} series have units in both universes (labelled "
                                           f"'{UNIVERSE_MIXED}')",
                            blocking=False,
                            examples=mixed_series[[SERIES_ID_COLUMN, "primer_mes", "ultimo_mes", COVERAGE_COLUMN]])

    # [3] a series with months to predict has something due in them
    predictable_routes = series_table[ROUTE_COLUMN].isin([ROUTE_PREDICTABLE, ROUTE_FUTURE_ONLY])
    empty_future = series_table[predictable_routes & (series_table["unidades_por_predecir"] == 0)]
    configuration.log_check(STEP_LABEL, check_log, "every series with something to predict has pipeline to predict",
                            empty_future.empty,
                            failure_detail=f"{len(empty_future):,} series have future months with 0 units due",
                            blocking=False,
                            examples=empty_future[[SERIES_ID_COLUMN, COVERAGE_COLUMN, "meses_proyeccion",
                                                   "unidades_por_predecir"]])


def log_routes_report(series_table: pd.DataFrame, configuration: Config) -> None:
    """Action 8: series and money to predict per route, as a table."""
    configuration.log_action(STEP_LABEL, 8, "series and money to predict per route:")
    total_usd = series_table["usd_por_predecir"].sum()
    route_rows = []
    for route in ROUTES_IN_ORDER:
        route_series = series_table[series_table[ROUTE_COLUMN] == route]
        route_usd = route_series["usd_por_predecir"].sum()
        route_rows.append({ROUTE_COLUMN: route, "series": len(route_series), "usd_por_predecir": route_usd,
                           "pct_usd": route_usd / total_usd if total_usd else 0.0})
    configuration.show_table(pd.DataFrame(route_rows))
