"""
step_02_apply_calendar.py — The calendar of the Config, applied to every row of the raw.

Steps 00 and 01 only looked at the raw. This is the first step that changes it:
  · every row gets its role in time, generated from the calendar of the Config
    (entrenamiento · examen · proyeccion) and the mark of the
    current month. The raw's own role columns, if any, are ignored.
  · the raw's renewals are kept, untouched, in two s0_ columns: what the raw said
    before any change (the core table reconciles against them)
  · closed months: a null renewal means nobody renewed (a LEFT JOIN with no match),
    so it becomes 0, counted
  · the current month and later: the renewals already booked are early results of a
    month still running, so they are wiped. The future must look like it has not
    started; the pipeline (what falls due) is kept, it is known.
The extra measures (reacquisitions, AUVs) are not touched: reacquisitions are not
forecast (a future extension), and no step uses the AUVs.

Actions (logged as they are done):
  1. keep the raw's renewals and pipeline in the s0_ columns
  2. give every row its role and the current-month mark
  3. closed months: read a null renewal as 0
  4. from the current month on: wipe the renewals already booked; and wipe the pipeline of the
     1-year licences sold or renewed from the current month on (due 12 months later or more):
     that event has not happened, or only in part, so its pipeline is not known yet and step 17
     projects it (the raw keeps it in s0_vencen_*)
  5. check the calendar and the money                               checks 1-9
  6. build the calendar table (one row per month) and write it      check 10
  7. count the checks; stop if any failed
  8. show the calendar per role, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. every row has one of the three roles
   2. training has at least a year of months                      (warning only)
   3. every exam month has rows                                   (warning only)
   4. the current month has pipeline                              (warning only)
   5. the pipeline is unchanged except the rows not known yet (units and USD)
   6. no 1-year row created from the current month on keeps its pipeline
   7. the renewals of the closed months are unchanged, a null read as 0
   8. no row from the current month on keeps a renewal
   9. the s0_ columns hold the raw's renewals and pipeline, untouched
  10. the table sff_calendario is written and read back

Output: a copy of the raw with six new columns (rol, es_mes_en_curso, s0_renovados_*, s0_vencen_*),
and the table sff_calendario (one row per month: its role and its money).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, is_one_year
from report_queries import calendar_query, roles_query
from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                         ROLE_TEST, ROLE_TRAIN, ROLES_IN_ORDER, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                         TABLE_CALENDAR, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "02"
STEP_NAME = "APPLY CALENDAR"
STEP_PURPOSE = ("give every row its role in time from the calendar of the Config, read a null renewal of a "
                "closed month as 0, and wipe the renewals already booked from the current month on: "
                "the future must look like it has not started")
STEP_ACTIONS = ["keep the raw's renewals and pipeline in the s0_ columns",
                "give every row its role and the current-month mark",
                "closed months: read a null renewal as 0",
                "from the current month on: wipe the renewals already booked, and the pipeline of 1-year licences "
                "sold or renewed from the current month on (due 12 months later): it is not known yet, it will be projected",
                "check the calendar and the money (checks 1-9)",
                "build the calendar table (one row per month) and write it (check 10)",
                "count the checks; stop if any failed",
                "show the calendar per role, as a table"]
STEP_OUTPUT = ("the raw with six new columns (rol, es_mes_en_curso, s0_renovados_unidades, s0_renovados_usd, "
               "s0_vencen_unidades, s0_vencen_usd) · "
               "table sff_calendario (one row per month)")

# ─── named constants ─────────────────────────────────────────────────────────────
# Below a year of training months no seasonal pattern can be learned.
MIN_TRAINING_MONTHS = 12
# Money must be conserved to the cent when nothing is supposed to change it.
MONEY_TOLERANCE = 0.01


def apply_calendar(validated: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Give every row its role and current-month mark, read null closed renewals as 0,
    wipe the early results of the future; log every check; stop if any fails."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    renewed_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]
    boundaries = configuration.calendar_boundaries()
    calendared = validated.copy()

    # [1] what the raw said, before any change
    calendared[S0_RENEWED_UNITS_COLUMN] = calendared[configuration.renewed_units_col]
    calendared[S0_RENEWED_USD_COLUMN] = calendared[configuration.renewed_usd_col]
    calendared[S0_PIPELINE_UNITS_COLUMN] = calendared[configuration.pipeline_units_col]
    calendared[S0_PIPELINE_USD_COLUMN] = calendared[configuration.pipeline_usd_col]
    configuration.log_action(STEP_LABEL, 1, f"renewals and pipeline kept as the raw had them in the s0_ columns")

    # [2] the role of every row and the mark of the current month, from the calendar
    calendared[CALENDAR_ROLE_COLUMN] = configuration.role_of_months(calendared[period_column])
    calendared[CURRENT_MONTH_COLUMN] = (calendared[period_column] == boundaries["current"]).astype(int).to_numpy()
    closed_rows = calendared[period_column] < boundaries["current"]
    projection_rows = calendared[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    configuration.log_action(STEP_LABEL, 2, f"roles given: {configuration.calendar_description()}")

    # [3] closed months: a null renewal means nobody renewed
    null_closed_rows = closed_rows & calendared[configuration.renewed_units_col].isna()
    null_renewals_read_as_zero = int(null_closed_rows.sum())
    calendared.loc[null_closed_rows, renewed_columns] = 0.0
    configuration.log_action(STEP_LABEL, 3, f"{null_renewals_read_as_zero:,} null renewals of closed months read as 0")

    # [4] the current month and later: wipe the renewals already booked
    early_results = calendared.loc[projection_rows, renewed_columns].fillna(0)
    rows_with_early_results = int((early_results != 0).any(axis=1).sum())
    wiped_units = float(early_results[configuration.renewed_units_col].sum())
    wiped_usd = float(early_results[configuration.renewed_usd_col].sum())
    calendared.loc[projection_rows, renewed_columns] = np.nan
    configuration.log_action(STEP_LABEL, 4, f"{rows_with_early_results:,} rows from {boundaries['current']} on had "
                                            f"renewals already booked: wiped {wiped_units:,.0f} units · ${wiped_usd:,.0f}")

    # [4b] the pipeline that the current month and later will create: a 1-year licence due in
    #      month m was sold or renewed in m − 12; if m − 12 is the current month or later, that
    #      event has not happened (or only in part): its pipeline is wiped and step 17 projects it
    not_known_yet = (calendared[period_column] >= boundaries["current"] + 12) & is_one_year(calendared, configuration)
    wiped_pipeline_units = float(calendared.loc[not_known_yet, configuration.pipeline_units_col].sum())
    wiped_pipeline_usd = float(calendared.loc[not_known_yet, configuration.pipeline_usd_col].sum())
    calendared.loc[not_known_yet, [configuration.pipeline_units_col, configuration.pipeline_usd_col]] = 0.0
    wiped_per_term = (calendared.loc[not_known_yet, configuration.term_column].astype(str).value_counts().to_dict()
                      if configuration.term_column else "no term_column: every licence is treated as 1-year")
    configuration.log_action(STEP_LABEL, "4b", f"{int(not_known_yet.sum()):,} rows of 1-year licences due from "
                                            f"{boundaries['current'] + 12} on (sold or renewed from {boundaries['current']} on): "
                                            f"pipeline wiped {wiped_pipeline_units:,.0f} units · ${wiped_pipeline_usd:,.0f} "
                                            f"(it will be projected) · rows wiped per term: {wiped_per_term}")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the calendar and the money")
    months_per_role = months_by_role(calendared, configuration)
    check_roles_and_calendar(calendared, configuration, check_log, months_per_role, boundaries)
    check_money_conserved(validated, calendared, configuration, check_log, closed_rows, projection_rows,
                          null_renewals_read_as_zero, rows_with_early_results, wiped_units, wiped_usd, not_known_yet)

    # [6] the calendar table: one row per month, written
    calendar_table = build_calendar_table(calendared, configuration)
    configuration.log_action(STEP_LABEL, 6, f"calendar table built: {len(calendar_table)} months; writing it")
    configuration.write_table(STEP_LABEL, check_log, calendar_table, TABLE_CALENDAR)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] what the calendar looks like on the raw
    log_calendar_report(calendared, configuration, months_per_role)
    return calendared


def build_calendar_table(calendared: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per month of the raw: its role, whether it is the current month, and its
    money (what fell due, what was renewed after step 02, and what the raw said)."""
    period_column = configuration.period_col
    calendar_table = (calendared
                      .groupby(period_column, sort=True)
                      .agg(rol=(CALENDAR_ROLE_COLUMN, "first"),
                           es_mes_en_curso=(CURRENT_MONTH_COLUMN, "max"),
                           filas=(period_column, "size"),
                           unidades_vencen=(configuration.pipeline_units_col, "sum"),
                           usd_vence=(configuration.pipeline_usd_col, "sum"),
                           unidades_renovadas=(configuration.renewed_units_col, lambda values: values.sum(min_count=1)),
                           usd_renovado=(configuration.renewed_usd_col, lambda values: values.sum(min_count=1)),
                           s0_renovados_unidades=(S0_RENEWED_UNITS_COLUMN, lambda values: values.sum(min_count=1)),
                           s0_renovados_usd=(S0_RENEWED_USD_COLUMN, lambda values: values.sum(min_count=1)))
                      .reset_index())
    calendar_table["tasa_unidades"] = calendar_table["unidades_renovadas"] / calendar_table["unidades_vencen"]
    return calendar_table


def months_by_role(calendared: pd.DataFrame, configuration: Config) -> dict:
    """The sorted months of the raw that fall in each role."""
    months_per_role = {}
    for role in ROLES_IN_ORDER:
        role_months = calendared.loc[calendared[CALENDAR_ROLE_COLUMN] == role, configuration.period_col].unique()
        months_per_role[role] = sorted(role_months)
    return months_per_role


def check_roles_and_calendar(calendared: pd.DataFrame, configuration: Config, check_log: list,
                             months_per_role: dict, boundaries: dict) -> None:
    """Checks 1 to 4: roles assigned, enough training, exam with data, pipeline this month."""
    period_column = configuration.period_col

    # [1] every row has one of the three roles
    unknown_roles = sorted(set(calendared[CALENDAR_ROLE_COLUMN]) - set(ROLES_IN_ORDER))
    rows_per_role = calendared[CALENDAR_ROLE_COLUMN].value_counts()
    configuration.log_check(STEP_LABEL, check_log, "every row has one of the three roles", not unknown_roles,
                            failure_detail=f"unknown roles: {unknown_roles}",
                            context=" · ".join(f"{role} {int(rows_per_role.get(role, 0)):,}" for role in ROLES_IN_ORDER))

    # [2] a year of training at least
    training_months = len(months_per_role[ROLE_TRAIN])
    configuration.log_check(STEP_LABEL, check_log, f"training has at least {MIN_TRAINING_MONTHS} months",
                            training_months >= MIN_TRAINING_MONTHS,
                            failure_detail=f"only {training_months} training months: no seasonal pattern can be learned",
                            context=f"{training_months} months", blocking=False)

    # [3] every exam month has rows (a month with no rows is a month the exam cannot score)
    exam_months = pd.period_range(boundaries["test_start"], boundaries["current"] - 1, freq="M")
    exam_months_without_rows = [str(month) for month in exam_months if month not in set(months_per_role[ROLE_TEST])]
    configuration.log_check(STEP_LABEL, check_log, "every exam month has rows", not exam_months_without_rows,
                            failure_detail=f"exam months with no row: {exam_months_without_rows}",
                            context=f"{len(exam_months)} exam months", blocking=False)

    # [4] the current month has pipeline: without it there is nothing to forecast this month
    current_month_rows = calendared[CURRENT_MONTH_COLUMN] == 1
    current_pipeline_units = float(calendared.loc[current_month_rows, configuration.pipeline_units_col].sum())
    configuration.log_check(STEP_LABEL, check_log, f"the current month ({boundaries['current']}) has pipeline",
                            current_pipeline_units > 0,
                            failure_detail=f"no units fall due in {boundaries['current']}",
                            context=f"{current_pipeline_units:,.0f} units due", blocking=False)


def check_money_conserved(validated: pd.DataFrame, calendared: pd.DataFrame, configuration: Config,
                          check_log: list, closed_rows: pd.Series, projection_rows: pd.Series,
                          null_renewals_read_as_zero: int, rows_with_early_results: int,
                          wiped_units: float, wiped_usd: float, not_known_yet: pd.Series) -> None:
    """Checks 5 to 8: the step only changed what it was meant to change."""
    renewed_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]

    # [5] the pipeline is unchanged, except the rows whose event has not happened yet
    pipeline_columns = [configuration.pipeline_units_col, configuration.pipeline_usd_col]
    pipeline_differences = {column_name: float(calendared.loc[~not_known_yet, column_name].sum()
                                               - validated.loc[~not_known_yet, column_name].sum())
                            for column_name in pipeline_columns}
    configuration.log_check(STEP_LABEL, check_log, "the pipeline is unchanged except the rows not known yet (units and USD)",
                            all(abs(difference) <= MONEY_TOLERANCE for difference in pipeline_differences.values()),
                            failure_detail=f"the pipeline changed: {pipeline_differences}")
    pipeline_left = float(calendared.loc[not_known_yet, pipeline_columns].abs().sum().sum())
    configuration.log_check(STEP_LABEL, check_log, "no 1-year row created from the current month on keeps its pipeline",
                            pipeline_left == 0, failure_detail=f"{pipeline_left:,.0f} left in rows not known yet",
                            context=f"{int(not_known_yet.sum()):,} rows wiped: "
                                    f"${validated.loc[not_known_yet, configuration.pipeline_usd_col].sum():,.0f} "
                                    f"(kept in {S0_PIPELINE_USD_COLUMN})")

    # [6] the renewals of the closed months are unchanged (a null counts as 0 before and after)
    closed_differences = {column_name: float(calendared.loc[closed_rows, column_name].sum()
                                             - validated.loc[closed_rows, column_name].fillna(0).sum())
                          for column_name in renewed_columns}
    configuration.log_check(STEP_LABEL, check_log, "the renewals of the closed months are unchanged, a null read as 0",
                            all(abs(difference) <= MONEY_TOLERANCE for difference in closed_differences.values())
                            and not calendared.loc[closed_rows, renewed_columns].isna().any().any(),
                            failure_detail=f"closed renewals changed: {closed_differences}",
                            context=f"{null_renewals_read_as_zero:,} null renewals read as 0")

    # [7] nothing booked survives from the current month on
    renewals_left = int(calendared.loc[projection_rows, renewed_columns].notna().any(axis=1).sum())
    configuration.log_check(STEP_LABEL, check_log, "no row from the current month on keeps a renewal", renewals_left == 0,
                            failure_detail=f"{renewals_left:,} projection rows still carry a renewal",
                            context=f"{rows_with_early_results:,} rows with early results wiped: "
                                    f"{wiped_units:,.0f} units · ${wiped_usd:,.0f}")

    # [8] the s0_ columns are the raw's renewals, untouched
    s0_intact = (calendared[S0_RENEWED_UNITS_COLUMN].equals(validated[configuration.renewed_units_col])
                 and calendared[S0_RENEWED_USD_COLUMN].equals(validated[configuration.renewed_usd_col])
                 and calendared[S0_PIPELINE_UNITS_COLUMN].equals(validated[configuration.pipeline_units_col])
                 and calendared[S0_PIPELINE_USD_COLUMN].equals(validated[configuration.pipeline_usd_col]))
    configuration.log_check(STEP_LABEL, check_log, "the s0_ columns hold the raw's renewals and pipeline, untouched", s0_intact,
                            failure_detail="the s0_ columns differ from the raw")


def log_calendar_report(calendared: pd.DataFrame, configuration: Config, months_per_role: dict) -> None:
    """Action 8: per role, its months, rows and money, as a table."""
    configuration.log_action(STEP_LABEL, 8, "per role: months, rows, what fell due and what was renewed "
                                            "(the projection shows no renewal: it was wiped)")
    summary_rows = []
    for role in ROLES_IN_ORDER:
        role_rows = calendared[calendared[CALENDAR_ROLE_COLUMN] == role]
        role_months = months_per_role[role]
        renewed_units = role_rows[configuration.renewed_units_col].sum(min_count=1)
        pipeline_units = role_rows[configuration.pipeline_units_col].sum()
        summary_rows.append({
            CALENDAR_ROLE_COLUMN: role,
            "meses": len(role_months),
            "desde": str(role_months[0]) if role_months else "",
            "hasta": str(role_months[-1]) if role_months else "",
            "filas": len(role_rows),
            "unidades_vencen": pipeline_units,
            "unidades_renovadas": renewed_units,
            "tasa_unidades": renewed_units / pipeline_units if pipeline_units > 0 else np.nan,
        })
    configuration.show_table(pd.DataFrame(summary_rows))
    configuration.show_query(STEP_LABEL, "the calendar per role", roles_query(configuration))
    configuration.show_query(STEP_LABEL, "the calendar per month (sff_calendario)", calendar_query(configuration))
