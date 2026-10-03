"""
test_step_04.py — Step 04 on the synthetic raw: one row per forecast unit, the money
conserved, the renewals null only in the future, and the units table written.

    python test_step_04.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from step_00_validate_raw import validate_raw
from step_02_apply_calendar import apply_calendar
from step_03_fine_table import build_fine_table
from step_04_forecast_units import build_forecast_units
from test_helpers import check, check_stops, console_of, count_status, finish, synthetic_with


def fine_synthetic(**overrides):
    """The synthetic raw after steps 00, 02 and 03, and its Config."""
    configuration = synthetic_with(**overrides)
    frames = {}
    def run_steps():
        calendared = apply_calendar(validate_raw(configuration.read_raw(), configuration), configuration)
        frames["fine"] = build_fine_table(calendared, configuration)
    console_of(run_steps)
    return frames["fine"], configuration


def run_step(fine, configuration):
    frames = {}
    console = console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    return frames["units"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_units() -> None:
    print("A · one row per forecast unit, the money conserved")
    fine, configuration = fine_synthetic()
    units, console = run_step(fine, configuration)
    check(len(units) == fine["fu_id"].nunique() and units["fu_id"].is_unique, "one row per fu_id")
    two_combinations = units[units["n_filas_finas"] == 2].iloc[0]
    fine_rows_of_unit = fine[fine["fu_id"] == two_combinations["fu_id"]]
    check(two_combinations["total_tr_usd"] == fine_rows_of_unit["total_tr_usd"].sum(),
          "a unit with two combinations carries the sum of its two fine rows")
    check(units["total_renewed_units"].sum() == fine["total_renewed_units"].sum(), "renewals conserved")
    projection = units["rol"] == "proyeccion"
    check(units.loc[projection, "total_renewed_units"].isna().all() and units.loc[~projection, "total_renewed_units"].notna().all(),
          "renewals null in the projection, numbers in closed months")
    check(count_status(console, "ok") == 5 and "5 checks: 5 ok" in console, "the 5 checks pass and are logged")
    written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_fact_fu", configuration.sql_engine)["n"].item()
    check(written == len(units), "sff_fact_fu written with every unit")


def test_the_failures() -> None:
    print("B · what stops and what warns")
    fine, configuration = fine_synthetic()
    unit_with_two_rows = fine["fu_id"][fine["fu_id"].duplicated()].iloc[0]
    mixed_flag = fine.copy()
    mixed_flag.loc[mixed_flag.index[mixed_flag["fu_id"] == unit_with_two_rows][0], "flag_time_series"] = 1
    check_stops(lambda: build_forecast_units(mixed_flag, configuration), "is the same in every fine row of a unit",
                "a unit whose fine rows disagree on the time_series flag stops")
    single_row_unit = fine["fu_id"].iloc[0]
    wiped = fine.copy()
    wiped.loc[wiped["fu_id"] == single_row_unit, "total_tr_units"] = 0          # the calendar wiped it, the extract had it
    _, console = run_step(wiped, configuration)
    check("WARN  every unit has something falling due in the extract" not in console
          and "1 wiped on purpose by the calendar" in console,
          "a unit the calendar wiped on purpose (due in the extract, 0 after the calendar) is counted apart, not warned")
    zero_in_extract = wiped.copy()
    zero_in_extract.loc[zero_in_extract["fu_id"] == single_row_unit, "s0_vencen_unidades"] = 0
    _, console = run_step(zero_in_extract, configuration)
    check("WARN  every unit has something falling due in the extract" in console,
          "a unit at 0 in the extract itself is a warning")


if __name__ == "__main__":
    test_the_units()
    test_the_failures()
    finish()
