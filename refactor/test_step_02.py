"""
test_step_02.py — Step 02 on the synthetic raw: the calendar lands on every row, the
null renewals of closed months become 0, the future's early results are wiped, and
nothing else changes.

    python test_step_02.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_00_validate_raw import validate_raw
from step_02_apply_calendar import apply_calendar
from test_helpers import check, console_of, count_status, finish, synthetic_with
from vocabulario import (CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN, ROLE_PROJECTION,
                         ROLE_TEST, ROLE_TRAIN, S0_RENEWED_UNITS_COLUMN)


def validated_synthetic(**overrides):
    """The synthetic raw after step 00, and its Config."""
    configuration = synthetic_with(**overrides)
    validated = {}
    console_of(lambda: validated.setdefault("raw", validate_raw(configuration.read_raw(), configuration)))
    return validated["raw"], configuration


def run_step(raw, configuration):
    """Step 02, returning (calendared raw, console)."""
    calendared = {}
    console = console_of(lambda: calendared.setdefault("raw", apply_calendar(raw, configuration)))
    return calendared["raw"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_calendar_lands_on_every_row() -> None:
    print("A · the calendar on the synthetic (current 2026-09, 3 exam months)")
    raw, configuration = validated_synthetic()
    calendared, console = run_step(raw, configuration)
    role_of_month = calendared.groupby(calendared["period"].astype(str))[CALENDAR_ROLE_COLUMN].first()
    check(role_of_month["2026-05"] == ROLE_TRAIN and role_of_month["2026-06"] == ROLE_TEST
          and role_of_month["2026-08"] == ROLE_TEST and role_of_month["2026-09"] == ROLE_PROJECTION,
          "entrenamiento ≤ 2026-05 · examen 2026-06..08 · proyeccion ≥ 2026-09")
    check(set(calendared[CALENDAR_ROLE_COLUMN]) == {ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION},
          "three roles: entrenamiento · examen · proyeccion")
    current_rows = calendared[CURRENT_MONTH_COLUMN] == 1
    check(set(calendared.loc[current_rows, "period"].astype(str)) == {"2026-09"}, "only 2026-09 is the current month")
    check(len(calendared) == len(raw) and list(calendared.columns[:len(raw.columns)]) == list(raw.columns),
          "no row lost, the raw's columns first and in order")
    check(count_status(console, "ok") == 10 and "10 checks: 10 ok" in console, "the 10 checks pass and are logged")
    calendar_table = pd.read_sql("SELECT * FROM sff_calendario", configuration.sql_engine)
    check(len(calendar_table) == 48 and calendar_table.loc[calendar_table["period"] == "2026-09", "rol"].item() == ROLE_PROJECTION
          and calendar_table.loc[calendar_table["period"] == "2026-09", "es_mes_en_curso"].item() == 1,
          "sff_calendario: 48 months, 2026-09 is proyeccion and the current month")
    september = calendar_table[calendar_table["period"] == "2026-09"].iloc[0]
    check(pd.isna(september["unidades_renovadas"]) and september["s0_renovados_unidades"] > 0,
          "sff_calendario keeps what the raw had booked (s0_) next to the wiped renewal")
    check("actions:" in console and "output:" in console and "▸ 4." in console,
          "the step logs its actions and its output, and each action when it is done")

    # an immature month (late renewals still arriving) is handled by setting current_month on it
    immature, _ = run_step(*validated_synthetic(current_month="2026-08"))
    august = immature[immature["period"].astype(str) == "2026-08"]
    check((august[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION).all() and august["total_renewed_units"].isna().all()
          and (august["s0_renovados_unidades"].fillna(0) > 0).any(),
          "current_month on the first immature month (2026-08): it is proyeccion and its partial renewals are wiped")


def test_the_renewals() -> None:
    print("B · the renewals: closed nulls → 0, the future wiped, s0_ untouched")
    raw, configuration = validated_synthetic()
    closed_row = raw.index[raw["period"].astype(str) == "2024-03"][0]
    raw.loc[closed_row, ["total_renewed_units", "total_renewed_usd"]] = np.nan
    calendared, console = run_step(raw, configuration)
    check(calendared.loc[closed_row, "total_renewed_units"] == 0 and np.isnan(calendared.loc[closed_row, S0_RENEWED_UNITS_COLUMN]),
          "a null closed renewal becomes 0; s0_ keeps the null the raw had")
    check("1 null renewals read as 0" in console, "the conversion is counted in the log")
    future = calendared["period"] >= pd.Period("2026-09", freq="M")
    check(calendared.loc[future, "total_renewed_units"].isna().all() and calendared.loc[future, "total_renewed_usd"].isna().all(),
          "every renewal from the current month on is wiped")
    booked_in_september = raw.loc[raw["period"].astype(str) == "2026-09", "total_renewed_units"].sum()
    check(booked_in_september > 0 and f"{booked_in_september:,.0f} units" in console,
          "the early results of 2026-09 were there, and the log says how much was wiped")
    check(calendared["total_tr_usd"].sum() == raw["total_tr_usd"].sum(), "the pipeline is untouched (nothing due from 2027-09 on)")


def test_the_warnings() -> None:
    print("C · the warnings")
    raw, configuration = validated_synthetic(test_months=36)
    _, console = run_step(raw, configuration)
    check("WARN  training has at least 12 months" in console, "less than a year of training is a warning")
    raw, configuration = validated_synthetic()
    _, console = run_step(raw[raw["period"].astype(str) != "2026-07"], configuration)
    check("WARN  every exam month has rows" in console and "2026-07" in console, "an exam month with no rows is a warning")
    no_pipeline = raw.copy()
    no_pipeline.loc[no_pipeline["period"].astype(str) == "2026-09", "total_tr_units"] = 0
    _, console = run_step(no_pipeline, configuration)
    check("WARN  the current month (2026-09) has pipeline" in console, "a current month with no pipeline is a warning")


def test_pipeline_not_known_yet() -> None:
    print("C · the pipeline of 1-year licences sold or renewed from the current month on is wiped (kept in s0_)")
    raw, configuration = validated_synthetic()
    september_2027 = raw[raw["period"].astype(str) == "2026-09"].head(2).copy()
    september_2027["period"] = pd.Period("2027-09", freq="M")
    august_2027 = raw[raw["period"].astype(str) == "2026-08"].head(2).copy()
    august_2027["period"] = pd.Period("2027-08", freq="M")
    extended = pd.concat([raw, september_2027, august_2027], ignore_index=True)
    calendared, console = run_step(extended, configuration)
    wiped = calendared["period"].astype(str) == "2027-09"
    kept = calendared["period"].astype(str) == "2027-08"
    check((calendared.loc[wiped, "total_tr_usd"] == 0).all() and (calendared.loc[wiped, "s0_vencen_usd"] > 0).all(),
          "due 2027-09 (sold or renewed in 2026-09, the current month): pipeline 0, the raw's value kept in s0_vencen_usd")
    check((calendared.loc[kept, "total_tr_usd"] > 0).all(), "due 2027-08 (sold or renewed in 2026-08, closed): kept")
    check("10 checks: 10 ok" in console, "the checks pass with the wipe")


if __name__ == "__main__":
    test_the_calendar_lands_on_every_row()
    test_the_renewals()
    test_the_warnings()
    test_pipeline_not_known_yet()
    finish()
