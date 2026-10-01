"""
test_step_21.py — The rows the framework adds or wipes, month by month in one format: the gaps of step 08,
what step 02 wiped, what step 17 created; the raw = what stays + what was wiped. A report, not a table:
it is not written to SQL, and its SQL over sff_nucleo is printed.

    python test_step_21.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from main import run
from test_helpers import check, console_of, finish, synthetic_with


def test_the_new_rows() -> None:
    print("A · the new rows on the synthetic")
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    check("1 checks: 1 ok" in console.split("STEP 21")[1], "the check of step 21 passes")
    check("sff_filas_nuevas" not in console and "the SQL that reproduces it" in console.split("STEP 21")[1],
          "step 21 is a report: it writes no table and prints the SQL that reproduces it from sff_nucleo")
    table = results["new_rows"]
    check(list(table.columns) == ["mes", "rol", "origen", "filas", "series", "unidades", "usd", "esperado_usd"],
          "one format for every origin: month, role, origin, rows, series, units, USD, USD expected")
    rated = results["rated_units"]
    gaps = table[table["origen"] == "hueco"]
    check(int(gaps["filas"].sum()) == int((rated["sintetica"] == 1).sum()) and gaps["unidades"].sum() == 0 and gaps["usd"].sum() == 0,
          "every gap of step 08 is counted, with no units and no money (a month, not money)")
    forecast = results["forecast"]["forecast"]
    projected = table[table["origen"] == "proyectada"]
    check(abs(projected["usd"].sum() - forecast.loc[forecast["origen_pipeline"] == "proyectada", "total_tr_usd"].sum()) < 1e-6
          and abs(projected["esperado_usd"].sum() - forecast.loc[forecast["origen_pipeline"] == "proyectada", "esperado_usd"].sum()) < 1e-6,
          "the projected rows are the forecast's, with their USD due and expected")
    fine = results["fine_table"]
    wiped_results = table[table["origen"] == "resultado_adelantado_borrado"]
    check(abs(wiped_results["unidades"].sum() - (fine["s0_renovados_unidades"].fillna(0) - fine["total_renewed_units"].fillna(0)).sum()) < 1e-6,
          "the wiped early results are the difference between the raw and what stays")
    check("raw" in console and "= kept" in console and "+ wiped" in console, "the reconciliation with the raw is on screen")


if __name__ == "__main__":
    test_the_new_rows()
    finish()
