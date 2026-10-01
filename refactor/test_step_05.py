"""
test_step_05.py — Step 05 on the synthetic raw: one row per id in each lookup, every key
of the facts resolved, the three lookups written.

    python test_step_05.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from step_04_forecast_units import build_forecast_units
from step_05_lookups import build_lookups
from test_helpers import check, check_stops, console_of, count_status, finish
from test_step_04 import fine_synthetic


def units_synthetic():
    fine, configuration = fine_synthetic()
    frames = {}
    console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    return fine, frames["units"], configuration


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_lookups() -> None:
    print("A · the three lookups")
    fine, units, configuration = units_synthetic()
    lookups = {}
    console = console_of(lambda: lookups.update(build_lookups(fine, units, configuration)))
    check(len(lookups["lookup_fs"]) == 17 and len(lookups["lookup_fu"]) == 747
          and len(lookups["lookup_uplift_cell"]) == fine["uplift_cell_id"].nunique(),
          "17 series, 747 units, one row per uplift cell")
    check(list(lookups["lookup_fs"].columns[2:]) == configuration.rate_series_columns,
          "the series lookup carries the series columns, in the order of the id")
    check(list(lookups["lookup_uplift_cell"].columns[2:]) == configuration.uplift_cell_columns,
          "the uplift cells lookup carries the cell columns (mandatory + extra_revalorizacion)")
    check(count_status(console, "ok") == 5 and "5 checks: 5 ok" in console, "the 5 checks pass and are logged")
    for table_name, lookup in lookups.items():
        written = pd.read_sql(f"SELECT COUNT(*) AS n FROM sff_{table_name}", configuration.sql_engine)["n"].item()
        check(written == len(lookup), f"sff_{table_name} written with every row")


def test_an_unresolved_key_stops() -> None:
    print("B · a key of the facts missing from its lookup stops")
    fine, units, configuration = units_synthetic()
    orphan_units = units.copy()
    orphan_units.loc[0, "fs_key"] = 1        # a series key that no fine row has
    check_stops(lambda: build_lookups(fine, orphan_units, configuration), "fact_fu.fs_key",
                "a units row whose series key is not in the series lookup stops, naming the key")


if __name__ == "__main__":
    test_the_lookups()
    test_an_unresolved_key_stops()
    finish()
