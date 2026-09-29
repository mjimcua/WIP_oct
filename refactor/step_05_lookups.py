"""
step_05_lookups.py — The lookups: every id next to its key and its attributes.

The fact tables carry keys (a hash of each id); the BI joins on them. The lookups are
where an id, its key and the columns that make it sit side by side, one row per id:
  · lookup_fs    one row per rate series: fs_id, fs_key and the series columns
  · lookup_fu    one row per forecast unit: fu_id, fu_key, its series key and its month
  · lookup_uplift_cell  one row per uplift cell: uplift_cell_id, uplift_cell_key and its columns

Actions (logged as they are done):
  1. build the series lookup from the fine table
  2. build the units lookup from the forecast units
  3. build the uplift cells lookup from the fine table
  4. check each lookup and that every key of the facts is in it      checks 1-2
  5. write the three lookups                                         checks 3-5
  6. count the checks; stop if any failed
  7. show the size of the lookups, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. one row per id and per key in each lookup
   2. every key of sff_fact_fine and sff_fact_fu is in its lookup
   3-5. tables sff_lookup_fs, sff_lookup_fu, sff_lookup_uplift_cell written and read back

Output: a dict with the three lookups · tables sff_lookup_fs, sff_lookup_fu, sff_lookup_uplift_cell.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import Config
from vocabulario import (SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_LOOKUP_SERIES, TABLE_LOOKUP_UNITS,
                         TABLE_LOOKUP_UPLIFT_CELLS, UNIT_ID_COLUMN, UNIT_KEY_COLUMN, UPLIFT_CELL_ID_COLUMN,
                         UPLIFT_CELL_KEY_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "05"
STEP_NAME = "LOOKUPS"
STEP_PURPOSE = ("put every id next to its key and the columns that make it (series, units, uplift cells), "
                "so the keys of the fact tables can be read outside the framework")
STEP_ACTIONS = ["build the series lookup from the fine table",
                "build the units lookup from the forecast units",
                "build the uplift cells lookup from the fine table",
                "check each lookup and that every key of the facts is in it (checks 1-2)",
                "write the three lookups (checks 3-5)",
                "count the checks; stop if any failed",
                "show the size of the lookups, as a table"]
STEP_OUTPUT = "three lookups (series, units, uplift cells) · tables sff_lookup_fs, sff_lookup_fu, sff_lookup_uplift_cell"


def build_lookups(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, configuration: Config) -> dict:
    """Build, check and write the three id ↔ key lookups."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] one row per rate series, with the columns that define it
    series_lookup = (fine_table[[SERIES_ID_COLUMN, SERIES_KEY_COLUMN] + configuration.rate_series_columns]
                     .drop_duplicates(SERIES_ID_COLUMN).reset_index(drop=True))
    configuration.log_action(STEP_LABEL, 1, f"series lookup: {len(series_lookup):,} series × "
                                            f"{len(configuration.rate_series_columns)} series columns")

    # [2] one row per forecast unit, with its series and its month
    units_lookup = forecast_units[[UNIT_ID_COLUMN, UNIT_KEY_COLUMN, SERIES_KEY_COLUMN, configuration.period_col]].copy()
    configuration.log_action(STEP_LABEL, 2, f"units lookup: {len(units_lookup):,} forecast units")

    # [3] one row per uplift cell, with its columns
    uplift_cells_lookup = (fine_table[[UPLIFT_CELL_ID_COLUMN, UPLIFT_CELL_KEY_COLUMN] + configuration.uplift_cell_columns]
                           .drop_duplicates(UPLIFT_CELL_ID_COLUMN).reset_index(drop=True))
    configuration.log_action(STEP_LABEL, 3, f"uplift cells lookup: {len(uplift_cells_lookup):,} cells × "
                                            f"{len(configuration.uplift_cell_columns)} cell columns")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the lookups and that every key of the facts is in them")
    lookups = {TABLE_LOOKUP_SERIES: (series_lookup, SERIES_ID_COLUMN, SERIES_KEY_COLUMN),
               TABLE_LOOKUP_UNITS: (units_lookup, UNIT_ID_COLUMN, UNIT_KEY_COLUMN),
               TABLE_LOOKUP_UPLIFT_CELLS: (uplift_cells_lookup, UPLIFT_CELL_ID_COLUMN, UPLIFT_CELL_KEY_COLUMN)}
    not_unique = {table_name: {"ids repeated": int(lookup[id_column].duplicated().sum()),
                               "keys repeated": int(lookup[key_column].duplicated().sum())}
                  for table_name, (lookup, id_column, key_column) in lookups.items()
                  if lookup[id_column].duplicated().any() or lookup[key_column].duplicated().any()}
    configuration.log_check(STEP_LABEL, check_log, "one row per id and per key in each lookup", not not_unique,
                            failure_detail=f"lookups with repeated ids or keys: {not_unique}",
                            context=" · ".join(f"{table_name} {len(lookup):,}" for table_name, (lookup, _, _) in lookups.items()))

    keys_to_resolve = {"fact_fine.fs_key": (fine_table[SERIES_KEY_COLUMN], series_lookup[SERIES_KEY_COLUMN]),
                       "fact_fine.fu_key": (fine_table[UNIT_KEY_COLUMN], units_lookup[UNIT_KEY_COLUMN]),
                       "fact_fine.uplift_cell_key": (fine_table[UPLIFT_CELL_KEY_COLUMN], uplift_cells_lookup[UPLIFT_CELL_KEY_COLUMN]),
                       "fact_fu.fs_key": (forecast_units[SERIES_KEY_COLUMN], series_lookup[SERIES_KEY_COLUMN])}
    unresolved = {key_name: int((~fact_keys.isin(set(lookup_keys))).sum())
                  for key_name, (fact_keys, lookup_keys) in keys_to_resolve.items()
                  if not fact_keys.isin(set(lookup_keys)).all()}
    configuration.log_check(STEP_LABEL, check_log, "every key of sff_fact_fine and sff_fact_fu is in its lookup",
                            not unresolved,
                            failure_detail=f"fact rows whose key is not in the lookup: {unresolved}",
                            context=f"{len(keys_to_resolve)} keys resolved")

    # [5] the three lookups, written
    configuration.log_action(STEP_LABEL, 5, "writing the three lookups")
    for table_name, (lookup, _, _) in lookups.items():
        configuration.write_table(STEP_LABEL, check_log, lookup, table_name)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the size of each lookup
    configuration.log_action(STEP_LABEL, 7, "the lookups:")
    configuration.show_table(pd.DataFrame([{"tabla": f"{configuration.table_prefix}{table_name}", "id": id_column,
                                            "clave": key_column, "filas": len(lookup), "columnas": len(lookup.columns)}
                                           for table_name, (lookup, id_column, key_column) in lookups.items()]))
    return {table_name: lookup for table_name, (lookup, _, _) in lookups.items()}
