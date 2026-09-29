"""
step_03_fine_table.py — The fine table: every row of the raw, with the ids that name it.

A row of the raw belongs to:
  · a RATE SERIES (fs_id): its mandatory, timevarying and extra_renovacion values. Every
    distinct combination of them is one series with its own renewal rate over time.
  · a FORECAST UNIT (fu_id): its series in its month (fs_id | month). The forecast is
    made per unit.
  · a REVALUATION COMBINATION (comb_id): its extra_revalorizacion values (the discount
    bucket, the new-customer mark...). A unit can hold several combinations: they share
    the renewal rate but not the price at which they renew.
Every id has a key (a hash of the id): the BI joins on keys. fu_comb_key names one fine
row: one unit in one combination. Nothing is aggregated here and no measure changes;
step 04 adds the combinations of each unit into the forecast units.

Actions (logged as they are done):
  1. build fs_id from the rate series columns
  2. build fu_id: fs_id | month
  3. build comb_id from the extra_revalorizacion columns (or "na" if none)
  4. derive the keys: fs_key, fu_key, comb_key, fu_comb_key
  5. check the grain, the keys and that nothing else changed         checks 1-3
  6. write the fine table                                            check 4
  7. count the checks; stop if any failed
  8. show a few rows with their ids, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. one row per forecast unit and combination (the grain of the raw)
   2. every distinct id has its own key (no hash collision), for the four keys
   3. same rows, same order, no column of step 02 changed
   4. table sff_fact_fine written and read back

Output: the raw of step 02 with seven new columns (fs_id, fu_id, comb_id, fs_key,
fu_key, comb_key, fu_comb_key) · table sff_fact_fine.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import COMBINED_ID_SEPARATOR, ID_FIELD_SEPARATOR, NO_COMBINATION_ID, Config, hash_key, join_columns
from vocabulario import (COMBINATION_ID_COLUMN, COMBINATION_KEY_COLUMN, SERIES_ID_COLUMN, SERIES_KEY_COLUMN,
                         TABLE_FINE, UNIT_COMBINATION_KEY_COLUMN, UNIT_ID_COLUMN, UNIT_KEY_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "03"
STEP_NAME = "FINE TABLE"
STEP_PURPOSE = ("name every row of the raw: its rate series (fs_id), its forecast unit (series in its month, fu_id) "
                "and its revaluation combination (comb_id), each with a key the BI joins on; nothing is aggregated")
STEP_ACTIONS = ["build fs_id from the rate series columns",
                "build fu_id: fs_id | month",
                "build comb_id from the extra_revalorizacion columns (or 'na' if none)",
                "derive the keys: fs_key, fu_key, comb_key, fu_comb_key",
                "check the grain, the keys and that nothing else changed (checks 1-3)",
                "write the fine table (check 4)",
                "count the checks; stop if any failed",
                "show a few rows with their ids, as a table"]
STEP_OUTPUT = "the raw with seven id and key columns · table sff_fact_fine (one row per raw row)"

# ─── named constants ─────────────────────────────────────────────────────────────
EXAMPLE_ROWS_SHOWN = 3
# Each id column and the key derived from it.
KEY_OF_ID = {SERIES_ID_COLUMN: SERIES_KEY_COLUMN, UNIT_ID_COLUMN: UNIT_KEY_COLUMN,
             COMBINATION_ID_COLUMN: COMBINATION_KEY_COLUMN}


def build_fine_table(calendared: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Add the ids and keys that name every row; check the grain; write the fine table."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    fine_table = calendared.copy()

    # [1] the rate series of every row
    fine_table[SERIES_ID_COLUMN] = join_columns(fine_table, configuration.rate_series_columns)
    configuration.log_action(STEP_LABEL, 1, f"fs_id from {len(configuration.rate_series_columns)} columns "
                                            f"({', '.join(configuration.rate_series_columns)}): "
                                            f"{fine_table[SERIES_ID_COLUMN].nunique():,} series")

    # [2] the forecast unit: the series in its month
    fine_table[UNIT_ID_COLUMN] = (fine_table[SERIES_ID_COLUMN] + ID_FIELD_SEPARATOR
                                  + fine_table[configuration.period_col].astype(str))
    configuration.log_action(STEP_LABEL, 2, f"fu_id = fs_id | month: {fine_table[UNIT_ID_COLUMN].nunique():,} forecast units")

    # [3] the revaluation combination
    if configuration.extra_revalorizacion:
        fine_table[COMBINATION_ID_COLUMN] = join_columns(fine_table, configuration.extra_revalorizacion)
        combination_source = ", ".join(configuration.extra_revalorizacion)
    else:
        fine_table[COMBINATION_ID_COLUMN] = NO_COMBINATION_ID
        combination_source = f"none declared: every row is '{NO_COMBINATION_ID}'"
    configuration.log_action(STEP_LABEL, 3, f"comb_id from {combination_source}: "
                                            f"{fine_table[COMBINATION_ID_COLUMN].nunique():,} combinations")

    # [4] the keys: a hash of each id
    for id_column, key_column in KEY_OF_ID.items():
        fine_table[key_column] = fine_table[id_column].map(hash_key)
    fine_table[UNIT_COMBINATION_KEY_COLUMN] = (fine_table[UNIT_ID_COLUMN] + COMBINED_ID_SEPARATOR
                                               + fine_table[COMBINATION_ID_COLUMN]).map(hash_key)
    configuration.log_action(STEP_LABEL, 4, "keys derived: fs_key, fu_key, comb_key, fu_comb_key")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the grain, the keys and that nothing else changed")
    check_grain(fine_table, configuration, check_log)
    check_keys(fine_table, configuration, check_log)
    new_columns = [column_name for column_name in fine_table.columns if column_name not in calendared.columns]
    configuration.log_check(STEP_LABEL, check_log, "same rows, same order, no column of step 02 changed",
                            fine_table[list(calendared.columns)].equals(calendared) and len(new_columns) == 7,
                            failure_detail="the fine table changed the rows or the columns it received",
                            context=f"{len(fine_table):,} rows · {len(new_columns)} columns added")

    # [6] the fine table, written
    configuration.log_action(STEP_LABEL, 6, "writing the fine table")
    configuration.write_table(STEP_LABEL, check_log, fine_table, TABLE_FINE)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] a few rows with their ids
    configuration.log_action(STEP_LABEL, 8, "a few rows with their ids:")
    configuration.show_table(fine_table[[configuration.period_col, SERIES_ID_COLUMN, UNIT_ID_COLUMN,
                                         COMBINATION_ID_COLUMN, UNIT_COMBINATION_KEY_COLUMN]].head(EXAMPLE_ROWS_SHOWN))
    return fine_table


def check_grain(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 1: no two raw rows are the same unit in the same combination. If they were,
    the raw would not be at the grain the Config declares (a column is missing from the
    taxonomy, or the extract repeats rows) and every later sum would double them."""
    repeated_rows = fine_table[fine_table.duplicated([UNIT_ID_COLUMN, COMBINATION_ID_COLUMN], keep=False)]
    repeated_pairs = int(fine_table.duplicated([UNIT_ID_COLUMN, COMBINATION_ID_COLUMN]).sum())
    configuration.log_check(STEP_LABEL, check_log, "one row per forecast unit and combination (the grain of the raw)",
                            repeated_pairs == 0,
                            failure_detail=f"{repeated_pairs:,} rows repeat a unit and combination already seen: the raw "
                                           f"is finer than the declared dimensions (a column missing from the taxonomy?) "
                                           f"or it repeats rows",
                            context=f"{len(fine_table):,} rows = {len(fine_table):,} unit × combination pairs",
                            examples=repeated_rows.sort_values([UNIT_ID_COLUMN, COMBINATION_ID_COLUMN])[
                                [configuration.period_col, UNIT_ID_COLUMN, COMBINATION_ID_COLUMN]
                                + configuration.core_measures])


def check_keys(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 2: every distinct id has its own key. Two ids with the same key would be
    joined as one by the BI (a hash collision: 48 bits, never observed, but checked)."""
    collisions = {}
    for id_column, key_column in KEY_OF_ID.items():
        distinct_ids = fine_table[id_column].nunique()
        distinct_keys = fine_table[key_column].nunique()
        if distinct_ids != distinct_keys:
            collisions[key_column] = distinct_ids - distinct_keys
    distinct_pairs = fine_table[[UNIT_ID_COLUMN, COMBINATION_ID_COLUMN]].drop_duplicates().shape[0]
    distinct_pair_keys = fine_table[UNIT_COMBINATION_KEY_COLUMN].nunique()
    if distinct_pairs != distinct_pair_keys:
        collisions[UNIT_COMBINATION_KEY_COLUMN] = distinct_pairs - distinct_pair_keys
    configuration.log_check(STEP_LABEL, check_log, "every distinct id has its own key (no hash collision)",
                            not collisions,
                            failure_detail=f"keys shared by different ids: {collisions}",
                            context=f"{fine_table[SERIES_KEY_COLUMN].nunique():,} fs_key · "
                                    f"{fine_table[UNIT_KEY_COLUMN].nunique():,} fu_key · "
                                    f"{fine_table[COMBINATION_KEY_COLUMN].nunique():,} comb_key · "
                                    f"{fine_table[UNIT_COMBINATION_KEY_COLUMN].nunique():,} fu_comb_key")
