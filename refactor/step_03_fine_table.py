"""
step_03_fine_table.py — The fine table: every row of the raw, with the ids that name it.

The forecast of a future row is: what falls due × renewal rate × uplift. Each row is
therefore named by the two contexts it takes its numbers from:
  · RATE SERIES (fs_id): its mandatory, timevarying and extra_renovacion values. Every
    distinct combination is one series with its own renewal rate over time.
  · FORECAST UNIT (fu_id): its series in its month (fs_id | month). The rate is
    predicted per unit.
  · UPLIFT CELL (uplift_cell_id): its uplift mandatory dims and extra_revalorizacion
    values (region, product, discount bucket...). The price at which it renews is
    estimated per cell, over every month of the cell.
Every id has a key (a hash of the id): the BI joins on keys. fila_key names one fine
row: its unit and its revaluation values (fu_id || extra_revalorizacion). Nothing is
aggregated here and no measure changes; step 04 adds the fine rows of each unit.

Actions (logged as they are done):
  1. build fs_id from the rate series columns
  2. build fu_id: fs_id | month
  3. build uplift_cell_id from the uplift cell columns
  4. derive the keys: fs_key, fu_key, uplift_cell_key, fila_key
  5. check the grain, the keys and that nothing else changed         checks 1-3
  6. write the fine table                                            check 4
  7. count the checks; stop if any failed
  8. show a few rows with their ids, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. one row per forecast unit and revaluation values (the grain of the raw)
   2. every distinct id has its own key (no hash collision), for the four keys
   3. same rows, same order, no column of step 02 changed
   4. table sff_fact_fine written and read back

Output: the raw of step 02 with seven new columns (fs_id, fu_id, uplift_cell_id,
fs_key, fu_key, uplift_cell_key, fila_key) · table sff_fact_fine.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import COMBINED_ID_SEPARATOR, ID_FIELD_SEPARATOR, NO_REVALUATION_VALUES, Config, hash_key, join_columns
from vocabulario import (ROW_KEY_COLUMN, SERIES_ID_COLUMN, SERIES_KEY_COLUMN, TABLE_FINE, UNIT_ID_COLUMN,
                         UNIT_KEY_COLUMN, UPLIFT_CELL_ID_COLUMN, UPLIFT_CELL_KEY_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "03"
STEP_NAME = "FINE TABLE"
STEP_PURPOSE = ("name every row of the raw by the contexts it takes its numbers from: its rate series (fs_id), its "
                "forecast unit (series in its month, fu_id) and its uplift cell (price context, uplift_cell_id), "
                "each with a key the BI joins on; nothing is aggregated")
STEP_ACTIONS = ["build fs_id from the rate series columns",
                "build fu_id: fs_id | month",
                "build uplift_cell_id from the uplift cell columns",
                "derive the keys: fs_key, fu_key, uplift_cell_key, fila_key",
                "check the grain, the keys and that nothing else changed (checks 1-3)",
                "write the fine table (check 4)",
                "count the checks; stop if any failed",
                "show a few rows with their ids, as a table"]
STEP_OUTPUT = "the raw with seven id and key columns · table sff_fact_fine (one row per raw row)"

# ─── named constants ─────────────────────────────────────────────────────────────
EXAMPLE_ROWS_SHOWN = 3
# Each id column and the key derived from it.
KEY_OF_ID = {SERIES_ID_COLUMN: SERIES_KEY_COLUMN, UNIT_ID_COLUMN: UNIT_KEY_COLUMN,
             UPLIFT_CELL_ID_COLUMN: UPLIFT_CELL_KEY_COLUMN}


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

    # [3] the uplift cell: the price context of the row
    fine_table[UPLIFT_CELL_ID_COLUMN] = join_columns(fine_table, configuration.uplift_cell_columns)
    configuration.log_action(STEP_LABEL, 3, f"uplift_cell_id from {', '.join(configuration.uplift_cell_columns)}: "
                                            f"{fine_table[UPLIFT_CELL_ID_COLUMN].nunique():,} uplift cells")

    # [4] the keys: a hash of each id; the row key is the unit and its revaluation values
    for id_column, key_column in KEY_OF_ID.items():
        fine_table[key_column] = fine_table[id_column].map(hash_key)
    fine_table[ROW_KEY_COLUMN] = (fine_table[UNIT_ID_COLUMN] + COMBINED_ID_SEPARATOR
                                  + revaluation_values(fine_table, configuration)).map(hash_key)
    configuration.log_action(STEP_LABEL, 4, "keys derived: fs_key, fu_key, uplift_cell_key, fila_key")

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
    configuration.show_table(fine_table[[configuration.period_col, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
                                         ROW_KEY_COLUMN]].head(EXAMPLE_ROWS_SHOWN))
    return fine_table


def revaluation_values(fine_table: pd.DataFrame, configuration: Config) -> pd.Series:
    """The extra_revalorizacion values of every row, joined ("na" when none is declared):
    with the unit, they are what makes a fine row unique."""
    if configuration.extra_revalorizacion:
        return join_columns(fine_table, configuration.extra_revalorizacion)
    return pd.Series(NO_REVALUATION_VALUES, index=fine_table.index)


def check_grain(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 1: no two raw rows are the same unit with the same revaluation values. If they
    were, the raw would not be at the grain the Config declares and every later sum would
    mix them. When it fails, the check says WHY: which columns of the raw take different
    values inside the repeated rows (a column missing from the taxonomy), or that only the
    money differs (the extract is below the declared grain and must be aggregated). The
    first repeated group is shown whole, with those columns."""
    grain_columns = [UNIT_ID_COLUMN] + configuration.extra_revalorizacion
    repeated_mask = fine_table.duplicated(grain_columns, keep=False)
    repeated_rows_count = int(fine_table.duplicated(grain_columns).sum())
    check_description = "one row per forecast unit and revaluation values (the grain of the raw)"
    if repeated_rows_count == 0:
        configuration.log_check(STEP_LABEL, check_log, check_description, True,
                                context=f"{len(fine_table):,} rows = {len(fine_table):,} distinct fine rows")
        return

    # which columns separate the repeated rows: the ones that vary inside a repeated group
    repeated_rows = fine_table[repeated_mask]
    splitting_columns = columns_that_split_the_grain(repeated_rows, grain_columns, configuration)
    if splitting_columns:
        cause_text = ("columns of the raw that differ inside the repeated rows (repeated groups where they differ): "
                      + ", ".join(f"{column_name} ({group_count:,})" for column_name, group_count in splitting_columns.items())
                      + " → declare them as dimensions, or aggregate the extract without them")
    else:
        cause_text = ("no column but the money differs inside the repeated rows: the extract is below the declared "
                      "grain (e.g. one row per transaction) → aggregate it to the declared dimensions")

    first_row_key = repeated_rows.iloc[0][ROW_KEY_COLUMN]
    first_group = repeated_rows[repeated_rows[ROW_KEY_COLUMN] == first_row_key]
    shown_columns = ([configuration.period_col, UNIT_ID_COLUMN] + configuration.extra_revalorizacion
                     + [column_name for column_name in splitting_columns if column_name not in configuration.extra_revalorizacion]
                     + configuration.core_measures)
    configuration.log_check(STEP_LABEL, check_log, check_description, False,
                            failure_detail=f"{repeated_rows_count:,} rows repeat a fine row already seen "
                                           f"({repeated_mask.sum():,} rows involved). {cause_text}",
                            examples=first_group[shown_columns])


def columns_that_split_the_grain(repeated_rows: pd.DataFrame, grain_columns: list, configuration: Config) -> dict:
    """The raw columns (not measures, not the ids added here) that take more than one value
    inside at least one repeated fine row, with how many groups: they are what makes the
    raw finer than the declared dimensions. Sorted by groups, most first."""
    measure_columns = set(configuration.core_measures) | set(configuration.extra_measure_cols)
    added_columns = set(KEY_OF_ID) | set(KEY_OF_ID.values()) | {ROW_KEY_COLUMN}
    candidate_columns = [column_name for column_name in repeated_rows.columns
                         if column_name not in measure_columns and column_name not in added_columns
                         and column_name not in grain_columns and not column_name.startswith("s0_")]
    grouped = repeated_rows.groupby(ROW_KEY_COLUMN, sort=False)
    splitting_columns = {}
    for column_name in candidate_columns:
        groups_where_it_varies = int((grouped[column_name].nunique(dropna=False) > 1).sum())
        if groups_where_it_varies:
            splitting_columns[column_name] = groups_where_it_varies
    return dict(sorted(splitting_columns.items(), key=lambda item: -item[1]))


def check_keys(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 2: every distinct id has its own key. Two ids with the same key would be
    joined as one by the BI (a hash collision: 48 bits, never observed, but checked)."""
    collisions = {}
    for id_column, key_column in KEY_OF_ID.items():
        distinct_ids = fine_table[id_column].nunique()
        distinct_keys = fine_table[key_column].nunique()
        if distinct_ids != distinct_keys:
            collisions[key_column] = distinct_ids - distinct_keys
    distinct_rows = fine_table[[UNIT_ID_COLUMN] + configuration.extra_revalorizacion].drop_duplicates().shape[0]
    distinct_row_keys = fine_table[ROW_KEY_COLUMN].nunique()
    if distinct_rows != distinct_row_keys:
        collisions[ROW_KEY_COLUMN] = distinct_rows - distinct_row_keys
    configuration.log_check(STEP_LABEL, check_log, "every distinct id has its own key (no hash collision)",
                            not collisions,
                            failure_detail=f"keys shared by different ids: {collisions}",
                            context=f"{fine_table[SERIES_KEY_COLUMN].nunique():,} fs_key · "
                                    f"{fine_table[UNIT_KEY_COLUMN].nunique():,} fu_key · "
                                    f"{fine_table[UPLIFT_CELL_KEY_COLUMN].nunique():,} uplift_cell_key · "
                                    f"{fine_table[ROW_KEY_COLUMN].nunique():,} fila_key")
