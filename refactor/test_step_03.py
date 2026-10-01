"""
test_step_03.py — Step 03 on the synthetic raw: every row gets its ids and keys, the
grain is checked, nothing else changes, and the fine table is written.

    python test_step_03.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from config import hash_key
from step_00_validate_raw import validate_raw
from step_02_apply_calendar import apply_calendar
from step_03_fine_table import build_fine_table
from test_helpers import check, check_stops, console_of, console_of_failure, count_status, finish, synthetic_with


def calendared_synthetic(**overrides):
    """The synthetic raw after steps 00 and 02, and its Config."""
    configuration = synthetic_with(**overrides)
    frames = {}
    def run_steps():
        frames["raw"] = apply_calendar(validate_raw(configuration.read_raw(), configuration), configuration)
    console_of(run_steps)
    return frames["raw"], configuration


def run_step(calendared, configuration):
    """Step 03, returning (fine table, console)."""
    frames = {}
    console = console_of(lambda: frames.setdefault("fine", build_fine_table(calendared, configuration)))
    return frames["fine"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_ids() -> None:
    print("A · the ids and keys of every row")
    calendared, configuration = calendared_synthetic()
    fine, console = run_step(calendared, configuration)
    first = fine.iloc[0]
    expected_series = "|".join(str(first[column]) for column in configuration.rate_series_columns)
    check(first["fs_id"] == expected_series, "fs_id = mandatory | timevarying | extra_renovacion, in that order")
    check(first["fu_id"] == f"{expected_series}|{first['period']}", "fu_id = fs_id | month")
    check(first["uplift_cell_id"] == f"{first['region']}|{first['product']}|{first['newcust']}|{first['tramo_descuento']}",
          "uplift_cell_id = mandatory | extra_revalorizacion | discount bucket (no timevarying, no extra_renovacion, no month)")
    check(first["fu_key"] == hash_key(first["fu_id"])
          and first["fila_key"] == hash_key(f"{first['fu_id']}||{first['newcust']}|"
                                            f"{'null' if pd.isna(first['discount_pct']) else first['discount_pct']}"),
          "every key is the hash of its id (fila_key: fu_id || extra_revalorizacion values | exact discount)")
    bucket_of_forty = fine.loc[fine["discount_pct"] == 0.4, "tramo_descuento"].unique().tolist()
    bucket_of_null = fine.loc[fine["discount_pct"].isna(), "tramo_descuento"].unique().tolist()
    check(bucket_of_forty == ["05 _ 40-50%"] and bucket_of_null == ["sin_dato"],
          "the bucket is derived like the extract's discount_interval: 0.40 → '05 _ 40-50%', null → 'sin_dato'")
    check(fine["uplift_cell_id"].nunique() < fine["fu_id"].nunique(),
          "many fine rows share an uplift cell: every month and signal state of the same price context")
    subset_cells = {}
    console_of(lambda: subset_cells.setdefault("fine", build_fine_table(calendared, synthetic_with(uplift_mandatory_dims=["region"]))))
    check(subset_cells["fine"]["uplift_cell_id"].iloc[0] == f"{first['region']}|{first['newcust']}|{first['tramo_descuento']}",
          "uplift_mandatory_dims narrows the cell to the declared mandatory dims")
    check(fine[list(calendared.columns)].equals(calendared) and len(fine.columns) == len(calendared.columns) + 8,
          "same rows, same order, eight columns added (bucket + seven ids and keys), nothing else changed")
    check(count_status(console, "ok") == 5 and "5 checks: 5 ok" in console, "the 5 checks pass and are logged")

    calendared_no_extras, configuration_no_extras = calendared_synthetic(
        extra_revalorizacion=[], discount_value_column=None,
        ignore_cols=["dataset_role", "is_current_month", "discount", "newcust", "discount_pct"])
    one_row_per_unit = calendared_no_extras.drop_duplicates(configuration_no_extras.rate_series_columns + ["period"])
    no_extras, _ = run_step(one_row_per_unit, configuration_no_extras)
    check(no_extras["fila_key"].iloc[0] == hash_key(f"{no_extras['fu_id'].iloc[0]}||na"),
          "with no extra_revalorizacion, the row key is fu_id || na")
    finer_console = console_of_failure(lambda: build_fine_table(calendared_no_extras, configuration_no_extras))
    check(count_status(finer_console, "FAIL") == 1,
          "a raw finer than the declared dimensions fails the grain, and only the grain (no false key collision)")
    check("differ inside the repeated rows" in finer_console and "discount_pct (" in finer_console and "newcust (" in finer_console,
          "the failure names the columns that split the grain (the ones left out of the taxonomy)")
    check("--    table sff_fact_fine written and read back" in finer_console and "not written" in finer_console,
          "when the grain fails, the table is not written")
    check("together they explain" in finer_console and "HAVING COUNT(*) > 1" in finer_console
          and "COUNT(DISTINCT discount_pct)" in finer_console and "GROUP BY period" in finer_console,
          "the failure says how many groups the columns explain and writes the SQL that reproduces it")


def test_one_row_per_exact_discount() -> None:
    print("D · the real extract: one row per exact discount inside the same bucket")
    calendared, configuration = calendared_synthetic()
    with_forty = calendared.index[calendared["discount_pct"] == 0.4][0]
    twin = calendared.loc[[with_forty]].copy()
    twin["discount_pct"] = 0.43                     # same unit, same bucket (40-50 %), another exact discount
    two_discounts = pd.concat([calendared, twin], ignore_index=True)
    fine, console = run_step(two_discounts, configuration)
    pair = fine[(fine["fu_id"] == fine.loc[with_forty, "fu_id"]) & fine["discount_pct"].isin([0.4, 0.43])]
    check("5 checks: 5 ok" in console and len(pair) == 2 and pair["fila_key"].nunique() == 2,
          "the grain passes: two fine rows, one per exact discount")
    check(pair["tramo_descuento"].nunique() == 1 and pair["uplift_cell_id"].nunique() == 1,
          "both fall in the same bucket and the same uplift cell (the cell does not split by exact discount)")
    from step_04_forecast_units import build_forecast_units
    frames = {}
    console_of(lambda: frames.setdefault("units", build_forecast_units(fine, configuration)))
    unit = frames["units"].set_index("fu_id").loc[pair["fu_id"].iloc[0]]
    check(unit["total_tr_usd"] == fine.loc[fine["fu_id"] == pair["fu_id"].iloc[0], "total_tr_usd"].sum(),
          "step 04 adds both rows into their forecast unit: the rate does not depend on the discount")


def test_the_discount_buckets() -> None:
    print("C · the discount buckets, like the extract's discount_interval")
    from config import discount_bucket_labels
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = discount_bucket_labels(pd.Series([0.0, 0.05, 0.1, 0.65, 0.6, 0.7, 0.999, 1.0, None]), edges).tolist()
    check(labels == ["01 _ 0-10%", "01 _ 0-10%", "02 _ 10-20%", "07 _ 60-70%", "07 _ 60-70%", "08 _ 70-80%",
                     "10 _ 90-100%", "10 _ 90-100%", "sin_dato"],
          "[low, high) in %, the last bucket closed at 100 %, null → sin_dato; 0.65 → '07 _ 60-70%'")
    check_stops(lambda: synthetic_with(discount_bucket_edges=[0, 50, 40, 100]), "discount_bucket_edges",
                "edges that do not go up from 0 to 100 stop when the Config is built")


def test_the_grain_and_the_table() -> None:
    print("B · the grain and the written table")
    calendared, configuration = calendared_synthetic()
    doubled = pd.concat([calendared, calendared.iloc[[0]]], ignore_index=True)
    check_stops(lambda: build_fine_table(doubled, configuration), "one row per forecast unit, revaluation values and exact discount",
                "two rows with the same unit and revaluation values stop the step")
    failing_console = console_of_failure(lambda: build_fine_table(doubled, configuration))
    check("example rows below" in failing_console, "the first repeated group is shown as a table under the check")
    check("no column but the money differs" in failing_console,
          "a row repeated whole is diagnosed as below the declared grain (only the money could differ)")

    fine, _ = run_step(calendared, configuration)
    written = pd.read_sql("SELECT * FROM sff_fact_fine", configuration.sql_engine)
    check(len(written) == len(fine) and set(fine.columns) == set(written.columns),
          "sff_fact_fine holds every row and every column")
    check(written["fila_key"].astype("int64").tolist() == fine["fila_key"].tolist(),
          "the keys survive the database as integers")


if __name__ == "__main__":
    test_the_ids()
    test_the_grain_and_the_table()
    test_the_discount_buckets()
    test_one_row_per_exact_discount()
    finish()
