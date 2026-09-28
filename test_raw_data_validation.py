"""test_raw_data_validation.py — checks for phase 0 (SFF v2).

Covers `run/raw_data_validation.py` (the operational part) and
`support_reference.py` (the immutable reference), function by function, on
small hand-built frames where every expected number can be verified by eye.

    python tests/test_raw_data_validation.py   →  exit 0 if healthy, 1 with the failures

Each block is the acceptance criterion of one function: the properties listed here are
what a regeneration of that function must satisfy. Blocks:

  0.1  validate_raw                 contract · period · money · nulls · all problems at once
       apply_current_month_doctrine current → projection · wipe · train untouched
  0.2  build_fine_table             ids · keys · one combination when no extras
       aggregate_to_forecast_units  sums · min_count · conservation · uniqueness
       build_key_lookups            one row per id, key = hash_key(id)
  0.3  route_from_coverage / label_universe_and_routes
                                    universe · fs_id · coverage · route · nothing dropped
  0.4  build_support_reference      the binomial bound and its dollars, persisted
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# every module lives in this folder; make sure it is importable.
# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import Config, hash_key
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from raw_data_validation import (apply_current_month_doctrine,
                                 aggregate_to_forecast_units, build_fine_table,
                                 build_key_lookups, label_universe_and_routes,
                                 route_from_coverage, validate_raw)
from support_reference import build_support_reference


# ─── named constants ─────────────────────────────────────────────────────────────
# The test taxonomy: one mandatory, one timevarying, one extra of each kind.
TEST_TAXONOMY = dict(business_mandatory_dims=["region"],
                     structural_timevarying_dims={"softcancel": "negative"},
                     extra_renovacion=["channel"],
                     extra_revalorizacion=["discount"])

# Column names of the test raw (the production defaults of Config).
PRODUCTION_COLUMNS = Config()

RECORDER = CheckRecorder()


def pattern(*roles) -> str:
    """The coverage pattern of a series: its roles sorted and joined with '_'."""
    return "_".join(sorted(roles))



# ─── fixtures ────────────────────────────────────────────────────────────────────

def build_test_config(**overrides) -> Config:
    """A Config on the test taxonomy; every other field is the production default."""
    config_arguments = dict(TEST_TAXONOMY)
    # the six-row raw spans four months: no pending month and one test month, so that
    # 2026-01 is train, 2026-02 test, 2026-03 (current) and later projection
    config_arguments.update(dict(pending_close_months=0, test_months=1))
    config_arguments.update(overrides)
    return Config(**config_arguments)


def build_test_raw() -> pd.DataFrame:
    """A six-row raw that exercises every path of phase 0.

    Rows (region | softcancel | channel | discount | period | role | current):
      0  EU 0 web d0   2026-01 train       0      series EU|0|web, history
      1  EU 0 web d40  2026-01 train       0      same forecast unit as row 0, other combination
      2  EU 0 web d0   2026-02 test        0
      3  EU 0 web d0   2026-03 projection  1      the CURRENT month, wrongly labeled? no: already projection
      4  EU 1 web d0   2026-03 test        1      the CURRENT month labeled test: must be reassigned
      5  NA 0 tele d0  2026-04 projection  0      future without history: heuristic route
    Rows 3 and 4 carry early results (renewed > 0) that the doctrine must wipe.
    """
    return pd.DataFrame({
        "region":            ["EU", "EU", "EU", "EU", "EU", "NA"],
        "softcancel":        [0, 0, 0, 0, 1, 0],
        "channel":           ["web", "web", "web", "web", "web", "tele"],
        "discount":          ["d0", "d40", "d0", "d0", "d0", "d0"],
        "period":            ["2026-01", "2026-01", "2026-02", "2026-03", "2026-03", "2026-04"],
        "dataset_role":      [ROLE_TRAIN, ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION, ROLE_TEST, ROLE_PROJECTION],
        "is_current_month":  [0, 0, 0, 1, 1, 0],
        "flag_time_series":  [0, 0, 0, 0, 0, 1],
        "total_tr_units":    [10.0, 5.0, 8.0, 6.0, 4.0, 3.0],
        "total_tr_usd":      [100.0, 50.0, 80.0, 60.0, 40.0, 30.0],
        "total_renewed_units": [7.0, 2.0, 5.0, 2.0, 1.0, 0.0],
        "total_renewed_usd": [77.0, 22.0, 55.0, 22.0, 11.0, 0.0],
        "total_reacquired_units": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "total_reacquired_usd":   [0.0, 0.0, 9.0, 9.0, 0.0, 0.0],
        "TR_AUV": [10.0] * 6, "REN_AUV": [11.0] * 6, "ReAC_AUV": [9.0] * 6,
        "dummy_field": [0] * 6, "row_id": list(range(6)), "_filter": [1] * 6,
    })


def expect_value_error(callable_under_test, expected_fragment: str, message: str) -> None:
    """Shorthand over the recorder for the blocking checks."""
    RECORDER.check_raises(callable_under_test, expected_fragment, message)


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.1 · validate_raw
# ═══════════════════════════════════════════════════════════════════════════════════

def test_validate_raw() -> None:
    RECORDER.start_block("0.1 · validate_raw")
    configuration = build_test_config()
    raw = build_test_raw()

    # a compliant raw passes and comes back normalized, untouched otherwise
    validated = validate_raw(raw, configuration)
    RECORDER.check(len(validated) == len(raw) and list(validated.columns) == list(raw.columns),
                   "a compliant raw keeps every row and every column")
    RECORDER.check(isinstance(validated["period"].dtype, pd.PeriodDtype)
                   and str(validated["period"].dtype) == "period[M]",
                   "the period is normalized to monthly Period")
    RECORDER.check(not isinstance(raw["period"].dtype, pd.PeriodDtype) and raw.loc[0, "period"] == "2026-01",
                   "the caller's raw is not modified (a copy is returned)")

    # a declared column missing from the raw stops, naming it
    expect_value_error(lambda: validate_raw(raw.drop(columns=["total_tr_usd"]), configuration),
                       "total_tr_usd", "a required column missing from the raw stops the run and is named")

    # the contract: an undeclared column stops
    raw_with_orphan = raw.copy()
    raw_with_orphan["orphan"] = 1
    expect_value_error(lambda: validate_raw(raw_with_orphan, configuration),
                       "orphan", "an undeclared column stops the run and is named")

    # the period: one unparseable value stops (pandas raises)
    raw_with_bad_period = raw.copy()
    raw_with_bad_period.loc[0, "period"] = "not-a-month"
    bad_period_raised = False
    try:
        validate_raw(raw_with_bad_period, configuration)
    except Exception:
        bad_period_raised = True
    RECORDER.check(bad_period_raised, "one unparseable period stops the run")

    # money: negative pipeline stops
    raw_with_negative = raw.copy()
    raw_with_negative.loc[0, "total_tr_usd"] = -1.0
    expect_value_error(lambda: validate_raw(raw_with_negative, configuration),
                       "total_tr_usd", "a negative pipeline USD stops the run and names the column")

    # money: renewing for free stops
    raw_renewing_free = raw.copy()
    raw_renewing_free.loc[0, "total_renewed_usd"] = 0.0
    expect_value_error(lambda: validate_raw(raw_renewing_free, configuration),
                       "USD<=0", "a renewal with USD<=0 stops the run (zero uplift forbidden)")

    # money: zero renewals with zero USD is legal (nobody renewed)
    raw_nobody_renewed = raw.copy()
    raw_nobody_renewed.loc[0, ["total_renewed_units", "total_renewed_usd"]] = [0.0, 0.0]
    nobody_renewed_accepted = True
    try:
        validate_raw(raw_nobody_renewed, configuration)
    except ValueError:
        nobody_renewed_accepted = False
    RECORDER.check(nobody_renewed_accepted, "zero renewed units with zero USD is legal")

    # dimensions: a null stops, with the count per column
    raw_with_null_dimension = raw.copy()
    raw_with_null_dimension.loc[[0, 1], "channel"] = None
    expect_value_error(lambda: validate_raw(raw_with_null_dimension, configuration),
                       "'channel': 2", "a null in a dimension stops the run with the count per column")

    # a null in an extra_revalorizacion is a dimension too
    raw_with_null_extra = raw.copy()
    raw_with_null_extra.loc[0, "discount"] = None
    expect_value_error(lambda: validate_raw(raw_with_null_extra, configuration),
                       "discount", "a null in an extra_revalorizacion stops the run too")

    # every problem is reported at once
    raw_with_two_problems = raw.copy()
    raw_with_two_problems.loc[0, "total_tr_units"] = -1.0
    raw_with_two_problems.loc[1, "region"] = None
    two_problems_message = None
    try:
        validate_raw(raw_with_two_problems, configuration)
    except ValueError as raised_error:
        two_problems_message = str(raised_error)
    RECORDER.check(two_problems_message is not None
                   and "total_tr_units" in two_problems_message and "region" in two_problems_message,
                   "several blocking problems are reported in one message")

    # an empty role is a warning, not a stop
    raw_without_test = raw[raw["dataset_role"] != ROLE_TEST].reset_index(drop=True)
    empty_role_accepted = True
    try:
        validate_raw(raw_without_test, configuration)
    except ValueError:
        empty_role_accepted = False
    RECORDER.check(empty_role_accepted, "an empty role does not stop the run (warning only)")

    # an empty raw passes the contract
    RECORDER.check(len(validate_raw(raw.head(0), configuration)) == 0,
                   "an empty raw passes the contract and returns empty")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.1 · apply_current_month_doctrine
# ═══════════════════════════════════════════════════════════════════════════════════

def test_current_month_doctrine() -> None:
    RECORDER.start_block("0.1 · apply_current_month_doctrine")
    configuration = build_test_config()
    validated = validate_raw(build_test_raw(), configuration)
    conditioned = apply_current_month_doctrine(validated, configuration)

    # the current month becomes projection, whatever it was labeled
    RECORDER.check((conditioned.loc[conditioned["is_current_month"] == 1, "dataset_role"]
                    == ROLE_PROJECTION).all(),
                   "every current-month row is reassigned to projection")
    RECORDER.check(conditioned.loc[4, "dataset_role"] == ROLE_PROJECTION,
                   "a current-month row labeled test is reassigned (never test)")

    # renewals and reacquisitions of the future are wiped
    projection_rows = conditioned[conditioned["dataset_role"] == ROLE_PROJECTION]
    wiped_columns = ["total_renewed_units", "total_renewed_usd",
                     "total_reacquired_units", "total_reacquired_usd"]
    RECORDER.check(projection_rows[wiped_columns].isna().all().all(),
                   "renewed and reacquired measures of every projection row are NaN")
    RECORDER.check(projection_rows[["total_tr_units", "total_tr_usd"]].notna().all().all(),
                   "the pipeline of projection rows is kept (it is known data)")

    # history is untouched
    history_rows = conditioned[conditioned["dataset_role"] != ROLE_PROJECTION]
    original_history = validated.loc[history_rows.index]
    RECORDER.check(history_rows[wiped_columns].equals(original_history[wiped_columns]),
                   "train and test rows keep their results untouched")
    RECORDER.check(len(conditioned) == len(validated), "no row is added or removed")
    RECORDER.check((validated.loc[4, "dataset_role"] == ROLE_TEST)
                   and validated.loc[3, "total_renewed_units"] == 2.0,
                   "the input frame is not modified (a copy is returned)")

    # the flag may arrive as text
    validated_text_flag = validated.copy()
    validated_text_flag["is_current_month"] = validated_text_flag["is_current_month"].map({1: "true", 0: "no"})
    conditioned_text_flag = apply_current_month_doctrine(validated_text_flag, configuration)
    RECORDER.check(conditioned_text_flag.loc[4, "dataset_role"] == ROLE_PROJECTION,
                   "a textual current-month flag ('true') is recognized")

    # with no current month flagged, only pre-labeled projection rows are wiped
    validated_no_flag = validated.copy()
    validated_no_flag["is_current_month"] = 0
    conditioned_no_flag = apply_current_month_doctrine(validated_no_flag, configuration)
    RECORDER.check(conditioned_no_flag.loc[4, "dataset_role"] == ROLE_TEST
                   and np.isnan(conditioned_no_flag.loc[3, "total_renewed_units"]),
                   "with no current month flagged, nothing is reassigned and projection is still wiped")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.2 · build_fine_table
# ═══════════════════════════════════════════════════════════════════════════════════

def test_build_fine_table() -> None:
    RECORDER.start_block("0.2 · build_fine_table")
    configuration = build_test_config()
    conditioned = apply_current_month_doctrine(validate_raw(build_test_raw(), configuration), configuration)
    fine_table = build_fine_table(conditioned, configuration)

    RECORDER.check(len(fine_table) == len(conditioned), "the fine table has one row per raw row")
    RECORDER.check(fine_table.loc[0, "fu_id"] == "EU|0|web|2026-01",
                   "fu_id = rate series columns | month, in order")
    RECORDER.check(fine_table.loc[0, "fu_id"] == fine_table.loc[1, "fu_id"],
                   "two raw rows differing only in the extras share the same fu_id")
    RECORDER.check(fine_table.loc[1, "comb_id"] == "d40",
                   "comb_id = the extra_revalorizacion joined")
    RECORDER.check(fine_table.loc[0, "fu_key"] == hash_key("EU|0|web|2026-01"),
                   "fu_key = hash_key(fu_id)")
    RECORDER.check(fine_table.loc[1, "comb_key"] == hash_key("d40"),
                   "comb_key = hash_key(comb_id)")
    RECORDER.check(fine_table.loc[1, "fu_comb_key"] == hash_key("EU|0|web|2026-01||d40"),
                   "fu_comb_key = hash_key(fu_id || comb_id): one column joins the raw to the bridge")
    RECORDER.check(fine_table["fu_comb_key"].is_unique,
                   "fu_comb_key is unique across the raw rows")
    RECORDER.check(fine_table["total_tr_usd"].equals(conditioned["total_tr_usd"]),
                   "no measure is touched")

    # no extras declared: one single combination
    configuration_without_extras = build_test_config(extra_revalorizacion=[])
    # without the extra, rows 0 and 1 would be the same (unit, combination): keep one of them,
    # as an extract at that grain would (the duplicate assertion is tested right after)
    conditioned_without_extras = conditioned.drop(columns=["discount"]).drop(index=1)
    fine_without_extras = build_fine_table(conditioned_without_extras, configuration_without_extras)
    RECORDER.check((fine_without_extras["comb_id"] == "na").all(),
                   "with no extra_revalorizacion, comb_id is 'na' for every row")

    # two raw rows for the same (unit, combination) stop the run: the extract is not at grain
    duplicate_grain = conditioned.drop(columns=["discount"])
    duplicate_caught = False
    try:
        build_fine_table(duplicate_grain, configuration_without_extras)
    except AssertionError as raised_error:
        duplicate_caught = "same forecast unit AND combination" in str(raised_error)
    RECORDER.check(duplicate_caught, "two raw rows for the same (unit, combination) stop the run")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.2 · aggregate_to_forecast_units
# ═══════════════════════════════════════════════════════════════════════════════════

def test_aggregate_to_forecast_units() -> None:
    RECORDER.start_block("0.2 · aggregate_to_forecast_units")
    configuration = build_test_config()
    conditioned = apply_current_month_doctrine(validate_raw(build_test_raw(), configuration), configuration)
    fine_table = build_fine_table(conditioned, configuration)
    forecast_units = aggregate_to_forecast_units(fine_table, configuration)

    RECORDER.check(len(forecast_units) == 5,
                   "six raw rows become five forecast units (rows 0 and 1 merge)")
    merged_unit = forecast_units[forecast_units["fu_id"] == "EU|0|web|2026-01"].iloc[0]
    RECORDER.check(merged_unit["total_tr_units"] == 15.0 and merged_unit["total_tr_usd"] == 150.0,
                   "measures are SUMMED across the combinations of a unit (10+5, 100+50)")
    RECORDER.check(merged_unit["total_renewed_units"] == 9.0 and merged_unit["total_renewed_usd"] == 99.0,
                   "renewed measures are summed too (7+2, 77+22)")
    RECORDER.check(forecast_units["fu_id"].is_unique, "fu_id is unique in the forecast units table")
    RECORDER.check(abs(forecast_units["total_tr_usd"].sum() - fine_table["total_tr_usd"].sum()) < 1e-9,
                   "pipeline USD is conserved to the cent between the two tables")

    # min_count=1: an all-NaN group stays NaN instead of becoming 0
    projection_unit = forecast_units[forecast_units["fu_id"] == "NA|0|tele|2026-04"].iloc[0]
    RECORDER.check(np.isnan(projection_unit["total_renewed_units"]),
                   "a unit whose renewed measures were wiped stays NaN (min_count=1), never 0")

    # the extras are not in the table: it is the rate-branch table
    RECORDER.check("discount" not in forecast_units.columns,
                   "the extra_revalorizacion columns do not enter the forecast units table")
    RECORDER.check(set(["dataset_role", "is_current_month", "flag_time_series"]).issubset(forecast_units.columns),
                   "role and flags travel with the unit")

    # keys match the fine table
    fine_keys = set(fine_table["fu_key"])
    RECORDER.check(set(forecast_units["fu_key"]) == fine_keys,
                   "the set of fu_key is the same in both tables")

    # an inconsistent extract (role varying inside a unit) is caught by uniqueness
    inconsistent_fine = fine_table.copy()
    inconsistent_fine.loc[1, "dataset_role"] = ROLE_PROJECTION
    inconsistency_caught = False
    try:
        aggregate_to_forecast_units(inconsistent_fine, configuration)
    except AssertionError as raised_error:
        inconsistency_caught = "duplicated fu_id" in str(raised_error)
    RECORDER.check(inconsistency_caught,
                   "a role varying inside a forecast unit stops the run (duplicated fu_id)")

    # an empty fine table is legal
    RECORDER.check(len(aggregate_to_forecast_units(fine_table.head(0), configuration)) == 0,
                   "an empty fine table yields an empty forecast units table")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.2 · build_key_lookups
# ═══════════════════════════════════════════════════════════════════════════════════

def test_build_key_lookups() -> None:
    RECORDER.start_block("0.2 · build_key_lookups")
    configuration = build_test_config()
    conditioned = apply_current_month_doctrine(validate_raw(build_test_raw(), configuration), configuration)
    fine_table = build_fine_table(conditioned, configuration)
    forecast_units = aggregate_to_forecast_units(fine_table, configuration)
    fu_lookup, comb_lookup = build_key_lookups(forecast_units, fine_table)

    RECORDER.check(len(fu_lookup) == 5 and fu_lookup["fu_id"].is_unique,
                   "fu_lookup has one row per forecast unit")
    RECORDER.check(len(comb_lookup) == 2 and set(comb_lookup["comb_id"]) == {"d0", "d40"},
                   "comb_lookup has one row per revaluation combination")
    RECORDER.check((fu_lookup["fu_key"] == fu_lookup["fu_id"].map(hash_key)).all()
                   and (comb_lookup["comb_key"] == comb_lookup["comb_id"].map(hash_key)).all(),
                   "every lookup row satisfies key = hash_key(id)")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.3 · route_from_coverage and label_universe_and_routes
# ═══════════════════════════════════════════════════════════════════════════════════

def test_routes_and_labels() -> None:
    RECORDER.start_block("0.3 · route_from_coverage")
    RECORDER.check(route_from_coverage(pattern(ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN)) == TREATMENT_PREDICTABLE,
                   "history + future → trainable")
    RECORDER.check(route_from_coverage(pattern(ROLE_PROJECTION, ROLE_TRAIN)) == TREATMENT_PREDICTABLE,
                   "train + projection (no test) → trainable")
    RECORDER.check(route_from_coverage(ROLE_PROJECTION) == TREATMENT_FUTURE_ONLY,
                   "future without history → heuristic")
    RECORDER.check(route_from_coverage(pattern(ROLE_PROJECTION, ROLE_TEST)) == TREATMENT_PREDICTABLE,
                   "test rows ARE closed months: test + projection → trainable (the forecast learns from them)")
    RECORDER.check(route_from_coverage(pattern(ROLE_PENDING, ROLE_PROJECTION)) == TREATMENT_FUTURE_ONLY,
                   "a pending month is not closed: pending + projection without train/test → heuristic")
    RECORDER.check(route_from_coverage(pattern(ROLE_PENDING, ROLE_TRAIN)) == TREATMENT_PREDICTABLE,
                   "a pending month is something to predict: train + pending → trainable")
    RECORDER.check(route_from_coverage(ROLE_TRAIN) == TREATMENT_HISTORY_ONLY
                   and route_from_coverage(pattern(ROLE_TEST, ROLE_TRAIN)) == TREATMENT_HISTORY_ONLY,
                   "no projection rows → no_impact")

    RECORDER.start_block("0.3 · label_universe_and_routes")
    configuration = build_test_config()
    conditioned = apply_current_month_doctrine(validate_raw(build_test_raw(), configuration), configuration)
    forecast_units = aggregate_to_forecast_units(build_fine_table(conditioned, configuration), configuration)
    labeled = label_universe_and_routes(forecast_units, configuration)

    RECORDER.check(len(labeled) == len(forecast_units), "no unit is dropped (label, never amputate)")
    RECORDER.check(set(["universo", "fs_id", "cobertura", "ruta"]).issubset(labeled.columns),
                   "the four label columns are added")
    by_id = labeled.set_index("fu_id")
    RECORDER.check(by_id.loc["EU|0|web|2026-01", "fs_id"] == "EU|0|web",
                   "fs_id = the rate series columns joined (no month)")
    RECORDER.check(by_id.loc["EU|0|web|2026-01", "cobertura"] == pattern(ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN),
                   "coverage = the sorted roles of the series joined with '_'")
    RECORDER.check(by_id.loc["EU|0|web|2026-01", "ruta"] == TREATMENT_PREDICTABLE,
                   "a series with history and future is trainable")
    RECORDER.check(by_id.loc["EU|1|web|2026-03", "ruta"] == TREATMENT_FUTURE_ONLY,
                   "a series present only in the (reassigned) current month is heuristic")
    RECORDER.check(by_id.loc["NA|0|tele|2026-04", "universo"] == UNIVERSE_TIME_SERIES
                   and by_id.loc["EU|0|web|2026-01", "universo"] == "normal",
                   "universo comes from the time-series flag")
    RECORDER.check((labeled.groupby("fs_id")["cobertura"].nunique() == 1).all(),
                   "every unit of a series carries the same coverage and route")


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.4 · build_support_reference (ANALYSIS)
# ═══════════════════════════════════════════════════════════════════════════════════

def test_support_reference() -> None:
    RECORDER.start_block("0.4 · build_support_reference (analysis)")
    with tempfile.TemporaryDirectory() as temporary_directory:
        sqlite_engine = create_engine(f"sqlite:///{os.path.join(temporary_directory, 'ref.db')}")
        configuration = build_test_config(sql_engine=sqlite_engine, sql_schema=None,
                                          outdir=temporary_directory)
        conditioned = apply_current_month_doctrine(validate_raw(build_test_raw(), configuration), configuration)
        forecast_units = aggregate_to_forecast_units(build_fine_table(conditioned, configuration), configuration)
        labeled = label_universe_and_routes(forecast_units, configuration)
        reference = build_support_reference(labeled, configuration)

        RECORDER.check(len(reference) == len(labeled), "one reference row per forecast unit")
        by_id = reference.set_index("fu_id")
        # n = 15 → se = 100·√(0.25/15) = 12.9099 pp · moe = 1.645·se = 21.2368 pp · $ = 0.212368·150
        unit_row = by_id.loc["EU|0|web|2026-01"]
        RECORDER.check(abs(unit_row["se_pp_max"] - 100 * np.sqrt(0.25 / 15)) < 1e-9,
                       "se_pp_max = 100·√(0.25/n) at p=0.5 (n=15 → 12.91 pp)")
        RECORDER.check(abs(unit_row["moe_pp_max"] - configuration.z * unit_row["se_pp_max"]) < 1e-9,
                       "moe_pp_max = z · se_pp_max")
        RECORDER.check(abs(unit_row["moe_usd_max"] - unit_row["moe_pp_max"] / 100 * 150.0) < 1e-9,
                       "moe_usd_max = moe_pp_max / 100 · pipeline USD of the unit")
        RECORDER.check(not isinstance(reference["period"].dtype, pd.PeriodDtype)
                       and unit_row["period"] == "2026-01",
                       "the period is persisted as text")
        RECORDER.check({"process_date", "execution_id"}.issubset(reference.columns),
                       "the written frame carries the traceability columns")

        persisted = pd.read_sql("SELECT COUNT(*) AS n FROM sff_fu_summary", sqlite_engine)
        RECORDER.check(int(persisted["n"].iloc[0]) == len(labeled),
                       "the reference lands in sff_fu_summary")

        # the bound of a unit with no pipeline: support clipped to 1
        labeled_zero = labeled.copy()
        labeled_zero.loc[0, "total_tr_units"] = 0.0
        reference_zero = build_support_reference(labeled_zero, configuration)
        RECORDER.check(abs(reference_zero.loc[0, "se_pp_max"] - 50.0) < 1e-9,
                       "n=0 is clipped to n=1: se = 50 pp (the bound of 'no evidence')")


# ═══════════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [test_validate_raw,
             test_current_month_doctrine,
             test_build_fine_table,
             test_aggregate_to_forecast_units,
             test_build_key_lookups,
             test_routes_and_labels,
             test_support_reference]


def main() -> int:
    print("═" * 74)
    print("TEST phase 0 · raw_data_validation (RUN) + support_reference (ANALYSIS)")
    print("═" * 74)
    for test_function in ALL_TESTS:
        test_function()
    return RECORDER.print_panel("PHASE 0 TEST")


if __name__ == "__main__":
    sys.exit(main())
