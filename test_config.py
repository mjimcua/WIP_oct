"""test_config.py — regression checks for `config.py` (SFF v2 · RUN).

WHY THIS FILE EXISTS. The equivalence gate proves that the framework AS A WHOLE still
produces the reference numbers. It does not prove that a bad taxonomy is rejected, that an
id is built in the contractual field order, or that a business parameter still holds its
sealed value: the error paths are never exercised by a healthy run, and a changed
parameter would surface as a confusing difference ten functions downstream. This file
checks `config.py` directly, so that a regression fails HERE, with a sentence naming
what broke, before it reaches the gate.

HOW TO USE IT. Run it after every change to `config.py`, before the gate:

    python test_config.py     →  exit 0 if healthy, 1 with the list of failures
    python test_pipeline.py   →  the system-level run on the v3 synthetic (logic checks)

The two are complementary: this one is unit-level and fast, the gate is system-level and
slow. A change that passes this file and fails the gate is a change in a PHASE; a change
that fails this file is a change in the CONTRACT.

WHAT IS PINNED. Block 5 freezes the production defaults (the ten mandatory dimensions,
the sealed business parameters, the physical table names). Those checks fail on purpose
whenever a value changes: the failure is the reminder to declare the change in
ASUNCIONES and to regenerate the reference, never a reason to edit the expected value
without that declaration.

No external test runner is required, on purpose: it runs like `smoke_multidim.py`.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine

# every module lives in this folder; make sure it is importable (a notebook may run
# from elsewhere). __file__ is undefined in a notebook, so fall back to the cwd.
# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import (Config, PHYSICAL_TABLE_NAMES, hash_key,
                    join_columns)
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)


# ─── named constants ─────────────────────────────────────────────────────────────
# A minimal taxonomy that exercises all four groups without dragging in the production
# column names: two mandatory, one timevarying, one extra of each kind.
TEST_MANDATORY_DIMS = ["region", "product"]
TEST_TIMEVARYING_DIMS = {"softcancel": "negative"}
TEST_EXTRA_RENOVACION = ["channel"]
TEST_EXTRA_REVALORIZACION = ["discount"]

# The key of a fixed id, pinned so a change in the hashing rule (digest, number of hex
# digits, encoding) fails here instead of silently invalidating every persisted key.
# Value of hash_key("EU|A|web|2026-01") = int(md5(id).hexdigest()[:12], 16).
EXPECTED_KEY_OF_SAMPLE_ID = 0xa7e319b2e18e

# The production taxonomy and the sealed business parameters (block 5). A change in any
# of these is a change of contract: it must be declared in ASUNCIONES and the reference
# must be regenerated.
EXPECTED_PRODUCTION_MANDATORY_DIMS = [
    "tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
    "tr_product_level_1", "tr_product_level_2", "tr_origin_type_SKU_based",
    "tr_term_level_1", "tr_term_level_2", "tr_band_level_1", "tr_band_level_2"]
EXPECTED_PRODUCTION_TIMEVARYING_DIMS = {"softcancel": "negative"}
EXPECTED_PRODUCTION_EXTRA_REVALORIZACION = [
    "price_cap", "msrp_increased", "discount_interval", "prev_OperationGroup"]
EXPECTED_BUSINESS_PARAMETERS = {"support_floor": 30.0, "z": 1.645, "rate_cap": 0.95,
                                "k_cred": 60.0, "uplift_floor": 30.0, "uplift_cap": 3.0,
                                "backtest_max_targets": 6}

# Physical names the reference and the BI already depend on. Sampled, not exhaustive: the
# full registry is checked for size and for prefixing.
EXPECTED_PHYSICAL_NAMES = {"forecast_units_raw_summary": "sff_fu_summary",
                           "forecast_series_raw_summary": "sff_fs_summary",
                           "decision_dynamics": "sff_decision_dynamics",
                           "decision_technique": "sff_decision_technique",
                           "forecast_detail": "sff_forecast_detail",
                           "fact_fine": "sff_fact_fine"}

# Size of the physical-name registry: the 36 tables of v3 (phase 0: 15, with the level-0/1
# profiles and the panorama · phase 1: 20 · phase 2: 4 · phase 3: 8 · phase 4: 5 · phase 5: 21).
EXPECTED_REGISTRY_SIZE = 76
DECISION_TABLES = ["decision_eta2", "decision_support", "decision_estacionalidad", "pool_reference", "decision_technique",
                   "decision_error_bands", "decision_uplift", "decision_aggregate_bands"]


RECORDER = CheckRecorder()


# ─── fixtures ────────────────────────────────────────────────────────────────────

def build_test_config(**overrides) -> Config:
    """Build a Config on the small test taxonomy, with `overrides` applied on top."""
    config_arguments = dict(business_mandatory_dims=list(TEST_MANDATORY_DIMS),
                            structural_timevarying_dims=dict(TEST_TIMEVARYING_DIMS),
                            extra_renovacion=list(TEST_EXTRA_RENOVACION),
                            extra_revalorizacion=list(TEST_EXTRA_REVALORIZACION))
    config_arguments.update(overrides)
    return Config(**config_arguments)


def build_test_raw(configuration: Config) -> pd.DataFrame:
    """Build a two-row raw whose columns are exactly the declared ones.

    INPUT:   configuration — the Config whose contract the frame must satisfy.
    OUTPUT:  a DataFrame with one column per declared role and two rows of plausible data.
    RULES:   this frame must pass `validate_column_contract` untouched; a test that needs
             a violation adds an extra column to a copy.
    EDGE CASES: the declared measures (reacquisitions, AUVs) are included, so the
             contract is exercised on its full surface.
    CONSOLE: nothing.
    STEPS:
      [1] Context columns.
      [2] Core and declared measures.
      [3] One value per dimension of the taxonomy.
    """
    # [1] context
    row_values = {configuration.period_col: ["2026-01", "2026-02"],
                  configuration.dataset_role_col: [ROLE_TRAIN, ROLE_PROJECTION],
                  configuration.current_month_col: [0, 1],
                  configuration.flag_time_series_col: [0, 0]}

    # [2] measures: the four core ones plus everything declared but not computed with
    for measure_column in configuration.core_measures:
        row_values[measure_column] = [100.0, 200.0]
    for measure_column in configuration.declared_measures:
        row_values[measure_column] = [1.0, 2.0]

    # [3] dimensions: rate series columns plus the revaluation extras
    dimension_columns = configuration.rate_series_columns + configuration.extra_revalorizacion
    for dimension_column in dimension_columns:
        row_values[dimension_column] = ["a", "b"]

    return pd.DataFrame(row_values)


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 1 · TAXONOMY AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════

def test_taxonomy_exclusions() -> None:
    """A misdeclared taxonomy must never reach a phase: it stops at construction."""
    RECORDER.start_block("BLOCK 1 · taxonomy exclusions")

    # a well-formed taxonomy builds and mints an execution id
    valid_configuration = build_test_config()
    RECORDER.check(len(valid_configuration.execution_id) == 10,
                   "a valid taxonomy builds and mints a 10-char execution_id")

    # two Configs are two executions: their ids must differ, or two runs would be
    # indistinguishable in the persisted tables
    second_configuration = build_test_config()
    RECORDER.check(valid_configuration.execution_id != second_configuration.execution_id,
                   "two Configs mint different execution_ids")

    # an invalid sign cannot be grouped in L1
    RECORDER.check_raises(
        lambda: build_test_config(structural_timevarying_dims={"softcancel": SIGN_NEGATIVE}),
        "softcancel", "a timevarying sign outside negative/positive is rejected")
    RECORDER.check_raises(
        lambda: build_test_config(structural_timevarying_dims={"softcancel": ""}),
        "softcancel", "an empty timevarying sign is rejected")

    # mandatory is exclusive with every other group
    RECORDER.check_raises(lambda: build_test_config(extra_renovacion=["region"]),
                          "region", "mandatory overlapping extra_renovacion is rejected")
    RECORDER.check_raises(lambda: build_test_config(extra_revalorizacion=["product"]),
                          "product", "mandatory overlapping extra_revalorizacion is rejected")
    RECORDER.check_raises(
        lambda: build_test_config(structural_timevarying_dims={"region": "negative"}),
        "region", "mandatory overlapping timevarying is rejected")

    # timevarying is exclusive with both extra groups
    RECORDER.check_raises(lambda: build_test_config(extra_renovacion=["softcancel"]),
                          "softcancel", "timevarying overlapping extra_renovacion is rejected")
    RECORDER.check_raises(lambda: build_test_config(extra_revalorizacion=["softcancel"]),
                          "softcancel", "timevarying overlapping extra_revalorizacion is rejected")

    # the two extra groups MAY overlap: the same column can serve both branches
    overlapping_extras_configuration = build_test_config(extra_renovacion=["discount"],
                                                         extra_revalorizacion=["discount"])
    enters_series_and_cell = ("discount" in overlapping_extras_configuration.rate_series_columns
                          and "discount" in overlapping_extras_configuration.uplift_cell_columns)
    RECORDER.check(enters_series_and_cell,
                   "the two extra groups may overlap: the column enters both the rate series and the uplift cell")

    # several offenders are reported together, not one per run
    RECORDER.check_raises(
        lambda: build_test_config(extra_renovacion=["region", "product"]),
        "product", "several overlapping columns are all named in one message")

    # a taxonomy with no timevarying and no extras is legal (mandatory alone)
    minimal_configuration = build_test_config(structural_timevarying_dims={},
                                              extra_renovacion=[], extra_revalorizacion=[])
    RECORDER.check(minimal_configuration.rate_series_columns == TEST_MANDATORY_DIMS,
                   "a taxonomy of mandatory dimensions alone is legal")


def test_read_raw_is_overridable() -> None:
    """The raw source is the caller's: the base Config refuses, a subclass provides."""
    RECORDER.start_block("RAW SOURCE · read_raw must be overridden")

    # the base Config has no source, on purpose
    base_configuration = build_test_config()
    base_refused = False
    try:
        base_configuration.read_raw()
    except NotImplementedError as raised_error:
        base_refused = "overridden" in str(raised_error)
    RECORDER.check(base_refused,
                   "Config.read_raw() raises NotImplementedError telling to override it")

    # a subclass that returns a frame is all a main needs
    class FrameConfig(Config):
        def read_raw(self) -> pd.DataFrame:
            return pd.DataFrame({"a": [1, 2]})

    subclass_configuration = FrameConfig(**dict(business_mandatory_dims=list(TEST_MANDATORY_DIMS),
                                                structural_timevarying_dims=dict(TEST_TIMEVARYING_DIMS),
                                                extra_renovacion=list(TEST_EXTRA_RENOVACION),
                                                extra_revalorizacion=list(TEST_EXTRA_REVALORIZACION)))
    RECORDER.check(len(subclass_configuration.read_raw()) == 2,
                   "a subclass overriding read_raw returns its own frame")
    RECORDER.check(subclass_configuration.rate_series_columns == ["region", "product", "softcancel", "channel"],
                   "the subclass inherits the taxonomy, the properties and the validation")


def test_column_contract() -> None:
    """The exhaustive contract (P3): a column with no role stops the run."""
    RECORDER.start_block("BLOCK 1 · exhaustive column contract")

    configuration = build_test_config()
    compliant_raw = build_test_raw(configuration)

    # a raw made of declared columns only passes silently
    contract_accepted = True
    try:
        configuration.validate_column_contract(compliant_raw)
    except ValueError:
        contract_accepted = False
    RECORDER.check(contract_accepted, "a raw with every column declared passes the contract")

    # an undeclared column is an orphan, named in the exception
    raw_with_one_orphan = compliant_raw.copy()
    raw_with_one_orphan["undeclared_column"] = 1
    RECORDER.check_raises(lambda: configuration.validate_column_contract(raw_with_one_orphan),
                          "undeclared_column",
                          "an undeclared column stops the run and is named")

    # every orphan is reported at once: a second run should not find a second surprise
    raw_with_two_orphans = raw_with_one_orphan.copy()
    raw_with_two_orphans["another_orphan"] = 2
    RECORDER.check_raises(lambda: configuration.validate_column_contract(raw_with_two_orphans),
                          "another_orphan",
                          "every orphan column is listed in the same message")

    # the same column declared as ignorable is accepted
    configuration_with_ignore = build_test_config(ignore_cols=["undeclared_column"])
    ignore_accepted = True
    try:
        configuration_with_ignore.validate_column_contract(raw_with_one_orphan)
    except ValueError:
        ignore_accepted = False
    RECORDER.check(ignore_accepted, "the same column listed in ignore_cols is accepted")

    # a declared column MISSING from the raw is not this function's job: the phases fail
    # on first access. Pinned so the behaviour is a decision, not an accident.
    raw_without_a_measure = compliant_raw.drop(columns=[configuration.renewed_usd_col])
    missing_column_accepted = True
    try:
        configuration.validate_column_contract(raw_without_a_measure)
    except ValueError:
        missing_column_accepted = False
    RECORDER.check(missing_column_accepted,
                   "a declared column missing from the raw does NOT stop the contract "
                   "(by design: the phases fail on first access)")

    # the contract reaches every group of the taxonomy
    declared_dimension_columns = set(configuration.rate_series_columns
                                     + configuration.extra_revalorizacion)
    every_group_declared = all(column_name in declared_dimension_columns
                               for column_name in (TEST_MANDATORY_DIMS
                                                   + list(TEST_TIMEVARYING_DIMS)
                                                   + TEST_EXTRA_RENOVACION
                                                   + TEST_EXTRA_REVALORIZACION))
    RECORDER.check(every_group_declared,
                   "the contract covers mandatory, timevarying and both extra groups")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 2 · GRAIN PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════════

def test_series_and_cell_columns() -> None:
    """Grains are derived from the taxonomy by formula, in a contractual order."""
    RECORDER.start_block("BLOCK 2 · series and cell columns")

    configuration = build_test_config()

    expected_rate_series_columns = ["region", "product", "softcancel", "channel"]
    RECORDER.check(configuration.rate_series_columns == expected_rate_series_columns,
                   "rate series columns = mandatory + timevarying + extra_renovacion, in that order")

    expected_uplift_cell_columns = ["region", "product", "discount"]
    RECORDER.check(configuration.uplift_cell_columns == expected_uplift_cell_columns,
                   "uplift cell columns = mandatory + extra_revalorizacion, in that order")

    # the mandatory dimensions open both the series and the cell: the pools of both branches share them
    mandatory_count = len(TEST_MANDATORY_DIMS)
    RECORDER.check(configuration.rate_series_columns[:mandatory_count] == TEST_MANDATORY_DIMS
                   and configuration.uplift_cell_columns[:mandatory_count] == TEST_MANDATORY_DIMS,
                   "the mandatory dimensions come first in both column lists")

    # no column list repeats a column: a duplicate would double a field in every id
    RECORDER.check(len(set(configuration.rate_series_columns)) == len(configuration.rate_series_columns),
                   "the rate series columns have no repeated column")
    RECORDER.check(len(set(configuration.uplift_cell_columns)) == len(configuration.uplift_cell_columns),
                   "the uplift cell columns have no repeated column")

    expected_core_measures = [configuration.pipeline_units_col, configuration.pipeline_usd_col,
                              configuration.renewed_units_col, configuration.renewed_usd_col]
    RECORDER.check(configuration.core_measures == expected_core_measures,
                   "core measures are pipeline and renewed, units and USD, in that order")

    RECORDER.check(configuration.reacq_units_col in configuration.declared_measures
                   and configuration.auv_pipeline_col in configuration.declared_measures,
                   "declared measures carry reacquisitions and AUVs (declared, never computed with)")

    # no measure is both core and declared: that would double-count it in the contract
    RECORDER.check(not set(configuration.core_measures) & set(configuration.declared_measures),
                   "core and declared measures do not overlap")

    # a measure configured as empty is simply not expected in the raw
    configuration_without_auv = build_test_config(auv_pipeline_col="", auv_renewed_col="",
                                                  auv_reacq_col="")
    RECORDER.check(all(column_name for column_name in configuration_without_auv.declared_measures),
                   "a measure configured as empty drops out of the declared measures")

    # the column lists are derived, never stored: changing the taxonomy changes it
    configuration_without_extras = build_test_config(extra_renovacion=[])
    RECORDER.check(configuration_without_extras.rate_series_columns == ["region", "product", "softcancel"],
                   "dropping extra_renovacion drops it from the rate series columns")
    configuration_with_two_timevarying = build_test_config(
        structural_timevarying_dims={"softcancel": "negative", "autorenew": "positive"})
    RECORDER.check(configuration_with_two_timevarying.rate_series_columns
                   == ["region", "product", "softcancel", "autorenew", "channel"],
                   "a second timevarying enters the rate series columns in declaration order")
    RECORDER.check("autorenew" not in configuration_with_two_timevarying.uplift_cell_columns,
                   "timevarying dimensions never enter the uplift cell columns")

    # the legacy Spanish aliases are gone: nothing in v3 answers to them
    RECORDER.check(not any(hasattr(configuration, alias) for alias in ("grano_tasa", "grano_uplift", "medidas", "validar_columnas")),
                   "the legacy aliases (grano_tasa, grano_uplift, medidas, validar_columnas) no longer exist")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 3 · KEYS AND IDS
# ═══════════════════════════════════════════════════════════════════════════════════

def test_join_columns() -> None:
    """Ids are '|'-joined in the field order given; the order is contractual."""
    RECORDER.start_block("BLOCK 3 · id construction (join_columns)")

    sample_frame = pd.DataFrame({"region": ["EU", "NA"],
                                 "product": ["A", "B"],
                                 "channel": ["web", "tele"]})

    joined_ids = join_columns(sample_frame, ["region", "product", "channel"])
    RECORDER.check(list(joined_ids) == ["EU|A|web", "NA|B|tele"],
                   "join_columns joins the fields with '|' in the order given")

    reordered_ids = join_columns(sample_frame, ["product", "region", "channel"])
    RECORDER.check(list(reordered_ids) != list(joined_ids),
                   "a different field order produces a different id (the order is contractual)")

    single_column_ids = join_columns(sample_frame, ["region"])
    RECORDER.check(list(single_column_ids) == ["EU", "NA"],
                   "a one-column id carries no separator")

    # non-text fields are cast, never formatted: the id must be reproducible
    mixed_types_frame = pd.DataFrame({"flag": [1, 0], "value": [2.5, 3.0]})
    mixed_ids = join_columns(mixed_types_frame, ["flag", "value"])
    RECORDER.check(list(mixed_ids) == ["1|2.5", "0|3.0"],
                   "non-text fields are cast with astype(str)")

    # A null in a dimension does NOT become text: `astype(str)` keeps it as NaN, so a
    # single-column id carries a NaN and a multi-column id raises TypeError when the
    # separator is concatenated. Neither outcome is a usable id — which is exactly why
    # phase 0 blocks nulls in dimensions BEFORE any id is built. Pinned here so a change
    # of pandas behaviour (text 'nan' instead) is noticed: that variant would produce a
    # legal-looking id for a broken row.
    frame_with_null = pd.DataFrame({"region": ["EU", None]})
    single_column_ids_with_null = join_columns(frame_with_null, ["region"])
    RECORDER.check(pd.isna(list(single_column_ids_with_null)[1]),
                   "a null dimension stays NaN in a one-column id, never becomes text")

    frame_with_null_and_second_column = pd.DataFrame({"region": ["EU", None],
                                                      "product": ["A", "B"]})
    multi_column_join_raised_type_error = False
    try:
        join_columns(frame_with_null_and_second_column, ["region", "product"])
    except TypeError:
        multi_column_join_raised_type_error = True
    RECORDER.check(multi_column_join_raised_type_error,
                   "a null dimension breaks a multi-column id (why phase 0 blocks nulls)")

    # duplicate index labels (the state of a frame after a merge) must not misalign
    duplicated_index_frame = sample_frame.copy()
    duplicated_index_frame.index = [0, 0]
    ids_on_duplicated_index = join_columns(duplicated_index_frame, ["region", "product"])
    RECORDER.check(list(ids_on_duplicated_index) == ["EU|A", "NA|B"],
                   "join_columns survives duplicate index labels (it works on arrays)")

    # a non-default index must be preserved, or a later assignment misaligns
    reindexed_frame = sample_frame.copy()
    reindexed_frame.index = [10, 20]
    ids_on_reindexed = join_columns(reindexed_frame, ["region"])
    RECORDER.check(list(ids_on_reindexed.index) == [10, 20],
                   "the returned Series keeps the caller's index")

    # an empty frame yields an empty Series instead of raising
    empty_frame = sample_frame.head(0)
    ids_on_empty = join_columns(empty_frame, ["region", "product"])
    RECORDER.check(len(ids_on_empty) == 0, "an empty frame yields an empty Series")


def test_hash_key() -> None:
    """Keys are a deterministic function of the id and fit a SQL bigint."""
    RECORDER.start_block("BLOCK 3 · key derivation (hash_key)")

    first_key = hash_key("EU|A|web|2026-01")
    second_key = hash_key("EU|A|web|2026-01")
    RECORDER.check(first_key == second_key,
                   "hash_key is stable: the same id yields the same key within a run")
    RECORDER.check(first_key == EXPECTED_KEY_OF_SAMPLE_ID,
                   "hash_key matches its pinned value (the hashing rule has not changed)")
    RECORDER.check(first_key != hash_key("EU|A|web|2026-02"),
                   "a different id yields a different key")
    RECORDER.check(0 < first_key < 2 ** 48,
                   "the key holds in 48 bits, so it fits a SQL bigint")
    RECORDER.check(isinstance(first_key, int), "the key is an int, never a string")

    # the id is cast with str(): an int id and its text form are the same key
    RECORDER.check(hash_key(42) == hash_key("42"),
                   "a non-text id is cast with str() before hashing")

    # keys of ids differing in one character must not collide in practice
    distinct_ids = [f"EU|A|web|2026-{month:02d}" for month in range(1, 13)]
    distinct_keys = {hash_key(one_id) for one_id in distinct_ids}
    RECORDER.check(len(distinct_keys) == len(distinct_ids),
                   "twelve near-identical ids yield twelve distinct keys")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 4 · SQL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════════

def test_physical_table_names() -> None:
    """Logical name → physical name: the registry, the prefix and the overrides."""
    RECORDER.start_block("BLOCK 4 · physical table names")

    configuration = build_test_config()
    RECORDER.check(
        configuration._resolve_physical_table_name("forecast_units_raw_summary") == "sff_fu_summary",
        "a logical name in the registry maps to prefix + registry suffix")
    RECORDER.check(configuration._resolve_physical_table_name("fact_fine") == "sff_fact_fine",
                   "a logical name that equals its suffix keeps its own name, prefixed")
    RECORDER.check(configuration._resolve_physical_table_name("unknown_table") == "sff_unknown_table",
                   "a logical name absent from the registry keeps its own name, prefixed")

    overriding_configuration = build_test_config(sql_table_names={"fact_fine": "my_own_table"})
    RECORDER.check(overriding_configuration._resolve_physical_table_name("fact_fine") == "my_own_table",
                   "an explicit override in sql_table_names wins over registry and prefix")

    prefixed_configuration = build_test_config(sql_table_prefix="prod_")
    RECORDER.check(prefixed_configuration._resolve_physical_table_name("fact_fu") == "prod_fact_fu",
                   "the prefix is configurable")

    # the schema is a config datum: qualification changes with it, and sqlite needs none
    schema_configuration = build_test_config(sql_schema="dbo")
    RECORDER.check(schema_configuration._qualified_table_name("sff_fact_fu") == "[dbo].[sff_fact_fu]",
                   "with a schema, the table is qualified as [schema].[table]")
    schemaless_configuration = build_test_config(sql_schema=None)
    RECORDER.check(schemaless_configuration._qualified_table_name("sff_fact_fu") == "[sff_fact_fu]",
                   "without a schema, the table is qualified as [table]")


def test_write_to_sql() -> None:
    """`write` persists under the physical name, stamped, honouring the write mode."""
    RECORDER.start_block("BLOCK 4 · write to SQL")

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = os.path.join(temporary_directory, "test_config.db")
        sqlite_engine = create_engine(f"sqlite:///{database_path}")
        configuration = build_test_config(sql_engine=sqlite_engine, sql_schema=None,
                                          outdir=temporary_directory)

        # a frame with a Period column, the case `_sanitize_for_persistence` exists for
        frame_to_write = pd.DataFrame({
            "fu_id": ["EU|A|2026-01", "NA|B|2026-01"],
            "period": pd.PeriodIndex(["2026-01", "2026-01"], freq="M"),
            "tasa": [0.80, 0.50]})

        written_frame = configuration.write(frame_to_write, "fact_fu")
        persisted_frame = pd.read_sql("SELECT * FROM sff_fact_fu", sqlite_engine)

        RECORDER.check(len(persisted_frame) == 2, "write persists every row of the frame")
        RECORDER.check("process_date" in persisted_frame.columns
                       and "execution_id" in persisted_frame.columns,
                       "write stamps process_date and execution_id on every row")
        RECORDER.check(set(persisted_frame["execution_id"]) == {configuration.execution_id},
                       "the stamped execution_id is the one of this Config")
        RECORDER.check(persisted_frame["tasa"].sum() == 1.30,
                       "the numbers survive the round trip untouched")
        RECORDER.check(persisted_frame["period"].tolist() == ["2026-01", "2026-01"]
                       and "period_date" in persisted_frame.columns,
                       "a Period column becomes text plus a `_date` companion")
        RECORDER.check("period" in frame_to_write.columns
                       and isinstance(frame_to_write["period"].dtype, pd.PeriodDtype),
                       "the caller's frame is never modified by write")
        RECORDER.check(written_frame is not frame_to_write and len(written_frame) == 2,
                       "write returns the sanitized frame, not the original")

        # replace mode: a second write leaves the table with the new content only
        configuration.write(frame_to_write.head(1), "fact_fu")
        after_replace = pd.read_sql("SELECT * FROM sff_fact_fu", sqlite_engine)
        RECORDER.check(len(after_replace) == 1,
                       "replace mode leaves only the rows of the last write")

        # truncate mode on an existing table: content replaced, definition kept
        truncating_configuration = build_test_config(sql_engine=sqlite_engine, sql_schema=None,
                                                     sql_write_mode="truncate",
                                                     outdir=temporary_directory)
        truncating_configuration.write(frame_to_write, "fact_fu")
        after_truncate = pd.read_sql("SELECT * FROM sff_fact_fu", sqlite_engine)
        RECORDER.check(len(after_truncate) == 2,
                       "truncate mode empties the table and appends the new rows")

        # truncate mode on a table that does not exist yet: first write creates it
        truncating_configuration.write(frame_to_write, "fact_fu_gaps")
        first_truncate_write = pd.read_sql("SELECT * FROM sff_fact_fu_gaps", sqlite_engine)
        RECORDER.check(len(first_truncate_write) == 2,
                       "truncate mode on a missing table falls back to replace")

        # an empty frame is a legal table: the write must not be skipped
        empty_frame = frame_to_write.head(0)
        configuration.write(empty_frame, "lookup_fu")
        empty_table = pd.read_sql("SELECT * FROM sff_lookup_fu", sqlite_engine)
        RECORDER.check(len(empty_table) == 0 and "execution_id" in empty_table.columns,
                       "an empty frame is written as an empty table, with its columns")

        # a column holding nested objects is a named error, not a driver crash
        frame_with_nested_objects = pd.DataFrame({"fu_id": ["EU|A"], "payload": [{"a": 1}]})
        RECORDER.check_raises(lambda: configuration.write(frame_with_nested_objects, "fact_fu"),
                              "payload",
                              "a column of nested objects stops the write and is named")


def test_write_to_csv_without_engine() -> None:
    """With no engine configured, `write` falls back to CSV under `outdir`."""
    RECORDER.start_block("BLOCK 4 · write without engine (CSV fallback)")

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory = os.path.join(temporary_directory, "salida")
        configuration = build_test_config(outdir=output_directory)
        RECORDER.check(os.path.isdir(output_directory),
                       "the output directory is created at construction")
        RECORDER.check(configuration.engine is None,
                       "no engine and no server configured means no engine at all")

        frame_to_write = pd.DataFrame({"fu_id": ["EU|A"], "tasa": [0.8]})
        configuration.write(frame_to_write, "fact_fu")

        csv_path = os.path.join(output_directory, "sff_fact_fu.csv")
        RECORDER.check(os.path.exists(csv_path),
                       "the CSV lands under outdir with the physical name")

        persisted_frame = pd.read_csv(csv_path)
        RECORDER.check("process_date" in persisted_frame.columns
                       and "execution_id" in persisted_frame.columns,
                       "the CSV carries the same traceability columns as the SQL table")


def test_engine_handling() -> None:
    """An injected engine is reused; a non-mssql engine is left alone."""
    RECORDER.start_block("BLOCK 4 · engine handling")

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = os.path.join(temporary_directory, "engine.db")
        sqlite_engine = create_engine(f"sqlite:///{database_path}")
        configuration = build_test_config(sql_engine=sqlite_engine, sql_schema=None,
                                          outdir=temporary_directory)

        RECORDER.check(configuration.engine is sqlite_engine,
                       "an injected engine is returned as is, never rebuilt")
        RECORDER.check(configuration.engine is configuration.engine,
                       "asking twice for the engine yields the same object")
        RECORDER.check(not getattr(sqlite_engine, "_sff_fast", False),
                       "a non-mssql engine is not patched with fast_executemany")

    # the connection fields alone are enough to know an engine WOULD be built; the
    # engine itself is not built here because no ODBC driver exists in the test machine
    configuration_with_server = build_test_config(sql_server="server", sql_database="Kamelot")
    RECORDER.check(configuration_with_server.sql_engine is None,
                   "the engine is lazy: nothing is built until it is asked for")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 5 · PRODUCTION DEFAULTS (pinned contract)
# ═══════════════════════════════════════════════════════════════════════════════════

def test_production_defaults() -> None:
    """`Config()` with no arguments is production: its values are the sealed contract.

    A failure here is not necessarily a bug: it is a CHANGE OF CONTRACT. Declare it in
    ASUNCIONES, regenerate the reference, and only then update the expected value.
    """
    RECORDER.start_block("BLOCK 5 · production defaults (pinned contract)")

    production_configuration = Config()

    RECORDER.check(production_configuration.business_mandatory_dims
                   == EXPECTED_PRODUCTION_MANDATORY_DIMS,
                   "the ten mandatory dimensions are unchanged, in order")
    RECORDER.check(production_configuration.structural_timevarying_dims
                   == EXPECTED_PRODUCTION_TIMEVARYING_DIMS,
                   "the timevarying dimensions and their signs are unchanged")
    RECORDER.check(production_configuration.extra_revalorizacion
                   == EXPECTED_PRODUCTION_EXTRA_REVALORIZACION,
                   "the revaluation extras are unchanged, in order")
    RECORDER.check(production_configuration.extra_renovacion == [],
                   "extra_renovacion is empty in production, on purpose")
    RECORDER.check(production_configuration.semantic_labels == [],
                   "semantic_labels is empty in production, on purpose")

    for parameter_name, expected_value in EXPECTED_BUSINESS_PARAMETERS.items():
        actual_value = getattr(production_configuration, parameter_name)
        RECORDER.check(actual_value == expected_value,
                       f"business parameter {parameter_name} is {expected_value} "
                       f"(found {actual_value})")

    RECORDER.check(len(production_configuration.rate_series_columns) == 11,
                   "the production rate series has 11 columns (10 mandatory + softcancel)")
    RECORDER.check(len(production_configuration.uplift_cell_columns) == 14,
                   "the production uplift cell has 14 columns (10 mandatory + 4 extras)")

    RECORDER.check(production_configuration.sql_table_prefix == "sff_"
                   and production_configuration.sql_schema == "dbo"
                   and production_configuration.sql_write_mode == "replace",
                   "the persistence defaults are prefix sff_, schema dbo, mode replace")


def test_physical_name_registry() -> None:
    """The registry is the name contract of the reference and of the BI."""
    RECORDER.start_block("BLOCK 5 · physical name registry (pinned contract)")

    production_configuration = Config()

    for logical_name, expected_physical_name in EXPECTED_PHYSICAL_NAMES.items():
        resolved_name = production_configuration._resolve_physical_table_name(logical_name)
        RECORDER.check(resolved_name == expected_physical_name,
                       f"{logical_name} → {expected_physical_name} (found {resolved_name})")

    RECORDER.check(len(PHYSICAL_TABLE_NAMES) == EXPECTED_REGISTRY_SIZE,
                   f"the registry holds {EXPECTED_REGISTRY_SIZE} logical names "
                   f"(found {len(PHYSICAL_TABLE_NAMES)})")

    # the six decision tables are the contract between ANALYSIS and RUN
    RECORDER.check(all(name in PHYSICAL_TABLE_NAMES for name in DECISION_TABLES),
                   "the registry holds the eight decision tables (ANALYSIS → RUN contract)")
    RECORDER.check(not any("showcase" in name for name in PHYSICAL_TABLE_NAMES),
                   "no showcase table: Simpson is explained by decomposition, not by case hunting")

    # two logical names must never resolve to the same physical table: one would
    # silently overwrite the other on every run
    resolved_names = [production_configuration._resolve_physical_table_name(logical_name)
                      for logical_name in PHYSICAL_TABLE_NAMES]
    RECORDER.check(len(set(resolved_names)) == len(resolved_names),
                   "no two logical names resolve to the same physical table")

    every_name_is_prefixed = all(resolved_name.startswith("sff_")
                                 for resolved_name in resolved_names)
    RECORDER.check(every_name_is_prefixed, "every physical name carries the sff_ prefix")


# ═══════════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [test_taxonomy_exclusions,
             test_read_raw_is_overridable,
             test_column_contract,
             test_series_and_cell_columns,
             test_join_columns,
             test_hash_key,
             test_physical_table_names,
             test_write_to_sql,
             test_write_to_csv_without_engine,
             test_engine_handling,
             test_production_defaults,
             test_physical_name_registry]


def main() -> int:
    """Run every test function and return the exit code.

    INPUT:   nothing.
    OUTPUT:  0 when every check passed, 1 otherwise.
    RULES:   a failing check never stops the run: one execution must report every
             broken check, so a single pass tells the whole story.
    EDGE CASES: an exception inside a test function propagates and stops the run — that
             is a bug in the test itself, not a failed check, and must be visible.
    CONSOLE: the per-check lines, the panel, and the final verdict.
    STEPS:
      [1] Header.
      [2] Every test function, in order.
      [3] Panel and verdict.
    """
    # [1] what is being tested
    print("═" * 74)
    print("TEST config.py · SFF v2 · RUN")
    print("Run this after every change to config.py, before the rest of the battery.")
    print("═" * 74)

    # [2] the tests, each opening its own block in the recorder
    for test_function in ALL_TESTS:
        test_function()

    # [3] the verdict
    return RECORDER.print_panel("CONFIG TEST")


if __name__ == "__main__":
    sys.exit(main())
