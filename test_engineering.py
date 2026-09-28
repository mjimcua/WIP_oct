"""test_engineering.py — the software engineer's battery for SFF v3.

    python test_engineering.py   →  exit 0 if healthy, 1 with the failures

What a software engineer checks that the phase tests do not: determinism (same input,
same output, twice), idempotency of the writes (running twice leaves the same tables),
referential integrity across every table (keys resolve, no orphans, no duplicates),
the ANALYSIS → RUN contract (the monthly run reproduces the analysis from the persisted
decisions only), robustness to degenerate inputs (empty projection, a single series,
a month with no expirations, unknown values), configuration validation (bad
parameters fail loudly and early), and a performance smoke (time per phase on a
mid-size synthetic within a budget).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import PHYSICAL_TABLE_NAMES, Config, hash_key
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from pipeline import DECISION_TABLES, run_analysis, run_pipeline
from synthetic_v3 import build_raw
from test_fixtures import quiet

RECORDER = CheckRecorder()
TIMEVARYING = {"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"}


class SyntheticConfig(Config):
    seed: int = 7

    def read_raw(self) -> pd.DataFrame:
        return build_raw(7)


def synthetic_config(folder: str, **overrides) -> Config:
    arguments = dict(sql_engine=create_engine(f"sqlite:///{os.path.join(folder, 'e.db')}"), sql_schema=None, outdir=folder,
                     business_mandatory_dims=["region"], structural_timevarying_dims=TIMEVARYING,
                     extra_renovacion=["product", "channel"], extra_revalorizacion=["discount", "newcust"],
                     extended_horizon_end="2027-12", uplift_parent_keep_columns=["newcust"],
                     benchmark_group_dims=["region", "product"], benchmark_min_support=100, sheets_top_series=0,
                     console_explanations=False)
    arguments.update(overrides)
    return SyntheticConfig(**arguments)


def read(configuration: Config, logical: str) -> pd.DataFrame:
    physical = configuration._resolve_physical_table_name(logical)
    return pd.read_sql(f"SELECT * FROM {physical}", configuration.engine)


# ═══════════════════════════════════════════════════════════════════════════════════

def test_determinism_and_idempotency() -> dict:
    RECORDER.start_block("E1 · determinism and idempotency")
    folder = tempfile.mkdtemp()
    configuration = synthetic_config(folder)
    with quiet():
        first = run_analysis(configuration)
        second = run_analysis(configuration)
    total_first = first["forecast"]["forecast_detail"]["esperado_usd"].sum()
    total_second = second["forecast"]["forecast_detail"]["esperado_usd"].sum()
    RECORDER.check(abs(total_first - total_second) < 1e-6, f"the same input gives the same total twice (${total_first:,.2f})")
    bands_first = first["decisions"]["decision_error_bands"].sort_values(["id_estimacion", "h"]).reset_index(drop=True)
    bands_second = second["decisions"]["decision_error_bands"].sort_values(["id_estimacion", "h"]).reset_index(drop=True)
    RECORDER.check(bands_first.equals(bands_second), "the decision tables are identical between two runs (no hidden randomness)")
    uplift_first = first["decisions"]["decision_uplift"].set_index("uplift_cell_id")["banda_low"]
    uplift_second = second["decisions"]["decision_uplift"].set_index("uplift_cell_id")["banda_low"]
    RECORDER.check(np.allclose(uplift_first, uplift_second, equal_nan=True), "the bootstrap band is seeded: identical between runs")
    tables = set(inspect(configuration.engine).get_table_names())
    expected = {configuration._resolve_physical_table_name(name) for name in PHYSICAL_TABLE_NAMES}
    RECORDER.check(expected <= tables, f"every registered table exists after the run ({len(expected - tables)} missing)")
    card_rows = len(read(configuration, "series_card"))
    RECORDER.check(card_rows == len(second["series_card"]), "writes are 'replace': the second run leaves one copy, not two")
    return dict(folder=folder, configuration=configuration, results=second)


def test_referential_integrity(context: dict) -> None:
    RECORDER.start_block("E2 · referential integrity across tables")
    configuration = context["configuration"]
    fact_fu, fine, card = read(configuration, "fact_fu"), read(configuration, "fact_fine"), read(configuration, "series_card")
    bridge, detail = read(configuration, "key_bridge"), read(configuration, "forecast_detail")
    RECORDER.check(fact_fu["fu_key"].is_unique and card["fs_key"].is_unique, "fu_key unique in fact_fu; fs_key unique in series_card")
    RECORDER.check(set(fine["fu_key"]) == set(fact_fu["fu_key"]), "fact_fine and fact_fu hold the same set of units")
    RECORDER.check(set(fact_fu["fs_key"]) <= set(card["fs_key"]), "every unit's series exists in series_card")
    RECORDER.check(set(bridge["fu_comb_key"]) == set(fine["fu_comb_key"]), "key_bridge covers every raw row exactly")
    RECORDER.check(set(detail["fs_key"]) <= set(card["fs_key"]), "every forecast row's series exists in series_card")
    reference = read(configuration, "pool_reference")
    trainable_with_history = card[(card["ruta"] == TREATMENT_PREDICTABLE) & (card["meses_historia"] > 0)]
    RECORDER.check(set(trainable_with_history["estimacion_key"]) <= set(reference["estimacion_key"]),
                   "every trainable series' estimation id has a pool reference (heuristic series have no history to pool)")
    technique = read(configuration, "decision_technique")
    with_support = reference[reference["gate"] == "nivel"]
    RECORDER.check(set(with_support["estimacion_key"]) <= set(technique["estimacion_key"]), "every pool with support has a technique decision")
    RECORDER.check(hash_key("EU|0|0|0|0|A|web") == card.set_index("fs_id").loc["EU|0|0|0|0|A|web", "fs_key"],
                   "keys are hash_key(id) and stable (a Power BI relationship can be built from the id)")
    fine_sum = fine.groupby("fu_key")["total_tr_usd"].sum()
    unit_sum = fact_fu.set_index("fu_key")["total_tr_usd"]
    RECORDER.check(np.allclose(fine_sum.reindex(unit_sum.index), unit_sum), "fine rows sum to their unit's pipeline $ for every unit")


def test_analysis_run_contract(context: dict) -> None:
    RECORDER.start_block("E3 · ANALYSIS → RUN contract")
    configuration = context["configuration"]
    with quiet():
        monthly = run_pipeline(configuration)
    analysis_total = context["results"]["forecast"]["forecast_detail"]["esperado_usd"].sum()
    monthly_total = monthly["forecast"]["forecast_detail"]["esperado_usd"].sum()
    RECORDER.check(abs(analysis_total - monthly_total) < 0.01, "the monthly run reproduces the analysis' total from the persisted decisions only")
    left = context["results"]["forecast"]["forecast_detail"].set_index("fu_comb_key")["tasa"].sort_index()
    right = monthly["forecast"]["forecast_detail"].set_index("fu_comb_key")["tasa"].reindex(left.index)
    RECORDER.check(np.allclose(left, right), "same rate on every row")
    RECORDER.check(all(name in PHYSICAL_TABLE_NAMES for name in DECISION_TABLES), "every decision table the run reads is registered")
    for name in DECISION_TABLES:
        RECORDER.check(len(read(configuration, name)) > 0, f"decision table '{name}' is persisted and not empty")


def test_degenerate_inputs() -> None:
    RECORDER.start_block("E4 · degenerate inputs")
    folder = tempfile.mkdtemp()
    base = build_raw(7)

    class NoProjection(SyntheticConfig):
        def read_raw(self):
            return base[base["period"] < "2026-09"]
    no_projection = NoProjection(**{k: v for k, v in vars(synthetic_config(folder)).items() if k in Config.__dataclass_fields__})
    no_projection.sql_engine = create_engine(f"sqlite:///{os.path.join(folder, 'np.db')}")
    survived = True
    try:
        with quiet():
            results = run_analysis(no_projection)
        survived = len(results["forecast"]["forecast_detail"]) >= 0
    except Exception as error:
        survived = False
        print(f"      (no projection: {type(error).__name__}: {error})")
    RECORDER.check(survived, "a raw whose last month is the current month (no known pipeline ahead) still runs (only the pending month and the simulated re-entries)")

    class OneSeries(SyntheticConfig):
        def read_raw(self):
            keep = (base["region"] == "EU") & (base["product"] == "A") & (base["channel"] == "web") & (base[["dormant", "softcancel", "no_instalado", "autorenew"]].sum(axis=1) == 0)
            return base[keep]
    one = OneSeries(**{k: v for k, v in vars(synthetic_config(folder)).items() if k in Config.__dataclass_fields__})
    one.sql_engine = create_engine(f"sqlite:///{os.path.join(folder, 'one.db')}")
    try:
        with quiet():
            results = run_analysis(one)
        ok = len(results["series_card"]) == 1 and results["forecast"]["forecast_detail"]["esperado_usd"].notna().all()
    except Exception as error:
        ok = False
        print(f"      (one series: {type(error).__name__}: {error})")
    RECORDER.check(ok, "a raw with a single series runs end to end (no relative to borrow from, no pairs to test)")


def test_configuration_validation() -> None:
    RECORDER.start_block("E5 · configuration validation")
    folder = tempfile.mkdtemp()
    bad = synthetic_config(folder, uplift_mandatory_dims=["not_a_dim"])
    raised = False
    try:
        _ = bad.uplift_cell_columns
    except ValueError as error:
        raised = "not mandatory" in str(error)
    RECORDER.check(raised, "uplift_mandatory_dims outside the mandatory dims fails loudly")
    outdated = synthetic_config(folder)
    delattr_ok = True
    try:
        from pipeline import check_configuration_version
        class Old:
            pass
        check_configuration_version(Old())
        delattr_ok = False
    except AttributeError as error:
        delattr_ok = "outdated" in str(error)
    RECORDER.check(delattr_ok, "an outdated config object is refused with a message naming the missing fields")
    strict = synthetic_config(folder, signed_ladder_max_loss=0.0)
    RECORDER.check(strict.signed_ladder_max_loss == 0.0 and synthetic_config(folder).signed_ladder_max_loss == 0.05,
                   "overrides win over defaults; defaults are the documented ones")


def test_performance_smoke() -> None:
    RECORDER.start_block("E6 · performance smoke (40× synthetic ≈ 700 series)")
    folder = tempfile.mkdtemp()
    base = build_raw(7)
    parts = []
    for k in range(40):
        part = base.copy()
        part["region"] = part["region"] + str(k % 8)
        part["product"] = part["product"] + str(k // 8)
        parts.append(part)
    big = pd.concat(parts, ignore_index=True)

    class Big(SyntheticConfig):
        def read_raw(self):
            return big
    configuration = Big(**{k: v for k, v in vars(synthetic_config(folder)).items() if k in Config.__dataclass_fields__})
    configuration.sql_engine = None
    configuration.backtest_persist = "none"
    started = time.time()
    with quiet():
        results = run_analysis(configuration)
    elapsed = time.time() - started
    RECORDER.check(elapsed < 240, f"{len(results['series_card'])} series end to end in {elapsed:.0f} s (budget 240 s)")
    RECORDER.check(results["forecast"]["forecast_detail"]["esperado_usd"].notna().all(), "no NaN in the forecast at scale")


ALL_TESTS = [test_determinism_and_idempotency, test_referential_integrity, test_analysis_run_contract,
             test_degenerate_inputs, test_configuration_validation, test_performance_smoke]


def main() -> int:
    print("═" * 74 + "\nTEST engineering · determinism, integrity, contracts, robustness, performance\n" + "═" * 74)
    context = test_determinism_and_idempotency()
    test_referential_integrity(context)
    test_analysis_run_contract(context)
    test_degenerate_inputs()
    test_configuration_validation()
    test_performance_smoke()
    return RECORDER.print_panel("ENGINEERING TEST")


if __name__ == "__main__":
    sys.exit(main())
