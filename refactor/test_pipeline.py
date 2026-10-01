"""
test_pipeline.py — The orchestrator: steps ordered by their dependencies, a step refused without its
inputs, nothing stale after a step runs again, a table that breaks its contract stopped at once, and
the checkpoints: run_from a step gives the same result without re-running the steps before it, and
warns when the code or the Config changed since the checkpoint.

    python test_pipeline.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
import tempfile

import pandas as pd

from pipeline import ContractError, Orchestrator, Step, TableContract, dependency_order, sff_steps
from test_helpers import check, check_stops, console_of, finish, synthetic_with


def test_the_order() -> None:
    print("A · the order comes from the dependencies")
    steps = sff_steps()
    order = dependency_order(list(reversed(steps)))
    position = {name: index for index, name in enumerate(order)}
    writer = {table: step.name for step in steps for table in step.writes}
    check(all(position[writer[table]] < position[step.name] for step in steps for table in step.reads),
          "every table is written before it is read, even with the steps declared backwards")
    check_stops(lambda: dependency_order([Step("a", "A", ("x",), ("y",), None), Step("b", "B", ("y",), ("x",), None)]),
                "cycle", "a cycle is refused")
    check_stops(lambda: dependency_order([Step("a", "A", ("nobody",), ("y",), None)]),
                "written by no step", "a table nobody writes is refused")


def test_one_step_at_a_time() -> None:
    print("B · one step at a time, as in a notebook")
    pipeline = Orchestrator(synthetic_with())
    check_stops(lambda: pipeline.run_step("forecast"), "run first", "a step without its inputs is refused, naming what to run first")
    console_of(pipeline.run)
    check("forecast_series" in pipeline.context and "card" in pipeline.context, "the whole run fills the catalogue")
    console_of(lambda: pipeline.run_step("backtest"))
    check("backtest" in pipeline.context and "forecast" not in pipeline.context and "core" not in pipeline.context,
          "running a step again drops what the later steps had produced: nothing stale can be used")
    check_stops(lambda: pipeline.run_step("core"), "run first", "and the later steps must run again before the core")


def test_the_contracts() -> None:
    print("C · a table that breaks its contract stops at the step that wrote it")
    contract = TableContract("series_rate", "forecast series", ("fs_id",), ("ruta",))
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a", "a"], "ruta": ["x", "y"]}), "rate_series"),
                "repeated keys", "a repeated key is refused, naming the table and the step")
    check_stops(lambda: contract.validate(pd.DataFrame({"fs_id": ["a"]}), "rate_series"),
                "lacks the columns", "a missing column is refused, naming it")


def total_2026(results) -> float:
    total = results["forecast_total"]
    return float(total.loc[(total["ano"] == 2026) & (total["origen"] == "TOTAL"), "usd_renovado"].item())


def test_the_checkpoints() -> None:
    print("D · checkpoints: run from any step, assuming the rest")
    folder = tempfile.mkdtemp()
    full = {}
    console_of(lambda: full.update(Orchestrator(synthetic_with(checkpoint_folder=folder)).run()))
    manifest = json.load(open(os.path.join(folder, "checkpoint_manifest.json"), encoding="utf-8"))
    check(len(manifest["steps"]) == len(sff_steps()) and os.path.exists(os.path.join(folder, "fine_table.pkl")),
          "a full run saves every step: its tables (.pkl) and its line in the manifest")
    resumed = {}
    console = console_of(lambda: resumed.update(Orchestrator(synthetic_with(checkpoint_folder=folder)).run_from("forecast")))
    check("STEP 03" not in console and "STEP 17" in console and "steps loaded" in console,
          "run_from('forecast') loads the steps before it and runs only it and the rest")
    check(abs(total_2026(resumed) - total_2026(full)) < 0.01, "and gives the same forecast as the full run")
    check("changed since the checkpoint" not in console, "no warning when nothing changed")
    console = console_of(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder, collapse_passes=3)).run_from("forecast"))
    check("the Config changed since the checkpoint in ['collapse_passes']" in console,
          "a Config that changed since the checkpoint is warned, naming the field")
    manifest["steps"]["ladder"]["code"] = "something-else"
    json.dump(manifest, open(os.path.join(folder, "checkpoint_manifest.json"), "w", encoding="utf-8"))
    console = console_of(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder)).load_until("forecast"))
    check("the code of ['ladder'] changed since the checkpoint" in console,
          "a loaded step whose code changed is warned, with the step to run from")
    del manifest["steps"]["backtest"]
    json.dump(manifest, open(os.path.join(folder, "checkpoint_manifest.json"), "w", encoding="utf-8"))
    check_stops(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder)).run_from("forecast"),
                "run from 'backtest'", "a step missing from the checkpoint stops the run, naming the step to run first")


def test_what_a_step_changes_in_the_config() -> None:
    print("E · what a step changes in the Config is saved and restored (step 02b: the mandatory dims)")
    folder, levels = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "levels.json")
    options = dict(checkpoint_folder=folder, leveled_dims={"product": {"type": "nominal"}}, levels_path=levels)
    full = {}
    console_of(lambda: full.update(Orchestrator(synthetic_with(**options)).run()))
    configuration = synthetic_with(**options)
    check(configuration.business_mandatory_dims == ["region", "product"], "a new Config starts with the declared dims")
    resumed = {}
    console = console_of(lambda: resumed.update(Orchestrator(configuration).run_from("ladder")))
    check(configuration.business_mandatory_dims == ["region", "product_level_1", "product_level_2"]
          and "changed since the checkpoint" not in console,
          "resuming after step 02b restores the dims it had set, without a false warning")
    check(abs(total_2026(resumed) - total_2026(full)) < 0.01, "and the forecast is the same as the full run")


if __name__ == "__main__":
    test_the_order()
    test_one_step_at_a_time()
    test_the_contracts()
    test_the_checkpoints()
    test_what_a_step_changes_in_the_config()
    finish()
