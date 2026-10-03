"""
pipeline.py — The steps of the SFF with formal interfaces, run over a common context.

Three pieces (the pattern of a data pipeline: node + data catalogue + pipeline):

  TableContract     a table declared once: its grain, its key (unique) and its required columns.
                    Checked when a step writes it: a missing column or a repeated key stops the run
                    at the step that produced it, with the table's name.
  Step              the common interface of a step: its name, the tables it READS, the tables it
                    WRITES, and run(context) → {table: value}. Every step of the SFF is one Step; the
                    step functions are unchanged (each Step adapts the context to its arguments).
  PipelineContext   the catalogue of the run: every table produced so far, by name, with the step
                    that produced it, plus the Config.
  Orchestrator      orders the steps by their dependencies (what each reads must be written before),
                    runs them all or one at a time, and refuses to run a step whose inputs are missing.
                    Running a step again drops what the steps after it had produced: nothing stale can
                    be used (the notebook problem of an old object passed to a new step).

CHECKPOINTS (save_checkpoints, on by default): every step saves the tables it writes to
checkpoint_folder (one .pkl per table) and records itself in checkpoint_manifest.json: when it
finished and the hash of its code. No step changes the Config, so a checkpoint is only tables.
run_from(step) loads every table of the steps before it and runs from that step on.
YOU decide when a full run is needed; the orchestrator only WARNS (it does not stop) when, since the
checkpoint was saved:
  - the code of a step that is loaded (not re-run) changed, or a shared module changed
    (config.py, prediction.py, techniques.py, vocabulario.py);
  - the Config changed (any field: the checkpoint was made with other parameters).
A step that is missing from the checkpoint stops the run: it names the step to run first.

In a notebook:
    pipeline = Orchestrator(configuration)
    results = pipeline.run()                 # every step, in the order of their dependencies (and saved)
    results = pipeline.run_from("forecast")  # the steps before "forecast" loaded from disk; it and the rest run
    pipeline.load_until("forecast")          # only load (to inspect the inputs of a step), run nothing
    pipeline.run_step("forecast")            # one step: its inputs must exist and be current
    pipeline.context["forecast"]             # any table of the run
    pipeline.order                           # the steps, in the order they run (the names run_from takes)
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from config import Config
from step_00_validate_raw import validate_raw
from step_01_values_and_levels import validate_values_and_build_levels
from step_02_apply_calendar import apply_calendar
from step_03_fine_table import build_fine_table
from step_04_forecast_units import build_forecast_units
from step_05_lookups import build_lookups
from step_06_series_routes import build_series_routes
from step_07_support_bound import build_support_bound
from step_08_rate_series import build_rate_series
from step_09_dimensions import analyse_dimensions
from step_10_ladder_groups import build_ladder_groups
from step_11_ladder import climb_the_ladder
from step_12_pool_series import build_pool_series
from step_13_dynamics import measure_dynamics
from step_14_backtest import run_backtest
from step_15_uplift import estimate_uplift
from step_16_uplift_backtest import backtest_uplift
from step_17_forecast import assemble_forecast
from step_18_validation import validate_chain
from step_19_series_exam import examine_series
from step_20_time_series import build_time_series_and_total
from step_21_new_rows import count_the_new_rows
from step_22_power_bi import build_power_bi_tables
from step_audit import build_audit_tables
from step_informe import build_report
from step_nucleo import build_core_table
from vocabulario import COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN, UNIT_ID_COLUMN


class ContractError(ValueError):
    """A table that does not keep its contract, or a step run without its inputs."""


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONTRACTS OF THE TABLES THAT CROSS STEPS
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TableContract:
    """A table declared once: grain, unique key, required columns. `part` reads a table inside a dict."""
    name: str
    grain: str
    key: tuple = ()
    required: tuple = ()
    part: str = None

    def validate(self, value, producer: str) -> None:
        frame = value[self.part] if self.part else value
        if not isinstance(frame, pd.DataFrame):
            raise ContractError(f"{self.name} (written by {producer}) is not a table")
        missing = [column for column in self.required + self.key if column not in frame.columns]
        if missing:
            raise ContractError(f"{self.name} (written by {producer}) lacks the columns {missing}; grain: {self.grain}")
        if self.key and frame.duplicated(list(self.key)).any():
            repeated = int(frame.duplicated(list(self.key)).sum())
            raise ContractError(f"{self.name} (written by {producer}): {repeated:,} repeated keys {list(self.key)}; grain: {self.grain}")


def contracts(configuration: Config) -> dict:
    """The contract of every table that crosses steps (by context name; several per name for a dict)."""
    period = configuration.period_col
    return {
        "fine_table": [TableContract("fine_table", "one row per fine row of the extract", (),
                                     (SERIES_ID_COLUMN, UNIT_ID_COLUMN, period))],
        "forecast_units": [TableContract("forecast_units", "forecast series × month", (UNIT_ID_COLUMN,), (SERIES_ID_COLUMN, period))],
        "series_table": [TableContract("series_table", "forecast series", (SERIES_ID_COLUMN,), ("ruta",))],
        "series_rate": [TableContract("series_rate", "forecast series", (SERIES_ID_COLUMN,),
                                      ("ruta", "signo", "n_propio", "tasa_propia", "usd_por_predecir"))],
        "rated_units": [TableContract("rated_units", "forecast series × month (gaps included)", (SERIES_ID_COLUMN, period), ("tasa",))],
        "ladder": [TableContract("ladder.groups", "estimable forecast series", (SERIES_ID_COLUMN,),
                                 (COMPOSITION_ID_COLUMN, "final_step", "group_support", "group_rate", "credibility_ref_id"), part="groups"),
                   TableContract("ladder.composition_members", "composition × series in its rate",
                                 (COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN), ("role",), part="composition_members"),
                   TableContract("ladder.steps", "forecast series × pass", (SERIES_ID_COLUMN, "ladder_step"), ("group_id",), part="steps")],
        "series_estimate": [TableContract("series_estimate", "forecast series", (SERIES_ID_COLUMN,),
                                          (COMPOSITION_ID_COLUMN, "z", "credibility_ref_id", "group_rate", "ref_rate", "tasa_estimada"))],
        "pool_series": [TableContract("pool_series", "composition × month", (COMPOSITION_ID_COLUMN, period), ("vencen", "renovadas", "tasa"))],
        "pool_reference": [TableContract("pool_reference", "composition", (COMPOSITION_ID_COLUMN,), ("gate", "n_pool"))],
        "forecast": [TableContract("forecast.forecast", "future fine row", (), ("tasa", "tasa_baja", "tasa_alta", "esperado_usd"),
                                   part="forecast")],
        "series_exam": [TableContract("series_exam.per_series", "forecast series", (SERIES_ID_COLUMN,),
                                      ("framework_predictions", "raw_predictions"), part="per_series")],
        "core": [TableContract("core", "fine row of the extract, gap, synthetic or time_series row", (), ("s03_fs_id",))],
        "forecast_series": [TableContract("forecast_series", "forecast series", ("s03_fs_id",), ())],
        "power_bi": [TableContract("power_bi.pipeline_source", "forecast pipeline source", ("forecast_pipeline_source",),
                                   ("source_label", "source_block", "source_order"), part="pipeline_source")],
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP INTERFACE, THE CONTEXT, THE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """A step: the tables it reads and writes, how it runs on the context, and the module of its code
    (its hash goes to the checkpoint). A step never changes the Config."""
    name: str
    label: str
    reads: tuple
    writes: tuple
    run: Callable
    module: str = None


@dataclass
class PipelineContext:
    """The catalogue of the run: every table produced so far, with the step that produced it."""
    configuration: Config
    tables: dict = field(default_factory=dict)
    producer: dict = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def __getitem__(self, name: str):
        if name not in self.tables:
            raise ContractError(f"'{name}' has not been produced in this run")
        return self.tables[name]

    def put(self, name: str, value, step_name: str, table_contracts: dict) -> None:
        for contract in table_contracts.get(name, []):
            contract.validate(value, step_name)
        self.tables[name] = value
        self.producer[name] = step_name

    def drop(self, names) -> None:
        for name in names:
            self.tables.pop(name, None)
            self.producer.pop(name, None)

    def results(self) -> dict:
        """The tables under the names the report, the validation and the tests read."""
        renamed = {"calendared_raw": "raw", "series_table": "series"}
        results = {renamed.get(name, name): value for name, value in self.tables.items()}
        if "series_exam" in self.tables:
            results["portfolio_exam"] = self.tables["series_exam"]["by_month"]
            results["portfolio_exam_summary"] = self.tables["series_exam"]["summary"]
        return results


class Orchestrator:
    """Orders the steps by their dependencies and runs them, all or one at a time."""

    def __init__(self, configuration: Config, steps: list = None):
        self.configuration = configuration
        self.steps = {step.name: step for step in (steps or sff_steps())}
        self.order = dependency_order(list(self.steps.values()))
        self.table_contracts = contracts(configuration)
        self.context = PipelineContext(configuration)
        self.producer_of = {table: step.name for step in self.steps.values() for table in step.writes}
        self.folder = configuration.checkpoint_folder or os.path.join(configuration.output_folder or ".", "checkpoints")
        self.manifest_path = os.path.join(self.folder, "checkpoint_manifest.json")
        self.seconds = {}                                  # how long every step took in this run

    def run(self) -> dict:
        """Every step, in the order of their dependencies; the results under their usual names.
        A full run starts a new checkpoint (the Config it runs with is recorded)."""
        if self.configuration.save_checkpoints:
            self.start_checkpoint()
        for step_name in self.order:
            self.run_step(step_name)
        self.show_timings()
        return self.context.results()

    def run_from(self, step_name: str) -> dict:
        """The steps before step_name loaded from the checkpoint; step_name and every step after it run
        (and are saved again). Warns when the code or the Config changed since the checkpoint."""
        self.load_until(step_name)
        for name in self.order[self.order.index(step_name):]:
            self.run_step(name)
        self.show_timings()
        return self.context.results()

    def show_timings(self) -> None:
        """How long every step took in this run, the slowest first: where a checkpoint saves the most."""
        if not self.seconds:
            return
        timings = pd.DataFrame({"step": list(self.seconds), "seconds": [round(value, 1) for value in self.seconds.values()]})
        timings["share"] = (timings["seconds"] / max(timings["seconds"].sum(), 1e-9)).map("{:.0%}".format)
        timings["run_from_saves_s"] = [round(sum(list(self.seconds.values())[:index]), 1) for index in range(len(timings))]
        self.configuration.logger.doc(f"[pipeline] time per step ({sum(self.seconds.values()):.0f}s in total; run_from_saves_s: "
                                      f"what run_from(step) skips, measured in this run):")
        self.configuration.show_table(timings.sort_values("seconds", ascending=False).head(12).reset_index(drop=True))

    def load_until(self, step_name: str) -> None:
        """Load from the checkpoint the tables of every step before step_name (nothing runs)."""
        if step_name not in self.steps:
            raise ContractError(f"no step '{step_name}'; the steps are: {', '.join(self.order)}")
        manifest = self.read_manifest()
        before = self.order[:self.order.index(step_name)]
        missing = [name for name in before if name not in manifest.get("steps", {})]
        if missing:
            raise ContractError(f"the checkpoint in {self.folder} lacks the steps {missing}: run from '{missing[0]}' "
                                f"(or run() everything) first")
        self.warn_about_changes(manifest, before)
        started = time.time()
        for name in before:
            saved = manifest["steps"][name]
            for table in self.steps[name].writes:
                with open(os.path.join(self.folder, f"{table}.pkl"), "rb") as handle:
                    self.context.tables[table] = pickle.load(handle)
                self.context.producer[table] = name
        last = manifest["steps"][before[-1]]["finished"] if before else "-"
        self.configuration.logger.doc(f"[checkpoint] {len(before)} steps loaded from {self.folder} "
                                      f"({before[0] if before else '-'} … {before[-1] if before else '-'}, saved {last}) "
                                      f"in {time.time() - started:.1f}s · running from '{step_name}'")

    def run_step(self, step_name: str):
        """One step: its inputs must exist; what the later steps had produced is dropped (no stale tables)."""
        step = self.steps[step_name]
        missing = [table for table in step.reads if table not in self.context]
        if missing:
            needed = sorted({self.producer_of.get(table, "?") for table in missing}, key=self.order.index)
            raise ContractError(f"step '{step_name}' needs {missing}: run first {needed}")
        self.context.drop([table for later in self.downstream_of(step_name) for table in self.steps[later].writes])
        started = time.time()
        outputs = step.run(self.context)
        if set(outputs) != set(step.writes):
            raise ContractError(f"step '{step_name}' wrote {sorted(outputs)} but declares {sorted(step.writes)}")
        for table, value in outputs.items():
            self.context.put(table, value, step_name, self.table_contracts)
        self.seconds[step_name] = time.time() - started
        self.configuration.logger.debug(f"[pipeline] {step_name} ({step.label}) in {self.seconds[step_name]:.1f}s")
        if self.configuration.save_checkpoints:
            self.save_step(step, outputs, time.time() - started)
        return outputs

    # ─── the checkpoint ───────────────────────────────────────────────────────────
    def start_checkpoint(self) -> None:
        """A new checkpoint: the Config of the run and the hash of the shared code; no step saved yet."""
        os.makedirs(self.folder, exist_ok=True)
        self.write_manifest({"created": datetime.now().isoformat(timespec="seconds"),
                             "config": config_fingerprint(self.configuration, self.steps.values()),
                             "shared_code": code_hash(SHARED_MODULES), "steps": {}})

    def save_step(self, step: Step, outputs: dict, seconds: float) -> None:
        """The tables of one step, and its line in the manifest."""
        os.makedirs(self.folder, exist_ok=True)
        for table, value in outputs.items():
            with open(os.path.join(self.folder, f"{table}.pkl"), "wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = self.read_manifest() or {"created": datetime.now().isoformat(timespec="seconds"),
                                            "config": config_fingerprint(self.configuration, self.steps.values()),
                                            "shared_code": code_hash(SHARED_MODULES), "steps": {}}
        manifest["steps"][step.name] = {
            "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
            "tables": list(step.writes), "code": code_hash([step.module] if step.module else [])}
        self.write_manifest(manifest)

    def read_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest(self, manifest: dict) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)

    def warn_about_changes(self, manifest: dict, loaded_steps: list) -> None:
        """Warnings (not errors): code of a loaded step, shared code or Config changed since the checkpoint."""
        logger = self.configuration.logger
        changed_steps = [name for name in loaded_steps
                         if self.steps[name].module and manifest["steps"][name]["code"] != code_hash([self.steps[name].module])]
        if changed_steps:
            logger.warning(f"[checkpoint] the code of {changed_steps} changed since the checkpoint: their tables are loaded, "
                           f"not recomputed. Run from '{changed_steps[0]}' if the change matters")
        if manifest.get("shared_code") != code_hash(SHARED_MODULES):
            logger.warning(f"[checkpoint] a shared module changed since the checkpoint ({', '.join(SHARED_MODULES)}): "
                           f"the loaded steps used the old one")
        now = config_fingerprint(self.configuration, self.steps.values())
        differences = sorted(name for name in set(now) | set(manifest.get("config", {}))
                             if now.get(name) != manifest.get("config", {}).get(name))
        if differences:
            logger.warning(f"[checkpoint] the Config changed since the checkpoint in {differences}: the loaded steps ran with "
                           f"the old values")

    def downstream_of(self, step_name: str) -> list:
        """Every step that reads, directly or not, what this step writes."""
        reached, frontier = set(), set(self.steps[step_name].writes)
        changed = True
        while changed:
            changed = False
            for step in self.steps.values():
                if step.name not in reached and step.name != step_name and frontier & set(step.reads):
                    reached.add(step.name)
                    frontier |= set(step.writes)
                    changed = True
        return [name for name in self.order if name in reached]


SHARED_MODULES = ["config", "prediction", "techniques", "vocabulario"]


def code_hash(module_names: list) -> str:
    """The hash of the source of some modules (an edit to any of them changes it)."""
    digest = hashlib.sha256()
    for module_name in module_names:
        module = sys.modules.get(module_name) or __import__(module_name)
        path = getattr(module, "__file__", None)
        if path and os.path.exists(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()[:16]


def config_fingerprint(configuration: Config, steps) -> dict:
    """The Config as text, field by field, without what cannot be compared (the engine, the SQL text)."""
    fingerprint = {}
    for config_field in dataclasses.fields(configuration):
        if config_field.name in ("sql_engine", "raw_extract_sql"):
            continue
        value = getattr(configuration, config_field.name)
        try:
            fingerprint[config_field.name] = json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            fingerprint[config_field.name] = str(type(value))
    return fingerprint


def dependency_order(steps: list) -> list:
    """The steps ordered so that every table is written before it is read (the declared order breaks ties).
    Stops on a table nobody writes or on a cycle."""
    writer = {}
    for step in steps:
        for table in step.writes:
            if table in writer:
                raise ContractError(f"'{table}' is written by both '{writer[table]}' and '{step.name}'")
            writer[table] = step.name
    unknown = sorted({table for step in steps for table in step.reads if table not in writer})
    if unknown:
        raise ContractError(f"tables read but written by no step: {unknown}")
    order, done = [], set()
    while len(order) < len(steps):
        ready = [step for step in steps if step.name not in done and all(writer[table] in done for table in step.reads)]
        if not ready:
            raise ContractError("the steps have a cycle: " + ", ".join(step.name for step in steps if step.name not in done))
        order.append(ready[0].name)
        done.add(ready[0].name)
    return order


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEPS OF THE SFF (each adapts the context to its step function, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════════

def sff_steps() -> list:
    """Every step of the SFF with what it reads and writes."""
    def read_raw(context):
        configuration = context.configuration
        configuration.logger.doc("[main] ═══ SFF · reading the raw ═══")
        started = time.time()
        raw = configuration.read_raw()
        configuration.logger.doc(f"[main] raw read: {len(raw):,} rows × {len(raw.columns)} columns ({time.time() - started:.1f}s)")
        return {"raw_extract": raw}

    def values_and_levels(context):
        leveled_raw, time_series_rows = validate_values_and_build_levels(context["validated_extract"], context.configuration)
        return {"leveled_raw": leveled_raw, "time_series_rows": time_series_rows}

    def rate_series(context):
        rated_units, series_rate = build_rate_series(context["forecast_units"], context["series_table"], context.configuration)
        return {"rated_units": rated_units, "series_rate": series_rate}

    def dimensions(context):
        decision, pairs = analyse_dimensions(context["series_rate"], context["lookups"]["lookup_fs"], context.configuration)
        return {"dimension_decision": decision, "dimension_pairs": pairs}

    def credibility(context):
        estimate, money = climb_the_ladder(context["series_rate"], context["ladder"], context.configuration)
        return {"series_estimate": estimate, "money_by_level": money}

    def compositions(context):
        series, reference = build_pool_series(context["rated_units"], context["ladder"], context["series_estimate"], context.configuration)
        return {"pool_series": series, "pool_reference": reference}

    def dynamics(context):
        pool_dynamics, profile, portfolio = measure_dynamics(context["pool_series"], context["pool_reference"], context.configuration)
        return {"pool_dynamics": pool_dynamics, "portfolio_profile": profile, "portfolio_dynamics": portfolio}

    def uplift(context):
        cells, contract_check = estimate_uplift(context["fine_table"], context.configuration)
        return {"uplift_cells": cells, "contract_check": contract_check}

    def uplift_backtest(context):
        backtest, verdict = backtest_uplift(context["fine_table"], context.configuration)
        return {"uplift_backtest": backtest, "uplift_verdict": verdict}

    def forecast(context):
        return {"forecast": assemble_forecast(context["fine_table"], context["forecast_units"], context["series_estimate"],
                                              context["pool_series"], context["pool_reference"], context["backtest"],
                                              context["uplift_cells"], context["uplift_verdict"], context["rated_units"],
                                              context.configuration)}

    def series_exam(context):
        return {"series_exam": examine_series(context["rated_units"], context["series_estimate"], context["pool_series"],
                                              context["backtest"], context.configuration,
                                              reference_members=context["ladder"]["reference_members"])}

    def time_series(context):
        table, total = build_time_series_and_total(context["time_series_rows"], context["fine_table"],
                                                   context["forecast"]["forecast"], context.configuration)
        return {"time_series": table, "forecast_total": total}

    def audit(context):
        return {"audit": build_audit_tables(context["ladder"], context["series_rate"], context["series_estimate"],
                                            context["rated_units"], context["pool_series"], context["pool_reference"],
                                            context["pool_dynamics"], context["backtest"], context["forecast"]["forecast"],
                                            context.configuration, series_exam_detail=context["series_exam"]["detail"])}

    def core(context):
        core_table, legend, dimension = build_core_table(
            context["fine_table"], context.configuration, forecast_units=context["forecast_units"],
            support_bound=context["support_bound"], rated_units=context["rated_units"], series_table=context["series_table"],
            series_rate=context["series_rate"], series_estimate=context["series_estimate"], pool_dynamics=context["pool_dynamics"],
            technique_decision=context["backtest"]["decision"], exam_by_pool=context["backtest"]["exam_by_pool"],
            forecast=context["forecast"]["forecast"], time_series_rows=context["time_series_rows"],
            time_series_table=context["time_series"], forecast_total=context["forecast_total"],
            ladder_stages=context["ladder"]["stages"], series_dynamics=context["audit"]["series_dynamics"],
            series_exam=context["series_exam"]["per_series"])
        return {"core": core_table, "core_legend": legend, "forecast_series": dimension}

    report_reads = ("raw_extract", "new_rows", "audit", "backtest", "contract_check", "fine_table", "forecast", "forecast_total", "ladder",
                    "money_by_level", "pool_dynamics", "pool_reference", "portfolio_dynamics", "portfolio_profile", "rated_units",
                    "series_table", "series_estimate", "series_exam", "series_rate", "time_series", "uplift_backtest",
                    "uplift_cells", "uplift_verdict", "validation", "core")
    return [
        Step("read_raw", "--", (), ("raw_extract",), read_raw, module="config"),
        Step("validate_raw", "00", ("raw_extract",), ("validated_extract",),
             lambda context: {"validated_extract": validate_raw(context["raw_extract"], context.configuration)}, module="step_00_validate_raw"),
        Step("values_and_levels", "01", ("validated_extract",), ("leveled_raw", "time_series_rows"), values_and_levels,
             module="step_01_values_and_levels"),
        Step("calendar", "02", ("leveled_raw",), ("calendared_raw",),
             lambda context: {"calendared_raw": apply_calendar(context["leveled_raw"], context.configuration)},
             module="step_02_apply_calendar"),
        Step("fine_table", "03", ("calendared_raw",), ("fine_table",),
             lambda context: {"fine_table": build_fine_table(context["calendared_raw"], context.configuration)}, module="step_03_fine_table"),
        Step("forecast_units", "04", ("fine_table",), ("forecast_units",),
             lambda context: {"forecast_units": build_forecast_units(context["fine_table"], context.configuration)}, module="step_04_forecast_units"),
        Step("lookups", "05", ("fine_table", "forecast_units"), ("lookups",),
             lambda context: {"lookups": build_lookups(context["fine_table"], context["forecast_units"], context.configuration)}, module="step_05_lookups"),
        Step("series_routes", "06", ("forecast_units",), ("series_table",),
             lambda context: {"series_table": build_series_routes(context["forecast_units"], context.configuration)}, module="step_06_series_routes"),
        Step("support_bound", "07", ("forecast_units",), ("support_bound",),
             lambda context: {"support_bound": build_support_bound(context["forecast_units"], context.configuration)}, module="step_07_support_bound"),
        Step("rate_series", "08", ("forecast_units", "series_table"), ("rated_units", "series_rate"), rate_series, module="step_08_rate_series"),
        Step("dimensions", "09", ("series_rate", "lookups"), ("dimension_decision", "dimension_pairs"), dimensions, module="step_09_dimensions"),
        Step("ladder", "10", ("rated_units", "series_rate", "lookups", "dimension_decision"), ("ladder",),
             lambda context: {"ladder": build_ladder_groups(context["rated_units"], context["series_rate"],
                                                            context["lookups"]["lookup_fs"], context["dimension_decision"],
                                                            context.configuration)}, module="step_10_ladder_groups"),
        Step("credibility", "11", ("series_rate", "ladder"), ("series_estimate", "money_by_level"), credibility, module="step_11_ladder"),
        Step("compositions", "12", ("rated_units", "ladder", "series_estimate"), ("pool_series", "pool_reference"), compositions, module="step_12_pool_series"),
        Step("dynamics", "13", ("pool_series", "pool_reference"), ("pool_dynamics", "portfolio_profile", "portfolio_dynamics"), dynamics, module="step_13_dynamics"),
        Step("backtest", "14", ("pool_series", "pool_reference"), ("backtest",),
             lambda context: {"backtest": run_backtest(context["pool_series"], context["pool_reference"], context.configuration)}, module="step_14_backtest"),
        Step("uplift", "15", ("fine_table",), ("uplift_cells", "contract_check"), uplift, module="step_15_uplift"),
        Step("uplift_backtest", "16", ("fine_table",), ("uplift_backtest", "uplift_verdict"), uplift_backtest, module="step_16_uplift_backtest"),
        Step("forecast", "17", ("fine_table", "forecast_units", "series_estimate", "pool_series", "pool_reference", "backtest",
                                "uplift_cells", "uplift_verdict", "rated_units"), ("forecast",), forecast, module="step_17_forecast"),
        Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("series_exam",), series_exam, module="step_19_series_exam"),
        Step("time_series", "20", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series, module="step_20_time_series"),
        Step("new_rows", "21", ("fine_table", "rated_units", "forecast"), ("new_rows",),
             lambda context: {"new_rows": count_the_new_rows(context["fine_table"], context["rated_units"],
                                                             context["forecast"]["forecast"], context.configuration)},
             module="step_21_new_rows"),
        Step("audit", "AUD", ("ladder", "series_rate", "series_estimate", "rated_units", "pool_series", "pool_reference",
                              "pool_dynamics", "backtest", "forecast", "series_exam"), ("audit",), audit, module="step_audit"),
        Step("core", "NU", ("fine_table", "forecast_units", "support_bound", "rated_units", "series_table", "series_rate",
                            "series_estimate", "pool_dynamics", "backtest", "forecast", "time_series_rows", "time_series",
                            "forecast_total", "ladder", "audit", "series_exam"), ("core", "core_legend", "forecast_series"), core, module="step_nucleo"),
        Step("power_bi", "22", ("core",), ("power_bi",),
             lambda context: {"power_bi": build_power_bi_tables(context["core"], context.configuration)},
             module="step_22_power_bi"),
        Step("validation", "18", ("raw_extract", "backtest", "fine_table", "forecast", "forecast_units", "pool_reference",
                                  "series_exam", "series_estimate", "time_series_rows", "core"), ("validation",),
             lambda context: {"validation": validate_chain(context["raw_extract"], context.results(), context.configuration)}, module="step_18_validation"),
        Step("report", "IN", report_reads, ("card",),
             lambda context: {"card": build_report(context["raw_extract"], context.results(), context.configuration)}, module="step_informe"),
    ]
