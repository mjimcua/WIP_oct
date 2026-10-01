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

In a notebook:
    pipeline = Orchestrator(configuration)
    results = pipeline.run()                 # every step, in the order of their dependencies
    pipeline.run_step("forecast")            # one step: its inputs must exist and be current
    pipeline.context["forecast"]             # any table of the run
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from config import Config
from step_00_validate_raw import validate_raw
from step_01_validate_values import validate_values
from step_02_apply_calendar import apply_calendar
from step_02b_dimension_levels import apply_dimension_levels
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
from step_20_time_series import build_time_series_and_total, split_time_series_rows
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
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP INTERFACE, THE CONTEXT, THE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """A step: the tables it reads and writes, and how it runs on the context."""
    name: str
    label: str
    reads: tuple
    writes: tuple
    run: Callable


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

    def run(self) -> dict:
        """Every step, in the order of their dependencies; the results under their usual names."""
        for step_name in self.order:
            self.run_step(step_name)
        return self.context.results()

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
        self.configuration.logger.debug(f"[pipeline] {step_name} ({step.label}) in {time.time() - started:.1f}s")
        return outputs

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

    def split(context):
        rows, time_series_rows = split_time_series_rows(context["validated_extract"], context.configuration)
        return {"validated_renewals": rows, "time_series_rows": time_series_rows}

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

    report_reads = ("raw_extract", "audit", "backtest", "contract_check", "fine_table", "forecast", "forecast_total", "ladder",
                    "money_by_level", "pool_dynamics", "pool_reference", "portfolio_dynamics", "portfolio_profile", "rated_units",
                    "series_table", "series_estimate", "series_exam", "series_rate", "time_series", "uplift_backtest",
                    "uplift_cells", "uplift_verdict", "validation", "core")
    return [
        Step("read_raw", "--", (), ("raw_extract",), read_raw),
        Step("validate_raw", "00", ("raw_extract",), ("validated_extract",),
             lambda context: {"validated_extract": validate_raw(context["raw_extract"], context.configuration)}),
        Step("split_time_series", "20a", ("validated_extract",), ("validated_renewals", "time_series_rows"), split),
        Step("validate_values", "01", ("validated_renewals",), ("validated_raw",),
             lambda context: {"validated_raw": validate_values(context["validated_renewals"], context.configuration)}),
        Step("calendar", "02", ("validated_raw",), ("calendared_extract",),
             lambda context: {"calendared_extract": apply_calendar(context["validated_raw"], context.configuration)}),
        Step("dimension_levels", "02b", ("calendared_extract",), ("calendared_raw",),
             lambda context: {"calendared_raw": apply_dimension_levels(context["calendared_extract"], context.configuration)}),
        Step("fine_table", "03", ("calendared_raw",), ("fine_table",),
             lambda context: {"fine_table": build_fine_table(context["calendared_raw"], context.configuration)}),
        Step("forecast_units", "04", ("fine_table",), ("forecast_units",),
             lambda context: {"forecast_units": build_forecast_units(context["fine_table"], context.configuration)}),
        Step("lookups", "05", ("fine_table", "forecast_units"), ("lookups",),
             lambda context: {"lookups": build_lookups(context["fine_table"], context["forecast_units"], context.configuration)}),
        Step("series_routes", "06", ("forecast_units",), ("series_table",),
             lambda context: {"series_table": build_series_routes(context["forecast_units"], context.configuration)}),
        Step("support_bound", "07", ("forecast_units",), ("support_bound",),
             lambda context: {"support_bound": build_support_bound(context["forecast_units"], context.configuration)}),
        Step("rate_series", "08", ("forecast_units", "series_table"), ("rated_units", "series_rate"), rate_series),
        Step("dimensions", "09", ("series_rate", "lookups"), ("dimension_decision", "dimension_pairs"), dimensions),
        Step("ladder", "10", ("rated_units", "series_rate", "lookups", "dimension_decision"), ("ladder",),
             lambda context: {"ladder": build_ladder_groups(context["rated_units"], context["series_rate"],
                                                            context["lookups"]["lookup_fs"], context["dimension_decision"],
                                                            context.configuration)}),
        Step("credibility", "11", ("series_rate", "ladder"), ("series_estimate", "money_by_level"), credibility),
        Step("compositions", "12", ("rated_units", "ladder", "series_estimate"), ("pool_series", "pool_reference"), compositions),
        Step("dynamics", "13", ("pool_series", "pool_reference"), ("pool_dynamics", "portfolio_profile", "portfolio_dynamics"), dynamics),
        Step("backtest", "14", ("pool_series", "pool_reference"), ("backtest",),
             lambda context: {"backtest": run_backtest(context["pool_series"], context["pool_reference"], context.configuration)}),
        Step("uplift", "15", ("fine_table",), ("uplift_cells", "contract_check"), uplift),
        Step("uplift_backtest", "16", ("fine_table",), ("uplift_backtest", "uplift_verdict"), uplift_backtest),
        Step("forecast", "17", ("fine_table", "forecast_units", "series_estimate", "pool_series", "pool_reference", "backtest",
                                "uplift_cells", "uplift_verdict", "rated_units"), ("forecast",), forecast),
        Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("series_exam",), series_exam),
        Step("time_series", "20", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series),
        Step("audit", "AUD", ("ladder", "series_rate", "series_estimate", "rated_units", "pool_series", "pool_reference",
                              "pool_dynamics", "backtest", "forecast", "series_exam"), ("audit",), audit),
        Step("core", "NU", ("fine_table", "forecast_units", "support_bound", "rated_units", "series_table", "series_rate",
                            "series_estimate", "pool_dynamics", "backtest", "forecast", "time_series_rows", "time_series",
                            "forecast_total", "ladder", "audit", "series_exam"), ("core", "core_legend", "forecast_series"), core),
        Step("validation", "18", ("raw_extract", "backtest", "fine_table", "forecast", "forecast_units", "pool_reference",
                                  "series_exam", "series_estimate", "time_series_rows", "core"), ("validation",),
             lambda context: {"validation": validate_chain(context["raw_extract"], context.results(), context.configuration)}),
        Step("report", "IN", report_reads, ("card",),
             lambda context: {"card": build_report(context["raw_extract"], context.results(), context.configuration)}),
    ]
