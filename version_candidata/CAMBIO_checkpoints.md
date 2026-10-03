# Cambio · checkpoints: ejecutar desde cualquier paso

Verificación: 21 ficheros de test en verde (298 comprobaciones; `test_pipeline` 19/19).

## Qué hace

- **Cada paso guarda sus tablas al terminar** en `checkpoint_folder` (por defecto `salida/checkpoints`), un `.pkl` por
  tabla, y se apunta en `checkpoint_manifest.json`: cuándo terminó, cuánto tardó, el hash de su código y lo que cambió en
  la Config (el paso 02b cambia las dimensiones mandatory; se restaura al reanudar).
- **`run_from("paso")`** carga del disco todo lo anterior a ese paso y ejecuta ese paso y los siguientes.
- **Tú decides cuándo hace falta una ejecución completa.** El orquestador solo **avisa** (no se detiene) si desde el
  checkpoint cambió el código de un paso que se carga, un módulo compartido (`config.py`, `prediction.py`,
  `techniques.py`, `vocabulario.py`) o algún campo de la Config. Si falta un paso en el checkpoint, se detiene y dice
  desde cuál ejecutar.
- **Al terminar, una tabla con lo que tardó cada paso** (el más lento primero) y lo que se ahorraría `run_from` en cada uno.

## Parámetros nuevos de la Config

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `save_checkpoints` | `True` | cada paso guarda sus tablas |
| `checkpoint_folder` | `None` (= `<output_folder>/checkpoints`) | dónde |

## Uso

En el notebook:

```python
import importlib, pipeline, main
importlib.reload(pipeline); importlib.reload(main)

configuration = main.production_configuration()
orchestrator = pipeline.Orchestrator(configuration)
results = orchestrator.run()                  # la primera vez: todo, y queda guardado
results = orchestrator.run_from("forecast")   # después: desde el forecast, cargando lo anterior
orchestrator.order                            # los nombres de los pasos, en orden
```

Desde la línea de comandos: `python main.py --desde forecast`.

Los nombres de los pasos, en orden: `read_raw`, `validate_raw`, `split_time_series`, `validate_values`, `calendar`,
`dimension_levels`, `fine_table`, `forecast_units`, `lookups`, `series_routes`, `support_bound`, `rate_series`,
`dimensions`, `ladder`, `credibility`, `compositions`, `dynamics`, `backtest`, `uplift`, `uplift_backtest`, `forecast`,
`series_exam`, `time_series`, `audit`, `core`, `validation`, `report`.

Un detalle: el paso que se reanuda y los posteriores se vuelven a guardar, así que el checkpoint queda siempre al día con
la última ejecución.

## pipeline.py

```diff
--- /tmp/pipeline_before_ckpt.py	2026-10-01 09:34:28.861409920 +0000
+++ pipeline.py	2026-10-01 09:35:21.648499293 +0000
@@ -16,16 +16,37 @@
                     Running a step again drops what the steps after it had produced: nothing stale can
                     be used (the notebook problem of an old object passed to a new step).
 
+CHECKPOINTS (save_checkpoints, on by default): every step saves the tables it writes to
+checkpoint_folder (one .pkl per table) and records itself in checkpoint_manifest.json: when it
+finished, the hash of its code, and the Config fields it changed (step 02b changes the mandatory
+dims). run_from(step) loads every table of the steps before it and runs from that step on.
+YOU decide when a full run is needed; the orchestrator only WARNS (it does not stop) when, since the
+checkpoint was saved:
+  - the code of a step that is loaded (not re-run) changed, or a shared module changed
+    (config.py, prediction.py, techniques.py, vocabulario.py);
+  - the Config changed (any field: the checkpoint was made with other parameters).
+A step that is missing from the checkpoint stops the run: it names the step to run first.
+
 In a notebook:
     pipeline = Orchestrator(configuration)
-    results = pipeline.run()                 # every step, in the order of their dependencies
+    results = pipeline.run()                 # every step, in the order of their dependencies (and saved)
+    results = pipeline.run_from("forecast")  # the steps before "forecast" loaded from disk; it and the rest run
+    pipeline.load_until("forecast")          # only load (to inspect the inputs of a step), run nothing
     pipeline.run_step("forecast")            # one step: its inputs must exist and be current
     pipeline.context["forecast"]             # any table of the run
+    pipeline.order                           # the steps, in the order they run (the names run_from takes)
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
+import dataclasses
+import hashlib
+import json
+import os
+import pickle
+import sys
 import time
 from dataclasses import dataclass, field
+from datetime import datetime
 from typing import Callable
 
 import pandas as pd
@@ -123,12 +144,15 @@
 
 @dataclass
 class Step:
-    """A step: the tables it reads and writes, and how it runs on the context."""
+    """A step: the tables it reads and writes, how it runs on the context, the module of its code
+    (its hash goes to the checkpoint) and the Config fields it changes while it runs (saved and restored)."""
     name: str
     label: str
     reads: tuple
     writes: tuple
     run: Callable
+    module: str = None
+    config_writes: tuple = ()
 
 
 @dataclass
@@ -177,13 +201,65 @@
         self.table_contracts = contracts(configuration)
         self.context = PipelineContext(configuration)
         self.producer_of = {table: step.name for step in self.steps.values() for table in step.writes}
+        self.folder = configuration.checkpoint_folder or os.path.join(configuration.output_folder or ".", "checkpoints")
+        self.manifest_path = os.path.join(self.folder, "checkpoint_manifest.json")
+        self.seconds = {}                                  # how long every step took in this run
 
     def run(self) -> dict:
-        """Every step, in the order of their dependencies; the results under their usual names."""
+        """Every step, in the order of their dependencies; the results under their usual names.
+        A full run starts a new checkpoint (the Config it runs with is recorded)."""
+        if self.configuration.save_checkpoints:
+            self.start_checkpoint()
         for step_name in self.order:
             self.run_step(step_name)
+        self.show_timings()
+        return self.context.results()
+
+    def run_from(self, step_name: str) -> dict:
+        """The steps before step_name loaded from the checkpoint; step_name and every step after it run
+        (and are saved again). Warns when the code or the Config changed since the checkpoint."""
+        self.load_until(step_name)
+        for name in self.order[self.order.index(step_name):]:
+            self.run_step(name)
+        self.show_timings()
         return self.context.results()
 
+    def show_timings(self) -> None:
+        """How long every step took in this run, the slowest first: where a checkpoint saves the most."""
+        if not self.seconds:
+            return
+        timings = pd.DataFrame({"step": list(self.seconds), "seconds": [round(value, 1) for value in self.seconds.values()]})
+        timings["share"] = (timings["seconds"] / max(timings["seconds"].sum(), 1e-9)).map("{:.0%}".format)
+        timings["run_from_saves_s"] = [round(sum(list(self.seconds.values())[:index]), 1) for index in range(len(timings))]
+        self.configuration.logger.doc(f"[pipeline] time per step ({sum(self.seconds.values()):.0f}s in total; run_from_saves_s: "
+                                      f"what run_from(step) skips, measured in this run):")
+        self.configuration.show_table(timings.sort_values("seconds", ascending=False).head(12).reset_index(drop=True))
+
+    def load_until(self, step_name: str) -> None:
+        """Load from the checkpoint the tables of every step before step_name (nothing runs)."""
+        if step_name not in self.steps:
+            raise ContractError(f"no step '{step_name}'; the steps are: {', '.join(self.order)}")
+        manifest = self.read_manifest()
+        before = self.order[:self.order.index(step_name)]
+        missing = [name for name in before if name not in manifest.get("steps", {})]
+        if missing:
+            raise ContractError(f"the checkpoint in {self.folder} lacks the steps {missing}: run from '{missing[0]}' "
+                                f"(or run() everything) first")
+        self.warn_about_changes(manifest, before)
+        started = time.time()
+        for name in before:
+            saved = manifest["steps"][name]
+            for table in self.steps[name].writes:
+                with open(os.path.join(self.folder, f"{table}.pkl"), "rb") as handle:
+                    self.context.tables[table] = pickle.load(handle)
+                self.context.producer[table] = name
+            for field_name, value in saved.get("config_writes", {}).items():
+                setattr(self.configuration, field_name, value)          # what the step had changed in the Config
+        last = manifest["steps"][before[-1]]["finished"] if before else "-"
+        self.configuration.logger.doc(f"[checkpoint] {len(before)} steps loaded from {self.folder} "
+                                      f"({before[0] if before else '-'} … {before[-1] if before else '-'}, saved {last}) "
+                                      f"in {time.time() - started:.1f}s · running from '{step_name}'")
+
     def run_step(self, step_name: str):
         """One step: its inputs must exist; what the later steps had produced is dropped (no stale tables)."""
         step = self.steps[step_name]
@@ -198,9 +274,63 @@
             raise ContractError(f"step '{step_name}' wrote {sorted(outputs)} but declares {sorted(step.writes)}")
         for table, value in outputs.items():
             self.context.put(table, value, step_name, self.table_contracts)
-        self.configuration.logger.debug(f"[pipeline] {step_name} ({step.label}) in {time.time() - started:.1f}s")
+        self.seconds[step_name] = time.time() - started
+        self.configuration.logger.debug(f"[pipeline] {step_name} ({step.label}) in {self.seconds[step_name]:.1f}s")
+        if self.configuration.save_checkpoints:
+            self.save_step(step, outputs, time.time() - started)
         return outputs
 
+    # ─── the checkpoint ───────────────────────────────────────────────────────────
+    def start_checkpoint(self) -> None:
+        """A new checkpoint: the Config of the run and the hash of the shared code; no step saved yet."""
+        os.makedirs(self.folder, exist_ok=True)
+        self.write_manifest({"created": datetime.now().isoformat(timespec="seconds"),
+                             "config": config_fingerprint(self.configuration, self.steps.values()),
+                             "shared_code": code_hash(SHARED_MODULES), "steps": {}})
+
+    def save_step(self, step: Step, outputs: dict, seconds: float) -> None:
+        """The tables of one step, and its line in the manifest."""
+        os.makedirs(self.folder, exist_ok=True)
+        for table, value in outputs.items():
+            with open(os.path.join(self.folder, f"{table}.pkl"), "wb") as handle:
+                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
+        manifest = self.read_manifest() or {"created": datetime.now().isoformat(timespec="seconds"),
+                                            "config": config_fingerprint(self.configuration, self.steps.values()),
+                                            "shared_code": code_hash(SHARED_MODULES), "steps": {}}
+        manifest["steps"][step.name] = {
+            "finished": datetime.now().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
+            "tables": list(step.writes), "code": code_hash([step.module] if step.module else []),
+            "config_writes": {field_name: getattr(self.configuration, field_name) for field_name in step.config_writes}}
+        self.write_manifest(manifest)
+
+    def read_manifest(self) -> dict:
+        if not os.path.exists(self.manifest_path):
+            return {}
+        with open(self.manifest_path, encoding="utf-8") as handle:
+            return json.load(handle)
+
+    def write_manifest(self, manifest: dict) -> None:
+        with open(self.manifest_path, "w", encoding="utf-8") as handle:
+            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
+
+    def warn_about_changes(self, manifest: dict, loaded_steps: list) -> None:
+        """Warnings (not errors): code of a loaded step, shared code or Config changed since the checkpoint."""
+        logger = self.configuration.logger
+        changed_steps = [name for name in loaded_steps
+                         if self.steps[name].module and manifest["steps"][name]["code"] != code_hash([self.steps[name].module])]
+        if changed_steps:
+            logger.warning(f"[checkpoint] the code of {changed_steps} changed since the checkpoint: their tables are loaded, "
+                           f"not recomputed. Run from '{changed_steps[0]}' if the change matters")
+        if manifest.get("shared_code") != code_hash(SHARED_MODULES):
+            logger.warning(f"[checkpoint] a shared module changed since the checkpoint ({', '.join(SHARED_MODULES)}): "
+                           f"the loaded steps used the old one")
+        now = config_fingerprint(self.configuration, self.steps.values())
+        differences = sorted(name for name in set(now) | set(manifest.get("config", {}))
+                             if now.get(name) != manifest.get("config", {}).get(name))
+        if differences:
+            logger.warning(f"[checkpoint] the Config changed since the checkpoint in {differences}: the loaded steps ran with "
+                           f"the old values")
+
     def downstream_of(self, step_name: str) -> list:
         """Every step that reads, directly or not, what this step writes."""
         reached, frontier = set(), set(self.steps[step_name].writes)
@@ -215,6 +345,37 @@
         return [name for name in self.order if name in reached]
 
 
+SHARED_MODULES = ["config", "prediction", "techniques", "vocabulario"]
+
+
+def code_hash(module_names: list) -> str:
+    """The hash of the source of some modules (an edit to any of them changes it)."""
+    digest = hashlib.sha256()
+    for module_name in module_names:
+        module = sys.modules.get(module_name) or __import__(module_name)
+        path = getattr(module, "__file__", None)
+        if path and os.path.exists(path):
+            with open(path, "rb") as handle:
+                digest.update(handle.read())
+    return digest.hexdigest()[:16]
+
+
+def config_fingerprint(configuration: Config, steps) -> dict:
+    """The Config as text, field by field, without what cannot be compared (the engine, the SQL text)
+    and without what the steps themselves change (restored from the checkpoint instead)."""
+    changed_by_steps = {field_name for step in steps for field_name in step.config_writes}
+    fingerprint = {}
+    for config_field in dataclasses.fields(configuration):
+        if config_field.name in changed_by_steps or config_field.name in ("sql_engine", "raw_extract_sql"):
+            continue
+        value = getattr(configuration, config_field.name)
+        try:
+            fingerprint[config_field.name] = json.dumps(value, sort_keys=True, default=str)
+        except (TypeError, ValueError):
+            fingerprint[config_field.name] = str(type(value))
+    return fingerprint
+
+
 def dependency_order(steps: list) -> list:
     """The steps ordered so that every table is written before it is read (the declared order breaks ties).
     Stops on a table nobody writes or on a cycle."""
@@ -322,51 +483,51 @@
                     "series_table", "series_estimate", "series_exam", "series_rate", "time_series", "uplift_backtest",
                     "uplift_cells", "uplift_verdict", "validation", "core")
     return [
-        Step("read_raw", "--", (), ("raw_extract",), read_raw),
+        Step("read_raw", "--", (), ("raw_extract",), read_raw, module="config"),
         Step("validate_raw", "00", ("raw_extract",), ("validated_extract",),
-             lambda context: {"validated_extract": validate_raw(context["raw_extract"], context.configuration)}),
-        Step("split_time_series", "20a", ("validated_extract",), ("validated_renewals", "time_series_rows"), split),
+             lambda context: {"validated_extract": validate_raw(context["raw_extract"], context.configuration)}, module="step_00_validate_raw"),
+        Step("split_time_series", "20a", ("validated_extract",), ("validated_renewals", "time_series_rows"), split, module="step_20_time_series"),
         Step("validate_values", "01", ("validated_renewals",), ("validated_raw",),
-             lambda context: {"validated_raw": validate_values(context["validated_renewals"], context.configuration)}),
+             lambda context: {"validated_raw": validate_values(context["validated_renewals"], context.configuration)}, module="step_01_validate_values"),
         Step("calendar", "02", ("validated_raw",), ("calendared_extract",),
-             lambda context: {"calendared_extract": apply_calendar(context["validated_raw"], context.configuration)}),
+             lambda context: {"calendared_extract": apply_calendar(context["validated_raw"], context.configuration)}, module="step_02_apply_calendar"),
         Step("dimension_levels", "02b", ("calendared_extract",), ("calendared_raw",),
-             lambda context: {"calendared_raw": apply_dimension_levels(context["calendared_extract"], context.configuration)}),
+             lambda context: {"calendared_raw": apply_dimension_levels(context["calendared_extract"], context.configuration)}, module="step_02b_dimension_levels", config_writes=("business_mandatory_dims",)),
         Step("fine_table", "03", ("calendared_raw",), ("fine_table",),
-             lambda context: {"fine_table": build_fine_table(context["calendared_raw"], context.configuration)}),
+             lambda context: {"fine_table": build_fine_table(context["calendared_raw"], context.configuration)}, module="step_03_fine_table"),
         Step("forecast_units", "04", ("fine_table",), ("forecast_units",),
-             lambda context: {"forecast_units": build_forecast_units(context["fine_table"], context.configuration)}),
+             lambda context: {"forecast_units": build_forecast_units(context["fine_table"], context.configuration)}, module="step_04_forecast_units"),
         Step("lookups", "05", ("fine_table", "forecast_units"), ("lookups",),
-             lambda context: {"lookups": build_lookups(context["fine_table"], context["forecast_units"], context.configuration)}),
+             lambda context: {"lookups": build_lookups(context["fine_table"], context["forecast_units"], context.configuration)}, module="step_05_lookups"),
         Step("series_routes", "06", ("forecast_units",), ("series_table",),
-             lambda context: {"series_table": build_series_routes(context["forecast_units"], context.configuration)}),
+             lambda context: {"series_table": build_series_routes(context["forecast_units"], context.configuration)}, module="step_06_series_routes"),
         Step("support_bound", "07", ("forecast_units",), ("support_bound",),
-             lambda context: {"support_bound": build_support_bound(context["forecast_units"], context.configuration)}),
-        Step("rate_series", "08", ("forecast_units", "series_table"), ("rated_units", "series_rate"), rate_series),
-        Step("dimensions", "09", ("series_rate", "lookups"), ("dimension_decision", "dimension_pairs"), dimensions),
+             lambda context: {"support_bound": build_support_bound(context["forecast_units"], context.configuration)}, module="step_07_support_bound"),
+        Step("rate_series", "08", ("forecast_units", "series_table"), ("rated_units", "series_rate"), rate_series, module="step_08_rate_series"),
+        Step("dimensions", "09", ("series_rate", "lookups"), ("dimension_decision", "dimension_pairs"), dimensions, module="step_09_dimensions"),
         Step("ladder", "10", ("rated_units", "series_rate", "lookups", "dimension_decision"), ("ladder",),
              lambda context: {"ladder": build_ladder_groups(context["rated_units"], context["series_rate"],
                                                             context["lookups"]["lookup_fs"], context["dimension_decision"],
-                                                            context.configuration)}),
-        Step("credibility", "11", ("series_rate", "ladder"), ("series_estimate", "money_by_level"), credibility),
-        Step("compositions", "12", ("rated_units", "ladder", "series_estimate"), ("pool_series", "pool_reference"), compositions),
-        Step("dynamics", "13", ("pool_series", "pool_reference"), ("pool_dynamics", "portfolio_profile", "portfolio_dynamics"), dynamics),
+                                                            context.configuration)}, module="step_10_ladder_groups"),
+        Step("credibility", "11", ("series_rate", "ladder"), ("series_estimate", "money_by_level"), credibility, module="step_11_ladder"),
+        Step("compositions", "12", ("rated_units", "ladder", "series_estimate"), ("pool_series", "pool_reference"), compositions, module="step_12_pool_series"),
+        Step("dynamics", "13", ("pool_series", "pool_reference"), ("pool_dynamics", "portfolio_profile", "portfolio_dynamics"), dynamics, module="step_13_dynamics"),
         Step("backtest", "14", ("pool_series", "pool_reference"), ("backtest",),
-             lambda context: {"backtest": run_backtest(context["pool_series"], context["pool_reference"], context.configuration)}),
-        Step("uplift", "15", ("fine_table",), ("uplift_cells", "contract_check"), uplift),
-        Step("uplift_backtest", "16", ("fine_table",), ("uplift_backtest", "uplift_verdict"), uplift_backtest),
+             lambda context: {"backtest": run_backtest(context["pool_series"], context["pool_reference"], context.configuration)}, module="step_14_backtest"),
+        Step("uplift", "15", ("fine_table",), ("uplift_cells", "contract_check"), uplift, module="step_15_uplift"),
+        Step("uplift_backtest", "16", ("fine_table",), ("uplift_backtest", "uplift_verdict"), uplift_backtest, module="step_16_uplift_backtest"),
         Step("forecast", "17", ("fine_table", "forecast_units", "series_estimate", "pool_series", "pool_reference", "backtest",
-                                "uplift_cells", "uplift_verdict", "rated_units"), ("forecast",), forecast),
-        Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("series_exam",), series_exam),
-        Step("time_series", "20", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series),
+                                "uplift_cells", "uplift_verdict", "rated_units"), ("forecast",), forecast, module="step_17_forecast"),
+        Step("series_exam", "19", ("rated_units", "series_estimate", "pool_series", "backtest", "ladder"), ("series_exam",), series_exam, module="step_19_series_exam"),
+        Step("time_series", "20", ("time_series_rows", "fine_table", "forecast"), ("time_series", "forecast_total"), time_series, module="step_20_time_series"),
         Step("audit", "AUD", ("ladder", "series_rate", "series_estimate", "rated_units", "pool_series", "pool_reference",
-                              "pool_dynamics", "backtest", "forecast", "series_exam"), ("audit",), audit),
+                              "pool_dynamics", "backtest", "forecast", "series_exam"), ("audit",), audit, module="step_audit"),
         Step("core", "NU", ("fine_table", "forecast_units", "support_bound", "rated_units", "series_table", "series_rate",
                             "series_estimate", "pool_dynamics", "backtest", "forecast", "time_series_rows", "time_series",
-                            "forecast_total", "ladder", "audit", "series_exam"), ("core", "core_legend", "forecast_series"), core),
+                            "forecast_total", "ladder", "audit", "series_exam"), ("core", "core_legend", "forecast_series"), core, module="step_nucleo"),
         Step("validation", "18", ("raw_extract", "backtest", "fine_table", "forecast", "forecast_units", "pool_reference",
                                   "series_exam", "series_estimate", "time_series_rows", "core"), ("validation",),
-             lambda context: {"validation": validate_chain(context["raw_extract"], context.results(), context.configuration)}),
+             lambda context: {"validation": validate_chain(context["raw_extract"], context.results(), context.configuration)}, module="step_18_validation"),
         Step("report", "IN", report_reads, ("card",),
-             lambda context: {"card": build_report(context["raw_extract"], context.results(), context.configuration)}),
+             lambda context: {"card": build_report(context["raw_extract"], context.results(), context.configuration)}, module="step_informe"),
     ]
```

## config.py

```diff
--- /tmp/config_before_ckpt.py	2026-10-01 09:34:28.866652844 +0000
+++ config.py	2026-10-01 09:34:28.951971488 +0000
@@ -199,6 +199,8 @@
     levels_path: Optional[str] = None                 # the JSON of the generated groups (None: <output_folder>/sff_levels.json);
                                                       # a later run reuses it; delete it to regenerate
     level_merge_max_pp: float = 5.0                   # two neighbouring values merge while their rates differ by at most this
+    save_checkpoints: bool = True                     # every step saves its tables, so a later run can start from any step
+    checkpoint_folder: Optional[str] = None           # where (None: <output_folder>/checkpoints); one .pkl per table + manifest
     structural_timevarying_dims: dict = field(default_factory=dict)   # column → "negative" | "positive"
     extra_renovacion: list = field(default_factory=list)              # enter the rate series only
     extra_revalorizacion: list = field(default_factory=list)          # enter the uplift cell only
```

## main.py

```diff
--- /mnt/user-data/outputs/sff_espejo/main.py	2026-10-01 08:39:22.696387000 +0000
+++ main.py	2026-10-01 09:34:56.748303449 +0000
@@ -121,13 +121,16 @@
 # THE RUN
 # ═══════════════════════════════════════════════════════════════════════════════════
 
-def run(configuration: Config) -> dict:
-    """Every step, in the order of their dependencies (pipeline.py: contracts, step interface, orchestrator)."""
-    return Orchestrator(configuration).run()
+def run(configuration: Config, from_step: str = None) -> dict:
+    """Every step, in the order of their dependencies (pipeline.py: contracts, step interface, orchestrator).
+    With from_step, the steps before it are loaded from the last checkpoint and it and the rest run."""
+    orchestrator = Orchestrator(configuration)
+    return orchestrator.run_from(from_step) if from_step else orchestrator.run()
 
 
 if __name__ == "__main__":
-    if SYNTHETIC_KEYWORD in sys.argv[1:]:
-        run(synthetic_configuration())
-    else:
-        run(production_configuration())
+    # python main.py [sintetico] [--desde <step>]    e.g. python main.py --desde forecast
+    arguments = sys.argv[1:]
+    from_step = arguments[arguments.index("--desde") + 1] if "--desde" in arguments else None
+    configuration = synthetic_configuration() if SYNTHETIC_KEYWORD in arguments else production_configuration()
+    run(configuration, from_step=from_step)
```

## test_pipeline.py

```diff
--- /mnt/user-data/outputs/sff_espejo/test_pipeline.py	2026-10-01 08:39:23.873448000 +0000
+++ test_pipeline.py	2026-10-01 09:35:44.267157485 +0000
@@ -1,11 +1,18 @@
 """
 test_pipeline.py — The orchestrator: steps ordered by their dependencies, a step refused without its
-inputs, nothing stale after a step runs again, and a table that breaks its contract stopped at once.
+inputs, nothing stale after a step runs again, a table that breaks its contract stopped at once, and
+the checkpoints: run_from a step gives the same result without re-running the steps before it, and
+warns when the code or the Config changed since the checkpoint.
 
     python test_pipeline.py
 """
 
 # ─── imports ─────────────────────────────────────────────────────────────────────
+import json
+import os
+import re
+import tempfile
+
 import pandas as pd
 
 from pipeline import ContractError, Orchestrator, Step, TableContract, dependency_order, sff_steps
@@ -47,8 +54,59 @@
                 "lacks the columns", "a missing column is refused, naming it")
 
 
+def total_2026(results) -> float:
+    total = results["forecast_total"]
+    return float(total.loc[(total["ano"] == 2026) & (total["origen"] == "TOTAL"), "usd_renovado"].item())
+
+
+def test_the_checkpoints() -> None:
+    print("D · checkpoints: run from any step, assuming the rest")
+    folder = tempfile.mkdtemp()
+    full = {}
+    console_of(lambda: full.update(Orchestrator(synthetic_with(checkpoint_folder=folder)).run()))
+    manifest = json.load(open(os.path.join(folder, "checkpoint_manifest.json"), encoding="utf-8"))
+    check(len(manifest["steps"]) == len(sff_steps()) and os.path.exists(os.path.join(folder, "fine_table.pkl")),
+          "a full run saves every step: its tables (.pkl) and its line in the manifest")
+    resumed = {}
+    console = console_of(lambda: resumed.update(Orchestrator(synthetic_with(checkpoint_folder=folder)).run_from("forecast")))
+    check("STEP 03" not in console and "STEP 17" in console and "steps loaded" in console,
+          "run_from('forecast') loads the steps before it and runs only it and the rest")
+    check(abs(total_2026(resumed) - total_2026(full)) < 0.01, "and gives the same forecast as the full run")
+    check("changed since the checkpoint" not in console, "no warning when nothing changed")
+    console = console_of(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder, collapse_passes=3)).run_from("forecast"))
+    check("the Config changed since the checkpoint in ['collapse_passes']" in console,
+          "a Config that changed since the checkpoint is warned, naming the field")
+    manifest["steps"]["ladder"]["code"] = "something-else"
+    json.dump(manifest, open(os.path.join(folder, "checkpoint_manifest.json"), "w", encoding="utf-8"))
+    console = console_of(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder)).load_until("forecast"))
+    check("the code of ['ladder'] changed since the checkpoint" in console,
+          "a loaded step whose code changed is warned, with the step to run from")
+    del manifest["steps"]["backtest"]
+    json.dump(manifest, open(os.path.join(folder, "checkpoint_manifest.json"), "w", encoding="utf-8"))
+    check_stops(lambda: Orchestrator(synthetic_with(checkpoint_folder=folder)).run_from("forecast"),
+                "run from 'backtest'", "a step missing from the checkpoint stops the run, naming the step to run first")
+
+
+def test_what_a_step_changes_in_the_config() -> None:
+    print("E · what a step changes in the Config is saved and restored (step 02b: the mandatory dims)")
+    folder, levels = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "levels.json")
+    options = dict(checkpoint_folder=folder, leveled_dims={"product": {"type": "nominal"}}, levels_path=levels)
+    full = {}
+    console_of(lambda: full.update(Orchestrator(synthetic_with(**options)).run()))
+    configuration = synthetic_with(**options)
+    check(configuration.business_mandatory_dims == ["region", "product"], "a new Config starts with the declared dims")
+    resumed = {}
+    console = console_of(lambda: resumed.update(Orchestrator(configuration).run_from("ladder")))
+    check(configuration.business_mandatory_dims == ["region", "product_level_1", "product_level_2"]
+          and "changed since the checkpoint" not in console,
+          "resuming after step 02b restores the dims it had set, without a false warning")
+    check(abs(total_2026(resumed) - total_2026(full)) < 0.01, "and the forecast is the same as the full run")
+
+
 if __name__ == "__main__":
     test_the_order()
     test_one_step_at_a_time()
     test_the_contracts()
+    test_the_checkpoints()
+    test_what_a_step_changes_in_the_config()
     finish()
```
