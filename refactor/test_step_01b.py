"""
test_step_01b.py — The leveled dimensions: the column keeps its raw value and the library adds
<column>_level_1 (its values grouped by the standardised rate of the training months), before the
calendar; the JSON written by the first run and reused by the next; the ladder collapses the fine
level before the generated one.

    python test_step_01b.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import tempfile

import pandas as pd

from config import roles_overview
from main import run
from step_01b_dimension_levels import natural_key
from step_09_dimensions import family_and_level
from test_helpers import check, check_stops, console_of, finish, synthetic_with


def leveled(**overrides):
    return synthetic_with(leveled_dims={"product": "nominal"}, levels_path=os.path.join(tempfile.mkdtemp(), "levels.json"),
                          **overrides)


def test_the_config() -> None:
    print("A · the Config knows the generated level from the start; nothing changes it later")
    configuration = leveled()
    check(configuration.business_mandatory_dims == ["region", "product", "product_level_1"],
          "the generated level is a mandatory dim right after its column, when the Config is built")
    check(synthetic_with(**{name: getattr(configuration, name) for name in ("leveled_dims", "business_mandatory_dims")})
          .business_mandatory_dims == ["region", "product", "product_level_1"],
          "building the Config again from its own fields gives the same dims (no duplicates)")
    check_stops(lambda: synthetic_with(leveled_dims={"channel": "nominal"}), "must be mandatory",
                "a leveled dim that is not mandatory stops when the Config is built")
    check_stops(lambda: synthetic_with(leveled_dims={"product": "alphabetical"}), "must be one of",
                "an unknown type stops when the Config is built")


def test_the_levels() -> None:
    print("B · product as a leveled dimension: generated, then reused")
    configuration = leveled()
    first = {}
    console = console_of(lambda: first.update(run(configuration)))
    check("not found, generating ['product']" in console and "STEP 01b" in console.split("STEP 02 ")[0],
          "the first run generates the groups, before the calendar")
    stored = json.load(open(configuration.levels_path, encoding="utf-8"))["dims"]["product"]
    check(set(stored["mapping"]) == {"A", "B"} and all("rate_by_year" in group for group in stored["groups"]),
          "the JSON maps every value to its group, with the evidence (rate per year) of every group")
    fine = first["fine_table"]
    check({"product", "product_level_1"} <= set(fine.columns) and "product_level_2" not in fine.columns
          and set(fine["product"]) == {"A", "B"},
          "the column keeps its name and its raw value; only product_level_1 is added")
    check(configuration.business_mandatory_dims == ["region", "product", "product_level_1"],
          "no step changed the Config")
    order = list(first["dimension_decision"].sort_values("orden_colapso")["dimension"]) if "orden_colapso" in first["dimension_decision"] else []
    if order:
        check(order.index("product") < order.index("product_level_1"), "the ladder collapses the raw value before its generated level")
    second = {}
    console = console_of(lambda: second.update(run(synthetic_with(leveled_dims={"product": "nominal"},
                                                                  levels_path=configuration.levels_path))))
    check("every dimension in it, reused" in console, "the next run finds the JSON and reuses it")
    check(set(first["ladder"]["groups"]["composition_id"]) == set(second["ladder"]["groups"]["composition_id"]),
          "the forecast series keep their compositions from one run to the next")


def test_the_rules() -> None:
    print("C · the rules: the family of a leveled dim, the ordinal order, the role table")
    dims = ["region", "product", "product_level_1", "tr_regional_level_1", "tr_regional_level_2"]
    check(family_and_level("product", dims) == ("product", 2) and family_and_level("product_level_1", dims) == ("product", 1)
          and family_and_level("region", dims) == ("region", 1) and family_and_level("tr_regional_level_2", dims) == ("tr_regional", 2),
          "a column X is the finest level of its family when X_level_N exist; the other hierarchies as before")
    check(sorted(["10 devices", "2 devices", "1 device"], key=natural_key) == ["1 device", "2 devices", "10 devices"],
          "an ordinal dimension is ordered by its numbers, not as text")
    configuration = leveled(extra_revalorizacion=[])
    raw = configuration.read_raw()
    before = roles_overview(raw.columns, configuration).set_index("rol")
    check(before.loc["extra_revalorizacion", "columnas"] == 0 and "product → product_level_1 (nominal)"
          in before.loc["niveles generados (01b)", "nombres"],
          "the role table lists every role (also an empty one) and the dims that get a generated level")
    after = roles_overview(list(raw.columns) + ["product_level_1"], configuration).set_index("rol")
    check(after.loc["mandatory", "nombres"] == "region, product, product_level_1",
          "with the level made, the mandatory role lists it after its column")


def test_the_merge_threshold_and_its_evidence() -> None:
    print("D · the merge threshold comes from the support floor; the evidence is on screen")
    import numpy as np
    import re
    configuration = leveled()
    results = {}
    console = re.sub(r"\x1b\[[0-9;]*m", "", console_of(lambda: results.update(run(configuration))))
    stored = json.load(open(configuration.levels_path, encoding="utf-8"))["dims"]["product"]
    rate = stored["criterion"]["global_rate"]
    check(abs(stored["criterion"]["threshold_pp"] - round(100 * np.sqrt(rate * (1 - rate) / configuration.support_floor), 2)) < 0.01
          and "the binomial noise of a series at the support floor" in console,
          "by default, two values merge while they differ by less than the binomial noise of a series at the floor")
    check(all("noise_pp" in value and value["rate_by_year"] for value in stored["values"]) and stored["decisions"],
          "the JSON keeps every value (its rate, its noise, its years) and every merge decision")
    check("what a difference in pp means" in console and "rate_2024" in console and "within_5_pp" in console,
          "on screen: the noise per size of series, the rate per year of every group and value, and the decisions")
    console = console_of(lambda: run(synthetic_with(leveled_dims={"product": "nominal"}, levels_path=configuration.levels_path,
                                                    level_merge_max_pp=20)))
    check("generating ['product']" in console, "a JSON made with another threshold is generated again")
    check("ONE group" in console and "no variation, not a pass of the ladder: ['product_level_1']" in console,
          "a dimension that ends in one group is told on screen and gets no pass of the ladder")


if __name__ == "__main__":
    test_the_config()
    test_the_levels()
    test_the_rules()
    test_the_merge_threshold_and_its_evidence()
    finish()
