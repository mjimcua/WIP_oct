"""
test_step_02b.py — The dimensions with generated levels: level_1 grouped by the standardised rate
(training months), level_2 raw; the JSON written by the first run and reused by the next.

    python test_step_02b.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import tempfile

import pandas as pd

from main import run
from step_02b_dimension_levels import natural_key, replace_in_mandatory
from test_helpers import check, console_of, finish, synthetic_with


def test_the_levels() -> None:
    print("A · product as a leveled dimension: generated, then reused")
    path = os.path.join(tempfile.mkdtemp(), "sff_levels.json")
    first = {}
    configuration = synthetic_with(leveled_dims={"product": {"type": "nominal"}}, levels_path=path)
    console = console_of(lambda: first.update(run(configuration)))
    check(os.path.exists(path) and "not found, generating ['product']" in console,
          "the first run generates the groups and writes the JSON")
    stored = json.load(open(path, encoding="utf-8"))["dims"]["product"]
    check(set(stored["mapping"]) == {"A", "B"} and all("rate_by_year" in group for group in stored["groups"]),
          "the JSON maps every value to its group, with the evidence (rate per year) of every group")
    check(configuration.business_mandatory_dims == ["region", "product_level_1", "product_level_2"],
          "the mandatory dims get the two levels in the place of the dimension")
    check({"product_level_1", "product_level_2"} <= set(first["fine_table"].columns), "the rows carry both levels")
    check("4 checks: 4 ok" in console.split("STEP 02b")[1], "the 4 checks of step 02b pass")
    second = {}
    console = console_of(lambda: second.update(run(synthetic_with(leveled_dims={"product": {"type": "nominal"}}, levels_path=path))))
    check("every dimension in it, reused" in console, "the next run finds the JSON and reuses it")
    check(set(first["ladder"]["groups"]["composition_id"]) == set(second["ladder"]["groups"]["composition_id"]),
          "the forecast series keep their compositions from one run to the next")


def test_the_rules() -> None:
    print("B · the rules: ordinal order, the place in the mandatory dims")
    check(sorted(["10 devices", "2 devices", "1 device"], key=natural_key) == ["1 device", "2 devices", "10 devices"],
          "an ordinal dimension is ordered by its numbers, not as text")
    from types import SimpleNamespace
    configuration = SimpleNamespace(business_mandatory_dims=["region", "tr_band_level_1", "tr_band_level_2", "partner"])
    replace_in_mandatory(configuration, "tr_band_level_2", "tr_band", "tr_band_level_1", "tr_band_level_2")
    check(configuration.business_mandatory_dims == ["region", "tr_band_level_1", "tr_band_level_2", "partner"],
          "an extract that already has level_1 and level_2 keeps them in their place (level_1 is overwritten)")


if __name__ == "__main__":
    test_the_levels()
    test_the_rules()
    finish()
