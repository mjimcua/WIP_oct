"""
test_step_01.py — Step 01 on the synthetic raw: the two universes, the values and the dimensions with a
generated level. It passes as generated; each blocking problem of the values stops the run naming itself;
the generated level is made before the calendar, kept in a JSON and reused.

    python test_step_01.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import tempfile

import numpy as np

from config import roles_overview
from config import (CORE_ISOLATED_PIPELINE_UNITS, CORE_ISOLATED_RENEWED_PIPELINE_USD, CORE_ISOLATED_RENEWED_UNITS,
                    CORE_ISOLATED_RENEWED_USD, CORE_RENEWED_PIPELINE_USD)
from main import run
from step_00_validate_raw import validate_raw
from step_01_values_and_levels import natural_key, validate_values_and_build_levels
from step_09_dimensions import family_and_level
from test_helpers import check, check_stops, console_of, count_status, finish, synthetic_with


def validate_values(raw, configuration):
    """Step 01 on a raw already validated by step 00; returns the renewal raw (the first of its two outputs)."""
    renewal_raw, time_series_rows = validate_values_and_build_levels(raw, configuration)
    return renewal_raw


def validated_synthetic():
    """The synthetic raw after step 00, and its Config."""
    configuration = synthetic_with()
    validated = {}
    console_of(lambda: validated.setdefault("raw", validate_raw(configuration.read_raw(), configuration)))
    return validated["raw"], configuration


def with_value(raw, row_filter, column_name, new_value):
    """A copy of the raw with column_name set to new_value in the first row of row_filter."""
    changed = raw.copy()
    first_row = changed.index[row_filter(changed)][0]
    if isinstance(new_value, str) and changed[column_name].dtype != object:
        changed[column_name] = changed[column_name].astype(object)    # a text value in a numeric column
    changed.loc[first_row, column_name] = new_value
    return changed


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_synthetic_values_are_usable() -> None:
    print("A · the synthetic raw is usable as generated")
    raw, configuration = validated_synthetic()
    console = console_of(lambda: validate_values(raw, configuration))
    check("0 failed" in console, "the synthetic passes step 01")
    check("closed months:" in console, "the report shows the money of the closed months")
    check(count_status(console, "ok") == 49 and "49 checks: 49 ok · 0 warnings · 0 failed" in console,
          "the 49 checks are logged, numbered, each with its status, and counted (one header, one count)")
    check("purpose:" in console and "STEP 01 · VALUES AND DIMENSION LEVELS" in console, "the step logs its name and its purpose")
    returned = {}
    console_of(lambda: returned.setdefault("outputs", validate_values_and_build_levels(raw, configuration)))
    renewal_raw, time_series_rows = returned["outputs"]
    flagged = raw["flag_time_series"] == 1
    check(renewal_raw.equals(raw[~flagged]) and time_series_rows.equals(raw[flagged]),
          "without leveled dims, the two universes come back with their rows and columns unchanged")


def test_money() -> None:
    print("B · money")
    raw, configuration = validated_synthetic()
    closed = lambda frame: frame["period"] < configuration.calendar_boundaries()["current"]
    future = lambda frame: frame["period"] > configuration.calendar_boundaries()["current"]
    check_stops(lambda: validate_values(with_value(raw, future, "total_tr_units", -5), configuration),
                "negative values in total_tr_units", "a negative pipeline stops, even in the future")
    check_stops(lambda: validate_values(with_value(raw, future, "total_tr_usd", np.nan), configuration),
                "nulls in total_tr_usd", "a null pipeline stops, even in the future")
    null_renewal = with_value(with_value(raw, lambda f: closed(f) & (f["period"] == "2024-03"),
                                         "total_renewed_units", np.nan),
                              lambda f: closed(f) & (f["period"] == "2024-03"), "total_renewed_usd", np.nan)
    console = console_of(lambda: validate_values(null_renewal, configuration))
    check(count_status(console, "WARN") == 1 and "1 rows null" in console and "2024-03" in console,
          "a null renewal in a closed month is a warning, counted, with the row shown as an example")
    example_lines = [line for line in console.splitlines() if "2024-03" in line and "NaN" in line]
    check(len(example_lines) == 1 and " | [01]" not in example_lines[0] and "example rows below" in console,
          "the example row is shown as a table under the check line, not as a log line")
    console = console_of(lambda: validate_values(raw, configuration))
    check("0 rows null" in console and count_status(console, "WARN") == 0,
          "the synthetic sends no renewal as 0: the check passes and still shows example rows")
    check_stops(lambda: validate_values(with_value(raw, closed, "total_renewed_usd", np.nan), configuration),
                "one of the two renewal columns null", "units present with USD null (or the reverse) stops")
    free = with_value(raw, lambda f: closed(f) & (f["total_renewed_units"] > 0), "total_renewed_usd", 0.0)
    check_stops(lambda: validate_values(free, configuration), "renewing for free", "a renewal for free stops")
    usd_only = with_value(raw, lambda f: closed(f) & (f["total_renewed_units"] > 0), "total_renewed_units", 0.0)
    check_stops(lambda: validate_values(usd_only, configuration), "USD with 0 renewed units",
                "renewed USD without renewed units stops")
    above = with_value(raw, lambda f: closed(f) & (f["total_tr_units"] > 0), "total_renewed_units", 10_000.0)
    console = console_of(lambda: validate_values(above, configuration))
    check("WARN" in console and "rate above 100 %" in console, "renewing more units than fall due is a warning")


def test_dimensions_and_flags() -> None:
    print("C · dimensions, flags and discount")
    raw, configuration = validated_synthetic()
    everywhere = lambda frame: frame.index >= 0
    check_stops(lambda: validate_values(with_value(raw, everywhere, "region", None), configuration),
                "'region': 1", "a null dimension stops")
    check_stops(lambda: validate_values(with_value(raw, everywhere, "channel", "  "), configuration),
                "'channel': 1", "an empty text dimension stops")
    check_stops(lambda: validate_values(with_value(raw, everywhere, "dormant", 2), configuration),
                "dormant must be 0 / 1", "a timevarying value other than 0 / 1 stops")
    check_stops(lambda: validate_values(with_value(raw, everywhere, "flag_time_series", "yes"), configuration),
                "flag_time_series must be 0 / 1", "a time_series flag other than 0 / 1 stops")
    check_stops(lambda: validate_values(with_value(raw, everywhere, "discount_pct", 25.0), configuration),
                "outside [0, 1]", "a discount written as a percentage (25) stops")
    doubled = raw.copy()
    doubled.loc[len(doubled)] = doubled.iloc[0]
    console = console_of(lambda: validate_values(doubled, configuration))
    check("double load" in console, "a row repeated in every column is a warning")
    two_problems = with_value(with_value(raw, everywhere, "region", None), everywhere, "dormant", 2)
    check_stops(lambda: validate_values(two_problems, configuration), "2. ", "every problem is reported at once")


def test_the_weight_in_money() -> None:
    print("Z · every flag and the unknown discount: the share of rows and the share of the USD due")
    console = console_of(lambda: run(synthetic_with()))
    check("of rows at 1 · " in console and "of the USD due" in console and "of rows unknown · " in console,
          "the flags and the unknown discount say how much money they weigh, not only how many rows")

def test_the_universes() -> None:
    print("D · the two universes are separated first, in step 01")
    raw, configuration = validated_synthetic()
    console = console_of(lambda: validate_values(raw, configuration))
    check("time_series rows (retail to subscription) out of" in console
          and "every time_series row has every region level" in console,
          "action 1 separates the time_series rows and checks their regions")
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    check(len(results["time_series_rows"]) == 48 and not (results["fine_table"]["flag_time_series"] == 1).any(),
          "the flagged rows leave the raw in step 01: none reaches the fine table")
    check("[00b]" not in console and "20a" not in console and "01b" not in console,
          "no step carries a letter: the separation and the levels are part of step 01")


def leveled(**overrides):
    return synthetic_with(leveled_dims={"product": "nominal"}, levels_path=os.path.join(tempfile.mkdtemp(), "levels.json"),
                          **overrides)


def test_the_config() -> None:
    print("E · the Config knows the generated level from the start; nothing changes it later")
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
    print("F · product as a leveled dimension: generated, then reused")
    configuration = leveled()
    first = {}
    console = console_of(lambda: first.update(run(configuration)))
    check("not found, generating ['product']" in console.split("STEP 02 ")[0] and "STEP 01 ·" in console.split("STEP 02 ")[0],
          "the first run generates the groups in step 01, before the calendar")
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
    print("G · the rules: the family of a leveled dim, the ordinal order, the role table")
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
          in before.loc["niveles generados (01)", "nombres"],
          "the role table lists every role (also an empty one) and the dims that get a generated level")
    after = roles_overview(list(raw.columns) + ["product_level_1"], configuration).set_index("rol")
    check(after.loc["mandatory", "nombres"] == "region, product, product_level_1",
          "with the level made, the mandatory role lists it after its column")


def test_the_merge_threshold_and_its_evidence() -> None:
    print("H · the merge threshold comes from the support floor; the evidence is on screen")
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


def test_the_extra_renewal_measure() -> None:
    print("I · the closed-month measures (the exact uplift base and the isolated renewals), each by its Config field")
    raw, configuration = validated_synthetic()
    measure = configuration.isolated_renewed_usd_col
    closed = raw[configuration.period_col] < configuration.calendar_boundaries()["current"]
    negative = raw.copy()
    negative.loc[negative[closed].index[0], measure] = -5.0
    check_stops(lambda: validate_values(negative, configuration), f"{measure} has no negatives",
                "a negative value of the measure in a closed month stops the step")
    above = raw.copy()
    first_closed = above[closed & (above[configuration.renewed_usd_col] > 0)].index[0]
    above.loc[first_closed, measure] = above.loc[first_closed, configuration.renewed_usd_col] + 100.0
    console = console_of(lambda: validate_values(above, configuration))
    check(f"WARN  {measure} is not above {configuration.renewed_usd_col}" in console,
          "a value above the total renewed is a warning, not a stop")

    results = {}
    console_of(lambda: results.update(run(synthetic_with())))
    core = results["core"]
    final_column = CORE_ISOLATED_RENEWED_USD
    actual_rows = (core["forecast_status"] == "actual") & (core["origen_fila"] == "raw")
    check(core.loc[actual_rows, final_column].notna().all() and core.loc[~actual_rows, final_column].isna().all(),
          "in the core it is filled in the closed months of the extract and empty elsewhere (it is not predicted)")
    check((core.loc[actual_rows & (core["softcancel"] == 1), final_column] == 0).all()
          and (core.loc[actual_rows, final_column] <= core.loc[actual_rows, "forecast_renewed_USD"] + 0.01).all(),
          "it is 0 where the row is marked softcancel and never above the renewed USD")
    check({CORE_ISOLATED_PIPELINE_UNITS, CORE_ISOLATED_RENEWED_UNITS, CORE_ISOLATED_RENEWED_PIPELINE_USD} <= set(core.columns)
          and (core.loc[actual_rows, CORE_ISOLATED_PIPELINE_UNITS]
               <= core.loc[actual_rows, "forecast_to_renew_units"] + 0.01).all(),
          "the isolated measures reach the core with the framework's fixed names, whatever the extract calls them")
    wiped = results["fine_table"]
    future = wiped[configuration.period_col] >= configuration.calendar_boundaries()["current"]
    check(wiped.loc[future, measure].isna().all(), "from the current month on it is wiped, like the renewals")

    base_column = configuration.renewed_pipeline_usd_col
    with_base = raw.copy()
    first_renewed = with_base[closed & (with_base[configuration.renewed_units_col] > 0)].index[0]
    with_base.loc[first_renewed, base_column] = 0.0
    console = console_of(lambda: validate_values(with_base, configuration))
    check(f"WARN  {base_column} is above 0 where something renewed" in console,
          "a renewer row without the uplift base is a warning (step 15 leaves it out of the uplift)")
    broken_moment = raw.copy()
    with_isolated_base = broken_moment[closed & (broken_moment[configuration.isolated_renewed_pipeline_usd_col] > 0)].index[0]
    broken_moment.loc[with_isolated_base, configuration.isolated_renewed_usd_sq_over_tr_col] = 0.0
    console = console_of(lambda: validate_values(broken_moment, configuration))
    check(f"WARN  {configuration.isolated_renewed_usd_sq_over_tr_col} gives a variance ≥ 0" in console,
          "a second moment below (Σ renewed)² / Σ base (a negative variance) is a warning: not built per licence")
    broken_bands = raw.copy()
    broken_bands.loc[with_isolated_base, configuration.isolated_tr_usd_renewed_ge120_col] += 50.0
    console = console_of(lambda: validate_values(broken_bands, configuration))
    check("WARN  the 6 ratio bands add up to" in console,
          "six bands that do not add up to the isolated base are a warning (a licence in no band, or in two)")
    check(CORE_RENEWED_PIPELINE_USD in core.columns
          and (core.loc[actual_rows, CORE_RENEWED_PIPELINE_USD]
               <= core.loc[actual_rows, "forecast_to_renew_USD"] + 0.01).all(),
          "the exact uplift base reaches the core as forecast_to_renew_USD_renewed, never above the USD due")


if __name__ == "__main__":
    test_the_synthetic_values_are_usable()
    test_money()
    test_dimensions_and_flags()
    test_the_universes()
    test_the_config()
    test_the_levels()
    test_the_rules()
    test_the_merge_threshold_and_its_evidence()
    test_the_weight_in_money()
    test_the_extra_renewal_measure()
    finish()
