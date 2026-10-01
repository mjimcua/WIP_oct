"""
test_step_01.py — Step 01 on the synthetic raw: it passes as generated, and each
blocking problem of the values stops the run naming itself.

    python test_step_01.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np

from step_00_validate_raw import validate_raw
from step_01_validate_values import validate_values
from test_helpers import check, check_stops, console_of, count_status, finish, synthetic_with


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
    check(count_status(console, "ok") == 20 and "20 checks: 20 ok · 0 warnings · 0 failed" in console,
          "the 20 checks are logged, numbered, each with its status, and counted")
    check("purpose:" in console and "STEP 01 · VALIDATE VALUES" in console, "the step logs its name and its purpose")
    returned = {}
    console_of(lambda: returned.setdefault("raw", validate_values(raw, configuration)))
    check(returned["raw"].equals(raw), "the raw comes back unchanged")


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


if __name__ == "__main__":
    test_the_synthetic_values_are_usable()
    test_money()
    test_dimensions_and_flags()
    finish()
