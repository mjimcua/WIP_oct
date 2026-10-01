"""
test_helpers.py — What every test_step_NN.py uses: the check recorder and the synthetic
Config with some fields changed.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import os
import re
import sys
import tempfile

from sqlalchemy import create_engine

from main import synthetic_configuration


# ─── the checks ──────────────────────────────────────────────────────────────────
passed_checks, failed_checks = [], []


def check(condition: bool, description: str) -> None:
    (passed_checks if condition else failed_checks).append(description)
    print(f"  {'ok  ' if condition else 'FAIL'} {description}")


def check_stops(function_under_test, expected_text: str, description: str) -> None:
    """The function must raise ValueError and its message must contain expected_text."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            function_under_test()
    except ValueError as error:
        check(expected_text in str(error), f"{description}  [{expected_text}]")
        return
    check(False, f"{description}  (did not stop)")


def console_of(function_under_test) -> str:
    """Run the function and return what it printed."""
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        function_under_test()
    return console.getvalue()


def count_status(console: str, status: str) -> int:
    """How many logged check lines carry this status ("ok", "FAIL", "WARN", "--")."""
    return len(re.findall(rf"\] +\d+\. {re.escape(status)}\s", console))


def console_of_failure(function_under_test) -> str:
    """Run a function that must stop, and return what it printed before stopping."""
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        try:
            function_under_test()
        except ValueError:
            pass
    return console.getvalue()


def synthetic_with(**overrides):
    """The synthetic Config with some fields changed; the log without colours (tests read it)
    and the tables in a SQLite file of its own, in a temporary folder."""
    configuration = synthetic_configuration()
    arguments = {name: getattr(configuration, name) for name in configuration.__dataclass_fields__}
    arguments["log_colors"] = False
    arguments["sql_engine"] = create_engine(f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'test.db')}")
    arguments.update(overrides)
    return type(configuration)(**arguments)


def finish() -> None:
    """Print the total and exit with 1 if any check failed."""
    print(f"\nTOTAL {len(passed_checks)}/{len(passed_checks) + len(failed_checks)} checks passed")
    sys.exit(1 if failed_checks else 0)
