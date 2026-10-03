"""
table_equivalence.py — Proves that a change did not touch the logic: every table the synthetic run
returns comes out identical, bit for bit, before and after the change.

    python table_equivalence.py save    before.pkl     # before touching the code
    python table_equivalence.py save    after.pkl      # after the change
    python table_equivalence.py compare before.pkl after.pkl

A change that does not give n identical tables of n is discarded, not "fixed" with a tolerance.
The synthetic does not walk every branch (e.g. a technique without its own band): a change to a lookup or
a loop also needs a direct test of the branches the synthetic misses (see test_prediction.py, part C).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import pickle
import sys

import pandas as pd

from main import run
from test_helpers import synthetic_with


def every_table(value, path: str, tables: dict) -> None:
    """Every DataFrame (or Series) in the results, nested dicts included, by its path ("forecast.forecast")."""
    if isinstance(value, pd.DataFrame):
        tables[path] = value
    elif isinstance(value, pd.Series):
        tables[path] = value.to_frame()
    elif isinstance(value, dict):
        for key, inner_value in value.items():
            every_table(inner_value, f"{path}.{key}" if path else str(key), tables)


def save_tables(file_path: str) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        results = run(synthetic_with())
    tables = {}
    every_table(results, "", tables)
    with open(file_path, "wb") as handle:
        pickle.dump(tables, handle)
    print(f"{len(tables)} tables saved to {file_path}")


def compare_tables(before_path: str, after_path: str) -> bool:
    with open(before_path, "rb") as handle:
        before = pickle.load(handle)
    with open(after_path, "rb") as handle:
        after = pickle.load(handle)
    names = sorted(set(before) | set(after))
    different = []
    for name in names:
        if name not in before or name not in after:
            different.append(f"{name}: only in one of the two runs")
        elif not before[name].reset_index(drop=True).equals(after[name].reset_index(drop=True)):
            different.append(f"{name}: differs")
    print(f"{len(names) - len(different)} of {len(names)} tables identical")
    for line in different:
        print(f"   {line}")
    return not different


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save_tables(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(0 if compare_tables(sys.argv[2], sys.argv[3]) else 1)
    else:
        print(__doc__)
