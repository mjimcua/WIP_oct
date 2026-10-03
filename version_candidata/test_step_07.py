"""
test_step_07.py — Step 07: the worst-case binomial bound of every forecast unit.

    python test_step_07.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_07_support_bound import build_support_bound
from test_helpers import check, console_of, count_status, finish
from test_step_05 import units_synthetic


def test_the_bound() -> None:
    print("A · the bound of every unit")
    _, units, configuration = units_synthetic()
    frames = {}
    console = console_of(lambda: frames.setdefault("bound", build_support_bound(units, configuration)))
    bound = frames["bound"]
    first = bound.iloc[0]
    expected_se = 100 * np.sqrt(0.25 / first["total_tr_units"])
    check(abs(first["se_pp_max"] - expected_se) < 1e-9 and abs(first["moe_pp_max"] - 1.645 * expected_se) < 1e-9,
          "se_pp_max = 100·√(0.25/n) and moe_pp_max = z · se_pp_max")
    check(abs(first["moe_usd_max"] - first["moe_pp_max"] / 100 * first["total_tr_usd"]) < 1e-9,
          "moe_usd_max = moe_pp_max / 100 × USD due")
    small, large = bound.nsmallest(1, "total_tr_units").iloc[0], bound.nlargest(1, "total_tr_units").iloc[0]
    check(small["moe_pp_max"] > large["moe_pp_max"], "the smaller the support, the wider the bound")
    check(len(bound) == len(units) and count_status(console, "ok") == 2, "one row per unit, the 2 checks pass")
    check(pd.read_sql("SELECT COUNT(*) AS n FROM sff_fu_soporte", configuration.sql_engine)["n"].item() == len(units),
          "sff_fu_soporte written")

    nothing_due = units.copy()
    nothing_due.loc[0, "total_tr_units"] = 0
    console_of(lambda: frames.update(bound=build_support_bound(nothing_due, configuration)))
    check(abs(frames["bound"].iloc[0]["se_pp_max"] - 50.0) < 1e-9, "a unit with nothing due is bounded as n = 1: ±50 pp of se")


if __name__ == "__main__":
    test_the_bound()
    finish()
