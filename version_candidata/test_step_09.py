"""
test_step_09.py — Step 09: the figures of every dimension and the collapse order.

    python test_step_09.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_05_lookups import build_lookups
from step_08_rate_series import build_rate_series
from step_09_dimensions import (analyse_dimensions, sequential_collapse_order, weighted_eta2, weighted_omega2,
                                weighted_r2)
from test_helpers import check, console_of, count_status, finish, synthetic_with
from test_step_08 import inputs_synthetic
from test_step_05 import units_synthetic


def test_the_statistics() -> None:
    print("A · η², ω², R² and the collapse order on a frame built by hand")
    frame = pd.DataFrame({"region": ["EU", "EU", "NA", "NA", "EU", "NA"],
                          "channel": ["web", "shop", "web", "shop", "web", "shop"],
                          "tasa_propia": [0.8, 0.8, 0.4, 0.4, 0.8, 0.4], "n_propio": [10, 10, 10, 10, 10, 10]})
    check(abs(weighted_eta2(frame, "region") - 1.0) < 1e-12 and weighted_eta2(frame, "channel") < 0.2,
          "region explains the whole rate (η² = 1); channel almost nothing")
    check(weighted_omega2(frame, "region") <= weighted_eta2(frame, "region"), "ω² is never above η²")
    check(abs(weighted_r2(frame, ["region", "channel"], synthetic_with().dimension_min_series) - 1.0) < 1e-9, "the additive model with region fits exactly")
    order = sequential_collapse_order(frame, ["region", "channel"], synthetic_with())
    check([dimension for dimension, _ in order] == ["channel", "region"],
          "the dimension that loses the least R² (channel) collapses first")

    nested = pd.DataFrame({"product_level_1": ["A", "A", "A", "A", "B", "B"],
                           "product_level_2": ["A1", "A1", "A2", "A2", "B1", "B1"],
                           "tasa_propia": [0.9, 0.9, 0.5, 0.5, 0.3, 0.3], "n_propio": [5] * 6})
    order = sequential_collapse_order(nested, ["product_level_1", "product_level_2"], synthetic_with())
    check([dimension for dimension, _ in order] == ["product_level_2", "product_level_1"],
          "in a hierarchy the finer level collapses first, whatever the loss")


def test_the_step() -> None:
    print("B · the step on the synthetic")
    units, series, configuration = inputs_synthetic()
    frames = {}
    console_of(lambda: frames.setdefault("rate", build_rate_series(units, series, configuration)))
    _, series_rate = frames["rate"]
    fine, _, _ = units_synthetic()
    console_of(lambda: frames.setdefault("lookups", build_lookups(fine, units, configuration)))
    console = console_of(lambda: frames.setdefault("dims", analyse_dimensions(series_rate, frames["lookups"]["lookup_fs"],
                                                                              configuration)))
    decision, pairs = frames["dims"]
    check(set(decision["dimension"]) == {"region", "product", "channel"}, "mandatory and extra_renovacion measured")
    check(sorted(decision.loc[decision["grupo"] == "mandatory", "orden_colapso"]) == [1, 2]
          and decision.loc[decision["dimension"] == "channel", "orden_colapso"].item() == 0,
          "the two mandatory dims get positions 1 and 2; the extra gets 0 (it is annulled, not collapsed)")
    check(count_status(console, "ok") == 6 and "6 checks: 6 ok" in console, "the 6 checks pass and are logged")
    check("collapse order:" in console, "the collapse order is logged")
    written = pd.read_sql("SELECT * FROM sff_decision_eta2", configuration.sql_engine)
    check(len(written) == 3, "sff_decision_eta2 written")


if __name__ == "__main__":
    test_the_statistics()
    test_the_step()
    finish()
