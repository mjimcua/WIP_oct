"""
test_step_10_11.py — Steps 10 and 11 on the synthetic: the mechanical rule of the ladder (every
series grouped by the same pattern at every pass; each uses the first pass whose group reaches the
floor; a big series keeps its own id but lends its history), every pass is a partition of the ids,
the sign is never mixed, every merge is recorded with its error alone and merged, stage 3 merges at
most collapse_passes dims, and the rate blends with the reference by credibility.

    python test_step_10_11.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_05_lookups import build_lookups
from step_08_rate_series import build_rate_series
from step_09_dimensions import analyse_dimensions
from step_10_ladder_groups import build_ladder_groups
from step_11_ladder import binomial_se_pp, climb_the_ladder
from test_helpers import check, console_of, finish
from test_step_05 import units_synthetic
from test_step_08 import inputs_synthetic


def inputs_of_the_ladder(**overrides):
    units, series, configuration = inputs_synthetic()
    frames = {}
    console_of(lambda: frames.setdefault("rate", build_rate_series(units, series, configuration)))
    rated_units, series_rate = frames["rate"]
    fine, _, _ = units_synthetic()
    console_of(lambda: frames.setdefault("lookups", build_lookups(fine, units, configuration)))
    console_of(lambda: frames.setdefault("dims", analyse_dimensions(series_rate, frames["lookups"]["lookup_fs"], configuration)))
    return rated_units, series_rate, frames["lookups"]["lookup_fs"], frames["dims"][0], configuration


def test_the_passes() -> None:
    print("A · step 10: the passes of the ladder")
    rated_units, series_rate, series_lookup, decision, configuration = inputs_of_the_ladder()
    frames = {}
    console = console_of(lambda: frames.setdefault("ladder", build_ladder_groups(rated_units, series_rate, series_lookup,
                                                                                 decision, configuration)))
    ladder = frames["ladder"]
    steps, summary, groups = ladder["steps"], ladder["summary"], ladder["groups"]
    check("9 checks: 9 ok" in console, "the 9 checks of step 10 pass")
    check(list(summary["step_name"][:2]) == ["itself", "sign"] and summary["step_name"].iloc[2].startswith("extra"),
          "the passes in order: itself → sign → extras → mandatory dims")
    check(summary["units_due"].nunique() == 1, "every pass is a partition: the units due add up to the same total")
    check((summary["groups"].diff().fillna(0) <= 0).all() and summary["groups"].iloc[-1] < summary["groups"].iloc[0],
          "the groups get fewer pass after pass")
    check(summary["median_group_support"].iloc[-1] > summary["median_group_support"].iloc[0],
          "and bigger: the median support grows")

    big = steps[(steps["fs_id"] == "EU|A|0|0|0|0|web")]
    check(big["group_id"].nunique() == 1 and (big["closed"] == 1).all(),
          "a series with support keeps its own id in every pass")
    negatives = ["EU|A|0|0|1|0|web", "EU|A|0|1|0|0|web", "EU|A|1|0|0|0|web", "EU|A|1|1|0|0|web"]
    at_sign = steps[(steps["ladder_step"] == 1) & steps["fs_id"].isin(negatives)]
    check(at_sign["group_id"].nunique() == 1 and at_sign["group_id"].iloc[0] == "EU|A|SIG=negativo|web"
          and (at_sign["closed"] == 1).all(),
          "the four small negative series merge at the sign pass into EU|A|SIG=negativo|web and close there")
    mixed = steps[steps["fs_id"] == "EU|B|1|0|0|1|web"]
    check(mixed["group_id"].nunique() == 1, "a mixed series is never merged")

    tele = groups.set_index("fs_id").loc["NA|A|0|0|0|0|tele"]
    check(tele["composition_id"] == "NA|A|SIG=neutro|*" and tele["group_series"] == 2 and tele["composition_users"] == 1,
          "the mechanical rule: NA|A|tele joins its sibling NA|A|web when the channel is removed (2 series in the rate, 1 uses it)")
    members = ladder["composition_members"]
    web_role = members[(members["composition_id"] == "NA|A|SIG=neutro|*") & (members["fs_id"] == "NA|A|0|0|0|0|web")]["role"]
    check(list(web_role) == ["lends"] and groups.set_index("fs_id").loc["NA|A|0|0|0|0|web", "composition_id"] == "NA|A|0|0|0|0|web",
          "NA|A|web predicts alone (own id) and lends its history to NA|A|tele's composition")
    merges = ladder["merges"]
    check(len(merges) > 0 and (merges["error_merged_pp"] < merges["error_alone_pp"]).all()
          and {"bias_pp", "error_alone_pp", "error_merged_pp", "improves"} <= set(merges.columns),
          "every merge is recorded with its bias and its error alone and merged; in the synthetic every merge improves")

    with_reference = groups.dropna(subset=["credibility_ref_id"])
    check((with_reference["ref_series"] > with_reference["group_series"]).all()
          and (with_reference["group_support"] < configuration.own_rate_floor).all(),
          "only groups below the own-rate floor take a reference, always wider than the group")
    closed_big = groups[groups["group_support"] >= configuration.own_rate_floor]
    check(closed_big["credibility_ref_id"].isna().all(), "a group with own precision takes no reference")


def test_the_collapse_limit() -> None:
    print("A2 · stage 3 merges at most collapse_passes dims; the later passes only give a reference")
    rated_units, series_rate, series_lookup, decision, configuration = inputs_of_the_ladder()
    configuration.collapse_passes = 0
    frames = {}
    console_of(lambda: frames.setdefault("ladder", build_ladder_groups(rated_units, series_rate, series_lookup, decision, configuration)))
    steps = frames["ladder"]["steps"]
    check(not steps["step_name"].str.startswith("without").any(),
          "with collapse_passes = 0 no mandatory dim is merged: the passes stop at the extras")
    groups = frames["ladder"]["groups"]
    references = groups.dropna(subset=["credibility_ref_id"])
    check(len(references) > 0 and (references["credibility_ref_step"] > references["final_step"]).all(),
          "a composition below the own-rate floor can still take its reference from a later (non-merging) pass")


def test_the_rate() -> None:
    print("B · step 11: the group lends its rate, blended with its reference by credibility")
    rated_units, series_rate, series_lookup, decision, configuration = inputs_of_the_ladder()
    frames = {}
    console_of(lambda: frames.setdefault("ladder", build_ladder_groups(rated_units, series_rate, series_lookup,
                                                                       decision, configuration)))
    console = console_of(lambda: frames.setdefault("estimate", climb_the_ladder(series_rate, frames["ladder"], configuration)))
    estimate = frames["estimate"][0]
    check("6 checks: 6 ok" in console, "the 6 checks of step 11 pass")

    big = estimate[estimate["fs_id"] == "EU|A|0|0|0|0|web"].iloc[0]
    check(big["z"] == 1.0 and abs(big["tasa_estimada"] - big["tasa_propia"]) < 1e-12 and big["nivel_riesgo"] == "A_propio",
          "a big series is its own group: z = 1, its own rate, A_propio")
    blended = estimate[estimate["z"] < 1].dropna(subset=["ref_rate"])
    check(len(blended) > 0, "some series blend with a reference")
    row = blended.iloc[0]
    expected_z = row["group_support"] / (row["group_support"] + row["k"])
    check(abs(row["z"] - round(expected_z, 3)) < 1e-9
          and abs(row["tasa_estimada"] - (expected_z * row["group_rate"] + (1 - expected_z) * row["ref_rate"])) < 1e-9,
          "z = n / (n + k) with the group's support; rate = z · group + (1 − z) · reference")
    rates_per_group = estimate.dropna(subset=["tasa_estimada"]).groupby("composition_id")["tasa_estimada"].nunique()
    check((rates_per_group == 1).all(), "every series of a group has the same rate: the group lends it")
    check((estimate["se_prediccion_pp"].dropna() >= estimate["se_estimacion_pp"].dropna() - 1e-9).all()
          and np.allclose(binomial_se_pp([0.5], [100]), [5.0]),
          "the prediction error never sheds the series' own noise; se(0.5, 100) = 5 pp")


if __name__ == "__main__":
    test_the_passes()
    test_the_collapse_limit()
    test_the_rate()
    finish()
