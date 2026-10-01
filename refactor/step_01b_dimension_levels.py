"""
step_01b_dimension_levels.py — The generated coarse level of the leveled dimensions.

Some mandatory dimensions come with many values (a band, a term, a product-usage bucket). The Config
declares them in leveled_dims ({column: "ordinal" | "nominal"}); the column keeps its raw value (the
fine level) and the library adds ONE column, <column>_level_1, with its values grouped by their renewal
rate. The Config already counts <column>_level_1 as a mandatory dim, right after its column, from the
moment it is built; this step only fills it, early, so every later step sees the same columns. The
ladder collapses the fine level first, then the coarse one.

HOW THE GROUPS ARE MADE (only the TRAINING months, taken from the calendar of the Config)
  1. the STANDARDISED rate of every value: renewed / expected, where expected = the rate of its cell
     (every other mandatory dim of the extract) × its units due, times the global rate. A value is
     compared with the others inside the same cell, not in bulk.
  2. values whose monthly support (median of the monthly units due) is below support_floor go to a
     RESIDUAL group: too little to tell their rate.
  3. the others are ordered (ordinal: by their value, numbers as numbers; nominal: by their rate) and
     the two NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
     the MERGE THRESHOLD.
  4. the rate of every value and of every group, year by year, goes with them as evidence.

THE MERGE THRESHOLD (the same principle as the ladder: merge while the bias it adds is smaller than the
noise it removes). The generated level is only used by the series that climb the ladder, the ones below
the support floor; their monthly rate carries at least the binomial noise of a series AT the floor,
√(p(1−p)/support_floor). Two values whose rates differ by less than that cannot be told apart by any
series that will use the merge: merging them adds less bias than the noise it removes. With p = 0.64 and
a floor of 30, the threshold is 8.8 pp. level_merge_max_pp fixes another value (5 pp = the noise of a
series of about 90 contracts a month).

A dimension that ends in a single group does not separate the renewal rate: its level_1 is constant and
the ladder gives it no pass (step 09).

THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
the ids of the forecast series do not move from one month to the next. A dimension missing from the file
(or with another type or another merge threshold) is generated and added. To regenerate everything, delete the file. A value the file
does not know goes to the residual group, with a warning.

Actions (logged as they are done):
  1. the declared dimensions and the file: use it, or generate
  2. generate the groups of every dimension that needs them (training months)
  3. fill <column>_level_1
  4. check the values the file knows                                     check 1
  5. write the levels tables                                             checks 2-3
  6. count the checks; stop if any failed
  7. show the criterion, the groups (rate per year), the values and the merge decisions, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
   2-3. tables sff_dimension_levels and sff_dimension_level_values written and read back
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import GENERATED_LEVEL_SUFFIX, Config, join_columns, roles_overview
from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVEL_VALUES, TABLE_DIMENSION_LEVELS


STEP_LABEL = "01b"
STEP_NAME = "DIMENSIONS WITH A GENERATED LEVEL"
STEP_PURPOSE = ("give every dimension declared in leveled_dims its coarse level, <column>_level_1: its values grouped by "
                "their standardised renewal rate in the training months, persisted in a JSON that later runs reuse so the "
                "forecast series keep their ids; the column itself keeps its raw value")
STEP_ACTIONS = ["the declared dimensions and the file: use it, or generate",
                "generate the groups of every dimension that needs them (training months)",
                "fill <column>_level_1",
                "check the values the file knows (check 1)",
                "write the levels tables (checks 2-3)",
                "count the checks; stop if any failed",
                "show the criterion, the groups (rate per year), the values and the merge decisions, as tables"]
STEP_OUTPUT = ("the raw with <column>_level_1 · the JSON of the groups · tables sff_dimension_levels (groups) and "
               "sff_dimension_level_values (values)")

RESIDUAL_GROUP = "residual"


def apply_dimension_levels(validated_raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The raw with the generated coarse level of every dimension in leveled_dims.

    INPUT:   the raw validated by steps 00 and 01 · the Config (leveled_dims, levels_path, level_merge_max_pp,
             support_floor, the calendar).
    OUTPUT:  the raw with <column>_level_1 for every leveled column.
    RULES:   see the module header.
    EDGE CASES: no leveled_dims → the raw unchanged, nothing written. A dimension with a single value → one
             group. All values below the floor → a single residual group.
    """
    if not configuration.leveled_dims:
        return validated_raw
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    raw = validated_raw.copy()

    # [1] the declared dimensions and the file
    path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
    stored = read_levels_file(path)
    to_generate = [column_name for column_name, level_type in configuration.leveled_dims.items()
                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type
                   or stored["dims"][column_name].get("criterion", {}).get("fixed_pp") != configuration.level_merge_max_pp]
    configuration.log_action(STEP_LABEL, 1, f"{len(configuration.leveled_dims)} dimensions with a generated level "
                                            f"{list(configuration.leveled_dims)} · file {path}: "
                                            + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
                                            + (f"generating {to_generate}" if to_generate else "every dimension in it, reused"))

    # [2] generate what the file does not have
    if to_generate:
        stored.setdefault("dims", {})
        for column_name in to_generate:
            stored["dims"][column_name] = generate_levels(raw, column_name, configuration.leveled_dims[column_name], configuration)
        stored["created"] = stored.get("created") or datetime.now().isoformat(timespec="seconds")
        stored["updated"] = datetime.now().isoformat(timespec="seconds")
        stored["current_month"] = str(configuration.current_month)
        write_levels_file(path, stored)
        configuration.log_action(STEP_LABEL, 2, f"groups generated from the training months and written to {path}")
    else:
        configuration.log_action(STEP_LABEL, 2, "nothing to generate")

    # [3] fill the coarse level
    unknown_values = {}
    for column_name in configuration.leveled_dims:
        mapping = stored["dims"][column_name]["mapping"]
        values = raw[column_name].astype(str)
        unknown = sorted(set(values.unique()) - set(mapping))
        if unknown:
            unknown_values[column_name] = unknown
        raw[f"{column_name}{GENERATED_LEVEL_SUFFIX}"] = values.map(mapping).fillna(RESIDUAL_GROUP)
    configuration.log_action(STEP_LABEL, 3, "filled: " + ", ".join(configuration.generated_columns))

    # [4] the check
    configuration.log_action(STEP_LABEL, 4, "checking the values the file knows")
    configuration.log_check(STEP_LABEL, check_log, "no value of the extract is unknown to the file", not unknown_values,
                            failure_detail="values sent to the residual: " + "; ".join(f"{column_name}: {values[:10]}"
                                                                                     for column_name, values in unknown_values.items()),
                            blocking=False)

    # [5] the table
    configuration.log_action(STEP_LABEL, 5, "writing the levels")
    levels_table, values_table, decisions_table = levels_as_tables(stored, configuration)
    configuration.write_table(STEP_LABEL, check_log, levels_table, TABLE_DIMENSION_LEVELS)
    configuration.write_table(STEP_LABEL, check_log, values_table, TABLE_DIMENSION_LEVEL_VALUES)

    # [6] the count of the checks
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the evidence: the criterion, the groups year by year, the values, the merge decisions
    show_the_evidence(stored, levels_table, values_table, decisions_table, configuration)
    configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, with the generated levels:")
    configuration.show_table(roles_overview(raw.columns, configuration))
    return raw


def natural_key(value: str) -> list:
    """'2 devices' before '10 devices': numbers compared as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def merge_threshold_pp(global_rate: float, configuration: Config) -> tuple:
    """(threshold in pp, how it was set): the binomial noise of a series at the support floor, or the fixed value."""
    if configuration.level_merge_max_pp is not None:
        return float(configuration.level_merge_max_pp), f"fixed in the Config (level_merge_max_pp = {configuration.level_merge_max_pp})"
    noise = 100 * np.sqrt(global_rate * (1 - global_rate) / configuration.support_floor)
    return float(noise), (f"the binomial noise of a series at the support floor: 100·√(p(1−p)/{configuration.support_floor:.0f}) "
                          f"with p = {global_rate:.2f}")


def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
    """The groups of one dimension, from the training months: standardised rate per value, residual for the
    rare ones, neighbours merged while their rates differ by at most the merge threshold. Keeps the evidence:
    every value, every group year by year, every merge decision."""
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
    train = raw[training & (raw[due].to_numpy() > 0)].copy()
    train["_value"] = train[column_name].astype(str)
    other_dims = [other for other in configuration.business_mandatory_dims
                  if other != column_name and other not in configuration.generated_columns]
    train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
    cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
    train["_expected"] = train["_cell"].map(cell_rate) * train[due]
    train["_year"] = train[period].map(lambda month: month.year)
    global_rate = float(train[renewed].sum() / train[due].sum())
    threshold_pp, threshold_rule = merge_threshold_pp(global_rate, configuration)

    per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
    per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
    rare = per_value["monthly_support"] < configuration.support_floor

    def standardised(values: list, rows: pd.DataFrame = None) -> float:
        if rows is None:
            block = per_value.loc[values]
            return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)
        expected = rows["_expected"].sum()
        return float(global_rate * rows[renewed].sum() / expected) if expected > 0 else np.nan

    def by_year(values: list) -> dict:
        block = train[train["_value"].isin(values)]
        return {str(year): round(standardised(values, rows), 4) for year, rows in block.groupby("_year")}

    groups = [[value] for value in per_value.index[~rare]]
    if level_type == "ordinal":
        groups.sort(key=lambda group: natural_key(group[0]))
    else:
        groups.sort(key=lambda group: standardised(group))

    def label(group: list) -> str:
        ordered = sorted(group, key=natural_key)
        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"

    # merge the closest neighbours while they differ by at most the threshold; every decision is kept
    decisions = []
    while len(groups) > 1:
        differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
        closest = int(np.argmin(differences))
        if 100 * differences[closest] > threshold_pp:
            break
        decisions.append({"left": label(groups[closest]), "right": label(groups[closest + 1]),
                          "difference_pp": round(100 * differences[closest], 2), "decision": "merged"})
        groups[closest:closest + 2] = [groups[closest] + groups[closest + 1]]
    for index in range(len(groups) - 1):                       # the neighbours that stay apart, and by how much
        decisions.append({"left": label(groups[index]), "right": label(groups[index + 1]),
                          "difference_pp": round(100 * abs(standardised(groups[index]) - standardised(groups[index + 1])), 2),
                          "decision": "kept apart"})

    mapping, group_evidence, value_evidence = {}, [], []
    final_groups = [(label(group), group) for group in groups]
    if rare.any():
        final_groups.append((RESIDUAL_GROUP, list(per_value.index[rare])))
    for group_label, values in final_groups:
        for value in values:
            mapping[value] = group_label
            value_rate = standardised([value])
            support = float(per_value.loc[value, "monthly_support"])
            value_evidence.append({"value": value, "group": group_label, "units_due": float(per_value.loc[value, "units_due"]),
                                   "monthly_support": support, "rate_std": round(value_rate, 4),
                                   "noise_pp": round(100 * np.sqrt(max(value_rate * (1 - value_rate), 0) / max(support, 1)), 2),
                                   "rate_by_year": by_year([value])})
        block = train[train["_value"].isin(values)]
        group_evidence.append({"group": group_label, "values": sorted(values, key=natural_key),
                               "units_due": float(per_value.loc[values, "units_due"].sum()),
                               "monthly_support": float(block.groupby(period)[due].sum().median()),
                               "rate_std": round(standardised(values), 4), "rate_by_year": by_year(values)})
    value_evidence.sort(key=lambda row: natural_key(row["value"]))
    return {"type": level_type,
            "criterion": {"threshold_pp": round(threshold_pp, 2), "rule": threshold_rule, "fixed_pp": configuration.level_merge_max_pp,
                          "global_rate": round(global_rate, 4), "support_floor": configuration.support_floor},
            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping,
            "groups": group_evidence, "values": value_evidence, "decisions": decisions}


def read_levels_file(path: str) -> dict:
    """The stored groups, or an empty dict when there is no file yet."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_levels_file(path: str, content: dict) -> None:
    """The groups, as indented JSON (readable and editable by hand)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)


def levels_as_tables(stored: dict, configuration: Config) -> tuple:
    """The evidence of the JSON as tables: groups, values and merge decisions (one row each), with the rate
    of every year as a column of its own, readable on screen and in Power BI."""
    group_rows, value_rows, decision_rows = [], [], []
    for column_name in configuration.leveled_dims:
        stored_dimension = stored["dims"][column_name]
        threshold = stored_dimension.get("criterion", {}).get("threshold_pp")
        for group in stored_dimension["groups"]:
            group_rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
                               "n_values": len(group["values"]), "units_due": group["units_due"],
                               "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                               **{f"rate_{year}": rate for year, rate in group["rate_by_year"].items()}})
        for value in stored_dimension.get("values", []):
            value_rows.append({"dimension": column_name, "value": value["value"], "group": value["group"],
                               "units_due": value["units_due"], "monthly_support": value["monthly_support"],
                               "rate_std": value["rate_std"], "noise_pp": value["noise_pp"],
                               **{f"rate_{year}": rate for year, rate in value["rate_by_year"].items()}})
        for decision in stored_dimension.get("decisions", []):
            decision_rows.append({"dimension": column_name, **decision, "threshold_pp": threshold,
                                  "within_5_pp": "yes" if decision["difference_pp"] <= 5 else "no"})
    return pd.DataFrame(group_rows), pd.DataFrame(value_rows), pd.DataFrame(decision_rows)


def show_the_evidence(stored: dict, levels_table: pd.DataFrame, values_table: pd.DataFrame,
                      decisions_table: pd.DataFrame, configuration: Config) -> None:
    """Action 7: the criterion, then the groups year by year, the values and the merge decisions."""
    for column_name in configuration.leveled_dims:
        criterion = stored["dims"][column_name].get("criterion", {})
        groups = [group for group in stored["dims"][column_name]["groups"] if group["group"] != RESIDUAL_GROUP]
        configuration.log_action(STEP_LABEL, 7, f"{column_name} ({stored['dims'][column_name]['type']}): neighbours merge while their "
                                                f"standardised rates differ by ≤ {criterion.get('threshold_pp', '?')} pp — "
                                                f"{criterion.get('rule', '?')} · {len(groups)} groups"
                                                + (" · ONE group: the dimension does not separate the renewal rate beyond that "
                                                   "noise; its level_1 is constant and gets no pass of the ladder (step 09)"
                                                   if len(groups) == 1 else ""))
    rate = next(iter(stored["dims"].values())).get("criterion", {}).get("global_rate", 0.64)
    noise_rows = [{"contracts_a_month": support, "binomial_noise_pp": round(100 * np.sqrt(rate * (1 - rate) / support), 1)}
                  for support in (10, 30, 50, 100, 271, 1000)]
    configuration.logger.doc(f"[{STEP_LABEL}] what a difference in pp means: the noise of the monthly rate of a series of n contracts "
                             f"(p = {rate:.2f}); a merge helps the series whose noise is larger than the difference it adds:")
    configuration.show_table(pd.DataFrame(noise_rows))
    configuration.logger.doc(f"[{STEP_LABEL}] the groups (rate_std: standardised by cell; rate_<year>: the same, year by year — "
                             f"a group is stable when its years stay close):")
    configuration.show_table(levels_table)
    configuration.logger.doc(f"[{STEP_LABEL}] every value (noise_pp: the binomial noise of its own monthly rate; two values whose "
                             f"rates differ by less than their noise cannot be told apart month by month):")
    configuration.show_table(values_table)
    configuration.logger.doc(f"[{STEP_LABEL}] the merge decisions, in order (within_5_pp: what the old fixed 5 pp would have said):")
    configuration.show_table(decisions_table)
