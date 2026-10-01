"""
step_02b_dimension_levels.py — The dimensions with generated levels.

Some dimensions come with many values (a band, a term, a product-usage bucket). Instead of grouping
them by hand, the Config declares them in leveled_dims and the library gives each two levels:

  <name>_level_2   the raw value, untouched (the fine detail)
  <name>_level_1   the values grouped by their renewal rate (the coarse level, optimised)

The ladder collapses level_2 before level_1, as with any _level_N family: the information moves in
the direction of greater homogeneity, and the coarse groups are now proposed by a rule.

HOW THE GROUPS ARE MADE (only the TRAINING months: the exam must not inform them)
  1. the STANDARDISED rate of every value: renewed / expected, where expected = the rate of its cell
     (every other mandatory dim) × its units due, times the global rate. A value is compared with the
     others inside the same cell, not in bulk: a band sold mostly where people renew less must not
     look worse because of where it is sold.
  2. values whose monthly support (median of the monthly units due) is below support_floor go to a
     RESIDUAL group: too little to tell their rate.
  3. the others are ordered (ordinal: by their value; nominal: by their standardised rate) and the two
     NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
     level_merge_max_pp points (the ±5 pp promise by default): a difference smaller than what we
     promise to measure is not worth a separate group.
  4. the rate of every group, year by year, goes with it as evidence of its stability.

THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it,
so the ids of the forecast series do not move from one month to the next. A dimension missing from the
file (or with another source or type) is generated and added. To regenerate everything, delete the file.
A value the file does not know goes to the residual group, with a warning.

Actions (logged as they are done):
  1. the declared dimensions and the file: use it, or generate
  2. generate the groups of every dimension that needs them (training months)
  3. apply them: <name>_level_1 and <name>_level_2, and the mandatory dims of the Config updated
  4. check the levels                                                    checks 1-3
  5. write the levels table                                              check 4
  6. count the checks; stop if any failed
  7. show the groups, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. every row has its level_1 and level_2 (no value left without a group)
   2. the units due add up the same grouped by level_1 and by level_2 (a partition)
   3. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
   4. table sff_dimension_levels written and read back
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config, join_columns
from vocabulario import CALENDAR_ROLE_COLUMN, ROLE_TRAIN, TABLE_DIMENSION_LEVELS


STEP_LABEL = "02b"
STEP_NAME = "DIMENSIONS WITH GENERATED LEVELS"
STEP_PURPOSE = ("give the dimensions declared in leveled_dims two levels: level_2 the raw value and level_1 the values "
                "grouped by their standardised renewal rate (training months only), persisted in a JSON that later runs "
                "reuse so the forecast series keep their ids")
STEP_ACTIONS = ["the declared dimensions and the file: use it, or generate",
                "generate the groups of every dimension that needs them (training months)",
                "apply them: <name>_level_1 and <name>_level_2, and the mandatory dims of the Config updated",
                "check the levels (checks 1-3)",
                "write the levels table (check 4)",
                "count the checks; stop if any failed",
                "show the groups, as a table"]
STEP_OUTPUT = "the raw with <name>_level_1 and <name>_level_2 · the JSON of the groups · table sff_dimension_levels"

RESIDUAL_GROUP = "residual"
LEVEL_TYPES = ("ordinal", "nominal")
UNITS_TOLERANCE = 1e-6


def apply_dimension_levels(calendared_raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The raw with the generated levels of every dimension in leveled_dims.

    INPUT:   the raw after step 02 (roles of the calendar) · the Config (leveled_dims, levels_path,
             level_merge_max_pp, support_floor).
    OUTPUT:  the raw with <name>_level_1 / <name>_level_2; the Config's business_mandatory_dims updated.
    RULES:   see the module header.
    EDGE CASES: no leveled_dims → the raw unchanged, nothing written. A dimension with a single value →
             one group. All values below the floor → a single residual group.
    """
    if not configuration.leveled_dims:
        return calendared_raw
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    raw = calendared_raw.copy()

    # [1] the declared dimensions and the file
    path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
    stored = read_levels_file(path)
    to_generate = [name for name, spec in configuration.leveled_dims.items()
                   if name not in stored.get("dims", {}) or stored["dims"][name]["source"] != source_of(name, spec)
                   or stored["dims"][name]["type"] != type_of(spec)]
    configuration.log_action(STEP_LABEL, 1, f"{len(configuration.leveled_dims)} dimensions with levels "
                                            f"{list(configuration.leveled_dims)} · file {path}: "
                                            + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
                                            + (f"generating {to_generate}" if to_generate else "every dimension in it, reused"))

    # [2] generate what the file does not have
    if to_generate:
        stored.setdefault("dims", {})
        for name in to_generate:
            stored["dims"][name] = generate_levels(raw, name, configuration.leveled_dims[name], configuration)
        stored["created"] = stored.get("created") or datetime.now().isoformat(timespec="seconds")
        stored["updated"] = datetime.now().isoformat(timespec="seconds")
        stored["current_month"] = str(configuration.current_month)
        write_levels_file(path, stored)
        configuration.log_action(STEP_LABEL, 2, f"groups generated from the training months and written to {path}")
    else:
        configuration.log_action(STEP_LABEL, 2, "nothing to generate")

    # [3] apply them
    unknown_values = {}
    for name, spec in configuration.leveled_dims.items():
        source = source_of(name, spec)
        mapping = stored["dims"][name]["mapping"]
        values = raw[source].astype(str)
        unknown = sorted(set(values.unique()) - set(mapping))
        if unknown:
            unknown_values[name] = unknown
        level_2, level_1 = f"{name}_level_2", f"{name}_level_1"
        raw[level_2] = raw[source]
        raw[level_1] = values.map(mapping).fillna(RESIDUAL_GROUP)
        replace_in_mandatory(configuration, source, name, level_1, level_2)
    configuration.log_action(STEP_LABEL, 3, "levels applied · mandatory dims now: " + ", ".join(configuration.business_mandatory_dims))

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the levels")
    missing = sum(int(raw[f"{name}_level_1"].isna().sum() + raw[f"{name}_level_2"].isna().sum()) for name in configuration.leveled_dims)
    configuration.log_check(STEP_LABEL, check_log, "every row has its level_1 and level_2", missing == 0,
                            failure_detail=f"{missing:,} rows without a level")
    due = configuration.pipeline_units_col
    partition = all(abs(raw.groupby(f"{name}_level_1")[due].sum().sum() - raw.groupby(f"{name}_level_2")[due].sum().sum())
                    <= UNITS_TOLERANCE for name in configuration.leveled_dims)
    configuration.log_check(STEP_LABEL, check_log, "the units due add up the same grouped by level_1 and by level_2", partition,
                            failure_detail="a level loses or duplicates units")
    configuration.log_check(STEP_LABEL, check_log, "no value of the extract is unknown to the file", not unknown_values,
                            failure_detail="values sent to the residual: " + "; ".join(f"{name}: {values[:10]}"
                                                                                     for name, values in unknown_values.items()),
                            blocking=False)

    # [5] the table
    configuration.log_action(STEP_LABEL, 5, "writing the levels")
    levels_table = levels_as_table(stored, configuration)
    configuration.write_table(STEP_LABEL, check_log, levels_table, TABLE_DIMENSION_LEVELS)

    # [6] the count of the checks
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the groups
    configuration.log_action(STEP_LABEL, 7, "the groups of every dimension (rate_std: standardised by cell, training months):")
    configuration.show_table(levels_table.drop(columns=["rate_by_year"], errors="ignore"))
    return raw


def source_of(name: str, spec: dict) -> str:
    """The raw column of a leveled dimension (its own name unless the spec says otherwise)."""
    return (spec or {}).get("source", name)


def type_of(spec: dict) -> str:
    """ordinal (only neighbouring values merge) or nominal (any two)."""
    level_type = (spec or {}).get("type", "nominal")
    if level_type not in LEVEL_TYPES:
        raise ValueError(f"leveled_dims type must be one of {LEVEL_TYPES}, not {level_type!r}")
    return level_type


def natural_key(value: str) -> list:
    """'2 devices' before '10 devices': numbers compared as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def generate_levels(raw: pd.DataFrame, name: str, spec: dict, configuration: Config) -> dict:
    """The groups of one dimension, from the training months: standardised rate per value, residual for
    the rare ones, neighbours merged while their rates differ by at most level_merge_max_pp."""
    source, level_type = source_of(name, spec), type_of(spec)
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    train = raw[(raw[CALENDAR_ROLE_COLUMN] == ROLE_TRAIN) & (raw[due] > 0)].copy()
    train["_value"] = train[source].astype(str)
    other_dims = [column for column in configuration.business_mandatory_dims
                  if column not in (source, f"{name}_level_1", f"{name}_level_2", name)]
    train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
    cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
    train["_expected"] = train["_cell"].map(cell_rate) * train[due]
    global_rate = train[renewed].sum() / train[due].sum()

    per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
    monthly = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
    per_value["monthly_support"] = monthly
    rare = per_value["monthly_support"] < configuration.support_floor
    groups = [[value] for value in per_value.index[~rare]]
    if level_type == "ordinal":
        groups.sort(key=lambda group: natural_key(group[0]))
    else:
        groups.sort(key=lambda group: per_value.loc[group[0], "observed"] / max(per_value.loc[group[0], "expected"], 1e-9))

    def standardised(group: list) -> float:
        block = per_value.loc[group]
        return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)

    # merge the closest neighbours while they differ by at most level_merge_max_pp
    while len(groups) > 1:
        differences = [abs(standardised(groups[i]) - standardised(groups[i + 1])) for i in range(len(groups) - 1)]
        closest = int(np.argmin(differences))
        if 100 * differences[closest] > configuration.level_merge_max_pp:
            break
        groups[closest:closest + 2] = [groups[closest] + groups[closest + 1]]

    def label(group: list) -> str:
        ordered = sorted(group, key=natural_key)
        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"

    mapping, evidence = {}, []
    for group in groups:
        group_label = label(group)
        for value in group:
            mapping[value] = group_label
        evidence.append(group_evidence(group_label, group, per_value, train, standardised(group), global_rate, configuration))
    if rare.any():
        for value in per_value.index[rare]:
            mapping[value] = RESIDUAL_GROUP
        evidence.append(group_evidence(RESIDUAL_GROUP, list(per_value.index[rare]), per_value, train,
                                       standardised(list(per_value.index[rare])), global_rate, configuration))
    return {"source": source, "type": level_type, "max_merge_pp": configuration.level_merge_max_pp,
            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping, "groups": evidence}


def group_evidence(group_label: str, values: list, per_value: pd.DataFrame, train: pd.DataFrame, rate: float,
                   global_rate: float, configuration: Config) -> dict:
    """The evidence of one group: its values, units, support, standardised rate and that rate year by year."""
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    block = train[train["_value"].isin(values)]
    by_year = {}
    for year, rows in block.groupby(block[period].map(lambda month: month.year)):
        expected = rows["_expected"].sum()
        by_year[str(year)] = round(float(global_rate * rows[renewed].sum() / expected), 4) if expected > 0 else None
    return {"group": group_label, "values": sorted(values, key=natural_key), "units_due": float(per_value.loc[values, "units_due"].sum()),
            "monthly_support": float(block.groupby(period)[due].sum().median()), "rate_std": round(float(rate), 4),
            "rate_by_year": by_year}


def replace_in_mandatory(configuration: Config, source: str, name: str, level_1: str, level_2: str) -> None:
    """The mandatory dims of the Config with the two levels in the place of the source (or of the name)."""
    dims = list(configuration.business_mandatory_dims)
    family = (source, name, level_1, level_2)
    positions = [index for index, column in enumerate(dims) if column in family]
    position = min(positions) if positions else len(dims)          # where the family already was
    dims = [column for column in dims if column not in family]
    dims[position:position] = [level_1, level_2]
    configuration.business_mandatory_dims = dims


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


def levels_as_table(stored: dict, configuration: Config) -> pd.DataFrame:
    """One row per dimension × group: the evidence of the JSON as a table (for Power BI and the report)."""
    rows = []
    for name in configuration.leveled_dims:
        for group in stored["dims"][name]["groups"]:
            rows.append({"dimension": name, "group": group["group"], "values": ", ".join(group["values"]),
                         "n_values": len(group["values"]), "units_due": group["units_due"],
                         "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                         "rate_by_year": json.dumps(group["rate_by_year"])})
    return pd.DataFrame(rows)
