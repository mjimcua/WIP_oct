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
     level_merge_max_pp points (the ±5 pp promise by default).
  4. the rate of every group, year by year, goes with it as evidence of its stability.

THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
the ids of the forecast series do not move from one month to the next. A dimension missing from the file
(or with another type) is generated and added. To regenerate everything, delete the file. A value the file
does not know goes to the residual group, with a warning.

Actions (logged as they are done):
  1. the declared dimensions and the file: use it, or generate
  2. generate the groups of every dimension that needs them (training months)
  3. fill <column>_level_1
  4. check the values the file knows                                     check 1
  5. write the levels table                                              check 2
  6. count the checks; stop if any failed
  7. show the groups and the columns by role, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. no value of the extract is unknown to the file                   (warning only: it goes to the residual)
   2. table sff_dimension_levels written and read back
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import GENERATED_LEVEL_SUFFIX, Config, join_columns, roles_overview
from vocabulario import ROLE_TRAIN, TABLE_DIMENSION_LEVELS


STEP_LABEL = "01b"
STEP_NAME = "DIMENSIONS WITH A GENERATED LEVEL"
STEP_PURPOSE = ("give every dimension declared in leveled_dims its coarse level, <column>_level_1: its values grouped by "
                "their standardised renewal rate in the training months, persisted in a JSON that later runs reuse so the "
                "forecast series keep their ids; the column itself keeps its raw value")
STEP_ACTIONS = ["the declared dimensions and the file: use it, or generate",
                "generate the groups of every dimension that needs them (training months)",
                "fill <column>_level_1",
                "check the values the file knows (check 1)",
                "write the levels table (check 2)",
                "count the checks; stop if any failed",
                "show the groups and the columns by role, as tables"]
STEP_OUTPUT = "the raw with <column>_level_1 · the JSON of the groups · table sff_dimension_levels"

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
                   if column_name not in stored.get("dims", {}) or stored["dims"][column_name]["type"] != level_type]
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
    levels_table = levels_as_table(stored, configuration)
    configuration.write_table(STEP_LABEL, check_log, levels_table, TABLE_DIMENSION_LEVELS)

    # [6] the count of the checks
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the groups and the columns by role
    configuration.log_action(STEP_LABEL, 7, "the groups of every dimension (rate_std: standardised by cell, training months):")
    configuration.show_table(levels_table.drop(columns=["rate_by_year"], errors="ignore"))
    configuration.logger.doc(f"[{STEP_LABEL}] the columns by role, with the generated levels:")
    configuration.show_table(roles_overview(raw.columns, configuration))
    return raw


def natural_key(value: str) -> list:
    """'2 devices' before '10 devices': numbers compared as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
    """The groups of one dimension, from the training months: standardised rate per value, residual for
    the rare ones, neighbours merged while their rates differ by at most level_merge_max_pp."""
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
    train = raw[training & (raw[due].to_numpy() > 0)].copy()
    train["_value"] = train[column_name].astype(str)
    other_dims = [other for other in configuration.business_mandatory_dims
                  if other != column_name and other not in configuration.generated_columns]
    train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
    cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
    train["_expected"] = train["_cell"].map(cell_rate) * train[due]
    global_rate = train[renewed].sum() / train[due].sum()

    per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
    per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
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
        differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
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
        rare_values = list(per_value.index[rare])
        for value in rare_values:
            mapping[value] = RESIDUAL_GROUP
        evidence.append(group_evidence(RESIDUAL_GROUP, rare_values, per_value, train, standardised(rare_values), global_rate,
                                       configuration))
    return {"type": level_type, "max_merge_pp": configuration.level_merge_max_pp,
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
    for column_name in configuration.leveled_dims:
        for group in stored["dims"][column_name]["groups"]:
            rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
                         "n_values": len(group["values"]), "units_due": group["units_due"],
                         "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                         "rate_by_year": json.dumps(group["rate_by_year"])})
    return pd.DataFrame(rows)
