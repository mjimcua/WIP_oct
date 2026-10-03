"""
step_15_uplift.py — The revaluation: at what price does a renewer renew?

What falls due has a price (USD due / units due = the pipeline AUV of the row). A renewer
pays a price too (renewed USD / renewed units). The UPLIFT is the second over the first:
1.0 = renews at the same price; 1.43 = pays 43 % more. The forecast of a row is
USD due × renewal rate × uplift.

Two ways to know the uplift of a future row:
  · STATISTICAL (the default, and the only one where the discount is unknown): the uplift
    observed in the past renewals of its UPLIFT CELL (uplift mandatory dims, revaluation
    extras, discount bucket), as a ratio of sums Σ renewed USD / Σ (renewed units ×
    pipeline AUV): weighted by money, over the renewers (a contract that did not renew has
    no price). Its n is the number of renewers. A cell with fewer than uplift_floor
    renewers takes its PARENT (same mandatory dims and discount bucket, the other extras
    set to '*'), then its mandatory cell, then the whole portfolio. The band of the
    uplift: bootstrap of the renewer rows (p5-p95).
  · CONTRACT (where the exact discount is known): a customer who paid list × (1 − d)
    renews at list, so the uplift is 1 / (1 − d). This step checks the rule against the past
    renewals with a known discount (how much of the renewed money it explains within ±2 %);
    step 16 decides, in the exam months, which path predicts better.

Actions (logged as they are done):
  1. the renewer rows of the closed months, with their pipeline AUV and their uplift
  2. the statistical uplift of every uplift cell: own, parent, cell, global; the band
  3. the contract rule against the past renewals with a known discount
  4. check the uplifts                                               checks 1-3
  5. write the uplift of the cells and the contract check            checks 4-5
  6. count the checks; stop if any failed
  7. show the uplift by origin and by discount bucket, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. every uplift cell of the fine table has an uplift (future cells included)
   2. every uplift is inside [0.01, uplift_cap]
   3. the band contains the uplift of every cell with its own or its parent's
   4-5. tables sff_uplift_celda and sff_uplift_contrato_check written and read back

Output: the uplift of every cell (with its origin and band) and the contract check ·
tables sff_uplift_celda, sff_uplift_contrato_check.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from vocabulario import (CALENDAR_ROLE_COLUMN, TABLE_CONTRACT_CHECK, TABLE_UPLIFT_CELLS, TRUTH_ROLES, UPLIFT_CELL,
                         UPLIFT_CELL_ID_COLUMN, UPLIFT_GLOBAL, UPLIFT_OWN, UPLIFT_PARENT, WILDCARD)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "15"
STEP_NAME = "UPLIFT"
STEP_PURPOSE = ("estimate at what price a renewer renews relative to what fell due: the statistical uplift of every "
                "uplift cell from its past renewals (borrowing from its parent when it has few renewers), and the check "
                "of the contract rule 1 / (1 − discount) where the discount is known")
STEP_ACTIONS = ["the renewer rows of the closed months, with their pipeline AUV and their uplift",
                "the statistical uplift of every uplift cell: own, parent, cell, global; the band",
                "the contract rule against the past renewals with a known discount",
                "check the uplifts (checks 1-3)",
                "write the uplift of the cells and the contract check (checks 4-5)",
                "count the checks; stop if any failed",
                "show the uplift by origin and by discount bucket, as tables"]
STEP_OUTPUT = "the uplift of every uplift cell (origin, band) · the contract check · tables sff_uplift_celda, sff_uplift_contrato_check"

# ─── named constants ─────────────────────────────────────────────────────────────
MIN_UPLIFT = 0.01
CONTRACT_TOLERANCE = 0.02          # a renewal "matches" the rule when its uplift is within ±2 % of 1/(1−d)
BAND_LOW, BAND_HIGH = 0.05, 0.95


def renewer_rows(fine_table: pd.DataFrame, configuration: Config, before_month=None) -> pd.DataFrame:
    """The rows of closed truth months that renewed (optionally only before a month), with
    their pipeline AUV, the two sides of the ratio and their parent and cell ids."""
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.renewed_units_col].fillna(0) > 0)
                        & (fine_table[configuration.pipeline_units_col] > 0)].copy()
    if before_month is not None:
        closed = closed[closed[configuration.period_col] < before_month]
    closed["auv_pipeline"] = closed[configuration.pipeline_usd_col] / closed[configuration.pipeline_units_col]
    closed["_numerador"] = closed[configuration.renewed_usd_col]
    closed["_denominador"] = closed[configuration.renewed_units_col] * closed["auv_pipeline"]
    return add_parent_ids(closed, configuration)


def add_parent_ids(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The parent of an uplift cell (mandatory dims and discount bucket kept, the other extras
    '*') and its mandatory cell (only the uplift mandatory dims)."""
    uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
    bucket = [configuration.discount_bucket_column] if configuration.discount_value_column else []
    rows = rows.copy()
    rows["uplift_padre_id"] = join_columns(rows, list(uplift_mandatory) + bucket) + "|" + WILDCARD
    rows["uplift_celda_mandatory_id"] = join_columns(rows, list(uplift_mandatory))
    return rows


def estimate_cell_uplifts(fine_table: pd.DataFrame, renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The statistical uplift of every uplift cell of the fine table (future ones included):
    own if ≥ uplift_floor renewers, else parent, else mandatory cell, else global. Clipped to
    [0.01, uplift_cap]. Band: bootstrap of the renewer rows the uplift was computed from."""
    renewed_units = configuration.renewed_units_col
    cells = add_parent_ids(fine_table, configuration)[[UPLIFT_CELL_ID_COLUMN, "uplift_padre_id", "uplift_celda_mandatory_id"]]
    cells = cells.drop_duplicates(UPLIFT_CELL_ID_COLUMN).reset_index(drop=True)

    def level(group_column):
        grouped = renewers.groupby(group_column)
        return pd.DataFrame({"uplift": grouped["_numerador"].sum() / grouped["_denominador"].sum(),
                             "renovadores": grouped[renewed_units].sum()})

    own, parent, cell = level(UPLIFT_CELL_ID_COLUMN), level("uplift_padre_id"), level("uplift_celda_mandatory_id")
    global_uplift = float(renewers["_numerador"].sum() / renewers["_denominador"].sum()) if len(renewers) else 1.0
    cells["renovadores"] = cells[UPLIFT_CELL_ID_COLUMN].map(own["renovadores"]).fillna(0.0)
    cells["uplift_propio"] = cells[UPLIFT_CELL_ID_COLUMN].map(own["uplift"])
    cells["renovadores_padre"] = cells["uplift_padre_id"].map(parent["renovadores"]).fillna(0.0)
    cells["uplift_padre"] = cells["uplift_padre_id"].map(parent["uplift"])
    cells["renovadores_celda"] = cells["uplift_celda_mandatory_id"].map(cell["renovadores"]).fillna(0.0)
    cells["uplift_celda"] = cells["uplift_celda_mandatory_id"].map(cell["uplift"])
    floor = configuration.uplift_floor
    conditions = [cells["renovadores"] >= floor, cells["renovadores_padre"] >= floor, cells["renovadores_celda"] > 0]
    cells["uplift_origen"] = np.select(conditions, [UPLIFT_OWN, UPLIFT_PARENT, UPLIFT_CELL], default=UPLIFT_GLOBAL)
    chosen = np.select(conditions, [cells["uplift_propio"], cells["uplift_padre"], cells["uplift_celda"]], default=global_uplift)
    cells["uplift"] = np.clip(chosen, MIN_UPLIFT, configuration.uplift_cap)
    cells["recortado"] = (cells["uplift"] != chosen).astype(int)

    # the band: bootstrap of the renewer rows of the group the uplift came from
    generator = np.random.default_rng(configuration.random_seed)
    group_column_of = {UPLIFT_OWN: UPLIFT_CELL_ID_COLUMN, UPLIFT_PARENT: "uplift_padre_id", UPLIFT_CELL: "uplift_celda_mandatory_id"}
    row_groups = {origin: renewers.groupby(column).indices for origin, column in group_column_of.items()}
    group_key_of = {UPLIFT_OWN: UPLIFT_CELL_ID_COLUMN, UPLIFT_PARENT: "uplift_padre_id", UPLIFT_CELL: "uplift_celda_mandatory_id"}
    numerator, denominator = renewers["_numerador"].to_numpy(dtype=float), renewers["_denominador"].to_numpy(dtype=float)
    lows, highs, cache = [], [], {}
    for _, cell_row in cells.iterrows():
        origin = cell_row["uplift_origen"]
        if origin == UPLIFT_GLOBAL:
            lows.append(np.nan); highs.append(np.nan); continue
        group_key = cell_row[group_key_of[origin]]
        if (origin, group_key) not in cache:
            positions = row_groups[origin].get(group_key, np.array([], dtype=int))
            if len(positions) < 2:
                cache[(origin, group_key)] = (np.nan, np.nan)
            else:
                picks = positions[generator.integers(0, len(positions), size=(configuration.uplift_bootstrap_samples, len(positions)))]
                samples = numerator[picks].sum(axis=1) / denominator[picks].sum(axis=1)
                cache[(origin, group_key)] = (float(np.quantile(samples, BAND_LOW)), float(np.quantile(samples, BAND_HIGH)))
        low, high = cache[(origin, group_key)]
        lows.append(low); highs.append(high)
    cells["uplift_bajo"] = np.clip(lows, MIN_UPLIFT, configuration.uplift_cap)
    cells["uplift_alto"] = np.clip(highs, MIN_UPLIFT, configuration.uplift_cap)
    cells.attrs["uplift_global"] = global_uplift
    return cells


def contract_check(renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The contract rule against the past renewals with a known discount, per discount bucket:
    observed uplift, rule uplift, realization ratio (observed / rule) and the share of renewed
    money within ±2 % of the rule."""
    if not configuration.discount_value_column:
        return pd.DataFrame(columns=["tramo", "renovadores", "usd_renovado", "uplift_observado", "uplift_regla",
                                     "ratio_realizacion", "pct_usd_dentro_2pct"])
    known = renewers[renewers[configuration.discount_value_column].notna()].copy()
    known["_regla"] = 1 / (1 - known[configuration.discount_value_column].clip(upper=0.99))
    known["_uplift_fila"] = known["_numerador"] / known["_denominador"]
    known["_dentro"] = ((known["_uplift_fila"] / known["_regla"] - 1).abs() <= CONTRACT_TOLERANCE).astype(float)
    known["_regla_por_denominador"] = known["_regla"] * known["_denominador"]
    known["_usd_dentro"] = known["_dentro"] * known["_numerador"]
    grouped = known.groupby(configuration.discount_bucket_column)
    check = pd.DataFrame({"renovadores": grouped[configuration.renewed_units_col].sum(),
                          "usd_renovado": grouped["_numerador"].sum(),
                          "uplift_observado": grouped["_numerador"].sum() / grouped["_denominador"].sum(),
                          "uplift_regla": grouped["_regla_por_denominador"].sum() / grouped["_denominador"].sum(),
                          "pct_usd_dentro_2pct": grouped["_usd_dentro"].sum() / grouped["_numerador"].sum()})
    check["ratio_realizacion"] = check["uplift_observado"] / check["uplift_regla"]
    return check.reset_index().rename(columns={configuration.discount_bucket_column: "tramo"})


def estimate_uplift(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """The statistical uplift of every cell and the contract check; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the renewers
    renewers = renewer_rows(fine_table, configuration)
    configuration.log_action(STEP_LABEL, 1, f"{len(renewers):,} renewer rows in the closed months "
                                            f"({renewers[configuration.renewed_units_col].sum():,.0f} renewed units, "
                                            f"${renewers['_numerador'].sum():,.0f})")

    # [2] the statistical uplift of every cell
    cells = estimate_cell_uplifts(fine_table, renewers, configuration)
    configuration.log_action(STEP_LABEL, 2, f"{len(cells):,} uplift cells · origins {cells['uplift_origen'].value_counts().to_dict()} · "
                                            f"global uplift {cells.attrs['uplift_global']:.3f}")

    # [3] the contract rule
    check = contract_check(renewers, configuration)
    known_share = renewers[configuration.discount_value_column].notna().mean() if configuration.discount_value_column else 0.0
    configuration.log_action(STEP_LABEL, 3, f"contract rule checked on the {known_share:.0%} of renewer rows with a known "
                                            f"discount" if configuration.discount_value_column else "no exact discount declared")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the uplifts")
    missing = set(fine_table[UPLIFT_CELL_ID_COLUMN]) - set(cells[UPLIFT_CELL_ID_COLUMN])
    configuration.log_check(STEP_LABEL, check_log, "every uplift cell of the fine table has an uplift (future cells included)",
                            not missing and cells["uplift"].notna().all(),
                            failure_detail=f"{len(missing)} cells without uplift", context=f"{len(cells):,} cells")
    configuration.log_check(STEP_LABEL, check_log, f"every uplift is inside [{MIN_UPLIFT}, {configuration.uplift_cap}]",
                            cells["uplift"].between(MIN_UPLIFT, configuration.uplift_cap).all(),
                            failure_detail="uplifts outside the range",
                            context=f"from {cells['uplift'].min():.3f} to {cells['uplift'].max():.3f}; {int(cells['recortado'].sum())} clipped")
    with_band = cells[cells["uplift_bajo"].notna() & (cells["recortado"] == 0)]
    outside = with_band[(with_band["uplift"] < with_band["uplift_bajo"] - 1e-9) | (with_band["uplift"] > with_band["uplift_alto"] + 1e-9)]
    configuration.log_check(STEP_LABEL, check_log, "the band contains the uplift of every cell that has one", outside.empty,
                            failure_detail=f"{len(outside)} cells whose uplift is outside its band",
                            context=f"{len(with_band):,} cells with a band", examples=outside)

    # [5] the tables
    configuration.log_action(STEP_LABEL, 5, "writing the uplift of the cells and the contract check")
    configuration.write_table(STEP_LABEL, check_log, cells, TABLE_UPLIFT_CELLS)
    configuration.write_table(STEP_LABEL, check_log, check, TABLE_CONTRACT_CHECK)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the uplift by origin, and the contract rule by bucket
    configuration.log_action(STEP_LABEL, 7, "uplift cells by origin (propia: ≥ floor renewers; padre: its mandatory dims and "
                                            "discount bucket; celda: its mandatory dims; global):")
    configuration.show_table(cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"),
                                                                uplift_medio=("uplift", "mean"),
                                                                renovadores=("renovadores", "sum")).reset_index())
    if len(check):
        configuration.logger.doc(f"[{STEP_LABEL}] the contract rule 1/(1 − d) against the past renewals with a known discount "
                                 f"(ratio_realizacion 1 = the rule is exact):")
        configuration.show_table(check)
    return cells, check
