"""raw_data_validation.py — SFF v2 · RUN · phase 0: the raw becomes computable.

"Load, review, condition, and establish a reference to compare against" (DISENO_V2 §5).
Not a single datum is modified: the raw is immutable — label, never amputate. This
module is the OPERATIONAL part of phase 0, the minimum that must run every month before
any rate can be computed:

  0.1  validate_raw                 exhaustive contract + blocking checks; period normalized
       apply_current_month_doctrine the running month is the FIRST month to project;
                                    its early results are wiped
  0.2  build_fine_table             raw rows + fu_id / comb_id / fu_comb_key and their keys
       aggregate_to_forecast_units  one row per forecast unit, measures SUMMED, money
                                    conserved to the cent, fu_id unique
       build_key_lookups            id ↔ key tables so the keys work outside the framework
  0.3  label_universe_and_routes    universe · coverage pattern · route, as LABELS

The ANALYTICAL part of phase 0 — the immutable reference with the binomial worst-case
error per forecast unit — lives in `support_reference.py`: it reports, it does
not compute anything the forecast needs.

Every function here has one job, a contract docstring, and a test in
`test_raw_data_validation.py`. Persisted column names stay in Spanish (`universo`,
`cobertura`, `ruta`): they are the contract of the tables and of the BI.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import COMBINED_ID_SEPARATOR, ID_FIELD_SEPARATOR, Config, hash_key, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)


# ─── named constants ─────────────────────────────────────────────────────────────
# The three roles the SQL extract labels every row with. An empty role is an alarm,
# not a stop: a raw with no test rows is legal (e.g. a very short history).
EXPECTED_DATASET_ROLES = (ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION)

# The role of every row of the future; the current month is reassigned to it.

# Raw values that mean "this row belongs to the current month". The extract may
# encode the flag as int, bool or text depending on the SQL client.
CURRENT_MONTH_TRUTHY_VALUES = (1, True, "1", "true", "yes", "si")

# Money conservation tolerance between the fine table and the forecast units table:
# floating sums of hundreds of thousands of rows.
MONEY_CONSERVATION_TOLERANCE_USD = 1e-6

# The comb_id of a raw with no extra_revalorizacion declared: one single combination.
NO_COMBINATION_ID = "na"

# Value of the time-series flag that puts a row in the time_series universe.
TIME_SERIES_FLAG_ON = 1

# Universe labels (persisted in `universo`).

# Route labels (persisted in `ruta`).

# Separator of the coverage pattern, e.g. "projection_test_train".
COVERAGE_SEPARATOR = "_"


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.1 · CONTRACT AND CONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════════

def validate_raw(raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Guarantee the raw is usable before computing anything; stop with EVERY problem.

    INPUT:   raw — every column of the extract, as `read_raw` returned it ·
             configuration — uses validate_column_contract, period_col, pipeline_*_col,
             renewed_*_col, rate_series_columns, extra_revalorizacion, dataset_role_col.
    OUTPUT:  a copy of the raw with `period_col` normalized to monthly Period. No row
             added or removed, no other column touched.
    RULES:   the checks are blocking and are collected, not short-circuited: one run
             reports every problem. Money must be non-negative; a renewal with USD ≤ 0
             is forbidden (renewing for free does not exist: it would be a zero
             uplift); a null in a dimension is forbidden (it breaks the ids). An empty
             role is printed as a warning, not raised.
    EDGE CASES: one unparseable period stops the run (pandas raises); a raw with no
             rows passes the contract and returns empty; a declared column MISSING from
             the raw stops the run naming it.
    CONSOLE: the contract line (columns, rows) and one warning per empty role.
    STEPS:
      [1] Exhaustive column contract (P3): every column has exactly one role.
      [2] Period normalized to monthly.
      [3] Money: pipeline non-negative; renewed USD > 0 wherever renewed units > 0.
      [4] Dimensions: no nulls.
      [5] Stop with the full list if anything failed.
      [6] Role census with an empty-role alarm.
    """
    # [1] a column with no role stops here, before anything is computed; and the
    #     columns the computation cannot live without must be present
    configuration.validate_column_contract(raw)
    required_columns = ([configuration.period_col, configuration.dataset_role_col, configuration.current_month_col,
                         configuration.flag_time_series_col] + configuration.core_measures
                        + configuration.rate_series_columns + configuration.extra_revalorizacion)
    missing_columns = [column for column in required_columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"validate_raw: required columns missing from the raw: {missing_columns}")
    declared_column_count = len(raw.columns)

    # [2] the period is monthly from here on; the raw itself is never modified
    validated = raw.copy()
    validated[configuration.period_col] = pd.PeriodIndex(
        validated[configuration.period_col].astype(str), freq="M")

    blocking_problems = []

    # [3] money: nothing negative, no renewal for free
    for money_column in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
        if (validated[money_column] < 0).any():
            blocking_problems.append(f"{money_column} has negative values")
    renewer_mask = validated[configuration.renewed_units_col].fillna(0) > 0
    renewed_usd_of_renewers = validated.loc[renewer_mask, configuration.renewed_usd_col].fillna(0)
    if (renewed_usd_of_renewers <= 0).any():
        blocking_problems.append(
            "renewal with USD<=0 (zero uplift forbidden: renewing for free does not exist)")

    # [4] dimensions: a null would silently produce a broken id (see join_columns)
    dimension_columns = configuration.rate_series_columns + configuration.extra_revalorizacion
    null_counts_by_dimension = {column_name: int(validated[column_name].isna().sum())
                                for column_name in dimension_columns
                                if validated[column_name].isna().any()}
    if null_counts_by_dimension:
        blocking_problems.append(
            f"nulls in dimensions {null_counts_by_dimension} — an empty dimension breaks the ids")

    # [5] everything at once
    if blocking_problems:
        raise ValueError("validate_raw: " + "; ".join(blocking_problems))

    # [6] the roles the extract labeled; an empty one is worth a look, not a stop
    role_census = validated[configuration.dataset_role_col].value_counts().to_dict()
    for expected_role in EXPECTED_DATASET_ROLES:
        if role_census.get(expected_role, 0) == 0:
            print(f"[0.1] WARNING role '{expected_role}' is EMPTY in the raw: check the SQL labeling")
    print(f"[0.1] contract OK: {declared_column_count} columns declared · {len(validated):,} rows")
    return validated


def apply_current_month_doctrine(validated: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The calendar of roles is decided from the CURRENT MONTH; the raw's roles are overwritten.

    INPUT:   validated — the frame returned by `validate_raw` · configuration — uses
             current_month_col, dataset_role_col, period_col, renewed_*_col, reacq_*_col,
             pending_close_months, test_months.
    OUTPUT:  a copy where dataset_role is: ROLE_PROJECTION for the current month and later
             (their results wiped), ROLE_PENDING for the `pending_close_months` just
             before it (results KEPT but never used to learn or to evaluate), ROLE_TEST for
             the `test_months` before those, ROLE_TRAIN for everything earlier.
    RULES:   the current month is the first month of the future: a month still running
             cannot be truth. The month before it is not closed either (renewals land
             after expiry), so it is neither truth nor future: pending. The exam is the
             most recent closed months; the forecast learns from every closed month.
    EDGE CASES: a raw with no current-month flag falls back to the raw's own roles and
             only wipes what projection rows carried. Reacquisition columns are wiped
             only when they exist.
    CONSOLE: the calendar (which months are what), rows reassigned, what was wiped.
    STEPS:
      [1] Detect the current month.
      [2] Assign the roles from it.
      [3] Wipe the early results of projection rows.
      [4] Report.
    """
    conditioned = validated.copy()
    period_column, role_column = configuration.period_col, configuration.dataset_role_col

    # [1] the flag may arrive as int, bool or text
    flag_values = conditioned[configuration.current_month_col]
    current_month_mask = flag_values.isin(CURRENT_MONTH_TRUTHY_VALUES) | (flag_values == 1)
    current_periods = sorted(conditioned.loc[current_month_mask, period_column].unique())

    # [2] the calendar from the current month
    original_roles = conditioned[role_column].copy()
    # step 0 of the core table: what the RAW itself said, kept before any change so the
    # core can tell "what the extract said" next to "what the framework uses"
    conditioned["s0_rol"] = original_roles.astype(str)
    conditioned["s0_renovados_unidades"] = conditioned[configuration.renewed_units_col]
    conditioned["s0_renovados_usd"] = conditioned[configuration.renewed_usd_col]
    if current_periods:
        current = current_periods[0]
        pending_start = current - int(configuration.pending_close_months)
        test_start = pending_start - int(configuration.test_months)
        periods = conditioned[period_column]
        conditioned[role_column] = np.select(
            [periods >= current, periods >= pending_start, periods >= test_start],
            [ROLE_PROJECTION, ROLE_PENDING, ROLE_TEST], default=ROLE_TRAIN)
        calendar = (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{pending_start - 1} ({configuration.test_months} months) · "
                    f"{ROLE_PENDING} {pending_start}..{current - 1} · {ROLE_PROJECTION} ≥ {current}")
    else:
        conditioned[role_column] = conditioned[role_column].map(RAW_ROLE_MAP).fillna(conditioned[role_column])
        calendar = "no current-month flag: the raw's own roles are kept (mapped to the vocabulary)"
    reassigned_row_count = int((conditioned[role_column] != original_roles).sum())

    # [3] the future must look like it has not started: wipe what was already booked
    projection_mask = conditioned[role_column] == ROLE_PROJECTION
    wipe_columns = [column_name for column_name in (configuration.renewed_units_col, configuration.renewed_usd_col,
                                                    configuration.reacq_units_col, configuration.reacq_usd_col)
                    if column_name in conditioned.columns]
    early_results = conditioned.loc[projection_mask, wipe_columns].fillna(0)
    rows_with_early_results = int((early_results != 0).any(axis=1).sum())
    wiped_renewed_units = float(conditioned.loc[projection_mask, configuration.renewed_units_col].fillna(0).sum())
    wiped_renewed_usd = float(conditioned.loc[projection_mask, configuration.renewed_usd_col].fillna(0).sum())
    conditioned.loc[projection_mask, wipe_columns] = np.nan

    # [4] what happened, in numbers
    final_role_census = conditioned[role_column].value_counts().to_dict()
    print(f"[0.1] calendar from the current month {[str(p) for p in current_periods]}: {calendar}")
    print(f"[0.1] {reassigned_row_count:,} rows changed role against the raw · projection cleansing: {rows_with_early_results} rows carried "
          f"early results — wiped {wiped_renewed_units:,.0f} renewed units / ${wiped_renewed_usd:,.0f} (reacquisitions included)")
    print(f"[0.1] roles after conditioning {final_role_census} · from here on, THIS is the raw")
    return conditioned


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.2 · THE TWO TABLES AND THEIR KEYS
# ═══════════════════════════════════════════════════════════════════════════════════

def build_fine_table(conditioned: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The fine table: every raw row, plus the identities that let it be joined.

    INPUT:   conditioned — the frame returned by `apply_current_month_doctrine` ·
             configuration — uses rate_series_columns, extra_revalorizacion, period_col.
    OUTPUT:  a copy with five new columns: fu_id, comb_id, fu_key, comb_key, fu_comb_key.
             Same rows, same order, no measure touched.
    RULES:   fu_id = rate series columns | month (the forecast unit the row belongs to);
             comb_id = extra_revalorizacion joined, or "na" if none declared (the
             revaluation combination of the row); keys are `hash_key` of each id;
             fu_comb_key = hash_key(fu_id || comb_id) — ONE column so the BI joins the
             raw anchor to the key bridge with one relationship.
    EDGE CASES: several raw rows can share fu_id (they differ in the extras): that is
             the point of the fine table. A raw with no extras has one comb_id.
    CONSOLE: nothing (the aggregation step prints the summary of both tables).
    STEPS:
      [1] fu_id and comb_id.
      [2] The three keys.
      [3] One row per (unit, combination); no key collision.
    """
    fine_table = conditioned.copy()

    # [1] identities: which forecast unit, which revaluation combination
    fine_table["fu_id"] = (join_columns(fine_table, configuration.rate_series_columns)
                           + ID_FIELD_SEPARATOR
                           + fine_table[configuration.period_col].astype(str))
    if configuration.extra_revalorizacion:
        fine_table["comb_id"] = join_columns(fine_table, configuration.extra_revalorizacion)
    else:
        fine_table["comb_id"] = NO_COMBINATION_ID

    # [2] keys: derived, never assigned (P7)
    fine_table["fu_key"] = fine_table["fu_id"].map(hash_key)
    fine_table["comb_key"] = fine_table["comb_id"].map(hash_key)
    fine_table["fu_comb_key"] = (fine_table["fu_id"] + COMBINED_ID_SEPARATOR
                                 + fine_table["comb_id"]).map(hash_key)

    # [3] every raw row is one (unit, combination): a duplicate means the extract has two
    #     rows for the same thing, or the hash collided — both must stop the run here
    duplicated_pairs = fine_table.duplicated(["fu_id", "comb_id"]).sum()
    assert duplicated_pairs == 0, (
        f"{duplicated_pairs} raw rows share the same forecast unit AND combination: the extract "
        f"is not at the declared grain (or the taxonomy misses a column)")
    assert fine_table["fu_comb_key"].is_unique, "fu_comb_key collision: two different (unit, combination) hashed alike"
    return fine_table


def aggregate_to_forecast_units(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per forecast unit: the rate-branch table, ratios coherent by construction.

    INPUT:   fine_table — from `build_fine_table` · configuration — uses
             rate_series_columns, period_col, dataset_role_col, current_month_col,
             flag_time_series_col, core_measures, pipeline_usd_col.
    OUTPUT:  the forecast units table: rate series columns + period + role + current
             month flag + time-series flag + the four core measures SUMMED + fu_id,
             fu_key.
    RULES:   measures are summed (`min_count=1`: a group whose measure is all NaN stays
             NaN, it does not become 0). Ratios are recomputed downstream from the sums,
             never averaged. Two assertions guard the contract: money conserved to
             the cent between the two tables, and fu_id unique (a duplicate means the
             role/flag columns vary inside a forecast unit, i.e. the extract is
             inconsistent or the taxonomy misdeclared).
    EDGE CASES: an empty fine table yields an empty table and passes both assertions.
    CONSOLE: rows of both tables and the conserved pipeline USD.
    STEPS:
      [1] Group by the forecast-unit identity plus its context columns; sum measures.
      [2] fu_id and fu_key of every group.
      [3] Conservation: Σ pipeline USD fine == Σ pipeline USD units, or stop.
      [4] fu_id unique, or stop.
    """
    # [1] the context columns are constant inside a forecast unit; if they are not,
    #     the uniqueness assertion below is what catches it
    grouping_columns = (configuration.rate_series_columns
                        + [configuration.period_col, configuration.dataset_role_col,
                           configuration.current_month_col, configuration.flag_time_series_col])
    forecast_units = (fine_table.groupby(grouping_columns, as_index=False, observed=True)
                      [configuration.core_measures].sum(min_count=1))

    # [2] identity of each unit
    forecast_units["fu_id"] = (join_columns(forecast_units, configuration.rate_series_columns)
                               + ID_FIELD_SEPARATOR
                               + forecast_units[configuration.period_col].astype(str))
    forecast_units["fu_key"] = forecast_units["fu_id"].map(hash_key)

    # [3] not a cent is lost between the two tables
    fine_pipeline_usd = fine_table[configuration.pipeline_usd_col].sum()
    units_pipeline_usd = forecast_units[configuration.pipeline_usd_col].sum()
    assert abs(units_pipeline_usd - fine_pipeline_usd) < MONEY_CONSERVATION_TOLERANCE_USD, (
        f"conservation BROKEN: fine ${fine_pipeline_usd:,.2f} vs units ${units_pipeline_usd:,.2f}")

    # [4] one row per forecast unit
    assert forecast_units["fu_id"].is_unique, (
        "duplicated fu_id in the forecast units table: role or flag columns vary inside "
        "a unit (inconsistent extract) or the taxonomy is misdeclared")

    print(f"[0.2] fine {len(fine_table):,} rows → forecast units {len(forecast_units):,} · "
          f"pipeline conserved (${units_pipeline_usd:,.0f})")
    return forecast_units


def build_key_lookups(forecast_units: pd.DataFrame, fine_table: pd.DataFrame) -> tuple:
    """id ↔ key tables, so the hash keys can be resolved outside the framework.

    INPUT:   forecast_units — from `aggregate_to_forecast_units` · fine_table — from
             `build_fine_table`.
    OUTPUT:  (fu_lookup, comb_lookup): two-column frames, one row per distinct id.
    RULES:   the lookups are the only place where an id and its key sit side by side
             for the BI; every other table carries the key.
    EDGE CASES: none.
    CONSOLE: the size of both lookups.
    STEPS:
      [1] Distinct forecast-unit ids.
      [2] Distinct revaluation combinations.
    """
    # [1] one row per forecast unit
    fu_lookup = forecast_units[["fu_id", "fu_key"]].drop_duplicates()

    # [2] one row per revaluation combination
    comb_lookup = fine_table[["comb_id", "comb_key"]].drop_duplicates()

    print(f"[0.2] lookups: {len(fu_lookup)} forecast units · {len(comb_lookup)} combinations")
    return fu_lookup, comb_lookup


# ═══════════════════════════════════════════════════════════════════════════════════
# 0.3 · UNIVERSE, COVERAGE AND ROUTE (labels, never filters)
# ═══════════════════════════════════════════════════════════════════════════════════

def route_from_coverage(coverage_pattern: str) -> str:
    """The management route of a series, read from which roles it has.

    INPUT:   coverage_pattern — the roles of the series joined with "_", sorted.
    OUTPUT:  one of TREATMENT_PREDICTABLE, TREATMENT_FUTURE_ONLY, TREATMENT_HISTORY_ONLY.
    RULES:   no projection nor pending rows → nothing to predict → no_impact (kept: the raw is labeled, never amputated);
             something to predict and a closed month (train or test) → trainable;
             something to predict without any closed month → heuristic (the assembly
             cascade gives it its cell's rate).
    EDGE CASES: a pending_close month alone is not history (not closed) but is a month
             to predict.
    CONSOLE: nothing.
    STEPS:
      [1] No future: no impact.
      [2] History or not.
    """
    # [1] nothing to project
    if ROLE_PROJECTION not in coverage_pattern and ROLE_PENDING not in coverage_pattern:
        return TREATMENT_HISTORY_ONLY

    # [2] with a closed month it is trainable; without, heuristic
    if ROLE_TRAIN in coverage_pattern or ROLE_TEST in coverage_pattern:
        return TREATMENT_PREDICTABLE
    return TREATMENT_FUTURE_ONLY


def label_universe_and_routes(forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Label every series with its universe, its coverage pattern and its route.

    INPUT:   forecast_units — from `aggregate_to_forecast_units` · configuration — uses
             flag_time_series_col, rate_series_columns, dataset_role_col.
    OUTPUT:  a copy with four new columns: `universo`, `fs_id`, `cobertura`, `ruta`.
             Same rows: nothing is filtered (P4: label, never amputate).
    RULES:   universo = time_series when the flag is 1, else normal · fs_id = the rate
             series columns joined (the series the unit belongs to) · cobertura = the
             sorted, "_"-joined set of roles the series has · ruta = route_from_coverage.
    EDGE CASES: a series present in one role only gets a one-word coverage.
    CONSOLE: the census of routes per series.
    STEPS:
      [1] Universe from the flag.
      [2] Series identity.
      [3] Coverage pattern per series, merged back onto every unit.
      [4] Route from the coverage; census.
    """
    labeled_units = forecast_units.copy()

    # [1] two universes; the time_series one gets its own treatment later
    labeled_units["universo"] = np.where(
        labeled_units[configuration.flag_time_series_col] == TIME_SERIES_FLAG_ON,
        UNIVERSE_TIME_SERIES, UNIVERSE_NORMAL)

    # [2] which series each unit belongs to
    labeled_units["fs_id"] = join_columns(labeled_units, configuration.rate_series_columns)

    # [3] which roles the series covers
    coverage_by_series = (labeled_units.groupby("fs_id")[configuration.dataset_role_col]
                          .agg(lambda roles: COVERAGE_SEPARATOR.join(sorted(set(roles))))
                          .rename("cobertura"))
    labeled_units = labeled_units.merge(coverage_by_series, on="fs_id")

    # [4] the route is a label the later phases read; nobody is dropped
    labeled_units["ruta"] = labeled_units["cobertura"].map(route_from_coverage)
    route_census = labeled_units.drop_duplicates("fs_id")["ruta"].value_counts().to_dict()
    print(f"[0.3] routes per series: {route_census} — no_impact is labeled, never dropped")
    return labeled_units
