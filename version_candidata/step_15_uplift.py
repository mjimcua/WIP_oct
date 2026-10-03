"""
step_15_uplift.py — The revaluation: at what price does a renewer renew?

What falls due has a price (USD due / units due = the pipeline AUV of the row). A renewer
pays a price too (renewed USD / renewed units). The UPLIFT is the second over the first:
1.0 = renews at the same price; 1.43 = pays 43 % more. The forecast of a row is
USD due × renewal rate × uplift.

Two ways to know the uplift of a future row:
  · STATISTICAL (the default, and the only one where the discount is unknown): the uplift
    observed in the past renewals of its UPLIFT CELL (uplift mandatory dims, revaluation
    extras, discount bucket), as a ratio of sums over the renewers (a contract that did not
    renew has no price), weighted by money:
        Σ renewed USD / Σ (the USD that was falling due of the contracts that RENEWED)
    when the extract carries that base (renewed_pipeline_usd_col): the revaluation pure,
    conditional on renewing. Without it, the base is approximated as renewed units × the
    row's pipeline AUV, which is exact only if renewers were worth the row's average: the
    difference between the two bases is the HOMOGENEITY of the cell, measured below.
    Its n is the number of renewers. A cell with fewer than uplift_floor
    renewers takes its PARENT (same mandatory dims and discount bucket, the other extras
    set to '*'), then its mandatory cell, then the whole portfolio. The band of the
    uplift: bootstrap of the renewer rows (p5-p95).
  · CONTRACT (where the exact discount is known): a customer who paid list × (1 − d)
    renews at list, so the uplift is 1 / (1 − d). This step checks the rule against the past
    renewals with a known discount (how much of the renewed money it explains within ±2 %);
    step 16 decides, in the exam months, which path predicts better.

Actions (logged as they are done):
  1. the renewer rows of the closed months, with the base of their uplift
  2. the statistical uplift of every uplift cell: own, parent, cell, global; the band
  3. the contract rule against the past renewals with a known discount
  4. the homogeneity of every cell (only with the exact base): renewers' average due price
     against the cell's, and the approximate uplift against the exact one
  5. check the uplifts                                               checks 1-3
  6. write the tables                                                checks 4-6
  7. count the checks; stop if any failed
  8. show the uplift by origin, the contract rule and the homogeneity by discount bucket
  9. the revaluation of the renewers, read for business: the population of the claim
     (no softcancel, no acquisition, renewing), where its uplift sits (percentiles,
     histogram by series), the discount split and the monthly path with its steps

Checks (logged as they are made, numbered, at the level of their status):
   1. every uplift cell of the fine table has an uplift (future cells included)
   2. every uplift is inside [0.01, uplift_cap]
   3. the band contains the uplift of every cell with its own or its parent's
   4-6. tables sff_uplift_celda, sff_uplift_contrato_check and sff_uplift_homogeneidad
        (the last only with the exact base) written and read back

Output: the uplift of every cell (with its origin and band), the contract check and the
homogeneity · tables sff_uplift_celda, sff_uplift_contrato_check, sff_uplift_homogeneidad.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, join_columns
from vocabulario import (CALENDAR_ROLE_COLUMN, SERIES_ID_COLUMN, TABLE_CONTRACT_CHECK, TABLE_UPLIFT_CELLS,
                         TABLE_UPLIFT_HOMOGENEITY, TRUTH_ROLES, UPLIFT_CELL, UPLIFT_CELL_ID_COLUMN, UPLIFT_GLOBAL,
                         UPLIFT_OWN, UPLIFT_PARENT, WILDCARD)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "15"
STEP_NAME = "UPLIFT"
STEP_PURPOSE = ("estimate at what price a renewer renews relative to what fell due: the statistical uplift of every "
                "uplift cell from its past renewals (borrowing from its parent when it has few renewers), and the check "
                "of the contract rule 1 / (1 − discount) where the discount is known")
STEP_ACTIONS = ["the renewer rows of the closed months, with the base of their uplift",
                "the statistical uplift of every uplift cell: own, parent, cell, global; the band",
                "the contract rule against the past renewals with a known discount",
                "the homogeneity of every cell: the renewers' average due price against the cell's (exact base only)",
                "check the uplifts (checks 1-3)",
                "write the tables (checks 4-6)",
                "count the checks; stop if any failed",
                "show the uplift by origin, the contract rule and the homogeneity by discount bucket",
                "the revaluation of the renewers: population, distribution, discount split, monthly path"]
STEP_OUTPUT = ("the uplift of every uplift cell (origin, band) · the contract check · the homogeneity · "
               "tables sff_uplift_celda, sff_uplift_contrato_check, sff_uplift_homogeneidad")

# ─── named constants ─────────────────────────────────────────────────────────────
MIN_UPLIFT = 0.01
# ─── the revaluation report (action 9): where does the renewers' revaluation sit ──
REPORT_WINDOW_MONTHS = 12          # the distribution is measured over the last N closed months
REPORT_MONTHLY_MONTHS = 24         # the monthly path shows the last N closed months
REPORT_UNITS_FLOOR = 271.0         # a series enters when it averages at least this many units due per month...
REPORT_TOP_SERIES = 50             # ...or is among this many biggest by USD due (the "more value" door)
REPORT_UPLIFT_BUCKETS = [0.0, 0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.20, 1.50, np.inf]
REPORT_HISTOGRAM_WIDTH = 40        # characters of the longest histogram bar
REPORT_STEP_THRESHOLD = 0.03       # a monthly step this big over the previous-12 average is above the threshold...
REPORT_PERSISTENCE_MONTHS = 3      # ...and an increase only when it stays this many months in a row (as step 22)
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
    # the base of the uplift: what the RENEWERS were worth before renewing. Exact when the extract carries it;
    # otherwise the renewed units valued at the row's average due price (the whole row, renewers or not)
    closed["_denominador_aproximado"] = closed[configuration.renewed_units_col] * closed["auv_pipeline"]
    if configuration.renewed_pipeline_usd_col:
        closed["_denominador"] = closed[configuration.renewed_pipeline_usd_col]
        closed = closed[closed["_denominador"] > 0]        # a renewer row without its base cannot opine (warned in step 01)
    else:
        closed["_denominador"] = closed["_denominador_aproximado"]
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



def homogeneity_table(fine_table: pd.DataFrame, renewers: pd.DataFrame, configuration: Config):
    """One row per uplift cell with renewers (exact base only): what the renewers were worth per unit
    against the cell's average due price, and the approximate uplift against the exact one.

    ratio_seleccion = (renewers' due USD / renewed units) / (cell's due USD / cell's due units). At 1 the
    renewers were worth the cell's average and the old approximation was exact; above 1 the dearer renew
    more and the approximation inflated the uplift by that ratio (uplift_aproximado = uplift × ratio).
    Where the ratio moves away from 1 the cell mixes prices: the answer is to segment better, not to add
    a factor to the forecast. Returns None when the exact base is not declared.
    """
    if not configuration.renewed_pipeline_usd_col:
        return None
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.pipeline_units_col] > 0)]
    cell_due = closed.groupby(UPLIFT_CELL_ID_COLUMN).agg(
        vencen_unidades=(configuration.pipeline_units_col, "sum"),
        vencen_usd=(configuration.pipeline_usd_col, "sum"))
    by_cell = renewers.groupby(UPLIFT_CELL_ID_COLUMN).agg(
        renovadores=(configuration.renewed_units_col, "sum"),
        usd_renovadores=("_denominador", "sum"),
        usd_renovado=("_numerador", "sum"),
        base_aproximada=("_denominador_aproximado", "sum"))
    bucket_of_cell = (renewers.drop_duplicates(UPLIFT_CELL_ID_COLUMN)
                      .set_index(UPLIFT_CELL_ID_COLUMN)[configuration.discount_bucket_column]
                      if configuration.discount_bucket_column in renewers.columns else None)
    homogeneity = by_cell.join(cell_due, how="left").reset_index()
    homogeneity["precio_medio_celda"] = homogeneity["vencen_usd"] / homogeneity["vencen_unidades"]
    homogeneity["precio_medio_renovadores"] = homogeneity["usd_renovadores"] / homogeneity["renovadores"]
    homogeneity["ratio_seleccion"] = homogeneity["precio_medio_renovadores"] / homogeneity["precio_medio_celda"]
    homogeneity["uplift_exacto"] = homogeneity["usd_renovado"] / homogeneity["usd_renovadores"]
    homogeneity["uplift_aproximado"] = homogeneity["usd_renovado"] / homogeneity["base_aproximada"]
    if bucket_of_cell is not None:
        homogeneity[configuration.discount_bucket_column] = homogeneity[UPLIFT_CELL_ID_COLUMN].map(bucket_of_cell)
    return homogeneity.drop(columns=["base_aproximada"])



def weighted_percentiles(values: np.ndarray, weights: np.ndarray, points: list) -> list:
    """The value below which each share of the total weight sits (weights all positive)."""
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    cumulative_share = np.cumsum(ordered_weights) / ordered_weights.sum()
    return [float(np.interp(point, cumulative_share, ordered_values)) for point in points]


def histogram_lines(series_table: pd.DataFrame, title: str) -> list:
    """The uplift histogram of a set of series, one text line per bucket: how many series fall in it,
    how much renewed USD, and a bar proportional to the USD share (the money is what matters)."""
    lines = [f"histogram · {title} (bar = % of the renewed USD)",
             f"{'uplift':>15}  {'series':>6}  {'% series':>8}  {'USD renovado':>14}  {'% USD':>6}"]
    assigned = pd.cut(series_table["uplift"], REPORT_UPLIFT_BUCKETS, right=False)
    grouped = series_table.groupby(assigned, observed=False).agg(series=("uplift", "size"), usd=("usd_renovado", "sum"))
    usd_share = grouped["usd"] / max(grouped["usd"].sum(), 1e-9)
    series_share = grouped["series"] / max(grouped["series"].sum(), 1e-9)
    longest = max(usd_share.max(), 1e-9)
    for (interval, row), usd_part, series_part in zip(grouped.iterrows(), usd_share, series_share):
        label = (f"      < {interval.right:.3f}" if interval.left == 0.0
                 else f"      ≥ {interval.left:.3f}" if np.isinf(interval.right)
                 else f"{interval.left:.3f} – {interval.right:.3f}")
        bar = "█" * int(round(REPORT_HISTOGRAM_WIDTH * usd_part / longest)) if row["usd"] > 0 else ""
        lines.append(f"{label:>15}  {int(row['series']):>6}  {series_part:>8.1%}  {row['usd']:>14,.0f}  {usd_part:>6.1%}  {bar}")
    return lines


def revaluation_report(renewers: pd.DataFrame, configuration: Config) -> None:
    """Action 9: the revaluation of the renewers, read for business, on screen only.

    INPUT:   the renewer rows of action 1 (closed months, renewed > 0, with the uplift base in
             _denominador: exact when the extract carries it, approximate otherwise).
    OUTPUT:  nothing returned, nothing written: the funnel of the population (no softcancel, no
             acquisition), where the uplift of its heavy series sits (aggregate, weighted percentiles,
             histogram), the discount split, the 15 biggest series and the monthly path with its steps.
    RULES:   the softcancel filter uses the column named "softcancel" when the extract has one (logged
             when it does not); the acquisition filter uses acquisition_column / acquisition_values of
             the Config. A series enters the distribution when it averages ≥ REPORT_UNITS_FLOOR units
             due per month in the window, or is among the REPORT_TOP_SERIES biggest by USD due.
    EDGE CASES: with no series past the floor, the report says so and stops; a single series still
             gets its percentiles (they are all its own uplift).
    """
    doc = configuration.logger.doc
    closed_periods = sorted(renewers[configuration.period_col].unique())
    window = closed_periods[-REPORT_WINDOW_MONTHS:]
    population = renewers[renewers[configuration.period_col].isin(window)]
    funnel_rows = [("renewer rows of the window", population)]
    if "softcancel" in population.columns:
        population = population[population["softcancel"] == 0]
        funnel_rows.append(("· without softcancel (the row's mark)", population))
    else:
        doc(f"[{STEP_LABEL}] no column named softcancel in the extract: that filter is skipped")
    if configuration.acquisition_column and configuration.acquisition_column in population.columns:
        population = population[~population[configuration.acquisition_column].isin(configuration.acquisition_values)]
        funnel_rows.append((f"· and not an acquisition ({configuration.acquisition_column})", population))
    population = population[population["_denominador"] > 0]
    base_text = ("exact (what the renewers were worth)" if configuration.renewed_pipeline_usd_col
                 else "APPROXIMATE (renewed units × the row's due price: declare renewed_pipeline_usd_col for the exact one)")
    doc(f"[{STEP_LABEL}] window {window[0]} to {window[-1]} · uplift base: {base_text}")
    configuration.show_table(pd.DataFrame([{"population": label, "rows": len(stage),
                                            "usd_renovado": round(float(stage["_numerador"].sum()))}
                                           for label, stage in funnel_rows]))

    # one row per forecast series over its months of the window; the heavy ones
    discount = configuration.discount_value_column
    per_series = (population.groupby(SERIES_ID_COLUMN)
                  .agg(meses=(configuration.period_col, "nunique"),
                       vencen_unidades=(configuration.pipeline_units_col, "sum"),
                       vencen_usd=(configuration.pipeline_usd_col, "sum"),
                       renovadas=(configuration.renewed_units_col, "sum"),
                       usd_renovado=("_numerador", "sum"),
                       base=("_denominador", "sum"),
                       usd_desconocido=("_numerador", lambda values:
                                        values[population.loc[values.index, discount].isna()].sum() if discount else 0.0))
                  .reset_index())
    per_series["unidades_mes"] = per_series["vencen_unidades"] / per_series["meses"]
    per_series["tasa_unidades"] = per_series["renovadas"] / per_series["vencen_unidades"]
    per_series["uplift"] = per_series["usd_renovado"] / per_series["base"]
    biggest = per_series.nlargest(REPORT_TOP_SERIES, "vencen_usd")[SERIES_ID_COLUMN]
    selected = per_series[(per_series["unidades_mes"] >= REPORT_UNITS_FLOOR)
                          | per_series[SERIES_ID_COLUMN].isin(set(biggest))].copy()
    if not len(selected):
        doc(f"[{STEP_LABEL}] no series past the floor ({REPORT_UNITS_FLOOR:,.0f} units due per month): nothing to show")
        return
    covered = selected["usd_renovado"].sum() / max(per_series["usd_renovado"].sum(), 1e-9)
    doc(f"[{STEP_LABEL}] {len(selected):,} series in the distribution (floor {REPORT_UNITS_FLOOR:,.0f} units/month or "
        f"top {REPORT_TOP_SERIES} by USD due): {covered:.0%} of the renewed USD of the population")

    # the claim, measured: the aggregate, the weighted percentiles and the money in each zone
    weights = selected["usd_renovado"].to_numpy(dtype=float)
    uplifts = selected["uplift"].to_numpy(dtype=float)
    p10, p25, p50, p75, p90 = weighted_percentiles(uplifts, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
    share_100_105 = weights[(uplifts >= 1.00) & (uplifts < 1.05)].sum() / weights.sum()
    share_095_105 = weights[(uplifts >= 0.95) & (uplifts < 1.05)].sum() / weights.sum()
    share_above_110 = weights[uplifts >= 1.10].sum() / weights.sum()
    doc(f"[{STEP_LABEL}] aggregate uplift {selected['usd_renovado'].sum() / selected['base'].sum():.3f} · weighted "
        f"percentiles p10 {p10:.3f} · p25 {p25:.3f} · MEDIAN {p50:.3f} · p75 {p75:.3f} · p90 {p90:.3f}")
    doc(f"[{STEP_LABEL}] the claim '100-105 %': {share_100_105:.0%} of the renewed USD sits in [1.00, 1.05) · "
        f"{share_095_105:.0%} in [0.95, 1.05) · {share_above_110:.0%} at or above 1.10")
    if share_above_110 > 0.15:
        doc(f"[{STEP_LABEL}] VALORACIÓN: a heavy tail at or above 1.10. Look at its discount buckets below: a tail in "
            f"high KNOWN buckets is the contract losing its discount (legitimate); one in the unknown bucket is the "
            f"statistical path and deserves the homogeneity table")
    elif share_095_105 > 0.80:
        doc(f"[{STEP_LABEL}] VALORACIÓN: the claim holds: most of the money renews within ±5 % of what it was worth")
    else:
        doc(f"[{STEP_LABEL}] VALORACIÓN: the money is spread beyond ±5 %: the buckets below say whether the spread "
            f"follows the discount (expected) or not")

    if discount:
        known_label = np.where(population[discount].notna(), "discount known", "discount unknown")
        by_known = (population.assign(_corte=known_label).groupby("_corte")
                    .agg(vencen_usd=(configuration.pipeline_usd_col, "sum"), usd_renovado=("_numerador", "sum"),
                         base=("_denominador", "sum")))
        by_known["uplift"] = (by_known["usd_renovado"] / by_known["base"]).round(3)
        configuration.show_table(by_known.reset_index().drop(columns=["base"]))
        by_bucket = (population.groupby(configuration.discount_bucket_column, dropna=False)
                     .agg(usd_renovado=("_numerador", "sum"), base=("_denominador", "sum")))
        by_bucket["uplift"] = (by_bucket["usd_renovado"] / by_bucket["base"]).round(3)
        configuration.show_table(by_bucket.reset_index().drop(columns=["base"]))

    for line in histogram_lines(selected, "every series in the distribution"):
        doc(f"[{STEP_LABEL}] {line}")
    mostly_unknown = selected[selected["usd_desconocido"] / selected["usd_renovado"] > 0.5]
    if len(mostly_unknown):
        for line in histogram_lines(mostly_unknown, "series whose renewed USD is mostly of UNKNOWN discount"):
            doc(f"[{STEP_LABEL}] {line}")

    top = selected.nlargest(15, "vencen_usd")[[SERIES_ID_COLUMN, "meses", "unidades_mes", "vencen_usd",
                                               "tasa_unidades", "uplift"]].copy()
    top["unidades_mes"] = top["unidades_mes"].round(0)
    top["tasa_unidades"] = top["tasa_unidades"].round(3)
    top["uplift"] = top["uplift"].round(3)
    doc(f"[{STEP_LABEL}] the 15 series with the most USD due:")
    configuration.show_table(top)

    # the monthly path of the same population, over a longer window: steps betray a price increase
    monthly_window = closed_periods[-REPORT_MONTHLY_MONTHS:]
    monthly_population = renewers[renewers[configuration.period_col].isin(monthly_window) & (renewers["_denominador"] > 0)]
    if "softcancel" in monthly_population.columns:
        monthly_population = monthly_population[monthly_population["softcancel"] == 0]
    if configuration.acquisition_column and configuration.acquisition_column in monthly_population.columns:
        monthly_population = monthly_population[~monthly_population[configuration.acquisition_column]
                                                .isin(configuration.acquisition_values)]
    monthly = (monthly_population.groupby(configuration.period_col)
               .agg(renovadas=(configuration.renewed_units_col, "sum"), usd_renovado=("_numerador", "sum"),
                    base=("_denominador", "sum")).reset_index())
    monthly["uplift"] = monthly["usd_renovado"] / monthly["base"]
    monthly["media_12_previos"] = monthly["uplift"].shift(1).rolling(window=12, min_periods=6).mean()
    monthly["escalon"] = monthly["uplift"] / monthly["media_12_previos"] - 1
    doc(f"[{STEP_LABEL}] month by month ({len(monthly)} closed, same population): escalon = the month over the average "
        f"of its previous 12")
    shown = monthly.drop(columns=["usd_renovado", "base"]).copy()
    shown["uplift"] = shown["uplift"].round(3)
    shown["media_12_previos"] = shown["media_12_previos"].round(3)
    shown["escalon"] = (100 * shown["escalon"]).round(1)
    configuration.show_table(shown.rename(columns={"escalon": "escalon_pct"}))
    recent = monthly.tail(12).dropna(subset=["escalon"]).reset_index(drop=True)
    above = (recent["escalon"] >= REPORT_STEP_THRESHOLD).to_numpy()
    longest_run, current_run = 0, 0
    for is_above in above:
        current_run = current_run + 1 if is_above else 0
        longest_run = max(longest_run, current_run)
    listed = " · ".join(f"{row[configuration.period_col]} ({row['escalon']:+.1%})"
                        for _, row in recent[above].iterrows())
    if longest_run >= REPORT_PERSISTENCE_MONTHS:
        doc(f"[{STEP_LABEL}] VALORACIÓN: a step of ≥ {REPORT_STEP_THRESHOLD:.0%} held {longest_run} months in a row in the "
            f"last 12 closed ({listed}): the shape of a price increase (its effect lasts one renewal cycle)")
    elif above.any():
        doc(f"[{STEP_LABEL}] VALORACIÓN: months above {REPORT_STEP_THRESHOLD:.0%} but never {REPORT_PERSISTENCE_MONTHS} in a "
            f"row ({listed}): noise, not an increase")
    elif len(recent):
        doc(f"[{STEP_LABEL}] VALORACIÓN: no step of ≥ {REPORT_STEP_THRESHOLD:.0%} in the last 12 closed months: no sign "
            f"of a price increase in this population")

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
    """The statistical uplift of every cell, the contract check and the homogeneity; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the renewers
    renewers = renewer_rows(fine_table, configuration)
    base_text = (f"exact base: {configuration.renewed_pipeline_usd_col}" if configuration.renewed_pipeline_usd_col
                 else "approximate base: renewed units × the row's pipeline AUV (renewed_pipeline_usd_col not declared)")
    configuration.log_action(STEP_LABEL, 1, f"{len(renewers):,} renewer rows in the closed months "
                                            f"({renewers[configuration.renewed_units_col].sum():,.0f} renewed units, "
                                            f"${renewers['_numerador'].sum():,.0f}) · {base_text}")

    # [2] the statistical uplift of every cell
    cells = estimate_cell_uplifts(fine_table, renewers, configuration)
    configuration.log_action(STEP_LABEL, 2, f"{len(cells):,} uplift cells · origins {cells['uplift_origen'].value_counts().to_dict()} · "
                                            f"global uplift {cells.attrs['uplift_global']:.3f}")

    # [3] the contract rule
    check = contract_check(renewers, configuration)
    known_share = renewers[configuration.discount_value_column].notna().mean() if configuration.discount_value_column else 0.0
    configuration.log_action(STEP_LABEL, 3, f"contract rule checked on the {known_share:.0%} of renewer rows with a known "
                                            f"discount" if configuration.discount_value_column else "no exact discount declared")

    # [4] the homogeneity: where renewers were worth the cell's average, the approximation was exact
    homogeneity = homogeneity_table(fine_table, renewers, configuration)
    if homogeneity is not None:
        weighted_ratio = (homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"]).sum() / homogeneity["usd_renovadores"].sum()
        configuration.log_action(STEP_LABEL, 4, f"homogeneity of {len(homogeneity):,} cells with renewers: the renewers' due "
                                                f"price over the cell's averages {weighted_ratio:.3f} weighted by their USD "
                                                f"(1.000 = renewers worth the cell's average: the old approximation was exact)")
    else:
        configuration.log_action(STEP_LABEL, 4, "no exact base declared: the homogeneity cannot be measured")

    # [5] the checks
    configuration.log_action(STEP_LABEL, 5, "checking the uplifts")
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

    # [6] the tables
    configuration.log_action(STEP_LABEL, 6, "writing the tables")
    configuration.write_table(STEP_LABEL, check_log, cells, TABLE_UPLIFT_CELLS)
    configuration.write_table(STEP_LABEL, check_log, check, TABLE_CONTRACT_CHECK)
    if homogeneity is not None:
        configuration.write_table(STEP_LABEL, check_log, homogeneity, TABLE_UPLIFT_HOMOGENEITY)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [8] the uplift by origin, the contract rule by bucket, the homogeneity by bucket
    configuration.log_action(STEP_LABEL, 8, "uplift cells by origin (propia: ≥ floor renewers; padre: its mandatory dims and "
                                            "discount bucket; celda: its mandatory dims; global):")
    configuration.show_table(cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"),
                                                                uplift_medio=("uplift", "mean"),
                                                                renovadores=("renovadores", "sum")).reset_index())
    if len(check):
        configuration.logger.doc(f"[{STEP_LABEL}] the contract rule 1/(1 − d) against the past renewals with a known discount "
                                 f"(ratio_realizacion 1 = the rule is exact):")
        configuration.show_table(check)
    # [9] the revaluation of the renewers, read for business
    configuration.log_action(STEP_LABEL, 9, "the revaluation of the renewers (no softcancel, no acquisition, renewing): "
                                            "population, distribution, discount split, monthly path")
    revaluation_report(renewers, configuration)

    if homogeneity is not None and configuration.discount_bucket_column in homogeneity.columns:
        by_bucket = (homogeneity.assign(_pesado=homogeneity["ratio_seleccion"] * homogeneity["usd_renovadores"])
                     .groupby(configuration.discount_bucket_column)
                     .agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), usd_renovadores=("usd_renovadores", "sum"),
                          _pesado=("_pesado", "sum")))
        by_bucket["ratio_seleccion"] = by_bucket.pop("_pesado") / by_bucket["usd_renovadores"]
        configuration.logger.doc(f"[{STEP_LABEL}] the homogeneity by discount bucket (ratio_seleccion 1 = renewers worth "
                                 f"the cell's average; above 1 = the dearer renew more):")
        configuration.show_table(by_bucket.reset_index())
    return cells, check, homogeneity
