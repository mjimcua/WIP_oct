"""analysis_discount_churn.py — SFF v3 · ANALYSIS · is the discount a driver of churn?

The belief in the company: a discounted customer renews at full price, so the price
jump at renewal makes them churn. The alternative: churn is multi-causal, and for some
customers (the ones who never installed the product) the discount is irrelevant. This
module tests both with the aggregated data we have, in four steps, each one a table and
a plain sentence:

  1. RATE BY DISCOUNT BUCKET, raw and STANDARDIZED. The raw rate by bucket mixes the
     composition of the cells (a bucket may live in one region). The standardized rate
     fixes the cell mix: within every mandatory cell, the rate of each bucket is weighted
     by the cell's share of the portfolio, and the buckets are then compared on the same
     footing. With Wilson intervals, so a 2 pp gap on 300 contracts is not a finding.
  2. DISCOUNT × SIGNAL STATE. The same comparison inside each signal state (neutral,
     no_instalado, dormant, softcancel…): does the discount gap survive among the
     customers who never installed? If their rate is low with and without discount,
     the flag dominates and the discount is not the driver there.
  3. ADJUSTED EFFECT AND RELATIVE IMPORTANCE. A weighted regression of logit(rate) on
     cell + signal state + newcust + discount bucket (weights = contracts): the effect of
     each bucket once cell and signals are held fixed, and how much of the variance each
     group of variables explains alone and on top of the others (R² increments). This is
     the "multi-causal" question answered with a number.
  4. PRICE JUMP VS RATE. Per cell × bucket, the renewal price ratio (renewed $ / pipeline
     $, the uplift) against the rate: if the price jump drives churn, the cells with the
     biggest jump renew worst inside the same bucket.

Everything is computed on closed months, at the grain of the fine table (cell ×
discount × newcust × signals), with no customer-level data: proportions and their
binomial errors are exact at that grain. Nothing here changes the forecast.

Tables: discount_churn_by_bucket, discount_churn_by_state, discount_churn_adjusted,
discount_churn_importance, discount_churn_price. Report chapter "8 · Precio y churn".
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from binomial_reference import logit, wilson_interval

# ─── named constants ─────────────────────────────────────────────────────────────
MIN_UNITS_PER_BUCKET = 30          # a bucket × cell with fewer contracts is not compared
NEUTRAL_STATE = "neutro"
MIXED_STATE = "mixto"


DISCOUNT_BUCKET_WIDTH = 0.10           # exact discounts are bucketed in 10 pp steps for the analysis
NO_DISCOUNT_BUCKET = "0"                # the bucket of discount = 0 (list price): the reference


def bucket_from_exact_discount(values: pd.Series) -> pd.Series:
    """0 → '0'; 0.3 → '0.3-0.4'; unknown → 'desconocido'."""
    exact = pd.to_numeric(values, errors="coerce")
    lower = np.floor(exact / DISCOUNT_BUCKET_WIDTH) * DISCOUNT_BUCKET_WIDTH
    labels = np.where(exact == 0, NO_DISCOUNT_BUCKET, [f"{l:.1f}-{l + DISCOUNT_BUCKET_WIDTH:.1f}" if np.isfinite(l) else "desconocido" for l in lower])
    return pd.Series(np.where(exact.isna(), "desconocido", labels), index=values.index)


def discount_bucket_column(configuration: Config) -> str:
    """The column with the discount bucket: the exact discount column when it exists (the
    analysis buckets it in 10 pp steps, 0 = the reference), else `discount_column`, else the
    first extra de revalorización whose name contains 'disc'."""
    if getattr(configuration, "discount_value_column", None):
        return configuration.discount_value_column
    if getattr(configuration, "discount_column", None):
        return configuration.discount_column
    for column in configuration.extra_revalorizacion:
        if "disc" in column.lower():
            return column
    raise ValueError("no discount column: set `discount_column` in the configuration")


def state_of(fine: pd.DataFrame, configuration: Config) -> pd.Series:
    """Signal state per row: 'neutro', the single active flag, or 'mixto'."""
    flags = list(configuration.structural_timevarying_dims)
    active = pd.DataFrame({flag: fine[flag].isin(configuration.timevarying_positive_values) for flag in flags})
    count = active.sum(axis=1)
    single = active.idxmax(axis=1).where(count == 1, "")
    return pd.Series(np.where(count == 0, NEUTRAL_STATE, np.where(count == 1, single, MIXED_STATE)), index=fine.index)


def closed_rows(fine: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Closed months with a pipeline, plus the helper columns: celda, estado, tramo (bucket)."""
    role = configuration.dataset_role_col
    frame = fine[fine[role].isin(TRUTH_ROLES) & (fine[configuration.pipeline_units_col] > 0)].copy()
    frame["celda"] = join_columns(frame, configuration.business_mandatory_dims)
    frame["estado"] = state_of(frame, configuration)
    column = discount_bucket_column(configuration)
    if column == getattr(configuration, "discount_value_column", None):
        frame["tramo"] = bucket_from_exact_discount(frame[column])
        frame = frame[frame["tramo"] != "desconocido"]           # unknown discounts cannot be compared
    else:
        frame["tramo"] = frame[column].astype(str)
    return frame


def rate_with_interval(renewed: float, units: float, z: float) -> tuple:
    rate = renewed / units if units > 0 else np.nan
    low, high = wilson_interval(rate, units, z) if units > 0 else (np.nan, np.nan)
    return rate, low, high


def rate_by_bucket(frame: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 1: raw and standardized rate per discount bucket, with Wilson intervals.

    Standardized = Σ_cell share_cell × rate(cell, bucket), over the cells where the bucket
    has ≥ MIN_UNITS_PER_BUCKET contracts; share_cell = the cell's share of all contracts.
    Reported next to the raw rate so the composition effect is visible as their difference."""
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    by_cell_bucket = frame.groupby(["celda", "tramo"]).agg(units=(pipe, "sum"), renewed=(ren, "sum"), usd=(configuration.pipeline_usd_col, "sum")).reset_index()
    by_cell_bucket["rate"] = by_cell_bucket["renewed"] / by_cell_bucket["units"]
    cell_share = frame.groupby("celda")[pipe].sum()
    cell_share = cell_share / cell_share.sum()
    rows = []
    for bucket, block in by_cell_bucket.groupby("tramo"):
        raw_rate, raw_low, raw_high = rate_with_interval(block["renewed"].sum(), block["units"].sum(), configuration.z)
        enough = block[block["units"] >= MIN_UNITS_PER_BUCKET]
        weights = cell_share.reindex(enough["celda"]).to_numpy()
        standardized = float(np.sum(weights * enough["rate"].to_numpy()) / max(weights.sum(), 1e-9)) if len(enough) else np.nan
        rows.append(dict(tramo=bucket, contratos=float(block["units"].sum()), usd=float(block["usd"].sum()), celdas=int(len(enough)),
                         tasa_bruta=raw_rate, tasa_bruta_low=raw_low, tasa_bruta_high=raw_high, tasa_estandarizada=standardized,
                         cobertura_estandarizada=float(weights.sum()) if len(enough) else 0.0))
    table = pd.DataFrame(rows).sort_values("tramo").reset_index(drop=True)
    table["efecto_composicion_pp"] = 100 * (table["tasa_bruta"] - table["tasa_estandarizada"])
    return table


def rate_by_bucket_and_state(frame: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 2: rate per signal state × discount bucket (standardized by cell), and the gap of
    every bucket to the reference bucket inside each state, in pp with its z."""
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    reference = NO_DISCOUNT_BUCKET if getattr(configuration, "discount_value_column", None) else str(getattr(configuration, "no_discount_value", None) or sorted(frame["tramo"].unique())[0])
    rows = []
    for state, block in frame.groupby("estado"):
        per_bucket = rate_by_bucket(block, configuration)
        per_bucket["tasa_comparada"] = per_bucket["tasa_estandarizada"].fillna(per_bucket["tasa_bruta"])   # standardized where possible, raw where the cells are too small
        per_bucket["base_comparacion"] = np.where(per_bucket["tasa_estandarizada"].notna(), "estandarizada", "bruta")
        base = per_bucket[per_bucket["tramo"] == reference]
        base_rate = float(base["tasa_comparada"].iloc[0]) if len(base) and pd.notna(base["tasa_comparada"].iloc[0]) else np.nan
        base_units = float(base["contratos"].iloc[0]) if len(base) else 0.0
        for _, row in per_bucket.iterrows():
            gap = 100 * (row["tasa_comparada"] - base_rate) if np.isfinite(base_rate) and pd.notna(row["tasa_comparada"]) else np.nan
            se = 100 * np.sqrt(max(row["tasa_bruta"] * (1 - row["tasa_bruta"]), 0.0475) / max(row["contratos"], 1) + max(base_rate * (1 - base_rate), 0.0475) / max(base_units, 1)) if np.isfinite(gap) else np.nan
            rows.append(dict(estado=state, tramo=row["tramo"], contratos=row["contratos"], usd=row["usd"], tasa_bruta=row["tasa_bruta"],
                             tasa_estandarizada=row["tasa_estandarizada"], base_comparacion=row["base_comparacion"], tramo_referencia=reference,
                             hueco_pp=gap, hueco_se_pp=se, z=gap / se if np.isfinite(gap) and se > 0 else np.nan))
    return pd.DataFrame(rows)


def adjusted_effects(frame: pd.DataFrame, configuration: Config) -> tuple:
    """Step 3: weighted regression of logit(rate) on cell + state + newcust + bucket.

    OUTPUT: (effects: bucket → effect in pp at the portfolio's mean rate, with se and z,
             cell, state and newcust held fixed;  importance: for each group of variables,
             R² alone and the R² lost when it is dropped from the full model.)
    RULES:  rows = cell × state × newcust × bucket aggregates over the closed months;
            weights = contracts; the reference bucket is the no-discount one. It is a
            descriptive linear model on the logit, not a causal claim: it says how much
            the buckets differ once the other groups are held fixed."""
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    newcust = [c for c in configuration.extra_revalorizacion if c != discount_bucket_column(configuration)]
    keys = ["celda", "estado", "tramo"] + newcust
    grouped = frame.groupby(keys).agg(units=(pipe, "sum"), renewed=(ren, "sum")).reset_index()
    grouped = grouped[grouped["units"] >= 5]
    rates = np.clip(grouped["renewed"] / grouped["units"], 1e-3, 1 - 1e-3)
    y = logit(rates.to_numpy())
    w = np.sqrt(grouped["units"].to_numpy(dtype=float))
    reference = NO_DISCOUNT_BUCKET if getattr(configuration, "discount_value_column", None) else str(getattr(configuration, "no_discount_value", None) or sorted(grouped["tramo"].unique())[0])

    def dummies(column: str, reference_value=None) -> pd.DataFrame:
        values = grouped[column].astype(str)
        levels = sorted(values.unique())
        ref = reference_value if reference_value in levels else levels[0]
        return pd.DataFrame({f"{column}={level}": (values == level).astype(float) for level in levels if level != ref}, index=grouped.index)
    groups = {"celda": dummies("celda"), "estado": dummies("estado", NEUTRAL_STATE), "tramo": dummies("tramo", reference)}
    for column in newcust:
        groups[column] = dummies(column)

    def fit(columns: list) -> tuple:
        design = np.column_stack([np.ones(len(grouped))] + [groups[g].to_numpy() for g in columns if groups[g].shape[1]]) if columns else np.ones((len(grouped), 1))
        coefficients, _, _, _ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        residual = y - design @ coefficients
        rss = float(np.sum((w * residual) ** 2))
        tss = float(np.sum((w * (y - np.average(y, weights=w ** 2))) ** 2))
        return coefficients, design, rss, 1 - rss / tss if tss > 0 else np.nan
    full_columns = list(groups)
    coefficients, design, rss, r2_full = fit(full_columns)
    mean_rate = float(np.average(rates, weights=grouped["units"]))
    to_pp = 100 * mean_rate * (1 - mean_rate)
    residual_variance = rss / max(len(y) - design.shape[1], 1)
    covariance = residual_variance * np.linalg.pinv((design * w[:, None]).T @ (design * w[:, None]))
    names = ["intercept"] + [name for g in full_columns for name in groups[g].columns]
    units_of_level = {("tramo", str(v)): float(u) for v, u in grouped.groupby("tramo")["units"].sum().items()}
    units_of_level.update({("estado", str(v)): float(u) for v, u in grouped.groupby("estado")["units"].sum().items()})
    effects = []
    for index, name in enumerate(names):
        if name.startswith("tramo=") or name.startswith("estado="):
            variable, value = name.split("=")[0], name.split("=", 1)[1]
            se_model_pp = to_pp * float(np.sqrt(max(covariance[index, index], 0)))
            # never below the sampling floor of the level's own contracts (a deterministic fit is not infinite precision)
            se_floor_pp = 100 * float(np.sqrt(max(mean_rate * (1 - mean_rate), 0.0475) / max(units_of_level.get((variable, value), 1.0), 1.0)))
            se_pp = max(se_model_pp, se_floor_pp)
            effects.append(dict(variable=variable, valor=value, referencia=reference if variable == "tramo" else NEUTRAL_STATE,
                                efecto_pp=to_pp * coefficients[index], se_pp=se_pp, z=to_pp * coefficients[index] / se_pp))
    importance = []
    for group in full_columns:
        _, _, _, r2_alone = fit([group])
        _, _, _, r2_without = fit([g for g in full_columns if g != group])
        importance.append(dict(grupo=group, r2_solo=r2_alone, r2_perdido_al_quitarlo=r2_full - r2_without, r2_modelo_completo=r2_full))
    return pd.DataFrame(effects), pd.DataFrame(importance)


def price_jump_vs_rate(frame: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Step 4: per cell × bucket, the renewal price ratio (uplift) and the rate, with the
    within-bucket correlation across cells (weighted by contracts)."""
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    per = frame.groupby(["celda", "tramo"]).agg(units=(pipe, "sum"), renewed=(ren, "sum"), pipe_usd=(configuration.pipeline_usd_col, "sum"),
                                                ren_usd=(configuration.renewed_usd_col, "sum")).reset_index()
    per = per[(per["units"] >= MIN_UNITS_PER_BUCKET) & (per["renewed"] > 0)]
    per["tasa"] = per["renewed"] / per["units"]
    per["uplift"] = (per["ren_usd"] / per["renewed"]) / (per["pipe_usd"] / per["units"])
    correlations = []
    for bucket, block in per.groupby("tramo"):
        if len(block) >= 4 and block["uplift"].std() > 0 and block["tasa"].std() > 0:
            w = block["units"].to_numpy(dtype=float)
            x, y = block["uplift"].to_numpy(), block["tasa"].to_numpy()
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            cov = np.average((x - mx) * (y - my), weights=w)
            corr = cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
            correlations.append(dict(tramo=bucket, celdas=int(len(block)), corr_uplift_tasa=float(corr)))
    per["corr_tramo"] = per["tramo"].map({c["tramo"]: c["corr_uplift_tasa"] for c in correlations})
    return per[["celda", "tramo", "units", "tasa", "uplift", "corr_tramo"]].rename(columns={"units": "contratos"})


def run_discount_churn(fine_table: pd.DataFrame, configuration: Config) -> dict:
    """The analysis end to end. Persists five tables, prints the findings in plain words."""
    try:
        bucket_column = discount_bucket_column(configuration)
    except ValueError as error:
        print(f"[1.5] discount and churn: skipped ({error})")
        return {}
    frame = closed_rows(fine_table, configuration)
    by_bucket = rate_by_bucket(frame, configuration)
    by_state = rate_by_bucket_and_state(frame, configuration)
    effects, importance = adjusted_effects(frame, configuration)
    price = price_jump_vs_rate(frame, configuration)
    for table, name in ((by_bucket, "discount_churn_by_bucket"), (by_state, "discount_churn_by_state"), (effects, "discount_churn_adjusted"),
                        (importance, "discount_churn_importance"), (price, "discount_churn_price")):
        configuration.write(table, name)
    print_discount_churn(by_bucket, by_state, effects, importance, price, bucket_column, configuration)
    return dict(by_bucket=by_bucket, by_state=by_state, effects=effects, importance=importance, price=price, bucket_column=bucket_column)


def print_discount_churn(by_bucket, by_state, effects, importance, price, bucket_column, configuration) -> None:
    print(f"[1.5] DISCOUNT AND CHURN · is the discount ({bucket_column}) a driver of churn? Closed months, composition held fixed by cell")
    print("   bucket · contracts · raw rate [Wilson 90 %] · standardized rate (cell mix fixed) · composition effect")
    for _, row in by_bucket.iterrows():
        print(f"   {row['tramo']:<14} {row['contratos']:>10,.0f}  {100 * row['tasa_bruta']:5.1f}% [{100 * row['tasa_bruta_low']:.1f}, {100 * row['tasa_bruta_high']:.1f}]  "
              f"std {100 * row['tasa_estandarizada']:5.1f}%  composition {row['efecto_composicion_pp']:+.1f} pp")
    print("   inside each signal state: gap of every bucket to the reference bucket (standardized), in pp with its z")
    for state, block in by_state.groupby("estado"):
        gaps = "  ".join(f"{r['tramo']}: {r['hueco_pp']:+.1f} pp (z {r['z']:+.1f}, {r['base_comparacion']})" for _, r in block.iterrows() if pd.notna(r["hueco_pp"]) and r["tramo"] != r["tramo_referencia"])
        print(f"      {state:<14} {gaps if gaps else '(sin contraste: un solo tramo o sin contratos)'}")
    if len(effects):
        print("   adjusted effects (cell, state, newcust held fixed), in pp at the mean rate:")
        for _, row in effects.iterrows():
            print(f"      {row['variable']}={row['valor']:<14} vs {row['referencia']:<10} {row['efecto_pp']:+6.1f} pp ± {row['se_pp']:.1f} (z {row['z']:+.1f})")
    if len(importance):
        print("   relative importance (share of the variance of renewal rates): alone / lost when dropped from the full model")
        for _, row in importance.sort_values("r2_perdido_al_quitarlo", ascending=False).iterrows():
            print(f"      {row['grupo']:<14} R² alone {row['r2_solo']:.3f} · lost {row['r2_perdido_al_quitarlo']:.3f}  (full model R² {row['r2_modelo_completo']:.3f})")
    if len(price) and price["corr_tramo"].notna().any():
        corr = price.drop_duplicates("tramo")[["tramo", "corr_tramo"]].dropna()
        print("   price jump at renewal vs rate, across cells inside the same bucket (weighted correlation): " + ", ".join(f"{t}: {c:+.2f}" for t, c in zip(corr["tramo"], corr["corr_tramo"])))
    explain(configuration,
            "Raw rate by bucket mixes WHO is in each bucket (regions, products): the standardized rate applies every bucket to the same cell mix, so the difference is behaviour, not composition.",
            "The gap inside each signal state answers the real question: if no_instalado customers renew at 20 % with and without discount, the flag dominates and the discount is not the driver there.",
            "Adjusted effects hold cell, state and newcust fixed; the relative importance says which group of variables explains more of the variance of the rates: cell, signals or discount.",
            "It is descriptive evidence on aggregated data, not a causal experiment: the customers who take a discount may differ in ways the dims do not capture.")
