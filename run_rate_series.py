"""run_rate_series.py — SFF v3 · RUN · phase 1.1: the forecast series and their summary.

A forecast series is the monthly sequence of forecast units of one combination of the
rate series columns. This module creates it (`fs_key`), fills the gaps inside its history
with synthetic rows whose rate is UNDEFINED (a month with no expirations says nothing
about the rate), computes the per-row rate with the null≠zero discipline, and leaves the
per-series summary that the ladder, the card and the report read: support, history,
sign of the active timevarying flags, own rate and its binomial error.

Persisted column names stay in Spanish (`tasa`, `sintetica`, `signo`): they are the
contract of the tables.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from binomial_reference import wilson_half_width_pp
from config import Config, explain, hash_key
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
# Sign labels persisted in `signo`


def sign_of_series(active_signs: set) -> str:
    """The sign of a series from the signs of its ACTIVE timevarying flags.

    INPUT:   active_signs — subset of {"negative", "positive"}.
    OUTPUT:  SIGN_NEUTRAL (no flag), SIGN_NEGATIVE, SIGN_POSITIVE, or SIGN_MIXED (both — never pooled).
    """
    if not active_signs:
        return SIGN_NEUTRAL
    if active_signs == {"negative"}:
        return SIGN_NEGATIVE
    if active_signs == {"positive"}:
        return SIGN_POSITIVE
    return SIGN_MIXED


def series_sign_table(forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per fs_id with its sign, read from the timevarying columns.

    INPUT:   forecast_units with fs_id and the timevarying columns · configuration.
    OUTPUT:  DataFrame(fs_id, signo).
    RULES:   a flag is active when its value is in `timevarying_positive_values`; the
             sign of the series is `sign_of_series` of the signs of its active flags.
    """
    one_per_series = forecast_units.drop_duplicates("fs_id").set_index("fs_id")
    sign_by_column = configuration.structural_timevarying_dims
    signs = []
    for series_id, row in one_per_series.iterrows():
        active = {sign_by_column[column] for column in sign_by_column
                  if row[column] in configuration.timevarying_positive_values}
        signs.append(dict(fs_id=series_id, signo=sign_of_series(active)))
    return pd.DataFrame(signs, columns=["fs_id", "signo"])


def fill_history_gaps(labeled_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Add a synthetic row for every month missing INSIDE the history of a trainable series.

    INPUT:   labeled_units (phase 0.3, with fs_id, ruta, universo) · configuration.
    OUTPUT:  the same frame plus synthetic rows (`sintetica` = 1): measures 0, role
             inherited from the previous real month, fu_id/fu_key minted here.
    RULES:   only trainable series of the normal universe; only between the first and
             the last real month of the history (never before, never after, never in
             the projection). The rate of a synthetic row is set later to NaN: a month
             with no expirations does not inform the rate (gap policy "no_rate").
    EDGE CASES: a series with no gaps adds nothing.
    NOTE:    built as ONE frame (cartesian months × series, minus the existing), not row
             by row: with thousands of series the per-row version took seconds per phase.
    """
    units = labeled_units.copy()
    units["sintetica"] = 0
    period_column, role_column = configuration.period_col, configuration.dataset_role_col
    trainable = units[(units["universo"] == UNIVERSE_NORMAL) & (units["ruta"] == TREATMENT_PREDICTABLE)
                      & (units[role_column].isin(TRUTH_ROLES))]
    if trainable.empty:
        return units
    span = trainable.groupby("fs_id")[period_column].agg(["min", "max"])
    full_months = pd.concat([pd.DataFrame({"fs_id": series_id, period_column: pd.period_range(first, last, freq="M")})
                             for series_id, (first, last) in span.iterrows()], ignore_index=True)
    existing = trainable[["fs_id", period_column]].drop_duplicates()
    missing = full_months.merge(existing, on=["fs_id", period_column], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if missing.empty:
        return units
    template = trainable.sort_values(period_column).drop_duplicates("fs_id").set_index("fs_id")
    synthetic = template.loc[missing["fs_id"]].reset_index()
    synthetic[period_column] = missing[period_column].to_numpy()
    for measure in configuration.core_measures:
        synthetic[measure] = 0.0
    synthetic[configuration.current_month_col] = 0
    synthetic["sintetica"] = 1
    synthetic["fu_id"] = synthetic["fs_id"] + "|" + synthetic[period_column].astype(str)
    synthetic["fu_key"] = synthetic["fu_id"].map(hash_key)
    # role = the role of the previous real month of the series (train, or test once the test starts)
    role_by_month = trainable[["fs_id", period_column, role_column]].sort_values(period_column)
    synthetic = pd.merge_asof(synthetic.sort_values(period_column).drop(columns=[role_column]), role_by_month,
                              on=period_column, by="fs_id", direction="backward")
    return pd.concat([units, synthetic[units.columns]], ignore_index=True)


def compute_row_rates(units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The rate of every forecast unit with the null≠zero discipline.

    RULES:   tasa = renewed / pipeline when pipeline > 0; NaN on projection rows, on
             synthetic rows and on real rows with pipeline 0. A NaN rate means "no
             information", never "0 %".
    """
    rated = units.copy()
    pipeline = rated[configuration.pipeline_units_col]
    renewed = rated[configuration.renewed_units_col]
    rated["tasa"] = np.where((rated["sintetica"] == 0) & (pipeline > 0), renewed / pipeline.replace(0, np.nan), np.nan)
    rated.loc[~rated[configuration.dataset_role_col].isin(TRUTH_ROLES), "tasa"] = np.nan   # projection and pending: no truth
    return rated


def build_series_summary(units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per forecast series: support, history, own rate, binomial error, money, sign.

    INPUT:   units with tasa and sintetica · configuration.
    OUTPUT:  fs_summary: fs_id, fs_key, celda_id, signo, ruta, universo, n_propio
             (median monthly pipeline units over real history months with pipeline > 0),
             meses_historia, huecos, ren, pipe, tasa_propia (Σren/Σpipe),
             error_binomial_pp (Wilson half-width at z with n_propio), usd_proyectado.
    RULES:   history = normal universe, non-projection rows. `n_propio` is the typical
             month, not the total: it is the n of the month we will predict.
    EDGE CASES: a series with no real history month gets n_propio 0 and tasa NaN.
    """
    period_column, role_column = configuration.period_col, configuration.dataset_role_col
    history = units[(units["universo"] == UNIVERSE_NORMAL) & (units[role_column].isin(TRUTH_ROLES))]
    summary = history.groupby(["fs_id", "fs_key"], as_index=False).agg(
        n_propio=(configuration.pipeline_units_col, lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0),
        meses_historia=("sintetica", "size"),
        huecos=("sintetica", "sum"),
        ren=(configuration.renewed_units_col, "sum"),
        pipe=(configuration.pipeline_units_col, "sum"))
    summary["tasa_propia"] = np.where(summary["pipe"] > 0, summary["ren"] / summary["pipe"].replace(0, np.nan), np.nan)
    summary["error_binomial_pp"] = [wilson_half_width_pp(rate if np.isfinite(rate) else 0.5, support, configuration.z)
                                    for rate, support in zip(summary["tasa_propia"], summary["n_propio"])]
    projected = (units[units[role_column].isin((ROLE_PROJECTION, ROLE_PENDING))]
                 .groupby("fs_id")[configuration.pipeline_usd_col].sum().rename("usd_proyectado"))
    summary = summary.merge(projected, on="fs_id", how="left")
    # series present only in the projection (heuristic) get a row too: they carry money
    heuristic_only = (units[~units["fs_id"].isin(summary["fs_id"])]
                      .drop_duplicates("fs_id")[["fs_id", "fs_key"]])
    if len(heuristic_only):
        heuristic_only = heuristic_only.assign(n_propio=0.0, meses_historia=0, huecos=0, ren=0.0, pipe=0.0,
                                               tasa_propia=np.nan, error_binomial_pp=np.nan)
        heuristic_only = heuristic_only.merge(projected, on="fs_id", how="left")
        summary = pd.concat([summary, heuristic_only], ignore_index=True)
    summary["usd_proyectado"] = summary["usd_proyectado"].fillna(0.0)
    labels = units.drop_duplicates("fs_id")[["fs_id", "ruta", "universo"] + configuration.business_mandatory_dims]
    summary = summary.merge(labels, on="fs_id", how="left")
    summary["celda_id"] = summary[configuration.business_mandatory_dims].astype(str).agg("|".join, axis=1)
    summary = summary.merge(series_sign_table(units, configuration), on="fs_id", how="left")
    return summary.drop(columns=configuration.business_mandatory_dims)


def build_rate_series(labeled_units: pd.DataFrame, configuration: Config) -> tuple:
    """Phase 1.1 end to end: fs_key, gaps, rates, summary. Persists `fact_fu_gaps` (the synthetic
    rows) and `forecast_series_raw_summary`.

    INPUT:   labeled_units (phase 0.3) · configuration.
    OUTPUT:  (units, series_summary).
    CONSOLE: series count, synthetic rows added, series below the floor and their money.
    STEPS:
      [1] fs_key.  [2] gaps.  [3] rates.  [4] summary.  [5] persist and report.
    """
    units = labeled_units.copy()
    units["fs_key"] = units["fs_id"].map(hash_key)
    units = fill_history_gaps(units, configuration)
    units = compute_row_rates(units, configuration)
    summary = build_series_summary(units, configuration)
    gaps = units[units["sintetica"] == 1][["fs_id", "fs_key", "fu_id", "fu_key", configuration.period_col]].copy()
    gaps[configuration.period_col] = gaps[configuration.period_col].astype(str)
    configuration.write(gaps, "fact_fu_gaps")
    configuration.write(summary, "forecast_series_raw_summary")
    below = summary[(summary["ruta"] == TREATMENT_PREDICTABLE) & (summary["n_propio"] < configuration.support_floor)]
    print(f"[1.1] {len(summary)} series · synthetic gap rows {int(units['sintetica'].sum())} (rate undefined) · "
          f"{len(below)} trainable series below the floor carry ${below['usd_proyectado'].sum():,.0f} "
          f"of ${summary['usd_proyectado'].sum():,.0f} projected")
    explain(configuration,
            "A series is one segment (region, product, plazo, banda, señales, cliente nuevo) followed month by month; its rate is renewed / expired.",
            f"'Below the floor' = fewer than {configuration.support_floor:.0f} contracts in a typical month: a rate measured on that few moves ±15 pp by chance alone,",
            "so those series will borrow support from a relative in the ladder instead of predicting with their own rate.")
    return units, summary
