"""run_uplift.py — SFF v3 · RUN · phase 4: the revaluation branch, simple and correct.

The uplift of a cell is the ratio of sums Σ renewed$ / Σ (renewed units × pipeline AUV):
what the renewers pay relative to what they paid, weighted by money. Its n is the number
of RENEWERS, not the pipeline: a renewal that did not happen has no price.

Support repair, deliberately minimal (the v2 credibility was a no-op and is dropped):
  · a cell with n_renovadores ≥ `uplift_floor` uses its own ratio;
  · below the floor, it takes its PARENT's: the same mandatory cell with the
    `uplift_parent_keep_columns` extras kept (the "starting point": e.g. newcust) and the
    other extras set to '*'; if the parent is also below the floor, the mandatory cell.
The error of a ratio has no clean formula: it is measured by BOOTSTRAP of the renewer
rows (resample, recompute the ratio, take p5/p95). Intuitive, no theory.

Output: `decision_uplift` (cell → uplift, n, band, origin) and `uplift_chain` (the trace).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import COMBINED_ID_SEPARATOR, Config, explain, hash_key, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
NEUTRAL_UPLIFT = 1.0
MIN_UPLIFT = 0.01
WILDCARD = "*"
ORIGIN_OWN = "propia"
ORIGIN_PARENT = "padre"
ORIGIN_CELL = "celda"
ORIGIN_NEUTRAL = "neutro"


def renewer_rows(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The rows that renewed, with per-row pipeline AUV and uplift; uplift cell ids."""
    renewers = fine_table[fine_table[configuration.renewed_units_col].fillna(0) > 0].copy()
    if configuration.uplift_window_months:
        recent = sorted(renewers[configuration.period_col].unique())[-int(configuration.uplift_window_months):]
        renewers = renewers[renewers[configuration.period_col].isin(recent)]
    renewers["auv_pipeline"] = renewers[configuration.pipeline_usd_col] / renewers[configuration.pipeline_units_col].clip(lower=1)
    renewers["uplift_fila"] = (renewers[configuration.renewed_usd_col] / renewers[configuration.renewed_units_col]) / renewers["auv_pipeline"]
    renewers["uplift_cell_id"] = join_columns(renewers, configuration.uplift_cell_columns)
    renewers["celda_id"] = join_columns(renewers, configuration.business_mandatory_dims)
    keep = set(configuration.uplift_parent_keep_columns)
    parent_fields = [renewers[c].astype(str) if c in keep or c in configuration.business_mandatory_dims
                     else pd.Series(WILDCARD, index=renewers.index) for c in configuration.uplift_cell_columns]
    renewers["uplift_parent_id"] = pd.concat(parent_fields, axis=1).agg("|".join, axis=1) if parent_fields else WILDCARD
    return renewers


def ratio_of_sums(rows: pd.DataFrame, configuration: Config) -> float:
    """Σ renewed$ / Σ (renewed units × pipeline AUV)."""
    denominator = float((rows[configuration.renewed_units_col] * rows["auv_pipeline"]).sum())
    return float(rows[configuration.renewed_usd_col].sum() / denominator) if denominator > 0 else np.nan


def bootstrap_band(rows: pd.DataFrame, configuration: Config, rng: np.random.Generator) -> tuple:
    """p5 / p95 of the ratio over resampled renewer rows.

    NOTE:    the resampling is done on two numpy arrays (renewed$ and renewed units ×
             pipeline AUV) with one index matrix of shape (samples × rows): no DataFrame
             is built per sample, so a cell with 50,000 renewer rows costs milliseconds.
    """
    if len(rows) < 2:
        return np.nan, np.nan
    numerator = rows[configuration.renewed_usd_col].to_numpy(dtype=float)
    denominator = (rows[configuration.renewed_units_col] * rows["auv_pipeline"]).to_numpy(dtype=float)
    picks = rng.integers(0, len(rows), size=(configuration.uplift_bootstrap_samples, len(rows)))
    sampled_denominator = denominator[picks].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = np.where(sampled_denominator > 0, numerator[picks].sum(axis=1) / sampled_denominator, np.nan)
    return float(np.nanquantile(samples, 0.05)), float(np.nanquantile(samples, 0.95))


def estimate_uplift_cells(renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per uplift cell: own ratio, n, parent ratio, cell ratio, decision and band.

    OUTPUT:  decision_uplift: uplift_cell_id, uplift_cell_key, celda_id,
             uplift_parent_id, n_renovadores, meses, uplift_propio, uplift_padre,
             uplift_celda, uplift, uplift_origen, banda_low, banda_high, recortado.
    """
    rng = np.random.default_rng(configuration.random_seed)
    by_parent = {pid: ratio_of_sums(g, configuration) for pid, g in renewers.groupby("uplift_parent_id")}
    n_parent = renewers.groupby("uplift_parent_id")[configuration.renewed_units_col].sum()
    by_cell = {cid: ratio_of_sums(g, configuration) for cid, g in renewers.groupby("celda_id")}
    rows = []
    for cell_id, group in renewers.groupby("uplift_cell_id"):
        n = float(group[configuration.renewed_units_col].sum())
        own = ratio_of_sums(group, configuration)
        parent_id, mandatory_id = group["uplift_parent_id"].iat[0], group["celda_id"].iat[0]
        parent, cell = by_parent.get(parent_id, np.nan), by_cell.get(mandatory_id, np.nan)
        if n >= configuration.uplift_floor and np.isfinite(own):
            chosen, origin, band_rows = own, ORIGIN_OWN, group
        elif n_parent.get(parent_id, 0) >= configuration.uplift_floor and np.isfinite(parent):
            chosen, origin, band_rows = parent, ORIGIN_PARENT, renewers[renewers["uplift_parent_id"] == parent_id]
        elif np.isfinite(cell):
            chosen, origin, band_rows = cell, ORIGIN_CELL, renewers[renewers["celda_id"] == mandatory_id]
        else:
            chosen, origin, band_rows = NEUTRAL_UPLIFT, ORIGIN_NEUTRAL, group.head(0)
        low, high = bootstrap_band(band_rows, configuration, rng)
        clipped = float(np.clip(chosen, MIN_UPLIFT, configuration.uplift_cap))
        rows.append(dict(uplift_cell_id=cell_id, uplift_cell_key=hash_key(cell_id), celda_id=mandatory_id,
                         uplift_parent_id=parent_id, n_renovadores=n, meses=int(group[configuration.period_col].nunique()),
                         uplift_propio=round(own, 4) if np.isfinite(own) else np.nan,
                         uplift_padre=round(parent, 4) if np.isfinite(parent) else np.nan,
                         uplift_celda=round(cell, 4) if np.isfinite(cell) else np.nan,
                         uplift=round(clipped, 4), uplift_origen=origin,
                         banda_low=round(low, 4) if np.isfinite(low) else np.nan,
                         banda_high=round(high, 4) if np.isfinite(high) else np.nan,
                         recortado=int(clipped != chosen)))
    return pd.DataFrame(rows)


def build_uplift_chain(decision_uplift: pd.DataFrame) -> pd.DataFrame:
    """The trace per cell: stages 0_propio / 1_padre / 2_celda / 9_final."""
    rows = []
    for _, cell in decision_uplift.iterrows():
        for stage, value in (("0_propio", cell["uplift_propio"]), ("1_padre", cell["uplift_padre"]),
                             ("2_celda", cell["uplift_celda"]), ("9_final", cell["uplift"])):
            rows.append(dict(uplift_cell_id=cell["uplift_cell_id"], uplift_cell_key=cell["uplift_cell_key"], etapa=stage,
                             uplift=value, n_renovadores=cell["n_renovadores"], uplift_origen=cell["uplift_origen"]))
    return pd.DataFrame(rows)


def run_uplift(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Phase 4: the statistical uplift per cell (as before) and, when the discount column
    exists, the validation of the contract rule (`uplift_contract_check`). The contract
    path itself is applied row by row in the assembly. Persists decision_uplift and
    uplift_chain. Returns decision_uplift (with attrs["realization_ratio"] per cell)."""
    renewers = renewer_rows(fine_table, configuration)
    check = realization_check(renewers, configuration) if configuration.discount_value_column else pd.DataFrame(
        columns=["uplift_cell_id", "autorenew", "n_renovadores", "usd_renovado", "uplift_observado", "uplift_regla", "pct_usd_dentro_2pct", "ratio_realizacion", "aplicar"])
    configuration.write(check, "uplift_contract_check")
    if not configuration.discount_value_column:
        configuration.write(pd.DataFrame(columns=["uplift_cell_id", "n_total", "n_desconocido", "n_conocido", "uplift_todas", "uplift_conocido",
                                                  "uplift_desconocido", "banda_low", "banda_high", "difiere", "usd_desconocido"]), "uplift_missing_check")
    if configuration.discount_value_column:
        known = known_discount(renewers, configuration)
        if len(check):
            weighted_within = float(np.average(check["pct_usd_dentro_2pct"], weights=np.maximum(check["usd_renovado"], 1e-9)))
            print(f"[4] contract rule check on past renewals with a known discount ({100 * known.mean():.0f}% of renewer rows): "
                  f"{weighted_within:.0f}% of renewed $ within ±2% of price_increase / (1 − discount) · realization ratio applied in "
                  f"{int(check['aplicar'].sum())} cell×autorenew groups (dollar-weighted mean {np.average(check.loc[check['aplicar'] == 1, 'ratio_realizacion'], weights=np.maximum(check.loc[check['aplicar'] == 1, 'usd_renovado'], 1e-9)) if check['aplicar'].any() else float('nan'):.3f})")
        missing = missing_discount_check(renewers, configuration)
        configuration.write(missing, "uplift_missing_check")
        if len(missing):
            differing = missing[missing["difiere"] == 1]
            print(f"[4] missing-discount check: in {len(differing)} of {len(missing)} cells with ≥ {configuration.uplift_floor:.0f} unknown-discount renewers, the uplift of ALL rows "
                  f"falls outside the band of the unknown-only rows (${differing['usd_desconocido'].sum():,.0f} renewed by unknown-discount rows there) → "
                  f"{'the missing discount is NOT random: the statistical path is estimated on unknown-discount rows only' if configuration.statistical_uplift_from_unknown_only else 'mixed estimate kept (statistical_uplift_from_unknown_only=False)'}")
        if configuration.statistical_uplift_from_unknown_only:
            renewers = renewers[~known]
    decision = estimate_uplift_cells(renewers, configuration)
    ratios = realization_ratio_by_cell(check, configuration) if len(check) else {}
    decision["ratio_realizacion"] = decision["uplift_cell_id"].map(ratios).fillna(1.0).round(4)
    decision.attrs["realization_ratio"] = ratios
    configuration.write(decision, "decision_uplift")
    configuration.write(build_uplift_chain(decision), "uplift_chain")
    below = decision[decision["n_renovadores"] < configuration.uplift_floor]
    print(f"[4] uplift: {len(decision)} cells · range [{decision['uplift'].min():.2f}, {decision['uplift'].max():.2f}] · "
          f"{len(below)} cells below the floor ({decision['uplift_origen'].value_counts().to_dict()}) · "
          f"{int(decision['recortado'].sum())} clipped at the cap {configuration.uplift_cap}")
    return decision


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONTRACT PATH · where the discount is known, the renewal price is a rule
# ═══════════════════════════════════════════════════════════════════════════════════

def price_increase_factor(periods: pd.Series, configuration: Config) -> np.ndarray:
    """The list-price factor of every period: the product of the increases dated at or
    before it (`price_increase_by_period`, {"2027-01": 1.05}). 1.0 with no increases."""
    factors = np.ones(len(periods), dtype=float)
    if not configuration.price_increase_by_period:
        return factors
    months = pd.PeriodIndex(periods.astype(str), freq="M")
    for start, factor in configuration.price_increase_by_period.items():
        factors = np.where(months >= pd.Period(start, freq="M"), factors * float(factor), factors)
    return factors


def known_discount(rows: pd.DataFrame, configuration: Config) -> pd.Series:
    """True where the row's discount is informed and below the cap (the contract path)."""
    column = configuration.discount_value_column
    if column is None or column not in rows.columns:
        return pd.Series(False, index=rows.index)
    discount = pd.to_numeric(rows[column], errors="coerce")
    return discount.notna() & (discount >= 0) & (discount <= configuration.discount_cap)


def contract_uplift(rows: pd.DataFrame, configuration: Config, with_price_increase: bool = False) -> pd.Series:
    """uplift = 1 / (1 − discount) (the customer renews at 100 % of the current list), NaN
    where the discount is unknown. The base forecast carries NO price increase: increases
    live in the scenario layer (with_price_increase=True applies price_increase_by_period)."""
    discount = pd.to_numeric(rows[configuration.discount_value_column], errors="coerce") if configuration.discount_value_column in rows.columns else pd.Series(np.nan, index=rows.index)
    increase = price_increase_factor(rows[configuration.period_col], configuration) if with_price_increase else np.ones(len(rows))
    rule = increase / (1.0 - discount.clip(upper=0.999))
    return rule.where(known_discount(rows, configuration))


def realization_check(renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The validation of the rule on past renewals with a known discount, per uplift cell
    (and per autorenew value when that flag exists): the observed uplift, the rule uplift,
    the share of renewed dollars within ±2 % of the rule, and the realization ratio
    = Σ renewed$ / Σ (renewed units × pipeline AUV × rule), dollar-weighted.
    OUTPUT: uplift_contract_check (uplift_cell_id, autorenew, n_renovadores, usd_renovado,
            uplift_observado, uplift_regla, pct_usd_dentro_2pct, ratio_realizacion, aplicar)."""
    rows = renewers[known_discount(renewers, configuration)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["uplift_cell_id", "autorenew", "n_renovadores", "usd_renovado", "uplift_observado", "uplift_regla",
                                     "pct_usd_dentro_2pct", "ratio_realizacion", "aplicar"])
    rows["uplift_regla"] = contract_uplift(rows, configuration)
    autorenew_column = next((c for c in configuration.structural_timevarying_dims if "autoren" in c.lower()), None)
    rows["autorenew"] = rows[autorenew_column].astype(str) if autorenew_column else "*"
    rows["within"] = (rows["uplift_fila"] / rows["uplift_regla"]).between(0.98, 1.02)
    rows["rule_denominator"] = rows[configuration.renewed_units_col] * rows["auv_pipeline"] * rows["uplift_regla"]
    out = []
    for (cell, autorenew), block in rows.groupby(["uplift_cell_id", "autorenew"]):
        renewed_usd = float(block[configuration.renewed_usd_col].sum())
        ratio = renewed_usd / float(block["rule_denominator"].sum()) if block["rule_denominator"].sum() > 0 else np.nan
        renewers_count = float(block[configuration.renewed_units_col].sum())
        out.append(dict(uplift_cell_id=cell, autorenew=autorenew, n_renovadores=renewers_count, usd_renovado=round(renewed_usd, 2),
                        uplift_observado=round(ratio_of_sums(block, configuration), 4),
                        uplift_regla=round(float(np.average(block["uplift_regla"], weights=np.maximum(block["rule_denominator"], 1e-9))), 4),
                        pct_usd_dentro_2pct=round(100 * float(block.loc[block["within"], configuration.renewed_usd_col].sum()) / max(renewed_usd, 1e-9), 1),
                        ratio_realizacion=round(ratio, 4) if np.isfinite(ratio) else np.nan,
                        aplicar=int(renewers_count >= configuration.uplift_floor and np.isfinite(ratio))))
    return pd.DataFrame(out)


def missing_discount_check(renewers: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The verification of the 'missing is not random' hypothesis, per uplift cell: the
    statistical uplift estimated on ALL renewer rows, on the rows with a KNOWN discount and
    on the rows with an UNKNOWN discount, with the bootstrap band of the unknown-only
    estimate. `difiere` = 1 when the all-rows estimate falls outside that band: mixing the
    populations would give the unknown rows an uplift that is not theirs.
    OUTPUT: uplift_missing_check (uplift_cell_id, n_total, n_desconocido, n_conocido,
            uplift_todas, uplift_conocido, uplift_desconocido, banda_low, banda_high,
            difiere, usd_desconocido)."""
    known = known_discount(renewers, configuration)
    rng = np.random.default_rng(configuration.seed if hasattr(configuration, "seed") else 7)
    rows = []
    for cell, block in renewers.groupby("uplift_cell_id"):
        block_known, block_unknown = block[known.loc[block.index]], block[~known.loc[block.index]]
        n_unknown = float(block_unknown[configuration.renewed_units_col].sum())
        if n_unknown < configuration.uplift_floor:
            continue
        low, high = bootstrap_band(block_unknown, configuration, rng)
        uplift_all = ratio_of_sums(block, configuration)
        rows.append(dict(uplift_cell_id=cell, n_total=float(block[configuration.renewed_units_col].sum()), n_desconocido=n_unknown,
                         n_conocido=float(block_known[configuration.renewed_units_col].sum()),
                         uplift_todas=round(uplift_all, 4), uplift_conocido=round(ratio_of_sums(block_known, configuration), 4) if len(block_known) else np.nan,
                         uplift_desconocido=round(ratio_of_sums(block_unknown, configuration), 4),
                         banda_low=round(low, 4) if np.isfinite(low) else np.nan, banda_high=round(high, 4) if np.isfinite(high) else np.nan,
                         difiere=int(np.isfinite(low) and np.isfinite(high) and not (low <= uplift_all <= high)),
                         usd_desconocido=round(float(block_unknown[configuration.renewed_usd_col].sum()), 2)))
    return pd.DataFrame(rows, columns=["uplift_cell_id", "n_total", "n_desconocido", "n_conocido", "uplift_todas", "uplift_conocido",
                                       "uplift_desconocido", "banda_low", "banda_high", "difiere", "usd_desconocido"])


def realization_ratio_by_cell(check: pd.DataFrame, configuration: Config) -> dict:
    """uplift_cell_id → realization ratio to apply (dollar-weighted over autorenew values),
    1.0 where the cell has no support or the correction is switched off."""
    if not configuration.contract_apply_realization_ratio or check is None or check.empty:
        return {}
    usable = check[check["aplicar"] == 1]
    ratios = {}
    for cell, block in usable.groupby("uplift_cell_id"):
        ratios[cell] = float(np.average(block["ratio_realizacion"], weights=np.maximum(block["usd_renovado"], 1e-9)))
    return ratios


# ═══════════════════════════════════════════════════════════════════════════════════
# THE BACKTEST OF THE UPLIFT · two paths, no leakage
# ═══════════════════════════════════════════════════════════════════════════════════

def backtest_uplift(fine_table: pd.DataFrame, backtest_holdout: pd.DataFrame, decision_support: pd.DataFrame,
                    configuration: Config) -> pd.DataFrame:
    """The error of the revaluation and of the final number on the EXAM months, without
    leakage: for every exam month t the statistical uplift of each cell is estimated with
    renewals BEFORE t only; the contract rule needs no estimation.

    Three lines of error, on the rows that really renewed in the exam months:
      1. the rate: the backtest of phase 3 (unchanged);
      2. the revaluation: renewed price predicted vs real — statistical for every row,
         and rule vs statistical on the rows with a known discount (WAPE in $, bias);
      3. the final number: pipeline × predicted rate (h=1, the phase-3 hold-out) × uplift
         vs real renewed $ — path A (all statistical) vs path B (rule where the discount
         is known, statistical elsewhere). A − B is what the discount contributes.
    Breakdown by region and autorenew; % of $ with a known discount. Past price
    increases are unknown: the rule is applied without them (an assumption: if there
    were increases, the rule looks worse through no fault of the method).
    OUTPUT: backtest_uplift (ambito, autorenew, n_meses, usd_real, pct_usd_descuento_conocido,
            wape_estadistico_pct, sesgo_estadistico_pct, wape_regla_pct, sesgo_regla_pct,
            wape_final_A_pct, wape_final_B_pct, aportacion_descuento_pct, veredicto)."""
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren, pipe_usd, ren_usd = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.pipeline_usd_col, configuration.renewed_usd_col
    exam = fine_table[(fine_table[role] == ROLE_TEST) & (fine_table[ren].fillna(0) > 0)].copy()
    if exam.empty or backtest_holdout is None or backtest_holdout.empty:
        return pd.DataFrame()
    history = renewer_rows(fine_table, configuration)
    exam["uplift_cell_id"] = join_columns(exam, configuration.uplift_cell_columns)
    exam["auv_pipeline"] = exam[pipe_usd] / exam[pipe].clip(lower=1)
    exam["fs_id"] = join_columns(exam, configuration.rate_series_columns)
    # the predicted rate at h=1 of the exam month, through the series' pool
    pool_of = dict(zip(decision_support["fs_id"], decision_support["id_estimacion"]))
    at_h1 = backtest_holdout[backtest_holdout["h"] == 1][["id_estimacion", "mes_objetivo", "tasa_pred"]]
    exam["id_estimacion"] = exam["fs_id"].map(pool_of)
    exam["mes_objetivo"] = exam[period].astype(str)
    exam = exam.merge(at_h1, on=["id_estimacion", "mes_objetivo"], how="left")
    # the statistical uplift of each cell with renewals BEFORE the exam month (no leakage)
    exam["uplift_estadistico"] = np.nan
    for month in sorted(exam[period].unique()):
        before = history[history[period] < month]
        cell_uplift = before.groupby("uplift_cell_id").apply(lambda g: ratio_of_sums(g, configuration), include_groups=False)
        mask = exam[period] == month
        exam.loc[mask, "uplift_estadistico"] = exam.loc[mask, "uplift_cell_id"].map(cell_uplift)
    global_uplift = ratio_of_sums(history[history[period] < exam[period].min()], configuration) if len(history) else 1.0
    exam["uplift_estadistico"] = exam["uplift_estadistico"].fillna(global_uplift if np.isfinite(global_uplift) else 1.0)
    exam["uplift_regla"] = contract_uplift(exam, configuration)
    known = known_discount(exam, configuration)
    exam["uplift_B"] = np.where(known, exam["uplift_regla"], exam["uplift_estadistico"])
    exam["uplift_real"] = (exam[ren_usd] / exam[ren]) / exam["auv_pipeline"]
    autorenew_column = next((c for c in configuration.structural_timevarying_dims if "autoren" in c.lower()), None)
    exam["autorenew"] = exam[autorenew_column].astype(str) if autorenew_column else "*"
    exam["region"] = exam[configuration.business_mandatory_dims[0]].astype(str)
    rows = []
    scopes = [("total", exam)] + [(r, b) for r, b in exam.groupby("region")]
    for scope, block in scopes:
        for autorenew, part in [("*", block)] + list(block.groupby("autorenew")):
            if scope != "total" and autorenew == "*":
                pass
            real_usd = float(part[ren_usd].sum())
            if real_usd <= 0:
                continue
            renewed_units = part[ren]
            pred_stat = renewed_units * part["auv_pipeline"] * part["uplift_estadistico"]
            pred_B = renewed_units * part["auv_pipeline"] * part["uplift_B"]
            known_part = known.loc[part.index]
            pred_rule_known = (renewed_units * part["auv_pipeline"] * part["uplift_regla"])[known_part]
            pred_stat_known = pred_stat[known_part]
            real_known = part.loc[known_part, ren_usd]
            with_rate = part["tasa_pred"].notna()
            final_A = (part[pipe] * part["tasa_pred"] * part["auv_pipeline"] * part["uplift_estadistico"])[with_rate]
            final_B = (part[pipe] * part["tasa_pred"] * part["auv_pipeline"] * part["uplift_B"])[with_rate]
            real_with_rate = part.loc[with_rate, ren_usd]
            def wape(pred, real):
                return 100 * float((pred - real).abs().sum() / real.sum()) if real.sum() > 0 else np.nan
            def bias(pred, real):
                return 100 * float((pred - real).sum() / real.sum()) if real.sum() > 0 else np.nan
            wape_A, wape_B = wape(final_A, real_with_rate), wape(final_B, real_with_rate)
            verdict = "mejora" if np.isfinite(wape_A) and np.isfinite(wape_B) and wape_B < wape_A - 0.5 else ("empeora" if np.isfinite(wape_A) and np.isfinite(wape_B) and wape_B > wape_A + 0.5 else "empata")
            rows.append(dict(ambito=scope, autorenew=autorenew, n_meses=int(part[period].nunique()), usd_real=round(real_usd, 2),
                             pct_usd_descuento_conocido=round(100 * float(real_known.sum()) / real_usd, 1),
                             wape_estadistico_pct=round(wape(pred_stat, part[ren_usd]), 2), sesgo_estadistico_pct=round(bias(pred_stat, part[ren_usd]), 2),
                             wape_regla_pct=round(wape(pred_rule_known, real_known), 2) if len(real_known) else np.nan,
                             wape_estadistico_conocido_pct=round(wape(pred_stat_known, real_known), 2) if len(real_known) else np.nan,
                             sesgo_regla_pct=round(bias(pred_rule_known, real_known), 2) if len(real_known) else np.nan,
                             wape_final_A_pct=round(wape_A, 2) if np.isfinite(wape_A) else np.nan, wape_final_B_pct=round(wape_B, 2) if np.isfinite(wape_B) else np.nan,
                             aportacion_descuento_pct=round(wape_A - wape_B, 2) if np.isfinite(wape_A) and np.isfinite(wape_B) else np.nan, veredicto=verdict))
    return pd.DataFrame(rows)


def print_backtest_uplift(table: pd.DataFrame, configuration: Config) -> None:
    if table is None or table.empty:
        return
    total = table[(table["ambito"] == "total") & (table["autorenew"] == "*")].iloc[0]
    print(f"[4] BACKTEST OF THE UPLIFT (exam months, no leakage) · {total['pct_usd_descuento_conocido']:.0f}% of renewed $ with a known discount")
    print(f"   revaluation: statistical WAPE {total['wape_estadistico_pct']:.1f}% (bias {total['sesgo_estadistico_pct']:+.1f}%) · on known-discount rows, rule {total['wape_regla_pct']:.1f}% vs statistical {total['wape_estadistico_conocido_pct']:.1f}%")
    print(f"   final number (pipeline × predicted rate × uplift vs real $): path A (all statistical) {total['wape_final_A_pct']:.1f}% · path B (rule where known) {total['wape_final_B_pct']:.1f}% "
          f"· the discount contributes {total['aportacion_descuento_pct']:+.1f} points → {total['veredicto'].upper()}")
    for _, row in table[(table["ambito"] == "total") & (table["autorenew"] != "*")].iterrows():
        print(f"      autorenew={row['autorenew']}: rule {row['wape_regla_pct']} vs statistical {row['wape_estadistico_conocido_pct']} on known-discount rows · final A {row['wape_final_A_pct']} / B {row['wape_final_B_pct']} → {row['veredicto']}")
    explain(configuration,
            "WAPE = Σ|predicted − real| / Σ real, in % of the renewed dollars; bias = the signed version. Path B beats A when the contract rule prices the known-discount rows better than the cell average.",
            "Criterion: if B improves, the change is justified; if it ties, it is adopted anyway for the scenarios; if it worsens, investigate before adopting (renewal offers, past price increases the rule does not know).")
