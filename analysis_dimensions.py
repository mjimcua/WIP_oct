"""analysis_dimensions.py — SFF v3 · ANALYSIS · phase 1.2: what separates behaviour, and
what it costs not to segment.

Three things are decided or measured here, all at the level of the forecast series:

  1. DIMENSION SEPARATION → `decision_eta2`. Which dimensions separate the renewal rate
     and in which order they should collapse when a series needs a relative. Three
     figures per dimension, not one:
       · η² individual  — how much the dimension explains ALONE (weighted by support).
                          Confounded when dimensions are correlated.
       · unique contribution (type II) — R²(all dims) − R²(all dims but this one): what
                          is LOST if this dimension is annulled while the others stay.
                          This is the ladder's question, so this fixes the order.
       · ω²              — η² discounted by the number of groups (many categories with
                          few series look like signal by chance).
     Plus pairs: η² of the joint key vs the individuals → interactions.
     Timevarying dims are OUT (they are handled by sign, not by η²).
     Method: weighted least squares of the series rate on the dimensions as categoricals
     (a weighted factorial ANOVA), numpy only. ANCOVA is not needed: everything is
     categorical (continuous discounts arrive already binned).

  2. COMPOSITION → `mandatory_only_cost`, `mix_shift_decomposition`. We COUNT composition,
     we do not hunt for its extreme cases. Per mandatory cell and month (walk-forward over
     the last months with truth): what predicting with the mandatory dims only (one rate
     per cell) would have cost against predicting the segmented series with the real
     weights of the month — the over/under-estimation of the mandatory-only view, in
     dollars. And the Kitagawa decomposition of the change of the aggregate rate into a
     behaviour term (Σ w·Δp) and a composition term (Σ p·Δw): how much of the movement is
     the mix of the pipeline, how much is people renewing differently.

  3. TIMEVARYING CALIBRATION → `timevarying_calibration`. The realized rate per flag
     and per sign, month by month, with its binomial error: the audit of every model
     that produces a flag, for free.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import itertools

import numpy as np
import pandas as pd

from binomial_reference import binomial_se_pp
from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from run_rate_series import ROLE_PROJECTION, TREATMENT_PREDICTABLE, UNIVERSE_NORMAL

# ─── named constants ─────────────────────────────────────────────────────────────
# (window, min history and max pairs are Config parameters: see config.py, phase 1.2)
RATE_BRANCH = "tasa"


# ═══════════════════════════════════════════════════════════════════════════════════
# 1 · DIMENSION SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════════

def weighted_eta2(frame: pd.DataFrame, group_column: str, value_column: str, weight_column: str) -> float:
    """η² of one grouping: between-group weighted sum of squares / total. 0 when no variance."""
    weights = frame[weight_column].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in frame.groupby(group_column, observed=True):
        gw = group[weight_column].to_numpy(dtype=float)
        between += float(np.sum(gw)) * (np.average(group[value_column], weights=gw) - grand_mean) ** 2
    return float(between / total)


def weighted_omega2(frame: pd.DataFrame, group_column: str, value_column: str, weight_column: str) -> float:
    """ω² = (SSB − (k−1)·MSW) / (SST + MSW): η² corrected for the number of groups k."""
    weights = frame[weight_column].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    n = len(frame)
    groups = frame.groupby(group_column, observed=True)
    k = groups.ngroups
    if total <= 0 or n <= k:
        return 0.0
    between = sum(float(np.sum(g[weight_column])) * (np.average(g[value_column], weights=g[weight_column]) - grand_mean) ** 2
                  for _, g in groups)
    within = total - between
    mean_square_within = within / (n - k)
    return float(max(0.0, (between - (k - 1) * mean_square_within) / (total + mean_square_within)))


def weighted_r2_factorial(frame: pd.DataFrame, dimensions: list, value_column: str, weight_column: str) -> float:
    """R² of a weighted least-squares fit of the value on the dimensions as categoricals
    (additive factorial model). Numpy only; rank-deficient designs are handled by lstsq."""
    if not dimensions:
        return 0.0
    design = pd.get_dummies(frame[dimensions].astype(str), drop_first=True).astype(float)
    design.insert(0, "intercept", 1.0)
    weights = np.sqrt(frame[weight_column].to_numpy(dtype=float))
    values = frame[value_column].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design.to_numpy() * weights[:, None], values * weights, rcond=None)
    fitted = design.to_numpy() @ coefficients
    grand_mean = np.average(values, weights=weights ** 2)
    total = float(np.sum(weights ** 2 * (values - grand_mean) ** 2))
    residual = float(np.sum(weights ** 2 * (values - fitted) ** 2))
    return float(max(0.0, 1 - residual / total)) if total > 0 else 0.0


def sequential_collapse_order(base: pd.DataFrame, mandatory_dims: list, value_column: str, weight_column: str) -> list:
    """The order in which the mandatory dims collapse, decided GREEDILY and SEQUENTIALLY:
    at every step, among the droppable dims (a `_level_N` only when no `_level_(N+1)`
    remains), drop the one whose removal from the CURRENT remaining set loses the least
    R². Extras are not in the model: by the time a mandatory collapses, they are gone.

    WHY NOT the type-II unique contribution: with nested hierarchies (level_1 ⊂ level_2
    ⊂ level_3) every coarser level has unique ≈ 0 while the finer one is present, and a
    dim fully carried by an extra (purchase_type by net_new) has unique ≈ 0 too — but the
    loss of dropping it AFTER the extra is annulled is large. The sequential loss is the
    question the ladder actually asks at each rung.

    OUTPUT:  list of (dim, loss_of_r2) from the first to collapse to the last.
    """
    def family_and_level(dim_name):
        if "_level_" in dim_name:
            head, _, tail = dim_name.rpartition("_level_")
            if tail.isdigit():
                return head, int(tail)
        return dim_name, 1
    remaining, order = list(mandatory_dims), []
    current_r2 = weighted_r2_factorial(base, remaining, value_column, weight_column) if len(base) >= 3 else 0.0
    while remaining:
        droppable = [dim for dim in remaining
                     if not any(family_and_level(other) == (family_and_level(dim)[0], family_and_level(dim)[1] + 1)
                                for other in remaining)]
        losses = {}
        for dim in droppable:
            without = [d for d in remaining if d != dim]
            r2_without = weighted_r2_factorial(base, without, value_column, weight_column) if (len(base) >= 3 and without) else 0.0
            losses[dim] = max(0.0, current_r2 - r2_without)
        next_dim = min(droppable, key=lambda d: (losses[d], d))
        order.append((next_dim, round(losses[next_dim], 4)))
        remaining.remove(next_dim)
        current_r2 = weighted_r2_factorial(base, remaining, value_column, weight_column) if (len(base) >= 3 and remaining) else 0.0
    return order


def dimension_separation(series_summary: pd.DataFrame, units: pd.DataFrame, configuration: Config,
                         branch: str = RATE_BRANCH) -> tuple:
    """The three figures per dimension and the pairs, for the rate branch.

    INPUT:   series_summary (tasa_propia, n_propio, signo) · units (dimension values per
             fs_id) · configuration.
    OUTPUT:  (decision_eta2, decision_eta2_pairs).
             decision_eta2: rama, dimension, grupo (mandatory/extra_renovacion),
             eta2_individual, contribucion_unica, omega2, anulable (0/1: extras only),
             orden_colapso (1 = first mandatory to collapse; 0 for extras),
             perdida_secuencial (R² lost when it collapses, given what remains).
    RULES:   base = trainable series with history, NEUTRAL sign only (timevarying are out
             by sign, and their series would contaminate the rates); weights = n_propio.
    EDGE CASES: fewer than 3 series → every figure 0 (no evidence, declared).
    """
    dims = configuration.business_mandatory_dims + configuration.extra_renovacion
    values = units.drop_duplicates("fs_id")[["fs_id"] + dims]
    base = series_summary[(series_summary["ruta"] == TREATMENT_PREDICTABLE) & series_summary["tasa_propia"].notna()
                          & (series_summary["signo"] == SIGN_NEUTRAL) & (series_summary["n_propio"] > 0)]
    base = base.merge(values, on="fs_id")
    rows = []
    r2_all = weighted_r2_factorial(base, dims, "tasa_propia", "n_propio") if len(base) >= 3 else 0.0
    for dim in dims:
        if len(base) < 3:
            individual = unique = omega = 0.0
        else:
            individual = weighted_eta2(base, dim, "tasa_propia", "n_propio")
            omega = weighted_omega2(base, dim, "tasa_propia", "n_propio")
            unique = max(0.0, r2_all - weighted_r2_factorial(base, [d for d in dims if d != dim], "tasa_propia", "n_propio"))
        rows.append(dict(rama=branch, dimension=dim,
                         grupo="mandatory" if dim in configuration.business_mandatory_dims else "extra_renovacion",
                         eta2_individual=round(individual, 4), contribucion_unica=round(unique, 4),
                         omega2=round(omega, 4), anulable=int(dim in configuration.extra_renovacion)))
    decision = pd.DataFrame(rows)
    # the collapse order of the mandatory dims: sequential loss, extras out of the model
    order = sequential_collapse_order(base, configuration.business_mandatory_dims, "tasa_propia", "n_propio")
    position = {dim: index + 1 for index, (dim, _) in enumerate(order)}
    loss = {dim: value for dim, value in order}
    decision["orden_colapso"] = decision["dimension"].map(position).fillna(0).astype(int)
    decision["perdida_secuencial"] = decision["dimension"].map(loss)
    pair_rows = []
    if len(base) >= 3:
        for a, b in itertools.combinations(dims, 2):
            joint = base.assign(_pair=base[a].astype(str) + "|" + base[b].astype(str))
            eta_pair = weighted_eta2(joint, "_pair", "tasa_propia", "n_propio")
            eta_a = decision.set_index("dimension").loc[a, "eta2_individual"]
            eta_b = decision.set_index("dimension").loc[b, "eta2_individual"]
            pair_rows.append(dict(rama=branch, par=f"{a}×{b}", eta2_par=round(eta_pair, 4),
                                  interaccion=round(eta_pair - max(eta_a, eta_b), 4)))
    pairs = pd.DataFrame(pair_rows, columns=["rama", "par", "eta2_par", "interaccion"])
    return decision, pairs.sort_values("interaccion", ascending=False).head(configuration.eta2_max_pairs)


# ═══════════════════════════════════════════════════════════════════════════════════
# 2 · MIX-SHIFT
# ═══════════════════════════════════════════════════════════════════════════════════

def counterfactual_and_decomposition(units: pd.DataFrame, configuration: Config) -> tuple:
    """The cost of the mandatory-only view and the Kitagawa decomposition, per mandatory cell × month.

    INPUT:   units with tasa (history) · configuration.
    OUTPUT:  (counterfactual, decomposition).
             counterfactual: celda_id, mes, tasa_real, tasa_plano, tasa_seg, err_plano_pp,
             err_seg_pp, ahorro_usd, gana_segmentado (0/1), cota_celda_pp.
             decomposition: celda_id, mes, delta_agregado_pp, delta_comportamiento_pp,
             delta_composicion_pp (vs the previous month with truth).
    RULES:   for month t only data ≤ t−1 is used (cumulative sums shifted by one month);
             the weights of t are the real pipeline of t (known data). Saving =
             (|err_plano| − |err_seg|) × pipeline$ of t, signed. Kitagawa: Δ = Σ w̄·Δp +
             Σ p̄·Δw with midpoint weights/rates, vs the cell's previous month with truth.
    EDGE CASES: cells with fewer than `counterfactual_min_history_months` past months are skipped.
    NOTE:    vectorized (one pass of cumulative sums), so it scales with the number of
             rows, not with cells × months × groupby calls.
    """
    period, ren_col, pipe_col, usd_col = (configuration.period_col, configuration.renewed_units_col,
                                          configuration.pipeline_units_col, configuration.pipeline_usd_col)
    history = units[(units["universo"] == UNIVERSE_NORMAL) & units["tasa"].notna() & (units["sintetica"] == 0)].copy()
    history["celda_id"] = join_columns(history, configuration.business_mandatory_dims)
    empty = (pd.DataFrame(columns=["celda_id", "mes", "tasa_real", "tasa_plano", "tasa_seg", "err_plano_pp", "err_seg_pp",
                                   "ahorro_usd", "gana_segmentado", "cota_celda_pp"]),
             pd.DataFrame(columns=["celda_id", "mes", "delta_agregado_pp", "delta_comportamiento_pp", "delta_composicion_pp"]))
    if history.empty:
        return empty
    # [1] one row per (cell, series, month), with the series' PAST (≤ t−1) as shifted cumsums
    monthly = (history.groupby(["celda_id", "fs_id", period], as_index=False)
               .agg(ren=(ren_col, "sum"), pipe=(pipe_col, "sum"), usd=(usd_col, "sum")).sort_values([period]))
    grouped = monthly.groupby(["celda_id", "fs_id"])
    monthly["ren_past"] = grouped["ren"].cumsum() - monthly["ren"]
    monthly["pipe_past"] = grouped["pipe"].cumsum() - monthly["pipe"]
    # [2] the cell's past and its number of past months
    cell = monthly.groupby(["celda_id", period], as_index=False).agg(ren=("ren", "sum"), pipe=("pipe", "sum"), usd=("usd", "sum"))
    cell = cell.sort_values(period)
    cell_grouped = cell.groupby("celda_id")
    cell["ren_past"] = cell_grouped["ren"].cumsum() - cell["ren"]
    cell["pipe_past"] = cell_grouped["pipe"].cumsum() - cell["pipe"]
    cell["meses_pasados"] = cell_grouped.cumcount()
    cell["mes_anterior"] = cell_grouped[period].shift(1)
    cell["tasa_real"] = cell["ren"] / cell["pipe"].replace(0, np.nan)
    cell["tasa_plano"] = cell["ren_past"] / cell["pipe_past"].replace(0, np.nan)
    window = sorted(history[period].unique())[-configuration.counterfactual_window_months:]
    judged = cell[(cell["meses_pasados"] >= configuration.counterfactual_min_history_months) & cell[period].isin(window)]
    # [3] segmented: the past rate of each series, weighted by the real pipeline of t
    rows = monthly.merge(judged[["celda_id", period, "tasa_plano"]], on=["celda_id", period])
    rows["tasa_serie_pasada"] = (rows["ren_past"] / rows["pipe_past"].replace(0, np.nan)).fillna(rows["tasa_plano"])
    rows["peso_x_tasa"] = rows["pipe"] * rows["tasa_serie_pasada"]
    segmented = rows.groupby(["celda_id", period])["peso_x_tasa"].sum() / rows.groupby(["celda_id", period])["pipe"].sum()
    judged = judged.merge(segmented.rename("tasa_seg").reset_index(), on=["celda_id", period])
    judged["err_plano_pp"] = 100 * (judged["tasa_plano"] - judged["tasa_real"])
    judged["err_seg_pp"] = 100 * (judged["tasa_seg"] - judged["tasa_real"])
    judged["ahorro_usd"] = (judged["err_plano_pp"].abs() - judged["err_seg_pp"].abs()) / 100 * judged["usd"]
    judged["gana_segmentado"] = (judged["err_seg_pp"].abs() < judged["err_plano_pp"].abs()).astype(int)
    judged["cota_celda_pp"] = [binomial_se_pp(r, n) * configuration.z for r, n in zip(judged["tasa_real"], judged["pipe"])]
    counterfactual = judged.assign(mes=judged[period].astype(str))[
        ["celda_id", "mes", "tasa_real", "tasa_plano", "tasa_seg", "err_plano_pp", "err_seg_pp", "ahorro_usd", "gana_segmentado", "cota_celda_pp"]
    ].round({"tasa_real": 4, "tasa_plano": 4, "tasa_seg": 4, "err_plano_pp": 2, "err_seg_pp": 2, "ahorro_usd": 2, "cota_celda_pp": 2})
    # [4] Kitagawa vs the cell's previous month with truth: outer join of the series present now / before
    now = monthly.merge(judged[["celda_id", period, "mes_anterior"]], on=["celda_id", period]).dropna(subset=["mes_anterior"])
    now["p1"] = now["ren"] / now["pipe"].replace(0, np.nan)
    now["w1"] = now["pipe"] / now.groupby(["celda_id", period])["pipe"].transform("sum")
    before = monthly.rename(columns={period: "mes_anterior"})[["celda_id", "fs_id", "mes_anterior", "ren", "pipe"]]
    before["p0"] = before["ren"] / before["pipe"].replace(0, np.nan)
    before["w0"] = before["pipe"] / before.groupby(["celda_id", "mes_anterior"])["pipe"].transform("sum")
    keys = judged[["celda_id", period, "mes_anterior"]].dropna()
    before = before.merge(keys, on=["celda_id", "mes_anterior"])
    joined = now[["celda_id", "fs_id", period, "p1", "w1"]].merge(before[["celda_id", "fs_id", period, "p0", "w0"]],
                                                                on=["celda_id", "fs_id", period], how="outer")
    joined["p0"], joined["p1"] = joined["p0"].fillna(joined["p1"]), joined["p1"].fillna(joined["p0"])
    joined["w0"], joined["w1"] = joined["w0"].fillna(0), joined["w1"].fillna(0)
    joined["comportamiento"] = (joined["w0"] + joined["w1"]) / 2 * (joined["p1"] - joined["p0"])
    joined["composicion"] = (joined["p0"] + joined["p1"]) / 2 * (joined["w1"] - joined["w0"])
    terms = joined.groupby(["celda_id", period], as_index=False)[["comportamiento", "composicion"]].sum()
    decomposition = pd.DataFrame(dict(celda_id=terms["celda_id"], mes=terms[period].astype(str),
                                      delta_agregado_pp=(100 * (terms["comportamiento"] + terms["composicion"])).round(2),
                                      delta_comportamiento_pp=(100 * terms["comportamiento"]).round(2),
                                      delta_composicion_pp=(100 * terms["composicion"]).round(2)))
    return counterfactual.reset_index(drop=True), decomposition


# ═══════════════════════════════════════════════════════════════════════════════════
# 3 · TIMEVARYING CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════════

def timevarying_calibration(units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Realized rate per active flag and per sign, month by month, with its binomial error.

    OUTPUT:  DataFrame(tipo ("flag"/"signo"), nombre, mes, n, tasa_realizada, error_pp,
             version_modelo). The audit of every model that produces a flag.
    """
    period, ren_col, pipe_col = configuration.period_col, configuration.renewed_units_col, configuration.pipeline_units_col
    history = units[(units["universo"] == UNIVERSE_NORMAL) & units["tasa"].notna() & (units["sintetica"] == 0)]
    rows = []
    for flag, sign in configuration.structural_timevarying_dims.items():
        active = history[history[flag].isin(configuration.timevarying_positive_values)]
        for month, month_rows in active.groupby(period):
            n, rate = month_rows[pipe_col].sum(), month_rows[ren_col].sum() / max(month_rows[pipe_col].sum(), 1)
            rows.append(dict(tipo="flag", nombre=flag, signo=sign, mes=str(month), n=float(n), tasa_realizada=round(rate, 4),
                             error_pp=round(binomial_se_pp(rate, n) * configuration.z, 2),
                             version_modelo=configuration.timevarying_model_version.get(flag, "")))
    for sign_value, sign_rows in history.merge(
            units.drop_duplicates("fs_id")[["fs_id"]], on="fs_id").assign(
            _sign=lambda f: f["fs_id"].map(_sign_lookup(units, configuration))).groupby("_sign"):
        for month, month_rows in sign_rows.groupby(period):
            n, rate = month_rows[pipe_col].sum(), month_rows[ren_col].sum() / max(month_rows[pipe_col].sum(), 1)
            rows.append(dict(tipo="signo", nombre=sign_value, signo=sign_value, mes=str(month), n=float(n),
                             tasa_realizada=round(rate, 4), error_pp=round(binomial_se_pp(rate, n) * configuration.z, 2),
                             version_modelo=""))
    return pd.DataFrame(rows, columns=["tipo", "nombre", "signo", "mes", "n", "tasa_realizada", "error_pp", "version_modelo"])


def _sign_lookup(units: pd.DataFrame, configuration: Config) -> pd.Series:
    from run_rate_series import series_sign_table
    table = series_sign_table(units, configuration)
    return table.set_index("fs_id")["signo"]


# ═══════════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════════

def print_mix_shift_by_cell(counterfactual: pd.DataFrame, decomposition: pd.DataFrame, top: int = 12) -> None:
    """Console: the cells where segmenting pays (or costs) and where the mix moves the rate.

    The same view in SQL (Power BI):
        SELECT c.celda_id, SUM(c.ahorro_usd) ahorro_usd, AVG(c.gana_segmentado) pct_gana,
               AVG(ABS(m.delta_composicion_pp)) riesgo_mix_pp, AVG(ABS(m.delta_comportamiento_pp)) mov_comportamiento_pp
        FROM sff_mandatory_only_cost c LEFT JOIN sff_mix_shift m ON m.celda_id = c.celda_id AND m.mes = c.mes
        GROUP BY c.celda_id ORDER BY ahorro_usd DESC
    Reading: `ahorro_usd` > 0 → in that cell, predicting the parts with the real weights of
    the month beat predicting the whole (the flat rate); `riesgo_mix_pp` is how much the
    cell's aggregate rate moves per month by composition alone (Kitagawa).
    """
    by_cell = counterfactual.groupby("celda_id").agg(ahorro_usd=("ahorro_usd", "sum"), pct_gana=("gana_segmentado", "mean"),
                                                      meses=("mes", "nunique"))
    if len(decomposition):
        mix = decomposition.groupby("celda_id").agg(riesgo_mix_pp=("delta_composicion_pp", lambda s: float(s.abs().mean())),
                                                    comportamiento_pp=("delta_comportamiento_pp", lambda s: float(s.abs().mean())))
        by_cell = by_cell.join(mix, how="left")
    total_saving = by_cell["ahorro_usd"].sum()
    winners = by_cell[by_cell["ahorro_usd"] > 0]
    print(f"[1.4] where segmenting pays: {len(winners)} of {len(by_cell)} cells with positive saving hold "
          f"${winners['ahorro_usd'].sum():,.0f}; the {len(by_cell) - len(winners)} others cost ${-by_cell.loc[by_cell['ahorro_usd'] <= 0, 'ahorro_usd'].sum():,.0f} "
          f"(net ${total_saving:,.0f})")
    print("   cell · saving $ · % months segmented wins · mix risk pp (composition) · behaviour pp")
    for cell, row in by_cell.sort_values("ahorro_usd", ascending=False).head(top).iterrows():
        print(f"   {str(cell):<40} ${row['ahorro_usd']:>11,.0f}  {row['pct_gana']:5.0%}  "
              f"{row.get('riesgo_mix_pp', float('nan')):5.2f}  {row.get('comportamiento_pp', float('nan')):5.2f}")
    worst = by_cell.sort_values("ahorro_usd").head(3)
    for cell, row in worst.iterrows():
        if row["ahorro_usd"] < 0:
            print(f"   {str(cell):<40} ${row['ahorro_usd']:>11,.0f}  {row['pct_gana']:5.0%}  (flat method wins here)")


def run_dimension_analysis(units: pd.DataFrame, series_summary: pd.DataFrame, configuration: Config) -> dict:
    """Phase 1.2 · what separates behaviour, and in which order the mandatory dims collapse.
    Persists decision_eta2 and decision_eta2_pairs. (Composition and signals: phase 1.4.)"""
    decision, pairs = dimension_separation(series_summary, units, configuration)
    configuration.write(decision, "decision_eta2")
    configuration.write(pairs, "decision_eta2_pairs")
    print("[1.2] dimension separation (neutral series, weights = support):")
    for _, row in decision.iterrows():
        collapse = f"collapses #{row['orden_colapso']} (loses R² {row['perdida_secuencial']:.3f})" if row["orden_colapso"] else "(annullable extra)"
        print(f"   {row['dimension']:<26} η²={row['eta2_individual']:.3f}  unique={row['contribucion_unica']:.3f}  ω²={row['omega2']:.3f}  {collapse}")
    explain(configuration,
            "η² = share of the variance of the series' rates that a dimension explains ALONE (0 = the rate is the same across its values; 1 = it decides the rate).",
            "'unique' = what is lost if that dimension is removed while the others stay (0 when another dimension carries the same information, e.g. nested levels).",
            "'collapses #k' = the order in which the ladder drops mandatory dims to find relatives: the ones that separate least fall first, so pooled cohorts stay alike.")
    return dict(decision_eta2=decision, decision_eta2_pairs=pairs)


def run_composition_analysis(units: pd.DataFrame, configuration: Config) -> dict:
    """Phase 1.4 · composition and signals, AFTER the ladder: how much of the movement of
    the cells is mix (Kitagawa), what the mandatory-only view would cost in dollars, and
    the realized rate of every flag and sign. Nothing downstream decides on these: they
    are the account of composition the business asked for, not a search for cases."""
    counterfactual, decomposition = counterfactual_and_decomposition(units, configuration)
    calibration = timevarying_calibration(units, configuration)
    configuration.write(counterfactual, "mandatory_only_cost")
    configuration.write(decomposition, "mix_shift_decomposition")
    configuration.write(calibration, "timevarying_calibration")
    if len(counterfactual):
        print(f"[1.4] the mandatory-only view (one rate per cell) vs the segmented series, walk-forward over "
              f"{counterfactual['mes'].nunique()} months: segmenting saves ${counterfactual['ahorro_usd'].sum():,.0f} net · "
              f"segmented is closer in {counterfactual['gana_segmentado'].mean():.0%} of cell-months")
        print_mix_shift_by_cell(counterfactual, decomposition)
    if len(decomposition):
        composition_share = (decomposition["delta_composicion_pp"].abs().sum()
                             / max((decomposition["delta_composicion_pp"].abs() + decomposition["delta_comportamiento_pp"].abs()).sum(), 1e-9))
        print(f"[1.4] composition: {composition_share:.0%} of the month-to-month movement of cell rates is the mix of the pipeline, not behaviour "
              f"(Σ|composición| / (Σ|composición| + Σ|comportamiento|) over every cell-month of the window)")
        explain(configuration,
                "A cell's rate can move because its people renew differently (behaviour) or because different people fall due this month (composition).",
                "Kitagawa splits each month's change into the two. A high composition share means the aggregate rate is not a behaviour to forecast:",
                "it is the mix of the known pipeline applied to the rates of its parts — which is exactly what predicting by series and summing does.")
    if len(calibration):
        by_sign = calibration[calibration["tipo"] == "signo"].groupby("nombre").apply(
            lambda g: float(np.average(g["tasa_realizada"], weights=np.maximum(g["n"], 1))), include_groups=False)
        print(f"[1.4] signals: realized rate by sign {by_sign.round(3).to_dict()} (the flags carve the risky and the safe customers out of the cohorts)")
    return dict(mandatory_only_cost=counterfactual, mix_shift_decomposition=decomposition, timevarying_calibration=calibration)
