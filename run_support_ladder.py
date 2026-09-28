"""run_support_ladder.py — SFF v3 · RUN · phase 1.3: one ladder, one loop.

"The series borrows support from the nearest relative that has enough, and trusts it in
proportion to how little it has itself."

For every forecast series an ORDERED LIST OF RELATIVES is built (`build_relatives`), from
the closest to the farthest; the series climbs it until the first relative whose monthly
support reaches the floor (`climb_ladder`); its rate is then a credibility blend of its
own rate and the relative's (`estimate_rates`). Every rung is recorded (`parent_ladder`),
the chosen one is the decision (`decision_support`).

THE SIGN NEVER GETS LOST. The timevarying flags are summarized into the sign of the
series (neutral / neg / pos / mixed) and every relative keeps that sign:

    series WITH sign (neg or pos)                series NEUTRAL (no active flag)
    R0  itself                                   R0  itself
    R1  same dims, flags summarized in the sign  —
    R2  the annullable extra set to '*'          R2  the annullable extra set to '*'
    R3  the mandatory cell × sign   ← TOP        R3  the mandatory cell, neutrals only
                                                 R4… mandatory dims collapsed in η² order
                                                 Rk  the total of neutrals

A signed series that reaches the top still below the floor keeps the best rate found
inside its sign, with NO credibility toward anyone, and is labeled `signed_under_floor`:
noisy, but more qualified than a big neutral series. A `mixed` series (both signs) is
never pooled: it keeps its own rate and is counted.

Pools are computed with EVERY series matching the pattern (big siblings included): the
pattern decides who computes the number; the ladder decides who receives it.

Persisted column names stay in Spanish (`etapa`, `peldano`, `padre_id`, `tasa`...).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from binomial_reference import binomial_se_pp, wilson_half_width_pp
from config import ID_FIELD_SEPARATOR, Config, explain
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from run_rate_series import (ROLE_PROJECTION, TREATMENT_PREDICTABLE, SIGN_MIXED, SIGN_NEUTRAL,
                             UNIVERSE_NORMAL)

# ─── named constants ─────────────────────────────────────────────────────────────
WILDCARD = "*"
SIGN_FIELD_PREFIX = "SIG="
# Risk levels (persisted in `nivel_riesgo`)
# (close_relative_max_rung and own_level_min_history_months are Config parameters)


# ═══════════════════════════════════════════════════════════════════════════════════
# RELATIVES
# ═══════════════════════════════════════════════════════════════════════════════════

def collapse_order(mandatory_dims: list, unique_contribution: dict) -> list:
    """The order in which mandatory dims collapse: finest level of a family first,
    then the family whose droppable dim contributes least (unique contribution).

    INPUT:   mandatory_dims · unique_contribution — dim → η² unique (from decision_eta2).
    OUTPUT:  ordered list of dims (first = collapses first).
    RULES:   a dim `x_level_N` is droppable only when no `x_level_(N+1)` remains; among
             droppable dims the lowest unique contribution goes first (ties by name).
    """
    def family_and_level(dim_name):
        if "_level_" in dim_name:
            head, _, tail = dim_name.rpartition("_level_")
            if tail.isdigit():
                return head, int(tail)
        return dim_name, 1
    pending, order = list(mandatory_dims), []
    while pending:
        droppable = [dim for dim in pending
                     if not any(family_and_level(other) == (family_and_level(dim)[0], family_and_level(dim)[1] + 1)
                                for other in pending)]
        next_dim = min(droppable, key=lambda d: (unique_contribution.get(d, 0.0), d))
        order.append(next_dim)
        pending.remove(next_dim)
    return order


def relative_pattern(series_values: dict, configuration: Config, sign: str,
                     annulled_extras: set, collapsed_mandatory: set, summarize_sign: bool) -> str:
    """The id pattern of one relative: the rate series columns in order, with the
    timevarying block replaced by `SIG=<sign>` when summarized, and '*' on annulled dims.

    The pattern is the KEY of the pool: two series with the same pattern compute together.
    """
    fields = []
    timevarying = list(configuration.structural_timevarying_dims)
    sign_written = False
    for column in configuration.rate_series_columns:
        if column in timevarying and summarize_sign:
            if not sign_written:
                fields.append(SIGN_FIELD_PREFIX + sign)
                sign_written = True
            continue
        if column in annulled_extras or column in collapsed_mandatory:
            fields.append(WILDCARD)
        else:
            fields.append(str(series_values[column]))
    return ID_FIELD_SEPARATOR.join(fields)


def build_relatives(series_values: dict, sign: str, configuration: Config,
                    annullable_extra, mandatory_collapse_order: list, collapse_loss: dict = None) -> list:
    """The ordered list of relatives of one series (pure: no data, no console).

    INPUT:   series_values — column → value of the series · sign — neutral/neg/pos/mixed ·
             configuration · annullable_extra — the extra_renovacion with the lowest
             unique η² (None if no extras) · mandatory_collapse_order.
    OUTPUT:  list of (rung, description, pattern). Rung 0 is the series itself.
             collapse_loss — dim → R² lost when it collapses (decision_eta2.perdida_secuencial).
    RULES:   see the module docstring. A `mixed` series has only rung 0. A signed series
             stops at the cell × sign unless `signed_ladder_max_loss` > 0: then it keeps
             collapsing mandatory dims (sign kept) while the cumulative loss ≤ that cap.
    """
    relatives = [(0, "itself", relative_pattern(series_values, configuration, sign, set(), set(), False))]
    if sign == SIGN_MIXED:
        return relatives
    if sign != SIGN_NEUTRAL and configuration.structural_timevarying_dims:
        relatives.append((1, "same sign", relative_pattern(series_values, configuration, sign, set(), set(), True)))
    annulled = {annullable_extra} if annullable_extra else set()
    if annulled:
        relatives.append((2, f"extra '{annullable_extra}' annulled",
                          relative_pattern(series_values, configuration, sign, annulled, set(), True)))
    all_extras = set(configuration.extra_renovacion)
    relatives.append((3, "mandatory cell × sign",
                      relative_pattern(series_values, configuration, sign, all_extras, set(), True)))
    collapsed, cumulative_loss = set(), 0.0
    for rung_offset, dim in enumerate(mandatory_collapse_order, start=4):
        if sign != SIGN_NEUTRAL:
            if configuration.signed_ladder_max_loss <= 0:
                break                              # default rule: the cell × sign is the top
            cumulative_loss += float((collapse_loss or {}).get(dim, 1.0))
            if cumulative_loss > configuration.signed_ladder_max_loss + 1e-12:
                break                              # cohorts start to differ: stop here, sign kept
        collapsed = collapsed | {dim}
        relatives.append((rung_offset, f"mandatory '{dim}' collapsed" + (" (sign kept)" if sign != SIGN_NEUTRAL else ""),
                          relative_pattern(series_values, configuration, sign, all_extras, set(collapsed), True)))
    return relatives


# ═══════════════════════════════════════════════════════════════════════════════════
# POOLS
# ═══════════════════════════════════════════════════════════════════════════════════

def series_patterns_table(series_summary: pd.DataFrame, units: pd.DataFrame, configuration: Config,
                          annullable_extra, mandatory_collapse_order: list, collapse_loss: dict = None) -> pd.DataFrame:
    """Every (fs_id, rung, pattern) of every series: the map used to compute pools."""
    one_per_series = (units[units["fs_id"].isin(series_summary["fs_id"])]
                      .drop_duplicates("fs_id").set_index("fs_id")[configuration.rate_series_columns])
    sign_by_series = series_summary.set_index("fs_id")["signo"]
    rows = []
    for series_id, values in one_per_series.iterrows():
        for rung, description, pattern in build_relatives(values.to_dict(), sign_by_series.get(series_id, SIGN_NEUTRAL),
                                                          configuration, annullable_extra, mandatory_collapse_order, collapse_loss):
            rows.append(dict(fs_id=series_id, peldano=rung, descripcion=description, patron=pattern))
    return pd.DataFrame(rows, columns=["fs_id", "peldano", "descripcion", "patron"])


def pool_support(units: pd.DataFrame, patterns: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Support and rate of every pattern, computed with ALL the series that match it.

    INPUT:   units with tasa (history) · patterns (fs_id, peldano, patron) · configuration.
    OUTPUT:  DataFrame(patron, n_pool, tasa_pool, ren_pool, pipe_pool): n = median of the
             monthly SUM of pipeline units over the matching series (sum by month first,
             then the median of the months with support); tasa = Σren/Σpipe.
    RULES:   only history rows of the normal universe with a defined rate.
    """
    history = units[(units["universo"] == UNIVERSE_NORMAL) & units["tasa"].notna()]
    if patterns.empty:
        return pd.DataFrame(columns=["patron", "n_pool", "ren_pool", "pipe_pool", "tasa_pool"])
    membership = patterns[["fs_id", "patron"]].drop_duplicates()
    joined = history.merge(membership, on="fs_id")
    monthly = joined.groupby(["patron", configuration.period_col]).agg(
        ren=(configuration.renewed_units_col, "sum"), pipe=(configuration.pipeline_units_col, "sum")).reset_index()
    pooled = monthly.groupby("patron").agg(
        n_pool=("pipe", lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0),
        ren_pool=("ren", "sum"), pipe_pool=("pipe", "sum")).reset_index()
    pooled["tasa_pool"] = np.where(pooled["pipe_pool"] > 0, pooled["ren_pool"] / pooled["pipe_pool"].replace(0, np.nan), np.nan)
    return pooled


# ═══════════════════════════════════════════════════════════════════════════════════
# CLIMB AND ESTIMATE
# ═══════════════════════════════════════════════════════════════════════════════════

def climb_ladder(series_summary: pd.DataFrame, patterns: pd.DataFrame, pools: pd.DataFrame,
                 configuration: Config) -> tuple:
    """Climb rung by rung until the first relative whose support reaches the floor.

    INPUT:   series_summary (signo, ruta, n_propio) · patterns · pools · configuration.
    OUTPUT:  (decision_support, parent_ladder).
             decision_support: fs_id → id_estimacion (pattern of the chosen relative),
             peldano, n_efectivo, tasa_pariente, alcanzo_suelo (0/1), signo, ruta.
             parent_ladder: fs_id × peldano with n, tasa and elegido (the trace).
    RULES:   rung 0 (the series itself) is chosen only when its n ≥ own_rate_floor (it
             predicts alone). Otherwise the chosen relative is the first rung ≥ 1 with
             n_pool ≥ support_floor (the series blends with it). If none reaches the
             support floor: rung 0 when the series itself has n ≥ support_floor (own rate,
             no reinforcement available), else the LAST rung (best available), flagged
             `alcanzo_suelo = 0`. Non-trainable series get no decision.
    """
    ladder = patterns.merge(pools, on="patron", how="left").fillna({"n_pool": 0.0})
    decisions, trace_rows = [], []
    decision_columns = ["fs_id", "signo", "ruta", "id_estimacion", "peldano", "n_efectivo", "tasa_pariente", "alcanzo_suelo"]
    trace_columns = ["fs_id", "peldano", "descripcion", "padre_id", "n_padre", "tasa_padre", "elegido"]
    info = series_summary.set_index("fs_id")
    for series_id, rungs in ladder.groupby("fs_id", sort=False):
        rungs = rungs.sort_values("peldano")
        chosen, chosen_trace_position, own_trace_position = None, None, None
        for _, rung in rungs.iterrows():
            trace_rows.append(dict(fs_id=series_id, peldano=int(rung["peldano"]), descripcion=rung["descripcion"],
                                   padre_id=rung["patron"], n_padre=round(float(rung["n_pool"]), 1),
                                   tasa_padre=round(float(rung["tasa_pool"]), 4) if np.isfinite(rung["tasa_pool"]) else np.nan,
                                   elegido=0))
            if rung["peldano"] == 0:
                own_trace_position = len(trace_rows) - 1
                if rung["n_pool"] >= configuration.own_rate_floor:      # precise enough to speak alone
                    chosen, chosen_trace_position = rung, own_trace_position
            elif chosen is None and rung["n_pool"] >= configuration.support_floor:
                chosen, chosen_trace_position = rung, len(trace_rows) - 1
        reached_floor = chosen is not None
        if not reached_floor and rungs.iloc[0]["n_pool"] >= configuration.support_floor:
            chosen, chosen_trace_position, reached_floor = rungs.iloc[0], own_trace_position, True   # own, unreinforced
        if not reached_floor:                       # nobody reached the floor: best available = last rung
            chosen, chosen_trace_position = rungs.iloc[-1], len(trace_rows) - 1
        trace_rows[chosen_trace_position]["elegido"] = 1
        decisions.append(dict(fs_id=series_id, signo=info.loc[series_id, "signo"], ruta=info.loc[series_id, "ruta"],
                              id_estimacion=chosen["patron"], peldano=int(chosen["peldano"]),
                              n_efectivo=float(chosen["n_pool"]), tasa_pariente=float(chosen["tasa_pool"]),
                              alcanzo_suelo=int(reached_floor)))
    return pd.DataFrame(decisions, columns=decision_columns), pd.DataFrame(trace_rows, columns=trace_columns)


def estimate_credibility_k(series_summary: pd.DataFrame, decision_support: pd.DataFrame,
                           configuration: Config) -> pd.Series:
    """Bühlmann-Straub k per chosen relative: within-series variance / between-series variance.

    INPUT:   series_summary (tasa_propia, n_propio, pipe) · decision_support · configuration.
    OUTPUT:  Series id_estimacion → k. Falls back to `k_cred` when fewer than 3 siblings
             with history or when the between variance is not positive.
    RULES:   within = mean over siblings of p_i(1−p_i); between = weighted variance of
             p_i minus the sampling part; k = within / between. A homogeneous group
             (between ≈ 0) gets a very large k → the series takes the pool rate; a
             heterogeneous group gets a small k → the series keeps its own rate.
    """
    joined = decision_support.merge(series_summary[["fs_id", "tasa_propia", "n_propio", "pipe"]], on="fs_id")
    k_by_relative = {}
    for relative, siblings in joined.groupby("id_estimacion"):
        siblings = siblings[siblings["tasa_propia"].notna() & (siblings["pipe"] > 0)]
        if len(siblings) < 3:
            k_by_relative[relative] = configuration.k_cred
            continue
        rates, weights = siblings["tasa_propia"].to_numpy(), siblings["pipe"].to_numpy()
        pooled = float(np.average(rates, weights=weights))
        within = float(np.mean(rates * (1 - rates)))
        raw_between = float(np.average((rates - pooled) ** 2, weights=weights))
        sampling_part = float(np.average(rates * (1 - rates) / np.maximum(weights, 1), weights=weights))
        between = raw_between - sampling_part
        k_by_relative[relative] = float(within / between) if between > 1e-6 else 10 * configuration.k_cred
    return pd.Series(k_by_relative, name="k")


def estimate_rates(series_summary: pd.DataFrame, decision_support: pd.DataFrame,
                   k_by_relative: pd.Series, configuration: Config) -> pd.DataFrame:
    """The estimated rate of every series and its TWO errors.

    INPUT:   series_summary · decision_support · k_by_relative · configuration.
    OUTPUT:  series_estimates: fs_id, id_estimacion, peldano, n_efectivo, tasa_pariente, k, z,
             tasa_estimada, se_estimacion_pp, se_prediccion_pp, nivel_riesgo.
    RULES:   tasa = z·own + (1−z)·parent with z = n_propio/(n_propio+k) ONLY when a relative
             with support was found and the relative is not the series itself; a series
             that is its own relative (rung 0 reached), or signed_under_floor, or mixed,
             uses the best rate found without blend (z = 1).
             se_estimacion = √(z²·se_own² + (1−z)²·se_parent²) (errors of the ESTIMATE);
             se_prediccion = √(se_estimacion² + binomial_se(p, n_propio)²): the month we
             will predict still samples with the series' own n — this never shrinks.
    """
    joined = decision_support.merge(series_summary, on=["fs_id", "signo", "ruta"], how="left")
    rows = []
    for _, series in joined.iterrows():
        own_rate, own_n = series["tasa_propia"], float(series["n_propio"])
        parent_rate, parent_n = series["tasa_pariente"], float(series["n_efectivo"])
        blends = (series["alcanzo_suelo"] == 1 and series["peldano"] > 0 and np.isfinite(parent_rate))
        k = float(k_by_relative.get(series["id_estimacion"], configuration.k_cred))
        if blends and np.isfinite(own_rate):
            z = own_n / (own_n + k)
            rate = z * own_rate + (1 - z) * parent_rate
            se_estimation = np.sqrt((z * binomial_se_pp(own_rate, own_n)) ** 2
                                    + ((1 - z) * binomial_se_pp(parent_rate, parent_n)) ** 2)
        else:
            z = 1.0
            rate = own_rate if series["peldano"] == 0 or not np.isfinite(parent_rate) else parent_rate
            n_used = own_n if series["peldano"] == 0 else parent_n
            se_estimation = binomial_se_pp(rate, n_used) if np.isfinite(rate) else np.nan
        se_prediction = (np.sqrt(se_estimation ** 2 + binomial_se_pp(rate, max(own_n, 1)) ** 2)
                         if np.isfinite(rate) else np.nan)
        rows.append(dict(fs_id=series["fs_id"], id_estimacion=series["id_estimacion"], peldano=series["peldano"],
                         n_efectivo=series["n_efectivo"], tasa_pariente=parent_rate, alcanzo_suelo=series["alcanzo_suelo"], k=round(k, 1),
                         z=round(z, 3), tasa_estimada=rate, se_estimacion_pp=se_estimation,
                         se_prediccion_pp=se_prediction,
                         nivel_riesgo=risk_level(series, configuration)))
    return pd.DataFrame(rows)


# One line per level: the RULE that assigns it and what it means for the prediction.
# Printed after the money table so the reader never has to guess what a level is.
LEVEL_DEFINITIONS = [
    (LEVEL_OWN,               "n ≥ own_rate_floor y ≥ 12 meses",   "precisión propia (±5 pp o mejor): predice sola, z = 1"),
    (LEVEL_OWN_SHORT,         "n ≥ own_rate_floor y < 12 meses",   "precisión propia pero historia corta: aún no ha visto todas las estaciones"),
    (LEVEL_OWN_REINFORCED,    "suelo ≤ n < own_rate_floor",         "evidencia propia sin precisión: usa su tasa y la completa con su primer pariente con soporte (z = n/(n+k)); si no lo hay, va sola"),
    (LEVEL_BORROWED,          "peldaño 1-2 con suelo",           "bajo el suelo; toma la tasa de un pariente que comparte TODAS las mandatory (mismo signo o la extra anulada): cohorte casi idéntica"),
    (LEVEL_FAR,               "peldaño ≥ 3 con suelo",           "bajo el suelo; el pariente es la celda mandatory entera o una mandatory colapsada: cohorte más gruesa, riesgo de heredar comportamiento ajeno"),
    (LEVEL_SIGNED_UNDER_FLOOR, "con señal y ningún peldaño llega", "tiene una señal (flag neg/pos) y su escalera termina en celda × signo sin llegar al suelo: predice con la mejor tasa de SU signo, ruidosa, porque no se le deja mezclarse con series sin señal"),
    (LEVEL_MIXED,             "flags de ambos signos",           "dos modelos marcan al mismo cliente en sentidos opuestos: se queda solo y se cuenta; debería ser ≈ 0"),
    (LEVEL_NO_HISTORY,        "sin historia (solo proyección)",  "no hay tasa que estimar: cascada celda → global; la banda es la binomial"),
    (LEVEL_NO_IMPACT,         "sin proyección (solo historia)",  "nada que predecir; se conserva para el backtest y los pools"),
    (LEVEL_TIME_SERIES,       "universo time_series",            "reservado: solo etiquetado, forecast por cascada"),
]


def print_level_definitions() -> None:
    """The legend of the risk levels (console, after the money table)."""
    print("[1.3] risk levels · rule → meaning")
    for level, rule, meaning in LEVEL_DEFINITIONS:
        print(f"   {level:<22} {rule:<34} {meaning}")


def risk_level(series: pd.Series, configuration: Config) -> str:
    """The predictive-quality level of a series. Rules in LEVEL_DEFINITIONS, in order:
    route / universe first, then sign, then whether the floor was reached, then the rung."""
    if series["ruta"] == TREATMENT_HISTORY_ONLY:
        return LEVEL_NO_IMPACT
    if series["universo"] != UNIVERSE_NORMAL:
        return LEVEL_TIME_SERIES
    if series["ruta"] == TREATMENT_FUTURE_ONLY or series["meses_historia"] == 0:
        return LEVEL_NO_HISTORY
    if series["signo"] == SIGN_MIXED:
        return LEVEL_MIXED
    if series["alcanzo_suelo"] == 0:
        return LEVEL_SIGNED_UNDER_FLOOR if series["signo"] != SIGN_NEUTRAL else LEVEL_FAR
    if series["n_propio"] >= configuration.support_floor and series["n_propio"] < configuration.own_rate_floor:
        return LEVEL_OWN_REINFORCED
    if series["peldano"] == 0:
        return LEVEL_OWN if series["meses_historia"] >= configuration.own_level_min_history_months else LEVEL_OWN_SHORT
    if series["peldano"] <= configuration.close_relative_max_rung:
        return LEVEL_BORROWED
    return LEVEL_FAR


# ═══════════════════════════════════════════════════════════════════════════════════
# CARD, CHAIN AND REPORT
# ═══════════════════════════════════════════════════════════════════════════════════

def build_support_chain(series_summary: pd.DataFrame, parent_ladder: pd.DataFrame,
                        series_estimates: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The money waterfall: per series and stage (0_raw, each rung climbed, final), the
    support and the binomial error on its own projected money."""
    rows = []
    money = series_summary.set_index("fs_id")["usd_proyectado"]
    estimates = series_estimates.set_index("fs_id")
    raw = series_summary.set_index("fs_id")
    for series_id, rungs in parent_ladder.groupby("fs_id"):
        chosen_rung = estimates.loc[series_id, "peldano"]
        for _, rung in rungs[rungs["peldano"] <= chosen_rung].iterrows():
            rate = rung["tasa_padre"] if np.isfinite(rung["tasa_padre"]) else raw.loc[series_id, "tasa_propia"]
            rows.append(dict(fs_id=series_id, etapa=f"{int(rung['peldano'])}_{rung['descripcion']}",
                             id_efectivo=rung["padre_id"], n_efectivo=rung["n_padre"], tasa=rate,
                             se_pp=binomial_se_pp(rate if np.isfinite(rate) else 0.5, rung["n_padre"]),
                             usd_proyectado=money.get(series_id, 0.0)))
        final = estimates.loc[series_id]
        rows.append(dict(fs_id=series_id, etapa="9_final", id_efectivo=f"z={final['z']}→{final['id_estimacion']}",
                         n_efectivo=final["n_efectivo"], tasa=final["tasa_estimada"],
                         se_pp=final["se_estimacion_pp"], usd_proyectado=money.get(series_id, 0.0)))
    chain = pd.DataFrame(rows, columns=["fs_id", "etapa", "id_efectivo", "n_efectivo", "tasa", "se_pp", "usd_proyectado"])
    chain["moe_usd"] = configuration.z * chain["se_pp"] / 100 * chain["usd_proyectado"]
    return chain


def build_series_card(series_summary: pd.DataFrame, series_estimates: pd.DataFrame) -> pd.DataFrame:
    """The card: one row per series with every attribute known so far (phases 0-1)."""
    return series_summary.merge(series_estimates, on="fs_id", how="left")


def risk_levels_report(series_card: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Money by risk level: the headline table of every phase.

    OUTPUT:  DataFrame(etapa, nivel_riesgo, series, usd, pct_usd, error_medio_pp) sorted by level.
    """
    total = max(series_card["usd_proyectado"].sum(), 1.0)
    report = series_card.groupby("nivel_riesgo").apply(lambda g: pd.Series({
        "series": len(g), "usd": g["usd_proyectado"].sum(),
        "pct_usd": 100 * g["usd_proyectado"].sum() / total,
        "error_medio_pp": (np.average(g["se_prediccion_pp"].fillna(g["error_binomial_pp"]).dropna(),
                                      weights=np.maximum(g.loc[g["se_prediccion_pp"].fillna(g["error_binomial_pp"]).notna(), "usd_proyectado"], 1e-9))
                           if g["se_prediccion_pp"].fillna(g["error_binomial_pp"]).notna().any() else np.nan)}),
        include_groups=False).reset_index()
    report.insert(0, "etapa", stage)
    return report.sort_values("nivel_riesgo").reset_index(drop=True)


def print_risk_levels(report: pd.DataFrame) -> None:
    """Console version of the report."""
    print(f"[1.3] money by risk level · {report['etapa'].iloc[0] if len(report) else ''}")
    for _, row in report.iterrows():
        error = f"error ±{row['error_medio_pp']:.1f} pp" if np.isfinite(row["error_medio_pp"]) else "error n/a (no rate)"
        print(f"   {row['nivel_riesgo']:<22} {int(row['series']):>4} series  ${row['usd']:>12,.0f}  {row['pct_usd']:5.1f}%  {error}")


def run_support_ladder(units: pd.DataFrame, series_summary: pd.DataFrame, decision_eta2: pd.DataFrame,
                       configuration: Config) -> tuple:
    """Phase 1.3 end to end. Persists decision_support, parent_ladder, support_chain,
    series_card and the risk-level report.

    INPUT:   units (1.1) · series_summary (1.1) · decision_eta2 (1.2: dimension, rama,
             contribucion_unica, anulable) · configuration.
    OUTPUT:  (series_estimates, series_card, decision_support, parent_ladder).
    STEPS:
      [1] Collapse order and annullable extra from decision_eta2.
      [2] Relatives of every series; pools of every pattern.
      [3] Climb; k per relative; estimate rates and levels.
      [4] Chain, card, report; persist.
    """
    rate_branch = decision_eta2[decision_eta2["rama"] == "tasa"]
    unique = dict(zip(rate_branch["dimension"], rate_branch["contribucion_unica"]))
    if "orden_colapso" in rate_branch.columns and (rate_branch["orden_colapso"] > 0).any():
        ordered = rate_branch[rate_branch["orden_colapso"] > 0].sort_values("orden_colapso")
        order = [d for d in ordered["dimension"] if d in configuration.business_mandatory_dims]
        order += [d for d in configuration.business_mandatory_dims if d not in order]     # safety: never lose a dim
    else:                                                                                   # older decision tables
        order = collapse_order(configuration.business_mandatory_dims, unique)
    collapse_loss = (dict(zip(rate_branch["dimension"], rate_branch["perdida_secuencial"].fillna(1.0)))
                     if "perdida_secuencial" in rate_branch.columns else {})
    extras = [d for d in configuration.extra_renovacion]
    annullable = min(extras, key=lambda d: unique.get(d, 1.0)) if extras else None
    trainable = series_summary[(series_summary["ruta"] == TREATMENT_PREDICTABLE) & (series_summary["universo"] == UNIVERSE_NORMAL)]
    patterns = series_patterns_table(trainable, units, configuration, annullable, order, collapse_loss)
    pools = pool_support(units, patterns, configuration)
    decision_support, parent_ladder = climb_ladder(trainable, patterns, pools, configuration)
    k_by_relative = estimate_credibility_k(series_summary, decision_support, configuration)
    non_trainable = series_summary[~series_summary["fs_id"].isin(trainable["fs_id"])]
    decision_support = pd.concat([decision_support, pd.DataFrame(dict(
        fs_id=non_trainable["fs_id"], signo=non_trainable["signo"], ruta=non_trainable["ruta"],
        id_estimacion=non_trainable["fs_id"], peldano=0, n_efectivo=0.0, tasa_pariente=np.nan, alcanzo_suelo=0))],
        ignore_index=True)
    series_estimates = estimate_rates(series_summary, decision_support, k_by_relative, configuration)
    card = build_series_card(series_summary, series_estimates)
    chain = build_support_chain(series_summary, parent_ladder, series_estimates, configuration)
    report = risk_levels_report(card, "1_soporte")
    configuration.write(decision_support.assign(k=decision_support["id_estimacion"].map(k_by_relative).fillna(configuration.k_cred)), "decision_support")
    configuration.write(parent_ladder, "parent_ladder")
    configuration.write(chain, "support_chain")
    configuration.write(card, "series_card")
    configuration.write(report, "risk_levels_report")
    chosen = decision_support[decision_support["ruta"] == TREATMENT_PREDICTABLE]
    print(f"[1.3] ladder: collapse order {order} · annullable extra {annullable!r} · "
          f"{(chosen['alcanzo_suelo'] == 1).mean():.0%} of trainable series reached the floor · "
          f"median rung {chosen['peldano'].median():.0f}")
    print_risk_levels(report)
    explain(configuration,
            "How to read: the error is the typical sampling error of one month for the series of that level (se_prediccion, weighted by money):",
            "even with a perfect estimate of the rate, next month's result moves that much by chance. It is a property of the size of the series, not of the method.",
            f"Levels A/A2 predict alone (≥ {configuration.own_rate_floor:.0f} contracts/month, ±5 pp or better); A3 blends its own rate with its pool; B/C borrow; S/M/D/N have no usable rate of their own.")
    print_level_definitions()
    print(f"[1.3] collapse route of the mandatory dims (first to collapse → last): {' → '.join(order)}")
    return series_estimates, card, decision_support, parent_ladder
