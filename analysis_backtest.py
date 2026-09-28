"""analysis_backtest.py — SFF v3 · ANALYSIS · phase 3: the judge, one judge.

For every estimation id, a ROLLING-ORIGIN backtest (Tashman 2000): every month with
truth that has at least `backtest_min_history_months` before it is a target; for every
horizon h in 1..H, the origin is h months before the target and only history ≤ origin
is used. All eligible techniques predict; the error is stored WITH SIGN in pp and
NORMALIZED by the binomial error of the pool-month (so a pool of n=1000 and a pool of
n=35 are judged on the same scale).

From the same table:
  · `decision_technique`  the champion per estimation id: the technique with the lowest
                          mean |normalized error| that beats the challenger (T2_mean) by
                          the margin, with at least `backtest_min_predictions` predictions;
                          ties inside the margin go to the richer family (time series
                          preferred). Otherwise the challenger, with `tecnica_origen`.
  · `decision_error_bands` the asymmetric band per (estimation id, h): the p5 and p95
                          of the signed normalized error (family quantiles — same
                          technique, every pool — when the id has too few predictions),
                          made monotone in h. Multiplied by the pool's own binomial
                          error at prediction time, the band respects each pool's size.
  · `backtest_holdout`    the report the business asked for: the months ≥
                          `backtest_test_start` (the 2026 months already happened),
                          predicted from origins with all the history before them.
  · calibration           % of hold-out realizations inside the band (target ≈ 90 %).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from binomial_reference import binomial_se_pp, inverse_logit, logit, weighted_quantile
from config import Config, explain
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from techniques import CATALOGUE, eligible_techniques, family_rank, memory_rank, month_numbers_of, technique_dimension

# ─── named constants ─────────────────────────────────────────────────────────────
# (intermittent_zero_share is a Config parameter: see config.py, phase 3)


def monthly_series_by_estimation_id(units: pd.DataFrame, decision_support: pd.DataFrame,
                                    parent_ladder: pd.DataFrame, configuration: Config) -> dict:
    """The monthly (rate, support) series of every estimation id.

    INPUT:   units with tasa · decision_support (the chosen ids) · parent_ladder (every
             fs_id × rung pattern: the MEMBERSHIP of each pattern) · configuration.
    OUTPUT:  dict id_estimacion → DataFrame(index=Period, columns ren, pipe, rate), built by
             SUMMING, month by month, EVERY series that matches the pattern — big siblings
             included, whether or not they chose that relative themselves. The pattern
             decides who computes the number; the ladder decides who receives it.
    RULES:   closed months with a defined rate. `technique_history_months` cuts what the
             techniques learn from (the last N months); the pools keep every month.
    """
    history = units[(units["universo"] == "normal") & units["tasa"].notna()]
    chosen_ids = set(decision_support["id_estimacion"])
    membership = parent_ladder[parent_ladder["padre_id"].isin(chosen_ids)][["fs_id", "padre_id"]].drop_duplicates()
    membership = membership.rename(columns={"padre_id": "id_estimacion"})
    own_ids = decision_support[~decision_support["id_estimacion"].isin(membership["id_estimacion"])][["fs_id", "id_estimacion"]]
    membership = pd.concat([membership, own_ids], ignore_index=True).drop_duplicates()
    joined = history.merge(membership, on="fs_id")
    window = configuration.technique_history_months
    series = {}
    for estimation_id, rows in joined.groupby("id_estimacion"):
        monthly = rows.groupby(configuration.period_col).agg(
            ren=(configuration.renewed_units_col, "sum"), pipe=(configuration.pipeline_units_col, "sum")).sort_index()
        monthly["rate"] = np.where(monthly["pipe"] > 0, monthly["ren"] / monthly["pipe"].replace(0, np.nan), np.nan)
        if window is not None:
            monthly = monthly.tail(int(window))
        series[estimation_id] = monthly
    return series


def build_pool_reference(monthly_series: dict, decision_estacionalidad: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per estimation id with what the backtest and the bands need — nothing more:
    meses, n_pool (median monthly support), tasa_pool, gate ('soporte' below the floor,
    'nivel' otherwise), estacional (1 only if the benchmark declared THIS series seasonal;
    a pool is a series only when it predicts alone), tendencia (the benchmark's verdict).
    Persisted as `pool_reference`. No per-pool diagnostics: the calendar decision is the
    benchmark's, taken once for the portfolio."""
    seasonal = set(decision_estacionalidad.loc[decision_estacionalidad["veredicto_estacional"] == 1, "fs_id"]) if len(decision_estacionalidad) else set()
    trend = dict(zip(decision_estacionalidad["fs_id"], decision_estacionalidad["veredicto_tendencia"])) if len(decision_estacionalidad) else {}
    rows = []
    for estimation_id, monthly in monthly_series.items():
        valid = monthly[monthly["rate"].notna() & (monthly["pipe"] > 0)]
        support = float(valid["pipe"].median()) if len(valid) else 0.0
        pooled = float(valid["ren"].sum() / max(valid["pipe"].sum(), 1)) if len(valid) else np.nan
        rows.append(dict(id_estimacion=estimation_id, meses=int(len(valid)), n_pool=round(support, 1),
                         tasa_pool=round(pooled, 4) if np.isfinite(pooled) else np.nan,
                         gate="soporte" if support < configuration.support_floor else "nivel",
                         estacional=int(estimation_id in seasonal), tendencia=int(trend.get(estimation_id, 0))))
    reference = pd.DataFrame(rows, columns=["id_estimacion", "meses", "n_pool", "tasa_pool", "gate", "estacional", "tendencia"])
    configuration.write(reference, "pool_reference")
    print(f"[2] pool reference: {len(reference)} estimation ids · {(reference['gate'] == 'nivel').sum()} with support · "
          f"{int(reference['estacional'].sum())} with month effects (the benchmark's seasonal series)")
    return reference


def dynamics_labels(pool_reference: pd.DataFrame) -> dict:
    """id_estimacion → dict(estacional, tendencia, intermitente): the labels the techniques'
    eligibility reads, taken from the pool reference (which carries the benchmark's verdict)."""
    labels = {}
    for _, row in pool_reference.iterrows():
        labels[row["id_estimacion"]] = dict(estacional=int(row["estacional"]), tendencia=int(row["tendencia"]), intermitente=0)
    return labels


def backtest_parameters(configuration: Config) -> dict:
    """The few numbers the backtest loop needs, picklable for the worker processes."""
    return dict(min_history=configuration.backtest_min_history_months, max_targets=configuration.backtest_max_targets,
                intermittent_zero_share=configuration.intermittent_zero_share, holdout_start=None)


def _backtest_chunk(arguments: tuple) -> list:
    """Worker: the raw prediction rows of a chunk of estimation ids (module-level so it pickles)."""
    chunk, labels, gate_by_id, horizons, parameters, techniques_by_id = arguments
    rows = []
    for estimation_id, monthly in chunk:
        if gate_by_id.get(estimation_id) == "soporte":
            continue
        valid = monthly[monthly["rate"].notna() & (monthly["pipe"] > 0)]
        if len(valid) <= parameters["min_history"]:
            continue
        rates, months = valid["rate"].to_numpy(dtype=float), pd.PeriodIndex(valid.index)
        supports = valid["pipe"].to_numpy(dtype=float)
        logit_rates, month_numbers = logit(rates), month_numbers_of(months)
        dynamics = dict(labels.get(estimation_id, {}))
        dynamics["intermitente"] = int(np.mean(rates < 0.02) >= parameters["intermittent_zero_share"])
        # the most recent `max_targets` DECISION months (before the hold-out) plus every
        # hold-out month: the exam never eats the evidence
        holdout_start = parameters.get("holdout_start")
        if holdout_start:
            decision_positions = [i for i, m in enumerate(months) if str(m) < holdout_start]
            first_target = max(parameters["min_history"], (decision_positions[-parameters["max_targets"]] if len(decision_positions) >= parameters["max_targets"] else (decision_positions[0] if decision_positions else len(valid))))
        else:
            first_target = max(parameters["min_history"], len(valid) - parameters["max_targets"])
        eligible_by_length = {}
        for target_position in range(first_target, len(valid)):
            real, n_real = rates[target_position], supports[target_position]
            se_pp = binomial_se_pp(real, n_real)
            target_month = str(months[target_position])
            for h in horizons:
                origin_position = target_position - h
                if origin_position < parameters["min_history"] - 1:
                    continue
                history_logit, history_months = logit_rates[:origin_position + 1], month_numbers[:origin_position + 1]
                origin_month = str(months[origin_position])
                length = len(history_logit)
                if length not in eligible_by_length:
                    eligible = eligible_techniques(length, dynamics)
                    if techniques_by_id is not None:
                        eligible = [t for t in eligible if t in techniques_by_id.get(estimation_id, ())]
                    eligible_by_length[length] = eligible
                for technique_id in eligible_by_length[length]:
                    try:
                        value = CATALOGUE[technique_id][4](history_logit, history_months, h, dynamics)
                    except Exception:
                        continue
                    if not np.isfinite(value):
                        continue
                    predicted = float(inverse_logit(value))
                    error_pp = 100 * (predicted - real)
                    rows.append((estimation_id, target_month, origin_month, h, technique_id, round(predicted, 4),
                                 round(real, 4), n_real, round(error_pp, 3), round(se_pp, 3), round(error_pp / se_pp, 4)))
    return rows


def rolling_origin_backtest(monthly_series: dict, pool_reference: pd.DataFrame, horizons: list,
                            configuration: Config, techniques_by_id: dict = None, holdout_start: str = None) -> pd.DataFrame:
    """The long table of predictions: one row per (id, target, h, technique).

    OUTPUT:  backtest_long: id_estimacion, mes_objetivo, origen, h, tecnica_id, tasa_pred,
             tasa_real, n_real, err_pp (signed: pred − real), se_binom_pp (of the target
             month with its own n), err_norm (err_pp / se_binom_pp).
    RULES:   history ≤ origin only; the target must have a defined rate and n > 0.
             Only estimation ids WITH SUPPORT are judged (gate ≠ 'soporte'): a series under
             the floor gets the challenger and a binomial band, and
             judging it would cost the same as judging a real pool for nothing. Targets =
             the most recent `backtest_max_targets` months; the history before each
             origin is the whole history.
             `techniques_by_id` (id → list) restricts the techniques judged per id (the
             second stage: champion + challenger only); None = every eligible technique.
    COST:    ≈ targets × horizons × techniques per id. The logit is computed once per id.
    """
    labels = dynamics_labels(pool_reference)
    gate_by_id = dict(zip(pool_reference["id_estimacion"], pool_reference["gate"]))
    items = list(monthly_series.items())
    parameters = backtest_parameters(configuration)
    parameters["holdout_start"] = holdout_start
    workers = max(1, int(configuration.backtest_workers))
    if workers == 1 or len(items) < 2 * workers:
        rows = _backtest_chunk((items, labels, gate_by_id, horizons, parameters, techniques_by_id))
    else:
        chunks = [items[index::workers] for index in range(workers)]
        arguments = [(chunk, labels, gate_by_id, horizons, parameters, techniques_by_id) for chunk in chunks]
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = [row for chunk_rows in pool.map(_backtest_chunk, arguments) for row in chunk_rows]
        except Exception as error:                       # a spawn problem must not kill the analysis
            print(f"[3] WARNING parallel backtest failed ({type(error).__name__}: {error}); running sequentially")
            rows = _backtest_chunk((items, labels, gate_by_id, horizons, parameters, techniques_by_id))
    return pd.DataFrame(rows, columns=["id_estimacion", "mes_objetivo", "origen", "h", "tecnica_id", "tasa_pred",
                                       "tasa_real", "n_real", "err_pp", "se_binom_pp", "err_norm"])


def band_of_horizon(h: int, bands: dict) -> str:
    """The horizon band a horizon belongs to; beyond the last band's top, the last band."""
    for name, (low, high) in bands.items():
        if low <= h <= high:
            return name
    return list(bands)[-1]


def select_technique(backtest_long: pd.DataFrame, configuration: Config, history_months: dict = None) -> pd.DataFrame:
    """One champion per estimation id AND horizon band, with its evidence.

    OUTPUT:  decision_technique: id_estimacion, tramo_h, h_min, h_max, tecnica,
             tecnica_origen (campeon/retador), err_norm_medio, err_pp_medio,
             n_predicciones, retador_err_norm.
    RULES:   per band, score = mean |err_norm| over the screen horizons inside the band
             and every origin. The margin is the band's (`challenger_margin_by_band`):
             0.10 near, 0 far. Champion = lowest score with n_predicciones ≥ minimum AND
             score ≤ challenger score − margin; among candidates within the margin of the
             best, the technique with the longest MEMORY wins (then the richest family)
             when the id has ≥ richer_family_min_history_months of history; else the
             simplest. Far from now, tying the challenger with more memory is enough.
             A band with no screen horizon inside it takes the previous band's champion.
    """
    rows = []
    challenger = configuration.challenger_technique
    bands = configuration.backtest_horizon_bands
    scored = backtest_long.assign(tramo_h=backtest_long["h"].map(lambda h: band_of_horizon(int(h), bands)))
    for (estimation_id, band_name), block in scored.groupby(["id_estimacion", "tramo_h"]):
        absolute = block.assign(abs_norm=block["err_norm"].abs(), abs_pp=block["err_pp"].abs())
        scores = absolute.groupby("tecnica_id").agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"),
                                                    n_predicciones=("err_norm", "size"))
        challenger_score = float(scores.loc[challenger, "err_norm_medio"]) if challenger in scores.index else np.inf
        margin = float((configuration.challenger_margin_by_band or {}).get(band_name, configuration.challenger_margin_normalized))
        candidates = scores[scores["n_predicciones"] >= configuration.backtest_min_predictions]
        best_score = candidates["err_norm_medio"].min() if len(candidates) else np.inf
        within_margin = candidates[candidates["err_norm_medio"] <= best_score + max(margin, 1e-9)]
        chosen, origin = challenger, CHALLENGER_ORIGIN
        # a champion must be at least `margin` better than the challenger; with margin 0
        # (far bands) tying the challenger is enough — memory then decides
        if len(within_margin) and best_score < challenger_score - margin + 1e-12 and best_score <= challenger_score:
            enough_history = (history_months or {}).get(estimation_id, 0) >= configuration.richer_family_min_history_months
            rank_sign = 1 if enough_history else -1
            chosen = max(within_margin.index, key=lambda t: (rank_sign * memory_rank(t), rank_sign * family_rank(t), -within_margin.loc[t, "err_norm_medio"]))
            origin = CHAMPION_ORIGIN if chosen != challenger else CHALLENGER_ORIGIN
        row = scores.loc[chosen] if chosen in scores.index else pd.Series(dict(err_norm_medio=np.nan, err_pp_medio=np.nan, n_predicciones=0))
        rows.append(dict(id_estimacion=estimation_id, tramo_h=band_name, h_min=bands[band_name][0], h_max=bands[band_name][1],
                         tecnica=chosen, tecnica_origen=origin,
                         err_norm_medio=round(float(row["err_norm_medio"]), 4), err_pp_medio=round(float(row["err_pp_medio"]), 3),
                         n_predicciones=int(row["n_predicciones"]), retador_err_norm=round(challenger_score, 4)))
    decision = pd.DataFrame(rows, columns=["id_estimacion", "tramo_h", "h_min", "h_max", "tecnica", "tecnica_origen", "err_norm_medio",
                                           "err_pp_medio", "n_predicciones", "retador_err_norm"])
    # a band without screen evidence inherits the previous band's champion
    filled = []
    for estimation_id, block in decision.groupby("id_estimacion"):
        previous = None
        for band_name in bands:
            hit = block[block["tramo_h"] == band_name]
            if len(hit):
                previous = hit.iloc[0].to_dict()
                filled.append(previous)
            elif previous is not None:
                inherited = dict(previous, tramo_h=band_name, h_min=bands[band_name][0], h_max=bands[band_name][1],
                                 tecnica_origen=previous["tecnica_origen"] + "_heredado", n_predicciones=0)
                filled.append(inherited)
    return pd.DataFrame(filled, columns=decision.columns)


def technique_for(decision_technique: pd.DataFrame) -> dict:
    """(id_estimacion, h) → technique, from the band table (h_min..h_max per row)."""
    lookup = {}
    for _, row in decision_technique.iterrows():
        for h in range(int(row["h_min"]), int(min(row["h_max"], 60)) + 1):
            lookup[(row["id_estimacion"], h)] = row["tecnica"]
    # beyond the last judged band, the last band's technique (the horizon cap)
    last = decision_technique.sort_values("h_max").groupby("id_estimacion").tail(1)
    for _, row in last.iterrows():
        for h in range(int(row["h_max"]) + 1, 61):
            lookup[(row["id_estimacion"], h)] = row["tecnica"]
    return lookup


def error_bands(backtest_long: pd.DataFrame, decision_technique: pd.DataFrame, horizons: list,
                configuration: Config) -> pd.DataFrame:
    """The asymmetric band per (estimation id, h) of its chosen technique.

    OUTPUT:  decision_error_bands: id_estimacion, tecnica, h, q_low_norm, q_high_norm,
             n_predicciones, banda_origen ("propia" / "familia" / "binomial"), and the
             widths made MONOTONE in h (a band never narrows as the horizon grows).
    RULES:   own quantiles when the id has ≥ band_min_predictions at that h; otherwise
             the family's (same technique, every id, that h). Quantiles are of err_norm
             (signed), so a trending series gets a shifted band by itself.
    NOTE:    the long table is grouped ONCE by (id, technique, h) and once by (technique,
             h); filtering it per id would scan millions of rows thousands of times.
    """
    own_groups = {key: group.to_numpy() for key, group in backtest_long.groupby(["id_estimacion", "tecnica_id", "h"])["err_norm"]}
    family_groups = {key: group.to_numpy() for key, group in backtest_long.groupby(["tecnica_id", "h"])["err_norm"]}
    technique_at = technique_for(decision_technique)
    rows = []
    for estimation_id in decision_technique["id_estimacion"].unique():
        low_so_far, high_so_far = 0.0, 0.0
        for h in horizons:
            technique = technique_at.get((estimation_id, h), configuration.challenger_technique)
            at_h = own_groups.get((estimation_id, technique, h), np.array([]))
            if len(at_h) >= configuration.band_min_predictions:
                origin, source = "propia", at_h
            else:
                origin, source = "familia", family_groups.get((technique, h), np.array([]))
            low = weighted_quantile(source, configuration.band_low_quantile)
            high = weighted_quantile(source, configuration.band_high_quantile)
            if not np.isfinite(low) or not np.isfinite(high):
                low, high, origin = -configuration.z, configuration.z, "binomial"
            low_so_far, high_so_far = min(low_so_far, low), max(high_so_far, high)
            rows.append(dict(id_estimacion=estimation_id, tecnica=technique, h=h,
                             q_low_norm=round(low_so_far, 4), q_high_norm=round(high_so_far, 4),
                             n_predicciones=int(len(source)), banda_origen=origin))
    return pd.DataFrame(rows)


def holdout_report(backtest_long: pd.DataFrame, decision_technique: pd.DataFrame, decision_error_bands: pd.DataFrame,
                   configuration: Config, holdout_start: str = None) -> pd.DataFrame:
    """The hold-out: months ≥ backtest_test_start predicted with the chosen technique.

    OUTPUT:  backtest_holdout: id_estimacion, mes_objetivo, h, tecnica, tasa_pred,
             tasa_real, err_pp, err_norm, banda_low_pp, banda_high_pp, dentro_banda.
    RULES:   the band of a prediction = q_norm × se_binom of the target month (the real n).
    """
    if backtest_long.empty:
        return pd.DataFrame()
    start = holdout_start or holdout_start_month(backtest_long, configuration)
    technique_at = technique_for(decision_technique)
    chosen = pd.DataFrame([dict(id_estimacion=i, h=h, tecnica_id=t) for (i, h), t in technique_at.items()])
    holdout = backtest_long[backtest_long["mes_objetivo"] >= start].merge(chosen, on=["id_estimacion", "h", "tecnica_id"])
    bands = decision_error_bands[["id_estimacion", "h", "q_low_norm", "q_high_norm"]]
    holdout = holdout.merge(bands, on=["id_estimacion", "h"], how="left")
    holdout["banda_low_pp"] = (holdout["q_low_norm"] * holdout["se_binom_pp"]).round(2)
    holdout["banda_high_pp"] = (holdout["q_high_norm"] * holdout["se_binom_pp"]).round(2)
    holdout["dentro_banda"] = ((holdout["err_pp"] >= holdout["banda_low_pp"]) & (holdout["err_pp"] <= holdout["banda_high_pp"])).astype(int)
    return holdout.rename(columns={"tecnica_id": "tecnica"})[["id_estimacion", "mes_objetivo", "origen", "h", "tecnica", "tasa_pred",
                                                             "tasa_real", "n_real", "err_pp", "err_norm", "banda_low_pp",
                                                             "banda_high_pp", "dentro_banda"]]


def print_candidates(decision_technique: pd.DataFrame, screen: pd.DataFrame, configuration: Config) -> None:
    """Console: a few pools worth looking at in detail, with the call that opens them.

    · biggest gain of the long-band champion over the challenger (a shape that pays far out)
    · pools whose short and long champions differ (the horizon matters there)
    · the biggest pools whose champion is the challenger in every band (nothing beats simple)
    """
    bands = list(configuration.backtest_horizon_bands)
    wide = decision_technique.pivot_table(index="id_estimacion", columns="tramo_h", values="tecnica", aggfunc="first")
    support = screen.groupby("id_estimacion")["n_real"].median()
    print("[3] candidates to look at in detail (open with sheet(\"<id_estimacion>\", configuration)):")
    long_band = bands[-1]
    far = decision_technique[(decision_technique["tramo_h"] == long_band) & (decision_technique["tecnica_origen"] == CHAMPION_ORIGIN)].copy()
    if len(far):
        far["gain"] = far["retador_err_norm"] - far["err_norm_medio"]
        far = far.assign(n=far["id_estimacion"].map(support)).sort_values(["gain", "n"], ascending=False).head(3)
        for _, row in far.iterrows():
            print(f"   shape pays far out    {row['id_estimacion'][:60]:<60} {row['tecnica']:<20} gains {row['gain']:.2f} binomial units over the challenger at {long_band} · n≈{row['n']:.0f}")
    if len(bands) > 1 and bands[0] in wide.columns and long_band in wide.columns:
        differ = wide[wide[bands[0]] != wide[long_band]].copy()
        differ["n"] = differ.index.map(support)
        for estimation_id, row in differ.sort_values("n", ascending=False).head(3).iterrows():
            print(f"   horizon matters       {estimation_id[:60]:<60} {row[bands[0]]:<20} near, {row[long_band]} far · n≈{row['n']:.0f}")
    simple = decision_technique.groupby("id_estimacion")["tecnica_origen"].apply(lambda s: (s == CHALLENGER_ORIGIN).all())
    simple_ids = simple[simple].index
    biggest = support.reindex(simple_ids).sort_values(ascending=False).head(2)
    for estimation_id, n in biggest.items():
        print(f"   nothing beats simple  {estimation_id[:60]:<60} {configuration.challenger_technique:<20} in every band · n≈{n:.0f}")


def holdout_aggregate(holdout: pd.DataFrame) -> pd.DataFrame:
    """The hold-out of the TOTAL: per target month and horizon, Σ n·predicted vs Σ n·real
    over every pool (units-weighted, ≈ money). This is the empirical error of the
    aggregate — the number the per-pool band cannot give, because pools that miss in the
    same direction do not cancel. OUTPUT: mes_objetivo, h, n, tasa_pred_agg, tasa_real_agg,
    err_agg_pp (signed), and per h the mean |err_agg_pp| and the bias."""
    if holdout is None or holdout.empty:
        return pd.DataFrame()
    weighted = holdout.assign(pred_n=holdout["tasa_pred"] * holdout["n_real"], real_n=holdout["tasa_real"] * holdout["n_real"])
    monthly = weighted.groupby(["mes_objetivo", "h"], as_index=False).agg(n=("n_real", "sum"), pred_n=("pred_n", "sum"), real_n=("real_n", "sum"),
                                                                          pools=("id_estimacion", "nunique"))
    monthly["tasa_pred_agg"] = (monthly["pred_n"] / monthly["n"]).round(4)
    monthly["tasa_real_agg"] = (monthly["real_n"] / monthly["n"]).round(4)
    monthly["err_agg_pp"] = (100 * (monthly["tasa_pred_agg"] - monthly["tasa_real_agg"])).round(3)
    return monthly.drop(columns=["pred_n", "real_n"])


def holdout_start_month(backtest_long: pd.DataFrame, configuration: Config, first_test_month: str = None) -> str:
    """The first hold-out month: `backtest_test_start` if set; else the first month with
    role ROLE_TEST in the extract; else the last `holdout_default_months` target months."""
    if configuration.backtest_test_start:
        return str(configuration.backtest_test_start)
    if first_test_month:
        return str(first_test_month)
    if backtest_long.empty:
        return None
    months = sorted(backtest_long["mes_objetivo"].unique())
    return months[-configuration.holdout_default_months] if len(months) >= configuration.holdout_default_months else months[0]


def aggregate_error_bands(backtest_long: pd.DataFrame, decision_technique: pd.DataFrame, holdout_start: str,
                          configuration: Config) -> tuple:
    """The COMMON error: the error of the whole portfolio summed per target month and
    horizon, with the chosen technique of each pool, over the DECISION months (before
    the hold-out). Its quantiles per h are the band component that the per-pool
    quadrature cannot see: pools missing together, in the same direction.
    OUTPUT: (aggregate_error table: mes_objetivo, h, n, tasa_pred_agg, tasa_real_agg,
             err_agg_pp;  decision_aggregate_bands: h, q_low_pp, q_high_pp, sesgo_pp, meses)."""
    technique_at = technique_for(decision_technique)
    chosen = pd.DataFrame([dict(id_estimacion=i, h=h, tecnica_id=t) for (i, h), t in technique_at.items()])
    rows = backtest_long[backtest_long["mes_objetivo"] < holdout_start].merge(chosen, on=["id_estimacion", "h", "tecnica_id"])
    aggregate = holdout_aggregate(rows)
    bands = []
    if len(aggregate):
        for h, block in aggregate.groupby("h"):
            errors = block["err_agg_pp"].to_numpy()
            bands.append(dict(h=int(h), q_low_pp=round(float(np.quantile(errors, configuration.band_low_quantile)), 3),
                              q_high_pp=round(float(np.quantile(errors, configuration.band_high_quantile)), 3),
                              sesgo_pp=round(float(errors.mean()), 3), meses=int(len(errors))))
    return aggregate, pd.DataFrame(bands, columns=["h", "q_low_pp", "q_high_pp", "sesgo_pp", "meses"])


def run_backtest_analysis(monthly_series: dict, pool_reference: pd.DataFrame, configuration: Config,
                          horizons: list, first_test_month: str = None) -> dict:
    """Phase 3 end to end, in two stages. Persists dim_tecnica, backtest_predictions,
    decision_technique, decision_error_bands, backtest_holdout.

    STAGE 1 · screen: every eligible technique at `backtest_screen_horizons` → the champion
             per id (select_technique on the screen table).
    STAGE 2 · judge: champion + challenger at EVERY judged horizon → bands, hold-out. The
             persisted `backtest_predictions` is the union of both stages.
    Two stages cost ~3× less than judging every technique at every horizon, and the
    champion is chosen where it matters most (the near horizons).
    """
    configuration.write(technique_dimension(), "dim_tecnica")
    screen_horizons = sorted({h for h in configuration.backtest_screen_horizons if h in horizons} | {horizons[0]})
    # the hold-out cut is known before predicting: `backtest_test_start`, else the first
    # ROLE_TEST month of the extract; the target window is counted before it
    provisional = holdout_start_month(pd.DataFrame({"mes_objetivo": []}), configuration, first_test_month) if (configuration.backtest_test_start or first_test_month) else None
    screen = rolling_origin_backtest(monthly_series, pool_reference, screen_horizons, configuration, holdout_start=provisional)
    holdout_start = holdout_start_month(screen, configuration, first_test_month) if len(screen) else None
    history_months = dict(zip(pool_reference["id_estimacion"], pool_reference["meses"]))
    # DECIDE on the months before the hold-out only: the champion and its bands never see
    # the months they are later judged on (otherwise the hold-out calibration is circular)
    decision_rows = screen[screen["mes_objetivo"] < holdout_start] if holdout_start else screen
    decision_technique = select_technique(decision_rows, configuration, history_months)
    remaining = [h for h in horizons if h not in screen_horizons]
    chosen_and_challenger = {}
    for _, row in decision_technique.iterrows():
        chosen_and_challenger.setdefault(row["id_estimacion"], {configuration.challenger_technique}).add(row["tecnica"])
    judged = rolling_origin_backtest(monthly_series, pool_reference, remaining, configuration, chosen_and_challenger, holdout_start) if remaining else screen.head(0)
    backtest_long = pd.concat([screen, judged], ignore_index=True).sort_values(["id_estimacion", "mes_objetivo", "h", "tecnica_id"])
    decision_long = backtest_long[backtest_long["mes_objetivo"] < holdout_start] if holdout_start else backtest_long
    bands = error_bands(decision_long, decision_technique, horizons, configuration)
    holdout = holdout_report(backtest_long, decision_technique, bands, configuration, holdout_start)
    aggregate_error, aggregate_bands = aggregate_error_bands(backtest_long, decision_technique, holdout_start, configuration)
    configuration.write(aggregate_error, "backtest_aggregate_error")
    configuration.write(aggregate_bands, "decision_aggregate_bands")
    if configuration.backtest_persist == "all":
        configuration.write(backtest_long, "backtest_predictions")
    elif configuration.backtest_persist == "chosen":
        keep = {(row["id_estimacion"], row["tecnica"]) for _, row in decision_technique.iterrows()}
        keep |= {(estimation_id, configuration.challenger_technique) for estimation_id in decision_technique["id_estimacion"].unique()}
        chosen_rows = backtest_long[[(i, t) in keep for i, t in zip(backtest_long["id_estimacion"], backtest_long["tecnica_id"])]]
        configuration.write(chosen_rows, "backtest_predictions")
        print(f"[3] backtest_pred persisted for champions + challenger only: {len(chosen_rows):,} of {len(backtest_long):,} rows "
              f"(backtest_persist='chosen'; the full table stays in results['backtest']['backtest_long'])")
    configuration.write(decision_technique, "decision_technique")
    configuration.write(bands, "decision_error_bands")
    configuration.write(holdout, "backtest_holdout")
    aggregate = holdout_aggregate(holdout)
    configuration.write(aggregate, "backtest_holdout_aggregate")
    if len(screen):
        bands_map = configuration.backtest_horizon_bands
        scored = screen.assign(tramo_h=screen["h"].map(lambda h: band_of_horizon(int(h), bands_map)), abs_norm=screen["err_norm"].abs())
        print(f"[3] backtest: {len(backtest_long):,} predictions · {backtest_long['id_estimacion'].nunique()} estimation ids with support · "
              f"{backtest_long['mes_objetivo'].nunique()} target months · screened at h={screen_horizons}, judged at h={horizons}")
        print("[3] leaderboard by horizon band (mean |error| in binomial units; 1.0 = one sampling error):")
        table = scored.groupby(["tecnica_id", "tramo_h"])["abs_norm"].mean().unstack("tramo_h").reindex(columns=list(bands_map))
        table = table.sort_values(list(bands_map)[0])
        print("   " + f"{'technique':<22}" + "".join(f"{b:>14}" for b in table.columns))
        for technique_id, row in table.iterrows():
            print("   " + f"{technique_id:<22}" + "".join(f"{v:14.2f}" if pd.notna(v) else f"{'-':>14}" for v in row))
        for band_name, block in decision_technique.groupby("tramo_h", sort=False):
            print(f"[3] champions {band_name:<6} {block['tecnica'].value_counts().head(6).to_dict()} · "
                  f"{(block['tecnica_origen'] == CHAMPION_ORIGIN).mean():.0%} beat the challenger")
        explain(configuration,
                "A binomial unit = the sampling error of the month being predicted (√(p(1−p)/n) of that pool that month). 1.0 is the floor no method can beat;",
                "2.0 means the technique misses by twice what chance alone would. Scores are comparable across pools of 50 and of 5,000 contracts because of this scaling.",
                f"'corto' = predicting next month with data up to the previous one; 'medio_largo' = predicting with data up to six months before. The challenger ({configuration.challenger_technique}) is 'the last quarter';",
                "a champion must beat it by the band's margin (0.10 units near; a tie is enough far away, where more memory wins). Decisions use only the months BEFORE the exam.")
        print_candidates(decision_technique, screen, configuration)
    if len(holdout):
        weighted = holdout.assign(abs_pp=holdout["err_pp"].abs(), w=holdout["n_real"])
        by_h = weighted.groupby("h").apply(lambda g: pd.Series(dict(
            err_pp=g["abs_pp"].mean(), err_pp_w=np.average(g["abs_pp"], weights=g["w"]), bias_w=np.average(g["err_pp"], weights=g["w"]),
            dentro=g["dentro_banda"].mean(), n=len(g))), include_groups=False)
        print(f"[3] HOLD-OUT (months ≥ {holdout['mes_objetivo'].min()}): chosen technique per id · "
              f"|error| plain and weighted by units (≈ money) · signed bias (− = under-forecast)")
        for h, row in by_h.iterrows():
            print(f"   h={int(h):>2}: |error| {row['err_pp']:.2f} pp · weighted {row['err_pp_w']:.2f} pp · bias {row['bias_w']:+.2f} pp · "
                  f"{row['dentro']:.0%} inside band · {int(row['n'])} predictions")
        if len(aggregate):
            by_h_agg = aggregate.groupby("h").agg(err=("err_agg_pp", lambda e: float(e.abs().mean())), bias=("err_agg_pp", "mean"),
                                                  worst=("err_agg_pp", lambda e: float(e.abs().max())), meses=("mes_objetivo", "nunique"))
            print(f"[3] HOLD-OUT OF THE TOTAL (months ≥ {holdout_start}, all pools summed, units-weighted): the error of the aggregate month")
            for h, row in by_h_agg.iterrows():
                print(f"   h={int(h):>2}: |error of the total| {row['err']:.2f} pp · bias {row['bias']:+.2f} pp · worst month {row['worst']:.2f} pp · {int(row['meses'])} months")
            explain(configuration,
                    "The hold-out is the exam: months never used to choose techniques or to measure bands, predicted as if unknown and compared with what happened.",
                    "'inside band' should be close to 90 % if the band is honest (a 90 % band): far above = too wide, far below = too narrow.",
                    "'bias' with a sign = we systematically over- (+) or under-forecast (−); a bias that grows with h says the level moved and history lags behind.",
                    "The TOTAL's error is what the business will see: pools that miss together do not cancel, which is why it is measured separately from the per-pool error.")
        if len(aggregate_bands):
            print(f"[3] COMMON ERROR BAND (aggregate error over the decision months < {holdout_start}, p5/p95 per h): what pools missing "
                  f"TOGETHER costs; added to the per-pool quadrature in the total's band")
            for _, row in aggregate_bands.iterrows():
                print(f"   h={int(row['h']):>2}: {row['q_low_pp']:+.2f} / {row['q_high_pp']:+.2f} pp · bias {row['sesgo_pp']:+.2f} pp · {int(row['meses'])} months")
    return dict(backtest_long=backtest_long, decision_technique=decision_technique, decision_error_bands=bands,
                backtest_holdout=holdout, backtest_holdout_aggregate=aggregate, backtest_aggregate_error=aggregate_error,
                decision_aggregate_bands=aggregate_bands, holdout_start=holdout_start)
