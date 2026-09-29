"""
step_14_backtest.py — The backtest of the renewal rate: which technique predicts each pool
best, how precise it is, and how wide its error band must be.

HOW THE TEST WORKS (it is also written in the log, with the months of the run):
  · WHAT IS JUDGED: the monthly series of every estimation id with support (gate 'nivel',
    step 12). An id below the floor is not judged: it takes the challenger.
  · TWO KINDS OF TARGET MONTHS, never mixed:
      selection  the backtest_selection_months (6) closed months BEFORE the exam: the
                 techniques are compared on them and the best one is CHOSEN
      exam       the exam months of the calendar (test_months, 3): used only to MEASURE
                 the chosen technique, which never saw them while being chosen
  · NO PEEKING: for a target month T and a horizon h, every technique sees only the
    months up to T − h, exactly as it would have been in real life h months earlier
    (h = 1: next month; h = 6: six months ahead). It needs at least
    backtest_min_history_months of history.
  · THE ERROR: err_pp = 100 · (predicted − real rate). It is compared with the error chance
    alone would cause in that month, the binomial error of the real rate with the month's
    own units: err_norm = err_pp / se_binom_pp. |err_norm| ≈ 1 is as good as any forecast
    can be; 3 means three times the noise.
  · THE CHOICE, per id and horizon band (corto = h 1, medio_largo = h 2-6): the technique
    with the lowest mean |err_norm| in the selection months REPLACES the challenger
    (T3_ma3, the moving average of 3 months) only if it beats it by the band's margin
    (0.10 near, 0 far). A technique competes where its minimum history is reached (the
    seasonal ones need 13 and 24 months): only the history limits, no seasonality verdict.
    The ranking of the techniques is computed on the targets where ALL of them competed.
  · THE BAND, per technique and horizon: the 5 % and 95 % quantiles of err_norm in the
    selection months, over every judged id. In the exam, 90 % of the errors of the chosen
    technique should fall inside it.
  · THE PRECISION, in the exam months: per id (chosen vs challenger) and for the TOTAL:
    Σ predicted renewals vs Σ real renewals of all judged ids, month by month.

OUTPUT TABLES (long format: one row per prediction, not one table or column per technique;
each id competes with the techniques its history allows, and Power BI pivots it):
  sff_dim_tecnica            the catalogue
  sff_backtest_predicciones  id × target month × horizon × technique: predicted, real, errors
  sff_decision_tecnica       id × horizon band: the chosen technique, its error, the challenger's
  sff_decision_bandas        technique × horizon: the error quantiles
  sff_backtest_examen        id × horizon band: the chosen vs the challenger in the exam
  sff_backtest_examen_total  exam month × horizon: the error of the total renewals

Actions (logged as they are done):
  1. the calendar of the test: selection and exam months, horizons, and how it works
  2. the predictions of every technique for every judged id, target and horizon
  3. the choice of every id and horizon band
  4. the error bands of every technique and horizon
  5. the precision in the exam: per id and for the total
  6. check the test                                                 checks 1-5
  7. write the six tables                                           checks 6-11
  8. count the checks; stop if any failed
  9. show the ranking of the techniques, the choices and the precision in the exam

Checks (logged as they are made, numbered, at the level of their status):
   1. no prediction saw its target month or anything after its origin
   2. every estimation id has a decision in every horizon band
   3. the challenger was predicted wherever another technique was
   4. the ranking compares the techniques on common targets (every technique competed)
   5. in the exam, the band holds at least 80 % of the chosen technique's errors (nominal 90 %)
                                                                     (warning only)
   6-11. the six tables written and read back

Output: (the predictions, the decision per id and band, the error bands, the exam per id,
the exam of the total) · the six tables above.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from techniques import CATALOGUE, eligible_techniques, inverse_logit, logit, predict_logit, technique_table
from vocabulario import (CHALLENGER_ORIGIN, CHAMPION_ORIGIN, ESTIMATION_ID_COLUMN, GATE_LEVEL, PURPOSE_EXAM,
                         PURPOSE_SELECTION, TABLE_BACKTEST_PREDICTIONS, TABLE_ERROR_BANDS, TABLE_EXAM_BY_POOL,
                         TABLE_EXAM_TOTAL, TABLE_TECHNIQUE_DECISION, TABLE_TECHNIQUES)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "14"
STEP_NAME = "BACKTEST OF THE RATE"
STEP_PURPOSE = ("choose, for every pool with support and every horizon, the technique that predicts its renewal rate "
                "best in the months before the exam, then measure it in the exam months it never saw: its precision "
                "per pool and for the total, and the width its error band must have")
STEP_ACTIONS = ["the calendar of the test: selection and exam months, horizons, and how it works",
                "the predictions of every technique for every judged id, target and horizon",
                "the choice of every id and horizon band",
                "the error bands of every technique and horizon",
                "the precision in the exam: per id and for the total",
                "check the test (checks 1-5)",
                "write the six tables (checks 6-11)",
                "count the checks; stop if any failed",
                "show the ranking of the techniques, the choices and the precision in the exam"]
STEP_OUTPUT = ("predictions (id × target × horizon × technique) · the chosen technique per id and band · error bands · "
               "the exam per id and for the total · six tables")

# ─── named constants ─────────────────────────────────────────────────────────────
PERCENTAGE_POINTS = 100
MIN_EXAM_COVERAGE = 0.80
SUPPORT_ORIGIN = "sin_soporte"      # an id below the floor: not judged, it takes the challenger
PREDICTION_COLUMNS = ["id_estimacion", "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
                      "tasa_pred", "tasa_real", "vencen_real", "err_pp", "se_binom_pp", "err_norm"]


def band_of_horizon(horizon: int, horizon_bands: dict) -> str:
    """The horizon band of a horizon; beyond the last band, the last one."""
    for band_name, (low, high) in horizon_bands.items():
        if low <= horizon <= high:
            return band_name
    return list(horizon_bands)[-1]


def run_backtest(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> dict:
    """The backtest of the rate: predictions, choice, bands and exam; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []

    # [1] the calendar of the test, and how it works
    selection_months, exam_months = test_calendar(pool_series, configuration)
    log_how_the_test_works(selection_months, exam_months, pool_reference, configuration)

    # [2] the predictions
    judged_ids = set(pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, ESTIMATION_ID_COLUMN])
    predictions = predict_every_target(pool_series, judged_ids, selection_months, exam_months, configuration)
    predictions["tramo_h"] = predictions["h"].map(lambda h: band_of_horizon(int(h), configuration.horizon_bands))
    configuration.log_action(STEP_LABEL, 2, f"{len(predictions):,} predictions: {predictions[ESTIMATION_ID_COLUMN].nunique():,} "
                                            f"ids × {predictions['mes_objetivo'].nunique()} target months × "
                                            f"{len(configuration.backtest_horizons)} horizons × up to {len(CATALOGUE)} techniques "
                                            f"(each where its history allows)")

    # [3] the choice
    decision = choose_techniques(predictions, pool_reference, configuration)
    judged_decision = decision[decision["tecnica_origen"] != SUPPORT_ORIGIN]
    configuration.log_action(STEP_LABEL, 3, f"choices: {judged_decision.groupby('tramo_h')['tecnica_origen'].value_counts().to_dict()}")

    # [4] the error bands of every technique and horizon (selection months)
    bands = error_bands(predictions, configuration)
    configuration.log_action(STEP_LABEL, 4, f"{len(bands)} bands (technique × horizon) from the selection months")

    # [5] the exam: per id and for the total
    exam_by_pool, exam_total, exam_rows = exam_precision(predictions, decision, bands, configuration)
    configuration.log_action(STEP_LABEL, 5, f"exam measured on {exam_by_pool[ESTIMATION_ID_COLUMN].nunique():,} ids and "
                                            f"{exam_total['mes_objetivo'].nunique()} months")

    # [6] the checks
    configuration.log_action(STEP_LABEL, 6, "checking the test")
    check_backtest(predictions, decision, pool_reference, exam_rows, configuration, check_log)

    # [7] the tables
    configuration.log_action(STEP_LABEL, 7, "writing the six tables")
    configuration.write_table(STEP_LABEL, check_log, technique_table(), TABLE_TECHNIQUES)
    configuration.write_table(STEP_LABEL, check_log, predictions, TABLE_BACKTEST_PREDICTIONS)
    configuration.write_table(STEP_LABEL, check_log, decision, TABLE_TECHNIQUE_DECISION)
    configuration.write_table(STEP_LABEL, check_log, bands, TABLE_ERROR_BANDS)
    configuration.write_table(STEP_LABEL, check_log, exam_by_pool, TABLE_EXAM_BY_POOL)
    configuration.write_table(STEP_LABEL, check_log, exam_total, TABLE_EXAM_TOTAL)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [9] what came out
    log_backtest_report(predictions, decision, exam_by_pool, exam_total, pool_reference, configuration)
    return dict(predictions=predictions, decision=decision, bands=bands, exam_by_pool=exam_by_pool, exam_total=exam_total)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CALENDAR AND ITS EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════════

def test_calendar(pool_series: pd.DataFrame, configuration: Config) -> tuple:
    """The selection months (the N closed months before the exam) and the exam months."""
    boundaries = configuration.calendar_boundaries()
    exam_months = pd.period_range(boundaries["test_start"], boundaries["pending_start"] - 1, freq="M")
    selection_months = pd.period_range(boundaries["test_start"] - configuration.backtest_selection_months,
                                       boundaries["test_start"] - 1, freq="M")
    return list(selection_months), list(exam_months)


def log_how_the_test_works(selection_months: list, exam_months: list, pool_reference: pd.DataFrame,
                           configuration: Config) -> None:
    """Action 1: the calendar of this run and the rules of the test, in the log."""
    judged = pool_reference["gate"] == GATE_LEVEL
    largest_horizon = max(configuration.backtest_horizons)
    example_target = selection_months[-1]
    configuration.log_action(STEP_LABEL, 1, "how the test works:")
    explanation = [
        f"WHAT: the monthly rate of the {int(judged.sum()):,} estimation ids with support (of {len(pool_reference):,}); "
        f"the other {int((~judged).sum()):,} are not judged and take the challenger {configuration.challenger_technique}",
        f"SELECTION months {selection_months[0]}..{selection_months[-1]} ({len(selection_months)}): every technique predicts "
        f"them and the best one is CHOSEN",
        f"EXAM months {exam_months[0]}..{exam_months[-1]} ({len(exam_months)}): only to MEASURE the chosen technique, "
        f"which never saw them while being chosen",
        f"NO PEEKING: for target month T and horizon h a technique sees only the months up to T − h. Example: "
        f"T = {example_target}, h = {largest_horizon} → it sees up to {example_target - largest_horizon}; "
        f"h = 1 → up to {example_target - 1}. It needs ≥ {configuration.backtest_min_history_months} months of history",
        "ERROR: err_pp = 100·(predicted − real rate); err_norm = err_pp / binomial error of the real rate with the "
        "month's own units. |err_norm| ≈ 1 = as close as chance allows; 3 = three times the noise",
        f"CHOICE per id and band {list(configuration.horizon_bands)}: the lowest mean |err_norm| in the selection months "
        f"replaces {configuration.challenger_technique} only if it beats it by the margin "
        f"{configuration.challenger_margin_by_band}",
        "WHO COMPETES: every technique whose minimum history is reached (seasonal ones: 13 and 24 months); "
        "only the history limits, no seasonality verdict; the ranking uses the targets where all of them competed",
        f"BAND per technique and horizon: quantiles {configuration.band_low_quantile:.0%}-{configuration.band_high_quantile:.0%} "
        f"of err_norm in the selection months; in the exam ≈ {configuration.band_high_quantile - configuration.band_low_quantile:.0%} "
        f"of the errors should fall inside",
        "PRECISION in the exam: per id (chosen vs challenger) and of the TOTAL renewals of the judged ids, month by month"]
    for line in explanation:
        configuration.logger.doc(f"[{STEP_LABEL}]      {line}")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def predict_every_target(pool_series: pd.DataFrame, judged_ids: set, selection_months: list, exam_months: list,
                         configuration: Config) -> pd.DataFrame:
    """One row per (id, target month, horizon, technique). The history of a prediction is
    every month of the id up to origin = target − h."""
    period_column = configuration.period_col
    purpose_of = {month.ordinal: PURPOSE_SELECTION for month in selection_months}
    purpose_of.update({month.ordinal: PURPOSE_EXAM for month in exam_months})
    judged_series = pool_series[pool_series["id_estimacion"].isin(judged_ids) & pool_series["tasa"].notna()
                                & (pool_series["vencen"] > 0)]
    rows = []
    for estimation_id, monthly in judged_series.groupby("id_estimacion", sort=False):
        monthly = monthly.sort_values(period_column)
        month_ordinals = np.array([month.ordinal for month in monthly[period_column]])
        calendar_months = np.array([month.month for month in monthly[period_column]])
        months = list(monthly[period_column])
        rates = monthly["tasa"].to_numpy(dtype=float)
        units_due = monthly["vencen"].to_numpy(dtype=float)
        logit_rates = logit(rates)
        for target_position, target_ordinal in enumerate(month_ordinals):
            purpose = purpose_of.get(int(target_ordinal))
            if purpose is None:
                continue
            real_rate, real_units = rates[target_position], units_due[target_position]
            clipped = np.clip(real_rate, 0.5 / real_units, 1 - 0.5 / real_units)
            se_binomial_pp = PERCENTAGE_POINTS * np.sqrt(clipped * (1 - clipped) / real_units)
            for horizon in configuration.backtest_horizons:
                origin_ordinal = target_ordinal - horizon
                seen = month_ordinals <= origin_ordinal
                if seen.sum() < configuration.backtest_min_history_months:
                    continue
                history = logit_rates[seen]
                history_months = calendar_months[seen]
                last_seen = months[int(np.flatnonzero(seen)[-1])]
                for technique_id in eligible_techniques(len(history)):
                    predicted_logit = predict_logit(technique_id, history, history_months, horizon)
                    if not np.isfinite(predicted_logit):
                        continue
                    predicted_rate = float(inverse_logit(predicted_logit))
                    error_pp = PERCENTAGE_POINTS * (predicted_rate - real_rate)
                    rows.append((estimation_id, purpose, months[target_position], horizon,
                                 months[target_position] - horizon, last_seen, technique_id,
                                 predicted_rate, real_rate, real_units, error_pp, se_binomial_pp,
                                 error_pp / se_binomial_pp))
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHOICE, THE BANDS, THE EXAM
# ═══════════════════════════════════════════════════════════════════════════════════

def choose_techniques(predictions: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per id and band, on the selection months: the lowest mean |err_norm| replaces the
    challenger only if it beats it by the band's margin; ties go to the catalogue order.
    Ids below the floor get the challenger (sin_soporte)."""
    challenger = configuration.challenger_technique
    catalogue_order = {technique_id: position for position, technique_id in enumerate(CATALOGUE)}
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION].assign(
        abs_norm=lambda frame: frame["err_norm"].abs(), abs_pp=lambda frame: frame["err_pp"].abs())
    scores = (selection.groupby(["id_estimacion", "tramo_h", "tecnica"])
              .agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"), n_predicciones=("abs_norm", "size"))
              .reset_index())
    decision_rows = []
    for (estimation_id, band_name), block in scores.groupby(["id_estimacion", "tramo_h"]):
        block = block.set_index("tecnica")
        challenger_score = block.loc[challenger, "err_norm_medio"] if challenger in block.index else np.inf
        margin = float(configuration.challenger_margin_by_band.get(band_name, 0.0))
        candidates = block[block["n_predicciones"] >= configuration.backtest_min_predictions]
        chosen, origin = challenger, CHALLENGER_ORIGIN
        if len(candidates):
            best = min(candidates.index, key=lambda technique_id: (candidates.loc[technique_id, "err_norm_medio"],
                                                                   catalogue_order[technique_id]))
            if best != challenger and candidates.loc[best, "err_norm_medio"] < challenger_score - margin:
                chosen, origin = best, CHAMPION_ORIGIN
        chosen_row = block.loc[chosen] if chosen in block.index else pd.Series(dict(err_norm_medio=np.nan, err_pp_medio=np.nan,
                                                                                    n_predicciones=0))
        decision_rows.append({"id_estimacion": estimation_id, "tramo_h": band_name, "tecnica": chosen,
                              "tecnica_origen": origin, "err_norm_seleccion": chosen_row["err_norm_medio"],
                              "err_pp_seleccion": chosen_row["err_pp_medio"], "n_predicciones": int(chosen_row["n_predicciones"]),
                              "retador_err_norm_seleccion": challenger_score})
    decision = pd.DataFrame(decision_rows)

    # every id and band gets a decision: the unjudged ones (and a judged id with no selection
    # month in a band) take the challenger
    all_pairs = pd.MultiIndex.from_product([pool_reference[ESTIMATION_ID_COLUMN], list(configuration.horizon_bands)],
                                           names=["id_estimacion", "tramo_h"]).to_frame(index=False)
    decision = all_pairs.merge(decision, on=["id_estimacion", "tramo_h"], how="left")
    unjudged = decision["tecnica"].isna()
    decision.loc[unjudged, "tecnica"] = challenger
    decision.loc[unjudged, "tecnica_origen"] = SUPPORT_ORIGIN
    decision["n_predicciones"] = decision["n_predicciones"].fillna(0).astype(int)
    return decision


def error_bands(predictions: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per technique and horizon, the quantiles of err_norm in the selection months (every judged id)."""
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    return (selection.groupby(["tecnica", "h"])["err_norm"]
            .agg(q_low_norm=lambda errors: errors.quantile(configuration.band_low_quantile),
                 q_high_norm=lambda errors: errors.quantile(configuration.band_high_quantile),
                 n_predicciones="size")
            .reset_index())


def exam_precision(predictions: pd.DataFrame, decision: pd.DataFrame, bands: pd.DataFrame, configuration: Config) -> tuple:
    """The chosen technique and the challenger in the exam months: per id and for the total."""
    challenger = configuration.challenger_technique
    exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
    chosen_rows = exam.merge(decision[["id_estimacion", "tramo_h", "tecnica"]], on=["id_estimacion", "tramo_h", "tecnica"])
    chosen_rows = chosen_rows.merge(bands[["tecnica", "h", "q_low_norm", "q_high_norm"]], on=["tecnica", "h"], how="left")
    chosen_rows["dentro_banda"] = ((chosen_rows["err_norm"] >= chosen_rows["q_low_norm"])
                                   & (chosen_rows["err_norm"] <= chosen_rows["q_high_norm"])).astype(int)
    challenger_rows = exam[exam["tecnica"] == challenger]

    def summary(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (rows.assign(abs_pp=rows["err_pp"].abs(), abs_norm=rows["err_norm"].abs())
                .groupby(["id_estimacion", "tramo_h"])
                .agg(**{f"{prefix}_err_pp_medio": ("abs_pp", "mean"), f"{prefix}_sesgo_pp": ("err_pp", "mean"),
                        f"{prefix}_err_norm_medio": ("abs_norm", "mean")}))

    exam_by_pool = summary(chosen_rows, "elegida").join(summary(challenger_rows, "retador"), how="left").reset_index()
    exam_by_pool = exam_by_pool.merge(decision[["id_estimacion", "tramo_h", "tecnica", "tecnica_origen"]],
                                      on=["id_estimacion", "tramo_h"], how="left")
    exam_by_pool["mejora_pp"] = exam_by_pool["retador_err_pp_medio"] - exam_by_pool["elegida_err_pp_medio"]
    exam_by_pool["dentro_banda"] = chosen_rows.groupby(["id_estimacion", "tramo_h"])["dentro_banda"].mean().to_numpy()

    # the TOTAL: Σ predicted renewals vs Σ real renewals of every judged id, per exam month and horizon
    def total_of(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (rows.assign(pred_units=rows["tasa_pred"] * rows["vencen_real"], real_units=rows["tasa_real"] * rows["vencen_real"])
                .groupby(["mes_objetivo", "h"])
                .agg(**{f"{prefix}_renovadas_pred": ("pred_units", "sum"), "renovadas_reales": ("real_units", "sum"),
                        "vencen": ("vencen_real", "sum")}))

    exam_total = total_of(chosen_rows, "elegida").join(total_of(challenger_rows, "retador")[["retador_renovadas_pred"]]).reset_index()
    for prefix in ("elegida", "retador"):
        exam_total[f"{prefix}_error_pct"] = (exam_total[f"{prefix}_renovadas_pred"] - exam_total["renovadas_reales"]) / exam_total["renovadas_reales"]
    return exam_by_pool, exam_total, chosen_rows


def check_backtest(predictions: pd.DataFrame, decision: pd.DataFrame, pool_reference: pd.DataFrame,
                   exam_rows: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    # [1] no peeking
    peeked = predictions[(predictions["ultimo_mes_visto"] > predictions["origen"])
                         | (predictions["origen"] >= predictions["mes_objetivo"])]
    configuration.log_check(STEP_LABEL, check_log, "no prediction saw its target month or anything after its origin",
                            peeked.empty, failure_detail=f"{len(peeked):,} predictions saw beyond their origin",
                            context=f"{len(predictions):,} predictions")

    # [2] a decision per id and band
    expected = len(pool_reference) * len(configuration.horizon_bands)
    configuration.log_check(STEP_LABEL, check_log, "every estimation id has a decision in every horizon band",
                            len(decision) == expected and decision["tecnica"].notna().all(),
                            failure_detail=f"{len(decision):,} decisions, {expected:,} expected",
                            context=f"{len(pool_reference):,} ids × {len(configuration.horizon_bands)} bands")

    # [3] the challenger is always there to compare with
    targets = predictions.groupby(["id_estimacion", "mes_objetivo", "h"])["tecnica"].apply(set)
    without_challenger = int(sum(configuration.challenger_technique not in techniques for techniques in targets))
    configuration.log_check(STEP_LABEL, check_log, "the challenger was predicted wherever another technique was",
                            without_challenger == 0,
                            failure_detail=f"{without_challenger:,} targets without the challenger")

    # [4] a fair ranking: the techniques are compared on the targets where all of them competed
    common = common_targets(predictions)
    configuration.log_check(STEP_LABEL, check_log, "the ranking compares the techniques on common targets",
                            len(common) > 0,
                            failure_detail="no target where every technique competed: the ranking would mix different targets",
                            context=f"{len(common):,} of {predictions.groupby(['id_estimacion', 'mes_objetivo', 'h']).ngroups:,} "
                                    f"targets have every technique; the others lack a history of 24 months")

    # [5] the band holds what it promises in the exam
    coverage = float(exam_rows["dentro_banda"].mean()) if len(exam_rows) else np.nan
    nominal = configuration.band_high_quantile - configuration.band_low_quantile
    configuration.log_check(STEP_LABEL, check_log,
                            f"in the exam, the band holds at least {MIN_EXAM_COVERAGE:.0%} of the chosen technique's errors",
                            bool(coverage >= MIN_EXAM_COVERAGE),
                            failure_detail=f"only {coverage:.0%} inside (nominal {nominal:.0%}): the band is too narrow",
                            context=f"{coverage:.0%} inside (nominal {nominal:.0%}, {len(exam_rows):,} exam predictions)",
                            blocking=False)


def common_targets(predictions: pd.DataFrame) -> pd.DataFrame:
    """The (id, target, horizon) where every technique of the catalogue competed."""
    techniques_per_target = predictions.groupby(["id_estimacion", "mes_objetivo", "h"])["tecnica"].nunique()
    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[["id_estimacion", "mes_objetivo", "h"]]


def log_backtest_report(predictions: pd.DataFrame, decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                        exam_total: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> None:
    """Action 9: the ranking, the choices and the precision in the exam, as tables."""
    configuration.log_action(STEP_LABEL, 9, "ranking of the techniques in the SELECTION months (mean |err_norm|: 1 = "
                                            "as close as chance allows; same targets for every technique):")
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    common = common_targets(predictions)
    selection = selection.merge(common, on=["id_estimacion", "mes_objetivo", "h"])
    ranking = (selection.assign(abs_norm=selection["err_norm"].abs(), abs_pp=selection["err_pp"].abs())
               .groupby(["tramo_h", "tecnica"]).agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"),
                                                    predicciones=("abs_norm", "size"))
               .reset_index().sort_values(["tramo_h", "err_norm_medio"]))
    configuration.show_table(ranking)

    configuration.logger.doc(f"[{STEP_LABEL}] the chosen technique per band (ids and the money their series predict):")
    chosen_money = decision.merge(pool_reference[[ESTIMATION_ID_COLUMN, "usd_por_predecir"]], on=ESTIMATION_ID_COLUMN)
    configuration.show_table(chosen_money.groupby(["tramo_h", "tecnica", "tecnica_origen"])
                             .agg(ids=("id_estimacion", "size"), usd_por_predecir=("usd_por_predecir", "sum"))
                             .reset_index().sort_values(["tramo_h", "usd_por_predecir"], ascending=[True, False]))

    configuration.logger.doc(f"[{STEP_LABEL}] precision in the EXAM, per band (mean over the judged ids; err in pp of rate; "
                             f"mejora = challenger's error − chosen's error):")
    configuration.show_table(exam_by_pool.groupby("tramo_h")
                             .agg(ids=("id_estimacion", "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
                                  retador_err_pp=("retador_err_pp_medio", "mean"), mejora_pp=("mejora_pp", "mean"),
                                  elegida_sesgo_pp=("elegida_sesgo_pp", "mean"), dentro_banda=("dentro_banda", "mean"))
                             .reset_index())

    configuration.logger.doc(f"[{STEP_LABEL}] precision of the TOTAL in the exam (Σ predicted renewals vs Σ real of the "
                             f"judged ids; error_pct = (predicted − real) / real):")
    configuration.show_table(exam_total[["mes_objetivo", "h", "vencen", "renovadas_reales", "elegida_renovadas_pred",
                                         "elegida_error_pct", "retador_renovadas_pred", "retador_error_pct"]])
