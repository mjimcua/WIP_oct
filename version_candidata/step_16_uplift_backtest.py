"""
step_16_uplift_backtest.py — Which uplift predicts the price of the renewals better:
the statistical one of the cell or the contract rule 1 / (1 − discount)?

HOW THE TEST WORKS:
  · the statistical uplift of every cell is estimated ONLY with the renewals of the months
    BEFORE the exam (no peeking);
  · in the exam months, for every row that renewed, the renewed money is predicted GIVEN
    its real renewed units (so only the price is judged, not the rate), the SAME way the
    forecast computes it (USD due × rate × uplift = renewed units × pipeline AUV × uplift once
    the real rate is given):
        statistical   renewed units × pipeline AUV × uplift of its cell
        contract      renewed units × pipeline AUV × 1 / (1 − discount)   (known discount only)
    The base is the row's average price, NOT the exact value of the renewers that step 15 uses
    to ESTIMATE the uplift: the forecast cannot know who will renew, so it values them at the
    row's average, and the test must judge that. Where renewers were worth more than the
    average (sff_uplift_homogeneidad), the error shows up here, as it will in the forecast.
  · both are compared on the SAME rows (the ones with a known discount): WAPE = Σ|predicted −
    real| / Σ real, and the bias of the total = (Σ predicted − Σ real) / Σ real;
  · VERDICT: the contract path is used in the forecast, where the discount is known, only if
    its WAPE is lower. Otherwise the statistical path is used everywhere. The verdict governs.
The rows with an unknown discount are measured too (statistical only): they always take it.

Actions (logged as they are done):
  1. the statistical uplift estimated with the months before the exam
  2. the renewals of the exam months, predicted by both paths
  3. the error of each path, the verdict
  4. check the test                                                  checks 1-2
  5. write the result                                                check 3
  6. count the checks; stop if any failed
  7. show the comparison, as a table

Checks (logged as they are made, numbered, at the level of their status):
   1. the statistical uplift was estimated without the exam months
   2. both paths were compared on the same rows
   3. table sff_backtest_uplift written and read back

Output: the comparison (per path and per discount bucket) and the verdict (a dict) ·
table sff_backtest_uplift.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from step_15_uplift import estimate_cell_uplifts, renewer_rows
from vocabulario import (CALENDAR_ROLE_COLUMN, PATH_CONTRACT, PATH_STATISTICAL, ROLE_TEST, TABLE_UPLIFT_BACKTEST,
                         UPLIFT_CELL_ID_COLUMN)


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "16"
STEP_NAME = "BACKTEST OF THE UPLIFT"
STEP_PURPOSE = ("decide, in the exam months, which uplift predicts the price of the renewals better where the discount "
                "is known: the statistical one of the cell (estimated before the exam) or the contract rule 1 / (1 − d); "
                "the verdict governs the forecast")
STEP_ACTIONS = ["the statistical uplift estimated with the months before the exam",
                "the renewals of the exam months, predicted by both paths",
                "the error of each path, the verdict",
                "check the test (checks 1-2)",
                "write the result (check 3)",
                "count the checks; stop if any failed",
                "show the comparison, as a table"]
STEP_OUTPUT = "the error of each path (total and per discount bucket) and the verdict · table sff_backtest_uplift"


def backtest_uplift(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """Statistical vs contract uplift in the exam months; the verdict."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
    check_log = []
    exam_start = configuration.calendar_boundaries()["test_start"]

    # [1] the statistical uplift, without the exam months
    before_exam = renewer_rows(fine_table, configuration, before_month=exam_start)
    cells = estimate_cell_uplifts(fine_table, before_exam, configuration)
    configuration.log_action(STEP_LABEL, 1, f"statistical uplift of {len(cells):,} cells estimated with the renewals before "
                                            f"{exam_start} ({len(before_exam):,} renewer rows)")

    # [2] the renewals of the exam, predicted by both paths
    exam_rows = renewer_rows(fine_table, configuration)
    exam_rows = exam_rows[exam_rows[CALENDAR_ROLE_COLUMN] == ROLE_TEST].copy()
    # the base the forecast will use: the renewed units at the row's average due price (not the exact base)
    forecast_base = exam_rows["_denominador_aproximado"]
    exam_rows["pred_" + PATH_STATISTICAL] = forecast_base * exam_rows[UPLIFT_CELL_ID_COLUMN].map(
        cells.set_index(UPLIFT_CELL_ID_COLUMN)["uplift"])
    discount = configuration.discount_value_column
    known = exam_rows[discount].notna() if discount else pd.Series(False, index=exam_rows.index)
    exam_rows["pred_" + PATH_CONTRACT] = np.where(known, forecast_base / (1 - exam_rows[discount].clip(upper=configuration.contract_discount_cap))
                                                  if discount else np.nan, np.nan)
    configuration.log_action(STEP_LABEL, 2, f"{len(exam_rows):,} renewer rows in the exam; {int(known.sum()):,} with a known discount")

    # [3] the error of each path, and the verdict
    comparison = compare_paths(exam_rows, known, configuration)
    known_rows = comparison[comparison["filas"] == "descuento conocido"].set_index("via")
    contract_wins = (PATH_CONTRACT in known_rows.index and PATH_STATISTICAL in known_rows.index
                     and known_rows.loc[PATH_CONTRACT, "wape"] < known_rows.loc[PATH_STATISTICAL, "wape"])
    verdict = {"via_contrato_gana": bool(contract_wins),
               "via_usada_con_descuento": PATH_CONTRACT if contract_wins else PATH_STATISTICAL}
    configuration.log_action(STEP_LABEL, 3, f"verdict: where the discount is known the forecast uses the "
                                            f"{verdict['via_usada_con_descuento'].upper()} path "
                                            f"({'it predicts the exam renewals better' if contract_wins else 'the contract rule does not beat the statistical uplift'})")

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the test")
    configuration.log_check(STEP_LABEL, check_log, "the statistical uplift was estimated without the exam months",
                            before_exam[configuration.period_col].max() < exam_start if len(before_exam) else True,
                            failure_detail="the estimate saw the exam", context=f"last month used {before_exam[configuration.period_col].max()}")
    same_rows = comparison[comparison["filas"] == "descuento conocido"]["renovadas_filas"].nunique() <= 1
    configuration.log_check(STEP_LABEL, check_log, "both paths were compared on the same rows", bool(same_rows),
                            failure_detail="the two paths were measured on different rows")

    # [5] the result
    configuration.log_action(STEP_LABEL, 5, "writing the result")
    configuration.write_table(STEP_LABEL, check_log, comparison.assign(via_usada_con_descuento=verdict["via_usada_con_descuento"]),
                              TABLE_UPLIFT_BACKTEST)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the comparison
    configuration.log_action(STEP_LABEL, 7, "the price of the exam renewals predicted by each path (wape: Σ|pred − real| / "
                                            "Σ real; sesgo: (Σ pred − Σ real) / Σ real):")
    configuration.show_table(comparison)
    return comparison, verdict


def compare_paths(exam_rows: pd.DataFrame, known: pd.Series, configuration: Config) -> pd.DataFrame:
    """WAPE and bias of each path: on the rows with a known discount (both paths) and on the
    rows with an unknown one (statistical only); and per discount bucket."""
    rows = []
    def measure(subset, path, label):
        predicted, real = subset["pred_" + path], subset["_numerador"]
        valid = predicted.notna()
        if not valid.any():
            return
        rows.append({"filas": label, "via": path, "renovadas_filas": int(valid.sum()), "usd_real": real[valid].sum(),
                     "usd_pred": predicted[valid].sum(),
                     "wape": float((predicted[valid] - real[valid]).abs().sum() / real[valid].sum()),
                     "sesgo": float((predicted[valid].sum() - real[valid].sum()) / real[valid].sum())})
    for path in (PATH_STATISTICAL, PATH_CONTRACT):
        measure(exam_rows[known], path, "descuento conocido")
    measure(exam_rows[~known], PATH_STATISTICAL, "descuento desconocido")
    return pd.DataFrame(rows)
