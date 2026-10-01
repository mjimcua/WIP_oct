"""
test_step_19.py — The exam of every forecast series: three methods on the same rows (the series
alone, the framework, the spreadsheet), without peeking, every prediction traced and with its interval.

    python test_step_19.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from test_helpers import check, console_of, finish, synthetic_with


def test_the_series_exam() -> None:
    print("A · the exam of every forecast series on the synthetic")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    check("8 checks: 8 ok" in console.split("STEP 19")[1], "the 8 checks of step 19 pass")
    exam = results["series_exam"]
    detail, per_series, summary = exam["detail"], exam["per_series"], exam["summary"]
    rated = results["rated_units"]
    exam_real = rated[(rated["sintetica"] == 0) & (rated["rol"] == "examen")]
    check(abs(exam["by_month"][exam["by_month"]["h"] == 1]["renovadas_reales"].sum() - exam_real["total_renewed_units"].sum()) < 1e-6,
          "the real renewals of the exam are the renewals of every series in the exam months (nothing counted twice)")
    check(set(summary["metodo"]) == {"raw", "framework", "hoja_global", "hoja_mandatory"} and set(summary["h"]) == {1, 6},
          "three ways on the same rows: the series alone (raw), the framework and every spreadsheet grain, at every horizon")
    check((detail.groupby("method").size().nunique() == 1), "every method predicts exactly the same rows")
    check((detail["origin"] == detail["period"] - detail["h"]).all(), "every prediction knows only up to its origin, T − h")
    framework = detail[detail["method"] == "framework"]
    traced = framework.dropna(subset=["composition_rate"])
    no_shift = traced[traced["credibility_shift_logit"] == 0]
    check(len(traced) > 0 and np.allclose(no_shift["pred_rate"], no_shift["composition_rate"]),
          "every framework prediction is traced: without credibility, the series' rate is its composition's prediction")
    with_interval = detail[detail["method"].isin(["raw", "framework"])]
    check(with_interval["band_low"].notna().all() and (with_interval["band_low"] <= with_interval["pred_rate"] + 1e-12).all()
          and (with_interval["pred_rate"] <= with_interval["band_high"] + 1e-12).all()
          and detail.loc[detail["method"].str.startswith("hoja_"), "band_low"].isna().all(),
          "raw and framework predictions carry their interval; the spreadsheet has none")
    check({"raw_mae_pp", "framework_mae_pp", "improvement_mae_pp", "raw_coverage", "framework_coverage"} <= set(per_series.columns)
          and per_series["fs_id"].is_unique,
          "per forecast series, raw and framework side by side: the improvement is one subtraction")
    by_method = summary.groupby("metodo")["wape_series"].mean()
    check(by_method["framework"] < by_method["hoja_mandatory"],
          "series by series, the framework beats the spreadsheet by mandatory cell in the synthetic")
    report = open(f"{configuration.output_folder}/informe_sff.md", encoding="utf-8").read()
    check("framework vs mejor hoja de cálculo vs serie sola" in report, "the report puts the three ways in its headline")


if __name__ == "__main__":
    test_the_series_exam()
    finish()
