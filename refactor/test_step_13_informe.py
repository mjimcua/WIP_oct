"""
test_step_13_informe.py — Step 13 (dynamics) on series built by hand, and the report on the synthetic.

    python test_step_13_informe.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd

from main import run
from step_13_dynamics import dynamics_of_one_series
from test_helpers import check, console_of, finish, synthetic_with


def monthly_series(rates, units=2000, start="2023-01"):
    months = pd.period_range(start, periods=len(rates), freq="M")
    renewed = np.round(np.asarray(rates) * units)
    return pd.DataFrame({"period": months, "renovadas": renewed, "vencen": float(units)})


def test_the_dynamics() -> None:
    print("A · φ, trend and seasonality on series built by hand")
    configuration = synthetic_with()
    generator = np.random.default_rng(3)
    flat = monthly_series(generator.binomial(2000, 0.7, 36) / 2000)
    result = dynamics_of_one_series(flat, "period", configuration)
    check(0.4 < result["phi"] < 2.0 and result["estacional"] == 0 and result["tendencia"] == 0,
          "a constant rate with binomial noise: φ ≈ 1, no season, no trend")
    december_peak = np.array([0.70] * 11 + [0.85])
    seasonal = monthly_series(np.tile(december_peak, 3) + generator.normal(0, 0.005, 36))
    result = dynamics_of_one_series(seasonal, "period", configuration)
    check(result["estacional"] == 1 and result["meses_alto"] == "12" and result["phi"] > 5,
          "a December peak every year: seasonal, December is a high month, φ well above 1")
    trending = monthly_series(np.linspace(0.60, 0.75, 36))
    result = dynamics_of_one_series(trending, "period", configuration)
    check(result["tendencia"] == 1 and 4 < result["tendencia_pp_ano"] < 6, "a rate rising 5 pp a year: trend +1, ≈ +5 pp/year")


def test_the_report() -> None:
    print("B · the report and the card on the synthetic")
    configuration = synthetic_with(output_folder=os.path.join(os.path.dirname(configuration_path()), "salida_test"))
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    report_path = os.path.join(configuration.output_folder, "informe_sff.md")
    report = open(report_path, encoding="utf-8").read()
    check(report.count("\n## ") == 8 and "## 5 · Qué tal se predice la tasa" in report and "## 7 · El forecast en dinero" in report,
          "the report has its eight chapters")
    check("**Los huecos:** 11 en 1 series" in report and "Resultados adelantados borrados" in report
          and "| resultado_adelantado_borrado |" in report and "| proyectada |" in report,
          "the report tells, in one format, the rows the framework adds or wipes (gaps, wiped, created)")
    check("El error frente al ruido" in report and "parte_del_error_que_es_ruido" in report,
          "the report tells how much of the exam error is noise, by size of series and for the total")
    card = results["card"]
    check(len(card) == len(results["series"]) and {"phi", "tecnica_corto", "elegida_err_pp_medio_corto", "nivel_riesgo"} <= set(card.columns),
          "the card: one row per series with dynamics, technique and exam error")
    check("s13_phi" in results["forecast_series"].columns, "the forecast series dimension carries the dynamics of its composition")
    check("3 checks: 3 ok" in console.split("STEP IN")[1], "the checks of the report pass")


def configuration_path():
    import tempfile
    return tempfile.mkdtemp() + "/x"


def test_a_constant_series() -> None:
    print("C · series that never vary (0 % every month; 1-2 units that always renew): no numpy warning")
    import warnings
    import numpy as np
    from step_13_dynamics import dynamics_of_one_series
    tiny = np.r_[np.full(30, 1.0), np.full(6, 2.0)]
    for name, due, renewed in (("0 % every month", 10.0, 0.0), ("1-2 units that always renew", tiny, tiny)):
        months = pd.DataFrame({"period": pd.period_range("2023-01", periods=36, freq="M"), "vencen": due, "renovadas": renewed})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                dynamics_of_one_series(months, "period", synthetic_with())
                clean = True
            except RuntimeWarning:
                clean = False
        check(clean, f"{name}: measured without a RuntimeWarning (a profile that does not move has no correlation)")


if __name__ == "__main__":
    test_the_dynamics()
    test_the_report()
    test_a_constant_series()
    finish()
