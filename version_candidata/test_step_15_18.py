"""
test_step_15_18.py — Steps 15 to 18 on the synthetic: the uplift, its backtest, the forecast
in money and the final validation.

    python test_step_15_18.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from prediction import judged_horizon
from test_helpers import check, console_of, finish, synthetic_with


def test_acquisition_simulation() -> None:
    print("B · the acquisition of the window, simulated per acquisition value (hand-made fine table)")
    from step_17_forecast import simulate_acquisitions
    configuration = synthetic_with(acquisition_column="newcust", acquisition_values=["pure", "not_pure"],
                                   acquisition_discount=0.4)
    rows = []
    for due_month in pd.period_range("2025-01", "2027-08", freq="M"):
        acquired_in = due_month - 12
        for kind, units in (("pure", 100.0), ("not_pure", 40.0)):
            growth = 1.2 if acquired_in.year == 2026 else 1.0             # acquisitions of 2026 are 20 % above 2025
            rows.append({"period": due_month, "region": "EU", "product": "A", "dormant": False, "softcancel": False,
                         "no_instalado": False, "autorenew": False, "channel": "web", "newcust": kind,
                         "total_tr_units": units * growth, "total_tr_usd": units * growth * 10.0})
    fine = pd.DataFrame(rows)
    fine["rol"] = "entrenamiento"
    carried = ["region", "product", "dormant", "softcancel", "no_instalado", "autorenew", "channel", "newcust"]
    simulated = simulate_acquisitions(fine, list(pd.period_range("2026-09", "2026-12", freq="M")), carried, configuration)
    check(len(simulated) == 8 and set(simulated["newcust"]) == {"pure", "not_pure"},
          "one line per acquisition value and month: 2 lines × 4 months")
    pure_september = simulated[(simulated["newcust"] == "pure") & (simulated["period"].astype(str) == "2027-09")].iloc[0]
    check(abs(pure_september["total_tr_units"] - 120.0) < 1e-9 and abs(pure_september["total_tr_usd"] - 1200.0) < 1e-9,
          "pure, acquired 2026-09: 100 units of 2025-09 × level 1.2 = 120 units × $10 = $1,200, due 2027-09")
    check((simulated["discount_pct"] == 0.4).all() and (~simulated[["dormant", "softcancel", "no_instalado", "autorenew"]].astype(bool)).all().all(),
          "an acquisition carries the acquisition discount (0.4) and every timevarying flag off")


def test_steps_15_to_18() -> None:
    print("A · uplift, its backtest, the forecast and the validation")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    for label, count in (("15", 6), ("16", 3), ("17", 8)):
        check(f"{count} checks: {count} ok" in console.split(f"STEP {label}")[1], f"the checks of step {label} pass")
    check("7 checks:" in console.split("STEP 18")[1] and "0 failed" in console.split("STEP 18")[1].split("STEP NU")[0],
          "the final validation passes (warnings allowed)")

    cells = results["uplift_cells"]
    fine = results["fine_table"]
    renewers = fine[fine["rol"].isin(["entrenamiento", "examen"]) & (fine["total_renewed_units"] > 0)]
    one_cell = cells[cells["uplift_origen"] == "propia"].iloc[0]
    rows = renewers[renewers["uplift_cell_id"] == one_cell["uplift_cell_id"]]
    expected = rows["total_renewed_usd"].sum() / rows["total_tr_usd_renewed"].sum()
    check(abs(one_cell["uplift"] - expected) < 1e-9,
          "the uplift of a cell = Σ renewed USD / Σ (the USD that was falling due of the contracts that renewed)")
    homogeneity = results["uplift_homogeneity"]
    same_cell = homogeneity[homogeneity["uplift_cell_id"] == one_cell["uplift_cell_id"]].iloc[0]
    check(abs(same_cell["uplift_aproximado"] - same_cell["uplift_exacto"] * same_cell["ratio_seleccion"]) < 1e-9,
          "homogeneity: the approximate uplift = the exact one × the selection ratio (the old base hid that factor)")
    check("the claim '100-105 %':" in console and "histogram · every series in the distribution" in console
          and "█" in console and "month by month (24 closed" in console,
          "the revaluation report prints the claim, the histogram and the monthly path on screen (action 9)")
    check(results["uplift_verdict"]["via_usada_con_descuento"] == "estadistica",
          "in the synthetic the contract rule loses (observed 1.50 vs rule 1.67 at 40 %): the statistical path is used")

    forecast = results["forecast"]["forecast"]
    extended = forecast[forecast["origen_pipeline"].isin(["proyectada", "simulada"])]
    check(set(extended["period"].astype(str)) == {"2027-09", "2027-10", "2027-11", "2027-12"},
          "the simulation window (2026-09..2026-12, the current month included) falls due in 2027-09..2027-12")
    september_2026 = forecast[(forecast["origen_pipeline"] == "real") & (forecast["period"].astype(str) == "2026-09")]
    projected_september = extended[(extended["origen_pipeline"] == "proyectada") & (extended["period"].astype(str) == "2027-09")]
    check(abs(projected_september["total_tr_units"].sum() - september_2026["esperado_unidades"].sum()) < 1e-6
          and abs(projected_september["total_tr_usd"].sum() - september_2026["esperado_usd"].sum()) < 1e-6,
          "what is expected to renew in 2026-09 is the pipeline of 2027-09: units = renewed units, value = renewed USD")
    check((projected_september["discount_pct"] == 0).all(),
          "a renewal falls due again with discount 0 and the same dims")
    row = forecast.iloc[0]
    check(abs(row["esperado_usd"] - row["total_tr_usd"] * row["tasa"] * row["uplift"]) < 1e-9,
          "expected USD of a row = USD due × rate × uplift")
    check((forecast["esperado_usd_bajo"] <= forecast["esperado_usd"] + 1e-9).all()
          and (forecast["esperado_usd"] <= forecast["esperado_usd_alto"] + 1e-9).all(), "the band of every row contains its expected value")
    by_month = results["forecast"]["by_month"]
    check((by_month["banda_lineal_baja"] <= by_month["banda_cuadratura_baja"] + 1e-6).all()
          and (by_month["banda_cuadratura_alta"] <= by_month["banda_lineal_alta"] + 1e-6).all(),
          "by month, the quadrature band is inside the linear one (independent errors partly cancel)")
    check(judged_horizon(1, [1, 6]) == 1 and judged_horizon(3, [1, 6]) == 6 and judged_horizon(9, [1, 6]) == 6,
          "a horizon takes the band of the judged horizon above it, or the last")
    core = results["core"]
    check(abs(core["s17_esperado_usd"].sum() - forecast["esperado_usd"].sum()) < 0.01,
          "the core carries the forecast of every future row: Σ s17_esperado_usd = Σ of the forecast")
    report = open(f"{configuration.output_folder}/informe_sff.md", encoding="utf-8").read()
    check("## 7 · El forecast en dinero" in report and "Validación final" in report, "the report tells the forecast and the validation")


if __name__ == "__main__":
    test_steps_15_to_18()
    test_acquisition_simulation()
    finish()
