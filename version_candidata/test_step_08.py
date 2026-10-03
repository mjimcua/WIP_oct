"""
test_step_08.py — Step 08: gaps inside the history, the rate of every unit only where it is
truth, and the summary of every series (support, own rate, Wilson error, sign).

    python test_step_08.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from step_06_series_routes import build_series_routes
from step_08_rate_series import build_rate_series, wilson_half_width_pp
from test_helpers import check, check_stops, console_of, count_status, finish
from test_step_05 import units_synthetic


def inputs_synthetic():
    _, units, configuration = units_synthetic()
    frames = {}
    console_of(lambda: frames.setdefault("series", build_series_routes(units, configuration)))
    return units, frames["series"], configuration


def run_step(units, series, configuration):
    frames = {}
    console = console_of(lambda: frames.setdefault("out", build_rate_series(units, series, configuration)))
    rated_units, series_rate = frames["out"]
    return rated_units, series_rate, console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_wilson() -> None:
    print("A · the Wilson half-width")
    check(abs(wilson_half_width_pp(np.array([0.5]), np.array([30.0]), 1.645)[0] - 14.2) < 0.3,
          "p = 0.5, n = 30, z = 1.645 → about ±14 pp (the support floor is ±15 pp at p = 0.5)")
    check(wilson_half_width_pp(np.array([0.5]), np.array([0.0]), 1.645)[0] == 50.0, "no support → ±50 pp, the widest")
    narrow, wide = wilson_half_width_pp(np.array([0.8, 0.8]), np.array([600.0, 30.0]), 1.645)
    check(narrow < wide, "more support, narrower interval")


def test_rates_and_gaps() -> None:
    print("B · gaps and rates")
    units, series, configuration = inputs_synthetic()
    rated, summary, console = run_step(units, series, configuration)
    gaps = rated[rated["sintetica"] == 1]
    check(len(gaps) == 11 and gaps["tasa"].isna().all() and (gaps["total_tr_units"] == 0).all(),
          "11 gap rows: nothing due and no rate (a month with no expirations says nothing)")
    check(rated.loc[rated["rol"] == "proyeccion", "tasa"].isna().all(), "no rate in the future")
    closed_real = rated[(rated["sintetica"] == 0) & rated["rol"].isin(["entrenamiento", "examen"])]
    check(np.allclose(closed_real["tasa"], closed_real["total_renewed_units"] / closed_real["total_tr_units"]),
          "rate = renewed / due in every real closed unit")
    check(count_status(console, "ok") == 6 and "6 checks: 6 ok" in console, "the 6 checks pass and are logged")

    top = summary.set_index("fs_id").loc["EU|A|0|0|0|0|web"]
    history = rated[(rated["fs_id"] == "EU|A|0|0|0|0|web") & rated["rol"].isin(["entrenamiento", "examen"])]
    check(abs(top["tasa_propia"] - history["total_renewed_units"].sum() / history["total_tr_units"].sum()) < 1e-12,
          "own rate = Σ renewed / Σ due over the closed months (not an average of rates)")
    check(top["n_propio"] == history.loc[history["total_tr_units"] > 0, "total_tr_units"].median(),
          "n_propio = the median units due of a closed month")
    check(summary.set_index("fs_id").loc["EU|A|0|0|0|0|kiosk", "meses_historia"] == 0,
          "a series of the time_series universe has no own history here (it is treated apart)")
    check(set(summary["signo"]) == {"neutro", "negativo", "positivo", "mixto"},
          "the four signs appear (the synthetic has series with negative, positive and both kinds of flags)")
    written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_series_tasa", configuration.sql_engine)["n"].item()
    check(written == len(series), "sff_series_tasa written, one row per series")


def test_a_rate_above_one_warns() -> None:
    print("C · a rate above 100 % is a warning")
    units, series, configuration = inputs_synthetic()
    above = units.copy()
    closed_row = above.index[above["rol"] == "entrenamiento"][0]
    above.loc[closed_row, "total_renewed_units"] = above.loc[closed_row, "total_tr_units"] + 5
    _, _, console = run_step(above, series, configuration)
    check("WARN  every rate between 0 and 1" in console, "the unit is shown and the step goes on")


if __name__ == "__main__":
    test_wilson()
    test_rates_and_gaps()
    test_a_rate_above_one_warns()
    finish()
