"""
test_step_20.py — Step 20: the time_series universe (retail to subscription) and the total.
The rules on a history built by hand, and the step on the synthetic.

    python test_step_20.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from step_20_time_series import REGION_KEY, project_months, reentry_rows, region_parameters
from test_helpers import check, console_of, finish, synthetic_with


def hand_made_history():
    """Region A: 2025 and 2026, 100 units a month in 2025, 110 in 2026, $10 a unit.
    Region B: only 2026 (no same month a year before): it must fall back to its share of the total."""
    rows = []
    for month in pd.period_range("2025-01", "2026-08", freq="M"):
        units = 100.0 if month.year == 2025 else 110.0
        rows.append({"period": month, REGION_KEY: "A", "region": "A", "unidades": units, "valor": units * 10.0})
    for month in pd.period_range("2026-01", "2026-08", freq="M"):
        rows.append({"period": month, REGION_KEY: "B", "region": "B", "unidades": 50.0, "valor": 50.0 * 20.0})
    return pd.DataFrame(rows)


def test_the_rules() -> None:
    print("A · the rules on a history built by hand (current month 2026-09)")
    configuration = synthetic_with(ts_region_columns=["region"])
    current = pd.Period("2026-09", freq="M")
    history = hand_made_history()
    parameters = region_parameters(history, current, configuration)
    check(abs(parameters.loc["A", "nivel"] - 1.10) < 1e-12 and abs(parameters.loc["A", "valor_medio"] - 10.0) < 1e-12,
          "level of A = Σ units Jun-Aug 2026 / Σ same months 2025 = 1.10; value per unit = Σ value / Σ units = $10")
    months = list(pd.period_range("2026-09", "2026-12", freq="M"))
    projected, fallback = project_months(history, parameters, months, current, configuration)
    a_september = projected[(projected[REGION_KEY] == "A") & (projected["period"] == pd.Period("2026-09", freq="M"))].iloc[0]
    check(abs(a_september["unidades"] - 110.0) < 1e-9 and abs(a_september["valor"] - 1100.0) < 1e-9,
          "A, September 2026 = 100 units of September 2025 × 1.10 = 110 units × $10 = $1,100")
    b_rows = projected[projected[REGION_KEY] == "B"]
    check(fallback == {"B"} and (b_rows["por_cuota"] == 1).all() and len(b_rows) == 4,
          "B has no same month in 2025: it takes its share of the projected total (and it is flagged)")
    rates = {1: pd.Series({"A": 0.8})}
    reentry = reentry_rows(projected.assign(region=projected[REGION_KEY]), rates, 0.6, configuration)
    a_reentry = reentry[(reentry["region"] == "A") & (reentry["period"] == pd.Period("2027-09", freq="M"))].iloc[0]
    check(abs(a_reentry["valor"] - 1100.0) < 1e-9 and a_reentry["descuento"] == 0.4 and abs(a_reentry["revenue"] - 1100.0 / 0.6 * 0.8) < 1e-9,
          "September 2026 falls due in September 2027 as pipeline: $1,100 at 40 % off; renewed at 100 %: $1,100 / 0.6 × rate 0.8")
    check((reentry.loc[reentry["region"] == "B", "tasa_global"] == 1).all(), "a region with no rate takes the global one, flagged")

    two_levels = synthetic_with(ts_region_columns=["region", "product"])
    projected_two = projected.assign(region=projected[REGION_KEY], product="A1")
    level_rates = {1: pd.Series({"A": 0.8, "B": 0.5}), 2: pd.Series({"A|A1": 0.9})}
    reentry_two = reentry_rows(projected_two, level_rates, 0.6, two_levels)
    check((reentry_two.loc[reentry_two["region"] == "A", "nivel_tasa"] == "product").all()
          and (reentry_two.loc[reentry_two["region"] == "B", "nivel_tasa"] == "region").all()
          and abs(reentry_two.loc[reentry_two["region"] == "B", "tasa"].iloc[0] - 0.5) < 1e-12,
          "the rate climbs the levels: A has its finest-level rate (0.9); B has none there and takes its region's (0.5)")


def test_the_step() -> None:
    print("B · the step on the synthetic")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    check("2 checks: 2 ok" in console.split("[00b]")[-1].split("STEP 01")[0] and "6 checks: 6 ok" in console.split("STEP 20")[1],
          "the split and the checks of step 20 pass")
    time_series_rows = results["time_series_rows"]
    check(len(time_series_rows) == 48 and not (results["fine_table"]["flag_time_series"] == 1).any(),
          "the flagged rows leave the raw after step 00: none reaches the fine table")
    table, total = results["time_series"], results["forecast_total"]
    check(set(table["origen"]) == {"ts_real", "ts_proyectado", "ts_reentrada"}, "the three origins of the universe")
    projected_months = set(table.loc[table["origen"] == "ts_proyectado", "period"].astype(str))
    check(projected_months == {"2026-09", "2026-10", "2026-11", "2026-12"}, "projected: from the current month to December")
    parts = total[total["origen"] != "TOTAL"].groupby("ano")["usd_renovado"].sum()
    totals = total[total["origen"] == "TOTAL"].set_index("ano")["usd_renovado"]
    check(np.allclose(parts.sort_index(), totals.sort_index()), "the origins of every year add up to its total")
    total_2026 = total[total["ano"] == 2026].set_index("origen")["usd_renovado"]
    check(abs(total_2026["TOTAL"] - (total_2026["pipeline_renovado_real"] + total_2026["pipeline_real_esperado"]
                                     + total_2026["ts_real"] + total_2026["ts_proyectado"])) < 0.01,
          "total 2026 = renewals of the pipeline (booked + expected) + ts_real + ts_proyectado")
    reentry_2027 = total[(total["ano"] == 2027) & (total["origen"] == "ts_reentrada")].iloc[0]
    projected_value = table.loc[table["origen"] == "ts_proyectado", "valor"].sum()
    check(abs(reentry_2027["usd_vence"] - projected_value) < 0.01,
          "2027: the pipeline of the re-entry is the projected value (at 40 % off), and it renews at the region's rate")


if __name__ == "__main__":
    test_the_rules()
    test_the_step()
    finish()
