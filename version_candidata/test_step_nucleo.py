"""
test_step_nucleo.py — The core table on the synthetic: one row per fine row plus the gaps,
the money reconciles, the series values repeat on every row, the legend covers everything.

    python test_step_nucleo.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from main import run
from test_helpers import check, console_of, count_status, finish, synthetic_with


def test_the_core() -> None:
    print("A · the core (fact table) and the forecast series dimension")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    core, legend, dimension = results["core"], results["core_legend"], results["forecast_series"]
    check("10 checks: 10 ok" in console.split("STEP NU")[1], "the 10 checks of the core pass")
    extended = int(results["forecast"]["forecast"]["_fila"].isna().sum())
    time_series_rows = len(results["time_series_rows"])
    synthetic_time_series = int(results["time_series"]["origen"].isin(["ts_proyectado", "ts_reentrada"]).sum())
    check(len(core) == len(results["fine_table"]) + int((results["rated_units"]["sintetica"] == 1).sum()) + extended
          + time_series_rows + synthetic_time_series,
          "every original record (pipeline and time_series) and every synthetic row (gaps, extended, ts), once")
    check({"raw", "hueco", "proyectada", "ts_real", "ts_sin_resultado", "ts_proyectado", "ts_reentrada"}
          <= set(core["origen_fila"]) <= {"raw", "hueco", "proyectada", "simulada", "ts_real", "ts_sin_resultado",
                                          "ts_proyectado", "ts_reentrada"},
          "the origins of the rows: original and synthetic, of both universes")
    total = results["forecast_total"]
    renewed_2026 = core.loc[core["fin_ano"] == 2026, "fin_renovado_usd"].sum()
    pipeline_2027 = core.loc[core["fin_ano"] == 2027, "fin_vence_usd"].sum()
    check(abs(renewed_2026 - total.loc[(total["ano"] == 2026) & (total["origen"] == "TOTAL"), "usd_renovado"].item()) < 0.01
          and abs(pipeline_2027 - total.loc[(total["ano"] == 2027) & (total["origen"] == "TOTAL"), "usd_vence"].item()) < 0.01,
          "Q1 (renewed 2026) and Q2 (pipeline 2027) are a SUM of the core: equal to sff_forecast_total")
    real_2026 = core[(core["fin_ano"] == 2026) & (core["fin_estado"] == "real")]["fin_renovado_usd"].sum()
    expected_2026 = core[(core["fin_ano"] == 2026) & (core["fin_estado"] == "previsto")]["fin_renovado_usd"].sum()
    check(abs(real_2026 + expected_2026 - renewed_2026) < 0.01, "2026 splits into real (done) + previsto (to renew)")
    gaps = core[core["origen_fila"] == "hueco"]
    check(len(gaps) > 0 and (gaps["s00_vencen_unidades"] == 0).all() and gaps["s03_fs_id"].isin(dimension["s03_fs_id"]).all(),
          "a gap row has every measure at 0 and its forecast series is in the dimension")
    raw = configuration.read_raw()
    check(abs(core.loc[core["fin_universo"] == "pipeline", "s00_vencen_usd"].sum()
              - raw.loc[raw["flag_time_series"] != 1, "total_tr_usd"].sum()) < 0.01,
          "Σ USD due in the core = Σ in the extract without the time_series universe (it is simulated apart, step 20)")
    check(dimension["s03_fs_id"].is_unique and set(dimension["s03_fs_id"]) == set(core["s03_fs_id"].dropna())
          and not any(column.startswith(("s06_", "s08_", "s10_", "s11_", "s19_")) for column in core.columns),
          "every value of a forecast series lives once, in sff_forecast_series; the core keeps the rows")
    one_series = dimension[dimension["s03_fs_id"] == "EU|A|0|0|0|0|web"]
    check(not any(column.endswith("_key") for column in core.columns), "no hash keys in the core")
    core_legend_rows = legend[legend["tabla"] == "sff_nucleo"]
    dimension_legend_rows = legend[legend["tabla"] == "sff_forecast_series"]
    check(set(core_legend_rows["columna"]) == set(core.columns) and set(dimension_legend_rows["columna"]) == set(dimension.columns),
          "the legend describes every column of both tables")
    closed = core[core["s02_rol"].isin(["entrenamiento", "examen"]) & (core["s03_fs_id"] == "EU|A|0|0|0|0|web")]
    check(abs(closed["s02_renovadas_unidades"].sum() / closed["s00_vencen_unidades"].sum()
              - one_series["s08_tasa_propia"].iloc[0]) < 1e-12,
          "the ratio of sums over the closed rows of a series = its own rate (what the Power BI measure computes)")
    written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_nucleo", configuration.sql_engine)["n"].item()
    written_dimension = pd.read_sql("SELECT COUNT(*) AS n FROM sff_forecast_series", configuration.sql_engine)["n"].item()
    check(written == len(core) and written_dimension == len(dimension), "sff_nucleo and sff_forecast_series written")
    summed = dimension["s19_exam_real_units"].sum()
    check(abs(summed - results["series_exam"]["per_series"]["framework_real_units"].sum()) < 1e-6,
          "a SUM over the dimension counts every forecast series' exam once")


if __name__ == "__main__":
    test_the_core()
    finish()
