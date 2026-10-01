"""
test_report_queries.py — Every SQL query printed for an aggregated report reproduces that report: it is
run on the tables the synthetic run writes (SQLite) and compared with the table the library computed.

    python test_report_queries.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from main import run
from report_queries import (calendar_query, exam_by_month_query, exam_summary_query, forecast_total_query, new_rows_query,
                            roles_query, table_in_sql)
from test_helpers import check, console_of, finish, synthetic_with


def same(from_sql: pd.DataFrame, from_library: pd.DataFrame, keys: list, values: list) -> bool:
    """The two tables have the same keys and the same values (to 1e-6, nulls as 0)."""
    left = from_sql.assign(**{key: from_sql[key].astype(str) for key in keys}).set_index(keys)[values].astype(float).fillna(0.0)
    right = from_library.assign(**{key: from_library[key].astype(str) for key in keys}).set_index(keys)[values].astype(float).fillna(0.0)
    if set(left.index) != set(right.index):
        print("      keys only in SQL:", sorted(set(left.index) - set(right.index))[:5], "· only in the library:",
              sorted(set(right.index) - set(left.index))[:5])
        return False
    difference = (left.sort_index() - right.loc[left.sort_index().index]).abs().max().max()
    if difference > 1e-6:
        print("      largest difference:", difference)
    return bool(difference <= 1e-6)


def test_every_query() -> None:
    print("A · every printed query reproduces its report from the tables of the run")
    configuration = synthetic_with()
    results = {}
    console = console_of(lambda: results.update(run(configuration)))
    engine = configuration.sql_engine
    check(console.count("the SQL that reproduces it from the tables of the run") == 6, "six aggregated reports print their SQL")
    check(table_in_sql(configuration, "nucleo") == ("sff_nucleo" if not configuration.sql_schema else f"{configuration.sql_schema}.sff_nucleo"),
          "the queries name the tables as the run writes them (schema and prefix of the Config)")

    new_rows = pd.read_sql(new_rows_query(configuration), engine)
    check(same(new_rows, results["new_rows"], ["mes", "origen"], ["filas", "series", "unidades", "usd", "esperado_usd"]),
          "step 21: the rows the framework adds or wipes")

    fine = results["fine_table"]
    by_role = fine.groupby("rol").agg(filas=("rol", "size"), unidades_vencen=("total_tr_units", "sum"),
                                      unidades_renovadas=("total_renewed_units", "sum")).reset_index()
    roles = pd.read_sql(roles_query(configuration), engine)
    check(same(roles, by_role, ["rol"], ["filas", "unidades_vencen", "unidades_renovadas"]), "step 02: the calendar per role")

    calendar = pd.read_sql(calendar_query(configuration), engine)
    written = pd.read_sql(f"SELECT * FROM {table_in_sql(configuration, 'calendario')}", engine)
    check(same(calendar, written, ["period"], ["filas", "unidades_vencen", "usd_vence", "unidades_renovadas", "usd_renovado",
                                               "s0_renovados_unidades", "s0_renovados_usd"]),
          "step 02: the calendar per month (sff_calendario)")

    total = pd.read_sql(forecast_total_query(configuration), engine)
    check(same(total, results["forecast_total"], ["ano", "origen"], ["usd_vence", "usd_renovado"]),
          "step 20: the total per year and origin (sff_forecast_total)")

    exam = results["series_exam"]
    methods = sorted(exam["detail"]["method"].unique(), key=lambda method: (method != "raw", method != "framework", method))
    by_month = pd.read_sql(exam_by_month_query(configuration, methods), engine)
    value_columns = ["renovadas_reales"] + [f"{method}_{part}" for method in methods for part in ("pred", "error_total")]
    check(same(by_month, exam["by_month"], ["mes", "h"], value_columns), "step 19: the total of every exam month per method")
    summary = pd.read_sql(exam_summary_query(configuration), engine)
    check(same(summary, exam["summary"], ["metodo", "h"], ["error_total_medio", "sesgo_total_medio", "wape_series",
                                                           "error_vs_ruido", "en_intervalo"]),
          "step 19: the exam per method and horizon")


if __name__ == "__main__":
    test_every_query()
    finish()
