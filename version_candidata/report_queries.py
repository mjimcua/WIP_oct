"""
report_queries.py — The SQL that reproduces every aggregated report from the detail tables.

A report with few rows because it is very aggregated (per month, per role, per year and origin, per
method) is not a source of data: it is a query over the detail tables the run writes. The step shows the
report and prints the query, so anyone can reproduce it in SQL, or in Power BI, from the same tables, and
the report never becomes a second copy of the data.

Every query reads only tables the run writes (sff_nucleo, sff_series_exam_detail) and uses plain SQL that
SQL Server and SQLite both run (the tests run every query on SQLite and compare it with the report).

  new_rows_query          step 21  the rows the framework adds or wipes, month × origin   ← sff_nucleo
  roles_query             step 02  the calendar per role                                 ← sff_nucleo
  calendar_query          step 02  the calendar per month (sff_calendario)               ← sff_nucleo
  forecast_total_query    step 20  the total per year and origin (sff_forecast_total)    ← sff_nucleo
  exam_by_month_query     step 19  the total of every exam month per method              ← sff_series_exam_detail
  exam_summary_query      step 19  the exam per method and horizon                       ← sff_series_exam_detail

The reports computed with medians or quantiles (the ladder summary, the error bands) or from the JSON of
the generated levels have no query here: they are not a plain sum over a detail table.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from vocabulario import TABLE_CORE, TABLE_SERIES_EXAM_DETAIL


def table_in_sql(configuration, table_name: str) -> str:
    """The name of a table of the run as SQL writes it: schema.prefix+name (or prefix+name)."""
    physical_name = f"{configuration.table_prefix}{table_name}"
    return f"{configuration.sql_schema}.{physical_name}" if configuration.sql_schema else physical_name


def new_rows_query(configuration) -> str:
    """Step 21: the gaps, what the calendar wiped and what the forecast created, month × origin."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 21 · the rows the framework adds or wipes, month × origin
SELECT {period} AS mes, s02_rol AS rol, 'hueco' AS origen, COUNT(*) AS filas, COUNT(DISTINCT s03_fs_id) AS series,
       0 AS unidades, 0 AS usd, NULL AS esperado_usd
FROM {core} WHERE origen_fila = 'hueco' GROUP BY {period}, s02_rol
UNION ALL
SELECT {period}, s02_rol, 'resultado_adelantado_borrado', COUNT(*), COUNT(DISTINCT s03_fs_id),
       SUM(s00_renovadas_unidades - COALESCE(s02_renovadas_unidades, 0)),
       SUM(s00_renovado_usd - COALESCE(s02_renovado_usd, 0)), NULL
FROM {core} WHERE origen_fila = 'raw' AND COALESCE(s00_renovadas_unidades, 0) > COALESCE(s02_renovadas_unidades, 0)
GROUP BY {period}, s02_rol
UNION ALL
SELECT {period}, s02_rol, 'pipeline_parcial_borrada', COUNT(*), COUNT(DISTINCT s03_fs_id),
       SUM(s00_vencen_unidades - s02_vencen_unidades), SUM(s00_vencen_usd - s02_vencen_usd), NULL
FROM {core} WHERE origen_fila = 'raw' AND s00_vencen_unidades > s02_vencen_unidades
GROUP BY {period}, s02_rol
UNION ALL
SELECT {period}, s02_rol, origen_fila, COUNT(*), COUNT(DISTINCT s03_fs_id),
       SUM(fin_vence_unidades), SUM(fin_vence_usd), SUM(s17_esperado_usd)
FROM {core} WHERE origen_fila IN ('proyectada', 'simulada') GROUP BY {period}, s02_rol, origen_fila
ORDER BY mes, origen;"""


def roles_query(configuration) -> str:
    """Step 02: the calendar per role (the rows of the extract)."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 02 · the calendar per role
SELECT s02_rol AS rol, COUNT(DISTINCT {period}) AS meses, MIN({period}) AS desde, MAX({period}) AS hasta, COUNT(*) AS filas,
       SUM(s02_vencen_unidades) AS unidades_vencen, SUM(s02_renovadas_unidades) AS unidades_renovadas,
       SUM(s02_renovadas_unidades) / SUM(s02_vencen_unidades) AS tasa_unidades
FROM {core} WHERE origen_fila = 'raw'
GROUP BY s02_rol ORDER BY MIN({period});"""


def calendar_query(configuration) -> str:
    """Step 02: the calendar per month (the content of sff_calendario)."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 02 · the calendar per month (sff_calendario)
SELECT {period}, MIN(s02_rol) AS rol, MAX(s02_es_mes_en_curso) AS es_mes_en_curso, COUNT(*) AS filas,
       SUM(s02_vencen_unidades) AS unidades_vencen, SUM(s02_vencen_usd) AS usd_vence,
       SUM(s02_renovadas_unidades) AS unidades_renovadas, SUM(s02_renovado_usd) AS usd_renovado,
       SUM(s00_renovadas_unidades) AS s0_renovados_unidades, SUM(s00_renovado_usd) AS s0_renovados_usd,
       SUM(s02_renovadas_unidades) / SUM(s02_vencen_unidades) AS tasa_unidades
FROM {core} WHERE origen_fila = 'raw'
GROUP BY {period} ORDER BY {period};"""


def forecast_total_query(configuration) -> str:
    """Step 20: the total per year and origin, and the TOTAL of every year (the content of sff_forecast_total)."""
    core = table_in_sql(configuration, TABLE_CORE)
    first_year = configuration.calendar_boundaries()["current"].year
    # the rows that carry no money in the answer are left out: a gap (a month with nothing due) and an original
    # time_series row of a month not closed yet (its result is replaced by its projection)
    without_money = "origen_fila NOT IN ('hueco', 'ts_sin_resultado')"
    return f"""-- step 20 · the total per year and origin (sff_forecast_total)
SELECT fin_ano AS ano, fin_origen AS origen, SUM(fin_vence_usd) AS usd_vence, SUM(fin_renovado_usd) AS usd_renovado
FROM {core} WHERE fin_ano >= {first_year} AND {without_money}
GROUP BY fin_ano, fin_origen
UNION ALL
SELECT fin_ano, 'TOTAL', SUM(fin_vence_usd), SUM(fin_renovado_usd)
FROM {core} WHERE fin_ano >= {first_year} AND {without_money}
GROUP BY fin_ano
ORDER BY ano, origen;"""


def exam_by_month_query(configuration, methods: list) -> str:
    """Step 19: per exam month and horizon, the real renewals and every method's prediction and error of the total."""
    detail = table_in_sql(configuration, TABLE_SERIES_EXAM_DETAIL)
    period = configuration.period_col
    reference = methods[0]
    per_method = ",\n       ".join(
        f"SUM(CASE WHEN method = '{method}' THEN pred_units END) AS {method}_pred,\n"
        f"       SUM(CASE WHEN method = '{method}' THEN pred_units END) / SUM(CASE WHEN method = '{reference}' THEN real_units END) - 1"
        f" AS {method}_error_total" for method in methods)
    return f"""-- step 19 · the total of every exam month and horizon, per method
SELECT {period} AS mes, h, SUM(CASE WHEN method = '{reference}' THEN real_units END) AS renovadas_reales,
       {per_method}
FROM {detail}
GROUP BY {period}, h ORDER BY {period}, h;"""


def exam_summary_query(configuration) -> str:
    """Step 19: per method and horizon, the mean error of the total, series-by-series WAPE, the error
    against the noise and the share in the interval."""
    detail = table_in_sql(configuration, TABLE_SERIES_EXAM_DETAIL)
    period = configuration.period_col
    return f"""-- step 19 · the exam per method and horizon
WITH month_total AS (
    SELECT method, h, {period}, SUM(pred_units) / SUM(real_units) - 1 AS error_total
    FROM {detail} GROUP BY method, h, {period})
SELECT d.method AS metodo, d.h,
       (SELECT AVG(ABS(m.error_total)) FROM month_total m WHERE m.method = d.method AND m.h = d.h) AS error_total_medio,
       (SELECT AVG(m.error_total) FROM month_total m WHERE m.method = d.method AND m.h = d.h) AS sesgo_total_medio,
       SUM(ABS(d.err_units)) / SUM(d.real_units) AS wape_series,
       SQRT(AVG(d.err_pp * d.err_pp)) / NULLIF(SQRT(AVG(d.noise_pp * d.noise_pp)), 0) AS error_vs_ruido,
       AVG(d.in_band) AS en_intervalo
FROM {detail} d
GROUP BY d.method, d.h ORDER BY d.h, error_total_medio;"""
