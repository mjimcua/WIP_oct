/* ═══════════════════════════════════════════════════════════════════════════════════
   SFF · LOS DATOS DE LA PORTADA
   Una query por dato del comentario que acompaña al número forecasteado.

   Supuestos (ajustar si tu instalación es distinta):
     · esquema dbo, prefijo de tablas sff_ (config: sql_schema = "dbo", table_prefix = "sff_")
     · el bloque final del núcleo ya renombrado: forecast_year, forecast_status (actual / forecast),
       forecast_universe, forecast_pipeline_source, forecast_to_renew_*, forecast_renewed_*
     · la columna del descuento exacto se llama discount (config: discount_value_column)
   Comprobadas sobre la salida del sintético (SQLite, con la sintaxis equivalente);
   las cifras de los comentarios son las del sintético.
   ═══════════════════════════════════════════════════════════════════════════════════ */

-- El año en curso: el último año con meses cerrados (renovación real).
-- El año siguiente es el de la pipeline que el framework completa.
DECLARE @current_year INT = (SELECT MAX(forecast_year)
                             FROM dbo.sff_nucleo
                             WHERE forecast_status = 'actual');
DECLARE @next_year INT = @current_year + 1;


/* ───────────────────────────────────────────────────────────────────────────────────
   1. EL NÚMERO: el renovado del año (real + previsto) y sus bandas
      Sintético 2026: $422,481 · lineal $406,774 – $439,400 · cuadratura −$2,811 / +$3,037
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    forecast_year,
    SUM(forecast_renewed_USD)                                                     AS renewed_USD,
    SUM(CASE WHEN forecast_status = 'actual'   THEN forecast_renewed_USD ELSE 0 END) AS renewed_actual_USD,
    SUM(CASE WHEN forecast_status = 'forecast' THEN forecast_renewed_USD ELSE 0 END) AS renewed_forecast_USD,
    SUM(forecast_to_renew_USD)                                                    AS to_renew_USD,
    SUM(forecast_renewed_USD_low)                                                 AS band_linear_low_USD,
    SUM(forecast_renewed_USD_high)                                                AS band_linear_high_USD,
    SQRT(SUM(SQUARE(forecast_renewed_USD - forecast_renewed_USD_low)))            AS band_quadrature_minus_USD,
    SQRT(SUM(SQUARE(forecast_renewed_USD_high - forecast_renewed_USD)))           AS band_quadrature_plus_USD
FROM dbo.sff_nucleo
WHERE forecast_year IN (@current_year, @next_year)
GROUP BY forecast_year
ORDER BY forecast_year;


/* ───────────────────────────────────────────────────────────────────────────────────
   2. LA MAQUINARIA: X proyecciones elegidas de Y calculadas
      Una proyección candidata = una composición × una técnica × un mes de la ventana.
      Una prueba = una predicción contra el pasado (backtest de selección + examen).
      Sintético: 16 series · 11 composiciones · 10 técnicas · 8 meses ·
                 880 candidatas · 1,578 pruebas · Y = 2,458 · X = 88
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    (SELECT COUNT(*)                       FROM dbo.sff_forecast_series)           AS forecast_series,
    (SELECT COUNT(DISTINCT composition_id) FROM dbo.sff_composition_forecast_all)  AS compositions_projected,
    (SELECT COUNT(DISTINCT tecnica)        FROM dbo.sff_composition_forecast_all)  AS techniques_compared,
    (SELECT COUNT(DISTINCT h)              FROM dbo.sff_composition_forecast_all)  AS months_projected,
    (SELECT COUNT(*)                       FROM dbo.sff_composition_forecast_all)  AS candidate_projections,
    (SELECT COUNT(*)                       FROM dbo.sff_backtest_predicciones)     AS backtest_predictions,
    (SELECT COUNT(*) FROM dbo.sff_composition_forecast_all)
      + (SELECT COUNT(*) FROM dbo.sff_backtest_predicciones)                       AS projections_computed_Y,
    (SELECT SUM(is_chosen)                 FROM dbo.sff_composition_forecast_all)  AS projections_chosen_X;

-- 2b. Las pruebas, por propósito: selección (elegir la técnica) y examen (medir el error de la elegida)
--     Sintético: seleccion 1,058 · examen 520
SELECT proposito, COUNT(*) AS backtest_predictions
FROM dbo.sff_backtest_predicciones
GROUP BY proposito;


/* ───────────────────────────────────────────────────────────────────────────────────
   3. LA PIPELINE DEL AÑO QUE VIENE, POR ORIGEN, con su peso
      Lo que el framework proyecta para completarla: re-renovación de licencias de 1 año
      (pipeline_proyectada), captación simulada (pipeline_simulada) y retail a suscripción
      que vuelve a vencer (ts_reentrada). El resto ya está en el extracto.
      Sintético 2027: pipeline $137,521 · re-renovación $133,267 (97 %) · retail reentrada $4,254 (3 %)
      (el sintético no trae pipeline 2027 en el extracto ni genera captación simulada)
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    nucleo.forecast_pipeline_source,
    source.source_label,
    source.source_block,
    SUM(nucleo.forecast_to_renew_units)                                            AS to_renew_units,
    SUM(nucleo.forecast_to_renew_USD)                                              AS to_renew_USD,
    SUM(nucleo.forecast_renewed_USD)                                               AS renewed_USD,
    SUM(nucleo.forecast_to_renew_USD)
      / NULLIF(SUM(SUM(nucleo.forecast_to_renew_USD)) OVER (), 0)                  AS share_of_pipeline
FROM dbo.sff_nucleo AS nucleo
JOIN dbo.sff_forecast_pipeline_source AS source
  ON source.forecast_pipeline_source = nucleo.forecast_pipeline_source
WHERE nucleo.forecast_year = @next_year
  AND source.source_block <> 'Control'
GROUP BY nucleo.forecast_pipeline_source, source.source_label, source.source_block, source.source_order
ORDER BY source.source_order;


/* ───────────────────────────────────────────────────────────────────────────────────
   4. CUÁNTO SE HA PROYECTADO PARA COMPLETAR LA PIPELINE DEL AÑO QUE VIENE (y qué % es)
      Sintético 2027: $137,521 proyectados = 100 % de la pipeline
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    SUM(forecast_to_renew_USD)                                                     AS pipeline_next_year_USD,
    SUM(CASE WHEN forecast_pipeline_source IN ('pipeline_proyectada', 'pipeline_simulada', 'ts_reentrada')
             THEN forecast_to_renew_USD ELSE 0 END)                                AS projected_by_framework_USD,
    SUM(CASE WHEN forecast_pipeline_source IN ('pipeline_proyectada', 'pipeline_simulada', 'ts_reentrada')
             THEN forecast_to_renew_USD ELSE 0 END)
      / NULLIF(SUM(forecast_to_renew_USD), 0)                                      AS projected_share
FROM dbo.sff_nucleo
WHERE forecast_year = @next_year;


/* ───────────────────────────────────────────────────────────────────────────────────
   5. LA CAPTACIÓN SIMULADA (para completar el año en curso)
      La captación de la ventana no es renovación: no suma al forecast de este año. Vive
      como la pipeline que vencerá 12 meses después, a su valor de venta (con el descuento
      de captación). Por eso se lee en el año siguiente.
      Sintético: 0 filas (no trae pipeline más allá de 2026-12; ver nota en la respuesta)
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    SUM(CASE WHEN forecast_pipeline_source = 'pipeline_simulada'
             THEN forecast_to_renew_units ELSE 0 END)                              AS acquisition_simulated_units,
    SUM(CASE WHEN forecast_pipeline_source = 'pipeline_simulada'
             THEN forecast_to_renew_USD ELSE 0 END)                                AS acquisition_simulated_USD,
    SUM(CASE WHEN forecast_pipeline_source = 'pipeline_simulada'
             THEN forecast_renewed_USD ELSE 0 END)                                 AS its_expected_renewal_USD,
    SUM(CASE WHEN forecast_pipeline_source = 'pipeline_simulada'
             THEN forecast_to_renew_USD ELSE 0 END)
      / NULLIF(SUM(forecast_to_renew_USD), 0)                                      AS share_of_pipeline_next_year
FROM dbo.sff_nucleo
WHERE forecast_year = @next_year;


/* ───────────────────────────────────────────────────────────────────────────────────
   6. EL DESCUENTO
      6a. En cuánta pipeline conocemos el descuento exacto (columna discount no nula).
          Solo filas del extracto con algo que vence: los huecos y las filas sintéticas
          no tienen descuento propio.
          Sintético 2026: 100 % (el sintético trae el descuento en todas las filas)
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    forecast_year,
    SUM(forecast_to_renew_units)                                                   AS to_renew_units,
    SUM(CASE WHEN discount IS NOT NULL THEN forecast_to_renew_units ELSE 0 END)    AS with_discount_units,
    SUM(CASE WHEN discount IS NOT NULL THEN forecast_to_renew_units ELSE 0 END)
      / NULLIF(SUM(forecast_to_renew_units), 0)                                    AS with_discount_share_units,
    SUM(CASE WHEN discount IS NOT NULL THEN forecast_to_renew_USD ELSE 0 END)
      / NULLIF(SUM(forecast_to_renew_USD), 0)                                      AS with_discount_share_USD
FROM dbo.sff_nucleo
WHERE forecast_year IN (@current_year, @next_year)
  AND origen_fila = 'raw'
  AND forecast_to_renew_units > 0
GROUP BY forecast_year
ORDER BY forecast_year;

/* 6b. En cuánto del forecast se aplica el ajuste por descuento (uplift, paso 15):
       el renovado previsto cuya fila lleva s15_via_uplift.
       Sintético: 2026 $133,267 de $133,267 previstos (100 %) · 2027 $96,099 (100 %)
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    forecast_year,
    SUM(forecast_renewed_USD)                                                      AS renewed_forecast_USD,
    SUM(CASE WHEN s15_via_uplift IS NOT NULL THEN forecast_renewed_USD ELSE 0 END) AS with_uplift_USD,
    SUM(CASE WHEN s15_via_uplift IS NOT NULL THEN forecast_renewed_USD ELSE 0 END)
      / NULLIF(SUM(forecast_renewed_USD), 0)                                       AS with_uplift_share
FROM dbo.sff_nucleo
WHERE forecast_year IN (@current_year, @next_year)
  AND forecast_status = 'forecast'
  AND forecast_universe = 'pipeline'
GROUP BY forecast_year
ORDER BY forecast_year;


/* ───────────────────────────────────────────────────────────────────────────────────
   7. EL RETAIL A SUSCRIPCIÓN (universo time_series)
      Revenue del año en curso: real (meses cerrados) + proyectado (hasta diciembre);
      del año siguiente: lo proyectado que vuelve a vencer y su renovación.
      Con su peso en el renovado de cada año.
      Sintético: 2026 real $7,986 + proyectado $4,254 = 2.9 % del renovado ·
                 2027 reentrada $4,254 que vence → $5,218 renovado = 5.2 % del renovado
   ─────────────────────────────────────────────────────────────────────────────────── */
SELECT
    forecast_year,
    SUM(CASE WHEN forecast_pipeline_source = 'ts_real'      THEN forecast_renewed_USD ELSE 0 END) AS retail_actual_USD,
    SUM(CASE WHEN forecast_pipeline_source = 'ts_proyectado' THEN forecast_renewed_USD ELSE 0 END) AS retail_projected_USD,
    SUM(CASE WHEN forecast_pipeline_source = 'ts_reentrada'  THEN forecast_to_renew_USD ELSE 0 END) AS retail_reentry_to_renew_USD,
    SUM(CASE WHEN forecast_pipeline_source = 'ts_reentrada'  THEN forecast_renewed_USD ELSE 0 END) AS retail_reentry_renewed_USD,
    SUM(CASE WHEN forecast_universe = 'time_series' THEN forecast_renewed_USD ELSE 0 END)
      / NULLIF(SUM(forecast_renewed_USD), 0)                                       AS retail_share_of_renewed
FROM dbo.sff_nucleo
WHERE forecast_year IN (@current_year, @next_year)
GROUP BY forecast_year
ORDER BY forecast_year;


/* ───────────────────────────────────────────────────────────────────────────────────
   8. EL COMENTARIO, montado en una sola fila (SQL Server 2017+: CONCAT_WS / FORMAT)
      Para validar el texto antes de llevarlo a Power BI. (Sintaxis solo de SQL Server: no probada.)
   ─────────────────────────────────────────────────────────────────────────────────── */
WITH machinery AS (
    SELECT
        (SELECT COUNT(DISTINCT composition_id) FROM dbo.sff_composition_forecast_all) AS compositions,
        (SELECT COUNT(DISTINCT tecnica)        FROM dbo.sff_composition_forecast_all) AS techniques,
        (SELECT COUNT(DISTINCT h)              FROM dbo.sff_composition_forecast_all) AS months,
        (SELECT SUM(is_chosen)                 FROM dbo.sff_composition_forecast_all) AS chosen,
        (SELECT COUNT(*)                       FROM dbo.sff_backtest_predicciones)    AS tests,
        (SELECT COUNT(*) FROM dbo.sff_composition_forecast_all)
          + (SELECT COUNT(*) FROM dbo.sff_backtest_predicciones)                      AS computed
),
money AS (
    SELECT
        SUM(CASE WHEN forecast_year = @current_year THEN forecast_renewed_USD ELSE 0 END)  AS renewed_current_year,
        SUM(CASE WHEN forecast_year = @next_year    THEN forecast_to_renew_USD ELSE 0 END) AS pipeline_next_year,
        SUM(CASE WHEN forecast_year = @next_year
                  AND forecast_pipeline_source IN ('pipeline_proyectada', 'pipeline_simulada', 'ts_reentrada')
                 THEN forecast_to_renew_USD ELSE 0 END)                                    AS projected_next_year,
        SUM(CASE WHEN forecast_year = @next_year AND forecast_pipeline_source = 'pipeline_simulada'
                 THEN forecast_to_renew_USD ELSE 0 END)                                    AS acquisition_simulated,
        SUM(CASE WHEN forecast_year = @current_year AND forecast_universe = 'time_series'
                 THEN forecast_renewed_USD ELSE 0 END)                                     AS retail_current_year
    FROM dbo.sff_nucleo
)
SELECT CONCAT(
    'El renovado de ', @current_year, ' (', FORMAT(money.renewed_current_year, 'C0', 'en-US'), ') no es un forecast: ',
    'es la suma de ', FORMAT(machinery.chosen, 'N0'), ' proyecciones elegidas (', machinery.compositions, ' composiciones × ',
    machinery.months, ' meses), cada una seleccionada entre ', machinery.techniques, ' técnicas tras ',
    FORMAT(machinery.tests, 'N0'), ' pruebas contra el pasado; en total, ', FORMAT(machinery.computed, 'N0'),
    ' proyecciones calculadas. Incluye ', FORMAT(money.retail_current_year, 'C0', 'en-US'),
    ' de retail a suscripción. De la pipeline de ', @next_year, ' (', FORMAT(money.pipeline_next_year, 'C0', 'en-US'), '), ',
    FORMAT(money.projected_next_year / NULLIF(money.pipeline_next_year, 0), 'P0'), ' lo proyecta el framework, ',
    'con ', FORMAT(money.acquisition_simulated, 'C0', 'en-US'), ' de captación simulada.'
) AS cover_comment
FROM machinery CROSS JOIN money;
