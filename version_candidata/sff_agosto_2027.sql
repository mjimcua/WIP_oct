/* ═══════════════════════════════════════════════════════════════════════════════════
   SFF · ¿POR QUÉ CAE LA TASA PREVISTA EN 2027-08?
   Compara los meses que vencen 2027-05 … 2027-08 (pipeline del extracto, foto de hoy)
   con los mismos meses de 2026 (cerrados, marca final). Si 2027-08 es una cohorte
   recién vendida o renovada, sus marcas (not_installed, dormant, softcancel) están en
   su estado de los primeros días y su mezcla (adquisición frente a retención) es otra.
   ═══════════════════════════════════════════════════════════════════════════════════ */
SELECT period,
       forecast_status,
       SUM(forecast_to_renew_units)                                                          AS due_units,
       SUM(forecast_renewed_units) / SUM(forecast_to_renew_units)                            AS rate_units,
       -- the marks, as a share of the units due
       SUM(CASE WHEN softcancel = 1    THEN forecast_to_renew_units ELSE 0 END) / SUM(forecast_to_renew_units) AS share_softcancel,
       SUM(CASE WHEN dormant = 1       THEN forecast_to_renew_units ELSE 0 END) / SUM(forecast_to_renew_units) AS share_dormant,
       SUM(CASE WHEN not_installed = 1 THEN forecast_to_renew_units ELSE 0 END) / SUM(forecast_to_renew_units) AS share_not_installed,
       -- the mix: first renewal of an acquisition, or a retention
       SUM(CASE WHEN net_new IN ('Acquisition_Not-New', 'Acquisition_Pure-New')
                THEN forecast_to_renew_units ELSE 0 END) / SUM(forecast_to_renew_units)     AS share_acquisition,
       -- the term: is the month made of 1-year licences, or of multi-year ones
       SUM(CASE WHEN tr_term = '1 year' THEN forecast_to_renew_units ELSE 0 END) / SUM(forecast_to_renew_units) AS share_1_year
FROM kamelot.sff_nucleo
WHERE forecast_universe = 'pipeline'
  AND forecast_to_renew_units > 0
  AND period IN ('2026-05', '2026-06', '2026-07', '2026-08', '2027-05', '2027-06', '2027-07', '2027-08')
GROUP BY period, forecast_status
ORDER BY period;
