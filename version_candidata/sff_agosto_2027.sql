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

WITH licences AS (
    SELECT <tus columnas y dimensiones>,
           total_tr_units, total_tr_usd, total_renewed_units, total_renewed_usd,
           total_tr_usd_renewed, total_readquired_units,
           -- the isolated renewal, ONCE: your condition (softcancel, sku base key, same BU, licence type,
           -- Retention), with the amounts on the fixed-rate base
           CASE WHEN <tu condición de softcancel y SKU, con sus paréntesis>
                 AND total_tr_usd_renewed > 0
                 AND total_renewed_usd > 0
                THEN 1 ELSE 0 END                                      AS is_isolated_renewal
    FROM ...
),
licences_with_ratio AS (
    SELECT licences.*,
           -- the ratio depends on the flag: it cannot exist outside the isolated renewals
           CASE WHEN is_isolated_renewal = 1
                THEN total_renewed_usd / total_tr_usd_renewed END     AS isolated_renewal_ratio
    FROM licences
)
SELECT <tus dimensiones>,
       SUM(total_tr_units)          AS total_tr_units,
       SUM(total_tr_usd)            AS total_tr_usd,
       SUM(total_renewed_units)     AS total_renewed_units,
       SUM(total_renewed_usd)       AS total_renewed_usd,
       SUM(total_tr_usd_renewed)    AS total_tr_usd_renewed,
       SUM(total_readquired_units)  AS total_reacquired_units,

       -- the isolated renewals: renewed units and USD, and what they were worth (fixed rate)
       SUM(CASE WHEN is_isolated_renewal = 1 THEN total_renewed_units  ELSE 0 END) AS total_renewed_units_without_softcancel,
       SUM(CASE WHEN is_isolated_renewal = 1 THEN total_renewed_usd    ELSE 0 END) AS total_renewed_usd_without_softcancel,
       SUM(CASE WHEN is_isolated_renewal = 1 THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_without_softcancel,

       -- the second moment, on the same base
       SUM(CASE WHEN is_isolated_renewal = 1
                THEN total_renewed_usd * total_renewed_usd / total_tr_usd_renewed ELSE 0 END) AS total_renewed_usd_sq_over_tr_isolated,

       -- the six bands: what the isolated renewers were worth, by their ratio (they add up to the base above)
       SUM(CASE WHEN isolated_renewal_ratio <  0.95                                  THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_lt095,
       SUM(CASE WHEN isolated_renewal_ratio >= 0.95 AND isolated_renewal_ratio < 1.00 THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_095_100,
       SUM(CASE WHEN isolated_renewal_ratio >= 1.00 AND isolated_renewal_ratio < 1.05 THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_100_105,
       SUM(CASE WHEN isolated_renewal_ratio >= 1.05 AND isolated_renewal_ratio < 1.10 THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_105_110,
       SUM(CASE WHEN isolated_renewal_ratio >= 1.10 AND isolated_renewal_ratio < 1.20 THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_110_120,
       SUM(CASE WHEN isolated_renewal_ratio >= 1.20                                  THEN total_tr_usd_renewed ELSE 0 END) AS total_tr_usd_renewed_isolated_ge120
FROM licences_with_ratio
GROUP BY <tus dimensiones>;