# SFF · Informe de la ejecución
Mes en curso: **2026-09** · entrenamiento ≤ 2026-05 · examen 2026-06..2026-08 · proyeccion ≥ 2026-09

### Cifras principales

| indicador | valor |
|---|---|
| filas del extracto | 843 |
| forecast series | 16 |
| USD por predecir | $187,980 |
| USD con la tasa conocida a ±5 pp: antes → después de la escalera | 91% → 93% |
| error del TOTAL en el examen, h = 1: framework vs mejor hoja de cálculo | 2.1% vs 2.5% (hoja_mandatory) |
| error del TOTAL en el examen, h = 6: framework vs mejor hoja de cálculo | 2.2% vs 2.9% (hoja_mandatory) |
| error medio de la tasa por pool en el examen (corto) | 2.6 pp (retador 3.2 pp) |
| renovado 2026: real + esperado (± cuadratura) | $410,359 ± $2,791 |
| renovado 2027: real + esperado (± cuadratura) | $96,258 ± $2,604 |
| TOTAL 2026 renovado + revenue time_series (pipeline 558,330 $) | $422,599 |
| TOTAL 2027 renovado + revenue time_series (pipeline 137,639 $) | $101,477 |

## 1 · El raw y cómo lo mejoramos

El extracto tiene **843 filas × 18 columnas**, de 2023-01 a 2026-12. Cada columna tiene un rol declarado en la Config y el calendario se genera desde el mes en curso.

| rol | meses | filas | usd_vence |
|---|---|---|---|
| entrenamiento | 41 | 682 | 1,925,400 |
| examen | 3 | 47 | 138,600 |
| proyeccion | 4 | 66 | 187,980 |

**Comprobaciones de cada paso** (un fallo detiene la ejecución; un aviso se registra y sigue):

| paso | nombre | comprobaciones | ok | avisos | fallos | avisos_detalle |
|---|---|---|---|---|---|---|
| 00 | VALIDATE RAW | 10 | 10 | 0 | 0 |  |
| 00b | SPLIT | 2 | 2 | 0 | 0 |  |
| 01 | VALIDATE VALUES | 20 | 20 | 0 | 0 |  |
| 02 | APPLY CALENDAR | 10 | 10 | 0 | 0 |  |
| 03 | FINE TABLE | 5 | 5 | 0 | 0 |  |
| 04 | FORECAST UNITS | 5 | 5 | 0 | 0 |  |
| 05 | LOOKUPS | 5 | 5 | 0 | 0 |  |
| 06 | SERIES AND ROUTES | 4 | 4 | 0 | 0 |  |
| 07 | SUPPORT BOUND | 2 | 2 | 0 | 0 |  |
| 08 | RATE SERIES | 6 | 6 | 0 | 0 |  |
| 09 | DIMENSIONS | 6 | 6 | 0 | 0 |  |
| 10 | LADDER GROUPS | 7 | 7 | 0 | 0 |  |
| 11 | LADDER | 6 | 6 | 0 | 0 |  |
| 12 | POOL SERIES | 5 | 5 | 0 | 0 |  |
| 13 | DYNAMICS OF THE RATE | 4 | 4 | 0 | 0 |  |
| 14 | BACKTEST OF THE RATE | 11 | 11 | 0 | 0 |  |
| 15 | UPLIFT | 5 | 5 | 0 | 0 |  |
| 16 | BACKTEST OF THE UPLIFT | 3 | 3 | 0 | 0 |  |
| 17 | FORECAST | 8 | 8 | 0 | 0 |  |
| 19 | EXAM OF THE PORTFOLIO | 5 | 5 | 0 | 0 |  |
| 20 | TIME SERIES UNIVERSE AND TOTAL | 6 | 6 | 0 | 0 |  |
| NU | CORE TABLE | 7 | 7 | 0 | 0 |  |
| 18 | VALIDATION | 7 | 7 | 0 | 0 |  |

**Lo que se corrigió o completó en el raw:**

- Renovaciones nulas en meses cerrados leídas como 0 (nadie renovó): **0 filas**.
- Resultados adelantados borrados desde el mes en curso (el futuro no ha empezado): **14 filas**, 328 unidades y $10,079. El raw original queda en las columnas s0_.
- Pipeline de licencias de 1 año vendidas o renovadas desde el mes en curso (vence desde 2027-09): **aún no se conoce**, se borra y se proyecta: **0 filas**, $0 (el raw la conserva en s0_vencen_*).
- Descuento: tramo derivado del descuento exacto con los cortes [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] %; **4.5%** de las filas sin dato (tramo `sin_dato`).
- Grano: 795 filas = 795 filas finas distintas (unidad + valores de revalorización + descuento exacto); el dinero se conserva al agregar a 699 forecast units.

## 2 · Huecos rellenados

Un hueco es un mes sin vencimientos DENTRO de la historia de una serie estimable. Un mes sin vencimientos no dice nada de la tasa: su tasa queda **nula, nunca 0 %**, y el mes aparece como fila explícita con medidas a 0 para que la historia de la serie esté completa y las sumas sigan cuadrando.

- Huecos añadidos: **11** en **1 series**, el 1.7% de los meses de historia.

| fs_id | meses_historia | huecos | n_propio | usd_por_predecir |
|---|---|---|---|---|
| NA|B|0|0|0|0|tele | 44 | 11 | 40 | 3,600 |


## 3 · El soporte binomial, antes y después de la escalera

La tasa de un mes es k renovaciones de n contratos: aunque nada cambie, oscila por azar (error binomial √(p(1−p)/n)). Con **30** contratos al mes una serie tiene evidencia para prestar; con **271** su tasa se conoce a ±5 pp y puede ir sola. **Antes**: cada serie con su propio soporte. **Después**: la escalera junta las series pasada a pasada (signo, extras y dimensiones mandatory en el orden de colapso) hasta que cada grupo llega a 30; el grupo presta su tasa a sus series y, por debajo de 271, la mezcla con la de una referencia más amplia por credibilidad. Ver `DOC_escalera.md`.

**Antes · el dinero por soporte propio (el dial):**

| soporte | series | usd_por_predecir | pct_usd |
|---|---|---|---|
| 0 · sin historia propia | 1 | 2,700.00 | 1.4% |
| 1 · < 30 (no fiable solo) | 10 | 9,840.00 | 5.2% |
| 2 · 30-271 (evidencia, sin precisión) | 1 | 3,600.00 | 1.9% |
| 3 · ≥ 271 (precisa sola: ±5 pp) | 4 | 171,840.00 | 91.4% |

**Antes y después · el error con el que se CONOCE la tasa de cada serie** (antes: su error binomial con su propio soporte; después: el error de la estimación de la escalera). La predicción de un mes concreto conserva además el ruido de su propio tamaño, que ninguna escalera elimina: está en el nivel de riesgo.

| medida | antes_pp | despues_pp |
|---|---|---|
| error de la tasa (90 %), ponderado por USD | 5.3 | 4.6 |
| % del USD con la tasa conocida a ±5 pp | 91.4 | 93.3 |

**Pasada a pasada · cómo mejora el soporte** (cada pasada es un reparto: las unidades que vencen suman lo mismo en todas; los grupos son menos y más grandes; pct_usd_floor / pct_usd_own_rate: dinero por predecir en grupos que llegan a 30 / a 271):

| ladder_step | step_name | groups | open_groups | median_group_support | units_due | pct_usd_floor | pct_usd_own_rate |
|---|---|---|---|---|---|---|---|
| 0 | itself | 14 | 9 | 12.00 | 72,652.00 | 94.7% | 92.7% |
| 1 | sign | 11 | 5 | 40.00 | 72,652.00 | 97.4% | 92.7% |
| 2 | extra channel | 11 | 5 | 40.00 | 72,652.00 | 97.4% | 92.7% |
| 3 | without region | 10 | 4 | 41.00 | 72,652.00 | 97.4% | 92.7% |
| 4 | without product | 9 | 2 | 42.00 | 72,652.00 | 99.4% | 92.7% |

**Después · el dinero por nivel de riesgo** (error_pp: error de predicción del mes siguiente, ponderado por dinero):

| nivel_riesgo | series | usd_por_predecir | pct_usd | error_pp |
|---|---|---|---|---|
| A3_propio_reforzado | 1 | 3,600.00 | 1.9% | 8.64 |
| A_propio | 4 | 171,840.00 | 91.4% | 3.03 |
| B_prestado | 4 | 5,040.00 | 2.7% | 16.60 |
| C_lejano | 3 | 3,600.00 | 1.9% | 13.69 |
| D_sin_historia | 1 | 2,700.00 | 1.4% |  |
| M_signo_mixto | 1 | 480.00 | 0.3% | 34.52 |
| N_sin_impacto | 1 | 0.00 | 0.0% |  |
| S_signo_bajo_suelo | 1 | 720.00 | 0.4% | 12.03 |


## 4 · La dinámica de la tasa: ¿hay algo más que ruido?

φ compara lo que varía la tasa mes a mes con lo que variaría solo por muestreo: φ ≈ 1, nada que modelar (la media es la mejor técnica); φ > 1, algo la mueve (tendencia, estación, cambio de nivel o de mezcla). La estacionalidad se prueba sobre la tasa sin su tendencia (prueba F del mes del año, 5 %). Es descriptivo: no restringe ninguna técnica; el backtest decide.

**Cartera completa:** **hay efecto mes** más allá del ruido (p = 0.001, amplitud 3.8 pp, consistencia entre mitades 0.76); tendencia -3.0 pp/año (p = 0.000); φ 10.9. Un φ de cartera alto con tendencia suele ser cambio de mezcla, no comportamiento.

| mes | efecto_pp | error_pp | meses_observados | tasa_media |
|---|---|---|---|---|
| 1 | 0.28 | 0.62 | 4 | 0.73 |
| 2 | 1.51 | 0.62 | 4 | 0.74 |
| 3 | 0.85 | 0.62 | 4 | 0.74 |
| 4 | 1.76 | 0.62 | 4 | 0.75 |
| 5 | 1.00 | 0.62 | 4 | 0.74 |
| 6 | 0.20 | 0.62 | 4 | 0.73 |
| 7 | -0.75 | 0.62 | 4 | 0.72 |
| 8 | 0.03 | 0.62 | 4 | 0.73 |
| 9 | -1.51 | 0.71 | 3 | 0.71 |
| 10 | -1.95 | 0.71 | 3 | 0.71 |
| 11 | -2.02 | 0.71 | 3 | 0.71 |
| 12 | -1.08 | 0.71 | 3 | 0.72 |

**Los pools con soporte, uno a uno:**

| final_group_id | meses | phi | tendencia_pp_ano | estacional | amplitud_pp | meses_alto | meses_bajo | usd_por_predecir |
|---|---|---|---|---|---|---|---|---|
| EU|A|0|0|0|0|web | 44 | 5.11 | -0.25 | 1 | 11.70 | 2,3,4,5 | 7,9,10,11,12 | 63,360.00 |
| NA|B|0|0|0|0|web | 44 | 1.52 | -0.40 | 0 | 5.67 |  |  | 57,600.00 |
| EU|B|0|0|0|0|web | 44 | 8.40 | -6.21 | 0 | 3.60 |  |  | 31,680.00 |
| NA|A|0|0|0|0|web | 44 | 1.72 | -0.36 | 0 | 4.52 | 7 |  | 19,200.00 |
| EU|A|SIG=negativo|web | 44 | 1.40 | 1.85 | 0 | 14.53 |  |  | 5,040.00 |
| *|*|SIG=neutro|* | 44 | 1.61 | -2.39 | 0 | 13.73 |  |  | 3,600.00 |
| NA|B|0|0|0|0|tele | 33 | 1.07 | -1.73 | 0 | 10.06 |  |  | 3,600.00 |


## 5 · Qué tal se predice la tasa de renovación

**Cómo se mide.** Cada pool con soporte se predice en meses que ya ocurrieron, sin mirar el futuro: para el mes T a horizonte h, cada técnica solo ve hasta T − h. Los meses de **selección** (los 6 anteriores al examen) eligen la técnica; los meses de **examen** (2026-06 a 2026-08) la miden sin que la haya visto. El error se compara con el ruido binomial del mes (err_norm ≈ 1: tan cerca como permite el azar). Una técnica sustituye al retador (T3_ma3) solo si le gana por un margen. Compiten todas las técnicas que la historia permite, también las de series temporales.

Los pools juzgados cubren el **99%** del dinero por predecir; el resto toma el retador.

**Ranking en los meses de selección** (error normalizado medio):

| tramo_h | tecnica | err_norm_medio |
|---|---|---|
| corto | T15_level_seasonal | 1.12 |
| corto | T12_theta | 1.19 |
| corto | T3_ma3 | 1.22 |
| corto | T11_holt_winters | 1.22 |
| corto | T9_ses | 1.24 |
| corto | T4_ewma | 1.25 |
| corto | T10_holt_damped | 1.26 |
| corto | T3_ma6 | 1.33 |
| corto | T14_temporal_cred | 1.44 |
| corto | T2_mean | 1.58 |
| medio_largo | T12_theta | 1.19 |
| medio_largo | T10_holt_damped | 1.28 |
| medio_largo | T9_ses | 1.30 |
| medio_largo | T4_ewma | 1.33 |
| medio_largo | T3_ma6 | 1.33 |
| medio_largo | T3_ma3 | 1.34 |
| medio_largo | T15_level_seasonal | 1.35 |
| medio_largo | T11_holt_winters | 1.45 |
| medio_largo | T14_temporal_cred | 1.47 |
| medio_largo | T2_mean | 1.64 |

**Elecciones:**

| tramo_h | tecnica_origen | ids |
|---|---|---|
| corto | campeon | 4 |
| corto | retador | 3 |
| corto | sin_soporte | 2 |
| medio_largo | campeon | 7 |
| medio_largo | sin_soporte | 2 |

**Precisión en el examen por tramo** (error medio de la tasa en pp, ponderado por dinero; dentro_banda: proporción de errores dentro de la banda del 90 %):

| tramo | pools | error_elegida_pp | error_retador_pp | sesgo_elegida_pp | dentro_banda |
|---|---|---|---|---|---|
| corto | 7 | 2.63 | 3.22 | 1.56 | 95.2% |
| medio_largo | 7 | 2.28 | 2.97 | 0.93 | 100.0% |

**Precisión en el examen por nivel de riesgo** (tramo corto):

| nivel_riesgo | series | usd_por_predecir | error_elegida_pp | error_retador_pp |
|---|---|---|---|---|
| A3_propio_reforzado | 1 | 3,600.00 | 9.31 | 8.42 |
| A_propio | 4 | 171,840.00 | 2.16 | 2.89 |
| B_prestado | 4 | 5,040.00 | 9.78 | 9.78 |
| C_lejano | 3 | 3,600.00 | 8.38 | 4.73 |

**La cartera en el examen, serie a serie: el framework frente a la hoja de cálculo** (paso 19; cada serie predicha como la predice el forecast, con solo lo que se sabía h meses antes; la hoja: la tasa de los últimos 12 meses por grano × la pipeline real; error_total: de la suma de la cartera; wape_series: serie a serie, sin compensaciones):

| metodo | h | error_total_medio | sesgo_total_medio | wape_series |
|---|---|---|---|---|
| framework | 1 | 0.021 | 0.021 | 0.042 |
| hoja_mandatory | 1 | 0.025 | 0.019 | 0.051 |
| hoja_global | 1 | 0.033 | 0.033 | 0.208 |
| framework | 6 | 0.022 | 0.012 | 0.038 |
| hoja_mandatory | 6 | 0.029 | 0.029 | 0.056 |
| hoja_global | 6 | 0.054 | 0.054 | 0.209 |

| mes | h | renovadas_reales | framework_pred | framework_error_total | hoja_mandatory_pred | hoja_mandatory_error_total | hoja_global_pred | hoja_global_error_total |
|---|---|---|---|---|---|---|---|---|
| 2026-06 | 1 | 1,146.000 | 1,145.673 | -0.000 | 1,136.297 | -0.008 | 1,154.864 | 0.008 |
| 2026-06 | 6 | 1,146.000 | 1,127.191 | -0.016 | 1,147.182 | 0.001 | 1,178.076 | 0.028 |
| 2026-07 | 1 | 1,059.000 | 1,109.314 | 0.048 | 1,111.972 | 0.050 | 1,122.374 | 0.060 |
| 2026-07 | 6 | 1,059.000 | 1,091.968 | 0.031 | 1,121.604 | 0.059 | 1,143.632 | 0.080 |
| 2026-08 | 1 | 1,109.000 | 1,126.161 | 0.015 | 1,126.293 | 0.016 | 1,143.761 | 0.031 |
| 2026-08 | 6 | 1,109.000 | 1,131.004 | 0.020 | 1,140.085 | 0.028 | 1,169.878 | 0.055 |


## 6 · La revalorización: a qué precio se renueva

El uplift es lo que paga quien renueva respecto a lo que vencía (1,00 = mismo precio). Dos vías: la **estadística** (el uplift observado en las renovaciones pasadas de su celda de uplift: dimensiones mandatory, extras de revalorización y tramo de descuento; con menos de 30 renovadores toma el de su padre) y la del **contrato** (quien pagó con descuento d renueva a lista: 1 / (1 − d)).

**Las celdas por origen de su uplift:**

| uplift_origen | celdas | uplift_medio | renovadores |
|---|---|---|---|
| propia | 7 | 1.16 | 53,569.00 |

**La regla de contrato frente a las renovaciones pasadas con descuento conocido** (ratio_realizacion 1 = exacta):

| tramo | renovadores | usd_renovado | uplift_observado | uplift_regla | pct_usd_dentro_2pct | ratio_realizacion |
|---|---|---|---|---|---|---|
| 01 _ 0-10% | 43,615.000 | 1,349,009.365 | 1.031 | 1.000 | 30.0% | 1.031 |
| 05 _ 40-50% | 9,310.000 | 250,665.950 | 1.496 | 1.667 | 0.0% | 0.897 |

**El backtest del uplift** (meses de examen; precio de las renovaciones reales predicho por cada vía, con el uplift estadístico estimado ANTES del examen): la vía usada donde hay descuento es la **estadistica**.

| filas | via | renovadas_filas | usd_real | usd_pred | wape | sesgo |
|---|---|---|---|---|---|---|
| descuento conocido | estadistica | 46 | 100,997.370 | 100,090.927 | 0.017 | -0.009 |
| descuento conocido | contrato | 46 | 100,997.370 | 99,420.000 | 0.051 | -0.016 |


## 7 · El forecast en dinero

Cada fila futura: **USD que vence × tasa de su serie × uplift de su celda**. La tasa la predice la técnica elegida para el id de estimación de la serie a su horizonte (aprendiendo de todos los meses cerrados); una serie que toma prestado conserva su diferencia de nivel con el pool en proporción a su credibilidad (activado). Sin id de estimación: la tasa de su celda mandatory. Las bandas: **lineal** (todos los errores en el mismo sentido, el peor caso) y **cuadratura** (errores independientes); la verdad está entre ambas.

**La ventana de simulación** (del mes en curso, incluido, a diciembre): lo que ocurre en ella vence doce meses después. Las licencias de 1 año que vencen en la ventana renuevan como prevé el forecast y su renovación vence el mismo mes del año siguiente, con las mismas dimensiones y descuento 0 (**proyectada**). La captación de la ventana se simula por cada valor de captación (mismo mes del año anterior × nivel, valor medio de 12 meses, señales a 0, descuento 40%; **simulada**). La pipeline de las licencias de 1 año vendidas o renovadas desde el mes en curso aún no se conoce: el paso 02 la borra (el raw la conserva en s0_vencen_*) y se proyecta.

**Por mes:**

| period | usd_vence | esperado_usd | banda_lineal_baja | banda_lineal_alta | banda_cuadratura_baja | banda_cuadratura_alta | esperado_real | esperado_proyectada | esperado_simulada | tasa_usd |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09 | 46,620 | 33,084 | 29,514 | 36,302 | 31,727 | 34,212 | 33,084 | 0 | 0 | 1 |
| 2026-10 | 47,520 | 33,699 | 30,458 | 37,973 | 32,580 | 35,180 | 33,699 | 0 | 0 | 1 |
| 2026-11 | 46,320 | 32,721 | 29,669 | 36,736 | 31,616 | 34,183 | 32,721 | 0 | 0 | 1 |
| 2026-12 | 47,520 | 33,881 | 30,644 | 38,147 | 32,763 | 35,360 | 33,881 | 0 | 0 | 1 |
| 2027-09 | 33,084 | 23,872 | 21,038 | 27,509 | 22,882 | 25,166 | 0 | 23,872 | 0 | 1 |
| 2027-10 | 33,699 | 24,332 | 21,320 | 28,166 | 23,328 | 25,641 | 0 | 24,332 | 0 | 1 |
| 2027-11 | 32,721 | 23,427 | 20,548 | 27,079 | 22,430 | 24,726 | 0 | 23,427 | 0 | 1 |
| 2027-12 | 33,881 | 24,627 | 21,614 | 28,462 | 23,624 | 25,934 | 0 | 24,627 | 0 | 1 |

**Por año** (lo renovado en los meses cerrados + lo esperado en los futuros):

| ano | renovado_real_usd | esperado_real_usd | esperado_proyectada_usd | esperado_simulada_usd | esperado_usd | total_usd | banda_cuadratura_usd | banda_lineal_baja | banda_lineal_alta |
|---|---|---|---|---|---|---|---|---|---|
| 2,023 | 464,903 | 0 | 0 | 0 | 0 | 464,903 | 0 | 464,903 | 464,903 |
| 2,024 | 446,200 | 0 | 0 | 0 | 0 | 446,200 | 0 | 446,200 | 446,200 |
| 2,025 | 431,466 | 0 | 0 | 0 | 0 | 431,466 | 0 | 431,466 | 431,466 |
| 2,026 | 276,974 | 133,385 | 0 | 0 | 133,385 | 410,359 | 2,791 | 397,260 | 426,133 |
| 2,027 | 0 | 0 | 96,258 | 0 | 96,258 | 96,258 | 2,604 | 84,520 | 111,216 |

**De dónde sale la tasa de las filas futuras:**

| origen_tasa | filas | usd_vence | esperado_usd |
|---|---|---|---|
| celda_mandatory | 6 | 4,782 | 3,686 |
| pool | 126 | 316,583 | 225,957 |

**El total del forecast por año y origen** (paso 20; es la SUMA de `sff_nucleo` por `fin_ano` y `fin_origen`, comprobado en el núcleo: en Power BI, SUM(fin_renovado_usd) y SUM(fin_vence_usd)). Orígenes: renovaciones ya contabilizadas y esperadas de la pipeline real, reentradas y captación del horizonte extendido, y el universo time_series de retail a suscripción (ts_real y ts_proyectado cuentan como revenue del año, sin tasa; ts_reentrada es pipeline del año siguiente: comprado con descuento, renueva al 100 % con la tasa de su región). Total 2026 = renovaciones de la pipeline + ts_real + ts_proyectado · Total 2027 = forecast extendido + ts_reentrada. usd_vence: pipeline; usd_renovado: renovaciones o revenue:

| ano | origen | unidades_vencen | usd_vence | unidades_renovadas | usd_renovado |
|---|---|---|---|---|---|
| 2,026 | TOTAL | 19,907 | 558,330 | 13,912 | 422,599 |
| 2,026 | pipeline_real_esperado | 6,698 | 187,980 | 4,412 | 133,385 |
| 2,026 | pipeline_renovado_real | 13,209 | 370,350 | 9,106 | 276,974 |
| 2,026 | ts_proyectado | 0 | 0 | 137 | 4,254 |
| 2,026 | ts_real | 0 | 0 | 256 | 7,986 |
| 2,027 | TOTAL | 4,550 | 137,639 | 3,129 | 101,477 |
| 2,027 | pipeline_proyectada | 4,412 | 133,385 | 3,027 | 96,258 |
| 2,027 | ts_reentrada | 137 | 4,254 | 101 | 5,218 |

**El universo time_series, región × mes:**

| region | region_ts | period | unidades | valor | origen | revenue | nivel | valor_medio | por_cuota | mes_origen | descuento | valor_a_renovar | tasa | nivel_tasa | tasa_global | unidades_renovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EU | EU | 2026-01 | 25.00 | 752.08 | ts_real | 752.08 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-02 | 41.00 | 1,292.14 | ts_real | 1,292.14 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-03 | 32.00 | 1,001.66 | ts_real | 1,001.66 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-04 | 34.00 | 1,074.99 | ts_real | 1,074.99 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-05 | 29.00 | 902.71 | ts_real | 902.71 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-06 | 34.00 | 1,065.87 | ts_real | 1,065.87 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-07 | 34.00 | 1,070.00 | ts_real | 1,070.00 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-08 | 27.00 | 826.23 | ts_real | 826.23 |  |  |  | NaT |  |  |  |  |  |  |
| EU | EU | 2026-09 | 35.11 | 1,087.51 | ts_proyectado | 1,087.51 | 1.03 | 30.98 | 0.00 | NaT |  |  |  |  |  |  |
| EU | EU | 2026-10 | 35.11 | 1,087.51 | ts_proyectado | 1,087.51 | 1.03 | 30.98 | 0.00 | NaT |  |  |  |  |  |  |
| EU | EU | 2026-11 | 34.08 | 1,055.52 | ts_proyectado | 1,055.52 | 1.03 | 30.98 | 0.00 | NaT |  |  |  |  |  |  |
| EU | EU | 2026-12 | 33.04 | 1,023.54 | ts_proyectado | 1,023.54 | 1.03 | 30.98 | 0.00 | NaT |  |  |  |  |  |  |
| EU | EU | 2027-09 | 35.11 | 1,087.51 | ts_reentrada | 1,333.95 |  |  |  | 2026-09 | 0.40 | 1,812.51 | 0.74 | region | 0.00 | 25.84 |
| EU | EU | 2027-10 | 35.11 | 1,087.51 | ts_reentrada | 1,333.95 |  |  |  | 2026-10 | 0.40 | 1,812.51 | 0.74 | region | 0.00 | 25.84 |
| EU | EU | 2027-11 | 34.08 | 1,055.52 | ts_reentrada | 1,294.72 |  |  |  | 2026-11 | 0.40 | 1,759.20 | 0.74 | region | 0.00 | 25.08 |
| EU | EU | 2027-12 | 33.04 | 1,023.54 | ts_reentrada | 1,255.48 |  |  |  | 2026-12 | 0.40 | 1,705.89 | 0.74 | region | 0.00 | 24.32 |

**Validación final de la cadena:**

| estado | comprobacion | detalle |
|---|---|---|
| ok | Σ USD due: extract (without time_series and the pipeline not known yet) = fine table = forecast units | $2,251,980 |
| ok | the future USD due of the extract = Σ USD due of the forecast's extract rows | $187,980 (the extended horizon adds its own pipeline) |
| ok | every series with money to predict has a rate and a risk level | 15 series |
| ok | every future row of a series with an estimation id takes its rate from the pool | 126 of 132 rows from a pool |
| ok | the expected rate of the future is within ±15 pp of 2026's | future 71.0% (USD, uplift included) vs 2026 74.8% |
| ok | the error of the total renewals in the exam is within ±10% | worst month 4.8% |


## 8 · La ficha de cada serie y lo que falta

`sff_ficha_serie` tiene una fila por serie (16) con todo lo que el framework sabe de ella: ruta, soporte y tasa propios, grupo final y su referencia, credibilidad, tasa estimada y sus dos errores, nivel de riesgo, la dinámica y las técnicas de su id de estimación y su error en el examen. `sff_nucleo` tiene la misma información fila a fila con los meses (en Power BI: seleccionar `s03_fs_id`).

**Lo que falta** (siguientes pasos): el horizonte extendido (reentradas de 2026 y captación simulada de 2027, más allá de la pipeline que trae el extracto), y los análisis del bloque B (composición y mix, descuento y churn, maduración de las señales, escenarios de precio, baseline de la hoja de cálculo, top movers).
