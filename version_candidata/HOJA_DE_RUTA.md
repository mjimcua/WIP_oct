# SFF · Versión espejo — hoja de ruta

La versión espejo se construye paso a paso tomando como referencia el repo `WIP_oct`
(el orden de llamadas de `pipeline.run_analysis`, detallado en `PASOS_SFF.md`). En cada
paso: se revisa la lógica de la referencia y su documentación, se decide qué cambia, se
escribe el código mínimo, su test, y se añade una línea a `main.run()`. `config.py`
solo gana los parámetros que el paso necesita.

| Paso | Qué hace | Referencia en `WIP_oct` | Estado |
|---|---|---|---|
| 00 | Estructura: columnas (filas, repetidas, rol, presentes) y meses (se leen, el calendario cabe) | `config.validate_column_contract`, `raw_data_validation.validate_raw` [1-2] | **hecho** |
| 01 | Contenido, en un solo paso (`step_01_values_and_levels.py`): separa el universo time_series (sus filas solo llevan región y resultado; van al paso 20), valida los valores del universo de renovaciones (dinero, dimensiones, señales 0/1, descuento en [0,1], avisos) y genera `<columna>_level_1` de las dimensiones de `leveled_dims` (JSON reutilizable) · escribe `sff_dimension_levels`, `sff_dimension_level_values` | `raw_data_validation.validate_raw` [3-5] | **hecho** |
| 02 | Aplicar el calendario: rol y mes en curso generados, `s0_*`, limpiar la proyección, renovación nula en mes cerrado → 0 (contada) · pipeline de licencias de 1 año vendidas o renovadas desde el mes en curso (vence desde mes en curso + 12) borrada: aún no se conoce, se proyecta (el raw queda en s0_vencen_*) · escribe `sff_calendario` | `apply_current_month_doctrine` | **hecho** |
| 03 | Tabla fina: tramo de descuento derivado del descuento exacto (cortes de la Config, nulo → sin_dato), fs_id, fu_id, uplift_cell_id (con el tramo), fila_key (con el descuento exacto) y sus claves, grano comprobado con la SQL que lo reproduce · escribe `sff_fact_fine` | `build_fine_table` | **hecho** |
| 04 | Forecast units: una fila por serie × mes, medidas sumadas, dinero conservado · escribe `sff_fact_fu` | `aggregate_to_forecast_units` | **hecho** |
| 05 | Lookups id ↔ clave con sus atributos, toda clave de los hechos resuelta · escribe `sff_lookup_fs`, `sff_lookup_fu`, `sff_lookup_uplift_cell` | `build_key_lookups` | **hecho** |
| 06 | Una fila por serie: meses por rol, cobertura, ruta (predecible · solo_historia · solo_futuro), universo, USD por predecir · escribe `sff_series` | `label_universe_and_routes` | **hecho** |
| 07 | Cota de soporte por unidad: error binomial en el peor caso (p = 0,5) en pp y en $ · escribe `sff_fu_soporte` | `support_reference` | **hecho** |
| 08 | Series de tasa: huecos dentro de la historia (tasa nula), tasa de cada unidad solo donde es verdad, resumen por serie (n_propio, tasa propia, error de Wilson, signo) · escribe `sff_fu_huecos`, `sff_series_tasa` | `run_rate_series` | **hecho** |
| 09 | Qué dimensiones separan la tasa (η², contribución única, ω², pares) y orden de colapso de las mandatory (pérdida secuencial de R², jerarquías respetadas) · escribe `sff_decision_eta2`, `sff_decision_eta2_pares` | `analysis_dimensions.run_dimension_analysis` | **hecho** |
| 10 | Escalera como reparto: pasadas en orden (signo → extras → mandatory según el paso 09); en cada pasada los grupos con soporte ≥ 30 se cierran y conservan su id, los demás suben; cada pasada suma lo mismo que el raw. Referencia de credibilidad para los grupos < 271 · escribe `sff_ladder_steps`, `sff_ladder_summary`, `sff_ladder_groups` · ver `DOC_escalera.md` | nuevo modelo | **hecho** |
| 11 | El grupo presta su tasa; por debajo de 271 se mezcla con la referencia (z = n/(n+k)); errores y niveles de riesgo por la pasada en que se formó el grupo · escribe `sff_series_estimacion`, `sff_niveles_riesgo` | nuevo modelo | **hecho** |
| 12 | Serie mensual de cada id de estimación (suma de TODAS las series que encajan en su patrón) y su ficha: meses, soporte, tasa, series que lo usan, dinero, gate (nivel / soporte) · escribe `sff_pool_serie`, `sff_pool_referencia` | `analysis_backtest.monthly_series_by_estimation_id`, `build_pool_reference` | **hecho** |
| 13 | Dinámica de la tasa (descriptiva, NO restringe técnicas): φ, tendencia, estacionalidad sobre la tasa sin tendencia (prueba F), meses alto/bajo, consistencia; perfil mensual de la cartera · escribe `sff_dinamica_pool`, `sff_estacionalidad_cartera` | `analysis_seasonality_benchmark` (reorientado) | **hecho** |
| 14 | Backtest de la tasa: meses de SELECCIÓN (6 antes del examen) para elegir y meses de EXAMEN para medir; sin mirar el futuro (T − h); compiten todas las técnicas que la historia permite (también Holt-Winters y nivel + efecto mes); error normalizado por el binomial; campeón frente al retador ma3 con margen; bandas por cuantiles; precisión por pool y del total · escribe `sff_dim_tecnica`, `sff_backtest_predicciones`, `sff_decision_tecnica`, `sff_decision_bandas`, `sff_backtest_examen`, `sff_backtest_examen_total` | `analysis_backtest`, `techniques` | **hecho** |
| 15 | Revalorización: uplift estadístico por celda (Σ renovado $ / Σ renovadas × AUV vencida; padre y celda si faltan renovadores; banda bootstrap) y comprobación de la regla de contrato 1/(1 − d) · escribe `sff_uplift_celda`, `sff_uplift_contrato_check` | `run_uplift.run_uplift` | **hecho** |
| 16 | Backtest del uplift en los meses de examen (precio de las renovaciones reales, estadístico estimado antes del examen): su veredicto decide la vía donde hay descuento · escribe `sff_backtest_uplift` | `run_uplift.backtest_uplift` | **hecho** |
| 17 | Forecast en dinero: fila a fila, USD que vence × tasa × uplift, bandas; totales por mes y año. VENTANA DE SIMULACIÓN (mes en curso incluido → diciembre): lo que ocurre en ella vence 12 meses después — renovaciones de licencias de 1 año (`term_column` = `one_year_term_value`; mismas dims, valor renovado, descuento 0; proyectada), captación por cada valor de `acquisition_column` (nivel × mismo mes del año anterior, valor medio de 12 meses, señales a 0, `acquisition_discount`; simulada) · escribe `sff_forecast`, `sff_forecast_mes`, `sff_resumen_negocio` | `run_forecast_assembly`, `answers` | **hecho** |
| 18 | Validación final de la cadena: dinero del raw al forecast, cobertura, coherencia con el pasado y con el examen · escribe `sff_validacion` | `run_validation` | **hecho** |
| 19 | Examen de la cartera serie a serie (como predice el forecast, solo con lo sabido h meses antes) frente a la hoja de cálculo de negocio (tasa de los últimos N meses por grano) · escribe `sff_examen_cartera`, `sff_examen_cartera_resumen`. Sustituye al 'total' del paso 14, que sumaba pools solapados | `analysis_baseline` (reorientado) | **hecho** |
| 20 | Universo time_series (retail a suscripción): sus filas (los niveles regionales y el resultado) se separan tras el paso 00; se proyecta al nivel regional más fino (país) y la tasa de la reentrada sube país → subregión → región → global; se simula lo que queda del año (unidades del mismo mes del año anterior × nivel de los últimos 3 meses; valor = unidades × valor medio de 12 meses; reparto por cuota si falta historia), cuenta como revenue del año; lo proyectado vence al año siguiente como pipeline (valor / (1 − 0,4), tasa de su región en la pipeline real). Total por año y origen · escribe `sff_time_series`, `sff_forecast_total` | nuevo | **hecho** |
| 20 | PASO FINAL · universo time_series (retail a suscripción) y total del forecast: ts_real (meses cerrados del año), ts_proyectado (mismo mes del año anterior × nivel de los últimos 3 meses, al valor medio de la región; si falta, el total repartido por cuota), ts_reentrada en 2027 (valor / (1 − 0,4) × tasa de la región); total por año y origen · escribe `sff_time_series`, `sff_forecast_total`. Fuente: `read_time_series()` o las filas con flag del raw | nuevo (encargo del 30-sep) | **hecho** |
| NU | Tabla núcleo: UNA tabla con TODOS los registros originales (pipeline y universo time_series) y todos los sintéticos (huecos, horizonte extendido, ts_proyectado, ts_reentrada), sin claves hash, columnas con el prefijo del paso que las crea, y el bloque final `forecast_*` (universo, origen, estado real/previsto, año, pipeline y renovado) igual para todas las filas: las preguntas de negocio son SUMAS del núcleo, y `sff_forecast_total` se comprueba igual a esa suma. Se construye tras el paso 20 · escribe `sff_nucleo`, `sff_nucleo_leyenda` | `nucleo.run_core_table` | **hecho (00-20)** |
| 22 | Adaptación a Power BI: las tablas auxiliares del informe, para que Power BI solo relacione y sume · escribe `sff_forecast_pipeline_source` (código, etiqueta, bloque y orden de cada origen de la pipeline) | nuevo | **hecho** |
| AUD | Tablas de auditoría, satélites del núcleo con sus sumas de comprobación: `sff_composition`, `sff_composition_techniques`, `sff_credibility`, `sff_credibility_members`, `sff_series_dynamics`, `sff_series_backtest`, `sff_series_technique_summary`, `sff_composition_forecast_all` · ver `DOC_modelo_datos.md` | nuevo | **hecho** |
| 20 | Puente de claves para el BI | `pipeline.build_key_bridge` | bloque B |
| 21 | Perfil del raw | `analysis_data_profile.run_raw_profile` | bloque B |
| 22 | Panorama de la cartera | `analysis_portfolio_overview` | bloque B |
| 23 | Perfil de las series y el dial | `analysis_data_profile.run_fu_profile` | bloque B |
| 24 | Topología | `topologia.run_topology_profile` | bloque B |
| 25 | Efecto composición histórico y futuro | `analysis_composition` | bloque B |
| 26 | Composición mes a mes y calibración de señales | `analysis_dimensions.run_composition_analysis` | bloque B |
| 27 | Descuento y churn | `analysis_discount_churn` | bloque B |
| 28 | Maduración de señales | `analysis_signal_maturation`, `answers.carry_signal_adjustment` | bloque B |
| 29 | Escenarios de precio | `analysis_price_scenarios` | bloque B |
| 30 | Baseline de la hoja de cálculo | `analysis_baseline` | bloque B |
| 31 | Mayores movimientos | `analysis_top_movers` | bloque B |
| 32 | Fichas de las series mayores | `pipeline.draw_top_sheets` | bloque C |
| 33 | Informe | `informe.build_report` | bloque C |
| — | Persistencia avanzada (tipos SQL, append, trazabilidad por ejecución) | `config.write` y bloque SQL | pendiente |

Convenciones de cada paso: abre con título, propósito, acciones y output (azul); registra
cada acción al hacerla (azul) y cada comprobación numerada con su estado (verde / amarillo /
rojo); cuenta las comprobaciones y se detiene si alguna falla; las filas concretas se
muestran como tabla de pandas. Todo pasa por `configuration` (logger, tablas, escritura).

Fuera de alcance por ahora: las recompras (reacquisitions) no se prevén; sus columnas se
declaran como medidas extra y ningún paso las toca.

Orden de construcción por bloques:
  · bloque A (el forecast, de principio a fin): 09 → 19. Al terminarlo hay un número por
    fila futura, sus bandas y la tabla núcleo reconciliada con el raw.
  · bloque B (las explicaciones y los análisis que el informe lee): 20 → 31.
  · bloque C (la salida para negocio): 32 → 33.
Dentro de cada bloque, un paso cada vez: se revisa, se valida y se sigue.

El descuento: UNA columna del extracto (`discount`, el descuento exacto en tanto por 1) sirve a
los dos lados. Tal cual, alimenta la vía contrato del uplift y forma parte de la identidad de la
fila fina (el extracto viene una fila por descuento exacto). Cortado en tramos (los mismos de
`discount_interval`, preconfigurados en la Config), es dimensión de la celda de uplift. Nulo =
desconocido (tramo `sin_dato`): toma el uplift estadístico. Cuando hay descuento, el backtest
del uplift (paso 15) decide entre contrato y estadístico, y su veredicto gobierna el forecast.

La lista de 30 puntos de la revisión del 28-sep se aplica en el paso al que pertenece
cada punto (columna "Puntos" de `PASOS_SFF.md`), no antes.
| IN | Informe `informe_sff.md` (raw y correcciones, huecos, soporte binomial antes/después, dinámica, precisión de la tasa) y ficha por serie · escribe `sff_ficha_serie`. Se construye al final, tras el núcleo | `informe.build_report` | **hecho (00-14)** |

PENDIENTE DE REVISAR CON DATOS REALES: en el log del paso 17 (acción 6), las filas proyectadas y simuladas de la ventana y su dinero; en el paso 02 (acción 4b), cuánta pipeline de 1 año se borra por no conocerse aún; y que la pipeline de septiembre a diciembre de 2027 ya no tenga el escalón hacia abajo.

## Pasos sin letra (3-oct-2026)
- Ningún paso lleva versión con letra: la separación del universo time_series (antes `00b`/`20a`) y los niveles generados
  (antes `01b`) forman parte del paso 01; el antiguo `02b` y el examen de cartera duplicado (`step_19_portfolio_exam.py`) se
  retiraron.

## Rendimiento (3-oct-2026)
- Reglas de escritura a la escala del extracto real (~1 M filas) y protocolo de prueba de cualquier cambio de
  rendimiento (tablas idénticas bit a bit con `table_equivalence.py`): `DOC_rendimiento.md`.

## Uplift con base exacta (3-oct-2026)
- `total_tr_usd_renewed` en el extracto (licencia a licencia: lo que valía cada licencia QUE RENOVÓ, sumado) y declarado como `renewed_pipeline_usd_col`: el uplift pasa a ser revalorización pura, condicionada a renovar, comparable con la regla de contrato. Sin la columna, el paso 15 vuelve a la aproximación y lo dice en el log.
- `sff_uplift_homogeneidad` mide dónde la segmentación no basta (ratio_seleccion ≠ 1) y `sff_price_increase_monitor` detecta subidas de precio sobre el uplift de los que nunca tuvieron softcancel, con su ciclo de 12 meses.
- Pendiente de decisión: estimar el uplift estadístico solo con meses fuera de ciclo de subida (ver la propuesta del 3-oct), tras ejecutar F1/F2 con datos reales.

## Revisión de la librería, 3-oct-2026: pendientes de decidir (no tocados)
- **Dos aritméticas de uplift de respaldo**: el paso 15 calcula cada nivel (propia, padre, celda, global) como cociente de sumas; el 17, para una celda nunca vista (fila extendida con tramo nuevo), usa la media de uplifts ponderada por renovadores.
- **El padre de una celda de uplift descarta los extras de revalorización** (`msrp_increased`, `price_cap`): al heredar mezcla regímenes de precio (sospecha H2; pendiente de F2 con datos reales).
- **La auditoría no verifica el uplift por su cuenta**: sus `expected_usd_*` vienen de `esperado_usd` del forecast; su independencia cubre la cadena de tasas, no la de precio.
- **La GUIA describe el uplift con la redacción y los recuentos de una ejecución anterior**: actualizar al revisar la próxima salida de consola.
- Resueltos en la misma revisión: el detector de subidas usa la base exacta de las renovaciones isolated (`isolated_renewed_pipeline_usd_col`), y el backtest del uplift (paso 16) juzga con la base que usará el forecast (unidades renovadas × precio medio de la fila), no con la exacta de los renovadores.

## Renovaciones isolated (4-oct-2026)
- Las medidas sin softcancel pasan a ser configuraciones individuales, como el resto de medidas de la firma: `isolated_pipeline_units_col`, `isolated_renewed_units_col`, `isolated_renewed_usd_col`, `isolated_renewed_pipeline_usd_col`. "Isolated" = renovaciones del proceso normal, aisladas de cualquier evento de retención; "sin softcancel" es como lo construye Kamelot. En el núcleo, nombres fijos `forecast_isolated_*`.

## Dispersión de las renovaciones isolated (4-oct-2026)
- Siete medidas nuevas en la firma (el momento de segundo orden y seis tramos de ratio), construidas licencia a licencia: la acción 9 del paso 15 lee las isolated licencia a licencia (desviación típica, histograma por tramos, cerca de 1 y ≥ 1,10, por descuento, serie y mes) y el monitor del paso 22 las lleva a Power BI. Las isolated ya exigen mismo producto, sin upgrades, sin softcancel y no adquisición.

## Maduración del softcancel (4-oct-2026)
- Acción 7 del paso 17: ajuste al lado del forecast (no dentro) por las marcas que llegarán antes de vencer (intentos de pago, periodo de gracia), por celda mandatory y mes de calendario. Supuesto: la proporción final marcada de este año se parece a la del mismo mes del año pasado; validable cuando haya fotos mensuales del extracto acumuladas.

## Renovaciones proyectadas = retención (4-oct-2026)
- La renovación proyectada de una adquisición vence al año siguiente como retención (`renewed_acquisition_value` = Retention_Not-New), con `prev_OperationGroup` = Renewal (`dims_after_renewal`) y marcas timevarying neutras. Antes heredaba Acquisition_* y la marca de la fila que vencía: inflaba la vista de adquisición de sep-dic del año siguiente y se predecía con la tasa de primera renovación.

## Mejoras para la versión final
- Renumerar los pasos para que número = orden de ejecución (hoy: 17 → NU → 19 → 20 → 18 → IN).

## Política de idioma y nombres
- Todo en inglés (consola, tablas, columnas, valores e informe), migrando poco a poco: cada vez que se retoque un
  fichero, se proponen antes los nombres y explicaciones y se validan uno a uno antes de ejecutar.

DECISIÓN ABIERTA (escalera): ¿un grupo cerrado absorbe en pasadas posteriores las series abiertas que coinciden con él? Ver `DOC_escalera.md`, sección Decisión abierta.

## Principios

**De diseño.** Ante un problema, primero se cambia la estructura y el orden (comprometer más en las etapas iniciales)
antes que añadir comprobaciones, variables o lógica que compensen. Un dato se conforma una vez, al principio, con su
nombre definitivo; ningún paso cambia la Config.

**La regla del ruido.** Una diferencia menor que el ruido no es una diferencia; un error del tamaño del ruido no es un
fallo, es el límite. El ruido binomial σ = √(p(1−p)/n) decide qué se separa (escalera, niveles generados) y qué se puede
prometer (intervalos, examen en unidades de ruido). Ver `DOC_escalera.md`.

## Versión candidata (1-oct-2026): decisiones de la revisión crítica

### Implementado
1. **Hechos + dimensión.** `sff_nucleo` es la tabla de hechos (una fila por registro); `sff_forecast_series`, la dimensión
   (una fila por forecast serie, la que se filtra en Power BI). Los sumables del examen son columnas normales de la
   dimensión. Ver `DOC_modelo_datos.md`.
2. **Módulo único de predicción** (`prediction.py`): predicción de la composición, desplazamiento por credibilidad, nivel
   conocido en cada origen y banda. Lo usan el forecast (17), el examen (19) y la auditoría; la auditoría comprueba que
   dan la misma predicción.
3. **Examen único serie a serie** (paso 19, `step_19_series_exam.py`): raw (la serie sola), framework y hoja de cálculo
   sobre las mismas filas, cada predicción trazada y con su intervalo. Tablas `sff_series_exam_detail` y `sff_series_exam`;
   en la dimensión, raw y framework lado a lado (`s19_improvement_mae_pp`).
4. **Pasos con contrato y orquestador** (`pipeline.py`): `TableContract`, `Step`, `PipelineContext`, `Orchestrator`.
   `main.run` lo usa; `run_step` ejecuta un paso suelto y descarta lo que los pasos posteriores habían producido.
5. **Escalera mecánica.** En cada pasada todas las forecast series se agrupan por el mismo patrón; cada una usa la primera
   pasada en la que su grupo llega a 30; las de 271 o más predicen solas pero prestan su historia. La etapa 3 junta como
   mucho `collapse_passes` dimensiones (2). Tablas `sff_ladder_merges` (sesgo y error sola frente a junta de cada fusión) y
   `sff_composition_members` (`uses` / `lends`). Ver `DOC_escalera.md`.
6. **Dimensiones con un nivel generado** (paso 01, `leveled_dims={"tr_term": "ordinal", ...}`): la columna conserva su
   nombre y su valor raw; la librería añade `<columna>_level_1` (valores agrupados por tasa estandarizada en los meses de
   entrenamiento; vecinos a menos del ruido binomial de una serie en el suelo, 8,8 pp con p = 0,64; residual bajo 30). Se genera antes del calendario; la Config la cuenta como
   mandatory desde que se construye. La primera ejecución escribe `sff_levels.json` y las siguientes lo reutilizan.
7. **Corregido:** la auditoría tomaba como origen el último mes con datos de la composición en lugar de T − h.
8. **Corregido (estructura, no parche):** con term en una sola columna, el paso de niveles iba detrás del calendario y
   renombraba la columna; el paso 02 no encontraba el plazo y borraba la pipeline de todas las licencias desde 2027-09. Se
   resolvió reordenando (niveles antes del calendario), sin renombrar y sin que ningún paso modifique la Config.
9. **Checkpoints:** cada paso guarda sus tablas; `run_from("paso")` (o `python main.py --desde paso`) carga lo anterior y
   ejecuta desde ahí, avisando si cambió el código o la Config desde el checkpoint. Al terminar, el tiempo de cada paso.

### Aplazado
- **Separar análisis (decisiones persistidas) y ejecución mensual:** al productivizar. Mientras tanto todo se recalcula,
  salvo los niveles generados, que ya se fijan en su JSON.

### Por revisar con los datos reales
- Los grupos que propone el paso 01 para term y band (tabla `sff_dimension_levels`).
- Cuántas fusiones mejoran y cuántas series prestan su historia (paso 10).
- El framework frente a la serie sola y frente a la hoja (paso 19 y dimensión).
- La duración de los pasos 10 y 19.

### Hecho tras la revisión de la salida real
- **Paso 21 y capítulo 2 del informe:** las filas que el framework añade o borra (huecos, resultados adelantados y
  pipeline parcial borrados, proyectadas, simuladas), mes a mes y con el mismo formato, conciliadas con el raw. Es un
  informe, no una tabla: imprime su SQL sobre `sff_nucleo`.
- **Los informes agregados imprimen la consulta SQL que los reproduce** (`report_queries.py`, comprobadas en los tests):
  calendario por rol y por mes, examen por método y por mes, total por año y origen, filas nuevas.
- **Decidir:** si `sff_calendario`, `sff_examen_cartera(_resumen)` y `sff_forecast_total` dejan de escribirse, ya que se
  reproducen con su consulta.
- **El error del examen frente al ruido** (paso 19, dimensión e informe): qué parte del error es ruido inevitable, por
  tamaño de serie y para el total.
- **Niveles generados:** umbral derivado del ruido en el suelo, evidencia por pantalla, y una dimensión de un solo grupo
  no es una pasada.

### Pendiente de la revisión anterior
- Formateador único de tablas en consola; consola priorizada.
- Total del examen del paso 14 contado dos veces (#1a); etiqueta de la comprobación 2 del núcleo (#1b); comprobación 5 del
  paso 18 para 2027 (#1c); uplift medio del paso 15 (#1d).
- Política de idioma: renombrado del vocabulario (validar la lista antes).
- Nivel "padre" del uplift, degenerado sin extras de revalorización.
- Medir el % del dinero que se predice con una técnica de series temporales y si gana.
- Calibrar y comprobar la cobertura del intervalo del total (el que se da a negocio).
- Análisis de WIP_oct no traídos al espejo: mix, descuento y churn, maduración de señales, top movers, escenarios de
  precio, análisis inicial de cartera, 2026 frente a 2027 por región.
- Vocabulario de papeles: `uses` / `lends` en composiciones y `borrower` / `lender` en credibilidad (unificar).
