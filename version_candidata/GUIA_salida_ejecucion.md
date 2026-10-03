# Guía de la salida de una ejecución

Explicación de cada comprobación y de cada tabla que el SFF escribe en la consola, paso a paso, sobre la ejecución real
del 1-oct-2026 (extracto de 1.023.291 filas, mes en curso 2026-09). Las comprobaciones salen en verde (`ok`), en
amarillo (`WARN`: aviso, no detiene) o en rojo (`FAIL`: detiene el paso al contarlas). Se va completando a medida que se
revisa la salida.

---

## Paso 00 · Validar el raw

**Qué hace.** Comprueba que el extracto y la Config encajan antes de tocar ningún dato, y deja el raw igual (mismas filas y
columnas) con `period` convertido a mes. No escribe ninguna tabla.

**Acción 1** — "the Config declares 32 columns in 8 roles". La Config conoce 32 columnas repartidas en 8 roles con columnas
(`extra_revalorizacion` no cuenta porque está vacío). El raw tiene 28; la diferencia son 4:
- **2 seguras:** `tr_term_level_1` y `tr_band_level_1`. La Config las declara desde que se construye, pero no existen hasta
  el paso 01.
- **2 más:** lo más probable es que sean columnas de `ignore_cols` que no vienen en el extracto (la comprobación 4 permite
  que las ignoradas falten). Se confirma con `[c for c in configuration.column_roles() if c not in raw.columns]`.

**Acción 2** — "the raw has 28 columns and 1,023,291 rows". El volumen de partida: todo lo que viene después debe cuadrar
con él.

### Comprobaciones 1-5: la estructura

| # | Qué comprueba | Por qué importa | Resultado |
|---|---|---|---|
| 1 | el raw tiene filas | una consulta que devuelve 0 filas no debe llegar a producir un forecast vacío "correcto" | 1.023.291 |
| 2 | ningún nombre de columna se repite | pandas admite columnas duplicadas, y entonces `raw["x"]` devuelve dos columnas y los cálculos fallan más adelante de forma confusa | ok |
| 3 | cada columna del raw tiene un rol en la Config | una columna sin rol no se sabe si se suma, si abre la serie o si se ignora; obliga a decidir qué es cada columna | 28 de 28 |
| 4 | cada columna declarada está en el raw | una dimensión que falta rompería los ids de las forecast series; se exceptúan las ignoradas y las generadas en el paso 01 | ok |
| 5 | cada fila tiene mes | una fila sin `period` no puede entrar en ningún mes del calendario | ok |

### Comprobaciones 6-10: el calendario

**Acción 3** — "61 months, 2022-12..2027-12". El extracto cubre 61 meses: 45 cerrados (de 2022-12 a 2026-08) y 16 de
pipeline futura (de 2026-09 a 2027-12).

| # | Qué comprueba | Resultado |
|---|---|---|
| 6 | cada valor de `period` es un mes válido (cada valor distinto se interpreta una sola vez) | 61 meses |
| 7 | `current_month` está declarado | 2026-09 |
| 8 | el mes en curso está dentro de los meses del raw; si no, el calendario no tendría sentido | 2026-09 dentro de 2022-12..2027-12 |
| 9 | quedan meses para entrenar después de reservar el examen | entrenamiento 2022-12..2026-02 |
| 10 | no hay meses vacíos entre el primero y el último: un hueco indicaría que la consulta perdió un mes entero | ok |

**Acción 5** — `period` convertido a mes. Se aplica a las filas la correspondencia valor → mes que ya leyó la comprobación
6 (61 valores distintos), en lugar de interpretar cada una de las filas.

### La tabla de roles (acción 6)

| Rol | Columnas | Qué hace la librería con ellas |
|---|---|---|
| period | `period` | el mes de cada fila; decide su rol en el calendario |
| time_series_flag | `flag_time_series` | las filas con 1 (retail a suscripción) se separan y se proyectan aparte en el paso 20 |
| measure | lo que vence (`total_tr_*`), lo que renueva (`total_renewed_*`), lo recomprado (`total_reacquired_*`) | se suman al agregar, en unidades y USD |
| mandatory | 9 del raw (+2 generadas) | abren la forecast serie y la celda de precio; la escalera las colapsa en la etapa 3. Con las generadas, el id de la serie tiene 11 dimensiones |
| timevarying | `dormant`, `softcancel`, `not_installed` | señales del cliente: abren la serie y en la etapa 1 se resumen en su signo |
| extra_renovacion | `net_new` | abre la serie, se anula en la etapa 2 y además es la columna de captación (`acquisition_column`) |
| extra_revalorizacion | ninguna | abriría la celda de precio (el uplift), no la tasa |
| formula_input | `discount`, `sku` | no son dimensiones: entradas de cálculo (el descuento exacto para los tramos del uplift, el SKU) |
| ignore | `dummy_field`, `_filter2` | se leen y no se usan |
| niveles generados (01) | `tr_term` → `tr_term_level_1` (nominal); `tr_band` → `tr_band_level_1` (ordinal) | la columna conserva su valor raw; el paso 01 añade su nivel agrupado |

### La tabla del calendario

"the calendar of the Config on these months: entrenamiento ≤ 2026-02 · examen 2026-03..2026-08 · proyeccion ≥ 2026-09".

| Rol | Meses | Filas | Para qué |
|---|---|---|---|
| entrenamiento | 39 (2022-12 … 2026-02) | 602.790 | se aprende de ellos: tasas, técnicas, niveles generados |
| examen | 6 (2026-03 … 2026-08) | 136.905 | se predicen como si no se conocieran, para medir el acierto |
| proyección | 16 (2026-09 … 2027-12) | 283.596 | el forecast; desde el mes en curso todo se trata como no conocido |

Las tres suman 1.023.291: cada fila del raw tiene exactamente un rol. Los meses cerrados (entrenamiento + examen) son
739.695 filas.

### Lo que se revisó en el paso 00

| Hallazgo | Decisión |
|---|---|
| `TR_AUV`, `REN_AUV`, `ReAC_AUV` declarados como medidas: se sumaban, y la suma de precios medios no significa nada | se quitan del extracto y de `extra_measure_cols`; un AUV se calcula donde haga falta como Σ USD / Σ unidades |
| `extra_revalorizacion` vacío: la celda de precio se abre solo con las mandatory | se deja así por ahora; se probará con datos |
| convertir `period` tardaba 35 s (interpretaba el mes fila a fila) | corregido en la librería: 12,5 s → 0,1 s con un millón de filas |

---

## Paso 01, acción 1 · Separación del universo time_series

> **Desde la versión sin pasos con letra (3-oct-2026)** la separación, la validación de valores y los niveles generados
> son un único paso 01 (`step_01_values_and_levels.py`), con una cabecera, las comprobaciones numeradas seguidas y un
> solo recuento. Las secciones de abajo describen la ejecución anterior; en la salida nueva la numeración es:
> comprobación 1 = la marca time_series es 0 / 1 (antes era una de las de señales); 2-3 = la separación (antes `[00b]`
> 1-2); 4-20 = los valores (antes 1-19, sin la marca time_series); y detrás, si hay `leveled_dims`, los niveles (antes
> `[01b]` 1-3). Acciones: 1 separación · 2-6 valores · 7-11 niveles · 12 recuento · 13 dinero de los meses cerrados ·
> 14 evidencia de los niveles y tabla de roles.

**Qué hace.** Saca del raw las filas del universo time_series (retail a suscripción, `flag_time_series = 1`) antes del
paso 01: solo llevan la región y el resultado, y el paso 20 las proyecta al final, aparte.

**Acción 1** — "0 time_series rows out of 1,023,291; 1,023,291 rows go on through steps 01-19 · region levels
[tr_regional_level_1, tr_regional_level_2, tr_regional_level_3]". **Este extracto no trae ninguna fila time_series**: todas
siguen por el flujo de renovaciones.

| # | Qué comprueba | Resultado |
|---|---|---|
| 1 | cada fila time_series tiene los tres niveles de región (se proyecta por país; sin región no se puede) | ok · 0 regiones |
| 2 | cada fila time_series cerrada tiene sus unidades y su valor | ok |

**Ojo:** las dos comprobaciones pasan porque no hay filas que comprobar, no porque se haya verificado nada. Con 0 filas, el
paso 20 no proyectará revenue time_series y el total no lo incluirá. Si el extracto debía traerlas, hay que revisar la
consulta o el valor de `flag_time_series` (que venga como 1 y no como `True`, texto o nulo).

---

## Paso 01, acciones 2-6 · Validar los valores

**Qué hace.** Comprueba que los valores del raw se pueden usar para calcular: dinero sin nulos ni negativos, renovaciones
coherentes con lo que vencía, dimensiones nunca vacías, señales 0/1 y descuento como proporción. No cambia el raw ni
escribe nada.

**Acción 1** — "renewals checked in 739,695 rows before 2026-09 · 283,596 rows from 2026-09 on: pipeline only, renewals
wiped in step 02". Las renovaciones solo se validan en los meses cerrados (602.790 + 136.905 = 739.695 filas). Desde el
mes en curso son parciales (el mes no ha terminado) o no han ocurrido, y el paso 02 las borra: validarlas no tendría
sentido.

### Comprobaciones 1-7: el dinero

| # | Qué comprueba | Ámbito | Por qué importa | Resultado |
|---|---|---|---|---|
| 1 | `total_tr_units` sin nulos | todos los meses | las unidades que vencen son el denominador de la tasa; un nulo la deja sin definir | ok |
| 2 | `total_tr_units` sin negativos | todos los meses | una unidad negativa (un ajuste o una devolución mal cargada) restaría pipeline | ok |
| 3 | `total_tr_usd` sin nulos | todos los meses | el dinero que vence es la base del forecast en USD | ok |
| 4 | `total_tr_usd` sin negativos | todos los meses | igual que la 2, en dinero | ok |
| 5 | `total_renewed_units` sin negativos | meses cerrados | una renovación negativa bajaría la tasa artificialmente | ok |
| 6 | `total_renewed_usd` sin negativos | meses cerrados | igual, en dinero | ok |
| 7 | las filas sin renovación llegan como 0, no como nulo | meses cerrados | un nulo significaría "no se sabe"; un 0 significa "no renovó". Son cosas distintas para la tasa | 0 nulos · 252.324 filas a 0 de 739.695 |

**Sobre la 7.** Es verde y además enseña 3 filas de ejemplo: son filas con algo que vencía y nada renovado, que es el
churn. Que el 34 % de las filas cerradas tengan 0 renovaciones es normal en el grano fino del extracto (una fila por
combinación de dimensiones, descuento y mes): muchas filas son de 1 o 2 licencias, y basta con que esas no renueven. En
unidades el peso es mucho menor. Los ejemplos:

| Producto | Renovación | Plazo | Band | Partner | Vencían | USD | Renovadas |
|---|---|---|---|---|---|---|---|
| Front Line · KTS MD | Auto Renewal | 1 year | 1 | NEXWAY | 2 | 90,93 | 0 |
| Standalone · Standalone product | Auto Renewal | 1 year | 5 | NEXWAY | 1 | 27,10 | 0 |

### Comprobaciones 8-10: la coherencia de las renovaciones (meses cerrados)

| # | Qué comprueba | Por qué importa | Resultado |
|---|---|---|---|
| 8 | unidades y USD renovados son nulos a la vez | si uno es nulo y el otro no, la fila está a medio cargar: no se sabe si renovó | ok |
| 9 | ninguna renovación gratis (unidades > 0 con USD ≤ 0) | una renovación sin dinero falsearía el uplift (el precio de renovación) | ok |
| 10 | ningún USD renovado sin unidades renovadas | dinero sin licencias: un ajuste contable, no una renovación | ok |

### Comprobaciones 11-15: las dimensiones y las señales

**Acción 3** — "checking 13 dimensions and 4 flags". Las 13 dimensiones son las que trae el extracto: 9 mandatory, 3
timevarying y `net_new`. Las 4 señales 0/1 son las 3 timevarying y `flag_time_series`.

| # | Qué comprueba | Resultado | Lectura |
|---|---|---|---|
| 11 | ninguna dimensión tiene un valor nulo o vacío | ok, 13 dimensiones | un vacío crearía una forecast serie fantasma (`…\|\|…`) |
| 12 | `dormant` es 0 / 1 | 36 % de las filas a 1 | más de un tercio de las filas tiene la señal de cliente dormido |
| 13 | `softcancel` es 0 / 1 | 18 % a 1 | |
| 14 | `not_installed` es 0 / 1 | 10 % a 1 | |
| 15 | `flag_time_series` es 0 / 1 | 0 % a 1 | confirma lo del paso de separación: la columna existe, pero ninguna fila es time_series |

Los porcentajes son de **filas**, no de unidades ni de dinero. Indican que muchas forecast series llevarán signo (alguna
señal activa) y pasarán por la etapa 1 de la escalera.

**Desde la próxima ejecución** cada señal dice también su peso en dinero ("36 % of rows at 1 · X % of the USD due"): una
señal que está en muchas filas pequeñas pesa poco en dinero. En el sintético, por ejemplo, `dormant` está en el 18 % de las
filas y solo en el 2 % del dinero.

### Comprobación 16: el descuento

**Acción 4.** La 16 comprueba que el descuento está entre 0 y 1 o es nulo (nulo = desconocido): ok, **39,8 % de las filas
sin descuento conocido**. En esas filas el uplift no puede usar el método de contrato (precio de renovación =
precio / (1 − descuento)) y depende del estadístico. Es un porcentaje de filas: en dinero puede ser distinto (en la
revisión anterior era el 59 % de la pipeline). En la tabla de tramos del paso 03, `sin_dato` es el 57 % del dinero que
vence. Desde la próxima ejecución, la comprobación lo dice: "% of rows unknown · % of the USD due".

### Comprobaciones 17-19: lo plausible que merece una mirada

**Acción 5.** Son comprobaciones que, si fallaran, saldrían como aviso (amarillo) y no detendrían la ejecución.

| # | Qué comprueba | Resultado |
|---|---|---|
| 17 | ninguna fila renueva más unidades de las que vencían (meses cerrados) | ok: la tasa de una fila nunca pasa del 100 % |
| 18 | ninguna fila vence con unidades pero USD = 0 | ok: no hay licencias regaladas en la pipeline |
| 19 | ninguna fila repetida en todas sus columnas | ok: la consulta no duplica filas (tardó unos 6 segundos, porque compara el millón de filas entero) |

19 de 19 en verde.

### El dinero de los meses cerrados (acción 7)

"closed months: 16,752,861 units due · 10,704,395 renewed (63.9%) · $580,740,540 due · $423,244,812 renewed".

| | Vence | Renueva | Tasa |
|---|---|---|---|
| Unidades | 16.752.861 | 10.704.395 | 63,9 % |
| USD | 580.740.540 | 423.244.812 | 72,9 % |

La tasa en dinero (72,9 %) es mayor que en unidades (63,9 %): 72,9 / 63,9 ≈ 1,14. El dinero renovado se cobra al precio
de renovación, no al de la pipeline, así que el cociente mezcla dos cosas: cuántos renuevan y a qué precio, que también
depende de qué contratos renuevan (si renuevan más los de más valor, el dinero sube). Esa es la separación que hace el
forecast: la tasa en unidades (pasos 08-14) y el uplift de precio (pasos 15-16).

---

## Paso 01, acciones 7-11 y 14 · Dimensiones con un nivel generado

**Qué hace.** Da a cada dimensión de `leveled_dims` su nivel agrupado, `<columna>_level_1`: sus valores agrupados por la
tasa estandarizada de los meses de entrenamiento. Lo guarda en un JSON que las ejecuciones siguientes reutilizan, para que
las forecast series conserven sus ids. La columna original conserva su valor raw.

**Acción 1** — "2 dimensions with a generated level ['tr_term', 'tr_band'] · file salida\sff_levels.json: not found,
generating". Primera ejecución: no hay JSON, así que se generan los grupos de las dos.

**Acción 2** — grupos generados y escritos en `salida\sff_levels.json` (13 segundos).

**Acción 3** — rellenadas `tr_term_level_1` y `tr_band_level_1`.

| # | Qué comprueba | Resultado |
|---|---|---|
| 1 | ningún valor del extracto es desconocido para el fichero (un valor nuevo iría a `residual`, con aviso) | ok: todos los valores aparecen en los meses de entrenamiento |
| 2 | la tabla `sff_dimension_levels` se escribe y se relee | 5 filas × 8 columnas |

### La tabla de grupos (acción 7)

`rate_std` es la tasa estandarizada: la de cada valor comparada con la de su celda (mismas mandatory), sobre la tasa
global. Se juntan dos grupos vecinos mientras difieren 5 pp o menos.

| Dimensión | Grupo | Valores | Unidades (entrenamiento) | Soporte mensual | rate_std |
|---|---|---|---|---|---|
| tr_term | 2 year + 3 year | 2 year, 3 year | 1.818.026 | 47.037 | 0,54 |
| tr_term | 1 year | 1 year | 12.812.675 | 316.229 | 0,65 |
| tr_band | 1 .. 5 (5 values) | 1, 2, 3, 4, 5 | 14.376.897 | 354.054 | 0,64 |
| tr_band | 6 + 7 | 6, 7 | 3.532 | 45 | 0,71 |
| tr_band | 10 + 20 | 10, 20 | 250.272 | 5.598 | 0,64 |

Las unidades de cada dimensión suman lo mismo (14.630.701): son todas las de entrenamiento, repartidas de dos formas.

**tr_term (nominal).** 2 y 3 años renuevan parecido (dentro de 5 pp) y se juntan. 1 año queda aparte: 11 pp más, con el
88 % de las unidades. No hay valores raros (no hay grupo `residual`).

**tr_band (ordinal: solo se juntan vecinos).**
- **Bands 1 a 5 en un solo grupo.** Comparadas dentro de su celda, ninguna se separa más de 5 pp de su vecina. Es distinto
  de la agrupación manual (1 aparte, 2-5 juntos): la diferencia de tasa en bruto entre band 1 y las demás puede venir de
  la mezcla de producto o región, y desaparece al comparar dentro de la misma celda.
- **"6 + 7"** tiene muy poco volumen (45 al mes, justo por encima de 30) y renueva más (0,71).
- **"10 + 20"** renueva igual que 1-5 (0,64) pero no se junta con ellos: al ser ordinal solo puede unirse a su vecino,
  "6 + 7", que difiere 7 pp. Con `nominal` se juntaría con 1-5. Si los packs grandes deben quedar aparte por
  interpretación, `ordinal` es correcto.
- **La tasa de cada año** de cada grupo: en esta ejecución no salía por pantalla. Desde la próxima, sí (ver más abajo).

### Lo que cambia en los niveles generados desde la próxima ejecución

**El umbral de fusión ya no es un 5 pp fijo.** Dos valores vecinos se juntan mientras sus tasas estandarizadas difieren
menos que el **ruido binomial de una serie en el suelo de soporte**: 100·√(p(1−p)/30). Con p = 0,64 son **8,8 pp**.

El razonamiento es el mismo de la escalera: se junta mientras el sesgo que se añade es menor que el ruido que se quita.
El nivel generado solo lo usan las series que suben la escalera, las de menos de 30 contratos al mes. Su tasa mensual
tiene, como mínimo, ese ruido: dos valores que difieren menos que eso no los distingue ninguna serie que vaya a usar la
fusión. Los 5 pp equivalían al ruido de una serie de unos 92 contratos al mes, que no sube la escalera. Para volver a un
valor fijo: `level_merge_max_pp = 5`.

**Por pantalla, en la acción 7:**
1. el criterio de cada dimensión y su umbral calculado;
2. qué significa una diferencia en pp: el ruido de la tasa mensual según el tamaño de la serie (10, 30, 50, 100, 271 y
   1.000 contratos al mes);
3. los grupos con su tasa de cada año en columnas (`rate_2023`, `rate_2024`…): un grupo es estable si sus años se
   parecen;
4. cada valor: su soporte, su tasa estandarizada, su ruido (`noise_pp`) y su tasa por año. Dos valores que difieren
   menos que su ruido no se distinguen mes a mes;
5. las decisiones de fusión, en orden: qué se juntó, la diferencia en pp, el umbral y qué habría dicho el 5 pp fijo
   (`within_5_pp`).

**Una dimensión que termina en un solo grupo** se avisa ("ONE group"): no separa la tasa más allá de ese ruido, su
`level_1` es constante y la escalera no le da pasada (paso 09: "no variation, not a pass of the ladder").

El JSON se regenera solo si cambia el umbral, así que en la próxima ejecución se regenerará.

Este criterio es una aplicación de **la regla del ruido** ("una diferencia menor que el ruido no es una diferencia; un
error del tamaño del ruido no es un fallo: es el límite"), explicada en `DOC_escalera.md`.

### La tabla de roles con los niveles generados

La misma tabla del paso 00, ahora con el rol mandatory en 11 columnas: `tr_term`, `tr_term_level_1`, `tr_band` y
`tr_band_level_1`, cada nivel generado justo detrás de su columna. La fila `niveles generados (01)` marca las dos como
`generado`.

---

## Paso 02 · Aplicar el calendario

**Qué hace.** Da a cada fila su rol en el tiempo, lee como 0 una renovación nula de un mes cerrado, y borra lo que todavía
no se conoce desde el mes en curso: el futuro debe parecer que no ha empezado. Añade seis columnas: `rol`,
`es_mes_en_curso`, `s0_renovados_unidades`, `s0_renovados_usd`, `s0_vencen_unidades` y `s0_vencen_usd`. Escribe la
tabla `sff_calendario` (una fila por mes).

| Acción | Qué dice | Lectura |
|---|---|---|
| 1 | renovaciones y pipeline guardadas tal como venían en las columnas `s0_` | el valor original nunca se pierde: el núcleo concilia con el extracto gracias a ellas |
| 2 | roles: entrenamiento ≤ 2026-02 · examen 2026-03..2026-08 · proyección ≥ 2026-09 | |
| 3 | 0 renovaciones nulas de meses cerrados leídas como 0 | coherente con el paso 01 (comprobación 7: 0 nulos) |
| 4 | 12.009 filas desde 2026-09 tenían renovaciones ya registradas: borradas 145.114 unidades · $6.018.579 | son resultados adelantados del mes en curso (septiembre aún no ha terminado) y renovaciones anticipadas de meses posteriores. Si se dejaran, el forecast sumaría un resultado parcial a uno previsto |
| 4b | 8.683 filas de licencias de 1 año que vencen desde 2027-09 (vendidas o renovadas desde 2026-09): pipeline borrada 183.057 unidades · $6.248.131 · por plazo: `{'1 year': 8683}` | esa pipeline la crea una venta o renovación que todavía no ha ocurrido; el paso 17 la proyecta. **Solo `1 year`**: confirma la corrección del plazo (antes se borraban también las de 2 y 3 años) |

### Comprobaciones 1-9

| # | Qué comprueba | Resultado |
|---|---|---|
| 1 | cada fila tiene uno de los tres roles | entrenamiento 602.790 · examen 136.905 · proyección 283.596 |
| 2 | el entrenamiento tiene al menos 12 meses (un año completo de estacionalidad) | 39 meses |
| 3 | cada mes de examen tiene filas | 6 meses |
| 4 | el mes en curso tiene pipeline: si no, no habría nada que predecir este mes | 358.686 unidades vencen en 2026-09 |
| 5 | la pipeline no cambia salvo en las filas aún no conocidas (unidades y USD) | ok |
| 6 | ninguna fila de 1 año creada desde el mes en curso conserva su pipeline | 8.683 filas borradas: $6.248.131, guardados en `s0_vencen_usd` |
| 7 | las renovaciones de los meses cerrados no cambian (un nulo se lee como 0) | 0 nulos |
| 8 | ninguna fila desde el mes en curso conserva una renovación | 12.009 filas con resultados adelantados borradas: 145.114 unidades · $6.018.579 |
| 9 | las columnas `s0_` guardan las renovaciones y la pipeline del raw, sin tocar | ok |

**Comprobación 10** (acción 6): `sff_calendario` escrita y releída, 61 filas × 11 columnas.

### La tabla por rol (acción 8)

| Rol | Meses | Desde | Hasta | Filas | Unidades que vencen | Renovadas | Tasa |
|---|---|---|---|---|---|---|---|
| entrenamiento | 39 | 2022-12 | 2026-02 | 602.790 | 14.630.701 | 9.346.408 | 0,64 |
| examen | 6 | 2026-03 | 2026-08 | 136.905 | 2.122.160 | 1.357.987 | 0,64 |
| proyección | 16 | 2026-09 | 2027-12 | 283.596 | 4.638.992 | — | — |

- **Entrenamiento y examen tienen la misma tasa (0,64):** el examen no cae en un periodo atípico, así que medir el
  acierto en él es representativo.
- **La proyección no tiene renovaciones:** se borraron (acción 4). La tasa es lo que hay que predecir.
- **La proyección vence menos por mes** (4,64 M en 16 meses, unos 290.000 al mes) que el entrenamiento (375.000 al mes) y
  el examen (354.000 al mes). Por dos motivos: la pipeline de 2027 solo contiene lo ya vendido (licencias plurianuales y
  de 1 año vendidas antes de 2026-09), y la de 2027-09 a 2027-12 de 1 año está borrada (acción 4b). El paso 17 añade lo
  que falta: las renovaciones proyectadas y la captación simulada.

---

## Paso 03 · La tabla fina

**Qué hace.** Añade a cada fila del raw sus ids y claves: la forecast serie, la forecast unit, el tramo de descuento y la
celda de precio. Escribe `sff_fact_fine` (una fila por fila del raw).

| Acción | Qué dice | Lectura |
|---|---|---|
| 1 | `fs_id` con 15 columnas: las 11 mandatory (con los dos niveles generados), las 3 señales y `net_new` → **33.024 forecast series** | cada combinación distinta de esas 15 columnas es una forecast serie. Los niveles generados no la dividen más, porque dependen de su columna (`tr_band` determina `tr_band_level_1`) |
| 2 | `fu_id` = `fs_id` \| mes → **556.516 forecast units** | una unidad es una forecast serie en un mes: el grano en el que se estudia la tasa. Media de 1,8 filas finas por unidad |
| 3 | `tramo_descuento` desde el descuento: 10 tramos, 39,8 % de las filas `sin_dato` | el descuento exacto se agrupa de 10 en 10 puntos |
| 4 | `uplift_cell_id` con las 11 mandatory + `tramo_descuento` → **22.638 celdas de precio** | la celda donde se mide el uplift (precio de renovación frente al de la pipeline). No usa las señales ni `net_new`, porque afectan a la tasa, no al precio |
| 5 | claves derivadas: `fs_key`, `fu_key`, `uplift_cell_key`, `fila_key` | números enteros para unir tablas en Power BI más rápido que con textos largos |

### Comprobaciones 1-5

| # | Qué comprueba | Resultado |
|---|---|---|
| 1 | cada descuento exacto cae en un único tramo (un nulo, en `sin_dato`) | límites 0, 10, …, 90, 100 % |
| 2 | una fila por forecast unit, valores de revalorización y descuento exacto (el grano del raw) | 1.023.291 filas = 1.023.291 filas finas distintas |
| 3 | cada id distinto tiene su propia clave (sin colisiones de hash) | 33.024 `fs_key` · 556.516 `fu_key` · 22.638 `uplift_cell_key` · 1.023.291 `fila_key` |
| 4 | mismas filas, mismo orden, ninguna columna del paso 02 cambiada | 8 columnas añadidas |
| 5 | `sff_fact_fine` escrita y releída | 1.023.291 filas × 44 columnas, en **166,5 segundos** |

La comprobación 2 confirma algo importante: no hay dos filas del raw con la misma forecast unit y el mismo descuento.
Si las hubiera, el extracto tendría filas partidas que habría que sumar antes.

**Escribir la tabla fina tarda casi 3 minutos.** Es el coste de subir un millón de filas × 44 columnas a SQL Server. Con
los checkpoints, al reanudar desde un paso posterior no se vuelve a escribir.

### Filas y dinero por tramo de descuento (acción 9)

| Tramo | Filas | Descuento mín | máx | USD que vence | % del USD |
|---|---|---|---|---|---|
| 0-10 % | 106.000 | 0,00 | 0,09 | 171.690.975 | 22,3 % |
| 10-20 % | 59.333 | 0,10 | 0,19 | 31.285.857 | 4,1 % |
| 20-30 % | 90.747 | 0,20 | 0,29 | 33.606.723 | 4,4 % |
| 30-40 % | 84.192 | 0,30 | 0,39 | 19.140.725 | 2,5 % |
| 40-50 % | 69.806 | 0,40 | 0,49 | 15.921.098 | 2,1 % |
| 50-60 % | 79.538 | 0,50 | 0,59 | 19.182.290 | 2,5 % |
| 60-70 % | 84.816 | 0,60 | 0,69 | 14.168.302 | 1,8 % |
| 70-80 % | 32.904 | 0,70 | 0,79 | 2.518.044 | 0,3 % |
| 80-90 % | 8.959 | 0,80 | 0,89 | 344.310 | 0,0 % |
| sin_dato | 406.996 | — | — | 441.125.712 | 57,4 % |
| **Total** | **1.023.291** | | | **768.983.736** | |

- **El 57 % del dinero no tiene descuento conocido** (el 39,8 % de las filas). Para ese dinero el uplift solo puede ser el
  estadístico.
- **Del dinero con descuento conocido** (328 M), la mitad está en 0-10 %: precio prácticamente de lista.
- **No hay ninguna fila con descuento del 90 % o más.**
- Cuanto más descuento, menos dinero por fila: en el tramo 60-70 % hay casi tantas filas como en el 0-10 %, pero
  12 veces menos dinero.

---

## Paso 21 · Las filas que añade o borra el framework (nuevo, desde la próxima ejecución)

**Qué hace.** Pone una al lado de otra, mes a mes y con el mismo formato, todas las filas que el framework añade o borra.
No calcula nada nuevo: cuenta lo que hicieron los pasos 02, 08 y 17, y lo concilia con el raw. **Es un informe, no una
tabla**: no se escribe en SQL (sería una segunda copia del núcleo). Se muestra por pantalla, alimenta el capítulo 2 del
informe, y el paso imprime la consulta SQL que lo reproduce desde `sff_nucleo`.

| Origen | Paso | Qué es | Por qué | Medidas |
|---|---|---|---|---|
| `hueco` | 08 | una forecast unit sin nada que vencer, DENTRO de la historia de una serie estimable (entre su primer y su último mes con vencimientos, hasta el último mes cerrado) | las técnicas leen meses consecutivos (tendencia, estación, media móvil): un mes que falta rompería la serie. El mes existe, solo que no vencía nada | todo a 0; su tasa queda **nula, nunca 0 %** (sin vencimientos no hay tasa). Añade un mes, no dinero |
| `resultado_adelantado_borrado` | 02 | una renovación ya registrada desde el mes en curso | el mes no ha terminado (o no ha empezado): si se dejara, el forecast sumaría un resultado parcial a uno previsto | unidades y USD renovados que se borran |
| `pipeline_parcial_borrada` | 02 | la pipeline de las licencias de 1 año que vencen 12 meses después del mes en curso o más tarde | la generan las ventas y renovaciones desde el mes en curso, que solo han empezado (el mes va por la mitad y los siguientes no han empezado): está a medio crear. Con ella, el forecast de esos meses se calcularía sobre una pipeline a medias | unidades y USD que vencían y se borran (el original queda en `s0_vencen_*`) |
| `proyectada` | 17 | la renovación esperada de cada licencia de 1 año que vence en la ventana de simulación, 12 meses después | reconstruye entera, con un solo método, la pipeline que se borró a medias | unidades y USD que vencen, y el USD que se espera renovar |
| `simulada` | 17 | la captación simulada en la ventana, 12 meses después | igual, para las ventas nuevas | igual |

**Ejemplo de por qué se borra la pipeline parcial.** Un cliente cuya licencia de 1 año venció el 5 de septiembre de 2026
y ya renovó tiene su fila de pipeline en 2027-09; uno que vence el 25 todavía no. La pipeline de 2027-09 del extracto solo
tiene a los primeros. El paso 17 la sustituye por la de todos: cada licencia que vence en 2026-09, por su tasa prevista.

**Formato:** `mes · rol · origen · filas · series · unidades · usd · esperado_usd`.

**Acciones por pantalla:**
1. los huecos (filas y series);
2. lo borrado por el calendario;
3. lo creado por el forecast;
4. la tabla;
5. la conciliación con el raw;
6. el recuento de comprobaciones;
7. los totales por origen, el detalle de los meses con pipeline borrada o creada (los huecos están repartidos por la
   historia), y la consulta SQL que lo reproduce.

| # | Qué comprueba | Por qué importa |
|---|---|---|
| 1 | lo que el raw tenía que vencer = lo que queda + la pipeline parcial borrada (unidades y USD) | nada se pierde sin explicar: cada unidad del extracto está en el forecast o en la fila que la borró |

**Qué esperar con tus datos:** 183.057 unidades y $6,2 M de `pipeline_parcial_borrada` en 2027-09..12, frente a lo
`proyectada` y `simulada` en los mismos meses. La comparación dice si el forecast reconstruye lo borrado con algo del
mismo orden.

---

## Paso 19 · El error frente al ruido (nuevo, desde la próxima ejecución)

Cada predicción del examen lleva el **ruido binomial** de la tasa real que intenta acertar, √(p(1−p)/n) con p su tasa
predicha y n las unidades que vencen: lo que se equivocaría incluso una predicción perfecta. Es la regla del ruido
aplicada al examen.

| Dónde | Qué |
|---|---|
| `sff_series_exam_detail` | `noise_pp` en cada predicción |
| `sff_series_exam` y la dimensión | `rmse_pp` (error cuadrático medio), `noise_pp` (ruido medio) y `error_over_noise` = error / ruido, para el framework y para la serie sola |
| resumen por método y horizonte | `error_vs_ruido` |
| informe, capítulo 5 | el error frente al ruido por tamaño de serie (< 30, 30-270, ≥ 271 contratos al mes) y para el total de la cartera, con la parte del error que es ruido |

**Cómo leer `error_vs_ruido`:**
- **≈ 1:** al límite, ya no se puede acertar más;
- **claramente mayor que 1:** falta algo que se podía saber;
- **claramente menor que 1:** sospechoso (el modelo ha visto el futuro).

En el sintético, en las series de menos de 30 contratos al mes, el framework está a 1,16 veces el ruido (el 74 % de su
error es ruido inevitable) y la serie sola a 1,36.

---

## Los informes agregados y su SQL (desde la próxima ejecución)

Un informe con pocas filas porque está muy agregado (por mes, por rol, por año y origen, por método) no es una fuente de
datos: es una consulta sobre las tablas de detalle que escribe la ejecución. El paso lo muestra por pantalla e imprime la
consulta SQL que lo reproduce ("the SQL that reproduces it from the tables of the run"), con los nombres reales de las
tablas (esquema y prefijo de la Config). Los tests ejecutan cada consulta sobre las tablas del sintético y comprueban que
da exactamente el mismo informe.

| Paso | Informe | Se reproduce desde |
|---|---|---|
| 02 | el calendario por rol | `sff_nucleo` |
| 02 | el calendario por mes (`sff_calendario`) | `sff_nucleo` |
| 19 | el examen por método y horizonte | `sff_series_exam_detail` |
| 19 | el total de cada mes de examen por método (`sff_examen_cartera`) | `sff_series_exam_detail` |
| 20 | el total por año y origen (`sff_forecast_total`) | `sff_nucleo` |
| 21 | las filas que añade o borra el framework | `sff_nucleo` |

Los informes que usan medianas o cuantiles (el resumen de la escalera, las bandas de error) o el JSON de los niveles
generados no tienen consulta: no son una suma sobre una tabla de detalle.
