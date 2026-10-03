# La escalera de soporte (pasos 01, 09, 10 y 11)

## Qué resuelve

Una forecast serie pequeña tiene una tasa mensual muy ruidosa: con 10 contratos al mes, el error binomial de un solo mes es de unos ±15 pp. La escalera la junta con las forecast series más parecidas hasta que el grupo tiene soporte, y predice con la tasa del grupo.

## El principio: segmentar homogeneiza

Las dimensiones separan clientes con tasas esperadas distintas. Juntar dos forecast series que solo se diferencian en una dimensión es correcto cuando **esa dimensión explica poco la renovación**: respecto a la tasa, la muestra sigue siendo homogénea. Es el ejemplo del género y la felicidad: el género separa a la población en muchos aspectos, pero si la tasa que se mide es igual en hombres y mujeres, separarla no aporta nada.

Formalmente es el equilibrio entre sesgo y varianza. Al predecir una serie con la tasa de un grupo mayor:

- **se gana varianza:** el ruido binomial baja, de √(p(1−p)/n) a √(p(1−p)/N);
- **se paga sesgo:** la diferencia real entre la tasa del grupo y la de la serie.

**Juntar es correcto cuando el sesgo que se introduce es menor que el ruido que se quita.** No exige que no haya ninguna diferencia (con mucha historia cualquier diferencia pequeña es "significativa"), sino que la diferencia sea menor que lo que no se podría medir de todas formas. La tabla `sff_ladder_merges` lo comprueba en cada fusión.

Por eso importa el orden: se quita primero lo que menos explica la renovación (paso 09, con las series grandes, donde la tasa se mide bien), y esa relevancia se aplica a las series pequeñas, que no tienen datos para aprenderla solas.

## La regla del ruido

> **Una diferencia menor que el ruido no es una diferencia. Un error del tamaño del ruido no es un fallo: es el límite.**

El ruido binomial de la tasa mensual de una forecast serie con n contratos al mes y tasa p es σ = √(p(1−p)/n). Es la
variación que tendría la tasa aunque el cliente medio no cambiara nada: puro azar de quién renueva este mes.

| Contratos al mes (p = 0,64) | σ (ruido) | Error medio de una predicción perfecta (≈ 0,8 σ) | El 95 % de los meses cae en |
|---|---|---|---|
| 10 | ±15,2 pp | 12,1 pp | ±29,8 pp |
| 30 (el suelo) | ±8,8 pp | 7,0 pp | ±17,2 pp |
| 92 | ±5,0 pp | 4,0 pp | ±9,8 pp |
| 271 | ±2,9 pp | 2,3 pp | ±5,7 pp |
| 1.000 | ±1,5 pp | 1,2 pp | ±3,0 pp |
| 354.000 (toda la cartera en un mes) | ±0,08 pp | 0,06 pp | ±0,16 pp |

**Primera mitad: para separar.** Dos segmentos cuyas tasas difieren menos que σ no se distinguen mes a mes. Separarlos no
aporta información y quita soporte. Es el criterio de la escalera (juntar mientras el sesgo que se añade es menor que el
ruido que se quita) y el de los niveles generados (paso 01: se juntan los valores que difieren menos que el ruido de una
serie en el suelo, 8,8 pp).

**Segunda mitad: para prometer.** Aunque supiéramos la tasa verdadera de una serie, el mes real caería a su alrededor
con ese ruido. Una predicción perfecta se equivoca, de media, en torno a 0,8 σ, y solo el 68 % de los meses cae dentro de
±σ. De ahí tres consecuencias:

- **Lo que hay dentro de la banda de ruido es azar: no se puede predecir.** La expectativa realista no es caer en el
  centro, sino que el error sea del orden de σ. Un error de ese tamaño quiere decir que se ha extraído todo lo que se podía.
- **Un error claramente mayor que σ** indica que falta algo que sí se podía saber: una tendencia, una estacionalidad o un
  cambio de mercado.
- **Un error sistemáticamente menor que σ** no es mérito, es sospechoso: el modelo ha visto datos del futuro o se ha
  ajustado al azar.

Por eso el intervalo que se da a negocio nunca puede ser más estrecho que la banda de ruido de la serie. Y por eso el
examen mide el error en unidades de ruido (`err_norm` = error / σ en `sff_series_backtest`): un |err_norm| cercano a 1 es
acertar al límite.

**La consecuencia para negocio, la más útil.** El ruido baja con la raíz del tamaño:
- **En una serie pequeña,** casi todo el error es ruido: de 30 contratos al mes, ±8,8 pp no se pueden mejorar.
- **En el total de la cartera,** el ruido es despreciable (±0,08 pp): todo el error que quede es del modelo (sesgo,
  cambios de mercado).

Por eso el forecast se juzga por serie frente a su ruido, y por total frente a la hoja de cálculo.

## Las pasadas y su orden

Cada pasada pone a `*` (cualquier valor) una dimensión más:

| Etapa | Pasadas | Qué se quita |
|---|---|---|
| 0 · raw | 1 | nada: cada forecast serie es su propio grupo |
| 1 · signo | 1 | las señales activas (dormant, softcancel…) se resumen en su signo; positivas y negativas nunca se juntan |
| 2 · extras | 1 por extra | cada extra de renovación (p. ej. `net_new`, el canal) |
| 3 · colapso | como mucho `collapse_passes` (2 por defecto) | las mandatory, la que menos explica primero; en una familia `_level_N`, primero el nivel más fino |
| 4 · credibilidad | — | no junta series: mezcla la tasa de la composición con la de una referencia más amplia |

Las pasadas de colapso posteriores a `collapse_passes` no juntan series: solo sirven para buscar la referencia de credibilidad (etapa 4). Las series con signo solo suben mientras la pérdida acumulada de R² es pequeña (`signed_ladder_max_loss`); las mixtas no suben nunca.

## La regla mecánica

> En cada pasada, **todas** las forecast series se agrupan por el mismo patrón. Cada forecast serie usa **la primera pasada en la que su grupo llega a 30** contratos al mes (`support_floor`). Ese grupo es su **composición**.

- Una serie que ya llega a 30 sola se queda en la pasada 0 y predice con su propia tasa.
- Su historia **sigue contando** en los grupos de las demás: una serie grande no se toca, pero presta sus datos a sus hermanas pequeñas.
- Una serie que no llega a 30 en ninguna pasada permitida se queda con el grupo más amplio que tuvo y depende de la credibilidad.

Por eso una composición tiene dos recuentos: las series **que entran en su tasa** (`group_series`) y las series **que la usan** (`composition_users`). La tabla `sff_composition_members` las lista con su papel: `uses` o `lends`.

## Las 4 etapas y sus ids

Cada forecast serie tiene un id por etapa. Un id **solo cambia cuando el grupo gana series**: si una pasada no le añade nadie, conserva el id anterior. Agrupando por el id de cualquier etapa, los totales del raw suman lo mismo, porque cada serie tiene exactamente un id por etapa.

El soporte que se informa en cada etapa es el del grupo con el que la serie predeciría: cuenta todas las series de su tasa, también las que solo prestan.

## Ejemplo (sintético)

| Forecast serie | Contratos/mes | Pasada que usa | Composición | Soporte |
|---|---|---|---|---|
| `NA·A·web` | 279 | 0 · raw | ella misma | 279 |
| `NA·A·tele` | 8 | 2 · sin canal | `NA\|A\|SIG=neutro\|*` (con su hermana web, que presta) | 287 |
| `EU·A·softcancel` | 12 | 1 · signo | `EU\|A\|SIG=negativo\|web` (las 4 negativas) | 42 |
| `NA·B·tele` | 40 | 0 · raw | ella misma, con credibilidad hacia `NA\|B\|*` | 40 |

El error de la tasa de un mes de `NA·A·tele` pasa de ±12,1 pp sola a ±3,9 pp con su composición (sesgo +3,4 pp).

## Los niveles generados (paso 01)

Las dimensiones declaradas en `leveled_dims` (por ejemplo `{"tr_term": "ordinal", "tr_band": "ordinal"}`) **conservan su
nombre y su valor raw**, que es el nivel fino. La librería añade una sola columna, `<columna>_level_1`, con sus valores
agrupados por la tasa estandarizada. La Config la cuenta como dimensión mandatory, justo detrás de su columna, desde que se
construye; ningún paso la modifica después.

**Cuándo:** justo después de validar el raw (pasos 00 y 01) y antes del calendario. El raw queda conformado una vez, al
principio, y todos los pasos siguientes ven las mismas columnas. Los meses de entrenamiento salen del calendario de la
Config (`current_month`, `test_months`), no del paso 02.

**Cómo se forman los grupos:**
1. Se usan solo los meses de entrenamiento.
2. Cada valor se compara dentro de su celda, con el resto de mandatory iguales.
3. Los valores con menos de 30 contratos al mes van a `residual`.
4. Se juntan los vecinos más parecidos mientras su diferencia sea menor que el **ruido binomial de una serie en el suelo
   de soporte**, 100·√(p(1−p)/30): 8,8 pp con p = 0,64. `ordinal`: solo valores contiguos, con los números comparados
   como números; `nominal`: cualquier par.

**Por qué ese umbral.** Es el mismo principio de la escalera: se junta mientras el sesgo que se añade es menor que el
ruido que se quita. El nivel generado solo lo usan las series que suben la escalera, las de menos de 30 contratos al
mes, y su tasa mensual tiene como mínimo ese ruido. Dos valores que difieren menos que eso no los distingue ninguna
serie que vaya a usar la fusión. `level_merge_max_pp` fija otro valor (5 pp equivale al ruido de una serie de unos 92
contratos al mes).

**Una dimensión que termina en un solo grupo** no separa la tasa más allá de ese ruido: su `level_1` es constante y el
paso 09 no le da pasada.

**En la escalera:** una columna `X` es el nivel más fino de su familia cuando existe `X_level_N`, así que se colapsa
primero `tr_term` y después `tr_term_level_1` (la misma regla de familias que `tr_product_level_2` → `tr_product_level_1`).

**El JSON:**
- La primera ejecución escribe los grupos en `levels_path` (por defecto `salida/sff_levels.json`).
- Las siguientes lo reutilizan, así que los ids de las forecast series no cambian de un mes a otro.
- Se regeneran si se borra el fichero o si cambia el tipo o el umbral.
- Un valor nuevo que el fichero no conoce va a `residual`, con un aviso.

## En el forecast

1. **Paso 12:** la serie mensual de cada composición es la suma de **todas** las series de su tasa.
2. **Paso 14:** el backtest elige la técnica de cada composición.
3. **Paso 17 (`prediction.py`):** la técnica predice la tasa de la composición. Si z < 1, la predicción se desplaza hacia la referencia en (1 − z) de la diferencia de niveles, en escala logit. Cada forecast serie que la usa recibe esa tasa, aplicada a su propia pipeline.

## La credibilidad (etapa 4)

Si la composición tiene menos de 271 contratos al mes (`own_rate_floor`), toma como referencia la primera pasada posterior (fusionadora o no) con más series y que llega a 30. Su tasa es z × composición + (1 − z) × referencia, con z = n / (n + k) y k de Bühlmann-Straub. La z de Bühlmann es exactamente el peso que minimiza sesgo² + varianza: es el mismo principio de la escalera, aplicado de forma continua.

## Niveles de riesgo

| Nivel | Regla |
|---|---|
| A_propio | la propia serie (pasada 0), ≥ 271 y al menos 12 meses |
| A2_propio_corto | la propia serie, ≥ 271, menos de 12 meses |
| A3_propio_reforzado | la propia serie, entre 30 y 271: credibilidad con su referencia |
| B_prestado | se juntó en la pasada de signo o de extras: su grupo comparte TODAS las mandatory |
| C_lejano | se juntó en una pasada mandatory, o su grupo no llegó a 30 |
| S_signo_bajo_suelo | con signo y su grupo no llegó a 30 |
| M_signo_mixto · D_sin_historia · N_sin_impacto | mixta · solo futuro · solo historia |

## Tablas

| Tabla | Una fila por | Para qué |
|---|---|---|
| `sff_dimension_levels` | dimensión × grupo | los grupos generados de `level_1`, con su tasa estandarizada por año |
| `sff_ladder_steps` | serie × pasada | su id, el soporte de su grupo y si ya tiene composición |
| `sff_ladder_summary` | pasada | ids, mediana de soporte, unidades (iguales en todas), % USD con soporte |
| `sff_ladder_merges` | serie × fusión | grupo antes y después, sesgo, error sola y junta, y si mejora |
| `sff_composition_members` | composición × serie | todas las series de su tasa, con su papel (`uses` / `lends`) |
| `sff_ladder_groups` | serie estimable | composición, pasada, series en la tasa, series que la usan, soporte, tasa, referencia |
| `sff_forecast_series` | forecast serie | las 4 etapas (`s10_stage*`) y la credibilidad (`s11_*`): ver `DOC_modelo_datos.md` |
