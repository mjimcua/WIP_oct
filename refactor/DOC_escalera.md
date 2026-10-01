# La escalera de soporte (pasos 01b, 09, 10 y 11)

## Qué resuelve

Una forecast serie pequeña tiene una tasa mensual muy ruidosa: con 10 contratos al mes, el error binomial de un solo mes es de unos ±15 pp. La escalera la junta con las forecast series más parecidas hasta que el grupo tiene soporte, y predice con la tasa del grupo.

## El principio: segmentar homogeneiza

Las dimensiones separan clientes con tasas esperadas distintas. Juntar dos forecast series que solo se diferencian en una dimensión es correcto cuando **esa dimensión explica poco la renovación**: respecto a la tasa, la muestra sigue siendo homogénea. Es el ejemplo del género y la felicidad: el género separa a la población en muchos aspectos, pero si la tasa que se mide es igual en hombres y mujeres, separarla no aporta nada.

Formalmente es el equilibrio entre sesgo y varianza. Al predecir una serie con la tasa de un grupo mayor:

- **se gana varianza:** el ruido binomial baja, de √(p(1−p)/n) a √(p(1−p)/N);
- **se paga sesgo:** la diferencia real entre la tasa del grupo y la de la serie.

**Juntar es correcto cuando el sesgo que se introduce es menor que el ruido que se quita.** No exige que no haya ninguna diferencia (con mucha historia cualquier diferencia pequeña es "significativa"), sino que la diferencia sea menor que lo que no se podría medir de todas formas. La tabla `sff_ladder_merges` lo comprueba en cada fusión.

Por eso importa el orden: se quita primero lo que menos explica la renovación (paso 09, con las series grandes, donde la tasa se mide bien), y esa relevancia se aplica a las series pequeñas, que no tienen datos para aprenderla solas.

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

## Los niveles generados (paso 01b)

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
4. Se juntan los vecinos más parecidos mientras su diferencia sea como mucho `level_merge_max_pp` (5 pp). `ordinal`:
   solo valores contiguos, con los números comparados como números; `nominal`: cualquier par.

**En la escalera:** una columna `X` es el nivel más fino de su familia cuando existe `X_level_N`, así que se colapsa
primero `tr_term` y después `tr_term_level_1` (la misma regla de familias que `tr_product_level_2` → `tr_product_level_1`).

**El JSON:**
- La primera ejecución escribe los grupos en `levels_path` (por defecto `salida/sff_levels.json`).
- Las siguientes lo reutilizan, así que los ids de las forecast series no cambian de un mes a otro.
- Para regenerarlos, se borra el fichero.
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
