# El escenario real (datos recogidos de las ejecuciones)

Cifras de las ejecuciones con datos reales, tal como aparecen en la consola, para entender el escenario sin volver a
ejecutar. Fuente y fecha en cada bloque. Se va completando.

## Volumen y calendario (ejecución del 1-oct-2026, 13:34)

| Dato | Valor |
|---|---|
| Filas del extracto | 1.023.291 |
| Columnas | 28 (32 declaradas en la Config: + 2 niveles generados + 2 que no vienen) |
| Meses | 61, de 2022-12 a 2027-12 |
| Mes en curso | 2026-09 |
| Filas time_series (`flag_time_series = 1`) | **0** |

| Rol | Meses | Rango | Filas | Unidades que vencen | Renovadas | Tasa |
|---|---|---|---|---|---|---|
| entrenamiento | 39 | 2022-12 … 2026-02 | 602.790 | 14.630.701 | 9.346.408 | 0,64 |
| examen | 6 | 2026-03 … 2026-08 | 136.905 | 2.122.160 | 1.357.987 | 0,64 |
| proyección | 16 | 2026-09 … 2027-12 | 283.596 | 4.638.992 (tras el borrado) | — | — |

**Meses cerrados en total:** 16.752.861 unidades que vencen, 10.704.395 renovadas (63,9 %); $580.740.540 que vencen,
$423.244.812 renovados (72,9 %). El cociente dinero/unidades (≈ 1,14) mezcla el precio de renovación con qué contratos
renuevan.

**Por mes:** entrenamiento ≈ 375.000 unidades, examen ≈ 354.000, proyección ≈ 290.000; el mes en curso (2026-09) tiene
358.686.

## Calidad de los datos (pasos 00 y 01)

- **Sin problemas estructurales:** 10/10 en el paso 00 y 19/19 en el 01. No hay nulos ni negativos en el dinero, ni
  dimensiones vacías, ni filas duplicadas, ni renovaciones por encima de lo que vencía.
- **Filas cerradas con 0 renovaciones:** 252.324 de 739.695 (34 % de las filas). Es churn en el grano fino; en unidades
  pesa mucho menos.
- **Señales, en % de filas:** `dormant` 36 %, `softcancel` 18 %, `not_installed` 10 %. El % del dinero se verá en la
  próxima ejecución.
- **Descuento desconocido:** 39,8 % de las filas, **57,4 % del dinero que vence** ($441,1 M de $769,0 M).

## Lo que borra el calendario (paso 02)

| Qué | Filas | Unidades | USD |
|---|---|---|---|
| renovaciones ya registradas desde 2026-09 (resultados adelantados) | 12.009 | 145.114 | 6.018.579 |
| pipeline de 1 año que vence desde 2027-09 (la crea una venta aún no ocurrida; la proyecta el 17) | 8.683 | 183.057 | 6.248.131 |

## Los niveles generados (paso 01, umbral fijo de 5 pp)

| Dimensión | Grupo | Unidades (entrenamiento) | Soporte mensual | rate_std |
|---|---|---|---|---|
| tr_term | 2 year + 3 year | 1.818.026 | 47.037 | 0,54 |
| tr_term | 1 year | 12.812.675 | 316.229 | 0,65 |
| tr_band | 1 .. 5 | 14.376.897 | 354.054 | 0,64 |
| tr_band | 6 + 7 | 3.532 | 45 | 0,71 |
| tr_band | 10 + 20 | 250.272 | 5.598 | 0,64 |

- `tr_term` es `nominal` y `tr_band` es `ordinal`.
- El 1 año es el 88 % del volumen.
- Las bands 6 y 7 son casi inexistentes (45 contratos al mes).

## Series, unidades y celdas (paso 03)

| Objeto | Cantidad |
|---|---|
| forecast series (`fs_id`, 15 columnas) | 33.024 |
| forecast units (serie × mes) | 556.516 (1,8 filas finas por unidad) |
| celdas de precio (11 mandatory + tramo de descuento) | 22.638 |

**Del dinero que vence por tramo de descuento:**
- 0-10 %: 22,3 %;
- 10-30 %: 8,5 %;
- 30-70 %: 8,9 %;
- 70-90 %: 0,3 %;
- sin dato: 57,4 %;
- 90 % o más: ninguna fila.

## Ejecución anterior (1-oct, 12:44, antes de la reestructuración de los niveles)

Son cifras orientativas: el extracto y los ids han cambiado desde entonces.
- 19.244 forecast series estimables. Su dinámica: 9.458 con poco soporte, 7.959 con historia corta y 1.827 medibles.
- 18.253 combinaciones serie × tramo.
- 20.658 combinaciones composición × mes de examen.
- 821.565 filas en el backtest serie × mes × horizonte × técnica.
- 932.160 tasas futuras composición × horizonte × técnica.

## Tiempos observados

| Qué | Tiempo | Nota |
|---|---|---|
| convertir `period` (paso 00) | 35 s | corregido: unos 0,1 s |
| generar los niveles (paso 01) | 13 s | |
| escribir `sff_fact_fine` (1.023.291 × 44) | 166,5 s | subir a SQL; al reanudar con checkpoints no se repite |
| auditoría completa (ejecución anterior) | ≈ 16 min | dinámica de 19.244 series y backtest de cada técnica en cada serie |

---

## Para revisar en la próxima ejecución

1. **Paso 01 (niveles), el umbral de fusión derivado** (8,8 pp con p = 0,64 y suelo 30, en lugar de 5 pp):
   - la tabla de cada valor: cuánto se separa band 1 de 2-5 y si las bands 6-7 y 10-20 se juntan con 1-5;
   - la tabla de decisiones (`within_5_pp`): qué habría cambiado con 5 pp;
   - la tasa por año de cada grupo: si son estables;
   - si `tr_band` termina en un solo grupo: el aviso "ONE group" y, en el paso 09, "no variation, not a pass of the
     ladder";
   - **decidir:** umbral derivado (por defecto) o fijo (`level_merge_max_pp = 5`).
2. **Paso 01, comprobaciones 12-16:** el % del dinero de cada señal y del descuento desconocido.
3. **Universo time_series:** 0 filas. Confirmar si el extracto debía traerlas.
4. **Tiempos:** la tabla de tiempos por paso al final de la ejecución, para decidir dónde poner el foco.
5. **Paso 21, las filas nuevas:** los huecos, y lo proyectado y lo simulado en 2027-09..12 frente a la pipeline parcial borrada (183.057
   unidades · $6,2 M). ¿El forecast sustituye lo borrado por algo del mismo orden?
6. **Paso 19 e informe, el error frente al ruido:** por tamaño de serie, cuánto del error es ruido inevitable; y el total
   de la cartera, donde el ruido debería ser de unos ±0,08 pp, así que casi todo su error es del modelo.
