# Rendimiento: buenas prácticas a la escala del extracto real

No es optimización temprana. Son hábitos de escritura que salen gratis si se aplican al escribir y cuestan
minutos de ejecución si no. Ningún arreglo de rendimiento cambia una línea de cálculo: solo cambia **cómo se
llega a los datos** dentro de un bucle.

## La escala para la que se escribe

Hay que escribir pensando en el extracto real, no en el sintético (el sintético cabe en cualquier código):

| Qué | Orden de magnitud (ejecución del 3-oct-2026) |
|---|---|
| filas finas del extracto | ~1 millón |
| forecast series | ~20.000 |
| composiciones | ~3.500 |
| filas futuras del forecast | ~350.000 |
| predicciones del examen (paso 19) | ~575.000 |
| filas del backtest por serie (auditoría) | ~900.000 |
| pasadas del examen (meses × horizontes) | ~12 |

**La cuenta que hay que hacer antes de escribir un bucle: iteraciones × coste de cada una.**

| Coste por iteración | × 1 millón |
|---|---|
| 0,1 µs (consulta a un `dict`, operación numpy ya vectorizada) | 0,1 s |
| 1 µs (función Python pequeña, `lambda` sobre un Period) | 1 s |
| 50 µs (`Series.get` o `.loc` sobre un MultiIndex) | 50 s |
| 1 ms (clave que no está en un MultiIndex: pasa por una excepción) | 17 min |

Un bucle que recorre un millón de filas y hace por fila algo de pandas que no sea vectorizado es un bug de
rendimiento, aunque en el sintético no se note.

## Las reglas

1. **Búsqueda por fila → `dict`, construido una vez.** Nunca `Series.get`, `.loc` ni `.map` sobre una Serie
   de tuplas dentro de una comprensión de cientos de miles de filas. Se vuelca la columna con `.to_dict()`
   antes del bucle y se consulta el `dict`. Si el valor es numérico, el valor por defecto es `np.nan` y la
   Serie resultante se crea con `dtype=float` (un `None` cambiaría el tipo y las comparaciones).
2. **Nada de filtrar, agrupar ni ordenar un DataFrame grande dentro de un bucle de orígenes o meses.** Si
   el mismo histórico se consulta en varios orígenes, se indexa una vez (`prediction.index_history`) y cada
   origen toma su prefijo con `months_known` (búsqueda binaria). Es el patrón del paso 14, que ya lo hacía
   bien con máscaras sobre arrays.
3. **Ordenar una vez, globalmente**, y agrupar con `sort=False`, no ordenar cada bloque dentro del bucle.
4. **Dentro de un bucle por clave, solo arrays numpy.** Nada de construir listas desde objetos Period
   (`[month.month for month in ...]`) en cada iteración: se precalcula una vez para todo el histórico.
5. **Los arrays compartidos entre iteraciones son de solo lectura** (`flags.writeable = False`). Así, si
   una técnica escribiera en su historia, fallaría en voz alta en lugar de cambiar la del origen siguiente.
6. **Los bucles que producen filas iteran las claves ordenadas**, en el mismo orden que daba `groupby`, para
   que la tabla salga en el mismo orden.
7. **`lambda` sobre un millón de Periods (p. ej. `month.year`) cuesta ~1 s: es aceptable.** Solo se
   vectoriza si está dentro de otro bucle o si los logs dicen que pesa.

## Lo que no se toca por rendimiento

- **El cálculo real**: las técnicas, las regresiones y los p-valores (vectorizarlos cambiaría los últimos
  bits).
- **La independencia de la auditoría**: recalcula todo por su cuenta; nunca reutiliza resultados del 17 o
  del 19 para ahorrar tiempo.
- **Las escrituras SQL**: con `fast_executemany=True` (en `main.py`) van al límite de la red; bajarlas
  exigiría quitar la relectura, que es una comprobación.

## Cómo se prueba un cambio de rendimiento

1. Antes de tocar nada: `python table_equivalence.py save antes.pkl`.
2. Después del cambio: `python table_equivalence.py save despues.pkl` y
   `python table_equivalence.py compare antes.pkl despues.pkl`. Tienen que salir **n tablas idénticas de
   n**, bit a bit. Si no, el cambio se descarta; no se ajusta una tolerancia.
3. Si el cambio toca una rama que el sintético no recorre (en el sintético todas las técnicas tienen banda
   propia, así que nunca se usa el fallback al retador ni ±z), esa rama lleva un test directo
   (`test_prediction.py`, parte C).
4. La suite completa en verde.
5. Para medir: los logs tienen marca de tiempo por acción. El salto entre dos acciones dice dónde está el
   tiempo. En real, la reejecución barata es `python main.py --desde forecast`.

## Arreglos aplicados (3-oct-2026)

| Dónde | Patrón corregido | Medida |
|---|---|---|
| `prediction.band_quantiles` (pasos 17, 19 y auditoría ▸5) | `Series.get` sobre MultiIndex, fila a fila | 900.000 filas: ~744 s → 1,7 s |
| paso 19, `predict_month` | `groupby` + `sort_values` + arrays por serie en cada una de las 12 pasadas | acceso a 600.000 filas: ~25 s por pasada → índice de 2 s, una vez |
| paso 17 `pool_predictions`, auditoría ▸6 `composition_forecast_all` | lo mismo, en una sola pasada | menor; por coherencia con el patrón |
| auditoría ▸5 `is_chosen`, ▸5 `chosen_rank`, comprobación 9; paso 17 confianza | `pd.Series(list(zip(...))).map(...)` | lineal en `dict` |
| auditoría ▸4 dinámicas por serie | un `sort` por serie | un `sort` global |

Resultado: 64 de 64 tablas del sintético idénticas antes y después, y la suite completa en verde.
