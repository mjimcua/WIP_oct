# Cambio · comprobación 7 de la auditoría y avisos de numpy (ejecución real del 1-oct)

## 1. "without credibility, every series predicts exactly its composition's rate — 32 differ"

`shifted_rate` (`prediction.py`) pasaba la tasa por la escala logit aunque el desplazamiento por credibilidad fuera 0, y
para eso la recortaba a [0,000001; 0,999999]. Una composición con tasa prevista exactamente 1 (o 0) salía como 0,999999.
La comprobación tenía razón. Corregido en la definición de la función: **sin desplazamiento, la tasa es la de la
composición, sin tocar**; solo la tasa desplazada pasa por la escala logit. Afecta igual al forecast, al examen y a la
auditoría, porque los tres usan la misma función.

## 2. "RuntimeWarning: invalid value encountered in divide" (c /= stddev)

Venía de la dinámica de cada forecast serie (paso 13, que usa la auditoría): la consistencia del perfil estacional es la
correlación entre la primera y la segunda mitad de la historia. En series que no varían (0 % todos los meses, o 1-2
unidades que siempre renuevan) la correlación no existe y numpy avisaba una vez por serie. Ahora solo se calcula si las
dos mitades varían; si no, es `NaN` (lo que numpy acababa devolviendo, sin el aviso). φ de esas series también es `NaN`:
con una tasa de 0 % o 100 % no hay ruido binomial con el que comparar.

Ambos casos reproducidos en los tests con el código anterior (`test_prediction`, `test_step_13_informe`); el sintético
corre entero tratando cualquier `RuntimeWarning` como error.

Verificación: 22 ficheros de test en verde.

## prediction.py

```diff
--- /tmp/prediction_before_fix.py	2026-10-01 10:19:24.754656263 +0000
+++ prediction.py	2026-10-01 10:19:24.838125314 +0000
@@ -56,12 +56,13 @@
 
 
 def shifted_rate(composition_rate, z, reference_level, composition_level, apply: bool = True) -> tuple:
-    """(rate of the forecast series, shift in logit): the composition's prediction moved toward its reference."""
+    """(rate of the forecast series, shift in logit): the composition's prediction moved toward its reference.
+    Without a shift the rate IS the composition's prediction, untouched (also when it is exactly 0 or 1);
+    only a shifted rate goes through the logit scale."""
     shift = credibility_shift(z, reference_level, composition_level, apply)
     composition_rate = np.asarray(pd.Series(composition_rate), dtype=float)
-    rate = np.where(np.isfinite(composition_rate),
-                    inverse_logit(logit(np.clip(np.nan_to_num(composition_rate, nan=0.5), LEVEL_CLIP, 1 - LEVEL_CLIP)) + shift),
-                    np.nan)
+    moved = inverse_logit(logit(np.clip(np.nan_to_num(composition_rate, nan=0.5), LEVEL_CLIP, 1 - LEVEL_CLIP)) + shift)
+    rate = np.where(shift != 0, moved, composition_rate)
     return rate, shift
 
 
```

## step_13_dynamics.py

```diff
--- /tmp/s13_before_fix.py	2026-10-01 10:19:24.757250307 +0000
+++ step_13_dynamics.py	2026-10-01 10:19:24.838720198 +0000
@@ -197,7 +197,9 @@
     first_half = detrended[years <= middle_year].groupby("mes")["residuo"].mean()
     second_half = detrended[years > middle_year].groupby("mes")["residuo"].mean()
     shared = first_half.index.intersection(second_half.index)
-    consistency = float(np.corrcoef(first_half[shared], second_half[shared])[0, 1]) if len(shared) >= 3 else np.nan
+    # a profile that does not move in one of the halves has no correlation (a series with the same rate every month)
+    both_move = len(shared) >= 3 and first_half[shared].std() > 0 and second_half[shared].std() > 0
+    consistency = float(np.corrcoef(first_half[shared], second_half[shared])[0, 1]) if both_move else np.nan
 
     profile = pd.DataFrame({"mes": month_effect.index.astype(int), "efecto_pp": deviation_pp.to_numpy(),
                             "error_pp": PERCENTAGE_POINTS * month_error.to_numpy(), "meses_observados": month_count.to_numpy(),
```
