"""synthetic_from_topology.py — SFF v3 · a synthetic raw with the shape of the real one.

`build_raw_from_topology(topology, seed, scale)` turns the anonymized profile written by
`topologia.run_topology_profile` into a raw with generic column names (mandatory_1 …,
tv_neg_1 …, extra_ren_1 …, extra_rev_1 …, discount_pct) that the framework reads with the
configuration returned by `topology_config(...)`. It reproduces:
  · the calendar (history, exam, pending, projection months) and the current month
  · the variety of series: how many (× scale), their dimension values by share (nested
    hierarchies kept), their monthly support drawn from the support quantiles, their
    months of history and the share born inside the history, their gaps by support
  · the rates: the neutral series' own rate drawn from its quantiles, the flagged series
    renewing at the flag's ratio, the extra monthly dispersion (φ) as beta-binomial noise
  · the volume: the monthly seasonal profile and the trend; the acquisition factor
  · the signals: the final share of every flag, and in the future months the share as
    it looks today by distance (the maturation)
  · price: AUV relative quantiles; the discount (unknown / zero / positive quantiles);
    the renewed price by the contract rule × the realization ratio of the bucket, or the
    statistical uplift for unknown discounts
It is a generator, not a copy: nothing identifies a market, a product or a customer.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json

import numpy as np
import pandas as pd

from config import Config

QUANTILE_KEYS = ["0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"]
QUANTILE_LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def sample_from_quantiles(quantiles: dict, size: int, rng: np.random.Generator, floor: float = None, ceiling: float = None) -> np.ndarray:
    """Inverse-CDF sampling from the stored quantiles (linear between them)."""
    if not quantiles:
        return np.full(size, 1.0)
    values = np.array([quantiles[k] for k in QUANTILE_KEYS if k in quantiles], dtype=float)
    levels = QUANTILE_LEVELS[[QUANTILE_KEYS.index(k) for k in QUANTILE_KEYS if k in quantiles]]
    u = rng.uniform(levels.min(), levels.max(), size=size)
    out = np.interp(u, levels, values)
    if floor is not None:
        out = np.maximum(out, floor)
    if ceiling is not None:
        out = np.minimum(out, ceiling)
    return out


def sample_values(block: dict, size: int, rng: np.random.Generator, parent_values: np.ndarray = None) -> np.ndarray:
    """Values of a dimension by their shares; if nested, children of the sampled parent."""
    codes = [f"v{i + 1:02d}" for i in range(block["cardinalidad"])]
    shares = np.array(block["cuotas"], dtype=float)
    shares = shares / shares.sum()
    if parent_values is None or "padre" not in block:
        return rng.choice(codes, size=size, p=shares)
    children_of = {}
    for code, parent in block["padre"].items():
        children_of.setdefault(parent, []).append(code)
    out = []
    for parent in parent_values:
        options = children_of.get(parent) or codes
        weights = np.array([shares[codes.index(c)] for c in options], dtype=float)
        out.append(rng.choice(options, p=weights / weights.sum()))
    return np.array(out)


def build_raw_from_topology(topology: dict, seed: int = 11, scale: float = 0.1, current_month: str = "2026-09") -> pd.DataFrame:
    """The synthetic raw. `scale` multiplies the number of series (0.1 = a tenth of the
    portfolio); supports and money keep their real distribution, so the totals scale too."""
    rng = np.random.default_rng(seed)
    roles, calendar = topology["roles"], topology["calendario"]
    n_series = max(20, int(topology["series"]["n_series"] * scale))
    # ── the calendar: months before the current month = history; from it, projection
    history_months = sum(v for k, v in calendar["meses_por_rol"].items() if k in ("entrenamiento", "examen", "pendiente_cierre"))
    projection_months = calendar["meses_por_rol"].get("proyeccion", 12)
    current = pd.Period(current_month, freq="M")
    months = pd.period_range(current - history_months, current + projection_months - 1, freq="M")
    # ── the series: dimensions
    frame = pd.DataFrame(index=range(n_series))
    previous = None
    for name in roles["mandatory"]:
        block = topology["dimensiones"][name]
        frame[name] = sample_values(block, n_series, rng, frame[previous].to_numpy() if previous and "padre" in block else None)
        previous = name
    for name in roles["extra_ren"] + roles["extra_rev"]:
        frame[name] = sample_values(topology["dimensiones"][name], n_series, rng)
    # ── the signals: neutral by default; a share of series carries each flag (units-weighted share ≈ series share)
    flags = list(roles["timevarying"])
    for flag in flags:
        share = topology["dimensiones"][flag]["cuota_unidades"]
        frame[flag] = (rng.uniform(size=n_series) < share).astype(int)
    # ── support, history, rate, price per series
    frame["soporte"] = sample_from_quantiles(topology["series"]["soporte_cuantiles"], n_series, rng, floor=1.0)
    frame["meses_historia"] = np.clip(sample_from_quantiles(topology["series"]["meses_historia_cuantiles"], n_series, rng, floor=3.0), 3, history_months).astype(int)
    born_inside = rng.uniform(size=n_series) < topology["series"]["cuota_nacen_dentro"]
    frame["meses_historia"] = np.where(born_inside, frame["meses_historia"], history_months)
    frame["tasa"] = sample_from_quantiles(topology["tasas"]["tasa_neutra_cuantiles"], n_series, rng, floor=0.02, ceiling=0.98)
    ratios = topology["tasas"].get("ratio_tasa_flag_vs_neutra", {})
    for flag in flags:
        ratio = ratios.get(flag, ratios.get(flag.replace("tv_", ""), 1.0))
        frame.loc[frame[flag] == 1, "tasa"] = np.clip(frame.loc[frame[flag] == 1, "tasa"] * ratio, 0.02, 0.98)
    auv_rel = sample_from_quantiles(topology["precio"]["auv_cuantiles_relativos"], n_series, rng, floor=0.05)
    frame["auv"] = 30.0 * auv_rel
    discount_block = topology["precio"].get("descuento")
    if discount_block:
        kind = rng.uniform(size=n_series)
        unknown = kind < discount_block["cuota_desconocido"]
        zero = (~unknown) & (kind < discount_block["cuota_desconocido"] + discount_block["cuota_cero"])
        positive = sample_from_quantiles(discount_block["positivo_cuantiles"], n_series, rng, floor=0.01, ceiling=0.9)
        frame["discount_pct"] = np.where(unknown, np.nan, np.where(zero, 0.0, positive))
    # ── the months: volume profile, trend, φ noise, gaps, maturation of the flags
    profile = np.array(topology["volumen"]["perfil_mensual"], dtype=float)
    trend = 1 + topology["volumen"]["tendencia_pct_ano"] / 100
    phi = topology["tasas"].get("phi_agregado") or 1.0
    gap_share = topology["series"].get("cuota_huecos_por_tramo", {})
    realization = {b: v.get("ratio_realizacion") or 1.0 for b, v in topology["precio"].get("uplift_por_tramo_descuento", {}).items()}
    unknown_uplift = topology["precio"].get("uplift_desconocido_cuantiles") or topology["precio"].get("uplift_fila_cuantiles") or {"0.5": 1.05}
    maturation = topology.get("senales", {})
    rows = []
    for index, series in frame.iterrows():
        start = len(months) - projection_months - int(series["meses_historia"])
        for position, month in enumerate(months):
            if position < start:
                continue
            years_from_start = (position - start) / 12
            units = series["soporte"] * profile[month.month - 1] * (trend ** years_from_start) * rng.lognormal(0, 0.15)
            units = max(1, int(round(units)))
            bucket = "0-30" if series["soporte"] < 30 else "30-271" if series["soporte"] < 271 else "271-752" if series["soporte"] < 752 else "752-inf"
            if rng.uniform() < gap_share.get(bucket, 0.0):
                continue                                              # a month with nothing due: absent, as in the raw
            is_future = month >= current
            # extra dispersion beyond binomial: the month's rate wobbles around the series' rate
            rate = series["tasa"]
            if phi and phi > 1:
                rate = float(np.clip(rng.normal(rate, np.sqrt((phi - 1) * rate * (1 - rate) / max(units, 1))), 0.01, 0.99))
            renewed = 0 if is_future else int(rng.binomial(units, rate))
            discount = series.get("discount_pct", np.nan)
            if np.isfinite(discount):
                bucket_key = "0" if discount == 0 else "0-0.2" if discount < 0.2 else "0.2-0.4" if discount < 0.4 else "0.4-0.6" if discount < 0.6 else "0.6-0.8" if discount < 0.8 else "0.8-1"
                uplift = (1 / (1 - min(discount, 0.9))) * realization.get(bucket_key, 1.0) * rng.lognormal(0, 0.03)
            else:
                uplift = float(sample_from_quantiles(unknown_uplift, 1, rng, floor=0.3)[0])
            row = {name: series[name] for name in roles["mandatory"] + roles["extra_ren"] + roles["extra_rev"]}
            for flag in flags:                                       # in the future, a flag shows only the share matured by today (by distance h)
                value = int(series[flag])
                if is_future and value == 1:
                    h = str(int((month - current).n))
                    block = maturation.get(flag, {})
                    final, now = block.get("cuota_final", 0.0) or 0.0, (block.get("cuota_futuro_por_h") or {}).get(h, 0.0)
                    if final > 0 and rng.uniform() > now / final:
                        value = 0
                row[flag] = value
            row.update(dict(period=str(month), dataset_role="projection" if is_future else "train", is_current_month=int(month == current), flag_time_series=0,
                            total_tr_units=units, total_tr_usd=round(units * series["auv"], 2), total_renewed_units=renewed,
                            total_renewed_usd=round(renewed * series["auv"] * uplift, 2)))
            if "discount_pct" in frame.columns:
                row["discount_pct"] = discount
            rows.append(row)
    raw = pd.DataFrame(rows)
    # one row per grain (dimensions × month): series that share every dimension merge, and the
    # exact discount — a formula input, not a dimension — becomes the units-weighted mean of the row
    keys = [c for c in raw.columns if not c.startswith("total_") and c not in ("is_current_month", "flag_time_series", "discount_pct")]
    if "discount_pct" in raw.columns:
        raw["_disc_w"] = raw["discount_pct"] * raw["total_tr_units"]
        raw["_disc_n"] = raw["total_tr_units"].where(raw["discount_pct"].notna(), 0.0)
    aggregations = {c: (c, "sum") for c in raw.columns if c.startswith("total_")}
    aggregations.update(is_current_month=("is_current_month", "max"), flag_time_series=("flag_time_series", "max"))
    if "discount_pct" in raw.columns:
        aggregations.update(_disc_w=("_disc_w", "sum"), _disc_n=("_disc_n", "sum"))
    raw = raw.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
    if "discount_pct" in raw.columns or "_disc_w" in raw.columns:
        raw["discount_pct"] = raw["_disc_w"] / raw["_disc_n"].where(raw["_disc_n"] > 0)
        raw = raw.drop(columns=["_disc_w", "_disc_n"])
    return raw


def topology_config(topology: dict, folder: str, **overrides) -> Config:
    """The configuration that reads the synthetic built from a topology."""
    roles = topology["roles"]
    arguments = dict(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=list(roles["mandatory"]),
                     structural_timevarying_dims={k: v for k, v in roles["timevarying"].items()},
                     extra_renovacion=list(roles["extra_ren"]), extra_revalorizacion=list(roles["extra_rev"]),
                     discount_value_column="discount_pct" if roles.get("descuento_exacto") else None,
                     benchmark_group_dims=list(roles["mandatory"][:2]) if len(roles["mandatory"]) >= 2 else list(roles["mandatory"][:1]),
                     console_explanations=False, sheets_top_series=0)
    arguments.update(overrides)
    return Config(**arguments)


def load_topology(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
