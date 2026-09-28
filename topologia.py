"""topologia.py — SFF v3 · the anonymized topology of the raw, so a realistic synthetic can be built.

WHY. The synthetic dataset the tests run on is invented. To test the LOGIC on situations
that look like the real portfolio without moving any data, the raw is summarized into
indicators that keep its SHAPE and drop its identity: how many series, how big, how they
renew by sign, how the volume moves along the year, how the flags mature, how the
discount relates to the revaluation, how concentrated the money is. Dimensions are named
by their ROLE (mandatory_1, tv_neg_1, extra_ren_1 …) and their values by rank of money
(v01, v02 …); no product, region, price or count travels as such — only shares,
quantiles and ratios (the totals are scaled to 1).

OUTPUT. `<outdir>/topologia.json` (the input of `synthetic_from_topology.build_raw_from_topology`)
and the table `topologia` (one row per indicator, for the report). Blocks of the JSON:

  calendario     months of history, exam, pending, projection; the current month's position
  dimensiones    per role: cardinality, shares of pipeline $ by ranked value, and the
                 parent value in the previous mandatory dim when the hierarchy is nested
  series         number of series (scaled), quantiles of monthly support, of months of
                 history, share born inside the history, gap share by support bucket
  tasas          quantiles of the series' own rate (neutral), rate ratio of every flag
                 to neutral, monthly dispersion of the aggregate rate beyond binomial (φ)
  volumen        the monthly profile of the units due (12 indices), trend %/year, the
                 acquisition factor pipeline(t)/renewed(t−12) quantiles
  senales        share of units with each flag in the closed months (final) and in the
                 future months by distance h (the maturation as it looks today)
  precio         AUV quantiles (scaled to the median), discount: share unknown, share 0,
                 quantiles of the known positive discount; observed uplift by discount
                 bucket and its realization ratio against 1/(1−d); uplift quantiles of
                 the unknown-discount rows
  concentracion  share of pipeline $ in the top 1 %, 10 %, 50 % of series
  extras         cardinality and shares of the extras (renovación, revalorización)
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os

import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
SUPPORT_BUCKETS = [(0, 30), (30, 271), (271, 752), (752, np.inf)]


def ranked_shares(frame: pd.DataFrame, column: str, weight: str) -> dict:
    """value (anonymized by rank of weight) → share of the weight, plus the mapping."""
    shares = frame.groupby(column)[weight].sum().sort_values(ascending=False)
    shares = shares / max(shares.sum(), 1e-9)
    mapping = {value: f"v{index + 1:02d}" for index, value in enumerate(shares.index)}
    return dict(cardinalidad=int(len(shares)), cuotas=[round(float(s), 5) for s in shares.values]), mapping


def role_names(configuration: Config) -> dict:
    """real column → role name (mandatory_1, tv_neg_1, tv_pos_1, extra_ren_1, extra_rev_1)."""
    names = {}
    for index, column in enumerate(configuration.business_mandatory_dims):
        names[column] = f"mandatory_{index + 1}"
    negatives = [c for c, p in configuration.structural_timevarying_dims.items() if p == "negative"]
    positives = [c for c, p in configuration.structural_timevarying_dims.items() if p == "positive"]
    for index, column in enumerate(negatives):
        names[column] = f"tv_neg_{index + 1}"
    for index, column in enumerate(positives):
        names[column] = f"tv_pos_{index + 1}"
    for index, column in enumerate(configuration.extra_renovacion):
        names[column] = f"extra_ren_{index + 1}"
    for index, column in enumerate(configuration.extra_revalorizacion):
        names[column] = f"extra_rev_{index + 1}"
    return names


def quantiles_of(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {}
    return {str(q): round(float(values.quantile(q)), 5) for q in QUANTILES}


def profile_calendar(fine: pd.DataFrame, configuration: Config) -> dict:
    period, role = configuration.period_col, configuration.dataset_role_col
    months = sorted(fine[period].unique())
    by_role = fine.groupby(role)[period].nunique().to_dict()
    return dict(meses_total=len(months), meses_por_rol={k: int(v) for k, v in by_role.items()},
                mes_en_curso_posicion=int(months.index(fine.loc[fine[configuration.current_month_col] == 1, period].min())) if (fine[configuration.current_month_col] == 1).any() else len(months))


def profile_dimensions(fine: pd.DataFrame, configuration: Config, names: dict) -> tuple:
    """Shares by ranked value per dimension role; nested parent for mandatory dims."""
    usd = configuration.pipeline_usd_col
    out, mappings = {}, {}
    previous = None
    for column in configuration.business_mandatory_dims:
        block, mapping = ranked_shares(fine, column, usd)
        mappings[column] = mapping
        if previous is not None:
            parents = fine.groupby(column)[previous].nunique()
            if (parents == 1).all():                     # nested: every value has exactly one parent
                parent_of = fine.drop_duplicates(column).set_index(column)[previous]
                block["padre"] = {mapping[v]: mappings[previous][parent_of[v]] for v in mapping}
                block["anidada_en"] = names[previous]
        out[names[column]] = block
        previous = column
    for column in list(configuration.structural_timevarying_dims):
        flagged = fine[column].isin(configuration.timevarying_positive_values)
        out[names[column]] = dict(cuota_unidades=round(float(fine.loc[flagged, configuration.pipeline_units_col].sum() / max(fine[configuration.pipeline_units_col].sum(), 1e-9)), 5))
    for column in list(configuration.extra_renovacion) + list(configuration.extra_revalorizacion):
        block, mapping = ranked_shares(fine, column, usd)
        mappings[column] = mapping
        out[names[column]] = block
    return out, mappings


def profile_series(units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> dict:
    period, role = configuration.period_col, configuration.dataset_role_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES)]
    first_month = closed[period].min()
    card = series_card[series_card["ruta"] == TREATMENT_PREDICTABLE] if "ruta" in series_card.columns else series_card
    born_inside = float((closed.groupby("fs_id")[period].min() > first_month).mean())
    gaps_by_bucket = {}
    if "sintetica" in units.columns:
        per_series = units.groupby("fs_id").agg(huecos=("sintetica", "sum"), meses=(period, "nunique"))
        per_series["n"] = per_series.index.map(series_card.set_index("fs_id")["n_propio"])
        for low, high in SUPPORT_BUCKETS:
            block = per_series[(per_series["n"] >= low) & (per_series["n"] < high)]
            if len(block):
                gaps_by_bucket[f"{low}-{high if np.isfinite(high) else 'inf'}"] = round(float(block["huecos"].sum() / max(block["meses"].sum(), 1)), 5)
    return dict(n_series=int(len(card)), soporte_cuantiles=quantiles_of(card["n_propio"]), meses_historia_cuantiles=quantiles_of(card["meses_historia"]),
                cuota_nacen_dentro=round(born_inside, 4), cuota_huecos_por_tramo=gaps_by_bucket,
                cuota_series_por_signo={str(k): round(float(v), 4) for k, v in card["signo"].value_counts(normalize=True).items()})


def profile_rates(units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config, names: dict = None) -> dict:
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES) & (units[pipe] > 0)]
    neutral = series_card[(series_card["signo"] == SIGN_NEUTRAL) & series_card["tasa_propia"].notna()]
    weights = neutral["n_propio"].clip(lower=1)
    out = dict(tasa_neutra_cuantiles=quantiles_of(neutral["tasa_propia"]),
               tasa_neutra_media_ponderada=round(float(np.average(neutral["tasa_propia"], weights=weights)), 5) if len(neutral) else None)
    neutral_rows = closed[~pd.concat([closed[f].isin(configuration.timevarying_positive_values) for f in configuration.structural_timevarying_dims], axis=1).any(axis=1)]
    neutral_rate = neutral_rows[ren].sum() / max(neutral_rows[pipe].sum(), 1)
    ratios = {}
    for flag in configuration.structural_timevarying_dims:
        flagged = closed[closed[flag].isin(configuration.timevarying_positive_values)]
        if len(flagged) and flagged[pipe].sum() > 0:
            ratios[(names or {}).get(flag, flag)] = round(float((flagged[ren].sum() / flagged[pipe].sum()) / max(neutral_rate, 1e-9)), 4)
    out["ratio_tasa_flag_vs_neutra"] = ratios
    monthly = closed.groupby(period).agg(ren=(ren, "sum"), pipe=(pipe, "sum"))
    rate = monthly["ren"] / monthly["pipe"]
    binomial_var = float((rate.mean() * (1 - rate.mean()) / monthly["pipe"]).mean())
    out["phi_agregado"] = round(float(rate.var(ddof=1) / binomial_var), 3) if binomial_var > 0 and len(rate) > 6 else None
    out["sd_mensual_tasa_agregada_pp"] = round(100 * float(rate.std(ddof=1)), 3) if len(rate) > 2 else None
    return out


def profile_volume(units: pd.DataFrame, configuration: Config) -> dict:
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES)]
    monthly = closed.groupby(period)[pipe].sum()
    idx = pd.PeriodIndex(monthly.index, freq="M")
    x = np.arange(len(monthly), dtype=float)
    slope, intercept = np.polyfit(x, np.log(monthly.clip(lower=1).to_numpy(dtype=float)), 1)
    detrended = monthly.to_numpy(dtype=float) / np.exp(intercept + slope * x)
    profile = pd.Series(detrended, index=idx.month).groupby(level=0).mean()
    profile = profile / profile.mean()
    entries = closed.groupby(period)[ren].sum()
    factor = (monthly / entries.shift(12, freq="M").reindex(monthly.index)).replace([np.inf, -np.inf], np.nan).dropna()
    return dict(perfil_mensual=[round(float(profile.get(m, 1.0)), 4) for m in range(1, 13)], tendencia_pct_ano=round(100 * (np.exp(slope * 12) - 1), 3),
                factor_adquisicion_cuantiles=quantiles_of(factor))


def profile_signals(units: pd.DataFrame, configuration: Config) -> dict:
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe = configuration.pipeline_units_col
    real = units[units.get("sintetica", 0) == 0]
    closed, future = real[real[role].isin(TRUTH_ROLES)], real[real[role].isin(FUTURE_ROLES)]
    current = future[period].min() if len(future) else None
    out = {}
    for flag in configuration.structural_timevarying_dims:
        final = float(closed.loc[closed[flag].isin(configuration.timevarying_positive_values), pipe].sum() / max(closed[pipe].sum(), 1))
        by_h = {}
        if current is not None:
            for month, block in future.groupby(period):
                h = int((month - current).n)
                by_h[str(h)] = round(float(block.loc[block[flag].isin(configuration.timevarying_positive_values), pipe].sum() / max(block[pipe].sum(), 1)), 5)
        out[flag] = dict(cuota_final=round(final, 5), cuota_futuro_por_h=by_h)
    return out


def profile_price(fine: pd.DataFrame, configuration: Config) -> dict:
    pipe, usd, ren, ren_usd = configuration.pipeline_units_col, configuration.pipeline_usd_col, configuration.renewed_units_col, configuration.renewed_usd_col
    role = configuration.dataset_role_col
    closed = fine[fine[role].isin(TRUTH_ROLES) & (fine[pipe] > 0)].copy()
    closed["auv"] = closed[usd] / closed[pipe]
    median_auv = float(closed["auv"].median()) if len(closed) else 1.0
    out = dict(auv_cuantiles_relativos={k: round(v / max(median_auv, 1e-9), 4) for k, v in quantiles_of(closed["auv"]).items()})
    renewers = closed[closed[ren].fillna(0) > 0].copy()
    renewers["uplift_fila"] = (renewers[ren_usd] / renewers[ren]) / renewers["auv"]
    column = configuration.discount_value_column
    if column and column in fine.columns:
        discount = pd.to_numeric(closed[column], errors="coerce")
        known = discount.notna()
        out["descuento"] = dict(cuota_desconocido=round(float(1 - known.mean()), 4), cuota_cero=round(float((discount == 0).mean()), 4),
                                positivo_cuantiles=quantiles_of(discount[discount > 0]))
        renewers_discount = pd.to_numeric(renewers[column], errors="coerce")
        buckets = pd.cut(renewers_discount.fillna(-1), bins=[-2, -0.5, 0.0001, 0.2, 0.4, 0.6, 0.8, 1.01], labels=["desconocido", "0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
        by_bucket = {}
        for bucket, block in renewers.groupby(buckets, observed=True):
            observed = float(block[ren_usd].sum() / max((block[ren] * block["auv"]).sum(), 1e-9))
            d = renewers_discount.loc[block.index]
            rule = float((block[ren] * block["auv"] / (1 - d.clip(upper=0.95))).sum() / max((block[ren] * block["auv"]).sum(), 1e-9)) if str(bucket) not in ("desconocido", "0") else 1.0
            by_bucket[str(bucket)] = dict(cuota_renovadores=round(float(block[ren].sum() / max(renewers[ren].sum(), 1)), 4), uplift_observado=round(observed, 4),
                                          ratio_realizacion=round(observed / rule, 4) if rule > 0 else None)
        out["uplift_por_tramo_descuento"] = by_bucket
        out["uplift_desconocido_cuantiles"] = quantiles_of(renewers.loc[renewers_discount.isna(), "uplift_fila"])
    else:
        out["uplift_fila_cuantiles"] = quantiles_of(renewers["uplift_fila"])
    return out


def profile_concentration(series_card: pd.DataFrame) -> dict:
    money = series_card["usd_proyectado"].sort_values(ascending=False).to_numpy(dtype=float)
    total = money.sum() if len(money) else 1.0
    return {f"top_{int(100 * q)}pct": round(float(money[:max(1, int(np.ceil(q * len(money))))].sum() / max(total, 1e-9)), 4) for q in (0.01, 0.1, 0.5)}


def run_topology_profile(fine: pd.DataFrame, units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> dict:
    """The profile end to end. Writes topologia.json and the table topologia."""
    names = role_names(configuration)
    dimensions, _ = profile_dimensions(fine, configuration, names)
    topology = dict(version="sff_v3_topologia_1", roles=dict(mandatory=[names[c] for c in configuration.business_mandatory_dims],
                                                             timevarying={names[c]: p for c, p in configuration.structural_timevarying_dims.items()},
                                                             extra_ren=[names[c] for c in configuration.extra_renovacion], extra_rev=[names[c] for c in configuration.extra_revalorizacion],
                                                             descuento_exacto=bool(configuration.discount_value_column)),
                    calendario=profile_calendar(fine, configuration), dimensiones=dimensions,
                    series=profile_series(units, series_card, configuration), tasas=profile_rates(units, series_card, configuration, names),
                    volumen=profile_volume(units, configuration), senales={names[k]: v for k, v in profile_signals(units, configuration).items()},
                    precio=profile_price(fine, configuration), concentracion=profile_concentration(series_card))
    os.makedirs(configuration.outdir, exist_ok=True)
    path = os.path.join(configuration.outdir, "topologia.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(topology, handle, ensure_ascii=False, indent=1, default=float)
    rows = []
    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, inner in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), inner)
        elif isinstance(value, list):
            rows.append(dict(indicador=prefix, valor=json.dumps(value)))
        else:
            rows.append(dict(indicador=prefix, valor=str(value)))
    flatten("", topology)
    configuration.write(pd.DataFrame(rows), "topologia")
    print_topology_report(topology, fine, units, series_card, configuration)
    figure_path = draw_topology_figure(topology, units, series_card, configuration)
    print(f"[topología] written: {path} (the anonymized profile, input of synthetic_from_topology) · {figure_path} (the six panels to screenshot)")
    explain(configuration,
            "This report is what to screenshot: it describes the SHAPE of the pipeline (series, supports, rates by sign, volume season, signal maturation, discount ↔ price, concentration, extremes) so a synthetic can cover the same assumptions.",
            "The JSON next to it is the same shape without names or values; it is what synthetic_from_topology.build_raw_from_topology reads.")
    return topology


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PRINTED REPORT · what to screenshot so the shape of the pipeline can be reasoned about
# ═══════════════════════════════════════════════════════════════════════════════════

def bar(value: float, scale: float, width: int = 24) -> str:
    """A text bar for the console tables."""
    filled = int(round(width * min(max(value / max(scale, 1e-9), 0.0), 1.0)))
    return "█" * filled + "·" * (width - filled)


def series_volatility(units: pd.DataFrame, configuration: Config) -> pd.Series:
    """Per series: the sd of its monthly rate beyond the binomial floor, in units of that floor
    (φ-like; 1 = pure sampling). Only closed months with support."""
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren = configuration.pipeline_units_col, configuration.renewed_units_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES) & (units[pipe] > 0)].copy()
    closed["rate"] = closed[ren] / closed[pipe]
    out = {}
    for series_id, block in closed.groupby("fs_id"):
        if len(block) < 6:
            continue
        p = float(block["rate"].mean())
        binomial = float(np.mean(p * (1 - p) / block[pipe]))
        out[series_id] = float(np.sqrt(block["rate"].var(ddof=1) / binomial)) if binomial > 0 else np.nan
    return pd.Series(out, name="phi_serie")


def extreme_examples(units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """A dozen forecast units at the extremes: the biggest by money, the smallest with money,
    the most volatile, the youngest (born inside), the ones with the most gaps, the ones with
    a signal that renew worst, the ones with the highest own rate."""
    card = series_card.copy()
    card["phi_serie"] = card["fs_id"].map(series_volatility(units, configuration))
    picks = []
    def take(frame, label, n):
        for _, row in frame.head(n).iterrows():
            picks.append(dict(por_que=label, **row[["fs_id", "signo", "n_propio", "meses_historia", "huecos", "tasa_propia", "phi_serie", "usd_proyectado", "nivel_riesgo", "peldano"]].to_dict()))
    take(card.sort_values("usd_proyectado", ascending=False), "más dinero", 3)
    take(card[(card["usd_proyectado"] > 0) & (card["n_propio"] > 0)].sort_values("n_propio"), "menos soporte con dinero", 2)
    take(card[card["phi_serie"].notna()].sort_values("phi_serie", ascending=False), "más volátil (φ)", 2)
    take(card[card["meses_historia"] < card["meses_historia"].max()].sort_values("meses_historia"), "más joven", 2)
    take(card[card["huecos"] > 0].sort_values("huecos", ascending=False), "más huecos", 2)
    take(card[card["signo"] == SIGN_NEGATIVE].sort_values("tasa_propia"), "señal negativa, peor tasa", 2)
    take(card[card["signo"] == SIGN_NEUTRAL].sort_values("tasa_propia", ascending=False), "neutra, mejor tasa", 1)
    return pd.DataFrame(picks).drop_duplicates("fs_id")


def print_topology_report(topology: dict, fine: pd.DataFrame, units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> None:
    """The report to screenshot: calendar, dimensions, series, rates, volume, signals, price,
    concentration and the extreme examples. Real names: this report is for reasoning, not
    for sharing outside; the JSON next to it is the anonymized version."""
    names = role_names(configuration)
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, usd = configuration.pipeline_units_col, configuration.pipeline_usd_col
    rule = "─" * 100
    print(f"[topología] THE SHAPE OF THE PIPELINE · what a synthetic must reproduce\n{rule}")
    # 1 · calendar and money by role
    print("1 · CALENDAR AND MONEY BY ROLE")
    by_role = fine.groupby(role).agg(meses=(period, "nunique"), unidades=(pipe, "sum"), usd=(usd, "sum"))
    total_usd = float(by_role["usd"].sum())
    for role_name, row in by_role.iterrows():
        print(f"   {role_name:<18} {int(row['meses']):>3} months  {row['unidades']:>12,.0f} units  ${row['usd']:>15,.0f}  {100 * row['usd'] / total_usd:5.1f}%  {bar(row['usd'], total_usd)}")
    # 2 · dimensions
    print(f"{rule}\n2 · DIMENSIONS · cardinality and the share of pipeline $ of the top values")
    for column in list(configuration.business_mandatory_dims) + list(configuration.extra_renovacion) + list(configuration.extra_revalorizacion):
        shares = fine.groupby(column)[usd].sum().sort_values(ascending=False)
        shares = shares / max(shares.sum(), 1e-9)
        nested = topology["dimensiones"].get(names[column], {}).get("anidada_en")
        top = ", ".join(f"{v}={100 * s:.0f}%" for v, s in shares.head(6).items())
        rest = f", other {len(shares) - 6} values={100 * shares.iloc[6:].sum():.0f}%" if len(shares) > 6 else ""
        print(f"   {names[column]:<12} {column:<22} {len(shares):>4} values{' · nested in ' + nested if nested else '':<26} {top}{rest}")
    for column in configuration.structural_timevarying_dims:
        flagged = fine[column].isin(configuration.timevarying_positive_values)
        closed = fine[role].isin(TRUTH_ROLES)
        print(f"   {names[column]:<12} {column:<22} flag  units share closed {100 * fine.loc[flagged & closed, pipe].sum() / max(fine.loc[closed, pipe].sum(), 1):5.1f}% · future {100 * fine.loc[flagged & ~closed, pipe].sum() / max(fine.loc[~closed, pipe].sum(), 1):5.1f}%")
    # 3 · series and support
    series = topology["series"]
    print(f"{rule}\n3 · SERIES AND SUPPORT · {series['n_series']} predictable series · {100 * series['cuota_nacen_dentro']:.0f}% born inside the history")
    print("   support quantiles (median monthly units due): " + "  ".join(f"p{int(100 * float(q))}={v:,.0f}" for q, v in series["soporte_cuantiles"].items()))
    print("   months of history quantiles:                  " + "  ".join(f"p{int(100 * float(q))}={v:,.0f}" for q, v in series["meses_historia_cuantiles"].items()))
    card = series_card[series_card["ruta"] == TREATMENT_PREDICTABLE] if "ruta" in series_card.columns else series_card
    total_money = max(card["usd_proyectado"].sum(), 1e-9)
    for low, high in SUPPORT_BUCKETS:
        block = card[(card["n_propio"] >= low) & (card["n_propio"] < high)]
        label = f"{low}-{high if np.isfinite(high) else '∞'}"
        print(f"   dial {label:<9} {len(block):>6} series  {100 * len(block) / max(len(card), 1):5.1f}%  ${block['usd_proyectado'].sum():>14,.0f}  {100 * block['usd_proyectado'].sum() / total_money:5.1f}% of the money  {bar(block['usd_proyectado'].sum(), total_money)}"
              + (f"  gaps {100 * series['cuota_huecos_por_tramo'].get(label.replace('∞', 'inf'), 0):.0f}% of months" if series["cuota_huecos_por_tramo"] else ""))
    print("   series by sign: " + " · ".join(f"{k}: {100 * v:.0f}%" for k, v in series["cuota_series_por_signo"].items()))
    # 4 · rates
    rates = topology["tasas"]
    print(f"{rule}\n4 · RATES · neutral series' own rate, quantiles: " + "  ".join(f"p{int(100 * float(q))}={100 * v:.1f}%" for q, v in rates["tasa_neutra_cuantiles"].items()))
    print(f"   weighted mean {100 * (rates['tasa_neutra_media_ponderada'] or 0):.1f}% · rate of flagged vs neutral: " + " · ".join(f"{k}: ×{v:.2f}" for k, v in rates["ratio_tasa_flag_vs_neutra"].items())
          + f" · aggregate φ {rates['phi_agregado']} (1 = pure sampling) · monthly sd of the aggregate rate {rates['sd_mensual_tasa_agregada_pp']} pp")
    # 5 · volume
    volume = topology["volumen"]
    print(f"{rule}\n5 · VOLUME · units due by month of the year (1 = the mean) · trend {volume['tendencia_pct_ano']:+.1f}%/year")
    for month, index in enumerate(volume["perfil_mensual"], start=1):
        print(f"   {'EFMAMJJASOND'[month - 1]}  {index:5.2f}  {bar(index, 1.5, 30)}")
    if volume["factor_adquisicion_cuantiles"]:
        print("   acquisition factor pipeline(t) / renewed(t−12), quantiles: " + "  ".join(f"p{int(100 * float(q))}={v:.2f}" for q, v in volume["factor_adquisicion_cuantiles"].items()))
    # 6 · signals maturation
    print(f"{rule}\n6 · SIGNALS · share of units with the flag: final (closed months) vs today in the future, by distance h")
    for flag, block in topology["senales"].items():
        by_h = block["cuota_futuro_por_h"]
        line = "  ".join(f"h{h}={100 * v:.1f}%" for h, v in list(by_h.items())[:8])
        print(f"   {flag:<10} final {100 * block['cuota_final']:5.1f}%  · future {line}")
    # 7 · price
    price = topology["precio"]
    closed_rows = fine[fine[role].isin(TRUTH_ROLES) & (fine[pipe] > 0)]
    median_auv = float((closed_rows[usd] / closed_rows[pipe]).median()) if len(closed_rows) else np.nan
    print(f"{rule}\n7 · PRICE · AUV median ${median_auv:,.2f} · relative quantiles: " + "  ".join(f"p{int(100 * float(q))}={v:.2f}" for q, v in price["auv_cuantiles_relativos"].items()))
    if "descuento" in price:
        d = price["descuento"]
        print(f"   discount: unknown {100 * d['cuota_desconocido']:.0f}% of rows · zero {100 * d['cuota_cero']:.0f}% · positive quantiles: " + "  ".join(f"p{int(100 * float(q))}={100 * v:.0f}%" for q, v in d["positivo_cuantiles"].items()))
        print("   renewed price by discount bucket: share of renewers · observed uplift · realization vs 1/(1−d)")
        for bucket, block in price["uplift_por_tramo_descuento"].items():
            print(f"      {bucket:<12} {100 * block['cuota_renovadores']:5.1f}%   uplift {block['uplift_observado']:5.2f}   realization {block['ratio_realizacion'] if block['ratio_realizacion'] is not None else '—'}")
        if price.get("uplift_desconocido_cuantiles"):
            print("   uplift of unknown-discount rows, quantiles: " + "  ".join(f"p{int(100 * float(q))}={v:.2f}" for q, v in price["uplift_desconocido_cuantiles"].items()))
    else:
        print("   row uplift quantiles: " + "  ".join(f"p{int(100 * float(q))}={v:.2f}" for q, v in price.get("uplift_fila_cuantiles", {}).items()))
    # 8 · concentration
    concentration = topology["concentracion"]
    print(f"{rule}\n8 · CONCENTRATION · the top 1% of series hold {100 * concentration['top_1pct']:.0f}% of the projected money, the top 10% {100 * concentration['top_10pct']:.0f}%, the top 50% {100 * concentration['top_50pct']:.0f}%")
    # 9 · extremes
    examples = extreme_examples(units, series_card, configuration)
    print(f"{rule}\n9 · EXTREME FORECAST UNITS · the ones that stretch the assumptions")
    print(f"   {'why':<26} {'series':<44} {'sign':<9} {'n':>6} {'months':>6} {'gaps':>4} {'rate':>6} {'φ':>5} {'$ proj':>12} {'level':<20} rung")
    for _, row in examples.iterrows():
        print(f"   {row['por_que']:<26} {str(row['fs_id'])[:44]:<44} {str(row['signo']):<9} {row['n_propio']:>6,.0f} {row['meses_historia']:>6,.0f} {row['huecos']:>4,.0f} "
              f"{100 * row['tasa_propia'] if pd.notna(row['tasa_propia']) else float('nan'):>5.1f}% {row['phi_serie'] if pd.notna(row['phi_serie']) else float('nan'):>5.1f} ${row['usd_proyectado']:>11,.0f} {str(row['nivel_riesgo']):<20} {row['peldano']}")
    print(rule)


def draw_topology_figure(topology: dict, units: pd.DataFrame, series_card: pd.DataFrame, configuration: Config) -> str:
    """Six panels to screenshot: support × money, rate by sign, volume profile, signal
    maturation by h, uplift by discount bucket, concentration (Lorenz)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    card = series_card[series_card["ruta"] == TREATMENT_PREDICTABLE] if "ruta" in series_card.columns else series_card
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    # support × money
    axis = axes[0, 0]
    positive = card[card["n_propio"] > 0]
    axis.hist(np.log10(positive["n_propio"]), bins=30, weights=positive["usd_proyectado"], color="#2E7D7D", alpha=0.85)
    for x, label in ((np.log10(30), "30"), (np.log10(271), "271"), (np.log10(752), "752")):
        axis.axvline(x, color="black", linestyle="--", linewidth=0.8); axis.text(x, axis.get_ylim()[1] * 0.95, label, fontsize=8, ha="center")
    axis.set_xlabel("log10(monthly support)"); axis.set_ylabel("projected $"); axis.set_title("Where the money is on the dial", loc="left", fontsize=10)
    # rate by sign
    axis = axes[0, 1]
    for index, (sign, block) in enumerate(card[card["tasa_propia"].notna()].groupby("signo")):
        axis.hist(100 * block["tasa_propia"], bins=25, alpha=0.6, label=f"{sign} ({len(block)})", weights=block["n_propio"].clip(lower=1))
    axis.set_xlabel("own rate, %"); axis.set_title("Own rate of the series by sign (weighted by support)", loc="left", fontsize=10); axis.legend(fontsize=8)
    # volume profile
    axis = axes[0, 2]
    profile = topology["volumen"]["perfil_mensual"]
    axis.bar(range(1, 13), profile, color="#3B8BC8", alpha=0.85); axis.axhline(1, color="black", linewidth=0.8)
    axis.set_xticks(range(1, 13)); axis.set_xticklabels(list("EFMAMJJASOND"))
    axis.set_title(f"Units due by month of the year (trend {topology['volumen']['tendencia_pct_ano']:+.1f}%/yr)", loc="left", fontsize=10)
    # maturation
    axis = axes[1, 0]
    for index, (flag, block) in enumerate(topology["senales"].items()):
        by_h = block["cuota_futuro_por_h"]
        if by_h:
            hs = [int(h) for h in by_h]; axis.plot(hs, [100 * v for v in by_h.values()], "-o", markersize=3, label=f"{flag} (final {100 * block['cuota_final']:.1f}%)")
            axis.axhline(100 * block["cuota_final"], color=f"C{index}", linestyle=":", linewidth=0.8)
    axis.set_xlabel("months ahead (h)"); axis.set_ylabel("% of units with the flag"); axis.set_title("Signals today vs final: the maturation", loc="left", fontsize=10); axis.legend(fontsize=7)
    # uplift by discount
    axis = axes[1, 1]
    buckets = topology["precio"].get("uplift_por_tramo_descuento", {})
    if buckets:
        labels = list(buckets); observed = [buckets[b]["uplift_observado"] for b in labels]
        axis.bar(range(len(labels)), observed, color="#E8B84F", alpha=0.85)
        rule_values = [1 / (1 - float(b.split("-")[0]) - 0.1) if "-" in b and b[0].isdigit() else 1.0 for b in labels]
        axis.plot(range(len(labels)), rule_values, "k--", linewidth=1, label="rule 1/(1−d) at bucket midpoint")
        axis.set_xticks(range(len(labels))); axis.set_xticklabels(labels, fontsize=8, rotation=20); axis.legend(fontsize=8)
    axis.set_title("Renewed price / pipeline price by discount bucket", loc="left", fontsize=10)
    # Lorenz
    axis = axes[1, 2]
    money = np.sort(card["usd_proyectado"].to_numpy(dtype=float))[::-1]
    if money.sum() > 0:
        cumulative = np.cumsum(money) / money.sum()
        axis.plot(np.linspace(0, 100, len(money)), 100 * cumulative, color="#C8553D", linewidth=2); axis.plot([0, 100], [0, 100], "k:", linewidth=0.8)
    axis.set_xlabel("% of series (largest first)"); axis.set_ylabel("% of projected $"); axis.set_title("Concentration of the money", loc="left", fontsize=10)
    figure.suptitle("The shape of the pipeline", fontsize=13, x=0.01, ha="left")
    figure.tight_layout()
    path = os.path.join(configuration.outdir, "topologia.png")
    figure.savefig(path, dpi=110); plt.close(figure)
    return path
