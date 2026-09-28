"""analysis_data_profile.py — SFF v3 · ANALYSIS · the two levels BELOW the series.

LEVEL 0 · the raw (`raw_profile`, `dimension_domains`): is the extract coherent with what
the configuration says about it?
  · calendar by role: which months each role has, calendar holes, contiguity of test and
    projection, projection starting at the current month;
  · dimension domains: cardinality and empties per declared column, and their STABILITY
    in time — a value that appears or disappears inside the history is a portfolio move
    or a catalogue change (`dimension_domains`, one row per dimension × value);
  · measure coherence per row: renewed > pipeline, negative values, pipeline 0 with
    renewals, AUV (pipeline and renewed) outside a plausible range;
  · coverage of the term mapping: values of `term_column` not in `term_months_by_value`
    and the money they carry;
  · shares: reacquisitions, time_series universe.

LEVEL 1 · the forecast units (`fu_profile`, `dial_buckets`): is there something to
predict with?
  · completeness per series: first / last month, months, gaps, series BORN or DEAD
    inside the history;
  · the dial applied to the units: money by monthly-support bucket (< 30, 30-271,
    271-752, ≥ 752) BEFORE any support is borrowed — the raw picture;
  · revaluation combinations per unit: how many, and the weight of the largest;
  · extreme months: 0 % or 100 % with a small pipeline;
  · unit ↔ combination consistency per unit.

Both are read, not decided: nothing downstream changes because of them. They are the
tables that later explain the odd cases (cells where the flat method wins, absurd
uplifts, "seasonal" series with one cycle).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from binomial_reference import support_for_half_width
from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
AUV_PLAUSIBLE_RANGE = (1.0, 10_000.0)          # pipeline / renewed AUV outside this is a data problem
UPLIFT_ROW_PLAUSIBLE_RANGE = (0.3, 3.0)        # renewed AUV / pipeline AUV per row
SMALL_PIPELINE_FOR_EXTREME_MONTH = 10          # a 0 % or 100 % month with fewer units than this is sampling
DIAL_HALF_WIDTHS_PP = (15.0, 5.0, 3.0)         # the dial: ±15 / ±5 / ±3 pp at 90 %
MIN_ROWS_TO_REPORT_A_VALUE = 1


def profile_row(section: str, concept: str, value, detail: str = "") -> dict:
    return dict(seccion=section, concepto=concept, valor=str(value), detalle=detail)


# ═══════════════════════════════════════════════════════════════════════════════════
# LEVEL 0 · RAW
# ═══════════════════════════════════════════════════════════════════════════════════

def calendar_by_role(raw: pd.DataFrame, configuration: Config) -> list:
    """Months per role, calendar holes, contiguity, and the current month as the first projection month."""
    period, role = configuration.period_col, configuration.dataset_role_col
    rows = []
    months_all = pd.PeriodIndex(sorted(raw[period].unique()), freq="M")
    full = pd.period_range(months_all.min(), months_all.max(), freq="M")
    holes = full.difference(months_all)
    rows.append(profile_row("calendario", "meses en el raw", len(months_all), f"{months_all.min()} .. {months_all.max()}"))
    rows.append(profile_row("calendario", "meses de calendario sin ninguna fila", len(holes), ", ".join(map(str, holes[:12]))))
    for role_value, block in raw.groupby(role):
        months = pd.PeriodIndex(sorted(block[period].unique()), freq="M")
        contiguous = len(months) == len(pd.period_range(months.min(), months.max(), freq="M"))
        rows.append(profile_row("calendario", f"rol {role_value}", f"{months.min()} .. {months.max()} ({len(months)} meses)",
                                "contiguo" if contiguous else "CON HUECOS"))
    projection_months = raw.loc[raw[role] == ROLE_PROJECTION, period]
    current = raw.loc[raw[configuration.current_month_col].isin([1, True, "1"]), period]
    if len(projection_months) and len(current):
        first_projection, current_month = projection_months.min(), current.min()
        rows.append(profile_row("calendario", "la proyección empieza en el mes en curso", int(first_projection == current_month),
                                f"proyección desde {first_projection}, mes en curso {current_month}"))
    train_max = raw.loc[raw[role] == ROLE_TRAIN, period].max() if (raw[role] == ROLE_TRAIN).any() else None
    test_min = raw.loc[raw[role] == ROLE_TEST, period].min() if (raw[role] == ROLE_TEST).any() else None
    if train_max is not None and test_min is not None:
        rows.append(profile_row("calendario", "test empieza justo después de train", int((test_min - train_max).n == 1),
                                f"train hasta {train_max}, test desde {test_min}"))
    return rows


def dimension_domains(raw: pd.DataFrame, configuration: Config) -> tuple:
    """Per declared dimension: cardinality and empties (profile rows); per dimension × value:
    rows, money, first / last month, and whether it appears or disappears INSIDE the history."""
    period, usd = configuration.period_col, configuration.pipeline_usd_col
    history = raw[raw[configuration.dataset_role_col].isin(TRUTH_ROLES)]
    first_month, last_month = history[period].min(), history[period].max()
    dims = (configuration.business_mandatory_dims + list(configuration.structural_timevarying_dims)
            + configuration.extra_renovacion + configuration.extra_revalorizacion)
    rows, domain_rows = [], []
    for dim in dims:
        values = raw[dim]
        empties = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        rows.append(profile_row("dimensiones", f"{dim}: valores distintos", values.nunique(dropna=True),
                                f"{empties} filas vacías/nulas" if empties else ""))
        for value, block in history.groupby(dim, dropna=False):
            first, last = block[period].min(), block[period].max()
            domain_rows.append(dict(dimension=dim, valor=str(value), filas=len(block), usd=round(float(block[usd].sum()), 2),
                                    primer_mes=str(first), ultimo_mes=str(last), meses=block[period].nunique(),
                                    aparece_dentro=int(first > first_month), desaparece_dentro=int(last < last_month)))
    domains = pd.DataFrame(domain_rows)
    if len(domains):
        moving = domains[(domains["aparece_dentro"] == 1) | (domains["desaparece_dentro"] == 1)]
        rows.append(profile_row("dimensiones", "valores que aparecen o desaparecen dentro de la historia", len(moving),
                                f"${moving['usd'].sum():,.0f} de pipeline histórico; ver dimension_domains"))
    return rows, domains


def measure_coherence(raw: pd.DataFrame, configuration: Config) -> list:
    """Per-row checks on the measures, with counts and money."""
    pipe_u, pipe_usd = configuration.pipeline_units_col, configuration.pipeline_usd_col
    ren_u, ren_usd = configuration.renewed_units_col, configuration.renewed_usd_col
    history = raw[raw[configuration.dataset_role_col].isin(TRUTH_ROLES)]
    rows = []
    over = history[history[ren_u] > history[pipe_u]]
    rows.append(profile_row("medidas", "filas con renovados > pipeline", len(over), f"${over[pipe_usd].sum():,.0f}"))
    negative = raw[(raw[[pipe_u, pipe_usd]] < 0).any(axis=1) | (raw[[ren_u, ren_usd]].fillna(0) < 0).any(axis=1)]
    rows.append(profile_row("medidas", "filas con medidas negativas", len(negative)))
    ghost = history[(history[pipe_u] <= 0) & (history[ren_u].fillna(0) > 0)]
    rows.append(profile_row("medidas", "filas con pipeline 0 y renovados > 0", len(ghost), f"{ghost[ren_u].sum():,.0f} renovados sin vencimiento"))
    zero_pipe = raw[raw[pipe_u] <= 0]
    rows.append(profile_row("medidas", "filas con pipeline 0", len(zero_pipe)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pipeline_auv = raw[pipe_usd] / raw[pipe_u].replace(0, np.nan)
        renewed_auv = history[ren_usd] / history[ren_u].replace(0, np.nan)
        row_uplift = renewed_auv / (history[pipe_usd] / history[pipe_u].replace(0, np.nan))
    low, high = AUV_PLAUSIBLE_RANGE
    odd_auv = raw[(pipeline_auv < low) | (pipeline_auv > high)]
    rows.append(profile_row("medidas", f"filas con AUV de pipeline fuera de [{low}, {high}]", len(odd_auv), f"${odd_auv[pipe_usd].sum():,.0f}"))
    rows.append(profile_row("medidas", "AUV de pipeline: p5 / mediana / p95",
                            f"{pipeline_auv.quantile(.05):.2f} / {pipeline_auv.median():.2f} / {pipeline_auv.quantile(.95):.2f}"))
    ulow, uhigh = UPLIFT_ROW_PLAUSIBLE_RANGE
    odd_uplift = history[(row_uplift < ulow) | (row_uplift > uhigh)]
    rows.append(profile_row("medidas", f"filas con uplift de fila fuera de [{ulow}, {uhigh}]", len(odd_uplift),
                            f"${odd_uplift[ren_usd].sum():,.0f} renovados; son el origen de los uplifts recortados"))
    rows.append(profile_row("medidas", "uplift de fila: p5 / mediana / p95",
                            f"{row_uplift.quantile(.05):.3f} / {row_uplift.median():.3f} / {row_uplift.quantile(.95):.3f}"))
    reacquired = configuration.reacq_usd_col if hasattr(configuration, "reacq_usd_col") else None
    if reacquired and reacquired in history.columns:
        share = history[reacquired].fillna(0).sum() / max(history[ren_usd].fillna(0).sum(), 1e-9)
        rows.append(profile_row("medidas", "readquisiciones sobre renovado $", f"{share:.1%}"))
    if configuration.flag_time_series_col in raw.columns:
        ts = raw[raw[configuration.flag_time_series_col].isin([1, True, "1"])]
        rows.append(profile_row("medidas", "universo time_series", f"{len(ts):,} filas", f"${ts[pipe_usd].sum():,.0f} de pipeline"))
    return rows


def term_mapping_coverage(raw: pd.DataFrame, configuration: Config) -> list:
    """Values of the term column not covered by the mapping, and their money."""
    if not configuration.term_column or configuration.term_column not in raw.columns:
        return [profile_row("plazo", "columna de plazo", "no declarada", "todas las filas reentran a renewal_term_months")]
    mapped = {str(k) for k in configuration.term_months_by_value}
    values = raw[configuration.term_column].astype(str)
    unmapped = raw[~values.isin(mapped)]
    rows = [profile_row("plazo", f"valores de {configuration.term_column}", ", ".join(sorted(values.unique())[:12]))]
    rows.append(profile_row("plazo", "filas con plazo sin mapear (caen en el defecto)", len(unmapped),
                            f"${unmapped[configuration.pipeline_usd_col].sum():,.0f} · valores {sorted(unmapped[configuration.term_column].astype(str).unique())[:8]}"))
    return rows


def run_raw_profile(conditioned_raw: pd.DataFrame, configuration: Config) -> dict:
    """Level 0 end to end. Persists raw_profile and dimension_domains; prints the headline."""
    rows = calendar_by_role(conditioned_raw, configuration)
    domain_rows, domains = dimension_domains(conditioned_raw, configuration)
    rows += domain_rows + measure_coherence(conditioned_raw, configuration) + term_mapping_coverage(conditioned_raw, configuration)
    profile = pd.DataFrame(rows)
    configuration.write(profile, "raw_profile")
    configuration.write(domains, "dimension_domains")
    print("[0.4] raw profile (level 0):")
    for _, row in profile.iterrows():
        if row["seccion"] == "dimensiones" and row["concepto"].endswith("valores distintos"):
            continue
        print(f"   {row['seccion']:<12} {row['concepto']:<58} {row['valor']:<28} {row['detalle'][:70]}")
    if len(domains):
        moving = domains[(domains["aparece_dentro"] == 1) | (domains["desaparece_dentro"] == 1)].sort_values("usd", ascending=False)
        for _, row in moving.head(configuration.console_top_rows).iterrows():
            kind = "aparece" if row["aparece_dentro"] else "desaparece"
            print(f"   {'movimiento':<12} {row['dimension']}={row['valor'][:30]:<30} {kind} {row['primer_mes']}..{row['ultimo_mes']} ${row['usd']:,.0f}")
    return dict(raw_profile=profile, dimension_domains=domains)


# ═══════════════════════════════════════════════════════════════════════════════════
# LEVEL 1 · FORECAST UNITS
# ═══════════════════════════════════════════════════════════════════════════════════

def dial_bucket(support: float, thresholds: tuple) -> str:
    """The dial bucket of a monthly support: below the first threshold, between, or above
    the last. The numeric prefix keeps the buckets in order when sorted as text."""
    if support < thresholds[0]:
        return f"0_menos_de_{int(thresholds[0])}"
    for index, (lower, upper) in enumerate(zip(thresholds, thresholds[1:]), start=1):
        if support < upper:
            return f"{index}_{int(lower)}_a_{int(upper)}"
    return f"{len(thresholds)}_desde_{int(thresholds[-1])}"


def series_completeness(units: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per series: months, gaps, birth/death inside the history, support, dial
    bucket, combinations per unit, extreme months, unit ↔ combination consistency.
    NOTE: vectorized (one groupby per measure); the per-series loop took minutes on 10k series."""
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe_u, pipe_usd, ren_u = configuration.pipeline_units_col, configuration.pipeline_usd_col, configuration.renewed_units_col
    real = units[units.get("sintetica", 0) == 0] if "sintetica" in units.columns else units
    history = real[real[role].isin(TRUTH_ROLES)].copy()
    if history.empty:
        return pd.DataFrame(columns=["fs_id"])
    first_month, last_month = history[period].min(), history[period].max()
    thresholds = tuple(round(support_for_half_width(w, configuration.z)) for w in DIAL_HALF_WIDTHS_PP)
    # months, span, support
    grouped = history.groupby("fs_id")
    profile = grouped.agg(primer_mes=(period, "min"), ultimo_mes=(period, "max"), meses=(period, "nunique")).reset_index()
    profile["huecos"] = [(last - first).n + 1 - months for first, last, months in zip(profile["primer_mes"], profile["ultimo_mes"], profile["meses"])]
    profile["nace_dentro"] = (profile["primer_mes"] > first_month).astype(int)
    profile["muere_dentro"] = (profile["ultimo_mes"] < last_month).astype(int)
    support = history[history[pipe_u] > 0].groupby("fs_id")[pipe_u].median()
    profile["n_mediana"] = profile["fs_id"].map(support).fillna(0.0).round(1)
    profile["tramo_dial"] = profile["n_mediana"].map(lambda n: dial_bucket(n, thresholds))
    projected = real[real[role].isin((ROLE_PROJECTION, ROLE_PENDING))].groupby("fs_id")[pipe_usd].sum()
    profile["usd_proyectado"] = profile["fs_id"].map(projected).fillna(0.0).round(2)
    # combinations per unit
    combos = fine_table[fine_table[role].isin(TRUTH_ROLES)].groupby("fu_id").agg(
        combinaciones=("comb_id", "nunique"), usd_fina=(pipe_usd, "sum"), mayor=(pipe_usd, "max"))
    combos["peso_mayor"] = combos["mayor"] / combos["usd_fina"].replace(0, np.nan)
    per_unit = history[["fs_id", "fu_id", pipe_usd, pipe_u, ren_u]].merge(combos, left_on="fu_id", right_index=True, how="left")
    by_series = per_unit.groupby("fs_id")
    profile["combinaciones_mediana"] = profile["fs_id"].map(by_series["combinaciones"].median())
    profile["combinaciones_max"] = profile["fs_id"].map(by_series["combinaciones"].max()).fillna(0).astype(int)
    profile["peso_combo_mayor"] = profile["fs_id"].map(by_series["peso_mayor"].median()).round(3)
    # extreme months on a small pipeline
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = per_unit[ren_u] / per_unit[pipe_u].replace(0, np.nan)
    small = per_unit[pipe_u] < SMALL_PIPELINE_FOR_EXTREME_MONTH
    per_unit["zero"] = ((rate == 0) & small).astype(int)
    per_unit["full"] = ((rate == 1) & small).astype(int)
    per_unit["mismatch"] = ((per_unit[pipe_usd] - per_unit["usd_fina"].fillna(0)).abs() > 1e-6).astype(int)
    extremes = per_unit.groupby("fs_id")[["zero", "full", "mismatch"]].sum()
    profile["meses_0pct"] = profile["fs_id"].map(extremes["zero"]).fillna(0).astype(int)
    profile["meses_100pct"] = profile["fs_id"].map(extremes["full"]).fillna(0).astype(int)
    profile["unidades_descuadradas"] = profile["fs_id"].map(extremes["mismatch"]).fillna(0).astype(int)
    profile["primer_mes"], profile["ultimo_mes"] = profile["primer_mes"].astype(str), profile["ultimo_mes"].astype(str)
    return profile


def dial_buckets(profile: pd.DataFrame) -> pd.DataFrame:
    """Money and series by dial bucket: the raw picture before any support is borrowed."""
    total = max(profile["usd_proyectado"].sum(), 1e-9)
    buckets = profile.groupby("tramo_dial").agg(series=("fs_id", "size"), usd=("usd_proyectado", "sum")).reset_index()
    buckets["pct_usd"] = (100 * buckets["usd"] / total).round(1)
    return buckets.sort_values("tramo_dial")


def run_fu_profile(units: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> dict:
    """Level 1 end to end. Persists fu_profile and dial_buckets; prints the headline."""
    profile = series_completeness(units, fine_table, configuration)
    buckets = dial_buckets(profile)
    configuration.write(profile, "fu_profile")
    configuration.write(buckets, "dial_buckets")
    born, dead = profile[profile["nace_dentro"] == 1], profile[profile["muere_dentro"] == 1]
    print(f"[1.1b] fu profile (level 1): {len(profile)} series with history · "
          f"{len(born)} born inside the history (${born['usd_proyectado'].sum():,.0f} projected) · "
          f"{len(dead)} dead before its end (${dead['usd_proyectado'].sum():,.0f} projected) · "
          f"{int(profile['unidades_descuadradas'].sum())} units where fine and unit money differ")
    print("[1.1b] the dial on the units (monthly support, before borrowing): bucket · series · $ projected · %")
    for _, row in buckets.iterrows():
        print(f"   {row['tramo_dial']:<20} {int(row['series']):>6} series  ${row['usd']:>13,.0f}  {row['pct_usd']:5.1f}%")
    many = profile[profile["combinaciones_max"] >= 10]
    extreme = profile[(profile["meses_0pct"] + profile["meses_100pct"]) >= 3]
    print(f"[1.1b] combinations: median per unit {profile['combinaciones_mediana'].median():.0f}, {len(many)} series with ≥ 10 in a unit · "
          f"{len(extreme)} series with ≥ 3 months at 0 % or 100 % on a small pipeline")
    return dict(fu_profile=profile, dial_buckets=buckets)
