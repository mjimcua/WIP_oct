"""analysis_composition.py — SFF v3 · the composition effect (one function, two moments).

DECISION. We do not hunt for cases of Simpson's paradox (sign reversals): they are rare
and they are not what matters. We always measure how much of the change of the
aggregate renewal rate is due to COMPOSITION (different customers fall due) and how much
to BEHAVIOUR (the same customers renew differently). "Simpson" appears only in
METODOS.md as the theory behind.

THE ONE FUNCTION. Between two periods 0 and 1, with cells i, weight w (share of the
pipeline units) and rate r (Kitagawa decomposition):

    total change     = r_aggregate_1 − r_aggregate_0
    composition      = Σ (w1ᵢ − w0ᵢ) · (r0ᵢ + r1ᵢ) / 2
    behaviour        = Σ (r1ᵢ − r0ᵢ) · (w0ᵢ + w1ᵢ) / 2
    composition + behaviour = total change  (exact; a test checks it)
    in dollars: each effect × pipeline $ of period 1

Example. Autorenewal renews at 90 %, manual at 40 %. Year 0: 70 % of the pipeline is
autorenewal → rate 75 %. Year 1: it falls to 60 %, the rates of both groups unchanged →
rate 70 %. Change −5 pp, all composition: (0.6 − 0.7)·0.9 + (0.4 − 0.3)·0.4 = −0.05.
With $100M of pipeline, a single rate would be wrong by $5M with nobody changing.

FOUR LEVELS OF SEGMENTATION, same pair of periods: regional_level_1 (context); the
mandatory cell (the reference: the business method); mandatory × who the customer is
(net_new); mandatory × who × how the customer is (the sign of the signals). What the
mandatory cell leaves as "behaviour" and the finer levels move to "composition" is what
knowing the customer contributes. Headline: the $ a forecast by mandatory cell would
have missed just by the composition of customer type and state.

TWO MOMENTS. (1) The panorama, historical: each closed year against the previous, each
month against the same month a year earlier (this feeds the "rate at fixed composition"
line). (2) After the forecast: this year against last, next year against this one — "of
what changes in the projected rate, how much is the portfolio and how much is people".

Tables: efecto_composicion (comparison × level × scope), efecto_composicion_celda (the
contribution of every segment, for audit).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
LEVEL_REGION = "regional"
LEVEL_MANDATORY = "mandatory"
LEVEL_WHO = "mandatory+quien"
LEVEL_WHO_HOW = "mandatory+quien+como"
LEVELS_IN_ORDER = (LEVEL_REGION, LEVEL_MANDATORY, LEVEL_WHO, LEVEL_WHO_HOW)
SCOPE_TOTAL = "total"
TOP_CONTRIBUTIONS = 25


def decompose_rate_change(frame: pd.DataFrame, cell_columns: list, period_a, period_b, configuration: Config,
                          period_column: str = None) -> tuple:
    """Kitagawa decomposition of the change of the aggregate rate between two periods.

    INPUT:   frame — rows with the cell columns, a period column, pipeline units, renewed
             units and pipeline $ (any grain: fine rows, units, aggregates) ·
             cell_columns — the columns that define a cell at this level of segmentation
             (empty list = one cell: everything is behaviour) · period_a, period_b — the
             two values of `period_column` to compare (a year, a month, a label) ·
             period_column — defaults to configuration.period_col.
    OUTPUT:  (summary dict, contributions DataFrame). summary: tasa_0, tasa_1,
             cambio_total_pp, composicion_pp, comportamiento_pp, pct_composicion,
             composicion_usd, comportamiento_usd, pipeline_usd_1, celdas. contributions:
             one row per cell: w0, w1, r0, r1, composicion_pp, comportamiento_pp,
             composicion_usd, comportamiento_usd.
    RULES:   weights = share of pipeline UNITS of the period; a cell absent in one period
             has weight 0 and takes the rate of the other period (its rate difference is
             then 0: the whole effect is composition, which is what it is). Dollars =
             effect × pipeline $ of period 1. composition + behaviour = total change,
             exactly, by construction.
    EDGE CASES: no rows in one of the periods → NaNs; a single cell → composition 0.
    """
    period_column = period_column or configuration.period_col
    pipe, ren, usd = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.pipeline_usd_col
    if cell_columns:
        keyed = frame.assign(_cell=join_columns(frame, cell_columns))
    else:
        keyed = frame.assign(_cell="*")
    per_cell = keyed.groupby([period_column, "_cell"]).agg(units=(pipe, "sum"), renewed=(ren, "sum"), usd=(usd, "sum"))
    a = per_cell.xs(period_a, level=period_column) if period_a in per_cell.index.get_level_values(0) else per_cell.iloc[0:0].droplevel(0)
    b = per_cell.xs(period_b, level=period_column) if period_b in per_cell.index.get_level_values(0) else per_cell.iloc[0:0].droplevel(0)
    cells = sorted(set(a.index) | set(b.index))
    if not cells or a["units"].sum() == 0 or b["units"].sum() == 0:
        return dict(tasa_0=np.nan, tasa_1=np.nan, cambio_total_pp=np.nan, composicion_pp=np.nan, comportamiento_pp=np.nan,
                    pct_composicion=np.nan, composicion_usd=np.nan, comportamiento_usd=np.nan, pipeline_usd_1=float(b["usd"].sum()) if len(b) else 0.0, celdas=len(cells)), pd.DataFrame()
    a, b = a.reindex(cells), b.reindex(cells)
    w0 = (a["units"] / a["units"].sum()).fillna(0.0)
    w1 = (b["units"] / b["units"].sum()).fillna(0.0)
    r0 = (a["renewed"] / a["units"].replace(0, np.nan))
    r1 = (b["renewed"] / b["units"].replace(0, np.nan))
    r0 = r0.fillna(r1)                   # absent in 0: takes the rate of 1 (no behaviour change)
    r1 = r1.fillna(r0)
    composition = (w1 - w0) * (r0 + r1) / 2
    behaviour = (r1 - r0) * (w0 + w1) / 2
    rate_0 = float(a["renewed"].sum() / a["units"].sum())
    rate_1 = float(b["renewed"].sum() / b["units"].sum())
    pipeline_usd_1 = float(b["usd"].sum())
    contributions = pd.DataFrame(dict(celda=cells, w0=w0.to_numpy(), w1=w1.to_numpy(), r0=r0.to_numpy(), r1=r1.to_numpy(),
                                      composicion_pp=100 * composition.to_numpy(), comportamiento_pp=100 * behaviour.to_numpy(),
                                      composicion_usd=composition.to_numpy() * pipeline_usd_1, comportamiento_usd=behaviour.to_numpy() * pipeline_usd_1))
    total_change = 100 * (rate_1 - rate_0)
    summary = dict(tasa_0=rate_0, tasa_1=rate_1, cambio_total_pp=total_change,
                   composicion_pp=100 * float(composition.sum()), comportamiento_pp=100 * float(behaviour.sum()),
                   pct_composicion=100 * abs(float(composition.sum())) / max(abs(float(composition.sum())) + abs(float(behaviour.sum())), 1e-12),
                   composicion_usd=float(composition.sum()) * pipeline_usd_1, comportamiento_usd=float(behaviour.sum()) * pipeline_usd_1,
                   pipeline_usd_1=pipeline_usd_1, celdas=len(cells))
    return summary, contributions


def sign_of_rows(frame: pd.DataFrame, configuration: Config) -> pd.Series:
    """'neutro', 'negativo', 'positivo' or 'mixto' per row from its timevarying flags."""
    negatives = [f for f, polarity in configuration.structural_timevarying_dims.items() if polarity == "negative"]
    positives = [f for f, polarity in configuration.structural_timevarying_dims.items() if polarity == "positive"]
    negative = pd.concat([frame[f].isin(configuration.timevarying_positive_values) for f in negatives], axis=1).any(axis=1) if negatives else pd.Series(False, index=frame.index)
    positive = pd.concat([frame[f].isin(configuration.timevarying_positive_values) for f in positives], axis=1).any(axis=1) if positives else pd.Series(False, index=frame.index)
    return pd.Series(np.select([negative & positive, negative, positive], [SIGN_MIXED, SIGN_NEGATIVE, SIGN_POSITIVE], default=SIGN_NEUTRAL), index=frame.index)


def level_columns(level: str, configuration: Config) -> list:
    """The cell columns of each level of segmentation."""
    who = [c for c in configuration.extra_renovacion if "new" in c.lower()] or list(configuration.extra_renovacion[:1])
    if level == LEVEL_REGION:
        return [configuration.business_mandatory_dims[0]]
    if level == LEVEL_MANDATORY:
        return list(configuration.business_mandatory_dims)
    if level == LEVEL_WHO:
        return list(configuration.business_mandatory_dims) + who
    return list(configuration.business_mandatory_dims) + who + ["_signo"]


def composition_by_levels(frame: pd.DataFrame, period_a, period_b, comparison_label: str, configuration: Config,
                          period_column: str = None) -> tuple:
    """The decomposition at the four levels, for the total and for every region.
    OUTPUT: (rows of efecto_composicion, rows of efecto_composicion_celda)."""
    region_dim = configuration.business_mandatory_dims[0]
    frame = frame.assign(_signo=sign_of_rows(frame, configuration))
    summaries, contributions = [], []
    scopes = [(SCOPE_TOTAL, frame)] + [(str(region), block) for region, block in frame.groupby(region_dim)]
    for scope, block in scopes:
        for level in LEVELS_IN_ORDER:
            if scope != SCOPE_TOTAL and level == LEVEL_REGION:
                continue
            summary, cells = decompose_rate_change(block, level_columns(level, configuration), period_a, period_b, configuration, period_column)
            summaries.append(dict(comparacion=comparison_label, periodo_0=str(period_a), periodo_1=str(period_b), nivel=level, ambito=scope, **summary))
            if len(cells) and scope == SCOPE_TOTAL:
                cells = cells.assign(comparacion=comparison_label, nivel=level, ambito=scope)
                cells["abs_usd"] = cells["composicion_usd"].abs()
                contributions.append(cells.sort_values("abs_usd", ascending=False).head(TOP_CONTRIBUTIONS).drop(columns="abs_usd"))
    return summaries, contributions


def headline_from_levels(rows: list) -> dict:
    """The headline: what the mandatory cell leaves as behaviour and the finer levels
    reveal as composition = the $ a mandatory-cell forecast misses by not knowing the
    customer. Computed for the total scope of one comparison."""
    by_level = {r["nivel"]: r for r in rows if r["ambito"] == SCOPE_TOTAL}
    if LEVEL_MANDATORY not in by_level or LEVEL_WHO_HOW not in by_level:
        return {}
    mandatory, finest = by_level[LEVEL_MANDATORY], by_level[LEVEL_WHO_HOW]
    return dict(comparacion=mandatory["comparacion"],
                composicion_mandatory_usd=mandatory["composicion_usd"], composicion_cliente_usd=finest["composicion_usd"],
                aporta_conocer_cliente_usd=(finest["composicion_usd"] - mandatory["composicion_usd"]) if pd.notna(finest["composicion_usd"]) and pd.notna(mandatory["composicion_usd"]) else np.nan)


def run_composition_effect(frame: pd.DataFrame, comparisons: list, configuration: Config, period_column: str,
                           table_suffix: str = "") -> dict:
    """One moment of the composition effect: `comparisons` is a list of (label, period_a,
    period_b) over `period_column` of `frame`. Persists efecto_composicion[_suffix] and
    efecto_composicion_celda[_suffix]. Returns the tables and the headlines."""
    all_rows, all_cells, headlines = [], [], []
    for label, a, b in comparisons:
        rows, cells = composition_by_levels(frame, a, b, label, configuration, period_column)
        all_rows.extend(rows)
        all_cells.extend(cells)
        headline = headline_from_levels(rows)
        if headline:
            headlines.append(headline)
    effect = pd.DataFrame(all_rows)
    per_cell = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    configuration.write(effect, "efecto_composicion" + table_suffix)
    configuration.write(per_cell, "efecto_composicion_celda" + table_suffix)
    return dict(efecto_composicion=effect, efecto_composicion_celda=per_cell, titulares=pd.DataFrame(headlines))


def print_composition_effect(effect: pd.DataFrame, headlines: pd.DataFrame, configuration: Config, moment: str) -> None:
    total = effect[effect["ambito"] == SCOPE_TOTAL]
    print(f"[composición · {moment}] of what changes the aggregate rate, how much is the portfolio (composition) and how much is people (behaviour)")
    for label, block in total.groupby("comparacion", sort=False):
        line = " · ".join(f"{r['nivel']}: {r['pct_composicion']:.0f}% composición ({r['composicion_pp']:+.1f} pp, ${r['composicion_usd']:,.0f})"
                          for _, r in block.iterrows() if pd.notna(r["pct_composicion"]))
        first = block.iloc[0]
        print(f"   {label}: tasa {100 * first['tasa_0']:.1f}% → {100 * first['tasa_1']:.1f}% ({first['cambio_total_pp']:+.1f} pp) · {line}")
    if len(headlines):
        for _, h in headlines.iterrows():
            print(f"   HEADLINE {h['comparacion']}: knowing WHO the customer is and HOW they are moves ${h['aporta_conocer_cliente_usd']:,.0f} from 'behaviour' to 'composition' "
                  f"— the $ a forecast by mandatory cell would miss just by customer type and state")
    explain(configuration,
            "Composition = the aggregate rate moves because DIFFERENT customers fall due (weights change); behaviour = the SAME customers renew differently (rates change).",
            "The four levels answer 'what does knowing the customer add': what the mandatory cell calls behaviour, the finer cells (new vs recurring, neutral vs signed) reveal as composition.",
            "A forecast by segment (Σ pipeline_segment × rate_segment) absorbs composition automatically: the pipeline of every segment is known in advance.")
