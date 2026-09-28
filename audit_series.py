"""audit_series.py — SFF v3 · the audit of ONE forecast series, end to end.

    from audit_series import audit_series
    audit = audit_series("EU|0|0|0|0|A|tele", configuration)          # reads the sff_* tables
    audit = audit_series("EU|0|0|0|0|A|tele", results=results)        # from run_analysis' dict

Prints the story of the series in the order the pipeline built it — raw history, sign
and support, the ladder rung by rung, the estimate and its two errors, the dynamics of
its estimation id, the technique chosen and its backtest, the hold-out months, the
uplift of its cells, and the forecast month by month with its band — and returns a dict
with every filtered table so the same audit can be done in a notebook or in Power BI.

This is exactly the filter path the BI model follows (see AUDITORIA.md): one series
(`fs_key`) → its estimation id (`estimacion_key`) → its uplift cells (`uplift_cell_key`)
→ its mandatory cell (`celda_key`). If it can be told here, it can be filtered there.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys

import pandas as pd

# The project is a flat folder imported from notebooks and scripts alike: make sure the
# folder of this file is importable BEFORE importing the sibling modules below (that is
# why those imports come after this block, not at the top).
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if PROJECT_FOLDER not in sys.path:
    sys.path.insert(0, PROJECT_FOLDER)

from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
RULE = "─" * 74
# logical table → the column(s) that filter it for a series, in order of precedence
TABLES_BY_SERIES = ["forecast_series_raw_summary", "series_card", "parent_ladder", "support_chain", "decision_support",
                    "fact_fu_gaps", "key_bridge", "forecast_detail", "forecast_bands", "forecast_units_extended"]
TABLES_BY_ESTIMATION = ["pool_reference", "decision_technique", "decision_error_bands", "backtest_holdout",
                        "backtest_predictions"]
TABLES_BY_SERIES_OPTIONAL = ["decision_estacionalidad"]
TABLES_BY_CELL = ["mandatory_only_cost", "mix_shift_decomposition"]
TABLES_BY_UPLIFT_CELL = ["decision_uplift", "uplift_chain"]
MAX_ROWS_PRINTED = 40


def read_table(logical_name: str, configuration: Config = None, results: dict = None) -> pd.DataFrame:
    """One logical table, from the results dict if given, else from the engine."""
    if results is not None:
        located = _locate_in_results(logical_name, results)
        if located is not None:
            return located
    if configuration is None or configuration.engine is None:
        return pd.DataFrame()
    physical = configuration._resolve_physical_table_name(logical_name)
    try:
        return pd.read_sql(f"SELECT * FROM {configuration._qualified_table_name(physical)}", configuration.engine)
    except Exception:
        return pd.DataFrame()


def _locate_in_results(logical_name: str, results: dict):
    """Map a logical table name to where run_analysis keeps it in its dict."""
    mapping = {
        "forecast_series_raw_summary": results.get("series_summary"), "series_card": results.get("series_card"),
        "parent_ladder": results.get("parent_ladder"), "key_bridge": results.get("key_bridge"),
        "decision_support": results.get("decisions", {}).get("decision_support"),
        "pool_reference": results.get("decisions", {}).get("pool_reference"),
        "decision_estacionalidad": results.get("decisions", {}).get("decision_estacionalidad"),
        "decision_technique": results.get("decisions", {}).get("decision_technique"),
        "decision_error_bands": results.get("decisions", {}).get("decision_error_bands"),
        "decision_uplift": results.get("decisions", {}).get("decision_uplift"),
        "backtest_holdout": results.get("backtest", {}).get("backtest_holdout"),
        "backtest_predictions": results.get("backtest", {}).get("backtest_long"),
        "mandatory_only_cost": results.get("composition", {}).get("mandatory_only_cost"),
        "mix_shift_decomposition": results.get("composition", {}).get("mix_shift_decomposition"),
        "forecast_detail": results.get("forecast", {}).get("forecast_detail"),
        "forecast_bands": results.get("forecast", {}).get("forecast_bands"),
        "forecast_units_extended": results.get("forecast", {}).get("forecast_units_extended"),
    }
    return mapping.get(logical_name)


def filter_for_series(series_id: str, configuration: Config = None, results: dict = None) -> dict:
    """Every table of the audit, filtered to the series. Returns dict logical name → frame."""
    card = read_table("series_card", configuration, results)
    card = card[card["fs_id"] == series_id] if len(card) else card
    if card.empty:
        raise KeyError(f"series '{series_id}' is not in series_card: check the id (rate series columns joined by '|')")
    estimation_id, cell_id = card["id_estimacion"].iloc[0], card["celda_id"].iloc[0]
    bridge = read_table("key_bridge", configuration, results)
    uplift_cells = sorted(bridge.loc[bridge["fs_id"] == series_id, "uplift_cell_id"].unique()) if len(bridge) else []
    filtered = {}
    for name in TABLES_BY_SERIES:
        table = read_table(name, configuration, results)
        filtered[name] = table[table["fs_id"] == series_id] if len(table) and "fs_id" in table.columns else table.head(0)
    for name in TABLES_BY_ESTIMATION:
        table = read_table(name, configuration, results)
        filtered[name] = table[table["id_estimacion"] == estimation_id] if len(table) else table
    for name in TABLES_BY_SERIES_OPTIONAL:
        table = read_table(name, configuration, results)
        filtered[name] = table[table["fs_id"] == series_id] if len(table) and "fs_id" in table.columns else table.head(0)
    for name in TABLES_BY_CELL:
        table = read_table(name, configuration, results)
        filtered[name] = table[table["celda_id"] == cell_id] if len(table) and "celda_id" in table.columns else table.head(0)
    for name in TABLES_BY_UPLIFT_CELL:
        table = read_table(name, configuration, results)
        filtered[name] = table[table["uplift_cell_id"].isin(uplift_cells)] if len(table) else table
    filtered["_keys"] = dict(fs_id=series_id, id_estimacion=estimation_id, celda_id=cell_id, uplift_cells=uplift_cells)
    return filtered


def tell(filtered: dict) -> None:
    """Print the story of the series from the filtered tables."""
    keys = filtered["_keys"]
    card = filtered["series_card"].iloc[0]
    print(RULE, f"\nAUDIT · series {keys['fs_id']}\n", RULE)
    print(f"[history]   route {card['ruta']} · universe {card['universo']} · sign {card['signo']} · "
          f"{int(card['meses_historia'])} history months ({int(card['huecos'])} gaps) · n_propio {card['n_propio']:.0f} units/month · "
          f"own rate {card['tasa_propia']:.4f} ± {card['error_binomial_pp']:.1f} pp (binomial, z) · "
          f"projected ${card['usd_proyectado']:,.0f}")
    ladder = filtered["parent_ladder"].sort_values("peldano")
    print("[ladder]    rung · relative · n · rate · chosen")
    for _, rung in ladder.iterrows():
        mark = "<- chosen" if rung["elegido"] == 1 else ""
        print(f"            {int(rung['peldano'])}  {rung['descripcion']:<32} {rung['padre_id']:<30} n={rung['n_padre']:>8.1f}  "
              f"rate={rung['tasa_padre'] if pd.notna(rung['tasa_padre']) else float('nan'):.4f}  {mark}")
    print(f"[estimate]  id_estimacion {card['id_estimacion']} (rung {int(card['peldano'])}, n_efectivo {card['n_efectivo']:.0f}) · "
          f"k={card['k']} → z={card['z']} · tasa_estimada {card['tasa_estimada']:.4f} = z·own + (1−z)·parent "
          f"({card['tasa_propia']:.4f} / {card['tasa_pariente'] if pd.notna(card['tasa_pariente']) else float('nan'):.4f}) · "
          f"se_estimacion {card['se_estimacion_pp']:.1f} pp · se_prediccion {card['se_prediccion_pp']:.1f} pp · level {card['nivel_riesgo']}")
    reference = filtered["pool_reference"]
    if len(reference):
        d = reference.iloc[0]
        print(f"[pool]      {keys['id_estimacion']}: {int(d['meses'])} months · n_pool {d['n_pool']:.0f} · rate {d['tasa_pool']} · "
              f"{'with support' if d['gate'] == 'nivel' else 'under the floor'} · month effects {'yes' if d['estacional'] else 'no'} · trend {int(d['tendencia'])}")
    benchmark = filtered["decision_estacionalidad"]
    if len(benchmark):
        b = benchmark.iloc[0]
        print(f"[benchmark] in the seasonality benchmark: φ={b['phi']} amplitude {b['amplitud_pp']} pp ({b['mes_alto']}/{b['mes_bajo']}) "
              f"consistency {b['consistencia_alto']}/{b['consistencia_bajo']} · shape vs level {b['mejora_h1_pct']:+.0f}% h1 / {b['mejora_h6_pct']:+.0f}% h6 · "
              f"verdict {'SEASONAL' if b['veredicto_estacional'] else b['motivo']} · trend {b['pendiente_pp_anio']} pp/yr")
    technique = filtered["decision_technique"]
    if len(technique):
        for _, t in technique.iterrows():
            band = f" {t['tramo_h']} (h {int(t['h_min'])}-{int(min(t['h_max'], 99))})" if "tramo_h" in technique.columns else ""
            print(f"[technique]{band} {t['tecnica']} ({t['tecnica_origen']}) · mean |error| {t['err_pp_medio']} pp = {t['err_norm_medio']} binomial units "
                  f"over {int(t['n_predicciones'])} predictions · challenger at {t['retador_err_norm']}")
    else:
        print("[technique] none judged (series without support: challenger + binomial band)")
    bands = filtered["decision_error_bands"]
    if len(bands):
        print("[bands]     h · q_low · q_high (× binomial se of the pool month) · origin")
        for _, b in bands.iterrows():
            print(f"            {int(b['h']):>2}  {b['q_low_norm']:+.2f}  {b['q_high_norm']:+.2f}  {b['banda_origen']} ({int(b['n_predicciones'])} predictions)")
    holdout = filtered["backtest_holdout"]
    if len(holdout):
        by_h = holdout.groupby("h").agg(err=("err_pp", lambda e: float(e.abs().mean())), bias=("err_pp", "mean"), inside=("dentro_banda", "mean"), n=("err_pp", "size"))
        print(f"[hold-out]  months ≥ {holdout['mes_objetivo'].min()} predicted with {holdout['tecnica'].iloc[0]}:")
        for h, row in by_h.iterrows():
            print(f"            h={int(h):>2}: |error| {row['err']:.2f} pp · bias {row['bias']:+.2f} pp · {row['inside']:.0%} inside band · {int(row['n'])} months")
    uplift = filtered["decision_uplift"]
    if len(uplift):
        print("[uplift]    cell · renewers · uplift · origin · band")
        for _, u in uplift.iterrows():
            print(f"            {u['uplift_cell_id']:<24} n={u['n_renovadores']:>8.0f}  {u['uplift']:.4f}  {u['uplift_origen']:<7} "
                  f"[{u['banda_low'] if pd.notna(u['banda_low']) else float('nan'):.4f}, {u['banda_high'] if pd.notna(u['banda_high']) else float('nan'):.4f}]"
                  f"{'  (clipped at the cap)' if u['recortado'] else ''}")
    detail, forecast_bands = filtered["forecast_detail"], filtered["forecast_bands"]
    if len(detail):
        joined = detail.merge(forecast_bands[["fu_comb_key", "banda_low_pp", "banda_high_pp", "banda_origen"]], on="fu_comb_key", how="left")
        period = [c for c in ("period", "periodo") if c in joined.columns][0]
        monthly = joined.groupby(period).agg(pipe=("total_tr_usd", "sum") if "total_tr_usd" in joined.columns else ("esperado_usd", "size"),
                                             esperado=("esperado_usd", "sum"), tasa=("tasa", "mean"), uplift=("uplift", "mean"),
                                             h=("h", "max"), low=("banda_low_pp", "mean"), high=("banda_high_pp", "mean"),
                                             sim=("simulada", "max"), tecnica=("tecnica", "first"), origen=("tasa_origen", "first"))
        print("[forecast]  month · h · rate (origin, technique) · uplift · band pp · expected $ · simulated")
        for month, row in monthly.iterrows():
            print(f"            {month}  h={row['h'] if pd.notna(row['h']) else '-':>3}  {row['tasa']:.4f} ({row['origen']}, {row['tecnica']})  ×{row['uplift']:.3f}  "
                  f"[{row['low']:+.1f}, {row['high']:+.1f}] pp  ${row['esperado']:>10,.0f}  {'sim' if row['sim'] else ''}")
        print(f"            total ${monthly['esperado'].sum():,.0f} over {len(monthly)} months")
    print(RULE)


def audit_series(series_id: str, configuration: Config = None, results: dict = None, verbose: bool = True) -> dict:
    """The audit: filter every table to the series and tell its story. Returns the filtered tables."""
    filtered = filter_for_series(series_id, configuration, results)
    if verbose:
        tell(filtered)
    return filtered
