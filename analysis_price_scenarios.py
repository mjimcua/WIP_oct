"""analysis_price_scenarios.py — SFF v3 · the price scenario layer.

PRINCIPLE. The library computes the heavy part once (the renewal rate and the assembled
forecast). Price and scenarios are a FINAL LAYER applied on the finished forecast rows,
without recomputing the engine, so two or three scenarios can be tried by changing a
table.

WHY. Because the renewal tactic removes the discount, part of the future portfolio has
no growth margin left from discount removal. We want to see how much of the portfolio
is in that situation and how much a small increase per SKU moves the number, offset by
a drop of the rate so that it is not free money upwards.

INPUT. `Config.read_price_table()`: escenario, sku, precio_lista (informative),
fecha_efectiva, subida_pct, subida_importe, delta_tasa_pp. Several scenarios in one
table through `escenario`.

ROWS AFFECTED. Fine rows of the future renewal pipeline (real and projected re-entries),
each with the rate of its forecast series. The simulated acquisition is left out.

MARGIN CLASS per row: con_margen (discount > 0), sin_margen (discount = 0, including the
projected re-entries, which already had it removed), desconocido (discount unknown).

APPLICATION, when the expiry month ≥ fecha_efectiva of the row's SKU:
  price: 100 % of list + increase (pct and/or importe); rows with an unknown discount get
         the increase on their estimated price when scenario_increase_unknown_discount;
  rate:  + delta_tasa_pp in absolute points (60 % → 59 %), bounded to [0, 1], only on
         NEUTRAL rows when scenario_delta_neutral_only (elsewhere the signal already
         carries the rejection).
The band of the rate is not widened by the assumption: it is reported shifted.

OUTPUT. escenarios_precio per escenario × SKU × region × month × margin class: $ base,
$ scenario, $ by price, $ by retention, net, and the break-even rate (the rate that can be
lost before losing money: r_equilibrio = r × precio_base / precio_escenario). Summary: %
of the portfolio sin_margen per year.

EXAMPLE. 10,000 units sin_margen, list $50, rate 60 % → $300,000. With +5 % and −1 pp:
10,000 × 0.59 × 52.5 = $309,750 (+$9,750). Break-even: 57.1 % (2.9 pp can be lost).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config, explain
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
MARGIN_WITH, MARGIN_WITHOUT, MARGIN_UNKNOWN = "con_margen", "sin_margen", "desconocido"


def margin_class(discount: pd.Series, configuration: Config) -> pd.Series:
    """con_margen (d > 0), sin_margen (d = 0), desconocido (null or above the cap)."""
    exact = pd.to_numeric(discount, errors="coerce")
    known = exact.notna() & (exact >= 0) & (exact <= configuration.discount_cap)
    return pd.Series(np.where(~known, MARGIN_UNKNOWN, np.where(exact > 0, MARGIN_WITH, MARGIN_WITHOUT)), index=discount.index)


def scenario_rows(detail: pd.DataFrame, fine_future: pd.DataFrame, units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The forecast rows the scenarios act on: real and projected (not simulated), with
    their SKU, discount, margin class, neutral flag, list price per unit and base $."""
    period = configuration.period_col
    columns = ["fu_comb_key", configuration.sku_column] + ([configuration.discount_value_column] if configuration.discount_value_column else [])
    columns += [c for c in configuration.structural_timevarying_dims]
    carried = fine_future[[c for c in columns if c in fine_future.columns]].drop_duplicates("fu_comb_key")
    rows = detail[detail["origen_pipeline"].isin((PIPELINE_REAL, PIPELINE_PROJECTED))].merge(carried, on="fu_comb_key", how="left")
    discount = rows[configuration.discount_value_column] if configuration.discount_value_column in rows.columns else pd.Series(np.nan, index=rows.index)
    rows["margen"] = margin_class(discount, configuration)
    rows.loc[rows["origen_pipeline"] == PIPELINE_PROJECTED, "margen"] = MARGIN_WITHOUT       # already at list after renewing
    negatives = [f for f, p in configuration.structural_timevarying_dims.items() if p == "negative" and f in rows.columns]
    rows["neutra"] = ~pd.concat([rows[f].isin(configuration.timevarying_positive_values) for f in negatives], axis=1).any(axis=1) if negatives else True
    # base: the renewed price per unit = pipeline AUV × uplift (already 100 % of list on the contract path)
    rows["auv_base"] = rows[configuration.pipeline_usd_col] / rows[configuration.pipeline_units_col].clip(lower=1e-9) * rows["uplift"]
    rows["usd_base"] = rows["esperado_usd"]
    return rows


def apply_scenarios(rows: pd.DataFrame, price_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One output row per scenario × SKU × region × month × margin class."""
    period = configuration.period_col
    region_dim = configuration.business_mandatory_dims[0]
    rows = rows.assign(_region=rows["fs_id"].str.split("|").str[0], _month=pd.PeriodIndex(rows[period].astype(str), freq="M"))
    out = []
    for scenario, table in price_table.groupby("escenario"):
        table = table.copy()
        table["fecha_efectiva"] = pd.PeriodIndex(pd.to_datetime(table["fecha_efectiva"]).dt.to_period("M"), freq="M")
        affected = rows.merge(table[["sku", "fecha_efectiva", "subida_pct", "subida_importe", "delta_tasa_pp"]], left_on=configuration.sku_column, right_on="sku", how="inner")
        if affected.empty:
            continue
        applies = affected["_month"] >= affected["fecha_efectiva"]
        pct = affected["subida_pct"].fillna(0.0) if configuration.scenario_use_pct_and_importe else affected["subida_pct"].fillna(0.0)
        importe = affected["subida_importe"].fillna(0.0) if configuration.scenario_use_pct_and_importe else 0.0
        increase_allowed = applies & ((affected["margen"] != MARGIN_UNKNOWN) | configuration.scenario_increase_unknown_discount)
        auv_scenario = np.where(increase_allowed, affected["auv_base"] * (1 + pct) + importe, affected["auv_base"])
        delta_allowed = applies & ((affected["neutra"]) | (not configuration.scenario_delta_neutral_only))
        rate_scenario = np.where(delta_allowed, (affected["tasa"] + affected["delta_tasa_pp"].fillna(0.0) / 100).clip(0, 1), affected["tasa"])
        units = affected[configuration.pipeline_units_col]
        usd_price_only = units * affected["tasa"] * auv_scenario                 # price moved, rate not
        usd_scenario = units * rate_scenario * auv_scenario
        affected["usd_escenario"] = usd_scenario
        affected["usd_por_precio"] = usd_price_only - affected["usd_base"]
        affected["usd_por_retencion"] = usd_scenario - usd_price_only
        affected["auv_escenario"] = auv_scenario
        affected["tasa_escenario"] = rate_scenario
        grouped = affected.groupby(["sku", "_region", "_month", "margen"]).agg(
            unidades=(configuration.pipeline_units_col, "sum"), usd_base=("usd_base", "sum"), usd_escenario=("usd_escenario", "sum"),
            usd_por_precio=("usd_por_precio", "sum"), usd_por_retencion=("usd_por_retencion", "sum"),
            tasa_base=("tasa", "mean"), tasa_escenario=("tasa_escenario", "mean"),
            auv_base=("auv_base", "mean"), auv_escenario=("auv_escenario", "mean")).reset_index()
        grouped["neto_usd"] = grouped["usd_escenario"] - grouped["usd_base"]
        grouped["tasa_equilibrio"] = grouped["tasa_base"] * grouped["auv_base"] / grouped["auv_escenario"].replace(0, np.nan)
        grouped["margen_pp_hasta_equilibrio"] = 100 * (grouped["tasa_escenario"] - grouped["tasa_equilibrio"])
        grouped.insert(0, "escenario", scenario)
        grouped = grouped.rename(columns={"_region": region_dim, "_month": period})
        grouped[period] = grouped[period].astype(str)
        out.append(grouped)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_price_scenarios(detail: pd.DataFrame, fine_table: pd.DataFrame, units_extended: pd.DataFrame, units: pd.DataFrame,
                        configuration: Config) -> dict:
    """The layer end to end: margin classification of the future portfolio (always), the
    scenarios (when the price table has rows). Persists escenarios_precio and
    escenarios_margen."""
    period = configuration.period_col
    if not configuration.sku_column or configuration.sku_column not in fine_table.columns:
        print("[escenarios] no sku_column: the margin classification and the price scenarios are skipped")
        empty_margin = pd.DataFrame(columns=["anio", "margen", "unidades", "usd", "usd_esperado", "pct_usd_anio"])
        empty_scenarios = pd.DataFrame(columns=["escenario", "sku", configuration.business_mandatory_dims[0], period, "margen", "unidades", "usd_base", "usd_escenario",
                                                "usd_por_precio", "usd_por_retencion", "tasa_base", "tasa_escenario", "auv_base", "auv_escenario", "neto_usd", "tasa_equilibrio", "margen_pp_hasta_equilibrio"])
        configuration.write(empty_margin, "escenarios_margen")
        configuration.write(empty_scenarios, "escenarios_precio")
        return dict(escenarios_precio=empty_scenarios, escenarios_margen=empty_margin)
    fine_future = pd.concat([fine_table[fine_table[configuration.dataset_role_col].isin(FUTURE_ROLES)], units_extended], ignore_index=True) if units_extended is not None and len(units_extended) else fine_table[fine_table[configuration.dataset_role_col].isin(FUTURE_ROLES)]
    rows = scenario_rows(detail, fine_future, units, configuration)
    rows["anio"] = pd.PeriodIndex(rows[period].astype(str), freq="M").year
    margin = rows.groupby(["anio", "margen"]).agg(unidades=(configuration.pipeline_units_col, "sum"), usd=(configuration.pipeline_usd_col, "sum"), usd_esperado=("esperado_usd", "sum")).reset_index()
    margin["pct_usd_anio"] = 100 * margin["usd"] / margin.groupby("anio")["usd"].transform("sum")
    configuration.write(margin, "escenarios_margen")
    price_table = configuration.read_price_table()
    scenarios = apply_scenarios(rows, price_table, configuration) if len(price_table) else pd.DataFrame()
    configuration.write(scenarios, "escenarios_precio")
    without = margin[margin["margen"] == MARGIN_WITHOUT]
    print("[escenarios] margin of the future renewal portfolio (real + projected; simulated acquisition left out): "
          + " · ".join(f"{int(y)}: {p:.0f}% sin margen" for y, p in zip(without["anio"], without["pct_usd_anio"])))
    if len(scenarios):
        for scenario, block in scenarios.groupby("escenario"):
            print(f"[escenarios] {scenario}: base ${block['usd_base'].sum():,.0f} → ${block['usd_escenario'].sum():,.0f} "
                  f"(price {block['usd_por_precio'].sum():+,.0f}, retention {block['usd_por_retencion'].sum():+,.0f}, net {block['neto_usd'].sum():+,.0f})")
    explain(configuration,
            "sin_margen = the discount is already 0 (or will be, after the projected renewal): no growth left from removing discounts; only a list increase moves it.",
            "A scenario applies, from its effective date, an increase per SKU and a drop of the rate (only on neutral rows: signed rows already carry the rejection).",
            "$ por precio = the increase with the rate unchanged; $ por retención = what the rate drop costs; neto = both. tasa_equilibrio = the rate below which the increase loses money.")
    return dict(escenarios_precio=scenarios, escenarios_margen=margin)
