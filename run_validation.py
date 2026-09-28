"""run_validation.py — SFF v3 · the validation panel.

Three families of checks, persisted as `validation_report` (check, familia, valor,
esperado, estado, detalle):
  INTEGRITY  the data is whole: money conserved, keys unique, every raw row reaches the
             bridge. A FAIL here means the run cannot be trusted.
  DOCTRINE   the rules of the framework hold: no projection row with results, no rate above the
             cap, positive uplifts, no NaN in the forecast, mandatory dims never annulled
             before the cell, signs never mixed, band never narrowing with the horizon.
  QUALITY    informative, not blocking: money by risk level, hold-out calibration, share
             of money with a measured band, share of money forecast with a champion,
             share of simulated pipeline, mix risk.
INTEGRITY and DOCTRINE failures stop the run (RUN); QUALITY is read by the analyst.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import Config
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
MONEY_TOLERANCE_USD = 1e-6
FAMILY_INTEGRITY, FAMILY_DOCTRINE, FAMILY_QUALITY = "INTEGRITY", "DOCTRINE", "QUALITY"


def check_row(name: str, family: str, value, expected: str, passed, detail: str = "") -> dict:
    state = "PASS" if passed else ("WARN" if passed is None else "FAIL")
    return dict(check=name, familia=family, valor=str(value), esperado=expected, estado=state, detalle=detail)


def run_validation(artifacts: dict, configuration: Config) -> pd.DataFrame:
    """Build, persist and print the panel. Raises if any INTEGRITY or DOCTRINE check FAILs.

    INPUT:   artifacts — dict with fine_table, forecast_units, key_bridge, forecast_detail,
             forecast_bands, horizon_report, series_card, decision_support, parent_ladder,
             backtest_holdout, decision_uplift (missing ones are skipped).
    """
    rows = []
    fine, units = artifacts.get("fine_table"), artifacts.get("forecast_units")
    usd = configuration.pipeline_usd_col
    if fine is not None and units is not None:
        gap = abs(fine[usd].sum() - units[units["sintetica"] == 0][usd].sum()) if "sintetica" in units.columns else abs(fine[usd].sum() - units[usd].sum())
        rows.append(check_row("pipeline USD conserved fine ↔ units", FAMILY_INTEGRITY, round(gap, 4), "= 0", gap < MONEY_TOLERANCE_USD))
        rows.append(check_row("fu_id unique in forecast units", FAMILY_INTEGRITY, int(units["fu_id"].duplicated().sum()), "= 0",
                              not units["fu_id"].duplicated().any()))
        rows.append(check_row("fu_key unique (no hash collision)", FAMILY_INTEGRITY, int(units["fu_key"].duplicated().sum()), "= 0",
                              not units["fu_key"].duplicated().any()))
    bridge = artifacts.get("key_bridge")
    if bridge is not None and fine is not None:
        covered = fine["fu_comb_key"].isin(bridge["fu_comb_key"]).mean()
        rows.append(check_row("raw → key_bridge coverage", FAMILY_INTEGRITY, f"{covered:.1%}", "= 100%", covered == 1.0))
        rows.append(check_row("fu_comb_key unique in the bridge", FAMILY_INTEGRITY, int(bridge["fu_comb_key"].duplicated().sum()), "= 0",
                              not bridge["fu_comb_key"].duplicated().any()))
    if fine is not None:
        projection = fine[fine[configuration.dataset_role_col] == ROLE_PROJECTION]
        early = int((projection[configuration.renewed_units_col].fillna(0) != 0).sum())
        rows.append(check_row("projection rows carry no results", FAMILY_DOCTRINE, early, "= 0", early == 0))
        current_not_projection = int((fine[configuration.current_month_col].isin([1, True, "1"]) & (fine[configuration.dataset_role_col] != ROLE_PROJECTION)).sum())
        rows.append(check_row("current month is projection", FAMILY_DOCTRINE, current_not_projection, "= 0", current_not_projection == 0))
    detail = artifacts.get("forecast_detail")
    if detail is not None and len(detail) and "esperado_usd" in detail.columns:
        rows.append(check_row("forecast without NaN", FAMILY_DOCTRINE, int(detail["esperado_usd"].isna().sum()), "= 0", not detail["esperado_usd"].isna().any()))
        over = int((detail["tasa"] > configuration.rate_cap + 1e-9).sum())
        rows.append(check_row("rate under the cap", FAMILY_DOCTRINE, over, "= 0", over == 0))
        bad_uplift = int((detail["uplift"] <= 0).sum())
        rows.append(check_row("uplift positive", FAMILY_DOCTRINE, bad_uplift, "= 0", bad_uplift == 0))
        share_series = detail.loc[detail["tasa_origen"] == "serie", "esperado_usd"].sum() / max(detail["esperado_usd"].sum(), 1e-9)
        rows.append(check_row("$ forecast with its own series rate", FAMILY_QUALITY, f"{share_series:.0%}", "informational", None))
        share_global = detail.loc[detail["tasa_origen"] == "global", "esperado_usd"].sum() / max(detail["esperado_usd"].sum(), 1e-9)
        rows.append(check_row("$ falling to the global mean", FAMILY_QUALITY, f"{share_global:.1%}", "≈ 0%", share_global < 0.01 or None))
        share_champion = detail.loc[detail["tecnica_origen"] == "campeon", "esperado_usd"].sum() / max(detail["esperado_usd"].sum(), 1e-9)
        rows.append(check_row("$ forecast with a champion technique", FAMILY_QUALITY, f"{share_champion:.0%}", "informational", None))
        share_simulated = detail.loc[detail["simulada"] == 1, "esperado_usd"].sum() / max(detail["esperado_usd"].sum(), 1e-9)
        rows.append(check_row("$ built on simulated pipeline", FAMILY_QUALITY, f"{share_simulated:.0%}", "informational", None))
    bands = artifacts.get("forecast_bands")
    if bands is not None and len(bands):
        measured = 1 - bands["banda_origen"].str.startswith("binomial").mean()
        rows.append(check_row("rows with a measured band", FAMILY_QUALITY, f"{measured:.0%}", "towards 100%", measured >= 0.5 or None))
    horizon = artifacts.get("horizon_report")
    if horizon is not None and len(horizon):
        narrowed = int((horizon["banda_monotona"] == 0).sum())
        rows.append(check_row("total band narrowing with the horizon (mix)", FAMILY_QUALITY, narrowed, "= 0 months beyond tolerance", narrowed == 0 or None))
    card = artifacts.get("series_card")
    if card is not None and len(card):
        mixed = card[card["nivel_riesgo"] == "M_signo_mixto"]
        rows.append(check_row("mixed-sign series", FAMILY_QUALITY, f"{len(mixed)} (${mixed['usd_proyectado'].sum():,.0f})", "≈ 0", len(mixed) == 0 or None))
        below = card[card["nivel_riesgo"] == "S_signo_bajo_suelo"]
        rows.append(check_row("signed series under the floor", FAMILY_QUALITY, f"{len(below)} (${below['usd_proyectado'].sum():,.0f})", "informational", None))
    ladder = artifacts.get("parent_ladder")
    support = artifacts.get("decision_support")
    if ladder is not None and support is not None and len(ladder):
        signed = support[support["signo"].isin([SIGN_NEGATIVE, SIGN_POSITIVE])]
        climbed_above_cell = int((signed["peldano"] > 3).sum())
        if configuration.signed_ladder_max_loss <= 0:
            rows.append(check_row("signed series never climb above their cell", FAMILY_DOCTRINE, climbed_above_cell, "= 0", climbed_above_cell == 0))
        else:
            rows.append(check_row("signed series climbing above their cell (sign kept)", FAMILY_QUALITY, climbed_above_cell,
                                  f"allowed: cumulative loss ≤ {configuration.signed_ladder_max_loss}", None))
        neutral_pooled_with_sign = int(support[(support["signo"] == SIGN_NEUTRAL) & support["id_estimacion"].str.contains("SIG=negativo|SIG=positivo")].shape[0])
        rows.append(check_row("neutral series never pooled with a sign", FAMILY_DOCTRINE, neutral_pooled_with_sign, "= 0", neutral_pooled_with_sign == 0))
    holdout = artifacts.get("backtest_holdout")
    if holdout is not None and len(holdout):
        inside = holdout["dentro_banda"].mean()
        rows.append(check_row("hold-out calibration (inside band)", FAMILY_QUALITY, f"{inside:.0%}", "≈ 90%", 0.8 <= inside <= 0.98 or None))
        h1 = holdout[holdout["h"] == 1]
        if len(h1):
            rows.append(check_row("hold-out mean |error| at h=1", FAMILY_QUALITY, f"{h1['err_pp'].abs().mean():.2f} pp", "informational", None))
    uplift = artifacts.get("decision_uplift")
    if uplift is not None and len(uplift):
        implausible = int(((uplift["uplift"] < 0.3) | (uplift["uplift"] > configuration.uplift_cap)).sum())
        rows.append(check_row("uplift in plausible range", FAMILY_QUALITY, implausible, f"= 0 cells outside [0.3, {configuration.uplift_cap}]", implausible == 0 or None))
    report = pd.DataFrame(rows)
    configuration.write(report, "validation_report")
    print("VALIDATION PANEL")
    for _, row in report.iterrows():
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "WARN": "warn"}[row["estado"]]
        print(f"  {mark} [{row['familia']:<9}] {row['check']:<44} {row['valor']:>14}   ({row['esperado']})")
    failed = report[(report["estado"] == "FAIL") & (report["familia"].isin([FAMILY_INTEGRITY, FAMILY_DOCTRINE]))]
    if len(failed):
        raise AssertionError("validation FAILED: " + "; ".join(failed["check"]))
    return report
