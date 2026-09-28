"""sheet.py — SFF v3 · the sheet of anything you can point at with a key.

    from sheet import sheet
    s = sheet("EU|0|0|0|0|A|tele", configuration)          # a series by id
    s = sheet(233977162747724, configuration)              # ANY key: fs_key, estimacion_key, fu_key,
                                                           # fu_comb_key, uplift_cell_key, celda_key
    s["summary"]        the diagnostics in words
    s["tables"]         every table filtered to the series (audit_series)
    s["figure"]         the PNG path of the sheet (diagnostics_plots)
    s["keys"]           what the key resolved to (kind, fs_id, id_estimacion, celda_id, uplift cells)

The keys are the four the BI joins on (see AUDITORIA.md): a SERIES (`fs_key`), its
ESTIMATION ID (`estimacion_key`, the pool it takes the rate from), its UPLIFT CELLS
(`uplift_cell_key`) and its MANDATORY CELL (`celda_key`); plus the two row-level keys
(`fu_key` = a unit = series × month, `fu_comb_key` = a raw row). Give any of them:
  · a series key → the sheet of that series;
  · an estimation key / cell key / uplift cell key → the sheet of the series with the
    most money inside it, and the list of every member series in `keys["members"]`;
  · a unit or row key → the sheet of the series it belongs to (month noted in `keys`).
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

from audit_series import filter_for_series, read_table, tell
from config import Config, hash_key
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)
from diagnostics_plots import series_sheet, sheet_summary

KEY_KINDS = ("fs_key", "estimacion_key", "celda_key", "uplift_cell_key", "fu_key", "fu_comb_key")


def resolve_key(key, configuration: Config = None, results: dict = None) -> dict:
    """What a key points at. Returns dict(kind, fs_id, id_estimacion, celda_id, members, period).

    RULES:   an id with '|' is hashed first (key = hash_key(id), a bigint); an integer is
             the key itself. It is looked up, in order, as fs_key, estimacion_key,
             celda_key (in series_card), then uplift_cell_key, fu_key, fu_comb_key (in
             key_bridge). A key that matches nothing raises KeyError naming the six kinds.
    """
    card = read_table("series_card", configuration, results)
    if card.empty:
        raise RuntimeError("series_card is not available: run the analysis first")
    for column in ("fs_key", "estimacion_key", "celda_key"):
        if column not in card.columns:
            source = {"fs_key": "fs_id", "estimacion_key": "id_estimacion", "celda_key": "celda_id"}[column]
            card[column] = card[source].astype(str).map(hash_key)
    # a key is an int (bigint: the first 12 hex digits of MD5(id)); an id contains '|'
    if isinstance(key, str) and "|" in key:
        probe = hash_key(key)
    else:
        try:
            probe = int(key)
        except (TypeError, ValueError):
            raise KeyError(f"'{key}' is neither an id (with '|') nor an integer key") from None
    kind, members, period = None, None, None
    if (card["fs_key"] == probe).any():
        kind, members = "fs_key", card[card["fs_key"] == probe]
    elif (card["estimacion_key"] == probe).any():
        kind, members = "estimacion_key", card[card["estimacion_key"] == probe]
    elif (card["celda_key"] == probe).any():
        kind, members = "celda_key", card[card["celda_key"] == probe]
    else:
        bridge = read_table("key_bridge", configuration, results)
        if len(bridge):
            for column in ("uplift_cell_key", "fu_key", "fu_comb_key"):
                if column in bridge.columns and (bridge[column] == probe).any():
                    rows = bridge[bridge[column] == probe]
                    kind, members = column, card[card["fs_id"].isin(rows["fs_id"])]
                    if column in ("fu_key", "fu_comb_key"):
                        period = rows["fu_id"].iloc[0].rsplit("|", 1)[-1]
                    break
    if kind is None or members is None or members.empty:
        raise KeyError(f"'{key}' matches none of {KEY_KINDS} (nor an fs_id)")
    chosen = members.sort_values("usd_proyectado", ascending=False).iloc[0]
    return dict(kind=kind, key=probe, fs_id=chosen["fs_id"], id_estimacion=chosen["id_estimacion"], celda_id=chosen["celda_id"],
                members=list(members["fs_id"]), period=period)


def sheet(key, configuration: Config = None, results: dict = None, figure: bool = True, verbose: bool = True,
          output_folder: str = None) -> dict:
    """The sheet: resolve the key, filter every table, write the summary, draw the figure.

    OUTPUT:  dict(keys, summary, tables, figure). Prints the audit story and the summary
             when verbose. `figure=False` skips the PNG (fast, for loops).
    """
    keys = resolve_key(key, configuration, results)
    tables = filter_for_series(keys["fs_id"], configuration, results)
    card = tables["series_card"].iloc[0]
    dynamics = tables["pool_reference"]
    technique = tables["decision_technique"]
    holdout = tables["backtest_holdout"]
    holdout_h1 = holdout[holdout["h"] == 1].sort_values("mes_objetivo") if len(holdout) else holdout
    summary = sheet_summary(card, dynamics.iloc[0] if len(dynamics) else pd.Series(dtype=object),
                            technique.iloc[0] if len(technique) else pd.Series(dtype=object), holdout_h1)
    if keys["kind"] != "fs_key":
        summary = (f"KEY    {keys['key']} is a {keys['kind']} with {len(keys['members'])} member series; showing the one with most money"
                   + (f" · unit of {keys['period']}" if keys["period"] else "") + "\n" + summary)
    if verbose:
        tell(tables)
        print(summary)
    path = series_sheet(keys["fs_id"], configuration, results, output_folder, verbose=verbose) if figure else None
    return dict(keys=keys, summary=summary, tables=tables, figure=path)
