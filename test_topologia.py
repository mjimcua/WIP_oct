"""test_topologia.py — profile → synthetic → framework → profile again: the shape survives.

    python test_topologia.py   →  exit 0 if healthy, 1 with the failures
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# make the flat project folder importable before the sibling imports below
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, PROJECT_FOLDER)

from checks import CheckRecorder
from config import Config
import synthetic_from_topology as sft
from pipeline import run_analysis
from synthetic_v3 import build_raw
from test_fixtures import quiet

RECORDER = CheckRecorder()
TIMEVARYING = {"dormant": "negative", "softcancel": "negative", "no_instalado": "negative", "autorenew": "positive"}


def test_profile_then_regenerate() -> None:
    RECORDER.start_block("T1 · the topology profile and the synthetic built from it")
    folder = tempfile.mkdtemp()

    class Synthetic(Config):
        def read_raw(self):
            return build_raw(7, with_discount_pct=True)
    configuration = Synthetic(sql_engine=None, sql_schema=None, outdir=folder, business_mandatory_dims=["region"], structural_timevarying_dims=TIMEVARYING,
                              extra_renovacion=["product", "channel"], extra_revalorizacion=["discount", "newcust"], extended_horizon_end="2027-12",
                              benchmark_group_dims=["region", "product"], benchmark_min_support=100, sheets_top_series=0, console_explanations=False,
                              discount_value_column="discount_pct")
    with quiet():
        results = run_analysis(configuration)
    topology = results["topologia"]
    RECORDER.check(os.path.exists(os.path.join(folder, "topologia.json")) and topology["series"]["n_series"] > 0, "topologia.json is written with the series block")
    text = open(os.path.join(folder, "topologia.json"), encoding="utf-8").read()
    RECORDER.check(all(word not in text for word in ("EU", "NA", "web", "tele", "kiosk", "region", "product", "channel", "dormant", "softcancel")),
                   "no real value or column name travels in the topology (roles and ranked codes only)")
    raw = sft.build_raw_from_topology(topology, seed=5, scale=1.0)
    RECORDER.check(len(raw) > 100 and set(raw.columns) >= {"mandatory_1", "tv_neg_1", "extra_ren_1", "discount_pct", "period", "dataset_role", "is_current_month"},
                   f"a synthetic raw with generic column names is built ({len(raw)} rows)")
    generated_folder = tempfile.mkdtemp()

    class FromTopology(Config):
        def read_raw(self):
            return raw
    generated = FromTopology(**{k: v for k, v in vars(sft.topology_config(topology, generated_folder, extended_horizon_end="2027-12", benchmark_min_support=30)).items() if k in Config.__dataclass_fields__})
    with quiet():
        results_2 = run_analysis(generated)
    RECORDER.check(results_2["forecast"]["forecast_detail"]["esperado_usd"].notna().all() and bool(results_2["nucleo"]["reconciliacion"]["cuadra"].all()),
                   "the framework runs end to end on the generated raw and its core reconciles")
    profile_2 = results_2["topologia"]
    support_1, support_2 = topology["series"]["soporte_cuantiles"]["0.5"], profile_2["series"]["soporte_cuantiles"]["0.5"]
    rate_1, rate_2 = topology["tasas"]["tasa_neutra_cuantiles"]["0.5"], profile_2["tasas"]["tasa_neutra_cuantiles"]["0.5"]
    RECORDER.check(support_2 / support_1 < 4 and support_1 / support_2 < 4, f"the median support keeps its order of magnitude ({support_1} → {support_2})")
    RECORDER.check(abs(rate_1 - rate_2) < 0.15, f"the median neutral rate stays close ({rate_1:.3f} → {rate_2:.3f})")
    flags_1 = {k: v["cuota_unidades"] for k, v in topology["dimensiones"].items() if k.startswith("tv_")}
    flags_2 = {k: v["cuota_unidades"] for k, v in profile_2["dimensiones"].items() if k.startswith("tv_")}
    RECORDER.check(all(abs(flags_1[k] - flags_2.get(k, 0)) < 0.15 for k in flags_1), f"the share of units with each flag stays close ({flags_1} → {flags_2})")
    d1, d2 = topology["precio"]["descuento"], profile_2["precio"]["descuento"]
    RECORDER.check(abs(d1["cuota_desconocido"] - d2["cuota_desconocido"]) < 0.2 and abs(d1["cuota_cero"] - d2["cuota_cero"]) < 0.25,
                   f"the discount structure (unknown / zero shares) stays close")


def main() -> int:
    print("═" * 74 + "\nTEST topología · the shape of the portfolio, without the data\n" + "═" * 74)
    test_profile_then_regenerate()
    return RECORDER.print_panel("TOPOLOGÍA TEST")


if __name__ == "__main__":
    sys.exit(main())
