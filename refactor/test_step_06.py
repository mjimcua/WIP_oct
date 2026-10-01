"""
test_step_06.py — Step 06 on the synthetic raw: every series gets its coverage, route and
universe; nothing is dropped; the series table is written.

    python test_step_06.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import pandas as pd

from step_06_series_routes import build_series_routes, route_from_coverage
from test_helpers import check, console_of, count_status, finish
from test_step_05 import units_synthetic


def run_step(units, configuration):
    frames = {}
    console = console_of(lambda: frames.setdefault("series", build_series_routes(units, configuration)))
    return frames["series"], console


# ═══════════════════════════════════════════════════════════════════════════════════
def test_the_routes() -> None:
    print("A · coverage and route of every series")
    check(route_from_coverage({"entrenamiento", "examen", "proyeccion"}) == "predecible"
          and route_from_coverage({"entrenamiento"}) == "solo_historia"
          and route_from_coverage({"proyeccion"}) == "solo_futuro"
          and route_from_coverage({"examen", "proyeccion"}) == "predecible",
          "the route rule: history + future → predecible · only history → solo_historia · only future → solo_futuro")
    _, units, configuration = units_synthetic()
    series, console = run_step(units, configuration)
    check(len(series) == 17 and series["fs_id"].is_unique, "one row per series, none dropped")
    check(series["ruta"].value_counts().to_dict() == {"predecible": 15, "solo_futuro": 1, "solo_historia": 1},
          "15 predecible, 1 solo_futuro, 1 solo_historia")
    only_history = series[series["ruta"] == "solo_historia"].iloc[0]
    check(only_history["cobertura"] == "entrenamiento" and only_history["usd_por_predecir"] == 0,
          "the solo_historia series covers only training and has nothing to predict")
    check((series["meses_entrenamiento"] + series["meses_examen"] + series["meses_proyeccion"] == series["meses"]).all(), "the months per role add up to the months of the series")
    check(abs(series["usd_por_predecir"].sum() - units.loc[units["rol"] == "proyeccion", "total_tr_usd"].sum()) < 0.01,
          "the USD to predict is what falls due in the projection")
    check(count_status(console, "ok") == 4 and "4 checks: 4 ok" in console, "the 4 checks pass and are logged")
    written = pd.read_sql("SELECT COUNT(*) AS n FROM sff_series", configuration.sql_engine)["n"].item()
    check(written == 17, "sff_series written")


def test_a_mixed_universe_warns() -> None:
    print("B · a series in two universes is a warning, labelled mixto")
    _, units, configuration = units_synthetic()
    mixed = units.copy()
    first_series = mixed["fs_id"].iloc[0]
    mixed.loc[mixed.index[mixed["fs_id"] == first_series][0], "flag_time_series"] = 1
    series, console = run_step(mixed, configuration)
    check(series.loc[series["fs_id"] == first_series, "universo"].item() == "mixto"
          and "WARN  flag_time_series is the same in every unit of a series" in console,
          "the series is labelled mixto and the check warns")


if __name__ == "__main__":
    test_the_routes()
    test_a_mixed_universe_warns()
    finish()
