"""
test_step_22.py — The adaptation to Power BI: the table of pipeline sources covers every source of the
core, once, and the core summed through it gives the same totals as summed alone.

    python test_step_22.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from main import run
from test_helpers import check, console_of, finish, synthetic_with


def test_the_power_bi_tables() -> None:
    print("A · the Power BI tables on the synthetic")
    results = {}
    console = console_of(lambda: results.update(run(synthetic_with())))
    step_console = console.split("STEP 22")[1]
    check("3 checks: 3 ok" in step_console, "the three checks of step 22 pass")

    pipeline_source_table = results["power_bi"]["pipeline_source"]
    core_table = results["core"]
    check(list(pipeline_source_table.columns) == ["forecast_pipeline_source", "source_label", "source_block", "source_order"],
          "the table has code, label, block and order")
    check(set(core_table["forecast_pipeline_source"]) <= set(pipeline_source_table["forecast_pipeline_source"]),
          "every source of the core has its row")
    check(pipeline_source_table["forecast_pipeline_source"].is_unique and pipeline_source_table["source_order"].is_unique,
          "one row per source, each with its own order")

    # the core summed through the table = the core summed alone (the relation adds nothing and loses nothing)
    joined_table = core_table.merge(pipeline_source_table, on="forecast_pipeline_source", how="left")
    check(abs(joined_table["forecast_renewed_USD"].sum() - core_table["forecast_renewed_USD"].sum()) < 0.01
          and len(joined_table) == len(core_table),
          "the core joined to the table keeps its rows and its money")


if __name__ == "__main__":
    test_the_power_bi_tables()
    finish()
