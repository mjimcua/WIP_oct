"""
test_term_column.py — The term of a licence when it is a leveled dimension: the column keeps its name and
its raw value in every step (step 01 only adds term_level_1), so step 02 wipes ONLY the pipeline of the
1-year licences that is not known yet and step 17 projects only them; a term_column that is not a
mandatory dim stops when the Config is built.

    python test_term_column.py
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import tempfile

import pandas as pd

from main import SyntheticConfig
from pipeline import Orchestrator
from test_helpers import check, check_stops, console_of, finish, synthetic_with


class TermSyntheticConfig(SyntheticConfig):
    """The synthetic extract with a term column ("1 year" for product A, "3 year" for B) and its pipeline
    of 2027 (the months 2026-09..2026-12 again, 12 months later, nothing renewed yet)."""
    def read_raw(self) -> pd.DataFrame:
        raw = super().read_raw()
        months = pd.PeriodIndex(pd.to_datetime(raw["period"]), freq="M")
        future = raw[months >= pd.Period("2026-09", "M")].copy()
        future["period"] = (pd.PeriodIndex(pd.to_datetime(future["period"]), freq="M") + 12).to_timestamp()
        for column in ("total_renewed_units", "total_renewed_usd"):
            future[column] = float("nan")
        raw = pd.concat([raw, future], ignore_index=True)
        raw["term"] = raw["product"].map({"A": "1 year", "B": "3 year"})
        return raw


def term_configuration(term_column: str = "term"):
    base = synthetic_with()
    arguments = {name: getattr(base, name) for name in base.__dataclass_fields__}
    arguments.update(business_mandatory_dims=list(base.business_mandatory_dims) + ["term"], term_column=term_column,
                     leveled_dims={"term": "ordinal"}, levels_path=os.path.join(tempfile.mkdtemp(), "levels.json"),
                     checkpoint_folder=tempfile.mkdtemp())
    return TermSyntheticConfig(**arguments)


def wiped_by_term(orchestrator) -> dict:
    """Units due in the extract and after the calendar, from 2027-09 on, per term."""
    calendared = orchestrator.context["calendared_raw"]
    late = calendared[calendared["period"] >= pd.Period("2027-09", "M")]
    return late.groupby("term").agg(extract=("s0_vencen_unidades", "sum"), after=("total_tr_units", "sum")).to_dict("index")


def test_the_term_column() -> None:
    print("A · term in one column, leveled: only the 1-year pipeline not known yet is wiped")
    orchestrator = Orchestrator(term_configuration())
    console = console_of(lambda: [orchestrator.run_step(name) for name in orchestrator.order[:orchestrator.order.index("fine_table")]])
    wiped = wiped_by_term(orchestrator)
    check(wiped["1 year"]["extract"] > 0 and wiped["1 year"]["after"] == 0
          and wiped["3 year"]["after"] == wiped["3 year"]["extract"] > 0,
          "the 1-year pipeline from 2027-09 is wiped, the 3-year one is kept")
    check("rows wiped per term: {'1 year':" in console, "step 02 says which terms it wiped")
    check({"term", "term_level_1"} <= set(orchestrator.context["calendared_raw"].columns),
          "the term keeps its name in every step; its generated level is next to it")
    results = {}
    console = console_of(lambda: results.update(Orchestrator(term_configuration()).run()))
    check("units at 0 units due:" in console and "wiped on purpose by the calendar" in console,
          "step 04 tells the units the calendar wiped on purpose from the ones at 0 in the extract")
    forecast = results["forecast"]["forecast"]
    projected = forecast[forecast["origen_pipeline"] == "proyectada"]
    check(len(projected) > 0 and set(projected["term"].astype(str)) == {"1 year"},
          "step 17 projects again only the 1-year licences")


def test_a_term_column_that_is_not_mandatory() -> None:
    print("B · a term_column that is not a mandatory dim stops when the Config is built")
    check_stops(lambda: term_configuration("term_level_2"), "must be one of the mandatory dims",
                "a term_column that is not a declared mandatory dim stops before any data is read")


if __name__ == "__main__":
    test_the_term_column()
    test_a_term_column_that_is_not_mandatory()
    finish()
