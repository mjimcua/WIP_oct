"""pipeline.py — SFF v3 · the two runners.

  run_analysis(configuration)   ANALYSIS. Phases 0-5 with every diagnostic, writes the
                                decision tables (`decision_*`) and the analysis tables,
                                and — because the decisions are in memory — also produces
                                the forecast. This is the run of the evaluation period,
                                the one the analyst keeps.
  run_pipeline(configuration)   RUN. Phases 0, 1.1, 1.3 (ladder), 4, 5 reading the
                                decisions from the persisted `decision_*` tables written
                                by the last analysis. No η², no backtest, no bands are
                                computed: they are read. This is the monthly run, the
                                one that can be delegated. It refuses to run if a decision
                                table is missing, and warns when it is older than
                                `decision_max_age_months`.

Both call `configuration.read_raw()` (overridden by the caller) and write through
`configuration.write`. No file, no query, no legacy in this module.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import datetime
import os
import sys
import time

import pandas as pd

# The project is a flat folder imported from notebooks and scripts alike: make sure the
# folder of this file is importable BEFORE importing the sibling modules below (that is
# why those imports come after this block, not at the top).
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if PROJECT_FOLDER not in sys.path:
    sys.path.insert(0, PROJECT_FOLDER)

import analysis_backtest
import analysis_baseline
import analysis_data_profile
import analysis_dimensions
import analysis_seasonality_benchmark
import analysis_top_movers
import analysis_discount_churn
import analysis_portfolio_overview
import analysis_composition
import analysis_price_scenarios
import analysis_signal_maturation
import answers
import informe
import nucleo
import topologia
import raw_data_validation
import run_forecast_assembly
import run_rate_series
import run_support_ladder
import run_uplift
import run_validation
import support_reference
from config import Config, hash_key, join_columns
from vocabulario import *  # the persisted labels (roles, signs, treatments, origins, levels)

# ─── named constants ─────────────────────────────────────────────────────────────
SECTION_RULE = "═" * 74


def section(title: str, started_at: float) -> None:
    """Print the section header with the seconds the previous section took."""
    print(SECTION_RULE, f"\n{title}   (previous section {time.time() - started_at:.1f} s)")
DECISION_TABLES = ("decision_eta2", "decision_support", "decision_estacionalidad", "pool_reference", "decision_technique",
                   "decision_error_bands", "decision_uplift", "decision_aggregate_bands")


# ═══════════════════════════════════════════════════════════════════════════════════
# SHARED STEPS
# ═══════════════════════════════════════════════════════════════════════════════════

REQUIRED_CONFIG_FIELDS = ("timevarying_model_version", "uplift_floor", "backtest_test_start", "extended_horizon_end",
                          "backtest_horizons", "benchmark_min_support", "acquisition_min_pairs", "test_months")


def check_configuration_version(configuration: Config) -> None:
    """Stop with a clear message when config.py is older than the phases that read it."""
    missing = [name for name in REQUIRED_CONFIG_FIELDS if not hasattr(configuration, name)]
    if missing:
        raise AttributeError(f"config.py is outdated: Config has no {missing}. Replace config.py with the one of this "
                             f"delivery (your subclass with read_raw keeps working) and restart the kernel.")


def phase_0(configuration: Config) -> dict:
    """Raw → validated, conditioned, the two tables, keys, labels, reference."""
    check_configuration_version(configuration)
    started = time.time()
    print(SECTION_RULE, "\nPHASE 0 — raw data validation")
    raw = configuration.read_raw()
    print(f"[0] raw read in {time.time() - started:.1f} s · {len(raw):,} rows")
    validated = raw_data_validation.validate_raw(raw, configuration)
    conditioned = raw_data_validation.apply_current_month_doctrine(validated, configuration)
    fine_table = raw_data_validation.build_fine_table(conditioned, configuration)
    forecast_units = raw_data_validation.aggregate_to_forecast_units(fine_table, configuration)
    fu_lookup, comb_lookup = raw_data_validation.build_key_lookups(forecast_units, fine_table)
    labeled_units = raw_data_validation.label_universe_and_routes(forecast_units, configuration)
    # both fact tables carry the series id (fs_id; write stamps fs_key) so the BI joins
    # unit -> series without deriving it from fu_id
    fine_table["fs_id"] = join_columns(fine_table, configuration.rate_series_columns)
    configuration.write(labeled_units, "fact_fu")
    configuration.write(fine_table, "fact_fine")
    configuration.write(fu_lookup, "lookup_fu")
    configuration.write(comb_lookup, "lookup_comb")
    labeled_units = support_reference.add_support_reference(labeled_units, configuration)
    return dict(fine_table=fine_table, labeled_units=labeled_units, conditioned=conditioned)


def build_key_bridge(fine_table: pd.DataFrame, units: pd.DataFrame, decision_support: pd.DataFrame,
                     configuration: Config) -> pd.DataFrame:
    """One thin spine from every raw row to its series, its estimation id, its uplift cell
    and its mandatory cell. Persisted as `key_bridge`."""
    columns = list(dict.fromkeys(["fu_comb_key", "fu_id", "fu_key", "comb_id", "comb_key"]
                                 + configuration.uplift_cell_columns + configuration.business_mandatory_dims))
    bridge = fine_table.drop_duplicates("fu_comb_key")[columns].copy()
    bridge["fs_id"] = bridge["fu_id"].str.rsplit("|", n=1).str[0]
    bridge["fs_key"] = bridge["fs_id"].map(hash_key)
    bridge = bridge.merge(decision_support[["fs_id", "id_estimacion", "peldano", "signo"]], on="fs_id", how="left")
    bridge["id_estimacion"] = bridge["id_estimacion"].fillna(bridge["fs_id"])
    bridge["uplift_cell_id"] = join_columns(bridge, configuration.uplift_cell_columns)
    bridge["uplift_cell_key"] = bridge["uplift_cell_id"].map(hash_key)
    bridge["celda_id"] = join_columns(bridge, configuration.business_mandatory_dims)
    labels = units.drop_duplicates("fs_id")[["fs_id", "universo", "ruta"]]
    bridge = bridge.merge(labels, on="fs_id", how="left")
    keep = ["fu_comb_key", "fu_id", "fu_key", "comb_id", "comb_key", "fs_id", "fs_key", "id_estimacion", "peldano",
            "signo", "uplift_cell_id", "uplift_cell_key", "celda_id", "universo", "ruta"]
    configuration.write(bridge[keep], "key_bridge")
    return bridge[keep]


def forecast_horizons(fine_table: pd.DataFrame, units: pd.DataFrame, configuration: Config) -> list:
    """1..H, with H = months from the last month with truth to the end of the forecast
    (known projection or extended horizon), or `max_forecast_horizon` if configured."""
    if configuration.max_forecast_horizon is not None:
        return list(range(1, configuration.max_forecast_horizon + 1))
    last_truth = units[units["tasa"].notna()][configuration.period_col].max()
    end = fine_table[configuration.period_col].max()
    if configuration.extended_horizon_end is not None:
        end = max(end, pd.Period(configuration.extended_horizon_end, freq="M"))
    return list(range(1, max(int((end - last_truth).n), 1) + 1))


def raw_measure_totals(conditioned: pd.DataFrame, configuration: Config) -> dict:
    """The four measures of the raw as extracted (before the future's partial results
    were wiped): the numbers the core table must reproduce exactly."""
    renewed_units = conditioned["s0_renovados_unidades"] if "s0_renovados_unidades" in conditioned.columns else conditioned[configuration.renewed_units_col]
    renewed_usd = conditioned["s0_renovados_usd"] if "s0_renovados_usd" in conditioned.columns else conditioned[configuration.renewed_usd_col]
    return dict(pipeline_unidades=float(conditioned[configuration.pipeline_units_col].sum()),
                pipeline_usd=float(conditioned[configuration.pipeline_usd_col].sum()),
                renovados_unidades=float(renewed_units.sum()), renovados_usd=float(renewed_usd.sum()))


def historical_composition_effect(units: pd.DataFrame, configuration: Config) -> dict:
    """Moment 1 of the composition effect: each complete closed year against the previous one."""
    period, role = configuration.period_col, configuration.dataset_role_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES)].copy()
    closed["anio"] = pd.PeriodIndex(closed[period], freq="M").year
    complete = [y for y, n in closed.groupby("anio")[period].nunique().items() if n >= 12]
    comparisons = [(f"{a}→{b}", a, b) for a, b in zip(complete[:-1], complete[1:])]
    if not comparisons:
        return dict(efecto_composicion=pd.DataFrame(), efecto_composicion_celda=pd.DataFrame(), titulares=pd.DataFrame())
    result = analysis_composition.run_composition_effect(closed, comparisons, configuration, "anio")
    analysis_composition.print_composition_effect(result["efecto_composicion"], result["titulares"], configuration, "histórico, año contra año")
    return result


def future_composition_effect(units: pd.DataFrame, detail: pd.DataFrame, configuration: Config) -> dict:
    """Moment 2: this year (closed months + forecast) against last year, and next year
    against this one, with the forecast rows carrying pipeline × rate as expected renewals."""
    period, role = configuration.period_col, configuration.dataset_role_col
    pipe, ren, usd = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.pipeline_usd_col
    closed = units[(units.get("sintetica", 0) == 0) & units[role].isin(TRUTH_ROLES)]
    if detail is None or detail.empty or pipe not in detail.columns:              # no future rows: nothing to compare
        return dict(efecto_composicion=pd.DataFrame(), efecto_composicion_celda=pd.DataFrame(), titulares=pd.DataFrame())
    dims = list(configuration.business_mandatory_dims) + list(configuration.extra_renovacion) + list(configuration.structural_timevarying_dims)
    one_per_series = units.drop_duplicates("fs_id").set_index("fs_id")[[c for c in dims if c in units.columns]]
    future = detail.merge(one_per_series, left_on="fs_id", right_index=True, how="left")
    future = future.assign(**{ren: future[pipe] * future["tasa"]})
    keep = [period, pipe, ren, usd] + [c for c in dims if c in closed.columns]
    frame = pd.concat([closed[keep], future[[c for c in keep if c in future.columns]]], ignore_index=True)
    frame["anio"] = pd.PeriodIndex(frame[period].astype(str), freq="M").year
    years = sorted(frame["anio"].unique())
    comparisons = [(f"{a}→{b}", a, b) for a, b in zip(years[:-1], years[1:])][-2:]
    if not comparisons:
        return dict(efecto_composicion=pd.DataFrame(), efecto_composicion_celda=pd.DataFrame(), titulares=pd.DataFrame())
    result = analysis_composition.run_composition_effect(frame, comparisons, configuration, "anio", table_suffix="_futuro")
    analysis_composition.print_composition_effect(result["efecto_composicion"], result["titulares"], configuration, "futuro, con el forecast")
    return result


def draw_top_sheets(results: dict, configuration: Config) -> list:
    """The sheets of the `sheets_top_series` series with the most projected money, drawn
    into <outdir>/diagnostics/ at the end of every analysis. Never stops the run."""
    if not configuration.sheets_top_series:
        return []
    from sheet import sheet
    card = results["series_card"]
    top = card[card["ruta"] == TREATMENT_PREDICTABLE].sort_values("usd_proyectado", ascending=False).head(int(configuration.sheets_top_series))
    print(SECTION_RULE, f"\nSHEETS — the {len(top)} series with the most projected money")
    paths = []
    for _, row in top.iterrows():
        try:
            drawn = sheet(row["fs_id"], configuration, results=results, figure=True, verbose=False)
            paths.append(drawn["figure"])
            print(f"   {row['fs_id'][:70]:<70} ${row['usd_proyectado']:>13,.0f}  →  {drawn['figure']}")
        except Exception as error:
            print(f"   {row['fs_id'][:70]:<70} sheet failed: {type(error).__name__}: {error}")
    return paths


def validate(results: dict, configuration: Config, started: float) -> pd.DataFrame:
    section("VALIDATION", started)
    return run_validation.run_validation(results, configuration)


# ═══════════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════

def run_analysis(configuration: Config) -> dict:
    """The evaluation run: every diagnostic, every decision written, the forecast produced."""
    started = time.time()
    results = phase_0(configuration)
    fine_table, labeled_units = results["fine_table"], results["labeled_units"]
    raw_totals = raw_measure_totals(results["conditioned"], configuration)     # what the reporting says: the core must match it
    profiles = analysis_data_profile.run_raw_profile(results["conditioned"], configuration)

    section("PHASE 1 — rate series, dimensions, support ladder", started); started = time.time()
    units, series_summary = run_rate_series.build_rate_series(labeled_units, configuration)
    panorama = analysis_portfolio_overview.run_portfolio_overview(units, configuration)
    composition_history = historical_composition_effect(units, configuration)
    profiles.update(analysis_data_profile.run_fu_profile(units, fine_table, configuration))
    dimensions = analysis_dimensions.run_dimension_analysis(units, series_summary, configuration)
    series_estimates, series_card, decision_support, parent_ladder = run_support_ladder.run_support_ladder(
        units, series_summary, dimensions["decision_eta2"], configuration)
    key_bridge = build_key_bridge(fine_table, units, decision_support, configuration)
    topology = topologia.run_topology_profile(fine_table, units, series_card, configuration)
    composition = analysis_dimensions.run_composition_analysis(units, configuration)
    discount_churn = analysis_discount_churn.run_discount_churn(fine_table, configuration)

    section("PHASE 2 — seasonality benchmark (big series) and pool reference", started); started = time.time()
    test_months = units.loc[units[configuration.dataset_role_col] == ROLE_TEST, configuration.period_col]
    first_test_month = str(test_months.min()) if len(test_months) else None
    benchmark = analysis_seasonality_benchmark.run_seasonality_benchmark(series_card, units, fine_table, configuration)
    monthly_series = analysis_backtest.monthly_series_by_estimation_id(units, decision_support, parent_ladder, configuration)
    pool_reference = analysis_backtest.build_pool_reference(monthly_series, benchmark["decision_estacionalidad"], configuration)

    section("PHASE 3 — backtest, technique, bands", started); started = time.time()
    forecast_horizon = min(forecast_horizons(fine_table, units, configuration)[-1], configuration.backtest_horizon_cap)
    judged_horizons = sorted({h for h in configuration.backtest_horizons if h <= forecast_horizon} | {forecast_horizon})
    backtest = analysis_backtest.run_backtest_analysis(monthly_series, pool_reference, configuration, judged_horizons, first_test_month)

    section("PHASE 4 — uplift", started); started = time.time()
    decision_uplift = run_uplift.run_uplift(fine_table, configuration)

    section("PHASE 5 — assembly, extended horizon, bands", started); started = time.time()
    decisions = dict(decision_eta2=dimensions["decision_eta2"], decision_support=decision_support,
                     decision_estacionalidad=benchmark["decision_estacionalidad"], pool_reference=pool_reference,
                     decision_technique=backtest["decision_technique"],
                     decision_error_bands=backtest["decision_error_bands"], decision_uplift=decision_uplift,
                     backtest_holdout_aggregate=backtest["backtest_holdout_aggregate"],
                     decision_aggregate_bands=backtest["decision_aggregate_bands"])
    backtest_uplift_table = run_uplift.backtest_uplift(fine_table, backtest["backtest_holdout"], decision_support, configuration)
    configuration.write(backtest_uplift_table, "backtest_uplift")
    run_uplift.print_backtest_uplift(backtest_uplift_table, configuration)
    forecast = run_forecast_assembly.run_forecast_assembly(fine_table, units, series_estimates, series_card,
                                                           decisions, monthly_series, configuration, backtest["backtest_holdout"],
                                                           composition["mix_shift_decomposition"])
    composition_future = future_composition_effect(units, forecast["forecast_detail"], configuration)
    scenarios = analysis_price_scenarios.run_price_scenarios(forecast["forecast_detail"], fine_table, forecast["forecast_units_extended"], units, configuration)
    maturation = analysis_signal_maturation.run_signal_maturation(units, series_card, forecast, configuration)
    forecast = answers.carry_signal_adjustment(forecast, maturation["ajuste_por_mes"], configuration)
    baseline = analysis_baseline.run_baseline(fine_table, forecast["forecast_units_extended"], forecast["business_summary"], configuration)
    movers = analysis_top_movers.run_top_movers(units, series_card, forecast, decisions, backtest["backtest_holdout"], configuration)
    results_so_far = dict(fine_table=fine_table, units=units, series_summary=series_summary, series_estimates=series_estimates,
                          series_card=series_card, key_bridge=key_bridge, parent_ladder=parent_ladder, monthly_series=monthly_series,
                          decisions=decisions, backtest=backtest, dimensions=dimensions, composition=composition, forecast=forecast)
    sheets = draw_top_sheets(results_so_far, configuration)
    report = validate(dict(
        fine_table=fine_table, forecast_units=units, key_bridge=key_bridge, forecast_detail=forecast["forecast_detail"],
        forecast_bands=forecast["forecast_bands"], horizon_report=forecast["horizon_report"], series_card=series_card,
        decision_support=decision_support, parent_ladder=parent_ladder, backtest_holdout=backtest["backtest_holdout"],
        decision_uplift=decision_uplift), configuration, started)
    results = dict(fine_table=fine_table, units=units, series_summary=series_summary, series_estimates=series_estimates,
                   series_card=series_card, key_bridge=key_bridge, parent_ladder=parent_ladder, monthly_series=monthly_series,
                   decisions=decisions, backtest=backtest, dimensions=dimensions, forecast=forecast, validation=report,
                   profiles=profiles, baseline=baseline, benchmark=benchmark, composition=composition, sheets=sheets, top_movers=movers,
                   signal_maturation=maturation, discount_churn=discount_churn, panorama=panorama,
                   composition_history=composition_history, composition_future=composition_future, scenarios=scenarios,
                   backtest_uplift=backtest_uplift_table, topologia=topology)
    section("THE CORE TABLE — the whole story in one table, left to right", started); started = time.time()
    results["nucleo"] = nucleo.run_core_table(results, configuration, raw_totals)
    try:
        results["informe"] = informe.build_report(results, configuration)
    except Exception as error:                              # the report never stops the analysis
        print(f"[informe] failed: {type(error).__name__}: {error}")
        results["informe"] = None
    print(SECTION_RULE, "\nanalysis complete · decisions and tables in", configuration.outdir)
    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# RUN (monthly, delegable)
# ═══════════════════════════════════════════════════════════════════════════════════

def read_decisions(configuration: Config) -> dict:
    """Read the decision tables from the configured engine; refuse if any is missing."""
    if configuration.engine is None:
        raise RuntimeError("run_pipeline needs an engine with the decision tables written by run_analysis")
    decisions = {}
    for logical_name in DECISION_TABLES:
        physical = configuration._resolve_physical_table_name(logical_name)
        try:
            table = pd.read_sql(f"SELECT * FROM {configuration._qualified_table_name(physical)}", configuration.engine)
        except Exception as error:
            raise RuntimeError(f"decision table '{logical_name}' ({physical}) is missing: run the analysis first") from error
        if "process_date" in table.columns and len(table):
            written = pd.to_datetime(table["process_date"].iloc[0])
            age_months = (datetime.datetime.now() - written).days / 30.4
            if age_months > configuration.decision_max_age_months:
                print(f"[run] WARNING decision '{logical_name}' is {age_months:.1f} months old "
                      f"(max {configuration.decision_max_age_months}): the analysis has expired")
        decisions[logical_name] = table.drop(columns=[c for c in ("process_date", "execution_id") if c in table.columns])
    return decisions


def run_pipeline(configuration: Config) -> dict:
    """The monthly run: read decisions, apply them, produce the forecast. No decision is taken."""
    started = time.time()
    decisions = read_decisions(configuration)
    results = phase_0(configuration)
    fine_table, labeled_units = results["fine_table"], results["labeled_units"]
    section("PHASE 1 — rate series and support (decisions read)", started); started = time.time()
    units, series_summary = run_rate_series.build_rate_series(labeled_units, configuration)
    series_estimates, series_card, decision_support, parent_ladder = run_support_ladder.run_support_ladder(
        units, series_summary, decisions["decision_eta2"], configuration)
    key_bridge = build_key_bridge(fine_table, units, decision_support, configuration)
    monthly_series = analysis_backtest.monthly_series_by_estimation_id(units, decision_support, parent_ladder, configuration)
    section("PHASE 4 — uplift", started); started = time.time()
    decision_uplift = run_uplift.run_uplift(fine_table, configuration)
    section("PHASE 5 — assembly (decisions read)", started); started = time.time()
    applied = dict(decisions, decision_support=decision_support, decision_uplift=decision_uplift)
    forecast = run_forecast_assembly.run_forecast_assembly(fine_table, units, series_estimates, series_card,
                                                           applied, monthly_series, configuration)
    maturation = analysis_signal_maturation.run_signal_maturation(units, series_card, forecast, configuration)
    forecast = answers.carry_signal_adjustment(forecast, maturation["ajuste_por_mes"], configuration)
    report = validate(dict(
        fine_table=fine_table, forecast_units=units, key_bridge=key_bridge, forecast_detail=forecast["forecast_detail"],
        forecast_bands=forecast["forecast_bands"], horizon_report=forecast["horizon_report"], series_card=series_card,
        decision_support=decision_support, parent_ladder=parent_ladder, decision_uplift=decision_uplift), configuration, started)
    monthly = dict(fine_table=fine_table, units=units, series_card=series_card, parent_ladder=parent_ladder, forecast=forecast,
                   decisions=applied, backtest={}, signal_maturation=maturation, validation=report)
    monthly["nucleo"] = nucleo.run_core_table(monthly, configuration, raw_measure_totals(results["conditioned"], configuration))
    print(SECTION_RULE, "\nmonthly run complete · tables in", configuration.outdir)
    return monthly
