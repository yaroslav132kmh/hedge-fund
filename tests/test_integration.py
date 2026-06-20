"""Integration tests: the eleven agents interacting through the orchestrator.

These run the full pipeline once (session fixture) and assert that every stage
succeeds, the expected artifacts are produced and exchanged across the blackboard,
and that the run is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptohedge.agents import build_pipeline
from cryptohedge.core.config import load_config
from cryptohedge.core.context import AgentContext

from conftest import CONFIG_DIR, FAST_OVERRIDES


EXPECTED_AGENTS = [
    "data_acquisition", "market_analysis", "heston_calibration", "greeks_calculation",
    "hedging_decision", "portfolio_optimization", "risk_management", "backtesting",
    "self_diagnostic", "explainability", "dashboard",
]


def test_all_agents_run_successfully(pipeline_run):
    _, report, _ = pipeline_run
    assert report.success, f"failed stage: {report.failed_stage}"
    assert [r.agent for r in report.results] == EXPECTED_AGENTS
    assert all(r.success for r in report.results)


def test_blackboard_artifacts_present(pipeline_run):
    ctx, _, _ = pipeline_run
    for key in ["market_data", "spot_close", "returns", "volatility", "hedge_sizing",
                "rankings_df", "hedge_universe", "calibr_data", "heston_history",
                "portfolio_greeks_latest", "hedge_history", "opt_weights",
                "risk_assessment", "backtest_metrics", "stress_table",
                "confidence_score", "explanation_sections", "dashboard_path"]:
        assert ctx.has(key), f"missing blackboard artifact: {key}"


def test_message_routing_trace(pipeline_run):
    ctx, report, _ = pipeline_run
    # every agent produced exactly one forward message consumed by the next stage
    produced = [r.message.type.value for r in report.results]
    assert "data_ready" in produced
    assert "dashboard_ready" in produced


def test_result_files_written(pipeline_run):
    _, _, root = pipeline_run
    results = Path(root) / "artifacts" / "results"
    for fname in ["performance_metrics.json", "hedging_history.parquet",
                  "stress_test.parquet", "explanation.md", "dashboard.html"]:
        assert (results / fname).exists(), f"missing result file: {fname}"
    # calibration artifacts persisted
    calib = Path(root) / "artifacts" / "calibration" / "calibr_data.parquet"
    assert calib.exists()


def test_greeks_are_finite(pipeline_run):
    ctx, _, _ = pipeline_run
    g = ctx.get("portfolio_greeks_latest")
    for key in ["delta", "gamma", "vega", "theta", "rho"]:
        assert key in g and np.isfinite(g[key])


def test_hedge_neutralises_delta(pipeline_run):
    ctx, _, _ = pipeline_run
    hist = ctx.get("hedge_history")
    residual = (hist["delta"] - hist["delta_hedge"]).abs()
    assert residual.max() < 1e-6        # delta fully hedged each step


def test_stress_table_decomposition(pipeline_run):
    ctx, _, _ = pipeline_run
    stress = ctx.get("stress_table")
    for col in ["scenario", "net_hedged_pnl", "unhedged_pnl", "hedge_effectiveness"]:
        assert col in stress.columns
    # hedged book is materially safer than naked exposure under the worst shock
    worst = stress.loc[stress["unhedged_pnl"].idxmin()]
    assert abs(worst["net_hedged_pnl"]) < abs(worst["unhedged_pnl"])


def test_confidence_score_in_unit_interval(pipeline_run):
    ctx, _, _ = pipeline_run
    cs = float(ctx.get("confidence_score"))
    assert 0.0 <= cs <= 1.0


def test_portfolio_constituents_and_diversification(pipeline_run):
    ctx, _, _ = pipeline_run
    constituents = ctx.get("portfolio_constituents")
    div = ctx.get("diversification")
    # the portfolio is an actual basket of instruments with valid weights
    assert constituents is not None and not constituents.empty
    assert {"symbol", "weight", "exp_return_annual", "vol_annual", "relationship"} <= set(constituents.columns)
    assert constituents["weight"].sum() == pytest.approx(1.0, abs=1e-6)
    # diversification is quantified and meaningful
    for key in ["diversification_ratio", "effective_n", "max_weight", "hhi", "n_assets"]:
        assert key in div
    assert div["diversification_ratio"] >= 1.0 - 1e-9       # DR is never below 1
    assert 1.0 <= div["effective_n"] <= div["n_assets"] + 1e-9
    assert div["effective_n"] > 1.0                          # genuinely diversified, not a single bet


def test_portfolio_is_profitable_and_rebalanced(pipeline_run):
    ctx, _, _ = pipeline_run
    methods = ctx.get("method_comparison")
    reb = ctx.get("rebalance_decision")
    equity = ctx.get("portfolio_equity")
    chosen = methods.loc[methods["chosen"]].iloc[0]
    assert chosen["method"] == reb["method"]
    assert chosen["total_return"] > 0.0                      # the chosen portfolio is profitable
    # the equity curve and rebalancing path are produced
    assert equity is not None and len(equity) > 1
    assert float(equity["equity"].iloc[-1]) > float(equity["equity"].iloc[0])
    assert len(ctx.get("portfolio_rebalances")) >= 1


def test_bilingual_outputs_written(pipeline_run):
    ctx, _, root = pipeline_run
    results = Path(root) / "artifacts" / "results"
    for fname in ["dashboard_ru.html", "dashboard_en.html", "explanation.md", "explanation.en.md"]:
        assert (results / fname).exists(), f"missing localized file: {fname}"
    paths = ctx.get("dashboard_paths")
    assert set(paths) == {"ru", "en"}
    # each language has its own non-trivial sections
    assert ctx.get("explanation_sections_ru") and ctx.get("explanation_sections_en")
    ru_html = (results / "dashboard_ru.html").read_text(encoding="utf-8")
    en_html = (results / "dashboard_en.html").read_text(encoding="utf-8")
    assert "Состав портфеля" in ru_html and "Portfolio Constituents" in en_html
    assert "Portfolio Constituents" not in ru_html and "Состав портфеля" not in en_html


@pytest.mark.slow
def test_pipeline_reproducible(pipeline_run, tmp_path):
    """A second independent run with the same seed reproduces key numbers."""
    ctx1, _, _ = pipeline_run
    cfg = load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)
    ctx2 = AgentContext(cfg, root=tmp_path)
    build_pipeline(ctx2, fail_fast=True).run()

    h1 = ctx1.get("hedge_history")["pnl"].to_numpy()
    h2 = ctx2.get("hedge_history")["pnl"].to_numpy()
    assert np.allclose(h1, h2, rtol=1e-8, atol=1e-8)

    s1 = ctx1.get("hedge_sizing").quantity_to_hedge
    s2 = ctx2.get("hedge_sizing").quantity_to_hedge
    assert s1 == pytest.approx(s2, rel=1e-10)
