"""Shared pytest fixtures.

The integration fixture runs the full multi-agent pipeline exactly once per test
session in an isolated temporary root, using a small but representative
configuration so the suite stays fast while still exercising every agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# A small, fast but complete configuration that still touches every code path.
FAST_OVERRIDES = {
    "seed": 12345,
    "data": {"universe_size": 6},
    "horizons": {"analysis_days": 24},
    "market_analysis": {
        "top_n_hedge_instruments": 3,
        "regime_n_states": 3,
        "correlation": {"methods": ["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]},
    },
    "hedging": {"calibration_subsample": 4},
    "backtest": {"train_window": 10, "test_window": 4, "step": 4},
    "runtime": {"resume": False, "checkpointing": True},
}

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="session")
def fast_config():
    from cryptohedge.core.config import load_config

    return load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """Run the whole pipeline once in an isolated root; reused by integration tests."""
    from cryptohedge.agents import build_pipeline
    from cryptohedge.core.config import load_config
    from cryptohedge.core.context import AgentContext

    root = tmp_path_factory.mktemp("run")
    config = load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)
    context = AgentContext(config, root=root)
    orchestrator = build_pipeline(context, fail_fast=True)
    report = orchestrator.run()
    return context, report, root


@pytest.fixture
def rng():
    return np.random.default_rng(0)
