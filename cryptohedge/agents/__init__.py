"""The eleven autonomous agents of the hedging system.

Each agent lives in its own module and implements the unified
:class:`cryptohedge.core.agent.BaseAgent` interface. :func:`build_pipeline` wires
them into an :class:`cryptohedge.core.Orchestrator` in dependency order.
"""

from __future__ import annotations

from typing import List

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.orchestrator import Orchestrator

from cryptohedge.agents.data_acquisition import DataAcquisitionAgent
from cryptohedge.agents.market_analysis import MarketAnalysisAgent
from cryptohedge.agents.heston_calibration import HestonCalibrationAgent
from cryptohedge.agents.greeks_calculation import GreeksCalculationAgent
from cryptohedge.agents.hedging_decision import HedgingDecisionAgent
from cryptohedge.agents.portfolio_optimization import PortfolioOptimizationAgent
from cryptohedge.agents.risk_management import RiskManagementAgent
from cryptohedge.agents.backtesting import BacktestingAgent
from cryptohedge.agents.self_diagnostic import SelfDiagnosticAgent
from cryptohedge.agents.explainability import ExplainabilityAgent
from cryptohedge.agents.dashboard import DashboardAgent

__all__ = [
    "DataAcquisitionAgent",
    "MarketAnalysisAgent",
    "HestonCalibrationAgent",
    "GreeksCalculationAgent",
    "HedgingDecisionAgent",
    "PortfolioOptimizationAgent",
    "RiskManagementAgent",
    "BacktestingAgent",
    "SelfDiagnosticAgent",
    "ExplainabilityAgent",
    "DashboardAgent",
    "all_agents",
    "build_pipeline",
]


def all_agents() -> List[BaseAgent]:
    """Instantiate the eleven agents in canonical pipeline order."""
    return [
        DataAcquisitionAgent(),
        MarketAnalysisAgent(),
        HestonCalibrationAgent(),
        GreeksCalculationAgent(),
        HedgingDecisionAgent(),
        PortfolioOptimizationAgent(),
        RiskManagementAgent(),
        BacktestingAgent(),
        SelfDiagnosticAgent(),
        ExplainabilityAgent(),
        DashboardAgent(),
    ]


def build_pipeline(context: AgentContext, fail_fast: bool = True) -> Orchestrator:
    """Create and wire the full agent pipeline."""
    orch = Orchestrator(context, fail_fast=fail_fast)
    for agent in all_agents():
        orch.register(agent)
    return orch
