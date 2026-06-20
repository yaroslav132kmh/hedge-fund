"""Decision-domain value objects produced by the agents."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Literal


@dataclass(frozen=True)
class HedgeDecision:
    """A delta/vega hedging instruction with quantitative justification."""

    timestamp: int
    instrument: Literal["spot", "vega_option"]
    side: Literal["buy", "sell", "hold"]
    quantity: float
    target_greek: str  # 'delta' | 'vega'
    pre_hedge_value: float
    post_hedge_value: float
    rationale: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RebalanceDecision:
    """A portfolio rebalancing instruction from the optimization agent."""

    method: str
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    turnover: float
    expected_return: float
    expected_risk: float
    transaction_cost: float
    triggered: bool
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StopLevel:
    """An adaptive stop-loss / trailing-stop level."""

    instrument: str
    side: Literal["long", "short"]
    stop_price: float
    reference_price: float
    distance_pct: float
    method: str
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAssessment:
    """Output of the risk-management agent for a single evaluation point."""

    var: float
    cvar: float
    expected_shortfall: float
    max_drawdown: float
    within_limits: bool
    breached_limits: List[str] = field(default_factory=list)
    utilization: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
