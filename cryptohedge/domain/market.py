"""Market-domain value objects."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class OptionContract:
    """A single vanilla option instrument."""

    underlying: str
    strike: float
    expiry_ts: int  # nanoseconds
    is_call: bool
    notional: float = 1.0

    @property
    def opt_type(self) -> str:
        return "call" if self.is_call else "put"


@dataclass(frozen=True)
class VolatilityEstimate:
    """Daily volatility, volatility-of-volatility and a confidence interval."""

    daily_vol: float
    annualized_vol: float
    vol_of_vol: float
    ci_low: float
    ci_high: float
    confidence_level: float
    horizon_days: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HestonParameters:
    """Calibrated Heston parameters (variance dynamics under the risk-neutral measure)."""

    v0: float
    kappa: float
    theta: float
    eps: float
    rho: float
    flat_yield: float = 0.0
    calibration_error: float = float("nan")
    feller_satisfied: bool = False

    def as_array(self) -> tuple:
        return (self.v0, self.kappa, self.theta, self.eps, self.rho)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @property
    def feller_condition(self) -> float:
        """``2*kappa*theta - eps^2`` (>= 0 keeps variance strictly positive)."""
        return 2.0 * self.kappa * self.theta - self.eps**2


@dataclass(frozen=True)
class InstrumentRanking:
    """Multi-criteria ranking of a candidate hedging instrument vs the primary asset."""

    symbol: str
    pearson: float
    spearman: float
    kendall: float
    dcc_mean: float
    cointegrated: bool
    stability: float
    liquidity: float
    hedge_cost: float
    risk_reduction: float
    score: float
    relationship: str  # 'positive' | 'inverse' | 'neutral'

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
