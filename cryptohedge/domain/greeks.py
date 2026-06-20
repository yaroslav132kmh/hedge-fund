"""Greeks value objects (per-instrument and aggregated portfolio sensitivities)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict


@dataclass(frozen=True)
class Greeks:
    """First- and second-order option sensitivities.

    All greeks are expressed for the held ``notional`` of the instrument.
    """

    premium: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0

    def scaled(self, qty: float) -> "Greeks":
        return Greeks(**{f.name: getattr(self, f.name) * qty for f in fields(self)})

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregated greeks across the whole position book."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    premium: float = 0.0

    @classmethod
    def from_greeks(cls, greeks: Greeks) -> "PortfolioGreeks":
        return cls(
            delta=greeks.delta,
            gamma=greeks.gamma,
            vega=greeks.vega,
            theta=greeks.theta,
            rho=greeks.rho,
            vanna=greeks.vanna,
            volga=greeks.volga,
            charm=greeks.charm,
            premium=greeks.premium,
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
