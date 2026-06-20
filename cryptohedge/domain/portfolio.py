"""Portfolio-domain value objects."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal


@dataclass(frozen=True)
class Position:
    """A held quantity of an instrument."""

    instrument: str
    quantity: float
    price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Trade:
    """An executed trade and the fee it incurred."""

    instrument: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float
    reason: str = ""

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity

    @property
    def cash_flow(self) -> float:
        """Cash impact of the trade including fees (negative = cash out)."""
        return -self.signed_quantity * self.price - self.fee

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
