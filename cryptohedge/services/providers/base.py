"""Provider abstractions shared by all data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# Instrument-type codes matching the bundled dataset schema (see *.parquet descriptions).
INSTR_ASSET = 7
INSTR_CALL = 5
INSTR_PUT = 6

MARKET_DATA_COLUMNS = [
    "sample_idx",
    "timestamp",
    "instrument_type",
    "strike",
    "expiry_ts",
    "time_to_maturity",
    "price",
    "best_bid_price",
    "best_ask_price",
    "bid_amount_total",
    "ask_amount_total",
    "bid_vwap",
    "ask_vwap",
]


@dataclass
class MarketDataBundle:
    """Everything the downstream agents need from a data source.

    Attributes:
        spot_bars: Long OHLCV frame with columns
            ``[timestamp, symbol, open, high, low, close, volume]``.
        spot_close: Wide close-price frame ``[timestamp x symbol]``.
        option_market_data: Option + spot quotes for the primary symbol following
            the bundled ``market_data`` schema (see :data:`MARKET_DATA_COLUMNS`).
        symbols: The instrument universe.
        primary_symbol: The asset whose risk is hedged (e.g. ``BTCUSDT``).
        meta: Free-form provenance metadata.
    """

    spot_bars: pd.DataFrame
    spot_close: pd.DataFrame
    option_market_data: pd.DataFrame
    symbols: List[str]
    primary_symbol: str
    meta: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> "MarketDataBundle":
        if self.spot_close.isna().all().any():
            raise ValueError("spot_close contains an all-NaN column")
        missing = set(MARKET_DATA_COLUMNS) - set(self.option_market_data.columns)
        if missing:
            raise ValueError(f"option_market_data missing columns: {sorted(missing)}")
        if self.primary_symbol not in self.symbols:
            raise ValueError("primary_symbol not present in symbols")
        return self


class MarketDataProvider(ABC):
    """Contract for a market-data source."""

    name: str = "base"

    @abstractmethod
    def load(self) -> MarketDataBundle:
        """Return a fully populated, validated :class:`MarketDataBundle`."""
