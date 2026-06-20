"""Market-data providers (pluggable data sources)."""

from __future__ import annotations

from cryptohedge.core.config import DataConfig
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider
from cryptohedge.services.providers.bundled import BundledProvider
from cryptohedge.services.providers.synthetic import SyntheticProvider

__all__ = [
    "MarketDataBundle",
    "MarketDataProvider",
    "BundledProvider",
    "SyntheticProvider",
    "build_provider",
]


def build_provider(config: DataConfig, root, seed: int, n_steps: int = 90) -> MarketDataProvider:
    """Factory selecting a provider by configuration (Open-Closed principle)."""
    provider = config.provider.lower()
    if provider == "synthetic":
        return SyntheticProvider(config, seed=seed, n_steps=n_steps)
    if provider == "bundled":
        return BundledProvider(config, root=root, seed=seed, n_steps=n_steps)
    if provider == "binance":
        from cryptohedge.services.providers.binance import BinanceProvider

        return BinanceProvider(config, seed=seed, n_steps=n_steps)
    raise ValueError(f"Unknown data provider: {config.provider}")
