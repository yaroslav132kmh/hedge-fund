"""Bundled provider: reads the pre-generated, version-controlled dataset.

This is the default, fully reproducible source. If the files are missing it
transparently regenerates them with the :class:`SyntheticProvider` (same seed),
so a fresh checkout always works offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptohedge.core.config import DataConfig
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider
from cryptohedge.services.providers.synthetic import SyntheticProvider


class BundledProvider(MarketDataProvider):
    name = "bundled"

    def __init__(self, config: DataConfig, root: str | Path, seed: int, n_steps: int = 90) -> None:
        self.config = config
        self.root = Path(root)
        self.seed = seed
        self.n_steps = n_steps
        self.raw_dir = self.root / "data" / "raw"

    def _paths(self):
        return (
            self.raw_dir / "spot_bars.parquet",
            self.raw_dir / "spot_close.parquet",
            self.raw_dir / "market_data.parquet",
            self.raw_dir / "universe.csv",
        )

    def exists(self) -> bool:
        return all(p.exists() for p in self._paths())

    def materialize(self) -> MarketDataBundle:
        """Generate the synthetic dataset and persist it to ``data/raw``."""
        bundle = SyntheticProvider(self.config, seed=self.seed, n_steps=self.n_steps).load()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        bars, close, md, universe = self._paths()
        bundle.spot_bars.to_parquet(bars)
        bundle.spot_close.to_parquet(close)
        bundle.option_market_data.to_parquet(md)
        pd.Series(bundle.symbols, name="symbol").to_csv(universe, index=False)
        pd.Series(bundle.meta).to_json(self.raw_dir / "meta.json")
        return bundle

    def load(self) -> MarketDataBundle:
        if not self.exists():
            return self.materialize()
        bars, close, md, universe = self._paths()
        spot_bars = pd.read_parquet(bars)
        spot_close = pd.read_parquet(close)
        option_market_data = pd.read_parquet(md)
        symbols = pd.read_csv(universe)["symbol"].tolist()
        return MarketDataBundle(
            spot_bars=spot_bars,
            spot_close=spot_close,
            option_market_data=option_market_data,
            symbols=symbols,
            primary_symbol=self.config.primary_symbol,
            meta={"provider": self.name, "source": str(self.raw_dir)},
        ).validate()
