"""Live spot provider using Binance public REST endpoints (no API key required).

Spot OHLCV for the universe is fetched from ``/api/v3/klines``. Liquid 30-day
option chains for 100 assets are not freely available without registration, so
the option chain for the primary symbol is generated Heston-consistently around
the *live* spot (clearly documented). If the network is unavailable the provider
degrades gracefully to the fully synthetic dataset, preserving reproducibility.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import requests

from cryptohedge.core.config import DataConfig
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider
from cryptohedge.services.providers.synthetic import DEFAULT_UNIVERSE, SyntheticProvider


class BinanceProvider(MarketDataProvider):
    name = "binance"

    def __init__(self, config: DataConfig, seed: int, n_steps: int = 90) -> None:
        self.config = config
        self.seed = seed
        self.n_steps = n_steps

    def _symbols(self) -> List[str]:
        primary = self.config.primary_symbol
        source = self.config.symbols or DEFAULT_UNIVERSE
        symbols = [s for s in dict.fromkeys(source) if s != primary]
        return [primary] + symbols[: max(0, self.config.universe_size - 1)]

    def _fetch_klines(self, symbol: str, limit: int) -> pd.DataFrame:
        url = f"{self.config.binance_base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": self.config.bar_interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=self.config.request_timeout_s)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "qav", "trades", "tbav", "tqav", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["symbol"] = symbol
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    def load(self) -> MarketDataBundle:
        symbols = self._symbols()
        limit = self.n_steps + 1
        frames = []
        good: List[str] = []
        try:
            for sym in symbols:
                try:
                    frames.append(self._fetch_klines(sym, limit))
                    good.append(sym)
                    time.sleep(0.05)
                except Exception:
                    continue
        except Exception:
            frames = []

        if len(good) < 5 or self.config.primary_symbol not in good:
            # network unavailable / too few symbols -> deterministic fallback
            return SyntheticProvider(self.config, seed=self.seed, n_steps=self.n_steps).load()

        spot_bars = pd.concat(frames, ignore_index=True)
        # align on common timestamps
        spot_close = spot_bars.pivot_table(index="timestamp", columns="symbol", values="close")
        spot_close = spot_close.dropna(axis=1, how="any").dropna(axis=0, how="any")
        good = [s for s in good if s in spot_close.columns]

        # Heston-consistent option chain around the live primary spot.
        synth = SyntheticProvider(self.config, seed=self.seed, n_steps=len(spot_close) - 1)
        live_primary = spot_close[self.config.primary_symbol].to_numpy()
        option_market_data = self._synthetic_options(synth, spot_close.index, live_primary)

        return MarketDataBundle(
            spot_bars=spot_bars[spot_bars["symbol"].isin(good)],
            spot_close=spot_close[good],
            option_market_data=option_market_data,
            symbols=good,
            primary_symbol=self.config.primary_symbol,
            meta={"provider": self.name, "live_symbols": len(good)},
        ).validate()

    def _synthetic_options(self, synth: SyntheticProvider, index, spot_path: np.ndarray) -> pd.DataFrame:
        from cryptohedge.services.providers.base import INSTR_ASSET, MARKET_DATA_COLUMNS

        ts_ns = index.astype("int64").to_numpy()
        n = len(spot_path)
        expiry_ns = int(ts_ns[-1] + self.config.option_expiry_days * 86_400_000_000_000)
        strikes = synth._strike_grid(float(spot_path[0]))
        # approximate instantaneous variance from realised returns
        rets = np.diff(np.log(spot_path))
        var = np.concatenate([[np.var(rets) * 365], pd.Series(rets).rolling(10, min_periods=1).var().to_numpy() * 365])
        records = []
        for t in range(n):
            spot = float(spot_path[t])
            records.append(
                {
                    "sample_idx": t, "timestamp": int(ts_ns[t]), "instrument_type": INSTR_ASSET,
                    "strike": 0.0, "expiry_ts": 0, "time_to_maturity": 0.0, "price": spot,
                    "best_bid_price": spot * 0.9999, "best_ask_price": spot * 1.0001,
                    "bid_amount_total": 100.0, "ask_amount_total": 100.0,
                    "bid_vwap": spot * 0.9999, "ask_vwap": spot * 1.0001,
                }
            )
            records.extend(
                synth._option_rows(t, int(ts_ns[t]), spot, float(max(var[t], 1e-3)), t, n - 1, expiry_ns, strikes)
            )
        md = pd.DataFrame.from_records(records, columns=MARKET_DATA_COLUMNS)
        md["timestamp"] = pd.to_datetime(md["timestamp"])
        return md
