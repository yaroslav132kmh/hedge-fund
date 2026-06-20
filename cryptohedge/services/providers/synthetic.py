"""Deterministic synthetic market generator.

Produces a fully self-contained, reproducible dataset:

* a 100-asset spot universe with a realistic correlation structure (positively
  correlated, inversely correlated, neutral and a few cointegrated names), driven
  by a Heston-simulated BTC factor;
* a Heston-consistent option chain (calls + puts, single 30-day-ish expiry) for
  the primary symbol at every daily slice, so that Heston re-calibration recovers
  meaningful, slowly drifting parameters.

Everything is a pure function of the configured ``seed``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from cryptohedge.core.config import DataConfig
from cryptohedge.domain.market import HestonParameters
from cryptohedge.services.heston_pricing import heston_premiums
from cryptohedge.services.providers.base import (
    INSTR_ASSET,
    INSTR_CALL,
    INSTR_PUT,
    MARKET_DATA_COLUMNS,
    MarketDataBundle,
    MarketDataProvider,
)

_NS_PER_DAY = 86_400_000_000_000
_YEAR_DAYS = 365.0

# A curated set of real tickers; padded with synthetic names up to universe_size.
DEFAULT_UNIVERSE: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT",
    "ATOMUSDT", "XLMUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "NEARUSDT", "ICPUSDT", "INJUSDT", "SUIUSDT", "AAVEUSDT", "UNIUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "EGLDUSDT", "FTMUSDT", "THETAUSDT", "ALGOUSDT", "XTZUSDT",
    "EOSUSDT", "GALAUSDT", "GRTUSDT", "CHZUSDT", "ZECUSDT", "DASHUSDT", "MKRUSDT",
    "SNXUSDT", "COMPUSDT", "CRVUSDT", "1INCHUSDT", "ENJUSDT", "BATUSDT", "ZILUSDT",
    "RUNEUSDT",
]

# Approximate starting price levels (USD) for realism.
_BASE_PRICES = {
    "BTCUSDT": 45_000.0, "ETHUSDT": 2_400.0, "BNBUSDT": 310.0, "SOLUSDT": 100.0,
    "XRPUSDT": 0.6, "ADAUSDT": 0.55, "DOGEUSDT": 0.08, "AVAXUSDT": 35.0,
    "DOTUSDT": 7.5, "LINKUSDT": 15.0, "MATICUSDT": 0.9, "LTCUSDT": 70.0,
}


class SyntheticProvider(MarketDataProvider):
    name = "synthetic"

    def __init__(self, config: DataConfig, seed: int, n_steps: int = 90) -> None:
        self.config = config
        self.seed = seed
        self.n_steps = int(n_steps)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ universe
    def _universe(self) -> List[str]:
        primary = self.config.primary_symbol
        source = list(dict.fromkeys(self.config.symbols)) if self.config.symbols else list(DEFAULT_UNIVERSE)
        symbols = [s for s in source if s != primary]
        i = 0
        while len(symbols) + 1 < self.config.universe_size:
            i += 1
            cand = f"SYN{i:03d}USDT"
            if cand != primary and cand not in symbols:
                symbols.append(cand)
        symbols = symbols[: max(0, self.config.universe_size - 1)]
        return [primary] + symbols

    # --------------------------------------------------------------- simulation
    def _simulate_btc(self, n_steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Full-truncation Euler Heston path for the BTC factor."""
        kappa, theta, eps, rho, mu, v0 = 3.0, 0.04 * 4, 0.6, -0.6, 0.05, 0.04
        S = np.empty(n_steps + 1)
        v = np.empty(n_steps + 1)
        S[0], v[0] = _BASE_PRICES["BTCUSDT"], v0
        z = self.rng.standard_normal((n_steps, 2))
        for t in range(n_steps):
            w1 = z[t, 0]
            w2 = rho * z[t, 0] + math.sqrt(1.0 - rho**2) * z[t, 1]
            vt = max(v[t], 0.0)
            S[t + 1] = S[t] * math.exp((mu - 0.5 * vt) * dt + math.sqrt(vt * dt) * w1)
            v[t + 1] = max(v[t] + kappa * (theta - vt) * dt + eps * math.sqrt(vt * dt) * w2, 1e-4)
        return S, v

    def _simulate_universe(self, symbols: List[str], btc_close: np.ndarray) -> pd.DataFrame:
        """Factor model: each asset loads on BTC returns plus idiosyncratic noise."""
        n_steps = len(btc_close) - 1
        btc_ret = np.diff(np.log(btc_close))
        n_assets = len(symbols)

        # relationship buckets: ~65% positive, ~15% inverse, ~12% neutral, ~8% cointegrated
        betas = np.empty(n_assets)
        idio = np.empty(n_assets)
        relationship = np.empty(n_assets, dtype=object)
        for i, sym in enumerate(symbols):
            if sym == self.config.primary_symbol:
                betas[i], idio[i], relationship[i] = 1.0, 0.0, "primary"
                continue
            u = self.rng.random()
            if u < 0.65:
                betas[i] = self.rng.uniform(0.4, 1.6)
                idio[i] = self.rng.uniform(0.005, 0.02)
                relationship[i] = "positive"
            elif u < 0.80:
                betas[i] = self.rng.uniform(-1.3, -0.3)
                idio[i] = self.rng.uniform(0.005, 0.02)
                relationship[i] = "inverse"
            elif u < 0.92:
                betas[i] = self.rng.uniform(-0.05, 0.05)
                idio[i] = self.rng.uniform(0.02, 0.05)
                relationship[i] = "neutral"
            else:
                betas[i] = self.rng.uniform(0.8, 1.2)
                idio[i] = self.rng.uniform(0.002, 0.008)
                relationship[i] = "cointegrated"

        eps_mat = self.rng.standard_normal((n_assets, n_steps)) * idio[:, None]
        log_ret = betas[:, None] * btc_ret[None, :] + eps_mat

        base = np.array([_BASE_PRICES.get(s, float(self.rng.uniform(0.2, 250.0))) for s in symbols])
        closes = np.empty((n_assets, n_steps + 1))
        closes[:, 0] = base
        closes[:, 1:] = base[:, None] * np.exp(np.cumsum(log_ret, axis=1))
        closes[0] = btc_close  # keep exact BTC path

        # overlay genuine cointegration for the 'cointegrated' bucket (OU residual)
        for i, rel in enumerate(relationship):
            if rel == "cointegrated":
                resid = self._ou_residual(n_steps + 1, sigma=base[i] * 0.01)
                closes[i] = base[i] / btc_close[0] * btc_close + resid
                closes[i] = np.clip(closes[i], base[i] * 0.2, None)

        return pd.DataFrame(closes.T, columns=symbols)

    def _ou_residual(self, n: int, sigma: float, kappa: float = 8.0) -> np.ndarray:
        dt = 1.0 / _YEAR_DAYS
        x = np.empty(n)
        x[0] = 0.0
        shocks = self.rng.standard_normal(n - 1)
        for t in range(n - 1):
            x[t + 1] = x[t] - kappa * x[t] * dt + sigma * math.sqrt(dt) * shocks[t]
        return x

    @staticmethod
    def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        n = len(close)
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev
        rng_amp = np.abs(rng.normal(0.0, 0.01, n))
        high = np.maximum(open_, close) * (1.0 + rng_amp)
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
        volume = np.abs(rng.normal(1.0, 0.3, n)) * 1_000.0
        return open_, high, low, close, volume

    # ----------------------------------------------------------------- options
    def _drifting_params(self, t: int, n_steps: int, v0: float) -> HestonParameters:
        phase = 2 * math.pi * t / max(n_steps, 1)
        kappa = 3.0 * (1.0 + 0.10 * math.sin(phase))
        theta = 0.16 * (1.0 + 0.15 * math.sin(phase + 0.7))
        eps = 0.6 * (1.0 + 0.10 * math.cos(phase))
        rho = -0.6 + 0.05 * math.sin(phase + 1.3)
        return HestonParameters(v0=float(max(v0, 1e-3)), kappa=kappa, theta=theta, eps=eps, rho=rho,
                                flat_yield=0.0)

    def _option_rows(
        self,
        sample_idx: int,
        ts_ns: int,
        spot: float,
        v0: float,
        t_index: int,
        n_steps: int,
        expiry_ns: int,
        strikes: np.ndarray,
    ) -> List[dict]:
        ttm = (expiry_ns - ts_ns) / (_NS_PER_DAY * _YEAR_DAYS)
        if ttm <= 0:
            return []
        params = self._drifting_params(t_index, n_steps, v0)
        rows: List[dict] = []
        for is_call, instr in ((True, INSTR_CALL), (False, INSTR_PUT)):
            calls = np.full(len(strikes), is_call)
            premia_usd = heston_premiums(spot, strikes, np.full(len(strikes), ttm), calls, params)
            coin_px = premia_usd / spot  # quote in units of the underlying (Deribit convention)
            for k, px in zip(strikes, coin_px):
                if not np.isfinite(px) or px <= 0:
                    continue
                spread = 0.02 + 0.01 * self.rng.random()
                bid = px * (1.0 - spread / 2)
                ask = px * (1.0 + spread / 2)
                liq = float(np.abs(self.rng.normal(50.0, 15.0)))
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "timestamp": ts_ns,
                        "instrument_type": instr,
                        "strike": float(k),
                        "expiry_ts": expiry_ns,
                        "time_to_maturity": float(ttm),
                        "price": float(px),
                        "best_bid_price": float(bid),
                        "best_ask_price": float(ask),
                        "bid_amount_total": liq,
                        "ask_amount_total": float(np.abs(self.rng.normal(50.0, 15.0))),
                        "bid_vwap": float(bid),
                        "ask_vwap": float(ask),
                    }
                )
        return rows

    # -------------------------------------------------------------------- build
    def load(self) -> MarketDataBundle:
        symbols = self._universe()
        n_steps = int(self.n_steps)
        dt = 1.0 / _YEAR_DAYS

        btc_close, btc_var = self._simulate_btc(n_steps, dt)
        closes = self._simulate_universe(symbols, btc_close)

        start = pd.Timestamp("2024-01-02 00:00:00")
        timestamps = pd.date_range(start, periods=n_steps + 1, freq="D")
        ts_ns = timestamps.astype("int64").to_numpy()

        # ---- spot bars (long OHLCV) + wide close matrix
        bar_frames = []
        ohlc_rng = np.random.default_rng(self.seed + 7)
        for sym in symbols:
            o, h, l, c, vol = self._ohlc_from_close(closes[sym].to_numpy(), ohlc_rng)
            bar_frames.append(
                pd.DataFrame(
                    {"timestamp": timestamps, "symbol": sym, "open": o, "high": h,
                     "low": l, "close": c, "volume": vol}
                )
            )
        spot_bars = pd.concat(bar_frames, ignore_index=True)
        spot_close = closes.copy()
        spot_close.index = timestamps
        spot_close.index.name = "timestamp"

        # ---- option market data for the primary symbol
        primary = self.config.primary_symbol
        spot_path = closes[primary].to_numpy()
        expiry_ns = int(ts_ns[-1] + self.config.option_expiry_days * _NS_PER_DAY)
        strikes = self._strike_grid(spot_path[0])

        records: List[dict] = []
        for t in range(n_steps + 1):
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
                self._option_rows(t, int(ts_ns[t]), spot, float(btc_var[t]), t, n_steps, expiry_ns, strikes)
            )

        option_market_data = pd.DataFrame.from_records(records, columns=MARKET_DATA_COLUMNS)
        option_market_data["timestamp"] = pd.to_datetime(option_market_data["timestamp"])

        meta = {
            "provider": self.name,
            "seed": self.seed,
            "n_samples": n_steps + 1,
            "expiry_ts": expiry_ns,
            "strikes": strikes.tolist(),
            "universe_size": len(symbols),
        }
        return MarketDataBundle(
            spot_bars=spot_bars,
            spot_close=spot_close,
            option_market_data=option_market_data,
            symbols=symbols,
            primary_symbol=primary,
            meta=meta,
        ).validate()

    def _strike_grid(self, spot0: float) -> np.ndarray:
        n = self.config.n_strikes_per_expiry
        width = self.config.strike_width_pct
        lo, hi = spot0 * (1 - width), spot0 * (1 + width)
        raw = np.linspace(lo, hi, n)
        step = 10 ** max(0, int(math.floor(math.log10(spot0))) - 2)
        return np.unique(np.round(raw / step) * step)
