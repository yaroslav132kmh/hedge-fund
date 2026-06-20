"""Adaptive stop-loss and trailing-stop logic.

The stop distance blends three risk signals - ATR (price action), parametric VaR
(tail risk) and the Heston instantaneous volatility (model risk) - and is clamped
to a configured band. The :class:`TrailingStop` ratchets the level as the trade
moves in the holder's favour and is re-calibrated on each update.
"""

from __future__ import annotations

import numpy as np

from cryptohedge.core.config import StopLossConfig
from cryptohedge.domain.decisions import StopLevel


def average_true_range(high, low, close, window: int = 14) -> np.ndarray:
    """Wilder's Average True Range (NaN-padded to the input length)."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    close = np.asarray(close, float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = np.full_like(tr, np.nan)
    if len(tr) >= window:
        atr[window - 1] = tr[:window].mean()
        for i in range(window, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


def adaptive_stop(
    reference_price: float,
    atr_value: float,
    daily_var: float,
    heston_vol: float,
    side: str,
    config: StopLossConfig,
) -> StopLevel:
    """Compute an adaptive stop level for a long/short position."""
    atr_pct = (config.atr_multiplier * atr_value / reference_price) if reference_price > 0 else 0.0
    var_pct = config.var_multiplier * float(daily_var)
    vol_pct = config.var_multiplier * float(heston_vol)  # heston daily vol = sqrt(v0/365)

    distance = max(atr_pct, var_pct, vol_pct)
    distance = float(np.clip(distance, config.min_stop_pct, config.max_stop_pct))

    if side == "long":
        stop_price = reference_price * (1.0 - distance)
    else:
        stop_price = reference_price * (1.0 + distance)

    return StopLevel(
        instrument="primary",
        side=side,  # type: ignore[arg-type]
        stop_price=float(stop_price),
        reference_price=float(reference_price),
        distance_pct=distance,
        method="atr+var+heston",
        components={"atr_pct": atr_pct, "var_pct": var_pct, "heston_vol_pct": vol_pct},
    )


class TrailingStop:
    """Stateful trailing stop that re-calibrates against live volatility."""

    def __init__(self, side: str, entry_price: float, config: StopLossConfig) -> None:
        self.side = side
        self.config = config
        self.entry_price = entry_price
        self.extreme = entry_price  # highest (long) / lowest (short) price seen
        self.stop_price = entry_price
        self.triggered = False

    def update(self, price: float, atr_value: float, daily_var: float, heston_vol: float) -> StopLevel:
        if self.side == "long":
            self.extreme = max(self.extreme, price)
        else:
            self.extreme = min(self.extreme, price)

        atr_pct = (self.config.trailing_atr_multiplier * atr_value / price) if price > 0 else 0.0
        distance = float(np.clip(max(atr_pct, self.config.var_multiplier * max(daily_var, heston_vol)),
                                 self.config.min_stop_pct, self.config.max_stop_pct))

        if self.side == "long":
            new_stop = self.extreme * (1.0 - distance)
            self.stop_price = max(self.stop_price, new_stop)
            self.triggered = price <= self.stop_price
        else:
            new_stop = self.extreme * (1.0 + distance)
            self.stop_price = min(self.stop_price, new_stop)
            self.triggered = price >= self.stop_price

        return StopLevel(
            instrument="primary",
            side=self.side,  # type: ignore[arg-type]
            stop_price=float(self.stop_price),
            reference_price=float(price),
            distance_pct=distance,
            method="trailing",
            components={"extreme": float(self.extreme), "atr_pct": atr_pct, "triggered": float(self.triggered)},
        )
