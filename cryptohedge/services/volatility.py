"""Volatility estimation and primary-instrument hedge sizing.

Computes daily volatility, volatility-of-volatility and a confidence interval for
the variance (chi-square based), then converts the investor's capital and risk
budget into the BTC notional that must be hedged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import stats

from cryptohedge.domain.market import VolatilityEstimate


def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def estimate_volatility(
    prices: np.ndarray,
    window: int = 30,
    vov_window: int = 30,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    trading_days: int = 365,
) -> VolatilityEstimate:
    """Estimate daily vol, vol-of-vol and a chi-square CI for the daily vol."""
    rets = log_returns(prices)
    if len(rets) < 2:
        raise ValueError("Need at least 3 prices to estimate volatility")

    daily_vol = float(np.std(rets, ddof=1))
    annualized = daily_vol * np.sqrt(trading_days)

    # rolling realised vol series -> vol of vol
    n = len(rets)
    win = min(window, max(2, n // 2))
    rolling = np.array([np.std(rets[max(0, i - win):i], ddof=1) for i in range(win, n + 1)])
    vol_of_vol = float(np.std(rolling, ddof=1)) if len(rolling) > 1 else 0.0

    # chi-square CI for the standard deviation
    dof = len(rets) - 1
    alpha = 1.0 - confidence_level
    chi2_low = stats.chi2.ppf(alpha / 2, dof)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, dof)
    var = daily_vol**2
    ci_low = float(np.sqrt(dof * var / chi2_high))
    ci_high = float(np.sqrt(dof * var / chi2_low))

    scale = np.sqrt(horizon_days)
    return VolatilityEstimate(
        daily_vol=daily_vol * scale,
        annualized_vol=annualized,
        vol_of_vol=vol_of_vol,
        ci_low=ci_low * scale,
        ci_high=ci_high * scale,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
    )


@dataclass(frozen=True)
class HedgeSizing:
    capital_usd: float
    spot: float
    daily_vol_used: float
    confidence_z: float
    unhedged_var_pct: float
    target_var_pct: float
    hedge_ratio: float
    notional_to_hedge_usd: float
    quantity_to_hedge: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "capital_usd": self.capital_usd,
            "spot": self.spot,
            "daily_vol_used": self.daily_vol_used,
            "confidence_z": self.confidence_z,
            "unhedged_var_pct": self.unhedged_var_pct,
            "target_var_pct": self.target_var_pct,
            "hedge_ratio": self.hedge_ratio,
            "notional_to_hedge_usd": self.notional_to_hedge_usd,
            "quantity_to_hedge": self.quantity_to_hedge,
        }


def size_primary_hedge(
    capital_usd: float,
    spot: float,
    vol: VolatilityEstimate,
    risk_budget_pct: float,
    confidence_level: float = 0.95,
    use_ci_high: bool = True,
) -> HedgeSizing:
    """Determine how much BTC notional must be hedged to respect the risk budget.

    Assumes the capital is exposed to BTC. The unhedged one-day parametric VaR is
    ``z * sigma``; if it exceeds the risk budget the excess fraction of capital is
    hedged.
    """
    z = float(stats.norm.ppf(confidence_level))
    sigma = vol.ci_high if use_ci_high else vol.daily_vol
    unhedged = z * sigma
    target = float(risk_budget_pct)
    hedge_ratio = float(np.clip(1.0 - target / unhedged, 0.0, 1.0)) if unhedged > 0 else 0.0
    notional = hedge_ratio * capital_usd
    return HedgeSizing(
        capital_usd=capital_usd,
        spot=spot,
        daily_vol_used=sigma,
        confidence_z=z,
        unhedged_var_pct=unhedged,
        target_var_pct=target,
        hedge_ratio=hedge_ratio,
        notional_to_hedge_usd=notional,
        quantity_to_hedge=notional / spot if spot > 0 else 0.0,
    )
