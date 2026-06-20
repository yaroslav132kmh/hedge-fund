"""Heston calibration via filtering + MLE, with Black-Scholes and SABR benchmarks.

* :func:`calibrate_mle` estimates the Heston parameters from the spot time series
  using an EWMA variance *filter* combined with maximum-likelihood estimation of
  the Euler-discretised dynamics (suited to time series, no look-ahead).
* :func:`sabr_calibrate` / :func:`black_scholes_benchmark` provide the two
  benchmark models required for model-risk comparison.
* :func:`parameter_stability` monitors the temporal stability of the calibrated
  parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from cryptohedge.domain.market import HestonParameters
from cryptohedge.services.heston_pricing import bs_implied_vol


# ----------------------------------------------------------------- variance filter
def ewma_variance(returns: np.ndarray, lam: float = 0.94, trading_days: int = 365) -> np.ndarray:
    """Filtered annualised instantaneous variance from returns (RiskMetrics EWMA)."""
    r = np.nan_to_num(np.asarray(returns, float))
    var = np.empty(len(r))
    var[0] = np.var(r) if np.var(r) > 0 else 1e-6
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var * trading_days


# --------------------------------------------------------------------------- MLE
def calibrate_mle(
    prices: np.ndarray,
    dt: float = 1.0 / 365.0,
    flat_yield: float = 0.0,
    init: Optional[Sequence[float]] = None,
    trading_days: int = 365,
) -> HestonParameters:
    """Maximum-likelihood Heston calibration on a price time series.

    The latent variance is filtered with EWMA; the (return, variance-increment)
    pairs are then modelled as conditionally bivariate-normal under the Euler
    discretisation, and the parameters maximise the joint log-likelihood.
    """
    prices = np.asarray(prices, float)
    rets = np.diff(np.log(prices))
    v = ewma_variance(rets, trading_days=trading_days)
    v = np.clip(v, 1e-6, None)
    dv = np.diff(v)
    r = rets[1:]
    vt = v[:-1]

    if init is None:
        init = [2.0, float(np.mean(v)), 0.5, -0.5, float(np.mean(r) / dt)]

    def neg_ll(p):
        kappa, theta, eps, rho, mu = p
        if kappa <= 0 or theta <= 0 or eps <= 0 or abs(rho) >= 0.999:
            return 1e8
        m_r = (mu - 0.5 * vt) * dt
        m_v = kappa * (theta - vt) * dt
        var_r = vt * dt
        var_v = eps**2 * vt * dt
        cov = rho * eps * vt * dt
        det = var_r * var_v - cov**2
        det = np.where(det <= 0, 1e-12, det)
        dr = r - m_r
        dvv = dv - m_v
        quad = (var_v * dr**2 - 2 * cov * dr * dvv + var_r * dvv**2) / det
        ll = -0.5 * (np.log(det) + quad + 2 * np.log(2 * np.pi))
        return float(-np.sum(ll))

    res = minimize(neg_ll, init, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    kappa, theta, eps, rho, _mu = res.x
    v0 = float(v[-1])
    return HestonParameters(
        v0=v0,
        kappa=float(abs(kappa)),
        theta=float(abs(theta)),
        eps=float(abs(eps)),
        rho=float(np.clip(rho, -0.999, 0.999)),
        flat_yield=flat_yield,
        calibration_error=float(res.fun),
        feller_satisfied=bool(2 * abs(kappa) * abs(theta) - eps**2 >= 0),
    )


# -------------------------------------------------------------------------- SABR
def sabr_lognormal_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Hagan (2002) lognormal SABR implied volatility approximation."""
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return float("nan")
    if abs(F - K) < 1e-12:
        term = (
            ((1 - beta) ** 2 / 24) * alpha**2 / F ** (2 - 2 * beta)
            + 0.25 * rho * beta * nu * alpha / F ** (1 - beta)
            + (2 - 3 * rho**2) / 24 * nu**2
        )
        return float(alpha / F ** (1 - beta) * (1 + term * T))
    logFK = np.log(F / K)
    fk_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * logFK
    xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    denom = fk_beta * (1 + ((1 - beta) ** 2 / 24) * logFK**2 + ((1 - beta) ** 4 / 1920) * logFK**4)
    term = (
        ((1 - beta) ** 2 / 24) * alpha**2 / fk_beta**2
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2 - 3 * rho**2) / 24 * nu**2
    )
    factor = z / xz if abs(xz) > 1e-12 else 1.0
    return float((alpha / denom) * factor * (1 + term * T))


@dataclass(frozen=True)
class SABRParameters:
    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float

    def to_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu, "rmse": self.rmse}


def sabr_calibrate(
    forward: float, strikes: np.ndarray, ttm: float, market_iv: np.ndarray, beta: float = 0.5
) -> SABRParameters:
    """Calibrate SABR (alpha, rho, nu) to a market implied-vol smile (beta fixed)."""
    strikes = np.asarray(strikes, float)
    market_iv = np.asarray(market_iv, float)
    mask = np.isfinite(market_iv) & (market_iv > 0)
    strikes, market_iv = strikes[mask], market_iv[mask]
    if len(strikes) < 3:
        return SABRParameters(np.nan, beta, np.nan, np.nan, np.nan)

    atm_iv = market_iv[np.argmin(np.abs(strikes - forward))]

    def loss(p):
        alpha, rho, nu = p
        if alpha <= 0 or abs(rho) >= 0.999 or nu < 0:
            return 1e6
        model = np.array([sabr_lognormal_vol(forward, k, ttm, alpha, beta, rho, nu) for k in strikes])
        return float(np.nanmean((model - market_iv) ** 2))

    res = minimize(loss, [atm_iv * forward ** (1 - beta), -0.3, 0.5], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
    alpha, rho, nu = res.x
    return SABRParameters(float(alpha), beta, float(np.clip(rho, -0.999, 0.999)), float(abs(nu)), float(np.sqrt(res.fun)))


# --------------------------------------------------------------- BS benchmark
def black_scholes_benchmark(
    spot: float, strikes: np.ndarray, ttm: float, market_prices: np.ndarray, is_call: np.ndarray, flat_yield: float = 0.0
) -> Dict[str, float]:
    """Flat-vol Black-Scholes benchmark: ATM IV and smile RMSE vs market."""
    strikes = np.asarray(strikes, float)
    ivs = np.array([
        bs_implied_vol(spot, k, ttm, flat_yield, p, bool(c))
        for k, p, c in zip(strikes, market_prices, is_call)
    ])
    valid = np.isfinite(ivs)
    if valid.sum() == 0:
        return {"atm_iv": float("nan"), "flat_vol": float("nan"), "iv_rmse": float("nan")}
    atm_iv = float(ivs[valid][np.argmin(np.abs(strikes[valid] - spot))])
    flat_vol = float(np.nanmean(ivs[valid]))
    rmse = float(np.sqrt(np.nanmean((ivs[valid] - flat_vol) ** 2)))
    return {"atm_iv": atm_iv, "flat_vol": flat_vol, "iv_rmse": rmse}


# ----------------------------------------------------------------- stability
def parameter_stability(history: List[HestonParameters], max_rel_change: float = 0.5) -> Dict[str, object]:
    """Quantify temporal stability of calibrated Heston parameters."""
    if len(history) < 2:
        return {"stable": True, "max_rel_change": 0.0, "per_param": {}}
    keys = ["v0", "kappa", "theta", "eps", "rho"]
    arr = np.array([[getattr(h, k) for k in keys] for h in history], float)
    rel = np.abs(np.diff(arr, axis=0)) / (np.abs(arr[:-1]) + 1e-8)
    per_param = {k: float(np.nanmean(rel[:, i])) for i, k in enumerate(keys)}
    max_change = float(np.nanmax(rel))
    return {
        "stable": bool(max_change < max_rel_change),
        "max_rel_change": max_change,
        "mean_rel_change": float(np.nanmean(rel)),
        "per_param": per_param,
    }
