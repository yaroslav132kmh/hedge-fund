"""Portfolio performance and risk metrics.

A single :func:`compute_metrics` returns ROI, Sharpe, Sortino, Calmar, Maximum
Drawdown, Profit Factor, Win Rate, VaR, CVaR, Expected Shortfall, Beta, Alpha and
Information Ratio. All ratios are annualised with the configured period count.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    roi: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    var: float
    cvar: float
    expected_shortfall: float
    beta: float
    alpha: float
    information_ratio: float
    volatility: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return np.array([])
    base = np.where(np.abs(equity[:-1]) < 1e-12, np.nan, equity[:-1])
    return np.nan_to_num((equity[1:] - equity[:-1]) / base)


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.where(np.abs(running_max) < 1e-12, np.nan, running_max)
    return float(np.nanmin(drawdown)) if len(drawdown) else 0.0


def value_at_risk(returns: np.ndarray, confidence: float = 0.95, method: str = "historical") -> float:
    """One-period VaR as a positive loss fraction."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    alpha = 1.0 - confidence
    if method == "gaussian":
        from scipy import stats

        return float(-(np.mean(r) + stats.norm.ppf(alpha) * np.std(r, ddof=1)))
    if method == "cornish_fisher":
        from scipy import stats

        mu, sigma = np.mean(r), np.std(r, ddof=1)
        s = stats.skew(r)
        k = stats.kurtosis(r)
        z = stats.norm.ppf(alpha)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(-(mu + z_cf * sigma))
    return float(-np.quantile(r, alpha))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall (historical)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    alpha = 1.0 - confidence
    threshold = np.quantile(r, alpha)
    tail = r[r <= threshold]
    return float(-np.mean(tail)) if len(tail) else float(-threshold)


def compute_metrics(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
    var_confidence: float = 0.95,
    var_method: str = "historical",
    equity: Optional[np.ndarray] = None,
) -> PerformanceMetrics:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return PerformanceMetrics(*([0.0] * 15))

    rf_per = risk_free / periods_per_year
    excess = r - rf_per
    mean, std = float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    ann_factor = np.sqrt(periods_per_year)

    total_return = float(np.prod(1.0 + r) - 1.0)
    n_years = len(r) / periods_per_year
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > -1 else 0.0
    volatility = std * ann_factor

    sharpe = float(np.mean(excess) / std * ann_factor) if std > 0 else 0.0
    downside = r[r < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(np.mean(excess) / dstd * ann_factor) if dstd > 0 else 0.0

    eq = np.asarray(equity, dtype=float) if equity is not None else np.cumprod(1.0 + r)
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    profit_factor = float(gains / losses) if losses > 1e-12 else float("inf") if gains > 0 else 0.0
    win_rate = float(np.mean(r > 0))

    var = value_at_risk(r, var_confidence, var_method)
    cvar = conditional_var(r, var_confidence)

    beta = alpha = information_ratio = 0.0
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        m = min(len(r), len(b))
        rr, bb = r[-m:], b[-m:]
        var_b = float(np.var(bb, ddof=1)) if m > 1 else 0.0
        if var_b > 0:
            beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b)
            alpha = float((np.mean(rr) - beta * np.mean(bb)) * periods_per_year)
        active = rr - bb
        astd = float(np.std(active, ddof=1)) if m > 1 else 0.0
        information_ratio = float(np.mean(active) / astd * ann_factor) if astd > 0 else 0.0

    return PerformanceMetrics(
        roi=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        var=var,
        cvar=cvar,
        expected_shortfall=cvar,
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
        volatility=volatility,
    )
