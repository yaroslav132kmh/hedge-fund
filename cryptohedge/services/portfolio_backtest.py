"""Portfolio diversification analytics and a rebalanced backtest engine.

This module turns a *one-shot* set of optimiser weights into a realistic, time
evolving portfolio: between rebalancing dates the weights drift with asset
returns, and on each rebalancing date the target weights are recomputed from a
trailing window and applied subject to transaction costs. It also exposes the
standard diversification diagnostics used to *confirm* that the resulting
portfolio is well diversified (diversification ratio, effective number of bets,
Herfindahl-Hirschman concentration).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------- diversification
def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average volatility divided by portfolio volatility (>= 1).

    A value of 1 means no diversification (a single bet); the higher the ratio,
    the larger the variance-reduction benefit of combining the assets.
    """
    w = np.asarray(weights, float)
    cov = np.asarray(cov, float)
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if port_vol <= 1e-12:
        return 1.0
    return float((w @ sigma) / port_vol)


def effective_number_of_bets(weights: np.ndarray) -> float:
    """Inverse Herfindahl index: number of *equally weighted* equivalent holdings."""
    w = np.asarray(weights, float)
    hhi = float(np.sum(w**2))
    return float(1.0 / hhi) if hhi > 0 else 0.0


def herfindahl_index(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(np.sum(w**2))


def diversification_report(weights: np.ndarray, cov: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, float)
    n = int(np.sum(w > 1e-6))
    return {
        "diversification_ratio": diversification_ratio(w, cov),
        "effective_n": effective_number_of_bets(w),
        "n_active": n,
        "max_weight": float(np.max(w)) if len(w) else 0.0,
        "hhi": herfindahl_index(w),
    }


# ------------------------------------------------------------------- backtest
@dataclass(frozen=True)
class PortfolioBacktest:
    """Result of a rebalanced portfolio backtest."""

    equity: pd.Series                # portfolio value, base 1.0
    weights_path: pd.DataFrame       # drifting weights per asset over time
    rebalance_dates: List[pd.Timestamp]
    returns: pd.Series               # net daily portfolio returns
    turnover: pd.Series              # per-day turnover at rebalances
    cum_cost: pd.Series              # cumulative transaction cost (fraction of capital)
    metrics: Dict[str, float]

    def to_summary(self) -> Dict[str, float]:
        return dict(self.metrics)


def backtest_rebalanced(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], np.ndarray],
    rebalance_days: int = 5,
    fee_pct: float = 0.0003,
    lookback: int = 30,
    periods_per_year: int = 365,
) -> PortfolioBacktest:
    """Backtest a periodically rebalanced long portfolio.

    Parameters
    ----------
    prices: wide price matrix (index = dates, columns = assets).
    weight_fn: maps a trailing *returns* window to a target weight vector
        (aligned to ``prices.columns``). Called at every rebalancing date.
    rebalance_days: rebalance cadence in days.
    fee_pct: proportional transaction cost charged on traded weight.
    lookback: trailing window (in days) used to estimate the weights.
    """
    prices = prices.dropna(axis=1, how="any")
    cols = list(prices.columns)
    n = len(cols)
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index

    if n == 0 or len(dates) < 3:
        empty = pd.Series(dtype=float)
        return PortfolioBacktest(empty, pd.DataFrame(), [], empty, empty, empty, {})

    w = np.ones(n) / n                      # start equally weighted
    first = min(lookback, len(dates) - 2)
    equity = [1.0]
    weights_path = [w.copy()]
    port_rets = [0.0]
    turnover_series = [0.0]
    cum_cost = [0.0]
    rebalance_dates: List[pd.Timestamp] = [dates[0]]
    total_cost = 0.0

    for t in range(1, len(dates)):
        r_t = rets.iloc[t].to_numpy()
        # drift weights with realised returns, then renormalise
        w_drift = w * (1.0 + r_t)
        gross = float(np.sum(w_drift))
        w_drift = w_drift / gross if gross > 1e-12 else np.ones(n) / n
        port_ret = float(np.sum(w * r_t))   # return earned over [t-1, t] with start weights

        cost = 0.0
        do_rebalance = (t >= first) and ((t - first) % max(rebalance_days, 1) == 0)
        if do_rebalance:
            train = rets.iloc[max(0, t - lookback): t]
            try:
                w_target = np.asarray(weight_fn(train), float)
                if w_target.shape[0] != n or not np.all(np.isfinite(w_target)):
                    w_target = w_drift
            except Exception:
                w_target = w_drift
            s = np.sum(w_target)
            w_target = w_target / s if s > 1e-12 else np.ones(n) / n
            trade = float(np.sum(np.abs(w_target - w_drift)))
            cost = trade * fee_pct
            total_cost += cost
            turnover_series.append(trade)
            rebalance_dates.append(dates[t])
            w = w_target
        else:
            turnover_series.append(0.0)
            w = w_drift

        equity.append(equity[-1] * (1.0 + port_ret - cost))
        port_rets.append(port_ret - cost)
        weights_path.append(w.copy())
        cum_cost.append(total_cost)

    equity_s = pd.Series(equity, index=dates, name="equity")
    weights_df = pd.DataFrame(weights_path, index=dates, columns=cols)
    rets_s = pd.Series(port_rets, index=dates, name="returns")
    turnover_s = pd.Series(turnover_series, index=dates, name="turnover")
    cost_s = pd.Series(cum_cost, index=dates, name="cum_cost")

    final_w = weights_df.iloc[-1].to_numpy()
    cov = rets.cov().to_numpy() * periods_per_year
    metrics = _equity_metrics(equity_s, rets_s, periods_per_year)
    metrics.update(diversification_report(final_w, cov))
    metrics["n_rebalances"] = int(max(0, len(rebalance_dates) - 1))
    metrics["total_cost"] = float(total_cost)
    metrics["avg_diversification_ratio"] = _avg_div_ratio(weights_df, cov)

    return PortfolioBacktest(equity_s, weights_df, rebalance_dates, rets_s,
                             turnover_s, cost_s, metrics)


def _avg_div_ratio(weights_df: pd.DataFrame, cov: np.ndarray) -> float:
    vals = [diversification_ratio(weights_df.iloc[i].to_numpy(), cov)
            for i in range(0, len(weights_df), max(1, len(weights_df) // 20))]
    return float(np.mean(vals)) if vals else 1.0


def _equity_metrics(equity: pd.Series, rets: pd.Series, ppy: int) -> Dict[str, float]:
    r = rets.to_numpy()[1:]
    if len(r) == 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_years = len(r) / ppy
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > -1 else 0.0
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ppy)) if len(r) > 1 and np.std(r, ddof=1) > 0 else 0.0
    running_max = np.maximum.accumulate(equity.to_numpy())
    dd = (equity.to_numpy() - running_max) / running_max
    mdd = float(np.min(dd)) if len(dd) else 0.0
    return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe,
            "volatility": vol, "max_drawdown": mdd}
