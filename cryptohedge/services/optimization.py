"""Portfolio construction / rebalancing.

Implements five optimisers - Mean-Variance, Risk Parity, Minimum Variance,
Maximum Diversification and CVaR (Rockafellar-Uryasev LP) - all honouring
long-only / max-weight bounds and an optional turnover budget that captures the
cost and frequency of rebalancing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linprog, minimize


def _bounds(n: int, long_only: bool, max_weight: float):
    lo = 0.0 if long_only else -max_weight
    return [(lo, max_weight)] * n


def _normalize(w: np.ndarray) -> np.ndarray:
    s = np.sum(w)
    return w / s if abs(s) > 1e-12 else np.ones_like(w) / len(w)


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _turnover_constraint(w_prev: Optional[np.ndarray], max_turnover: float):
    if w_prev is None:
        return []
    return [{"type": "ineq", "fun": lambda w: max_turnover - np.sum(np.abs(w - w_prev))}]


def mean_variance(mu, Sigma, risk_aversion, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    n = len(mu)
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)

    def neg_util(w):
        return -(w @ mu - 0.5 * risk_aversion * w @ Sigma @ w)

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])
    res = minimize(neg_util, np.ones(n) / n, method="SLSQP",
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return _normalize(res.x) if res.success else np.ones(n) / n


def min_variance(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    n = Sigma.shape[0]
    Sigma = np.asarray(Sigma, float)

    def variance(w):
        return w @ Sigma @ w

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])
    res = minimize(variance, np.ones(n) / n, method="SLSQP",
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return _normalize(res.x) if res.success else np.ones(n) / n


def risk_parity(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    """Equalise marginal risk contributions across assets."""
    n = Sigma.shape[0]
    Sigma = np.asarray(Sigma, float)
    target = np.ones(n) / n

    def objective(w):
        port_var = w @ Sigma @ w
        if port_var <= 0:
            return 1e6
        mrc = Sigma @ w
        rc = w * mrc / np.sqrt(port_var)
        rc = rc / np.sum(rc)
        return np.sum((rc - target) ** 2)

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])
    res = minimize(objective, target, method="SLSQP",
                   bounds=_bounds(n, max(long_only, True), max_weight), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return _normalize(res.x) if res.success else target


def max_diversification(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    """Maximise the diversification ratio (weighted avg vol / portfolio vol)."""
    n = Sigma.shape[0]
    Sigma = np.asarray(Sigma, float)
    sigma = np.sqrt(np.diag(Sigma))

    def neg_dr(w):
        pv = np.sqrt(w @ Sigma @ w)
        return -(w @ sigma) / pv if pv > 0 else 1e6

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])
    res = minimize(neg_dr, np.ones(n) / n, method="SLSQP",
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return _normalize(res.x) if res.success else np.ones(n) / n


def cvar_optimization(scenarios, alpha=0.95, long_only=True, max_weight=1.0, target_return=None, mu=None):
    """Minimise portfolio CVaR via the Rockafellar-Uryasev linear program.

    Variables: ``[w (n), var (1), u (S)]`` where ``u_s >= -scenario_s.w - var``.
    Objective: ``var + 1/((1-alpha) S) * sum(u)``.
    """
    R = np.asarray(scenarios, float)  # (S, n) scenario returns
    S, n = R.shape
    nvars = n + 1 + S

    c = np.zeros(nvars)
    c[n] = 1.0
    c[n + 1:] = 1.0 / ((1.0 - alpha) * S)

    # u_s >= -R_s . w - var  ->  -R_s.w - var - u_s <= 0
    A_ub = np.zeros((S, nvars))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    A_ub[np.arange(S), n + 1 + np.arange(S)] = -1.0
    b_ub = np.zeros(S)

    A_eq = np.zeros((1, nvars))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    if target_return is not None and mu is not None:
        add = np.zeros((1, nvars))
        add[0, :n] = -np.asarray(mu, float)
        A_ub = np.vstack([A_ub, add])
        b_ub = np.concatenate([b_ub, [-target_return]])

    lo = 0.0 if long_only else -max_weight
    bounds = [(lo, max_weight)] * n + [(None, None)] + [(0, None)] * S
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return _normalize(res.x[:n])
    return np.ones(n) / n


def optimize(
    method: str,
    mu: Optional[np.ndarray],
    Sigma: np.ndarray,
    scenarios: Optional[np.ndarray] = None,
    risk_aversion: float = 5.0,
    cvar_alpha: float = 0.95,
    long_only: bool = True,
    max_weight: float = 1.0,
    w_prev: Optional[np.ndarray] = None,
    max_turnover: Optional[float] = None,
) -> np.ndarray:
    """Dispatch to the requested optimiser (Strategy pattern)."""
    method = method.lower()
    if method == "mean_variance":
        return mean_variance(mu, Sigma, risk_aversion, long_only, max_weight, w_prev, max_turnover)
    if method == "min_variance":
        return min_variance(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "risk_parity":
        return risk_parity(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "max_diversification":
        return max_diversification(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "cvar":
        if scenarios is None:
            raise ValueError("CVaR optimisation requires return scenarios")
        return cvar_optimization(scenarios, cvar_alpha, long_only, max_weight, mu=mu)
    raise ValueError(f"Unknown optimisation method: {method}")


def turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w_new) - np.asarray(w_old))))


def transaction_cost(w_new: np.ndarray, w_old: np.ndarray, fee_pct: float, capital: float) -> float:
    return float(turnover(w_new, w_old) * fee_pct * capital)
