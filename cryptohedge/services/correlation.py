"""Dependence analysis and multi-criteria hedge-instrument ranking.

Implements linear (Pearson), rank (Spearman, Kendall) correlations, dynamic
DCC-GARCH correlation and cointegration tests (Engle-Granger & Johansen), then
ranks candidate instruments for hedging the primary asset by a weighted blend of
correlation strength, relationship stability, liquidity, hedging cost and risk
reduction.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cryptohedge.core.config import CorrelationConfig, RankingWeights
from cryptohedge.domain.market import InstrumentRanking

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- static corr
def static_correlations(returns: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Pearson, Spearman and Kendall of every column against ``primary``."""
    base = returns[primary].to_numpy()
    rows = []
    for col in returns.columns:
        if col == primary:
            continue
        other = returns[col].to_numpy()
        mask = np.isfinite(base) & np.isfinite(other)
        if mask.sum() < 5:
            rows.append({"symbol": col, "pearson": np.nan, "spearman": np.nan, "kendall": np.nan})
            continue
        x, y = base[mask], other[mask]
        rows.append(
            {
                "symbol": col,
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(stats.spearmanr(x, y).correlation),
                "kendall": float(stats.kendalltau(x, y).correlation),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def rolling_stability(returns: pd.DataFrame, primary: str, window: int = 30) -> pd.Series:
    """Stability of the relationship = 1 / (1 + std of rolling Pearson corr)."""
    base = returns[primary]
    out = {}
    for col in returns.columns:
        if col == primary:
            continue
        roll = base.rolling(window).corr(returns[col]).dropna()
        out[col] = float(1.0 / (1.0 + np.std(roll))) if len(roll) > 1 else 0.0
    return pd.Series(out, name="stability")


# ------------------------------------------------------------------ DCC-GARCH
def _garch_standardized(series: np.ndarray) -> Optional[np.ndarray]:
    """Standardised residuals from a GARCH(1,1) fit (returns in % for stability)."""
    try:
        from arch.univariate import arch_model

        r = series[np.isfinite(series)] * 100.0
        if len(r) < 20:
            return None
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        z = res.resid / res.conditional_volatility
        return np.asarray(z, dtype=float)
    except Exception:
        return None


def _ewma_standardized(series: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA standardisation fallback when GARCH is unavailable."""
    r = np.nan_to_num(series)
    var = np.empty_like(r)
    var[0] = np.var(r) if np.var(r) > 0 else 1e-8
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return r / np.sqrt(np.maximum(var, 1e-12))


def dcc_garch_correlations(
    returns: pd.DataFrame,
    primary: str,
    candidates: Sequence[str],
    a: float = 0.02,
    b: float = 0.95,
    estimate: bool = True,
    max_iter: int = 50,
) -> Dict[str, float]:
    """Mean dynamic conditional correlation between ``primary`` and each candidate.

    Two-step DCC: (1) GARCH(1,1) standardisation per series; (2) DCC(1,1)
    recursion. ``(a, b)`` are pooled-QML estimated across all candidate pairs when
    ``estimate`` is True, otherwise the configured constants are used.
    """
    series = {primary: returns[primary].to_numpy()}
    for c in candidates:
        series[c] = returns[c].to_numpy()

    std: Dict[str, np.ndarray] = {}
    for sym, arr in series.items():
        z = _garch_standardized(arr)
        std[sym] = z if z is not None else _ewma_standardized(arr)

    zp = std[primary]
    pairs = [(zp, std[c]) for c in candidates if len(std[c]) == len(zp)]

    if estimate and pairs:
        a, b = _estimate_dcc_ab(pairs, a, b, max_iter)

    result: Dict[str, float] = {}
    for c in candidates:
        zc = std[c]
        if len(zc) != len(zp):
            result[c] = float("nan")
            continue
        corr_path = _dcc_pair_corr(zp, zc, a, b)
        result[c] = float(np.nanmean(corr_path))
    return result


def _dcc_pair_corr(z1: np.ndarray, z2: np.ndarray, a: float, b: float) -> np.ndarray:
    Z = np.column_stack([z1, z2])
    Qbar = np.cov(Z, rowvar=False)
    T = len(z1)
    Q = Qbar.copy()
    out = np.empty(T)
    for t in range(T):
        if t > 0:
            zz = np.outer(Z[t - 1], Z[t - 1])
            Q = (1 - a - b) * Qbar + a * zz + b * Q
        d = np.sqrt(np.diag(Q))
        out[t] = Q[0, 1] / (d[0] * d[1]) if d[0] > 0 and d[1] > 0 else np.nan
    return out


def _estimate_dcc_ab(pairs, a0: float, b0: float, max_iter: int):
    from scipy.optimize import minimize

    def neg_ll(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e6
        total = 0.0
        for z1, z2 in pairs:
            Z = np.column_stack([z1, z2])
            Qbar = np.cov(Z, rowvar=False)
            Q = Qbar.copy()
            for t in range(len(z1)):
                if t > 0:
                    zz = np.outer(Z[t - 1], Z[t - 1])
                    Q = (1 - a - b) * Qbar + a * zz + b * Q
                d = np.sqrt(np.diag(Q))
                R = Q / np.outer(d, d)
                detR = R[0, 0] * R[1, 1] - R[0, 1] ** 2
                if detR <= 0:
                    return 1e6
                zt = Z[t]
                quad = (zt @ np.linalg.inv(R) @ zt)
                total += 0.5 * (np.log(detR) + quad)
        return total

    try:
        res = minimize(neg_ll, x0=[a0, b0], method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2})
        a, b = float(res.x[0]), float(res.x[1])
        if a < 0 or b < 0 or a + b >= 0.999:
            return a0, b0
        return a, b
    except Exception:
        return a0, b0


# ---------------------------------------------------------------- cointegration
def cointegration(
    prices: pd.DataFrame,
    primary: str,
    candidates: Sequence[str],
    method: str = "both",
    pvalue: float = 0.05,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Dict[str, bool]:
    """Test each candidate for cointegration with the primary price level."""
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    base = prices[primary].to_numpy()
    out: Dict[str, bool] = {}
    for c in candidates:
        other = prices[c].to_numpy()
        eg = jo = False
        if method in ("engle_granger", "both"):
            try:
                _, pval, _ = coint(base, other)
                eg = bool(pval < pvalue)
            except Exception:
                eg = False
        if method in ("johansen", "both"):
            try:
                jres = coint_johansen(np.column_stack([base, other]), det_order, k_ar_diff)
                jo = bool(jres.lr1[0] > jres.cvt[0, 1])  # trace stat vs 95% crit, r=0
            except Exception:
                jo = False
        out[c] = (eg or jo) if method == "both" else (eg if method == "engle_granger" else jo)
    return out


# ---------------------------------------------------------------------- ranking
def classify_relationship(pearson: float, pos: float, neg: float, zero_band: float) -> str:
    if np.isnan(pearson):
        return "neutral"
    if pearson >= pos:
        return "positive"
    if pearson <= neg:
        return "inverse"
    if abs(pearson) <= zero_band:
        return "neutral"
    return "weak"


def _minmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return np.zeros_like(v)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.nan_to_num(np.ones_like(v) * 0.5)
    return np.nan_to_num((v - lo) / (hi - lo))


def rank_instruments(
    static: pd.DataFrame,
    stability: pd.Series,
    dcc: Dict[str, float],
    cointegrated: Dict[str, bool],
    liquidity: pd.Series,
    hedge_cost: pd.Series,
    config: CorrelationConfig,
    weights: RankingWeights,
) -> List[InstrumentRanking]:
    """Combine all criteria into a weighted score and rank candidates.

    A good hedge has strong (positive or inverse) and *stable* dependence, deep
    liquidity, low hedging cost and high risk-reduction potential
    (``|corr|`` is a proxy for variance reduction of the hedged book).
    """
    syms = list(static.index)
    abs_corr = static["pearson"].abs().to_numpy()
    dcc_arr = np.array([abs(dcc.get(s, np.nan)) for s in syms])
    corr_strength = np.nanmax(np.column_stack([abs_corr, dcc_arr]), axis=1)
    risk_reduction = 1.0 - np.sqrt(np.clip(1.0 - corr_strength**2, 0.0, 1.0))

    n_corr = _minmax(corr_strength)
    n_stab = _minmax(stability.reindex(syms).to_numpy())
    n_liq = _minmax(liquidity.reindex(syms).to_numpy())
    n_cost = 1.0 - _minmax(hedge_cost.reindex(syms).to_numpy())  # lower cost is better
    n_rr = _minmax(risk_reduction)
    coint_bonus = np.array([0.1 if cointegrated.get(s, False) else 0.0 for s in syms])

    score = (
        weights.correlation * n_corr
        + weights.stability * n_stab
        + weights.liquidity * n_liq
        + weights.hedge_cost * n_cost
        + weights.risk_reduction * n_rr
        + coint_bonus
    )

    rankings: List[InstrumentRanking] = []
    for i, s in enumerate(syms):
        rankings.append(
            InstrumentRanking(
                symbol=s,
                pearson=float(static.loc[s, "pearson"]),
                spearman=float(static.loc[s, "spearman"]),
                kendall=float(static.loc[s, "kendall"]),
                dcc_mean=float(dcc.get(s, np.nan)),
                cointegrated=bool(cointegrated.get(s, False)),
                stability=float(stability.get(s, 0.0)),
                liquidity=float(liquidity.get(s, 0.0)),
                hedge_cost=float(hedge_cost.get(s, 0.0)),
                risk_reduction=float(risk_reduction[i]),
                score=float(score[i]),
                relationship=classify_relationship(
                    float(static.loc[s, "pearson"]),
                    config.positive_threshold,
                    config.negative_threshold,
                    config.zero_band,
                ),
            )
        )
    rankings.sort(key=lambda r: r.score, reverse=True)
    return rankings
