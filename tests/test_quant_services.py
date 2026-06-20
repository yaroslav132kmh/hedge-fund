"""Unit tests for the critical computational services."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptohedge.services import optimization as opt
from cryptohedge.services import correlation as corr
from cryptohedge.services import drift
from cryptohedge.services import metrics as mx
from cryptohedge.services.stops import average_true_range
from cryptohedge.services.volatility import estimate_volatility, log_returns, size_primary_hedge
from cryptohedge.services.walkforward import walk_forward_splits


# ------------------------------------------------------------------- volatility
def test_log_returns_length():
    p = np.array([100.0, 110.0, 121.0])
    r = log_returns(p)
    assert len(r) == 2
    assert np.allclose(r, np.log(p[1:] / p[:-1]))


def test_estimate_volatility_known_series():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, 500)
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))
    vol = estimate_volatility(prices, window=30, trading_days=365)
    assert 0.015 < vol.daily_vol < 0.025          # recovers ~2% daily vol
    assert vol.ci_low <= vol.daily_vol <= vol.ci_high
    assert vol.annualized_vol == pytest.approx(vol.daily_vol * np.sqrt(365), rel=1e-6)


def test_size_primary_hedge_monotonic():
    rng = np.random.default_rng(1)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.04, 200)))
    vol = estimate_volatility(prices)
    tight = size_primary_hedge(1e7, prices[-1], vol, risk_budget_pct=0.01)
    loose = size_primary_hedge(1e7, prices[-1], vol, risk_budget_pct=0.04)
    assert 0.0 <= loose.hedge_ratio <= tight.hedge_ratio <= 1.0
    assert tight.quantity_to_hedge >= loose.quantity_to_hedge


# ----------------------------------------------------------------------- metrics
def test_metrics_basic_properties():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 365)
    m = mx.compute_metrics(r, periods_per_year=365)
    assert -1.0 < m.roi
    assert m.volatility > 0
    assert 0.0 <= m.win_rate <= 1.0
    assert m.max_drawdown <= 0.0
    assert np.isfinite(m.sharpe)


def test_max_drawdown_simple():
    equity = np.array([100, 120, 90, 110.0])
    dd = mx.max_drawdown(equity)
    assert dd == pytest.approx((90 - 120) / 120)


def test_var_cvar_ordering():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.02, 5000)
    var = mx.value_at_risk(r, 0.95)
    cvar = mx.conditional_var(r, 0.95)
    assert cvar >= var > 0


def test_beta_against_self_is_one():
    rng = np.random.default_rng(4)
    b = rng.normal(0, 0.01, 300)
    m = mx.compute_metrics(b.copy(), benchmark=b.copy(), periods_per_year=365)
    assert m.beta == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ optimization
def _spd_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 0.01


def test_optimizers_simplex_constraints():
    Sigma = _spd_matrix(5)
    mu = np.linspace(0.01, 0.05, 5)
    scen = np.random.default_rng(7).multivariate_normal(mu, Sigma, size=400)
    for method in ["mean_variance", "min_variance", "risk_parity", "max_diversification", "cvar"]:
        w = opt.optimize(method, mu, Sigma, scenarios=scen, long_only=True, max_weight=1.0)
        assert w.shape == (5,)
        assert np.all(w >= -1e-6)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)


def test_min_variance_beats_equal_weight():
    Sigma = _spd_matrix(6, seed=11)
    w = opt.min_variance(Sigma)
    ew = np.ones(6) / 6
    assert w @ Sigma @ w <= ew @ Sigma @ ew + 1e-9


def test_transaction_cost_and_turnover():
    a = np.array([0.5, 0.5])
    b = np.array([0.2, 0.8])
    assert opt.turnover(a, b) == pytest.approx(0.6)
    assert opt.transaction_cost(a, b, 0.001, 1e6) == pytest.approx(0.6 * 0.001 * 1e6)


# ----------------------------------------------------------------------- drift
def test_psi_zero_for_identical():
    rng = np.random.default_rng(5)
    x = rng.normal(size=1000)
    assert drift.population_stability_index(x, x.copy()) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)
    assert drift.population_stability_index(ref, cur) > 0.25


def test_confidence_score_weighting():
    comps = {"a": 1.0, "b": 0.0}
    w = {"a": 1.0, "b": 1.0}
    assert drift.confidence_score(comps, w) == pytest.approx(0.5)
    assert 0.0 <= drift.confidence_score({"a": 5.0}, {"a": 1.0}) <= 1.0  # clipped


# ------------------------------------------------------------------- walkforward
def test_walk_forward_no_leakage():
    folds = walk_forward_splits(n=40, train_window=10, test_window=5, step=5, purge=1)
    assert len(folds) > 0
    for f in folds:
        assert f.train.max() < f.test.min()           # no look-ahead
        assert f.test.min() - f.train.max() >= 1       # purge gap respected
        assert len(f.test) == 5


def test_walk_forward_expanding_grows():
    folds = walk_forward_splits(n=50, train_window=10, test_window=5, step=5, expanding=True)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] <= sizes[-1]


# -------------------------------------------------------------------------- ATR
def test_atr_positive_and_padded():
    rng = np.random.default_rng(8)
    close = 100 + np.cumsum(rng.normal(0, 1, 100))
    high = close + np.abs(rng.normal(0, 1, 100))
    low = close - np.abs(rng.normal(0, 1, 100))
    atr = average_true_range(high, low, close, window=14)
    assert len(atr) == 100
    assert np.isnan(atr[:13]).all()
    assert np.all(atr[14:] > 0)


# ------------------------------------------------------------------ correlation
def test_static_correlations_self_excluded():
    rng = np.random.default_rng(9)
    base = rng.normal(0, 1, 200)
    df = pd.DataFrame({
        "BTCUSDT": base,
        "POS": base * 0.9 + rng.normal(0, 0.1, 200),
        "NEG": -base * 0.8 + rng.normal(0, 0.1, 200),
    })
    res = corr.static_correlations(df, "BTCUSDT")
    assert "BTCUSDT" not in res.index
    assert res.loc["POS", "pearson"] > 0.7
    assert res.loc["NEG", "pearson"] < -0.6


def test_classify_relationship():
    assert corr.classify_relationship(0.8, 0.5, -0.3, 0.1) == "positive"
    assert corr.classify_relationship(-0.5, 0.5, -0.3, 0.1) == "inverse"
    assert corr.classify_relationship(0.02, 0.5, -0.3, 0.1) == "neutral"
