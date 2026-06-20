"""Portfolio Optimization Agent.

Role: construct, *validate* and rebalance an investable portfolio.

The agent (1) selects a diversified, historically profitable investable universe,
(2) builds candidate portfolios with five optimisers (Mean-Variance, Risk Parity,
Minimum Variance, Maximum Diversification, CVaR), (3) runs a periodically
rebalanced backtest of each candidate accounting for transaction costs, and
(4) selects the method that is both profitable and well diversified. It publishes
the chosen portfolio's constituents, equity curve, rebalancing path and
diversification diagnostics so the dashboard can render them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.domain.decisions import RebalanceDecision
from cryptohedge.services import optimization as opt
from cryptohedge.services import portfolio_backtest as pbt


class PortfolioOptimizationAgent(BaseAgent):
    name = "portfolio_optimization"
    consumes = [MessageType.HEDGE_DECISION]
    produces = MessageType.PORTFOLIO_READY
    checkpoint_keys = [
        "optimization_results", "rebalance_decision", "opt_weights",
        "portfolio_universe", "portfolio_constituents", "portfolio_equity",
        "portfolio_weights_path", "portfolio_rebalances", "portfolio_costs",
        "diversification", "method_comparison",
    ]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        ocfg = context.config.optimization
        ppy = context.config.horizons.trading_days_per_year
        fee = context.config.investment.transaction_fee_pct

        returns: pd.DataFrame = context.require("returns")
        spot_close: pd.DataFrame = context.require("spot_close")
        primary: str = context.require("primary_symbol")

        # ---- 1. select a diversified, historically profitable investable universe
        universe = self._select_universe(returns, ocfg, ppy)
        prices = spot_close[universe].dropna(axis=1, how="any")
        universe = list(prices.columns)
        n = len(universe)
        R = returns[universe].dropna()
        mu = R.mean().to_numpy() * ppy
        Sigma = R.cov().to_numpy() * ppy
        scenarios = R.to_numpy()
        log.decision("selected investable universe", n=n, instruments=universe)

        # ---- 2-3. optimise + rebalanced backtest for every method
        results: Dict[str, dict] = {}
        backtests: Dict[str, pbt.PortfolioBacktest] = {}
        for method in ocfg.methods:
            w_full = self._optimise(method, mu, Sigma, scenarios, ocfg)
            fn = self._weight_fn(method, ocfg, ppy)
            with log.timer(f"backtest_{method}"):
                bt = pbt.backtest_rebalanced(prices, fn, ocfg.rebalance_frequency_days,
                                             fee, ocfg.lookback_days, ppy)
            backtests[method] = bt
            results[method] = {
                "weights": {c: float(wi) for c, wi in zip(universe, w_full)},
                "expected_return": float(w_full @ mu),
                "expected_risk": float(np.sqrt(max(w_full @ Sigma @ w_full, 0.0))),
                "sharpe": float((w_full @ mu) / np.sqrt(max(w_full @ Sigma @ w_full, 1e-12))),
                "backtest": bt.metrics,
            }

        # ---- 4. choose the method: profitable first, then most diversified / risk-adjusted
        chosen = self._select_method(results, ocfg)
        bt = backtests[chosen]
        w_chosen = np.array([results[chosen]["weights"][c] for c in universe])
        log.decision("selected optimisation method", method=chosen,
                     total_return=round(results[chosen]["backtest"]["total_return"], 4),
                     sharpe=round(results[chosen]["backtest"]["sharpe"], 3),
                     diversification_ratio=round(results[chosen]["backtest"]["diversification_ratio"], 3),
                     effective_n=round(results[chosen]["backtest"]["effective_n"], 2))

        # ---- equal-weight benchmark for a fair profitability comparison
        eq_bt = pbt.backtest_rebalanced(prices, lambda tr: np.ones(n) / n,
                                        ocfg.rebalance_frequency_days, fee, ocfg.lookback_days, ppy)

        # ---- assemble portfolio artefacts ------------------------------------
        corr = returns[universe].corrwith(returns[primary]) if primary in returns.columns else pd.Series(dtype=float)
        constituents = self._constituents(universe, w_chosen, mu, Sigma, corr, primary)
        equity = pd.DataFrame({
            "ts": bt.equity.index,
            "equity": bt.equity.to_numpy(),
            "benchmark": eq_bt.equity.reindex(bt.equity.index).to_numpy(),
        })
        equity["drawdown"] = bt.equity.to_numpy() / np.maximum.accumulate(bt.equity.to_numpy()) - 1.0

        weights_path = bt.weights_path.copy()
        weights_path.insert(0, "ts", weights_path.index)
        weights_path = weights_path.reset_index(drop=True)

        costs = pd.DataFrame({"ts": bt.cum_cost.index, "cum_cost": bt.cum_cost.to_numpy(),
                              "turnover": bt.turnover.to_numpy()})
        rebalances = [str(d) for d in bt.rebalance_dates]

        diversification = {k: bt.metrics[k] for k in
                           ["diversification_ratio", "avg_diversification_ratio", "effective_n",
                            "n_active", "max_weight", "hhi", "n_rebalances"]}
        diversification["n_assets"] = n
        diversification["benchmark_diversification_ratio"] = eq_bt.metrics.get("diversification_ratio", 1.0)

        method_comparison = pd.DataFrame([
            {"method": m,
             "total_return": r["backtest"]["total_return"],
             "cagr": r["backtest"]["cagr"],
             "sharpe": r["backtest"]["sharpe"],
             "volatility": r["backtest"]["volatility"],
             "max_drawdown": r["backtest"]["max_drawdown"],
             "diversification_ratio": r["backtest"]["diversification_ratio"],
             "effective_n": r["backtest"]["effective_n"],
             "chosen": (m == chosen)}
            for m, r in results.items()
        ])

        # ---- rebalance decision (kept for the explainability/legacy contract)
        w_prev = np.ones(n) / n
        tn = opt.turnover(w_chosen, w_prev)
        tcost = opt.transaction_cost(w_chosen, w_prev, fee, context.config.investment.capital_usd)
        decision = RebalanceDecision(
            method=chosen,
            target_weights=results[chosen]["weights"],
            current_weights={c: float(wp) for c, wp in zip(universe, w_prev)},
            turnover=tn,
            expected_return=results[chosen]["expected_return"],
            expected_risk=results[chosen]["expected_risk"],
            transaction_cost=tcost,
            triggered=bool(tn > 0.5 * ocfg.max_turnover),
            rationale=(f"Method '{chosen}' delivered total return "
                       f"{results[chosen]['backtest']['total_return']:.2%} with diversification ratio "
                       f"{results[chosen]['backtest']['diversification_ratio']:.2f} "
                       f"(effective {results[chosen]['backtest']['effective_n']:.1f} bets)."),
        )

        # ---- persist
        method_comparison.to_parquet(context.results_path("portfolio_methods.parquet"))
        constituents.to_parquet(context.results_path("portfolio_constituents.parquet"))
        equity.to_parquet(context.results_path("portfolio_equity.parquet"))
        weights_path.to_parquet(context.results_path("portfolio_weights_path.parquet"))

        context.put("optimization_results", results)
        context.put("rebalance_decision", decision.to_dict())
        context.put("opt_weights", results[chosen]["weights"])
        context.put("portfolio_universe", universe)
        context.put("portfolio_constituents", constituents)
        context.put("portfolio_equity", equity)
        context.put("portfolio_weights_path", weights_path)
        context.put("portfolio_rebalances", rebalances)
        context.put("portfolio_costs", costs)
        context.put("diversification", diversification)
        context.put("method_comparison", method_comparison)

        log.decision("portfolio optimization", chosen=chosen, n_assets=n,
                     turnover=round(tn, 3), transaction_cost=round(tcost, 2),
                     profitable=bool(results[chosen]["backtest"]["total_return"] > 0),
                     diversification_ratio=round(diversification["diversification_ratio"], 3))

        return Message(self.produces, self.name, "risk_management",
                       payload={"method": chosen, "n_assets": n,
                                "total_return": results[chosen]["backtest"]["total_return"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _select_universe(self, returns: pd.DataFrame, ocfg, ppy: int) -> List[str]:
        """Pick a diversified, historically profitable set of instruments.

        Profitable longs (positive mean return) ranked by risk-adjusted return;
        if too few are positive, fall back to the top names by mean return.
        """
        mu = returns.mean() * ppy
        vol = returns.std() * np.sqrt(ppy)
        sharpe = (mu / vol.replace(0, np.nan)).fillna(0.0)
        k = max(2, min(ocfg.portfolio_universe_size, returns.shape[1]))

        profitable = sharpe[mu > ocfg.min_expected_return].sort_values(ascending=False)
        if len(profitable) >= max(2, k // 2):
            return list(profitable.head(k).index)
        return list(mu.sort_values(ascending=False).head(k).index)

    def _optimise(self, method, mu, Sigma, scenarios, ocfg) -> np.ndarray:
        n = len(mu)
        # keep the bounds feasible for small universes (n * max_weight must be >= 1)
        max_weight = max(ocfg.max_weight, 1.0 / n + 1e-9)
        try:
            return opt.optimize(method, mu, Sigma, scenarios=scenarios,
                                risk_aversion=ocfg.risk_aversion, cvar_alpha=ocfg.cvar_alpha,
                                long_only=ocfg.long_only, max_weight=max_weight)
        except Exception:
            return np.ones(n) / n

    def _weight_fn(self, method, ocfg, ppy):
        def fn(train_returns: pd.DataFrame) -> np.ndarray:
            mu = train_returns.mean().to_numpy() * ppy
            Sigma = train_returns.cov().to_numpy() * ppy
            return self._optimise(method, mu, Sigma, train_returns.to_numpy(), ocfg)
        return fn

    def _select_method(self, results: Dict[str, dict], ocfg) -> str:
        """Pick the best *profitable* method, balancing risk-adjusted return and
        diversification on a common (min-max normalised) scale so the two very
        different magnitudes (Sharpe ~ units, diversification ratio ~ 1-2) are
        weighted fairly. ``diversification_weight`` controls the trade-off."""
        if not ocfg.auto_select_method:
            return ocfg.method if ocfg.method in results else next(iter(results))

        profitable = {m: r for m, r in results.items()
                      if r["backtest"]["total_return"] > ocfg.min_expected_return}
        pool = profitable or results
        if len(pool) == 1:
            return next(iter(pool))

        def col(metric):
            return {m: float(r["backtest"][metric]) for m, r in pool.items()}

        def norm(vals: Dict[str, float]) -> Dict[str, float]:
            lo, hi = min(vals.values()), max(vals.values())
            rng = hi - lo
            return {m: (v - lo) / rng if rng > 1e-12 else 0.5 for m, v in vals.items()}

        n_sharpe = norm(col("sharpe"))
        n_dr = norm(col("diversification_ratio"))
        n_eff = norm(col("effective_n"))
        w = ocfg.diversification_weight
        div = {m: 0.5 * (n_dr[m] + n_eff[m]) for m in pool}
        score = {m: (1.0 - w) * n_sharpe[m] + w * div[m] for m in pool}
        return max(score, key=score.get)

    def _constituents(self, universe, weights, mu, Sigma, corr, primary) -> pd.DataFrame:
        vol = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
        df = pd.DataFrame({
            "symbol": universe,
            "weight": weights,
            "exp_return_annual": mu,
            "vol_annual": vol,
            "relationship": [self._classify(s, primary, corr) for s in universe],
        })
        df = df[df["weight"] > 1e-4].sort_values("weight", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def _classify(symbol, primary, corr) -> str:
        if symbol == primary:
            return "primary"
        c = float(corr.get(symbol, 0.0)) if corr is not None and len(corr) else 0.0
        if c >= 0.5:
            return "positive"
        if c <= -0.3:
            return "inverse"
        if abs(c) < 0.2:
            return "neutral"
        return "weak"
