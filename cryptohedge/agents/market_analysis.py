"""Market Analysis Agent.

Role: characterise the market - volatility & vol-of-vol of the primary asset with
a confidence interval, the BTC notional that must be hedged, the dependence of
every other instrument on BTC (Pearson/Spearman/Kendall/DCC-GARCH/cointegration),
the market regime, and a multi-criteria ranking selecting the best hedging
instruments.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services import correlation as corr
from cryptohedge.services.volatility import estimate_volatility, size_primary_hedge


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"
    consumes = [MessageType.DATA_READY]
    produces = MessageType.ANALYSIS_READY
    checkpoint_keys = ["volatility", "hedge_sizing", "correlation_static", "rankings_df",
                       "hedge_universe", "regime"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        cfg = context.config.market_analysis
        primary = context.require("primary_symbol")
        spot_close: pd.DataFrame = context.require("spot_close")
        returns: pd.DataFrame = context.require("returns")

        # ---- volatility & hedge sizing
        with log.timer("volatility"):
            vol = estimate_volatility(
                spot_close[primary].to_numpy(), window=cfg.vol_window,
                vov_window=cfg.vol_of_vol_window, confidence_level=cfg.confidence_level,
                horizon_days=context.config.horizons.forecast_days,
                trading_days=context.config.horizons.trading_days_per_year,
            )
        sizing = size_primary_hedge(
            capital_usd=context.config.investment.capital_usd,
            spot=float(spot_close[primary].iloc[-1]),
            vol=vol,
            risk_budget_pct=context.config.investment.risk_budget_pct,
            confidence_level=cfg.confidence_level,
        )
        log.decision(
            "sized primary hedge",
            daily_vol=round(vol.daily_vol, 5), vol_of_vol=round(vol.vol_of_vol, 5),
            ci=[round(vol.ci_low, 5), round(vol.ci_high, 5)],
            hedge_ratio=round(sizing.hedge_ratio, 4),
            quantity_to_hedge=round(sizing.quantity_to_hedge, 4),
        )

        # ---- static correlations + stability
        with log.timer("static_correlations"):
            static = corr.static_correlations(returns, primary)
            stability = corr.rolling_stability(returns, primary, cfg.correlation.rolling_window)

        # candidate pool for the heavier dynamic / cointegration analysis
        candidates = static["pearson"].abs().sort_values(ascending=False)
        pool = list(candidates.head(min(len(candidates), 3 * cfg.top_n_hedge_instruments)).index)

        dcc = {}
        if "dcc_garch" in cfg.correlation.methods:
            with log.timer("dcc_garch", n=len(pool)):
                dcc = corr.dcc_garch_correlations(
                    returns, primary, pool, a=cfg.correlation.dcc_a, b=cfg.correlation.dcc_b,
                    estimate=True, max_iter=cfg.correlation.dcc_max_iter,
                )
        cointegrated = {}
        if "cointegration" in cfg.correlation.methods:
            with log.timer("cointegration", n=len(pool)):
                cointegrated = corr.cointegration(
                    spot_close, primary, pool, method=cfg.correlation.cointegration_method,
                    pvalue=cfg.correlation.cointegration_pvalue,
                    det_order=cfg.correlation.johansen_det_order,
                    k_ar_diff=cfg.correlation.johansen_k_ar_diff,
                )

        # ---- liquidity & hedge-cost proxies from spot bars
        liquidity, hedge_cost = self._liquidity_and_cost(context, static.index)

        rankings = corr.rank_instruments(
            static.loc[pool], stability, dcc, cointegrated,
            liquidity.loc[pool] if set(pool).issubset(liquidity.index) else liquidity,
            hedge_cost.loc[pool] if set(pool).issubset(hedge_cost.index) else hedge_cost,
            cfg.correlation, cfg.ranking_weights,
        )
        rankings_df = pd.DataFrame([r.to_dict() for r in rankings])
        hedge_universe = [r.symbol for r in rankings[: cfg.top_n_hedge_instruments]]

        regime = self._detect_regime(returns[primary], cfg.regime_window, cfg.regime_n_states)

        context.put("volatility", vol)
        context.put("hedge_sizing", sizing)
        context.put("correlation_static", static)
        context.put("rankings", rankings)
        context.put("rankings_df", rankings_df)
        context.put("hedge_universe", hedge_universe)
        context.put("regime", regime)

        log.decision("selected hedge universe", instruments=hedge_universe,
                     top_scores=[round(r.score, 3) for r in rankings[: cfg.top_n_hedge_instruments]],
                     regime=regime["label"])

        payload = {
            "hedge_universe": hedge_universe,
            "regime": regime["label"],
            "quantity_to_hedge": sizing.quantity_to_hedge,
        }
        return Message(self.produces, self.name, "heston_calibration", payload=payload,
                       correlation_id=message.correlation_id)

    def _liquidity_and_cost(self, context: AgentContext, symbols) -> tuple:
        bars: pd.DataFrame = context.require("spot_bars")
        grp = bars.groupby("symbol")
        dollar_vol = (grp["close"].mean() * grp["volume"].mean())
        spread = ((grp["high"].mean() - grp["low"].mean()) / grp["close"].mean())
        liquidity = dollar_vol.reindex([s for s in symbols]).fillna(dollar_vol.median())
        hedge_cost = spread.reindex([s for s in symbols]).fillna(spread.median())
        liquidity.name, hedge_cost.name = "liquidity", "hedge_cost"
        return liquidity, hedge_cost

    def _detect_regime(self, primary_returns: pd.Series, window: int, n_states: int) -> dict:
        """Volatility-regime classification via K-means on (return, rolling vol)."""
        r = primary_returns.dropna()
        roll_vol = r.rolling(window).std().bfill()
        feats = np.column_stack([r.to_numpy(), roll_vol.to_numpy()])
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_states, n_init=10, random_state=0).fit(feats)
            labels = km.labels_
            order = np.argsort(km.cluster_centers_[:, 1])  # by volatility
            rank = {int(c): i for i, c in enumerate(order)}
            current = rank[int(labels[-1])]
            names = {0: "calm", 1: "normal", 2: "stressed"}
            label = names.get(current, f"state_{current}")
        except Exception:
            current = int(roll_vol.iloc[-1] > roll_vol.median())
            label = "stressed" if current else "calm"
        return {"label": label, "current_vol": float(roll_vol.iloc[-1]),
                "median_vol": float(roll_vol.median())}
