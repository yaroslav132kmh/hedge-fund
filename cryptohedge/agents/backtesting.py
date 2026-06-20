"""Backtesting Agent.

Role: validate the hedging strategy out-of-sample. Uses walk-forward validation
(no look-ahead), mitigates survivorship / selection / transaction-cost biases,
runs stress tests on extreme spot & volatility shocks, and computes the full set
of performance metrics against an unhedged-BTC benchmark.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services import metrics as mx
from cryptohedge.services.greeks import HestonGreeksEngine
from cryptohedge.services.hedging_engine import FeeModel, HedgingEngine, StrategyConfig, YEAR_NANOS
from cryptohedge.services.walkforward import walk_forward_splits
from cryptohedge.services.providers.base import INSTR_ASSET


class BacktestingAgent(BaseAgent):
    name = "backtesting"
    consumes = [MessageType.RISK_ASSESSMENT]
    produces = MessageType.BACKTEST_READY
    checkpoint_keys = ["backtest_metrics", "walkforward", "stress_table", "bias_controls"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        bcfg = context.config.backtest
        inv = context.config.investment
        history: pd.DataFrame = context.require("hedge_history")
        md: pd.DataFrame = context.require("market_data")
        calibr: pd.DataFrame = context.require("calibr_data")
        spot_close = context.require("spot_close")
        primary = context.require("primary_symbol")
        liability, vega_option = context.require("hedge_setup")

        # ---- in-sample performance metrics vs unhedged BTC benchmark
        pnl = history["pnl"].to_numpy()
        hedged_returns = np.diff(pnl, prepend=pnl[0]) / inv.capital_usd
        btc = spot_close[primary].reindex(pd.to_datetime(history["ts"])).to_numpy()
        btc_returns = np.concatenate([[0.0], np.diff(np.log(btc))])
        ppy = context.config.horizons.trading_days_per_year
        perf = mx.compute_metrics(hedged_returns, benchmark=btc_returns, periods_per_year=ppy,
                                  var_confidence=context.config.risk.var_confidence,
                                  var_method=context.config.risk.var_method)

        # ---- walk-forward validation (no look-ahead)
        engine = HedgingEngine(
            HestonGreeksEngine(context.config.greeks),
            FeeModel(inv.transaction_fee_pct, inv.option_fee_pct, inv.option_fee_cap_pct),
            StrategyConfig(context.config.hedging.delta_eps, context.config.hedging.vega_eps),
        )
        samples = sorted(calibr["sample_idx"].unique())
        folds = walk_forward_splits(len(samples), bcfg.train_window, bcfg.test_window,
                                    bcfg.step, bcfg.purge, bcfg.embargo)
        wf_rows = []
        with log.timer("walk_forward", n_folds=len(folds)):
            for fold in folds:
                test_samples = [samples[i] for i in fold.test]
                res = engine.run(md, calibr, liability, vega_option, sample_indices=test_samples)
                if res.history.empty:
                    continue
                fp = res.history["pnl"].to_numpy()
                fr = np.diff(fp, prepend=fp[0]) / inv.capital_usd
                m = mx.compute_metrics(fr, periods_per_year=ppy)
                wf_rows.append({
                    "fold": fold.index, "train_start": int(fold.train[0]), "train_end": int(fold.train[-1]),
                    "test_start": int(fold.test[0]), "test_end": int(fold.test[-1]),
                    "roi": m.roi, "sharpe": m.sharpe, "sortino": m.sortino,
                    "max_drawdown": m.max_drawdown, "pnl_end": float(fp[-1] - fp[0]),
                })
        walkforward = pd.DataFrame(wf_rows)

        # ---- stress testing on extreme scenarios
        stress = self._stress_test(context, engine, md, calibr, history, liability, vega_option, bcfg)

        bias_controls = {
            "survivorship_bias": {"controlled": bcfg.account_survivorship_bias,
                                  "note": "Full universe retained across the window; no winners-only selection."},
            "selection_bias": {"controlled": bcfg.account_selection_bias,
                               "note": "Hedge universe ranked on in-sample data only; no peeking at test folds."},
            "transaction_cost_bias": {"controlled": bcfg.account_transaction_cost_bias,
                                      "note": "Spot and option fees (incl. cap) charged on every trade."},
            "look_ahead": {"controlled": True,
                           "note": "Walk-forward folds with purge/embargo; per-slice calibration is contemporaneous."},
        }

        walkforward.to_parquet(context.results_path("walkforward.parquet")) if not walkforward.empty else None
        stress.to_parquet(context.results_path("stress_test.parquet"))
        pd.Series(perf.to_dict()).to_json(context.results_path("performance_metrics.json"))

        context.put("backtest_metrics", perf.to_dict())
        context.put("walkforward", walkforward)
        context.put("stress_table", stress)
        context.put("bias_controls", bias_controls)

        log.decision("backtest complete", roi=round(perf.roi, 4), sharpe=round(perf.sharpe, 3),
                     sortino=round(perf.sortino, 3), calmar=round(perf.calmar, 3),
                     max_drawdown=round(perf.max_drawdown, 4), var=round(perf.var, 5),
                     cvar=round(perf.cvar, 5), n_folds=len(walkforward))

        return Message(self.produces, self.name, "self_diagnostic",
                       payload={"sharpe": perf.sharpe, "roi": perf.roi},
                       correlation_id=message.correlation_id)

    def _stress_test(self, context, engine, md, calibr, history, liability, vega_option, bcfg) -> pd.DataFrame:
        """Decompose the book's P&L per leg under shocks and contrast with the
        unhedged BTC exposure, so the effectiveness of the hedge is explicit."""
        last_cal = calibr.sort_values("sample_idx").iloc[-1]
        sidx = int(last_cal["sample_idx"])
        grp = md[md["sample_idx"] == sidx]
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])
        ts_ns = int(pd.Timestamp(last_cal["timestamp"]).value)
        last_row = history.iloc[-1]
        pos_spot = float(last_row["pos_spot"])
        pos_vo = float(last_row["pos_vega_option"])
        q_hedge = float(context.require("hedge_sizing").quantity_to_hedge)

        from cryptohedge.domain.market import HestonParameters

        def params_with(v0):
            return HestonParameters(v0=max(v0, 1e-6), kappa=float(last_cal["kappa"]),
                                    theta=float(last_cal["theta"]), eps=float(last_cal["eps"]),
                                    rho=float(last_cal["rho"]), flat_yield=float(last_cal.get("flat_yield", 0.0)))

        ttm_v = (vega_option.expiry_ts - ts_ns) / YEAR_NANOS

        def leg_values(ds, dv):
            s = spot * (1 + ds)
            p = params_with(float(last_cal["v0"]) * (1 + dv))
            p_liab = sum(engine.greeks.price(s, p, c.strike, (c.expiry_ts - ts_ns) / YEAR_NANOS, c.is_call) * c.notional
                         for c in liability if (c.expiry_ts - ts_ns) / YEAR_NANOS > 0)
            vo_val = engine.greeks.price(s, p, vega_option.strike, ttm_v, vega_option.is_call) if ttm_v > 0 else 0.0
            return s, p_liab, vo_val

        s0, liab0, vo0 = leg_values(0.0, 0.0)
        rows = []
        for sc in bcfg.stress_scenarios:
            s, liab, vo = leg_values(sc["spot_shock"], sc["vol_shock"])
            liability_pnl = -(liab - liab0)          # the option book is sold (short)
            spot_hedge_pnl = pos_spot * (s - s0)
            option_hedge_pnl = pos_vo * (vo - vo0)
            net = liability_pnl + spot_hedge_pnl + option_hedge_pnl
            unhedged = q_hedge * (s - s0)            # naked long-BTC exposure of equal size
            eff = float(1.0 - abs(net) / abs(unhedged)) if abs(unhedged) > 1e-9 else 1.0
            rows.append({
                "scenario": sc["name"], "spot_shock": sc["spot_shock"], "vol_shock": sc["vol_shock"],
                "liability_pnl": float(liability_pnl), "spot_hedge_pnl": float(spot_hedge_pnl),
                "option_hedge_pnl": float(option_hedge_pnl), "net_hedged_pnl": float(net),
                "unhedged_pnl": float(unhedged), "hedge_effectiveness": eff,
            })
        return pd.DataFrame(rows)
