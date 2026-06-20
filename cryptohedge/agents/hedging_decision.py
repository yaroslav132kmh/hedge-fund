"""Hedging Decision Agent.

Role: form and execute the delta-vega hedging strategy with the Heston model.
Running the hedging engine over the in-sample history produces every required
output - spot, PnL, fees, delta-hedge, vega-hedge, spot/option positions, option
portfolio premium, portfolio delta and vega - plus a current actionable decision.
"""

from __future__ import annotations

import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services.greeks import HestonGreeksEngine
from cryptohedge.services.hedging_engine import FeeModel, HedgingEngine, StrategyConfig


class HedgingDecisionAgent(BaseAgent):
    name = "hedging_decision"
    consumes = [MessageType.GREEKS_READY]
    produces = MessageType.HEDGE_DECISION
    checkpoint_keys = ["hedge_history", "hedge_decisions", "hedge_trades", "latest_decision"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        inv = context.config.investment
        engine = HedgingEngine(
            HestonGreeksEngine(context.config.greeks),
            FeeModel(inv.transaction_fee_pct, inv.option_fee_pct, inv.option_fee_cap_pct),
            StrategyConfig(context.config.hedging.delta_eps, context.config.hedging.vega_eps),
        )

        md: pd.DataFrame = context.require("market_data")
        calibr: pd.DataFrame = context.require("calibr_data")
        liability, vega_option = context.require("hedge_setup")

        with log.timer("hedging_run", n=len(calibr)):
            result = engine.run(md, calibr, liability, vega_option)

        history = result.history
        if history.empty:
            raise RuntimeError("Hedging engine produced no results")

        history.to_parquet(context.results_path("hedging_history.parquet"))
        decisions_df = pd.DataFrame([d.to_dict() for d in result.decisions])
        if not decisions_df.empty:
            decisions_df.to_parquet(context.results_path("hedge_decisions.parquet"))

        last = history.iloc[-1]
        latest = {
            "ts": str(last["ts"]), "spot": float(last["spot"]),
            "portfolio_delta": float(last["delta"]), "portfolio_vega": float(last["vega"]),
            "delta_hedge": float(last["delta_hedge"]), "vega_hedge": float(last["vega_hedge"]),
            "residual_delta": float(last["delta"] - last["delta_hedge"]),
            "residual_vega": float(last["vega"] - last["vega_hedge"]),
            "pos_spot": float(last["pos_spot"]), "pos_vega_option": float(last["pos_vega_option"]),
            "pnl": float(last["pnl"]), "fees": float(last["fee"]),
            "n_trades": int(len(result.trades)),
        }

        context.put("hedge_history", history)
        context.put("hedge_decisions", [d.to_dict() for d in result.decisions])
        context.put("hedge_trades", result.trades)
        context.put("latest_decision", latest)

        log.decision("executed delta-vega hedge", n_trades=latest["n_trades"],
                     residual_delta=round(latest["residual_delta"], 4),
                     residual_vega=round(latest["residual_vega"], 4),
                     pnl=round(latest["pnl"], 2), fees=round(latest["fees"], 2))

        return Message(self.produces, self.name, "portfolio_optimization",
                       payload={"pnl": latest["pnl"], "n_trades": latest["n_trades"]},
                       correlation_id=message.correlation_id)
