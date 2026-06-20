"""Risk Management Agent.

Role: enforce the risk budget. Computes VaR, CVaR and Expected Shortfall of the
hedged book, checks them against configured limits (VaR, drawdown, leverage),
and produces adaptive stop-loss levels (ATR + VaR + Heston-vol aware) together
with a dynamic trailing-stop trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.domain.decisions import RiskAssessment
from cryptohedge.services import metrics as mx
from cryptohedge.services.stops import TrailingStop, adaptive_stop, average_true_range


class RiskManagementAgent(BaseAgent):
    name = "risk_management"
    consumes = [MessageType.PORTFOLIO_READY]
    produces = MessageType.RISK_ASSESSMENT
    checkpoint_keys = ["risk_assessment", "stop_level", "trailing_stops", "risk_returns"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        rcfg = context.config.risk
        inv = context.config.investment
        history: pd.DataFrame = context.require("hedge_history")
        spot_bars: pd.DataFrame = context.require("spot_bars")
        primary = context.require("primary_symbol")
        calibr: pd.DataFrame = context.require("calibr_data")

        # ---- hedged PnL returns (per capital)
        pnl = history["pnl"].to_numpy()
        pnl_changes = np.diff(pnl, prepend=pnl[0]) / inv.capital_usd
        var = mx.value_at_risk(pnl_changes, rcfg.var_confidence, rcfg.var_method)
        cvar = mx.conditional_var(pnl_changes, rcfg.cvar_confidence)
        equity = inv.capital_usd + (pnl - pnl[0])
        mdd = mx.max_drawdown(equity)

        breached = []
        if var > rcfg.var_limit_pct:
            breached.append("VaR")
        if abs(mdd) > rcfg.max_drawdown_limit_pct:
            breached.append("MaxDrawdown")

        utilization = {
            "var_vs_limit": float(var / rcfg.var_limit_pct) if rcfg.var_limit_pct else 0.0,
            "drawdown_vs_limit": float(abs(mdd) / rcfg.max_drawdown_limit_pct) if rcfg.max_drawdown_limit_pct else 0.0,
        }
        assessment = RiskAssessment(
            var=var, cvar=cvar, expected_shortfall=cvar, max_drawdown=mdd,
            within_limits=(len(breached) == 0), breached_limits=breached, utilization=utilization,
        )

        # ---- adaptive stop-loss on the primary BTC exposure
        bars = spot_bars[spot_bars["symbol"] == primary].sort_values("timestamp")
        atr = average_true_range(bars["high"].to_numpy(), bars["low"].to_numpy(),
                                 bars["close"].to_numpy(), rcfg.stop_loss.atr_window)
        atr_last = float(np.nan_to_num(atr[-1]))
        ref_price = float(bars["close"].iloc[-1])
        btc_returns = np.diff(np.log(bars["close"].to_numpy()))
        daily_var = mx.value_at_risk(btc_returns, rcfg.var_confidence, "historical")
        v0_last = float(calibr.sort_values("sample_idx")["v0"].iloc[-1])
        heston_daily_vol = float(np.sqrt(max(v0_last, 0.0) / context.config.horizons.trading_days_per_year))

        stop = adaptive_stop(ref_price, atr_last, daily_var, heston_daily_vol, "long", rcfg.stop_loss)

        # ---- dynamic trailing stop trajectory
        trailing_rows = []
        if rcfg.stop_loss.trailing:
            tstop = TrailingStop("long", float(bars["close"].iloc[0]), rcfg.stop_loss)
            closes = bars["close"].to_numpy()
            for i, px in enumerate(closes):
                a = float(np.nan_to_num(atr[i])) if i < len(atr) else atr_last
                lvl = tstop.update(float(px), a, daily_var, heston_daily_vol)
                trailing_rows.append({"ts": bars["timestamp"].iloc[i], "price": float(px),
                                      "stop_price": lvl.stop_price, "triggered": bool(tstop.triggered)})
        trailing = pd.DataFrame(trailing_rows)

        context.put("risk_assessment", assessment.to_dict())
        context.put("stop_level", stop.to_dict())
        context.put("trailing_stops", trailing)
        context.put("risk_returns", pnl_changes)

        if not trailing.empty:
            trailing.to_parquet(context.results_path("trailing_stops.parquet"))

        log.decision("risk assessment", var=round(var, 5), cvar=round(cvar, 5),
                     max_drawdown=round(mdd, 4), within_limits=assessment.within_limits,
                     breached=breached, stop_price=round(stop.stop_price, 2),
                     stop_distance_pct=round(stop.distance_pct, 4))

        return Message(self.produces, self.name, "backtesting",
                       payload={"within_limits": assessment.within_limits, "var": var},
                       correlation_id=message.correlation_id)
