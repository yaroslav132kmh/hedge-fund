"""Greeks Calculation Agent.

Role: build the liability portfolio (protective put) and the vega-hedge option,
then compute per-instrument and aggregated portfolio greeks - delta, gamma, vega,
theta, rho, vanna, volga, charm - at the latest slice, across the full history
(for monitoring) and across the strike grid (for the dashboard heatmap).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.domain.greeks import Greeks
from cryptohedge.domain.market import HestonParameters, OptionContract
from cryptohedge.services.greeks import HestonGreeksEngine, aggregate
from cryptohedge.services.hedging_engine import YEAR_NANOS
from cryptohedge.services.portfolio_spec import available_strikes, build_hedge_setup
from cryptohedge.services.providers.base import INSTR_ASSET


class GreeksCalculationAgent(BaseAgent):
    name = "greeks_calculation"
    consumes = [MessageType.CALIBRATION_READY]
    produces = MessageType.GREEKS_READY
    checkpoint_keys = ["hedge_setup", "portfolio_greeks_latest", "greeks_per_instrument",
                       "greeks_timeseries", "chain_greeks", "hedge_status", "hedge_contracts"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        engine = HestonGreeksEngine(context.config.greeks)
        md: pd.DataFrame = context.require("market_data")
        calibr: pd.DataFrame = context.require("calibr_data")
        primary = context.require("primary_symbol")
        sizing = context.require("hedge_sizing")

        spot0 = float(md[md["instrument_type"] == INSTR_ASSET].sort_values("sample_idx")["price"].iloc[0])
        liability, vega_option = build_hedge_setup(
            md, spot0, sizing.quantity_to_hedge, primary,
            put_moneyness=context.config.hedging.liability_put_moneyness,
            call_moneyness=context.config.hedging.vega_call_moneyness,
        )

        # ---- full-history portfolio greeks (monitoring time series)
        with log.timer("greeks_timeseries", n=len(calibr)):
            ts_rows = self._timeseries(engine, md, calibr, liability, vega_option)
        greeks_timeseries = pd.DataFrame(ts_rows)

        # ---- latest-slice detailed greeks
        last = calibr.sort_values("sample_idx").iloc[-1]
        sidx = int(last["sample_idx"])
        grp = md[md["sample_idx"] == sidx]
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])
        params = self._params(last)
        ts_ns = int(pd.Timestamp(last["timestamp"]).value)

        per_instrument: Dict[str, Greeks] = {}
        for i, c in enumerate(liability):
            ttm = (c.expiry_ts - ts_ns) / YEAR_NANOS
            per_instrument[f"liability_put_{c.strike:.0f}"] = engine.compute(spot, params, c, ttm)
        ttm_v = (vega_option.expiry_ts - ts_ns) / YEAR_NANOS
        per_instrument["vega_hedge_call"] = engine.compute(spot, params, vega_option, ttm_v)

        portfolio = aggregate([g for name, g in per_instrument.items() if name.startswith("liability")])

        # ---- chain greeks for the heatmap
        chain = self._chain_greeks(engine, spot, params, md, ttm_v)

        # ---- delta/vega balance status (green/red zone)
        delta_usd = portfolio.delta * spot
        frac = abs(delta_usd) / context.config.investment.capital_usd
        zone = ("green" if frac <= context.config.hedging.delta_green_zone else
                "red" if frac >= context.config.hedging.delta_red_zone else "amber")
        status = {"delta_usd": delta_usd, "delta_fraction": frac, "zone": zone,
                  "portfolio_vega": portfolio.vega, "portfolio_gamma": portfolio.gamma}

        context.put("hedge_setup", (liability, vega_option))
        context.put("hedge_contracts", {
            "liability": [c.__dict__ for c in liability],
            "vega_option": vega_option.__dict__,
        })
        context.put("portfolio_greeks_latest", portfolio.to_dict())
        context.put("greeks_per_instrument", {k: v.to_dict() for k, v in per_instrument.items()})
        context.put("greeks_timeseries", greeks_timeseries)
        context.put("chain_greeks", chain)
        context.put("hedge_status", status)

        log.decision("computed portfolio greeks", **{k: round(v, 4) for k, v in portfolio.to_dict().items()})
        log.decision("delta balance", zone=zone, delta_fraction=round(frac, 4))

        return Message(self.produces, self.name, "hedging_decision",
                       payload={"zone": zone, "portfolio_delta": portfolio.delta,
                                "portfolio_vega": portfolio.vega},
                       correlation_id=message.correlation_id)

    @staticmethod
    def _params(row: pd.Series) -> HestonParameters:
        return HestonParameters(v0=float(row["v0"]), kappa=float(row["kappa"]), theta=float(row["theta"]),
                                eps=float(row["eps"]), rho=float(row["rho"]),
                                flat_yield=float(row.get("flat_yield", 0.0)))

    def _timeseries(self, engine, md, calibr, liability, vega_option) -> List[dict]:
        rows = []
        for _, row in calibr.sort_values("sample_idx").iterrows():
            sidx = int(row["sample_idx"])
            grp = md[md["sample_idx"] == sidx]
            asset = grp[grp["instrument_type"] == INSTR_ASSET]
            if asset.empty:
                continue
            spot = float(asset["price"].iloc[0])
            ts_ns = int(pd.Timestamp(row["timestamp"]).value)
            params = self._params(row)
            total = Greeks()
            for c in liability:
                ttm = (c.expiry_ts - ts_ns) / YEAR_NANOS
                if ttm > 0:
                    total = total + engine.compute(spot, params, c, ttm)
            d = total.to_dict()
            d.update({"ts": pd.to_datetime(ts_ns), "sample_idx": sidx, "spot": spot})
            rows.append(d)
        return rows

    def _chain_greeks(self, engine, spot, params, md, ttm) -> pd.DataFrame:
        strikes = available_strikes(md, True)
        rows = []
        for k in strikes:
            g = engine.compute(spot, params, OptionContract("primary", float(k), 0, True), ttm)
            d = g.to_dict()
            d["strike"] = float(k)
            d["moneyness"] = float(k / spot)
            rows.append(d)
        return pd.DataFrame(rows)
