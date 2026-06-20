"""Heston delta-vega hedging engine.

A faithful, config-driven refactor of ``heston_greeks_hedging.ipynb``: it walks a
time series of spot/option quotes and per-slice Heston parameters, computes the
greeks of the liability option portfolio, and dynamically neutralises its delta
(with spot) and vega (with a hedging option), accounting for transaction fees.

It returns the full set of required outputs - spot, strategy PnL, fees paid,
delta-hedge, vega-hedge, spot/option positions, option-portfolio premium and the
portfolio delta/vega (plus gamma, theta, rho, charm for monitoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cryptohedge.domain.decisions import HedgeDecision
from cryptohedge.domain.greeks import Greeks
from cryptohedge.domain.market import HestonParameters, OptionContract
from cryptohedge.services.greeks import HestonGreeksEngine
from cryptohedge.services.providers.base import INSTR_ASSET, INSTR_CALL, INSTR_PUT

YEAR_NANOS = 31_536_000_000_000_000


@dataclass
class FeeModel:
    spot_fee_pct: float = 0.0003
    option_fee_pct: float = 0.0003
    option_fee_cap_pct: float = 0.125


@dataclass
class StrategyConfig:
    delta_eps: float = 0.0
    vega_eps: float = 0.0


@dataclass
class HedgingResult:
    history: pd.DataFrame
    trades: List[dict] = field(default_factory=list)
    decisions: List[HedgeDecision] = field(default_factory=list)


class HedgingEngine:
    def __init__(self, greeks_engine: HestonGreeksEngine, fees: FeeModel, strategy: StrategyConfig) -> None:
        self.greeks = greeks_engine
        self.fees = fees
        self.strategy = strategy

    # -------------------------------------------------------------- data access
    @staticmethod
    def _asset_row(md_slice: pd.DataFrame) -> pd.Series:
        rows = md_slice[md_slice["instrument_type"] == INSTR_ASSET]
        if len(rows) != 1:
            raise ValueError("Expected exactly one ASSET row per sample")
        return rows.iloc[0]

    @staticmethod
    def _option_row(md_slice: pd.DataFrame, contract: OptionContract) -> Optional[pd.Series]:
        instr = INSTR_CALL if contract.is_call else INSTR_PUT
        rows = md_slice[
            (md_slice["instrument_type"] == instr)
            & (np.isclose(md_slice["strike"], contract.strike))
            & (md_slice["expiry_ts"].astype("int64") == contract.expiry_ts)
        ]
        return rows.iloc[0] if len(rows) else None

    @staticmethod
    def _params(cal_row: pd.Series) -> HestonParameters:
        return HestonParameters(
            v0=float(cal_row["v0"]), kappa=float(cal_row["kappa"]), theta=float(cal_row["theta"]),
            eps=float(cal_row["eps"]), rho=float(cal_row["rho"]),
            flat_yield=float(cal_row.get("flat_yield", 0.0)),
        )

    # ------------------------------------------------------------------- greeks
    def _portfolio_greeks(self, spot, params, contracts, ts) -> Greeks:
        total = Greeks()
        for c in contracts:
            ttm = (c.expiry_ts - ts) / YEAR_NANOS
            if ttm <= 0:
                continue
            total = total + self.greeks.compute(spot, params, c, ttm)
        return total

    # --------------------------------------------------------------------- run
    def run(
        self,
        market_data: pd.DataFrame,
        calibr_data: pd.DataFrame,
        portfolio_contracts: List[OptionContract],
        vega_option: OptionContract,
        sample_indices: Optional[List[int]] = None,
    ) -> HedgingResult:
        calibr_data = calibr_data.sort_values("sample_idx").reset_index(drop=True)
        if sample_indices is not None:
            calibr_data = calibr_data[calibr_data["sample_idx"].isin(sample_indices)].reset_index(drop=True)

        md_by_sample = {idx: grp for idx, grp in market_data.groupby("sample_idx")}

        pos = {"spot": 0.0, "vega_opt": 0.0}
        pos_quote = 0.0
        fee_paid = 0.0
        hedge_delta = 0.0
        hedge_vega = 0.0
        portf_init_premium: Optional[float] = None

        rows: List[dict] = []
        trades: List[dict] = []
        decisions: List[HedgeDecision] = []

        for _, cal_row in calibr_data.iterrows():
            sample_idx = int(cal_row["sample_idx"])
            if sample_idx not in md_by_sample:
                continue
            md_slice = md_by_sample[sample_idx]
            asset = self._asset_row(md_slice)
            ts = int(pd.Timestamp(cal_row["timestamp"]).value)
            spot = float(asset["price"])
            spot_bid = float(asset["best_bid_price"])
            spot_ask = float(asset["best_ask_price"])
            params = self._params(cal_row)

            portf = self._portfolio_greeks(spot, params, portfolio_contracts, ts)
            if portf_init_premium is None:
                portf_init_premium = portf.premium

            vo_row = self._option_row(md_slice, vega_option)
            vo_ttm = (vega_option.expiry_ts - ts) / YEAR_NANOS
            if vo_row is None or vo_ttm <= 0:
                vo_greeks = Greeks()
            else:
                vo_greeks = self.greeks.compute(spot, params, vega_option, vo_ttm)

            # ---- hedge vega with the option, then delta with spot
            self._recompute_hedge(pos, vo_greeks)
            hedge_vega = pos["vega_opt"] * vo_greeks.vega
            hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta

            diff_vega = portf.vega - hedge_vega
            if vo_row is not None and abs(vo_greeks.vega) > 1e-12 and abs(diff_vega) > self.strategy.vega_eps:
                amount = diff_vega / vo_greeks.vega
                side = "buy" if amount > 0 else "sell"
                t = self._trade("vega_opt", side, abs(amount), spot, spot_bid, spot_ask, vo_row, pos)
                pos_quote += t["cash"]
                fee_paid += t["fee"]
                trades.append(t)
                decisions.append(HedgeDecision(
                    timestamp=ts, instrument="vega_option", side=side, quantity=abs(amount),
                    target_greek="vega", pre_hedge_value=hedge_vega, post_hedge_value=portf.vega,
                    rationale="Neutralise portfolio vega via hedging option",
                    metrics={"portfolio_vega": portf.vega, "option_vega": vo_greeks.vega},
                ))
                self._recompute_hedge(pos, vo_greeks)
                hedge_vega = pos["vega_opt"] * vo_greeks.vega
                hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta

            diff_delta = portf.delta - hedge_delta
            if abs(diff_delta) > self.strategy.delta_eps:
                side = "buy" if diff_delta > 0 else "sell"
                t = self._trade("spot", side, abs(diff_delta), spot, spot_bid, spot_ask, None, pos)
                pos_quote += t["cash"]
                fee_paid += t["fee"]
                trades.append(t)
                decisions.append(HedgeDecision(
                    timestamp=ts, instrument="spot", side=side, quantity=abs(diff_delta),
                    target_greek="delta", pre_hedge_value=hedge_delta, post_hedge_value=portf.delta,
                    rationale="Neutralise portfolio delta via spot",
                    metrics={"portfolio_delta": portf.delta},
                ))
                self._recompute_hedge(pos, vo_greeks)
                hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta

            net_worth = self._mark_to_market(
                portf_init_premium, portf.premium, pos, pos_quote, spot, spot_bid, spot_ask, vo_row
            )

            rows.append({
                "ts": pd.to_datetime(ts), "sample_idx": sample_idx, "spot": spot,
                "premium": portf.premium, "delta": portf.delta, "gamma": portf.gamma,
                "vega": portf.vega, "theta": portf.theta, "rho": portf.rho, "charm": portf.charm,
                "delta_hedge": hedge_delta, "vega_hedge": hedge_vega,
                "vega_option_premium": vo_greeks.premium, "vega_option_delta": vo_greeks.delta,
                "vega_option_vega": vo_greeks.vega,
                "pos_spot": pos["spot"], "pos_vega_option": pos["vega_opt"], "pos_usd": pos_quote,
                "fee": fee_paid, "pnl": net_worth,
            })

        history = pd.DataFrame(rows)
        return HedgingResult(history=history, trades=trades, decisions=decisions)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _recompute_hedge(pos: Dict[str, float], vo_greeks: Greeks) -> None:
        # kept for symmetry / future extension (greeks recomputed by caller)
        return None

    def _trade(self, instr, side, amount, spot, spot_bid, spot_ask, opt_row, pos) -> dict:
        sign = 1.0 if side == "buy" else -1.0
        fee = 0.0
        if instr == "spot":
            price = spot_ask if sign > 0 else spot_bid
            fee = self.fees.spot_fee_pct * amount * price
        else:
            spot_mid = 0.5 * (spot_bid + spot_ask)
            coin_price = float(opt_row["best_ask_price"] if sign > 0 else opt_row["best_bid_price"])
            price = coin_price * spot_mid
            fee = min(amount * spot_mid * self.fees.option_fee_pct, self.fees.option_fee_cap_pct * price)
        key = "spot" if instr == "spot" else "vega_opt"
        pos[key] += sign * amount
        cash = -sign * amount * price - fee
        return {"instrument": instr, "side": side, "quantity": amount, "price": price, "fee": fee, "cash": cash}

    def _mark_to_market(self, init_premium, premium, pos, pos_quote, spot, spot_bid, spot_ask, opt_row) -> float:
        net = init_premium - premium + pos_quote
        # liquidate spot
        if pos["spot"] != 0:
            liq = spot_bid if pos["spot"] > 0 else spot_ask
            net += pos["spot"] * liq
        # liquidate hedging option
        if pos["vega_opt"] != 0 and opt_row is not None:
            spot_mid = 0.5 * (spot_bid + spot_ask)
            coin = float(opt_row["best_bid_price"] if pos["vega_opt"] > 0 else opt_row["best_ask_price"])
            net += pos["vega_opt"] * coin * spot_mid
        return float(net)
