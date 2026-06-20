"""Greeks computation and aggregation.

The default :class:`HestonGreeksEngine` computes first- and second-order
sensitivities by finite differences on the semi-analytical Heston price (fast,
deterministic, torch-free): delta, gamma, vega, theta, rho, vanna, volga and
charm. Greeks are consistent across instruments (vega in vol space), so the
delta/vega hedge ratios used by the hedging agent are well-defined. An optional
Monte-Carlo engine mirrors the reference autograd approach when ``engine='mc'``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

from cryptohedge.core.config import GreeksConfig
from cryptohedge.domain.greeks import Greeks, PortfolioGreeks
from cryptohedge.domain.market import HestonParameters, OptionContract
from cryptohedge.services.heston_pricing import heston_premiums


class HestonGreeksEngine:
    """Finite-difference greeks on the analytical Heston price."""

    def __init__(self, config: GreeksConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ pricing
    def price(self, spot: float, params: HestonParameters, K: float, T: float, is_call: bool) -> float:
        return float(heston_premiums(spot, [K], [T], [is_call], params)[0])

    def _with_v0(self, params: HestonParameters, v0: float) -> HestonParameters:
        return HestonParameters(v0=max(v0, 1e-6), kappa=params.kappa, theta=params.theta,
                                eps=params.eps, rho=params.rho, flat_yield=params.flat_yield)

    def _with_rate(self, params: HestonParameters, r: float) -> HestonParameters:
        return HestonParameters(v0=params.v0, kappa=params.kappa, theta=params.theta,
                                eps=params.eps, rho=params.rho, flat_yield=r)

    def _delta(self, spot, params, K, T, is_call, dS) -> float:
        up = self.price(spot + dS, params, K, T, is_call)
        dn = self.price(spot - dS, params, K, T, is_call)
        return (up - dn) / (2 * dS)

    # ------------------------------------------------------------------- greeks
    def compute(self, spot: float, params: HestonParameters, contract: OptionContract, ttm: float) -> Greeks:
        K, is_call = contract.strike, contract.is_call
        cfg = self.config
        dS = max(spot * cfg.spot_bump_pct, 1e-8)
        dsig = cfg.vol_bump
        dr = cfg.rate_bump
        dT = cfg.time_bump_days / 365.0

        sigma0 = math.sqrt(max(params.v0, 1e-8))
        base = self.price(spot, params, K, ttm, is_call)

        # spot greeks
        p_su = self.price(spot + dS, params, K, ttm, is_call)
        p_sd = self.price(spot - dS, params, K, ttm, is_call)
        delta = (p_su - p_sd) / (2 * dS)
        gamma = (p_su - 2 * base + p_sd) / (dS**2)

        # vol greeks (bump in volatility space; v0 = sigma^2)
        params_vu = self._with_v0(params, (sigma0 + dsig) ** 2)
        params_vd = self._with_v0(params, (sigma0 - dsig) ** 2)
        p_vu = self.price(spot, params_vu, K, ttm, is_call)
        p_vd = self.price(spot, params_vd, K, ttm, is_call)
        vega = (p_vu - p_vd) / (2 * dsig)
        volga = (p_vu - 2 * base + p_vd) / (dsig**2)

        # rho
        if dr > 0:
            p_ru = self.price(spot, self._with_rate(params, params.flat_yield + dr), K, ttm, is_call)
            p_rd = self.price(spot, self._with_rate(params, params.flat_yield - dr), K, ttm, is_call)
            rho = (p_ru - p_rd) / (2 * dr)
        else:
            rho = 0.0

        # theta (per day): value lost as maturity shortens by one day
        theta = 0.0
        if ttm - dT > 1e-6:
            p_tm = self.price(spot, params, K, ttm - dT, is_call)
            theta = (p_tm - base) / cfg.time_bump_days

        # vanna: d^2P / (dS dsigma)
        p_su_vu = self.price(spot + dS, params_vu, K, ttm, is_call)
        p_su_vd = self.price(spot + dS, params_vd, K, ttm, is_call)
        p_sd_vu = self.price(spot - dS, params_vu, K, ttm, is_call)
        p_sd_vd = self.price(spot - dS, params_vd, K, ttm, is_call)
        vanna = (p_su_vu - p_su_vd - p_sd_vu + p_sd_vd) / (4 * dS * dsig)

        # charm: d delta / d(time) per day
        charm = 0.0
        if ttm - dT > 1e-6:
            delta_tm = self._delta(spot, params, K, ttm - dT, is_call, dS)
            charm = (delta_tm - delta) / cfg.time_bump_days

        greeks = Greeks(
            premium=base, delta=delta, gamma=gamma, vega=vega, theta=theta,
            rho=rho, vanna=vanna, volga=volga, charm=charm,
        )
        return greeks.scaled(contract.notional)

    # ---------------------------------------------------------------- chain grid
    def chain_greeks(self, spot, params, strikes, ttm, is_call=True) -> List[Greeks]:
        return [
            self.compute(spot, params, OptionContract("primary", float(k), 0, is_call), ttm)
            for k in strikes
        ]


def aggregate(per_instrument: Iterable[Greeks]) -> PortfolioGreeks:
    """Aggregate per-instrument greeks into a single portfolio-level object."""
    total = Greeks()
    for g in per_instrument:
        total = total + g
    return PortfolioGreeks.from_greeks(total)


def greeks_to_frame(per_instrument: Dict[str, Greeks]):
    import pandas as pd

    return pd.DataFrame({name: g.to_dict() for name, g in per_instrument.items()}).T
