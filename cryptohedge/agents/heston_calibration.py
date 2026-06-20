"""Heston Calibration Agent.

Role: calibrate the Heston parameters at every time slice (implied-volatility
surface least squares), maintain a maximum-likelihood calibration on the spot
time series, monitor parameter stability over time and benchmark Heston against
Black-Scholes and SABR. All intermediate calibration artefacts are persisted.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.domain.market import HestonParameters
from cryptohedge.services import calibration as cal
from cryptohedge.services.heston_pricing import bs_implied_vol, calibrate_iv_surface, heston_premiums
from cryptohedge.services.providers.base import INSTR_ASSET, INSTR_CALL, INSTR_PUT


class HestonCalibrationAgent(BaseAgent):
    name = "heston_calibration"
    consumes = [MessageType.ANALYSIS_READY]
    produces = MessageType.CALIBRATION_READY
    checkpoint_keys = ["calibr_data", "heston_history", "heston_stability",
                       "heston_benchmarks", "heston_mle"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        hcfg = context.config.heston
        md: pd.DataFrame = context.require("market_data")
        spot_close = context.require("spot_close")
        primary = context.require("primary_symbol")

        # ---- MLE on the spot time series (filtering + maximum likelihood)
        with log.timer("mle"):
            mle = cal.calibrate_mle(
                spot_close[primary].to_numpy(),
                dt=1.0 / context.config.horizons.trading_days_per_year,
                flat_yield=hcfg.flat_yield_fallback,
                trading_days=context.config.horizons.trading_days_per_year,
            )
        log.decision("MLE calibration", v0=round(mle.v0, 5), kappa=round(mle.kappa, 3),
                     theta=round(mle.theta, 5), eps=round(mle.eps, 3), rho=round(mle.rho, 3))

        # ---- per-slice IV-surface calibration
        samples = sorted(md["sample_idx"].unique())
        subsample = max(1, context.config.hedging.calibration_subsample)
        init = list(hcfg.initial_params)
        history: List[HestonParameters] = []
        records = []
        last: Optional[HestonParameters] = None
        n_failed = 0

        with log.timer("per_slice_calibration", n=len(samples)):
            for k, sidx in enumerate(samples):
                grp = md[md["sample_idx"] == sidx]
                ts = grp["timestamp"].iloc[0]
                spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])

                if hcfg.calibration_method == "mle":
                    params = HestonParameters(v0=mle.v0, kappa=mle.kappa, theta=mle.theta,
                                              eps=mle.eps, rho=mle.rho, flat_yield=mle.flat_yield)
                elif k % subsample != 0 and last is not None:
                    params = last
                else:
                    params = self._calibrate_slice(grp, spot, init, hcfg, last, mle)

                if params is None:
                    n_failed += 1
                    params = last or mle
                last = params
                init = list(params.as_array())
                history.append(params)
                records.append({
                    "sample_idx": int(sidx), "timestamp": ts,
                    "v0": params.v0, "kappa": params.kappa, "theta": params.theta,
                    "eps": params.eps, "rho": params.rho, "flat_yield": params.flat_yield,
                    "calibration_error": params.calibration_error,
                })

        calibr_data = pd.DataFrame(records)

        stability = cal.parameter_stability(history, hcfg.stability_max_rel_change)
        benchmarks = self._benchmarks(md, samples, history, hcfg)

        # ---- persist intermediate calibration artefacts
        calibr_data.to_parquet(context.calibration_path("calibr_data.parquet"))
        pd.Series(stability).to_json(context.calibration_path("heston_stability.json"))
        pd.Series({k: str(v) for k, v in benchmarks.items()}).to_json(
            context.calibration_path("heston_benchmarks.json"))

        context.put("calibr_data", calibr_data)
        context.put("heston_history", history)
        context.put("heston_stability", stability)
        context.put("heston_benchmarks", benchmarks)
        context.put("heston_mle", mle)

        log.decision("calibration complete", n_slices=len(samples), n_failed=n_failed,
                     stable=stability["stable"], max_rel_change=round(stability["max_rel_change"], 3),
                     heston_iv_rmse=round(benchmarks.get("heston", {}).get("iv_rmse", float("nan")), 5),
                     bs_iv_rmse=round(benchmarks.get("black_scholes", {}).get("iv_rmse", float("nan")), 5),
                     sabr_iv_rmse=round(benchmarks.get("sabr", {}).get("rmse", float("nan")), 5))

        return Message(self.produces, self.name, "greeks_calculation",
                       payload={"n_slices": len(samples), "stable": stability["stable"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _slice_chain(self, grp: pd.DataFrame, spot: float):
        opts = grp[grp["instrument_type"].isin([INSTR_CALL, INSTR_PUT])]
        strikes = opts["strike"].to_numpy(float)
        ttm = opts["time_to_maturity"].to_numpy(float)
        is_call = (opts["instrument_type"] == INSTR_CALL).to_numpy()
        premiums_usd = opts["price"].to_numpy(float) * spot  # coin-quoted -> USD
        return strikes, ttm, is_call, premiums_usd

    def _calibrate_slice(self, grp, spot, init, hcfg, last, mle) -> Optional[HestonParameters]:
        strikes, ttm, is_call, premiums_usd = self._slice_chain(grp, spot)
        if len(strikes) < 6:
            return None
        try:
            return calibrate_iv_surface(
                spot, strikes, ttm, is_call, premiums_usd, flat_yield=hcfg.flat_yield_fallback,
                init_params=init, num_iter=hcfg.num_iter, tol=hcfg.tol,
            )
        except Exception:
            return None

    def _benchmarks(self, md, samples, history, hcfg) -> dict:
        """Compare Heston, Black-Scholes and SABR on a representative mid slice."""
        mid = samples[len(samples) // 2]
        grp = md[md["sample_idx"] == mid]
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])
        strikes, ttm, is_call, premiums_usd = self._slice_chain(grp, spot)
        if len(strikes) < 6:
            return {"heston": {}, "black_scholes": {}, "sabr": {}}

        T = float(np.median(ttm))
        market_iv = np.array([bs_implied_vol(spot, k, t, hcfg.flat_yield_fallback, p, bool(c))
                              for k, t, p, c in zip(strikes, ttm, premiums_usd, is_call)])

        result = {}
        params = history[len(history) // 2]
        heston_prices = heston_premiums(spot, strikes, ttm, is_call, params)
        heston_iv = np.array([bs_implied_vol(spot, k, t, hcfg.flat_yield_fallback, p, bool(c))
                              for k, t, p, c in zip(strikes, ttm, heston_prices, is_call)])
        mask = np.isfinite(market_iv) & np.isfinite(heston_iv)
        result["heston"] = {
            "iv_rmse": float(np.sqrt(np.nanmean((heston_iv[mask] - market_iv[mask]) ** 2))) if mask.any() else float("nan"),
            "params": params.to_dict(),
        }
        if "black_scholes" in hcfg.benchmarks:
            result["black_scholes"] = cal.black_scholes_benchmark(
                spot, strikes, T, premiums_usd, is_call, hcfg.flat_yield_fallback)
        if "sabr" in hcfg.benchmarks:
            forward = spot * np.exp(hcfg.flat_yield_fallback * T)
            sabr = cal.sabr_calibrate(forward, strikes, T, market_iv, beta=hcfg.sabr_beta)
            result["sabr"] = sabr.to_dict()
        return result
