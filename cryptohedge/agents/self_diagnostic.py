"""Self-Diagnostic Agent.

Role: assess the system's own adequacy. Monitors data drift (PSI/KS), model
degradation (calibration error & volatility-forecast error), hedge quality
(residual delta/vega) and risk compliance, and condenses them into a single
Confidence Score in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services import drift as dr


class SelfDiagnosticAgent(BaseAgent):
    name = "self_diagnostic"
    consumes = [MessageType.BACKTEST_READY]
    produces = MessageType.DIAGNOSTIC_READY
    checkpoint_keys = ["diagnostic", "confidence_score"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        dcfg = context.config.diagnostic
        returns: pd.DataFrame = context.require("returns")
        primary = context.require("primary_symbol")
        calibr: pd.DataFrame = context.require("calibr_data")
        history: pd.DataFrame = context.require("hedge_history")
        stability = context.require("heston_stability")
        risk = context.require("risk_assessment")

        r = returns[primary].to_numpy()
        half = len(r) // 2
        psi = dr.population_stability_index(r[:half], r[half:], bins=10)
        ks = dr.ks_drift(r[:half], r[half:])

        # volatility forecast error: Heston daily vol vs realised |return|
        v0 = calibr.sort_values("sample_idx")["v0"].to_numpy()
        ppy = context.config.horizons.trading_days_per_year
        pred_vol = np.sqrt(np.maximum(v0, 0) / ppy)
        realised = np.abs(np.concatenate([[0.0], np.diff(np.log(history["spot"].to_numpy()))]))
        m = min(len(pred_vol), len(realised))
        fe = dr.forecast_errors(realised[-m:], pred_vol[-m:])
        mean_realised = float(np.mean(realised[-m:])) or 1e-6

        # hedge quality: residual greeks
        resid_delta = np.abs(history["delta"] - history["delta_hedge"]).to_numpy()
        resid_delta_usd = float(np.mean(resid_delta * history["spot"].to_numpy()))
        resid_frac = resid_delta_usd / context.config.investment.capital_usd

        mean_cal_err = float(np.nanmean(calibr.get("calibration_error", pd.Series([np.nan]))))
        components = {
            "calibration": float(np.clip(1.0 / (1.0 + (0.0 if np.isnan(mean_cal_err) else mean_cal_err))
                                         * (0.5 if not stability.get("stable", True) else 1.0), 0, 1)),
            "data_drift": float(np.clip(1.0 - psi / max(dcfg.drift_threshold, 1e-6), 0, 1)),
            "forecast_error": float(np.clip(1.0 - fe["rmse"] / mean_realised, 0, 1)),
            "hedge_quality": float(np.clip(1.0 - resid_frac / max(context.config.hedging.delta_red_zone, 1e-6), 0, 1)),
            "risk_compliance": 1.0 if risk.get("within_limits", False) else 0.3,
        }
        confidence = dr.confidence_score(components, dcfg.confidence_weights)

        diagnostic = {
            "psi": psi, "ks": ks, "drift_detected": bool(psi > dcfg.drift_threshold or ks["pvalue"] < 0.05),
            "forecast_error": fe, "stability": stability,
            "residual_delta_usd": resid_delta_usd, "residual_delta_fraction": resid_frac,
            "components": components, "confidence_score": confidence,
            "self_assessment": self._label(confidence),
        }

        pd.Series({k: str(v) for k, v in diagnostic.items()}).to_json(
            context.results_path("diagnostic.json"))
        context.put("diagnostic", diagnostic)
        context.put("confidence_score", confidence)

        log.decision("self-diagnostic", confidence=round(confidence, 3),
                     drift_detected=diagnostic["drift_detected"], assessment=diagnostic["self_assessment"],
                     components={k: round(v, 3) for k, v in components.items()})

        return Message(self.produces, self.name, "explainability",
                       payload={"confidence": confidence, "assessment": diagnostic["self_assessment"]},
                       correlation_id=message.correlation_id)

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75:
            return "high_confidence"
        if score >= 0.5:
            return "moderate_confidence"
        if score >= 0.3:
            return "low_confidence"
        return "unreliable"
