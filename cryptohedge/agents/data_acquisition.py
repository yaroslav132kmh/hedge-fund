"""Data Acquisition Agent.

Role: load and validate market data (spot universe + primary-asset option chain)
from the configured provider, compute returns and publish a clean dataset.
Responsibility boundary: it is the *only* agent that touches external data
sources; everyone else consumes its validated output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.services.providers import build_provider


class DataAcquisitionAgent(BaseAgent):
    name = "data_acquisition"
    consumes = [MessageType.START]
    produces = MessageType.DATA_READY
    checkpoint_keys = ["spot_close", "returns", "market_data", "spot_bars", "symbols",
                       "primary_symbol", "data_meta"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)
        cfg = context.config
        n_steps = cfg.horizons.analysis_days

        provider = build_provider(cfg.data, root=context.root, seed=cfg.seed, n_steps=n_steps)
        with log.timer("load_data", provider=cfg.data.provider):
            bundle = provider.load()

        self._validate(bundle, log)

        returns = np.log(bundle.spot_close).diff().dropna(how="all")
        returns = returns.dropna(axis=1, how="any")

        context.put("spot_close", bundle.spot_close)
        context.put("spot_bars", bundle.spot_bars)
        context.put("returns", returns)
        context.put("market_data", bundle.option_market_data)
        context.put("symbols", bundle.symbols)
        context.put("primary_symbol", bundle.primary_symbol)

        meta = {
            "provider": cfg.data.provider,
            "n_symbols": len(bundle.symbols),
            "n_samples": int(bundle.spot_close.shape[0]),
            "n_option_rows": int(bundle.option_market_data.shape[0]),
            "primary_symbol": bundle.primary_symbol,
            "date_start": str(bundle.spot_close.index[0]),
            "date_end": str(bundle.spot_close.index[-1]),
        }
        context.put("data_meta", meta)
        log.decision("loaded and validated market data", **meta)

        return Message(self.produces, self.name, "market_analysis", payload=meta,
                       correlation_id=message.correlation_id)

    def _validate(self, bundle, log) -> None:
        close = bundle.spot_close
        if close.isna().any().any():
            n = int(close.isna().sum().sum())
            log.warning("forward-filling NaNs in spot_close", n_missing=n)
            bundle.spot_close = close.ffill().bfill()
        if (bundle.spot_close <= 0).any().any():
            raise ValueError("Non-positive spot prices detected")
        if bundle.primary_symbol not in bundle.spot_close.columns:
            raise ValueError("Primary symbol missing from spot data")
        opt = bundle.option_market_data
        if (opt["price"] < 0).any():
            raise ValueError("Negative option prices detected")
        log.info("validation passed", n_symbols=len(bundle.symbols),
                 n_samples=int(bundle.spot_close.shape[0]))
