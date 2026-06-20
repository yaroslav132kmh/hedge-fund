"""Helpers to derive the liability portfolio and the vega-hedge option from data.

Shared by the greeks, hedging and backtesting agents so they all reference the
exact same instruments.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from cryptohedge.domain.market import OptionContract
from cryptohedge.services.providers.base import INSTR_CALL, INSTR_PUT


def option_expiry(market_data: pd.DataFrame) -> int:
    opts = market_data[market_data["instrument_type"].isin([INSTR_CALL, INSTR_PUT])]
    if opts.empty:
        raise ValueError("No option rows found in market data")
    return int(opts["expiry_ts"].astype("int64").mode().iloc[0])


def available_strikes(market_data: pd.DataFrame, is_call: bool) -> np.ndarray:
    instr = INSTR_CALL if is_call else INSTR_PUT
    return np.sort(market_data[market_data["instrument_type"] == instr]["strike"].unique())


def nearest(strikes: np.ndarray, target: float) -> float:
    return float(strikes[np.argmin(np.abs(strikes - target))])


def build_hedge_setup(
    market_data: pd.DataFrame,
    spot0: float,
    quantity_to_hedge: float,
    primary_symbol: str,
    put_moneyness: float = 0.95,
    call_moneyness: float = 1.0,
) -> Tuple[List[OptionContract], OptionContract]:
    """Return (liability_portfolio, vega_hedge_option).

    Liability = a protective put on the primary asset, sized to the BTC notional
    that must be hedged. Vega is hedged with a near-ATM call at the same expiry.
    """
    expiry = option_expiry(market_data)
    call_strikes = available_strikes(market_data, True)
    put_strikes = available_strikes(market_data, False)
    if len(call_strikes) == 0 or len(put_strikes) == 0:
        raise ValueError("Need both call and put strikes to build hedge setup")

    vega_strike = nearest(call_strikes, spot0 * call_moneyness)
    put_strike = nearest(put_strikes, spot0 * put_moneyness)

    vega_option = OptionContract(primary_symbol, vega_strike, expiry, True, notional=1.0)
    liability = [OptionContract(primary_symbol, put_strike, expiry, False, notional=max(quantity_to_hedge, 1.0))]
    return liability, vega_option
