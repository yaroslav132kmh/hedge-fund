"""Thin, NumPy-friendly wrapper around the numba ``pyquant`` Heston engine.

This isolates the rest of the codebase from the low-level jitclass API. It exposes
vectorised premium pricing, implied-vol inversion and implied-volatility-surface
calibration (Levenberg-Marquardt least squares on option premiums), returning the
plain :class:`cryptohedge.domain.HestonParameters` value object.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from cryptohedge.domain.market import HestonParameters

# pyquant numba primitives ----------------------------------------------------
from pyquant.common import (
    ForwardRates,
    ForwardYield,
    DiscountYield,
    Forward,
    Spot,
    Strike,
    Strikes,
    StrikesMaturitiesGrid,
    OptionTypes,
    Premium,
    Premiums,
    TimeToMaturity,
    TimesToMaturity,
    CalibrationWeights,
    forward_curve_from_forward_rates,
)
from pyquant.black_scholes import BSCalc
from pyquant.heston import (
    HestonCalc,
    HestonParams,
    Variance,
    VarReversion,
    AverageVar,
    VolOfVar,
    Correlation,
    FlatForwardYield,
)
from pyquant.vol_surface import VolSurfaceChainSpace

_HESTON = HestonCalc()
_BS = BSCalc()


def _heston_params(p: HestonParameters) -> HestonParams:
    return HestonParams(
        Variance(p.v0),
        VarReversion(p.kappa),
        AverageVar(p.theta),
        VolOfVar(p.eps),
        Correlation(p.rho),
        FlatForwardYield(p.flat_yield),
    )


def heston_premiums(
    spot: float,
    strikes: Sequence[float],
    ttm: Sequence[float],
    is_call: Sequence[bool],
    params: HestonParameters,
) -> np.ndarray:
    """Vectorised Heston premiums for a set of (strike, maturity, type) points."""
    Ks = np.asarray(strikes, dtype=np.float64)
    Ts = np.asarray(ttm, dtype=np.float64)
    calls = np.asarray(is_call, dtype=np.bool_)
    grid = StrikesMaturitiesGrid(Spot(float(spot)), TimesToMaturity(Ts), Strikes(Ks))
    return np.asarray(_HESTON._grid_premiums(_heston_params(params), grid, OptionTypes(calls)))


def bs_implied_vol(
    spot: float, strike: float, ttm: float, rate: float, premium: float, is_call: bool
) -> float:
    """Black-Scholes implied volatility from a premium (NaN if it cannot invert)."""
    if premium <= 0 or ttm <= 0:
        return float("nan")
    fwd = Forward(Spot(float(spot)), ForwardYield(float(rate)), DiscountYield(float(rate)), TimeToMaturity(float(ttm)))
    try:
        return float(_BS.implied_vol(fwd, Strike(float(strike)), Premium(float(premium))).sigma)
    except Exception:
        return float("nan")


def calibrate_iv_surface(
    spot: float,
    strikes: Sequence[float],
    ttm: Sequence[float],
    is_call: Sequence[bool],
    premiums: Sequence[float],
    flat_yield: float = 0.0,
    init_params: Optional[Sequence[float]] = None,
    num_iter: int = 50,
    tol: float = 1e-8,
) -> HestonParameters:
    """Calibrate Heston to an option chain by least squares on premiums.

    This is the implied-volatility-surface calibration route (the LM optimiser
    fits model premiums, equivalently the IV surface, to the market quotes).
    """
    Ks = np.asarray(strikes, dtype=np.float64)
    Ts = np.asarray(ttm, dtype=np.float64)
    calls = np.asarray(is_call, dtype=np.bool_)
    pvs = np.asarray(premiums, dtype=np.float64)

    order = np.argsort(Ts, kind="stable")
    Ks, Ts, calls, pvs = Ks[order], Ts[order], calls[order], pvs[order]

    valid = (pvs > 0) & np.isfinite(pvs) & (Ts > 0)
    Ks, Ts, calls, pvs = Ks[valid], Ts[valid], calls[valid], pvs[valid]
    if len(pvs) < 6:
        raise ValueError("Need at least 6 valid option quotes to calibrate Heston")

    unique_T = np.unique(Ts)
    fwd_rates = float(spot) * np.exp(flat_yield * unique_T)
    fwd_curve = forward_curve_from_forward_rates(
        Spot(float(spot)), ForwardRates(fwd_rates), TimesToMaturity(unique_T)
    )
    chain = VolSurfaceChainSpace(
        fwd_curve, TimesToMaturity(Ts), Strikes(Ks), OptionTypes(calls), Premiums(pvs)
    )
    if len(chain.pvs) < 6:
        raise ValueError("Too few out-of-the-money quotes survived filtering")

    hc = HestonCalc()
    hc.num_iter = int(num_iter)
    hc.tol = float(tol)
    if init_params is not None:
        ip = np.asarray(init_params, dtype=np.float64)
        hc.update_cached_params(
            HestonParams(
                Variance(ip[0]), VarReversion(ip[1]), AverageVar(ip[2]),
                VolOfVar(ip[3]), Correlation(ip[4]), FlatForwardYield(flat_yield),
            )
        )
    weights = CalibrationWeights(np.ones_like(chain.pvs))
    params, err = hc.calibrate(chain, FlatForwardYield(float(flat_yield)), weights)

    return HestonParameters(
        v0=float(params.v0),
        kappa=float(params.kappa),
        theta=float(params.theta),
        eps=float(params.eps),
        rho=float(params.rho),
        flat_yield=float(flat_yield),
        calibration_error=float(err.v),
        feller_satisfied=bool(2.0 * params.kappa * params.theta - params.eps**2 >= 0.0),
    )


def heston_atm_iv(spot: float, ttm: float, params: HestonParameters) -> float:
    """ATM implied vol implied by Heston params (used for benchmarks / smiles)."""
    k = float(spot)
    prem = heston_premiums(spot, [k], [ttm], [True], params)[0]
    return bs_implied_vol(spot, k, ttm, params.flat_yield, float(prem), True)
