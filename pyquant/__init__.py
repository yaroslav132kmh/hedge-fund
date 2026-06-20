"""pyquant: low-level quantitative library (Black-Scholes, Heston, vol surfaces).

This package bundles the numerically heavy primitives used by the multi-agent
hedging system. The numba-accelerated, NumPy-only modules (``common``,
``utils``, ``black_scholes``, ``vol_surface``, ``heston``) can be imported
without PyTorch. The Monte-Carlo modules (``heston_sim``, ``barrier``, ``lsm``,
``torch_spline``) require PyTorch and are imported lazily by the callers that
need them, so the core analytical pipeline runs even when torch is absent.
"""

__all__ = [
    "utils",
    "common",
    "black_scholes",
    "vol_surface",
    "heston",
]
