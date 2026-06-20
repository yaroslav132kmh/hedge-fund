"""Centralised seeding for full reproducibility.

Every stochastic component draws from generators initialised here from a single
configuration value, so identical inputs always produce identical outputs.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed all known RNG sources and return a fresh NumPy ``Generator``.

    Seeds Python's ``random``, NumPy's legacy global RNG, the ``PYTHONHASHSEED``
    environment variable and (if importable) PyTorch. Returns a dedicated
    :class:`numpy.random.Generator` that callers should thread through their code
    instead of relying on global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    try:  # torch is optional for the analytical pipeline
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch absent or partial
        pass

    return np.random.default_rng(seed)


def spawn_rng(seed: int, stream: Optional[int] = None) -> np.random.Generator:
    """Return an independent generator for a named sub-stream.

    Using :class:`numpy.random.SeedSequence` guarantees statistically independent
    yet deterministic streams for different agents / components.
    """
    seq = np.random.SeedSequence(entropy=seed, spawn_key=() if stream is None else (stream,))
    return np.random.default_rng(seq)
