"""Walk-forward splitting without look-ahead bias.

Generates expanding/rolling train-test folds with optional purge & embargo gaps so
that no test observation can leak information into training (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray


def walk_forward_splits(
    n: int,
    train_window: int,
    test_window: int,
    step: int = None,
    purge: int = 0,
    embargo: int = 0,
    expanding: bool = False,
) -> List[Fold]:
    """Return rolling (or expanding) walk-forward folds over ``range(n)``."""
    step = step or test_window
    folds: List[Fold] = []
    start = 0
    fold_idx = 0
    while True:
        train_end = start + train_window
        test_start = train_end + purge
        test_end = test_start + test_window
        if test_end > n:
            break
        train_start = 0 if expanding else start
        train = np.arange(train_start, train_end)
        test = np.arange(test_start, test_end)
        folds.append(Fold(fold_idx, train, test))
        fold_idx += 1
        start += step + embargo
    return folds
