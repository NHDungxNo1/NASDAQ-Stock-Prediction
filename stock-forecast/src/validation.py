"""Walk-forward (rolling-origin) validation.

Random train/test splits leak the future into training for time series.
Instead we train on the past, forecast the next block, then roll forward:

    train: [==========]            test: [--]
    train: [============]          test:   [--]
    train: [==============]        test:     [--]
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def walk_forward_splits(
    n: int, initial_train: int, step: int = 1, expanding: bool = True
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs over rows 0..n-1, in time order.

    initial_train : rows in the first training window
    step          : rows forecast per fold before refitting
    expanding     : True  = training window grows each fold
                    False = fixed-size window that slides forward
    """
    if initial_train < 1 or step < 1:
        raise ValueError("initial_train and step must be positive")
    if initial_train >= n:
        raise ValueError(f"initial_train ({initial_train}) must be smaller than n ({n})")

    for test_start in range(initial_train, n, step):
        test_end = min(test_start + step, n)
        train_start = 0 if expanding else test_start - initial_train
        yield np.arange(train_start, test_start), np.arange(test_start, test_end)
