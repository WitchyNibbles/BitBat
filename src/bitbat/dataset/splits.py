"""Dataset splitting utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.Index
    test: pd.Index


def walk_forward(
    indices: Iterable[pd.Timestamp],
    windows: list[tuple[str, str, str, str]],
    embargo_bars: int,
) -> list[Fold]:
    """Generate walk-forward folds with embargo to avoid leakage."""
    index = pd.Index(sorted(pd.to_datetime(list(indices))), name="timestamp")
    folds: list[Fold] = []
    previous_tests = pd.Index([])

    for train_start, train_end, test_start, test_end in windows:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)

        train_idx = index[(index >= train_start_ts) & (index <= train_end_ts)]
        test_idx = index[(index >= test_start_ts) & (index <= test_end_ts)]

        if embargo_bars > 0 and not train_idx.empty and not test_idx.empty:
            embargo_start = test_idx[0]
            embargo_delta = pd.Timedelta(hours=embargo_bars)
            embargo_cutoff = embargo_start - embargo_delta
            train_idx = train_idx[train_idx < embargo_cutoff]

        if not previous_tests.empty:
            train_idx = train_idx.difference(previous_tests)

        folds.append(Fold(train=train_idx, test=test_idx))
        previous_tests = previous_tests.union(test_idx)

    return folds
