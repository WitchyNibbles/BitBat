"""Dataset splitting utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.Index
    test: pd.Index


def _as_timedelta(value: str | pd.Timedelta) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    parsed = pd.to_timedelta(value)
    if parsed <= pd.Timedelta(0):
        raise ValueError(f"Window duration must be positive: {value}")
    return parsed


def generate_rolling_windows(
    indices: Iterable[pd.Timestamp],
    *,
    train_window: str | pd.Timedelta,
    backtest_window: str | pd.Timedelta,
    step: str | pd.Timedelta | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> list[tuple[str, str, str, str]]:
    """Generate deterministic rolling train/backtest windows from duration controls."""
    index = pd.Index(sorted(pd.to_datetime(list(indices))), name="timestamp")
    if index.empty:
        return []

    train_delta = _as_timedelta(train_window)
    backtest_delta = _as_timedelta(backtest_window)
    step_delta = _as_timedelta(step) if step is not None else backtest_delta

    start_bound = pd.Timestamp(start) if start is not None else index.min()
    end_bound = pd.Timestamp(end) if end is not None else index.max()
    if start_bound >= end_bound:
        return []

    cursor = start_bound + train_delta
    windows: list[tuple[str, str, str, str]] = []
    while cursor + backtest_delta <= end_bound:
        train_start = cursor - train_delta
        train_end = cursor
        test_start = cursor
        test_end = cursor + backtest_delta

        windows.append((
            train_start.isoformat(sep=" "),
            train_end.isoformat(sep=" "),
            test_start.isoformat(sep=" "),
            test_end.isoformat(sep=" "),
        ))
        cursor += step_delta

    return windows


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
