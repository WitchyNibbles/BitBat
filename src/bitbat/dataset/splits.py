"""Dataset splitting utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.Index
    test: pd.Index
    embargo_bars: int = 0
    purge_bars: int = 0


def _as_timedelta(value: str | pd.Timedelta) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    parsed = pd.to_timedelta(value)
    if parsed <= pd.Timedelta(0):
        raise ValueError(f"Window duration must be positive: {value}")
    return parsed


def _infer_bar_delta(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(hours=1)
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return pd.Timedelta(hours=1)
    return positive.median()


def _horizon_to_bars(label_horizon: str | pd.Timedelta | None, index: pd.Index) -> int:
    if label_horizon in (None, ""):
        return 0
    horizon_delta = _as_timedelta(label_horizon)
    bar_delta = _infer_bar_delta(index)
    return max(int(math.ceil(horizon_delta / bar_delta)), 0)


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
    embargo_bars: int = 0,
    purge_bars: int = 0,
    label_horizon: str | pd.Timedelta | None = None,
) -> list[Fold]:
    """Generate walk-forward folds with optional purge/embargo leakage controls."""
    index = pd.Index(sorted(pd.to_datetime(list(indices))), name="timestamp")
    folds: list[Fold] = []
    previous_tests = pd.Index([])
    inferred_horizon_bars = _horizon_to_bars(label_horizon, index)
    resolved_purge_bars = max(int(purge_bars), inferred_horizon_bars)
    resolved_embargo_bars = max(int(embargo_bars), 0)

    for train_start, train_end, test_start, test_end in windows:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)

        train_idx = index[(index >= train_start_ts) & (index <= train_end_ts)]
        test_idx = index[(index >= test_start_ts) & (index <= test_end_ts)]

        if (resolved_embargo_bars > 0 or resolved_purge_bars > 0) and not test_idx.empty:
            first_test_bar = test_idx[0]
            first_test_pos = int(index.searchsorted(first_test_bar))
            exclusion_bars = resolved_embargo_bars + resolved_purge_bars
            cutoff_pos = max(first_test_pos - exclusion_bars, 0)
            if cutoff_pos == 0:
                train_idx = train_idx[:0]
            else:
                train_cutoff = index[cutoff_pos - 1]
                train_idx = train_idx[train_idx <= train_cutoff]

        if not previous_tests.empty:
            train_idx = train_idx.difference(previous_tests)

        folds.append(
            Fold(
                train=train_idx,
                test=test_idx,
                embargo_bars=resolved_embargo_bars,
                purge_bars=resolved_purge_bars,
            )
        )
        previous_tests = previous_tests.union(test_idx)

    return folds
