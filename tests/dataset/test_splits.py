from __future__ import annotations

import pandas as pd
import pytest

from bitbat.dataset.splits import walk_forward

pytestmark = pytest.mark.behavioral


def test_walk_forward_embargo_removes_overlap() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="1h")
    folds = walk_forward(
        indices=idx,
        windows=[
            ("2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05"),
            ("2024-01-02", "2024-01-04", "2024-01-05", "2024-01-06"),
        ],
        embargo_bars=2,
    )

    assert len(folds) == 2
    first = folds[0]
    second = folds[1]
    assert first.test[0] not in first.train
    assert second.test[0] not in second.train
    assert all(ts < second.test[0] for ts in second.train)


def test_walk_forward_no_leakage_across_folds() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="1h")
    folds = walk_forward(
        indices=idx,
        windows=[
            ("2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"),
            ("2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"),
        ],
        embargo_bars=1,
    )

    assert len(folds) == 2
    overlap = set(folds[0].test).intersection(set(folds[1].train))
    assert not overlap


def test_walk_forward_purge_bars_remove_training_tail() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="1h")
    folds = walk_forward(
        indices=idx,
        windows=[
            (
                "2024-01-01 00:00:00",
                "2024-01-01 15:00:00",
                "2024-01-01 16:00:00",
                "2024-01-01 20:00:00",
            ),
        ],
        embargo_bars=0,
        purge_bars=3,
    )

    assert len(folds) == 1
    assert folds[0].train.max() == idx[12]
    assert len(folds[0].train) == 13


def test_walk_forward_label_horizon_infers_purge_window() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="1h")
    folds = walk_forward(
        indices=idx,
        windows=[
            (
                "2024-01-01 00:00:00",
                "2024-01-01 15:00:00",
                "2024-01-01 16:00:00",
                "2024-01-01 20:00:00",
            ),
        ],
        embargo_bars=0,
        purge_bars=0,
        label_horizon="4h",
    )

    assert len(folds) == 1
    assert folds[0].train.max() == idx[11]
    assert len(folds[0].train) == 12
