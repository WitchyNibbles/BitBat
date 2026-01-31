from __future__ import annotations

import pandas as pd

from alpha.dataset.splits import walk_forward


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
