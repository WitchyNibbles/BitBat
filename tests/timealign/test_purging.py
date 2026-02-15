from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.timealign.purging import mask_future


def test_mask_future_includes_boundary_and_excludes_late_news() -> None:
    news = pd.Series([
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:59:59Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T01:00:01Z",
    ])
    bar_end = pd.Series([
        "2024-01-01T00:30:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T01:00:00Z",
    ])

    mask = mask_future(news, bar_end)
    expected = pd.Series([True, True, True, False])
    pd_testing.assert_series_equal(mask, expected)


def test_mask_future_requires_matching_lengths() -> None:
    news = pd.Series(["2024-01-01T00:00:00Z"])
    bar_end = pd.Series(
        ["2024-01-01T00:15:00Z", "2024-01-01T00:30:00Z"],
    )
    with pytest.raises(ValueError):
        mask_future(news, bar_end)


def test_mask_future_requires_series() -> None:
    with pytest.raises(TypeError):
        mask_future(["2024-01-01T00:00:00Z"], pd.Series(["2024-01-01T00:00:00Z"]))  # type: ignore[arg-type]
