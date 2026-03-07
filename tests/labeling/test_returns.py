from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.labeling.returns import forward_return, parse_horizon

pytestmark = pytest.mark.behavioral

def test_forward_return_hourly_alignment() -> None:
    index = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 02:00:00",
        "2024-01-01 03:00:00",
    ])
    prices = pd.DataFrame(
        {
            "close": [100.0, 110.0, 105.0, 120.0],
        },
        index=index,
    )

    expected = pd.Series(
        [0.10, -0.0454545, 0.1428571, float("nan")],
        index=index,
        name="fwd_return_1h",
    )

    result = forward_return(prices, "1h")
    pd_testing.assert_series_equal(
        result,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_return_gap_results_nan() -> None:
    index = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 03:00:00",
    ])
    prices = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=index)

    result = forward_return(prices, "1h")
    expected = pd.Series(
        [0.10, float("nan"), float("nan")],
        index=index,
        name="fwd_return_1h",
    )
    pd_testing.assert_series_equal(result, expected)


def test_forward_return_invalid_horizon() -> None:
    prices = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-01-01")])
    with pytest.raises(ValueError):
        forward_return(prices, "0h")


def test_parse_horizon_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        parse_horizon("bad")


def test_parse_horizon_returns_positive_timedelta() -> None:
    assert parse_horizon("90m") == pd.Timedelta(minutes=90)


def test_forward_return_requires_sorted_unique_index() -> None:
    unsorted = pd.DataFrame(
        {"close": [101.0, 100.0]},
        index=pd.to_datetime(["2024-01-01 01:00:00", "2024-01-01 00:00:00"]),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        forward_return(unsorted, "1h")

    duplicate = pd.DataFrame(
        {"close": [101.0, 100.0]},
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
    )
    with pytest.raises(ValueError, match="unique"):
        forward_return(duplicate, "1h")


def test_forward_return_requires_close_column() -> None:
    with pytest.raises(KeyError):
        forward_return(pd.DataFrame({"open": [1]}), "1h")
