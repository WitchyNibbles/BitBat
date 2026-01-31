from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.labeling.returns import forward_return


def test_forward_return_hourly_alignment() -> None:
    index = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
        ]
    )
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
    index = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 03:00:00",
        ]
    )
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


def test_forward_return_requires_close_column() -> None:
    with pytest.raises(KeyError):
        forward_return(pd.DataFrame({"open": [1]}), "1h")
