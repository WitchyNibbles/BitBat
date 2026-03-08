from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.labeling.targets import (
    classify,
    direction_from_prices,
    direction_from_returns,
)

pytestmark = pytest.mark.behavioral


def test_classify_threshold_edges() -> None:
    series = pd.Series([0.02, 0.01, 0.0, -0.01, -0.02])
    labels = classify(series, tau=0.01)
    expected = pd.Series(
        ["up", "flat", "flat", "flat", "down"],
        index=series.index,
        name="target",
        dtype="string",
    )
    pd_testing.assert_series_equal(labels, expected)


def test_classify_symmetry() -> None:
    values = pd.Series([0.015, -0.015, 0.002, -0.002])
    labels = classify(values, tau=0.01)
    expected = pd.Series(
        ["up", "down", "flat", "flat"],
        index=values.index,
        name="target",
        dtype="string",
    )
    pd_testing.assert_series_equal(labels, expected)


def test_classify_negative_tau_rejected() -> None:
    with pytest.raises(ValueError):
        classify(pd.Series([0.0]), tau=-0.1)


def test_direction_from_returns_defaults_to_label_column_name() -> None:
    returns = pd.Series([0.03, -0.02, 0.0], index=[1, 2, 3])
    labels = direction_from_returns(returns, tau=0.01)
    expected = pd.Series(
        ["up", "down", "flat"],
        index=returns.index,
        name="label",
        dtype="string",
    )
    pd_testing.assert_series_equal(labels, expected)


def test_direction_from_prices_uses_forward_return_horizon_alignment() -> None:
    index = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 03:00:00",
    ])
    prices = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=index)

    labels = direction_from_prices(prices, horizon="1h", tau=0.0)
    expected = pd.DataFrame(
        {
            "r_forward": [0.10, float("nan"), float("nan")],
            "label": pd.Series(["up", pd.NA, pd.NA], index=index, dtype="string"),
        },
        index=index,
    )

    pd_testing.assert_frame_equal(labels, expected)
