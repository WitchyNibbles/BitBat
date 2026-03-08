from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.labeling.triple_barrier import triple_barrier

pytestmark = pytest.mark.behavioral


def test_triple_barrier_hits_take_profit_stop_loss_and_timeout() -> None:
    index = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h")
    prices = pd.DataFrame({"close": [100.0, 103.0, 104.0, 101.0]}, index=index)

    result = triple_barrier(
        prices,
        horizon="3h",
        take_profit=0.02,
        stop_loss=0.02,
    )
    expected = pd.DataFrame(
        {
            "r_forward": [0.03, -0.0194174757, -0.0288461538, float("nan")],
            "label": pd.Series(
                ["take_profit", "timeout", "stop_loss", pd.NA],
                index=index,
                dtype="string",
            ),
        },
        index=index,
    )

    pd_testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-8)


def test_triple_barrier_uses_first_barrier_hit_ordering() -> None:
    index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="1h")
    prices = pd.DataFrame({"close": [100.0, 97.0, 103.0]}, index=index)

    result = triple_barrier(
        prices,
        horizon="2h",
        take_profit=0.02,
        stop_loss=0.02,
    )

    assert result.loc[index[0], "label"] == "stop_loss"
    assert result.loc[index[0], "r_forward"] == pytest.approx(-0.03)


def test_triple_barrier_rejects_invalid_thresholds() -> None:
    prices = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01 00:00:00", periods=2, freq="1h"),
    )
    with pytest.raises(ValueError, match="take_profit"):
        triple_barrier(prices, horizon="1h", take_profit=0.0, stop_loss=0.01)
    with pytest.raises(ValueError, match="stop_loss"):
        triple_barrier(prices, horizon="1h", take_profit=0.01, stop_loss=0.0)


def test_triple_barrier_requires_close_column() -> None:
    with pytest.raises(KeyError):
        triple_barrier(
            pd.DataFrame({"open": [1.0]}),
            horizon="1h",
            take_profit=0.01,
            stop_loss=0.01,
        )
