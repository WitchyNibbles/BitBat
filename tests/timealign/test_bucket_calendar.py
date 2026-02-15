from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.timealign.bucket import to_bar
from bitbat.timealign.calendar import ensure_utc


def test_to_bar_floors_expected_intervals() -> None:
    base = pd.date_range("2024-01-01 00:45:00+00:00", periods=10, freq="53min", tz="UTC")
    series = pd.Series(base.tz_convert("UTC"))

    hourly = to_bar(series, "1h")
    expected_hourly = pd.Series(
        base.floor("1h").tz_localize(None).to_numpy(),
        index=series.index,
        name=series.name,
    )
    pd_testing.assert_series_equal(hourly, expected_hourly)

    four_hour = to_bar(series, "4h")
    expected_four_hour = pd.Series(
        base.floor("4h").tz_localize(None).to_numpy(),
        index=series.index,
        name=series.name,
    )
    pd_testing.assert_series_equal(four_hour, expected_four_hour)

    daily = to_bar(series, "24h")
    expected_daily = pd.Series(
        base.floor("24h").tz_localize(None).to_numpy(),
        index=series.index,
        name=series.name,
    )
    pd_testing.assert_series_equal(daily, expected_daily)


def test_to_bar_rejects_unknown_frequency() -> None:
    series = pd.Series(pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))
    with pytest.raises(ValueError):
        to_bar(series, "15m")  # type: ignore[arg-type]


def test_ensure_utc_normalises_column() -> None:
    frame = pd.DataFrame({
        "timestamp": [
            "2024-01-01T00:00:00Z",
            pd.Timestamp("2024-01-01 01:15:00", tz="UTC"),
            pd.Timestamp("2024-01-01 02:30:00"),
        ],
        "value": [1, 2, 3],
    })

    normalised = ensure_utc(frame, "timestamp")
    expected = pd.Series(
        pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 01:15:00", "2024-01-01 02:30:00"],
            utc=True,
        )
        .tz_localize(None)
        .to_numpy(),
        index=normalised.index,
        name="timestamp",
    )
    pd_testing.assert_series_equal(normalised["timestamp"], expected, check_index_type=False)

    # Original left untouched
    assert frame["timestamp"].iloc[0] == "2024-01-01T00:00:00Z"


def test_ensure_utc_invalid_column_raises() -> None:
    frame = pd.DataFrame({"value": [1]})
    with pytest.raises(KeyError):
        ensure_utc(frame, "missing")


def test_ensure_utc_invalid_value_raises() -> None:
    frame = pd.DataFrame({"timestamp": ["invalid timestamp"]})
    with pytest.raises(ValueError):
        ensure_utc(frame, "timestamp")
