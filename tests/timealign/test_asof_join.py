from __future__ import annotations

import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.timealign.asof import align_features_asof, ensure_no_future_matches

pytestmark = pytest.mark.behavioral

def test_align_features_asof_is_boundary_inclusive() -> None:
    target = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h", name="timestamp_utc")
    source = pd.DataFrame(
        {"macro_rate": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 03:00:00"]),
    )

    aligned = align_features_asof(target, source, source_name="macro")

    expected = pd.DataFrame(
        {"macro_rate": [1.0, 1.0, 1.0, 2.0]},
        index=target,
    )
    pd_testing.assert_frame_equal(aligned, expected, check_freq=False)


def test_align_features_asof_handles_irregular_gaps_without_future_fill() -> None:
    target = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h", name="timestamp_utc")
    source = pd.DataFrame(
        {"onchain_hashrate": [10.0]},
        index=pd.to_datetime(["2024-01-01 02:00:00"]),
    )

    aligned = align_features_asof(target, source, source_name="onchain")

    expected = pd.DataFrame(
        {"onchain_hashrate": [float("nan"), float("nan"), 10.0, 10.0]},
        index=target,
    )
    pd_testing.assert_frame_equal(aligned, expected, check_freq=False)


def test_align_features_asof_requires_sorted_unique_indices() -> None:
    target = pd.to_datetime(["2024-01-01 01:00:00", "2024-01-01 00:00:00"])
    source = pd.DataFrame(
        {"x": [1.0]},
        index=pd.to_datetime(["2024-01-01 00:00:00"]),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        align_features_asof(target, source, source_name="macro")


def test_ensure_no_future_matches_rejects_future_rows() -> None:
    target = pd.Series(pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]))
    source = pd.Series(pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 02:00:00"]))
    with pytest.raises(ValueError, match="Future sentiment values detected"):
        ensure_no_future_matches(target, source, source_name="sentiment")
