from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pd_testing

from alpha.io.duck import query
from alpha.io.fs import read_parquet, write_parquet


def test_parquet_roundtrip_preserves_schema(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="int64"),
            "price": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "flag": pd.Series([True, False, True], dtype="bool"),
            "name": pd.Series(["alpha", "beta", "gamma"], dtype="object"),
            "event_at": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
                utc=False,
            ),
        }
    )

    target = tmp_path / "roundtrip.parquet"
    write_parquet(frame, target)

    reloaded = read_parquet(target)
    pd_testing.assert_frame_equal(reloaded, frame)


def test_partitioned_roundtrip_with_filters(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3, 4], dtype="int64"),
            "value": pd.Series([10.0, 20.0, 30.5, 40.0], dtype="float64"),
            "year": pd.Series([2022, 2023, 2023, 2024], dtype="int32"),
        }
    )
    target = tmp_path / "partitioned"

    write_parquet(frame, target, partition_cols=["year"])

    reloaded = read_parquet(target).sort_values("id").reset_index(drop=True)
    pd_testing.assert_frame_equal(reloaded, frame.sort_values("id").reset_index(drop=True))

    filtered = read_parquet(target, filters=[("year", "=", 2023)])
    expected_filtered = frame[frame["year"] == 2023].sort_values("id").reset_index(drop=True)
    pd_testing.assert_frame_equal(
        filtered.sort_values("id").reset_index(drop=True),
        expected_filtered,
    )


def test_duck_query_returns_dataframe() -> None:
    result = query(
        "SELECT $value::INTEGER AS value, $flag::BOOLEAN AS flag",
        value=42,
        flag=True,
    )

    assert list(result.columns) == ["value", "flag"]
    assert result.loc[0, "value"] == 42
    assert bool(result.loc[0, "flag"]) is True
