from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bitbat.ingest.prices import fetch_yf

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_fetch_yf_btcusd_hourly(tmp_path: Path) -> None:
    start = datetime(2017, 1, 1)
    frame = fetch_yf("BTC-USD", "1h", start, output_root=tmp_path)

    expected_columns = ["timestamp_utc", "open", "high", "low", "close", "volume", "source"]
    assert list(frame.columns) == expected_columns
    assert len(frame) > 0
    assert frame["timestamp_utc"].is_monotonic_increasing
    assert frame["timestamp_utc"].is_unique
    assert str(frame["timestamp_utc"].dtype) == "datetime64[ns]"
    assert (frame["source"] == "yfinance").all()

    output_path = tmp_path / "btcusd_yf_1h.parquet"
    assert output_path.exists()
    if output_path.is_dir():
        partition_dirs = sorted(path for path in output_path.iterdir() if path.is_dir())
        assert (
            partition_dirs
        ), "Expected parquet dataset partitions when storage is directory-based."
        assert all(path.name.startswith("year=") for path in partition_dirs)
    else:
        assert output_path.is_file()

    dataset = pd.read_parquet(output_path)
    assert len(dataset) == len(frame)
