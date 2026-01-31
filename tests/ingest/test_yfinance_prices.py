from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bitbat.ingest.prices import fetch_yf


@pytest.mark.slow
def test_fetch_yf_btcusd_hourly() -> None:
    output_dir = Path("data/raw/prices/btcusd_yf_1h.parquet")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    start = datetime(2017, 1, 1)
    frame = fetch_yf("BTC-USD", "1h", start)

    expected_columns = ["timestamp_utc", "open", "high", "low", "close", "volume", "source"]
    assert list(frame.columns) == expected_columns
    assert len(frame) > 0
    assert frame["timestamp_utc"].is_monotonic_increasing
    assert frame["timestamp_utc"].is_unique
    assert str(frame["timestamp_utc"].dtype) == "datetime64[ns]"
    assert (frame["source"] == "yfinance").all()

    assert output_dir.exists()
    partition_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert partition_dirs, "Expected partitioned parquet output by year."
    assert all(path.name.startswith("year=") for path in partition_dirs)

    dataset = pd.read_parquet(output_dir)
    assert len(dataset) == len(frame)
