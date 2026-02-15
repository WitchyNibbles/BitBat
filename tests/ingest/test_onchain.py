"""Tests for blockchain.info on-chain data ingestion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from bitbat.ingest.onchain import _fetch_metric, fetch_blockchain_info


class FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:  # type: ignore[type-arg]
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:  # type: ignore[type-arg]
        return self._payload


def _sample_chart(n: int = 30) -> dict:  # type: ignore[type-arg]
    """Build a fake blockchain.info chart response."""
    base_ts = int(datetime(2024, 1, 1).timestamp())
    values = [{"x": base_ts + i * 86400, "y": 100.0 + i} for i in range(n)]
    return {"values": values}


def test_fetch_metric_parses_values(monkeypatch: object) -> None:
    import bitbat.ingest.onchain as mod

    def mock_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(_sample_chart(10))

    monkeypatch.setattr(mod.requests, "get", mock_get)  # type: ignore[attr-defined]

    df = _fetch_metric("hash-rate", datetime(2024, 1, 1), datetime(2024, 1, 10))
    assert "date" in df.columns
    assert "value" in df.columns
    assert len(df) == 10


def test_fetch_blockchain_info_writes_parquet(tmp_path: Path, monkeypatch: object) -> None:
    import bitbat.ingest.onchain as mod

    def mock_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(_sample_chart(20))

    monkeypatch.setattr(mod.requests, "get", mock_get)  # type: ignore[attr-defined]

    frame = fetch_blockchain_info(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30),
        metrics={"hash-rate": "hashrate", "n-transactions": "tx_count"},
        output_root=tmp_path,
    )

    assert len(frame) > 0
    parquet_file = tmp_path / "blockchain_info.parquet"
    assert parquet_file.exists()

    stored = pd.read_parquet(parquet_file)
    assert "date" in stored.columns or "hashrate" in stored.columns
