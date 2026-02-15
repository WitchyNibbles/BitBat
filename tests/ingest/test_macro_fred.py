"""Tests for FRED macro data ingestion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from bitbat.ingest.macro_fred import _fetch_series, fetch_fred


class FakeResponse:
    """Minimal mock for requests.get return value."""

    status_code = 200

    def __init__(self, payload: dict) -> None:  # type: ignore[type-arg]
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:  # type: ignore[type-arg]
        return self._payload


def _sample_observations(series_id: str, n: int = 30) -> dict:  # type: ignore[type-arg]
    """Build a fake FRED API response with n daily observations."""
    obs = []
    for i in range(n):
        date = f"2024-01-{i + 1:02d}"
        value = str(round(5.0 + i * 0.01, 4)) if i % 7 != 0 else "."
        obs.append({"date": date, "value": value})
    return {"observations": obs}


def test_fetch_series_parses_observations(monkeypatch: object) -> None:
    import bitbat.ingest.macro_fred as mod

    payload = _sample_observations("DFF", 10)

    def mock_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr(mod.requests, "get", mock_get)  # type: ignore[attr-defined]

    df = _fetch_series("DFF", datetime(2024, 1, 1), datetime(2024, 1, 10))
    assert "date" in df.columns
    assert "value" in df.columns
    assert len(df) == 10
    # Missing values (.) should be None/NaN
    assert df["value"].isna().any()


def test_fetch_series_missing_marker_becomes_nan(monkeypatch: object) -> None:
    import bitbat.ingest.macro_fred as mod

    payload = {"observations": [{"date": "2024-01-01", "value": "."}]}

    def mock_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr(mod.requests, "get", mock_get)  # type: ignore[attr-defined]

    df = _fetch_series("DFF", datetime(2024, 1, 1), datetime(2024, 1, 1))
    assert pd.isna(df["value"].iloc[0])


def test_fetch_fred_writes_parquet(tmp_path: Path, monkeypatch: object) -> None:
    import bitbat.ingest.macro_fred as mod

    def mock_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(_sample_observations("TEST", 20))

    monkeypatch.setattr(mod.requests, "get", mock_get)  # type: ignore[attr-defined]

    frame = fetch_fred(
        series_ids={"DFF": "fed_funds_rate", "DGS10": "treasury_10y"},
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30),
        output_root=tmp_path,
    )

    assert len(frame) > 0
    parquet_file = tmp_path / "fred.parquet"
    assert parquet_file.exists()

    stored = pd.read_parquet(parquet_file)
    assert "date" in stored.columns or "fed_funds_rate" in stored.columns
