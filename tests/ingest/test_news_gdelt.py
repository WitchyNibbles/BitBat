from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pd_testing
import pytest

try:  # pragma: no cover - dependency guard
    import vaderSentiment  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("vaderSentiment not installed", allow_module_level=True)

from bitbat.features.sentiment import score_vader
from bitbat.ingest.news_gdelt import fetch
from bitbat.io.fs import read_parquet


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - compatibility shim
        if self.status_code >= 400:
            msg = f"GDELT request failed ({self.status_code})"
            raise RuntimeError(msg)


class FakeSession:
    def __init__(self, responses: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs.get("params", {})
        key = (params["startdatetime"], params["enddatetime"])
        self.calls.append(key)
        payload = self.responses.get(key, {"articles": []})
        return FakeResponse(payload)

    def close(self) -> None:
        return None


def _dataset(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    for column in ("year", "month", "day", "hour"):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    return frame.sort_values("published_utc").reset_index(drop=True)


def test_fetch_gdelt_range_and_writes(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 2, 0, 0)

    responses = {
        ("20240101000000", "20240101010000"): {
            "articles": [
                {
                    "seendate": "20240101000000",
                    "title": "Bitcoin rallies",
                    "url": "http://example.com/a",
                    "sourceCommonName": "ExampleNews",
                    "language": "en",
                },
                {
                    "seendate": "20240101003000",
                    "title": "Crypto regulation update",
                    "url": "http://example.com/b",
                    "source": "AnotherSource",
                    "language": "en",
                },
            ]
        },
        ("20240101010000", "20240101020000"): {
            "articles": [
                {
                    "seendate": "20240101011500",
                    "title": "BTC price analysis",
                    "url": "https://example.net/c",
                    "sourceCommonName": "CryptoDaily",
                    "language": "en",
                },
                {
                    "seendate": "20240101013000",
                    "title": "Duplicate should be removed",
                    "url": "http://example.com/b",
                    "source": "AnotherSource",
                    "language": "en",
                },
            ]
        },
    }

    session = FakeSession(responses)
    output_root = tmp_path / "gdelt"

    frame = fetch(start, end, session=session, output_root=output_root)

    assert session.calls == [
        ("20240101000000", "20240101010000"),
        ("20240101010000", "20240101020000"),
    ]

    assert list(frame.columns) == [
        "published_utc",
        "title",
        "url",
        "source",
        "lang",
        "sentiment_score",
    ]
    assert len(frame) == 3
    assert frame["published_utc"].min() >= start
    assert frame["published_utc"].max() <= end
    assert frame["published_utc"].isna().sum() == 0
    assert frame["url"].str.startswith("http").all()

    target_path = output_root / "gdelt_crypto_1h.parquet"
    assert target_path.exists()
    stored = _dataset(target_path)
    pd_testing.assert_frame_equal(stored, frame)

    expected_sentiment = score_vader(frame["title"])
    pd_testing.assert_series_equal(frame["sentiment_score"], expected_sentiment)

    # Re-run to confirm idempotency when merging with existing dataset
    frame_repeat = fetch(start, end, session=session, output_root=output_root)
    pd_testing.assert_frame_equal(frame_repeat, frame)
    stored_repeat = _dataset(target_path)
    pd_testing.assert_frame_equal(stored_repeat, frame)
