from __future__ import annotations

from datetime import UTC, datetime
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
from bitbat.ingest.news_cryptocompare import fetch
from bitbat.io.fs import read_parquet


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp())


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeNonJsonResponse:
    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text
        self.headers: dict[str, str] = {"Content-Type": "text/html; charset=utf-8"}

    def json(self) -> dict[str, Any]:
        raise ValueError("not valid json")


class SequenceSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[int] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        params = kwargs.get("params", {})
        self.calls.append(int(params["lTs"]))
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse({"Response": "Success", "Data": []})

    def close(self) -> None:
        return None


def _dataset(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    for column in ("year", "month", "day", "hour"):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    return frame.sort_values("published_utc").reset_index(drop=True)


def test_fetch_cryptocompare_range_and_writes(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 3, 0, 0)
    output_root = tmp_path / "cryptocompare"

    page_one = {
        "Response": "Success",
        "Data": [
            {
                "published_on": _ts("2024-01-01T02:30:00"),
                "title": "Bitcoin rallies",
                "url": "http://example.com/a",
                "source_info": {"name": "CoinDesk"},
                "lang": "EN",
            },
            {
                "published_on": _ts("2024-01-01T01:15:00"),
                "title": "Crypto regulation update",
                "url": "http://example.com/b",
                "source_info": {"name": "TheBlock"},
                "lang": "EN",
            },
        ],
    }
    page_two = {
        "Response": "Success",
        "Data": [
            {
                "published_on": _ts("2024-01-01T00:45:00"),
                "title": "BTC market opens strong",
                "url": "https://example.net/c",
                "source_info": {"name": "CryptoDaily"},
                "lang": "EN",
            },
            {
                "published_on": _ts("2023-12-31T22:10:00"),
                "title": "Older out-of-range item",
                "url": "http://example.com/old",
                "source_info": {"name": "OldNews"},
                "lang": "EN",
            },
        ],
    }

    session = SequenceSession([FakeResponse(page_one), FakeResponse(page_two)])
    frame = fetch(start, end, session=session, output_root=output_root)

    assert len(session.calls) == 2
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
    assert frame["url"].str.startswith("http").all()

    target_path = output_root / "cryptocompare_btc_1h.parquet"
    assert target_path.exists()
    stored = _dataset(target_path)
    pd_testing.assert_frame_equal(stored, frame)

    expected_sentiment = score_vader(frame["title"])
    pd_testing.assert_series_equal(frame["sentiment_score"], expected_sentiment)


def test_fetch_retries_non_json_then_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    output_root = tmp_path / "cryptocompare"

    good_payload = {
        "Response": "Success",
        "Data": [
            {
                "published_on": _ts("2024-01-01T00:15:00"),
                "title": "Bitcoin rebounds",
                "url": "https://example.com/recover",
                "source_info": {"name": "RecoveryWire"},
                "lang": "EN",
            }
        ],
    }
    session = SequenceSession(
        [
            FakeNonJsonResponse("{html payload}"),
            FakeResponse(good_payload),
        ]
    )

    monkeypatch.setattr("bitbat.ingest.news_cryptocompare.time.sleep", lambda _: None)
    frame = fetch(start, end, session=session, output_root=output_root, retry_limit=1)

    assert len(session.calls) >= 2
    assert len(frame) == 1
    assert frame["url"].iloc[0] == "https://example.com/recover"
