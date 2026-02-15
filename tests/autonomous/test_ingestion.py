"""Integration tests for SESSION 4 ingestion components.

Tests that can run without live API access use mocking or offline fixtures.
Tests that hit the real network are marked ``live`` and skipped by default.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bitbat.autonomous.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Rate-limiter tests (no network, no disk after cleanup)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_rate_limiter(tmp_path: Path) -> RateLimiter:
    state = tmp_path / "test_rate_limit.json"
    return RateLimiter("test", limit=5, period="minute", state_file=state)


class TestRateLimiter:
    def test_allows_requests_within_limit(self, tmp_rate_limiter: RateLimiter) -> None:
        for _ in range(5):
            assert tmp_rate_limiter.can_make_request()
            tmp_rate_limiter.record_request()

    def test_blocks_request_over_limit(self, tmp_rate_limiter: RateLimiter) -> None:
        for _ in range(5):
            tmp_rate_limiter.record_request()
        assert not tmp_rate_limiter.can_make_request()

    def test_requests_remaining(self, tmp_rate_limiter: RateLimiter) -> None:
        assert tmp_rate_limiter.requests_remaining() == 5
        tmp_rate_limiter.record_request()
        assert tmp_rate_limiter.requests_remaining() == 4

    def test_state_persisted_to_disk(self, tmp_path: Path) -> None:
        state_file = tmp_path / "persistent.json"
        limiter = RateLimiter("persist_test", limit=10, period="hour", state_file=state_file)
        limiter.record_request()

        # Load a fresh instance from the same file.
        limiter2 = RateLimiter("persist_test", limit=10, period="hour", state_file=state_file)
        assert limiter2.requests_remaining() == 9

    def test_old_requests_pruned(self, tmp_path: Path) -> None:
        """Requests older than the period window should be discarded."""
        state_file = tmp_path / "prune_test.json"
        limiter = RateLimiter("prune", limit=2, period="minute", state_file=state_file)

        # Inject a stale request (2 minutes ago).
        stale_ts = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=2)
        limiter.requests = [stale_ts]
        limiter._save_state()

        # Should be pruned on the next check, freeing the slot.
        assert limiter.can_make_request()
        assert limiter.requests_remaining() == 2

    def test_get_status_structure(self, tmp_rate_limiter: RateLimiter) -> None:
        status = tmp_rate_limiter.get_status()
        assert "service" in status
        assert "limit" in status
        assert "requests_remaining" in status
        assert status["limit"] == 5

    def test_time_until_reset_none_when_empty(self, tmp_rate_limiter: RateLimiter) -> None:
        assert tmp_rate_limiter.time_until_reset() is None

    def test_time_until_reset_positive_after_request(self, tmp_rate_limiter: RateLimiter) -> None:
        tmp_rate_limiter.record_request()
        delta = tmp_rate_limiter.time_until_reset()
        assert delta is not None
        assert delta.total_seconds() > 0

    def test_invalid_period_raises(self, tmp_path: Path) -> None:
        limiter = RateLimiter("bad", limit=1, period="fortnight", state_file=tmp_path / "x.json")
        with pytest.raises(ValueError, match="Invalid period"):
            limiter.can_make_request()


# ---------------------------------------------------------------------------
# Price ingestion tests (mocked yfinance)
# ---------------------------------------------------------------------------


class TestPriceIngestionService:
    def _make_fake_yf_history(self) -> pd.DataFrame:
        """Minimal yfinance-style DataFrame."""
        times = pd.date_range("2024-01-15 10:00", periods=3, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "Open": [40_000.0, 40_100.0, 40_200.0],
                "High": [40_500.0, 40_600.0, 40_700.0],
                "Low": [39_900.0, 40_000.0, 40_100.0],
                "Close": [40_100.0, 40_200.0, 40_300.0],
                "Volume": [1_000.0, 1_100.0, 1_200.0],
            },
            index=pd.Index(times, name="Datetime"),
        )

    def test_fetch_latest_stores_parquet(self, tmp_path: Path) -> None:
        from bitbat.autonomous.price_ingestion import PriceIngestionService

        fake_history = self._make_fake_yf_history()
        service = PriceIngestionService(symbol="BTC-USD", interval="1h", data_dir=tmp_path)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_history

        with patch("bitbat.autonomous.price_ingestion.yf.Ticker", return_value=mock_ticker):
            count = service.fetch_latest()

        assert count == 3
        parquet_files = list(service.prices_dir.glob("**/*.parquet"))
        assert len(parquet_files) >= 1

        df = pd.read_parquet(parquet_files[0])
        assert "timestamp_utc" in df.columns
        assert "close" in df.columns

    def test_fetch_latest_deduplicates(self, tmp_path: Path) -> None:
        from bitbat.autonomous.price_ingestion import PriceIngestionService

        fake_history = self._make_fake_yf_history()
        service = PriceIngestionService(symbol="BTC-USD", interval="1h", data_dir=tmp_path)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_history

        with patch("bitbat.autonomous.price_ingestion.yf.Ticker", return_value=mock_ticker):
            service.fetch_latest()
            # Fetch again — should not add duplicates.
            mock_ticker.history.return_value = fake_history
            service.fetch_latest()

        # Second fetch: the service advances start past the last timestamp,
        # so yfinance returns the same rows but they are deduped on write.
        parquet_files = list(service.prices_dir.glob("**/*.parquet"))
        assert len(parquet_files) >= 1
        total_rows = sum(len(pd.read_parquet(f)) for f in parquet_files)
        assert total_rows == 3  # No duplicates.

    def test_get_last_timestamp_none_when_empty(self, tmp_path: Path) -> None:
        from bitbat.autonomous.price_ingestion import PriceIngestionService

        service = PriceIngestionService(data_dir=tmp_path)
        assert service._get_last_timestamp() is None

    def test_fetch_with_retry_propagates_after_exhaustion(self, tmp_path: Path) -> None:
        from bitbat.autonomous.price_ingestion import PriceIngestionService

        service = PriceIngestionService(data_dir=tmp_path)

        with (
            patch.object(service, "fetch_latest", side_effect=RuntimeError("network down")),
            pytest.raises(RuntimeError, match="network down"),
        ):
            service.fetch_with_retry(max_retries=2)


# ---------------------------------------------------------------------------
# News ingestion tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestNewsIngestionService:
    def _fake_cryptocompare_response(self) -> dict:
        return {
            "Data": [
                {
                    "published_on": int(datetime(2024, 1, 15, 12, 0).timestamp()),
                    "title": "Bitcoin hits new high",
                    "url": "https://example.com/btc-high",
                    "source": "CryptoTimes",
                }
            ]
        }

    def test_fetch_cryptocompare_returns_articles(self, tmp_path: Path) -> None:
        from bitbat.autonomous.news_ingestion import NewsIngestionService

        service = NewsIngestionService(data_dir=tmp_path)
        fake_response = MagicMock()
        fake_response.json.return_value = self._fake_cryptocompare_response()
        fake_response.raise_for_status.return_value = None

        with patch("bitbat.autonomous.news_ingestion.requests.get", return_value=fake_response):
            articles = service.fetch_cryptocompare(max_results=5)

        assert len(articles) == 1
        assert articles[0]["title"] == "Bitcoin hits new high"
        assert "sentiment_score" in articles[0]

    def test_fetch_newsapi_skipped_without_key(self, tmp_path: Path) -> None:
        from bitbat.autonomous.news_ingestion import NewsIngestionService

        service = NewsIngestionService(data_dir=tmp_path)
        service.newsapi_key = None  # Ensure no key.
        articles = service.fetch_newsapi()
        assert articles == []

    def test_fetch_all_sources_stores_parquet(self, tmp_path: Path) -> None:
        from bitbat.autonomous.news_ingestion import NewsIngestionService

        service = NewsIngestionService(data_dir=tmp_path)
        fake_response = MagicMock()
        fake_response.json.return_value = self._fake_cryptocompare_response()
        fake_response.raise_for_status.return_value = None

        with patch("bitbat.autonomous.news_ingestion.requests.get", return_value=fake_response):
            count = service.fetch_all_sources()

        assert count >= 1
        parquet_files = list(service.news_dir.glob("**/*.parquet"))
        assert len(parquet_files) >= 1
        df = pd.read_parquet(parquet_files[0])
        assert "published_utc" in df.columns
        assert "sentiment_score" in df.columns

    def test_fetch_all_sources_deduplicates_by_url(self, tmp_path: Path) -> None:
        from bitbat.autonomous.news_ingestion import NewsIngestionService

        service = NewsIngestionService(data_dir=tmp_path)
        fake_response = MagicMock()
        # Two identical articles with same URL.
        raw = self._fake_cryptocompare_response()
        raw["Data"] = raw["Data"] * 2  # Duplicate entries.
        fake_response.json.return_value = raw
        fake_response.raise_for_status.return_value = None

        with patch("bitbat.autonomous.news_ingestion.requests.get", return_value=fake_response):
            count = service.fetch_all_sources()

        assert count == 1  # Deduped.

    def test_fetch_all_sources_empty_when_all_fail(self, tmp_path: Path) -> None:
        from bitbat.autonomous.news_ingestion import NewsIngestionService

        service = NewsIngestionService(data_dir=tmp_path)
        service.newsapi_key = None

        with patch(
            "bitbat.autonomous.news_ingestion.requests.get",
            side_effect=OSError("network error"),
        ):
            count = service.fetch_all_sources()

        assert count == 0
