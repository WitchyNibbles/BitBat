"""Rate limiter for API calls respecting free tier limits.

Tracks API usage per period and prevents exceeding limits.
State is persisted to a JSON file so it survives process restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class RateLimiter:
    """Track API usage and enforce rate limits.

    Usage::

        limiter = RateLimiter('newsapi', limit=100, period='day')
        if limiter.can_make_request():
            result = api_call()
            limiter.record_request()

    Args:
        service_name: Identifier for the API service.
        limit: Maximum number of requests allowed per period.
        period: Duration of one rate-limit window.  One of ``'minute'``,
            ``'hour'``, or ``'day'``.
        state_file: JSON file used to persist request history across
            restarts.  Defaults to ``data/<service_name>_rate_limit.json``.
    """

    def __init__(
        self,
        service_name: str,
        limit: int,
        period: str = "day",
        state_file: Path | None = None,
    ) -> None:
        self.service_name = service_name
        self.limit = limit
        self.period = period

        if state_file is None:
            state_file = Path("data") / f"{service_name}_rate_limit.json"

        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.requests: list[datetime] = []
        self._load_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load persisted request history from disk."""
        if not self.state_file.exists():
            return
        try:
            with self.state_file.open() as fh:
                data = json.load(fh)
            self.requests = [datetime.fromisoformat(ts) for ts in data.get("requests", [])]
        except Exception as exc:
            logger.error("Error loading rate-limit state for %s: %s", self.service_name, exc)
            self.requests = []

    def _save_state(self) -> None:
        """Persist the current request history to disk."""
        try:
            payload = {
                "service": self.service_name,
                "limit": self.limit,
                "period": self.period,
                "requests": [ts.isoformat() for ts in self.requests],
            }
            with self.state_file.open("w") as fh:
                json.dump(payload, fh)
        except Exception as exc:
            logger.error("Error saving rate-limit state for %s: %s", self.service_name, exc)

    def _get_period_delta(self) -> timedelta:
        """Return the timedelta that corresponds to *self.period*."""
        mapping = {
            "minute": timedelta(minutes=1),
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
        }
        if self.period not in mapping:
            raise ValueError(f"Invalid period '{self.period}'. Choose from: {list(mapping)}")
        return mapping[self.period]

    def _clean_old_requests(self) -> None:
        """Drop timestamps older than the current period window."""
        cutoff = datetime.now(UTC).replace(tzinfo=None) - self._get_period_delta()
        self.requests = [ts for ts in self.requests if ts > cutoff]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def can_make_request(self) -> bool:
        """Return ``True`` if another request is allowed within the current period."""
        self._clean_old_requests()
        return len(self.requests) < self.limit

    def record_request(self, timestamp: datetime | None = None) -> None:
        """Record that one request was made.

        Args:
            timestamp: Explicit timestamp of the request.  Defaults to *now* (UTC).
        """
        if timestamp is None:
            timestamp = datetime.now(UTC).replace(tzinfo=None)
        self.requests.append(timestamp)
        self._save_state()

    def requests_remaining(self) -> int:
        """Return the number of additional requests available in the current period."""
        self._clean_old_requests()
        return max(0, self.limit - len(self.requests))

    def time_until_reset(self) -> timedelta | None:
        """Return the time until the oldest request expires from the window.

        Returns ``None`` if no requests have been made yet.
        """
        self._clean_old_requests()
        if not self.requests:
            return None
        oldest = min(self.requests)
        reset_time = oldest + self._get_period_delta()
        remaining = reset_time - datetime.now(UTC).replace(tzinfo=None)
        return remaining if remaining.total_seconds() > 0 else timedelta(0)

    def get_status(self) -> dict:
        """Return a snapshot of the current rate-limit state."""
        self._clean_old_requests()
        until_reset = self.time_until_reset()
        return {
            "service": self.service_name,
            "limit": self.limit,
            "period": self.period,
            "requests_made": len(self.requests),
            "requests_remaining": self.requests_remaining(),
            "time_until_reset": str(until_reset) if until_reset is not None else None,
        }
