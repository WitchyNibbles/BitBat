"""Autonomous paper-trading loop for BitBat v2."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .coinbase import CoinbaseMarketDataClient
from .config import BitBatV2Config
from .domain import RuntimeOutcome, utc_now
from .runtime import BitBatRuntime


@dataclass(frozen=True)
class AutoSyncResult:
    status: str
    reason: str | None = None
    error: str | None = None
    outcome: RuntimeOutcome | None = None


@dataclass(frozen=True)
class AutoSyncSnapshot:
    enabled: bool
    interval_seconds: int
    running: bool
    last_cycle_status: str | None = None
    last_cycle_started_at: datetime | None = None
    last_cycle_completed_at: datetime | None = None
    last_error: str | None = None
    last_processed_candle_start: datetime | None = None
    last_action: str | None = None


class AutonomousPaperTrader:
    """Poll Coinbase and feed new candles into the paper-only runtime."""

    def __init__(
        self,
        runtime: BitBatRuntime,
        config: BitBatV2Config,
        market_data_client: CoinbaseMarketDataClient | Any,
        now_fn=None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.market_data_client = market_data_client
        self.now_fn = now_fn or utc_now
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_cycle_status: str | None = None
        self._last_cycle_started_at: datetime | None = None
        self._last_cycle_completed_at: datetime | None = None
        self._last_error: str | None = None
        self._last_processed_candle_start: datetime | None = None
        self._last_action: str | None = None

    def snapshot(self) -> AutoSyncSnapshot:
        return AutoSyncSnapshot(
            enabled=self.config.autorun_enabled,
            interval_seconds=self.config.autorun_interval_seconds,
            running=self._running,
            last_cycle_status=self._last_cycle_status,
            last_cycle_started_at=self._last_cycle_started_at,
            last_cycle_completed_at=self._last_cycle_completed_at,
            last_error=self._last_error,
            last_processed_candle_start=self._last_processed_candle_start,
            last_action=self._last_action,
        )

    def sync_once(self) -> AutoSyncResult:
        self._last_cycle_started_at = self.now_fn().astimezone(UTC)
        try:
            end = self.now_fn().astimezone(UTC)
            candles = self.market_data_client.fetch_candles(
                product_id=self.config.product_id,
                granularity_seconds=self.config.granularity_seconds,
                start=end - timedelta(seconds=self.config.granularity_seconds * 3),
                end=end,
            )
            if not candles:
                raise RuntimeError("Coinbase returned no candles for the requested window")
            outcome = self.runtime.process_candle(candles[-1])
            self._last_cycle_completed_at = self.now_fn().astimezone(UTC)
            self._last_error = None
            self._last_processed_candle_start = candles[-1].start.astimezone(UTC)
            if outcome.decision.reason == "duplicate candle":
                self._last_cycle_status = "skipped"
                return AutoSyncResult(status="skipped", reason="duplicate candle")
            self._last_action = outcome.decision.action
            self._last_cycle_status = "processed"
            return AutoSyncResult(status="processed", outcome=outcome)
        except Exception as exc:
            self._last_cycle_completed_at = self.now_fn().astimezone(UTC)
            self._last_cycle_status = "error"
            self._last_error = str(exc)
            return AutoSyncResult(status="error", error=str(exc))

    async def run_forever(self) -> None:
        self._running = True
        try:
            while True:
                self.sync_once()
                await asyncio.sleep(self.config.autorun_interval_seconds)
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        if self._task is None:
            self._running = False
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._running = False
