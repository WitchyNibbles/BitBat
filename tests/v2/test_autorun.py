from __future__ import annotations

from datetime import UTC, datetime

from bitbat_v2.autorun import AutonomousPaperTrader
from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
from bitbat_v2.runtime import BitBatRuntime
from bitbat_v2.storage import RuntimeStore


def build_runtime(tmp_path, now_fn=None) -> BitBatRuntime:
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        autorun_enabled=True,
        autorun_interval_seconds=15,
        signal_threshold=0.001,
    )
    store = RuntimeStore(config.database_url)
    runtime = BitBatRuntime(store=store, config=config, now_fn=now_fn)
    runtime.initialize()
    return runtime


def test_autonomous_trader_processes_new_candle_once_and_skips_duplicate(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 6, tzinfo=UTC)
    runtime = build_runtime(tmp_path, now_fn=lambda: now)
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )

    class FakeCoinbaseClient:
        def fetch_candles(self, product_id: str, granularity_seconds: int, start, end):
            return [candle]

    trader = AutonomousPaperTrader(
        runtime=runtime,
        config=runtime.config,
        market_data_client=FakeCoinbaseClient(),
        now_fn=lambda: now,
    )

    first = trader.sync_once()
    event_count_after_first = runtime.store.count_events()
    second = trader.sync_once()

    assert first.status == "processed"
    assert first.outcome is not None
    assert first.outcome.decision.action == "buy"
    assert second.status == "skipped"
    assert second.reason == "duplicate candle"
    assert second.outcome is None
    assert runtime.store.count_events() == event_count_after_first

    snapshot = trader.snapshot()
    assert snapshot.enabled is True
    assert snapshot.last_cycle_status == "skipped"
    assert snapshot.last_processed_candle_start == candle.start
    assert snapshot.last_action == "buy"


def test_autonomous_trader_records_fetch_errors_without_mutating_runtime(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 6, tzinfo=UTC)
    runtime = build_runtime(tmp_path, now_fn=lambda: now)

    class ExplodingCoinbaseClient:
        def fetch_candles(self, product_id: str, granularity_seconds: int, start, end):
            raise RuntimeError("coinbase unavailable")

    trader = AutonomousPaperTrader(
        runtime=runtime,
        config=runtime.config,
        market_data_client=ExplodingCoinbaseClient(),
        now_fn=lambda: now,
    )

    result = trader.sync_once()

    assert result.status == "error"
    assert "coinbase unavailable" in (result.error or "")
    assert runtime.store.count_events() == 0

    snapshot = trader.snapshot()
    assert snapshot.last_cycle_status == "error"
    assert "coinbase unavailable" in (snapshot.last_error or "")
