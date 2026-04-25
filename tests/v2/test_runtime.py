from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle, RuntimeEvent
from bitbat_v2.runtime import BitBatRuntime, EventBroker
from bitbat_v2.storage import RuntimeStore


def build_runtime(tmp_path, now_fn=None) -> BitBatRuntime:
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        max_position_size_btc=0.05,
        order_size_btc=0.02,
        stale_after_seconds=120,
        signal_threshold=0.001,
    )
    store = RuntimeStore(config.database_url)
    runtime = BitBatRuntime(store=store, config=config, now_fn=now_fn)
    runtime.initialize()
    return runtime


def test_process_candle_generates_signal_order_and_portfolio(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
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

    outcome = runtime.process_candle(candle)

    assert outcome.signal.direction == "buy"
    assert outcome.decision.action == "buy"
    assert outcome.order is not None
    assert outcome.order.status == "filled"
    assert outcome.portfolio.position_qty == runtime.config.order_size_btc
    assert outcome.portfolio.cash < runtime.config.starting_cash_usd
    assert any(reason.startswith("score=") for reason in outcome.signal.reasons)
    assert "confirmed positive trend and score above threshold" in outcome.decision.reason

    event_types = [event.event_type for event in runtime.store.list_events(limit=10)]
    assert "candle.closed" in event_types
    assert "features.computed" in event_types
    assert "signal.generated" in event_types
    assert "decision.made" in event_types
    assert "order.paper_filled" in event_types
    assert "portfolio.updated" in event_types


def test_pause_blocks_execution_and_emits_alert(tmp_path) -> None:
    runtime = build_runtime(tmp_path)
    runtime.pause_trading()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 5, tzinfo=UTC),
        open=100_000.0,
        high=100_200.0,
        low=99_900.0,
        close=100_150.0,
        volume=8.2,
    )

    outcome = runtime.process_candle(candle)

    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert runtime.store.get_orders(limit=10) == []
    assert runtime.store.get_control_state().trading_paused is True


def test_stale_candle_does_not_place_order(tmp_path) -> None:
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        stale_after_seconds=60,
        signal_threshold=0.001,
    )
    store = RuntimeStore(config.database_url)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    runtime = BitBatRuntime(store=store, config=config, now_fn=lambda: now)
    runtime.initialize()
    stale_candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=now - timedelta(minutes=15),
        open=100_000.0,
        high=100_200.0,
        low=99_800.0,
        close=100_100.0,
        volume=5.0,
    )

    outcome = runtime.process_candle(stale_candle)

    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert any(event.event_type == "alert.raised" for event in runtime.store.list_events(limit=20))


def test_recently_closed_candle_is_not_stale(tmp_path) -> None:
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        stale_after_seconds=180,
        signal_threshold=0.001,
    )
    candle_start = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    now = candle_start + timedelta(seconds=300 + 120)
    store = RuntimeStore(config.database_url)
    runtime = BitBatRuntime(store=store, config=config, now_fn=lambda: now)
    runtime.initialize()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=candle_start,
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )

    outcome = runtime.process_candle(candle)

    assert outcome.decision.stale_data is False
    assert outcome.decision.action == "buy"
    assert outcome.order is not None


def test_sell_signal_exits_existing_spot_position(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
    runtime = build_runtime(tmp_path, now_fn=lambda: now)
    buy_candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )
    runtime.process_candle(buy_candle)

    sell_now = datetime(2026, 4, 25, 10, 6, tzinfo=UTC)
    runtime.now_fn = lambda: sell_now
    sell_candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 5, tzinfo=UTC),
        open=100_700.0,
        high=100_720.0,
        low=99_100.0,
        close=99_300.0,
        volume=16.1,
    )

    outcome = runtime.process_candle(sell_candle)

    assert outcome.signal.direction == "sell"
    assert outcome.decision.action == "sell"
    assert outcome.order is not None
    assert outcome.order.side == "sell"
    assert outcome.portfolio.position_qty == 0.0
    assert "confirmed downside trend and score below sell threshold" in outcome.decision.reason


def test_risk_cap_blocks_second_buy(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        order_size_btc=0.02,
        max_position_size_btc=0.02,
        signal_threshold=0.001,
    )
    store = RuntimeStore(config.database_url)
    runtime = BitBatRuntime(store=store, config=config, now_fn=lambda: now)
    runtime.initialize()
    first_candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )
    runtime.process_candle(first_candle)

    second_now = datetime(2026, 4, 25, 10, 6, tzinfo=UTC)
    runtime.now_fn = lambda: second_now
    second_candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 5, tzinfo=UTC),
        open=100_700.0,
        high=101_200.0,
        low=100_500.0,
        close=101_000.0,
        volume=10.0,
    )

    outcome = runtime.process_candle(second_candle)

    assert outcome.signal.direction == "buy"
    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert any(
        event.payload.get("code") == "risk_cap"
        for event in runtime.store.list_events(limit=20)
        if event.event_type == "alert.raised"
    )


def test_retrain_and_acknowledge_update_control_state(tmp_path) -> None:
    runtime = build_runtime(tmp_path)

    retrain_state = runtime.request_retrain()
    acknowledged_state = runtime.acknowledge_alert("operator cleared ritual alert")

    assert retrain_state.retrain_requested is True
    assert acknowledged_state.last_acknowledged_alert == "operator cleared ritual alert"


def test_sell_signal_without_position_holds(tmp_path) -> None:
    runtime = build_runtime(tmp_path, now_fn=lambda: datetime(2026, 4, 25, 10, 1, tzinfo=UTC))

    outcome = runtime.process_candle(
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            open=100_000.0,
            high=100_100.0,
            low=98_900.0,
            close=99_000.0,
            volume=11.0,
        )
    )

    assert outcome.signal.direction == "sell"
    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert outcome.decision.reason == "no valid spot action"


def test_filtered_strategy_holds_when_range_and_body_filters_fail(tmp_path) -> None:
    runtime = build_runtime(tmp_path, now_fn=lambda: datetime(2026, 4, 25, 10, 1, tzinfo=UTC))
    base_start = datetime(2026, 4, 25, 8, 55, tzinfo=UTC)
    for idx in range(13):
        current_start = base_start + timedelta(seconds=300 * idx)
        runtime.now_fn = lambda current_start=current_start: current_start + timedelta(seconds=60)
        runtime.process_candle(
            Candle(
                product_id="BTC-USD",
                granularity_seconds=300,
                start=current_start,
                open=100_000.0 + (idx * 80.0),
                high=100_260.0 + (idx * 80.0),
                low=99_980.0 + (idx * 80.0),
                close=100_220.0 + (idx * 80.0),
                volume=9.0 + idx,
            )
        )

    runtime.now_fn = lambda: datetime(2026, 4, 25, 10, 1, tzinfo=UTC)

    outcome = runtime.process_candle(
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            open=101_300.0,
            high=104_800.0,
            low=99_800.0,
            close=102_600.0,
            volume=11.0,
        )
    )

    assert outcome.signal.direction == "hold"
    assert outcome.decision.action == "hold"
    assert outcome.decision.reason == "volatility/range filter blocked entry"


def test_process_candle_skips_duplicate_candle_start(tmp_path) -> None:
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

    first = runtime.process_candle(candle)
    event_count_after_first = runtime.store.count_events()
    second = runtime.process_candle(candle)

    assert first.order is not None
    assert second.decision.action == "hold"
    assert second.decision.reason == "duplicate candle"
    assert second.order is None
    assert runtime.store.count_events() == event_count_after_first


def test_event_broker_delivers_future_event_to_subscriber() -> None:
    async def exercise_broker() -> None:
        broker = EventBroker()
        queue = broker.subscribe()
        broker.publish(
            RuntimeEvent(
                id=1,
                event_type="signal.generated",
                occurred_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
                payload={"direction": "buy"},
            )
        )
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        broker.unsubscribe(queue)
        assert event.event_type == "signal.generated"
        assert event.payload["direction"] == "buy"

    asyncio.run(exercise_broker())
