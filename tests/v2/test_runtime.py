from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle, RuntimeEvent
from bitbat_v2.runtime import BitBatRuntime, EventBroker
from bitbat_v2.signals import SignalEvaluation
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
    assert outcome.portfolio.position_qty >= runtime.config.order_size_btc
    assert outcome.portfolio.position_qty <= runtime.config.max_position_size_btc
    assert outcome.portfolio.cash < runtime.config.starting_cash_usd
    assert outcome.signal.expected_value_return > 0
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


def test_runtime_holds_buy_lean_signal_when_expected_value_after_costs_is_non_positive(
    tmp_path,
) -> None:
    now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
    runtime = build_runtime(tmp_path, now_fn=lambda: now)

    class StubSignalProvider:
        def evaluate(self, context) -> SignalEvaluation:
            del context
            return SignalEvaluation(
                model_name="stub_model",
                direction="buy",
                predicted_return=0.0006,
                predicted_price=100_100.0,
                confidence=0.81,
                reasons=["stub=buy_lean"],
                block_reason=None,
                p_up=0.66,
                p_down=0.24,
                p_flat=0.10,
                expected_move_return=0.0006,
                expected_cost_return=0.001,
                expected_value_return=-0.0004,
                abstain_reason=None,
            )

    runtime.signal_provider = StubSignalProvider()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_400.0,
        low=99_900.0,
        close=100_100.0,
        volume=7.5,
    )

    outcome = runtime.process_candle(candle)

    assert outcome.signal.direction == "buy"
    assert outcome.signal.expected_value_return < 0
    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert "expected value" in outcome.decision.reason


def test_runtime_holds_sell_lean_signal_when_expected_value_after_costs_is_non_negative(
    tmp_path,
) -> None:
    buy_now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
    runtime = build_runtime(tmp_path, now_fn=lambda: buy_now)
    runtime.process_candle(
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            open=100_000.0,
            high=100_900.0,
            low=99_800.0,
            close=100_700.0,
            volume=12.5,
        )
    )

    class StubSignalProvider:
        def evaluate(self, context) -> SignalEvaluation:
            del context
            return SignalEvaluation(
                model_name="stub_model",
                direction="sell",
                predicted_return=-0.0006,
                predicted_price=100_000.0,
                confidence=0.78,
                reasons=["stub=sell_lean"],
                block_reason=None,
                p_up=0.22,
                p_down=0.58,
                p_flat=0.20,
                expected_move_return=-0.0006,
                expected_cost_return=0.001,
                expected_value_return=0.0004,
                abstain_reason=None,
            )

    runtime.signal_provider = StubSignalProvider()
    sell_now = datetime(2026, 4, 25, 10, 6, tzinfo=UTC)
    runtime.now_fn = lambda: sell_now
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 5, tzinfo=UTC),
        open=100_700.0,
        high=100_750.0,
        low=100_000.0,
        close=100_100.0,
        volume=9.1,
    )

    outcome = runtime.process_candle(candle)

    assert outcome.signal.direction == "sell"
    assert outcome.signal.expected_value_return > 0
    assert outcome.decision.action == "hold"
    assert outcome.order is None
    assert "expected value" in outcome.decision.reason


def test_runtime_applies_fees_slippage_and_dynamic_sizing(tmp_path) -> None:
    now = datetime(2026, 4, 25, 10, 1, tzinfo=UTC)
    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        order_size_btc=0.01,
        max_position_size_btc=0.05,
        min_order_size_btc=0.005,
        signal_threshold=0.001,
        fee_bps=10.0,
        slippage_bps=10.0,
    )
    runtime = BitBatRuntime(
        store=RuntimeStore(config.database_url),
        config=config,
        now_fn=lambda: now,
    )
    runtime.initialize()

    outcome = runtime.process_candle(
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            open=100_000.0,
            high=100_900.0,
            low=99_800.0,
            close=100_700.0,
            volume=12.5,
        )
    )

    assert outcome.order is not None
    assert outcome.order.fill_price > 100_700.0
    assert outcome.order.quantity_btc > config.order_size_btc
    assert outcome.portfolio.avg_entry_price >= outcome.order.fill_price
    assert outcome.portfolio.equity < config.starting_cash_usd


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


def test_runtime_legacy_ml_path_executes_buy_with_direction_artifact_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.json").write_text("fake-model", encoding="utf-8")
    (model_dir / "xgb.meta.json").write_text(
        json.dumps({
            "family": "xgb",
            "label_mode": "direction",
            "version": "test-model-v1",
            "freq": "5m",
            "horizon": "30m",
        }),
        encoding="utf-8",
    )

    price_index = pd.date_range("2026-04-25 06:00:00", periods=60, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100_000.0 + idx for idx in range(60)],
            "high": [100_050.0 + idx for idx in range(60)],
            "low": [99_950.0 + idx for idx in range(60)],
            "close": [100_020.0 + idx for idx in range(60)],
            "volume": [10.0] * 60,
        },
        index=price_index,
    )
    prices.index.name = "timestamp_utc"

    def _fake_generate_price_features(
        prices: pd.DataFrame,
        enable_garch: bool,
        freq: str,
    ) -> pd.DataFrame:
        del enable_garch, freq
        aligned_index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        features = pd.DataFrame({"feat_stub": [0.5] * len(prices)}, index=aligned_index)
        features.index.name = "timestamp_utc"
        return features

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(
        signal_module,
        "get_runtime_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(
        signal_module,
        "load_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")
    monkeypatch.setattr(signal_module, "_load_ingested_prices", lambda data_dir, freq: prices)
    monkeypatch.setattr(signal_module, "generate_price_features", _fake_generate_price_features)
    monkeypatch.setattr(
        signal_module,
        "load_model",
        lambda path, expected_label_mode=None: type(
            "FakeBooster",
            (),
            {"feature_names": ["feat_stub"]},
        )(),
    )
    monkeypatch.setattr(
        signal_module,
        "predict_bar",
        lambda booster, feature_row, timestamp, current_price, tau: {
            "predicted_direction": "up",
            "predicted_price": current_price * 1.01,
            "p_up": 0.72,
            "p_down": 0.18,
            "p_flat": 0.10,
            "confidence": 0.72,
        },
    )

    config = BitBatV2Config(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        signal_source="legacy_ml",
        legacy_signal_freq="5m",
        legacy_signal_horizon="30m",
        signal_threshold=0.001,
    )
    runtime = BitBatRuntime(
        store=RuntimeStore(config.database_url),
        config=config,
        now_fn=lambda: datetime(2026, 4, 25, 10, 1, tzinfo=UTC),
    )
    runtime.initialize()

    outcome = runtime.process_candle(
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            open=100_000.0,
            high=100_900.0,
            low=99_800.0,
            close=100_700.0,
            volume=12.5,
        )
    )

    assert outcome.signal.model_name == "legacy_xgb_5m_30m"
    assert outcome.signal.expected_value_return > 0
    assert outcome.decision.action == "buy"
    assert outcome.order is not None
    assert "artifact_label_mode=direction" in outcome.signal.reasons


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
