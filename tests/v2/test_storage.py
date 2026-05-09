from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle, PredictionSignal
from bitbat_v2.runtime import BitBatRuntime
from bitbat_v2.storage import RuntimeStore


def test_runtime_store_round_trips_state_across_reopen(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'bitbat_v2.db'}"
    config = BitBatV2Config(database_url=database_url, demo_mode=False, signal_threshold=0.001)
    store = RuntimeStore(database_url)
    runtime = BitBatRuntime(
        store=store,
        config=config,
        now_fn=lambda: datetime(2026, 4, 25, 10, 1, tzinfo=UTC),
    )
    runtime.initialize()
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
    runtime.request_retrain()
    runtime.acknowledge_alert("operator cleared ritual alert")

    reopened = RuntimeStore(database_url)
    reopened.create_schema()

    signal = reopened.get_latest_signal()
    portfolio = reopened.get_portfolio()
    orders = reopened.get_orders(limit=5)
    control = reopened.get_control_state()

    assert signal is not None
    assert signal.product_id == "BTC-USD"
    assert portfolio.equity > 0
    assert orders and orders[0].status == "filled"
    assert control.retrain_requested is True
    assert control.last_acknowledged_alert == "operator cleared ritual alert"

    portfolio_events = reopened.list_events_by_type("portfolio.updated", limit=10)
    assert portfolio_events
    assert portfolio_events[-1].event_type == "portfolio.updated"


def test_runtime_store_round_trips_signal_evidence_fields(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'bitbat_v2.db'}"
    store = RuntimeStore(database_url)
    store.create_schema()
    signal = PredictionSignal(
        signal_id="sig-123",
        generated_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        product_id="BTC-USD",
        venue="coinbase",
        model_name="legacy_xgb_5m_30m",
        direction="buy",
        confidence=0.72,
        predicted_return=0.0054,
        predicted_price=101_020.2,
        reasons=["signal_source=legacy_ml"],
        p_up=0.72,
        p_down=0.18,
        p_flat=0.10,
        expected_move_return=0.0054,
        expected_cost_return=0.001,
        expected_value_return=0.0044,
        abstain_reason=None,
    )

    store.save_latest_signal(signal)
    loaded = store.get_latest_signal()

    assert loaded is not None
    assert loaded.p_up == 0.72
    assert loaded.p_down == 0.18
    assert loaded.p_flat == 0.10
    assert loaded.expected_move_return == 0.0054
    assert loaded.expected_cost_return == 0.001
    assert loaded.expected_value_return == 0.0044
    assert loaded.abstain_reason is None


def test_runtime_store_migrates_legacy_latest_signal_schema(tmp_path) -> None:
    sqlite_path = tmp_path / "legacy_signal.db"
    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE v2_latest_signal (
            id INTEGER PRIMARY KEY,
            generated_at TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            venue TEXT NOT NULL,
            model_name TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            predicted_return REAL NOT NULL,
            predicted_price REAL NOT NULL,
            reasons_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    store = RuntimeStore(f"sqlite:///{sqlite_path}")
    store.create_schema()

    signal = PredictionSignal(
        signal_id="sig-migrated",
        generated_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        product_id="BTC-USD",
        venue="coinbase",
        model_name="legacy_xgb_5m_30m",
        direction="buy",
        confidence=0.72,
        predicted_return=0.0054,
        predicted_price=101_020.2,
        reasons=["signal_source=legacy_ml"],
        p_up=0.72,
        p_down=0.18,
        p_flat=0.10,
        expected_move_return=0.0054,
        expected_cost_return=0.001,
        expected_value_return=0.0044,
        abstain_reason=None,
    )

    store.save_latest_signal(signal)
    loaded = store.get_latest_signal()

    assert loaded is not None
    assert loaded.signal_id == "sig-migrated"
    assert loaded.expected_value_return == 0.0044
