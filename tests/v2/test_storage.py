from __future__ import annotations

from datetime import UTC, datetime

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
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
