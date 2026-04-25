from __future__ import annotations

from datetime import UTC, datetime

from bitbat_v2.api.app import create_app
from bitbat_v2.domain import Candle
from tests.api.client import SyncASGIClient

AUTH_HEADERS = {"X-BitBat-Operator-Token": "test-token"}
AUTH_TOKEN = AUTH_HEADERS["X-BitBat-Operator-Token"]


def test_v2_api_health_and_simulation_flow(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    health = client.get("/v1/health", headers=AUTH_HEADERS)
    assert health.status_code == 200
    assert health.json()["product_id"] == "BTC-USD"
    assert health.json()["autorun"]["enabled"] is False

    simulated = client.request(
        "POST",
        "/v1/control/simulate-candle",
        headers=AUTH_HEADERS,
        json={"close": 101_100.0, "open": 100_000.0, "high": 101_200.0, "low": 100_000.0},
    )
    assert simulated.status_code == 200
    body = simulated.json()
    assert body["decision"]["action"] == "buy"
    assert any(reason.startswith("score=") for reason in body["signal"]["reasons"])
    assert "confirmed positive trend and score above threshold" in body["decision"]["reason"]

    signal = client.get("/v1/signals/latest", headers=AUTH_HEADERS)
    assert signal.status_code == 200
    assert signal.json()["product_id"] == "BTC-USD"

    orders = client.get("/v1/orders", headers=AUTH_HEADERS)
    assert orders.status_code == 200
    assert "orders" in orders.json()

    portfolio = client.get("/v1/portfolio", headers=AUTH_HEADERS)
    assert portfolio.status_code == 200
    assert portfolio.json()["equity"] > 0


def test_v2_latest_signal_returns_404_before_any_candle(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.get("/v1/signals/latest", headers=AUTH_HEADERS)

    assert response.status_code == 404


def test_v2_auth_rejects_missing_token(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    health = client.get("/v1/health")

    assert health.status_code == 401


def test_v2_control_pause_resume_reset_retrain_and_acknowledge(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    paused = client.request("POST", "/v1/control/pause", headers=AUTH_HEADERS)
    assert paused.status_code == 200
    assert paused.json()["control"]["trading_paused"] is True

    resumed = client.request("POST", "/v1/control/resume", headers=AUTH_HEADERS)
    assert resumed.status_code == 200
    assert resumed.json()["control"]["trading_paused"] is False

    retrained = client.request("POST", "/v1/control/retrain", headers=AUTH_HEADERS)
    assert retrained.status_code == 200
    assert retrained.json()["control"]["retrain_requested"] is True

    acknowledged = client.request(
        "POST",
        "/v1/control/acknowledge",
        headers=AUTH_HEADERS,
        json={"message": "operator cleared ritual alert"},
    )
    assert acknowledged.status_code == 200
    assert (
        acknowledged.json()["control"]["last_acknowledged_alert"]
        == "operator cleared ritual alert"
    )

    reset = client.request("POST", "/v1/control/reset-paper", headers=AUTH_HEADERS)
    assert reset.status_code == 200
    assert reset.json()["portfolio"]["cash"] == 10_000.0


def test_v2_event_stream_supports_backlog_reads(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)
    client.request(
        "POST",
        "/v1/control/simulate-candle",
        headers=AUTH_HEADERS,
        json={"close": 101_200.0, "open": 100_000.0, "high": 101_500.0, "low": 99_800.0},
    )

    response = client.get("/v1/stream/events?limit=6&once=true&token=test-token")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: candle.closed" in response.text
    assert "event: signal.generated" in response.text
    assert "event: portfolio.updated" in response.text


def test_v2_stream_rejects_missing_token(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.get("/v1/stream/events?limit=0")

    assert response.status_code == 401


def test_v2_sync_market_uses_coinbase_adapter(tmp_path) -> None:
    class FakeCoinbaseClient:
        def fetch_candles(self, product_id: str, granularity_seconds: int, start, end):
            return [
                Candle(
                    product_id=product_id,
                    granularity_seconds=granularity_seconds,
                    start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
                    open=100_000.0,
                    high=100_900.0,
                    low=99_900.0,
                    close=100_800.0,
                    volume=9.5,
                )
            ]

    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        market_data_client=FakeCoinbaseClient(),
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.request("POST", "/v1/control/sync-market", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["signal"]["product_id"] == "BTC-USD"
    assert body["portfolio"]["equity"] > 0
    assert any(reason.startswith("score=") for reason in body["signal"]["reasons"])


def test_v2_sync_market_skips_duplicate_coinbase_candle(tmp_path) -> None:
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_900.0,
        close=100_800.0,
        volume=9.5,
    )

    class FakeCoinbaseClient:
        def fetch_candles(self, product_id: str, granularity_seconds: int, start, end):
            return [candle]

    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        market_data_client=FakeCoinbaseClient(),
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    first = client.request("POST", "/v1/control/sync-market", headers=AUTH_HEADERS)
    second = client.request("POST", "/v1/control/sync-market", headers=AUTH_HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["decision"]["action"] == "hold"
    assert second.json()["decision"]["reason"] == "duplicate candle"


def test_v2_health_reports_autorun_enabled(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
        autorun_enabled=True,
    )
    client = SyncASGIClient(app)

    health = client.get("/v1/health", headers=AUTH_HEADERS)

    assert health.status_code == 200
    assert health.json()["autorun"]["enabled"] is True
    assert health.json()["autorun"]["interval_seconds"] > 0


def test_v2_simulate_candle_rejects_invalid_ohlc(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.request(
        "POST",
        "/v1/control/simulate-candle",
        headers=AUTH_HEADERS,
        json={"close": 101_000.0, "open": 100_000.0, "high": 99_000.0, "low": 101_500.0},
    )

    assert response.status_code == 422
