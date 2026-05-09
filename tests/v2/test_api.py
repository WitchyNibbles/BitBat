from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from bitbat.api.cors import ALLOWED_BROWSER_ORIGINS
from bitbat_v2.api.app import create_app
from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle, PortfolioSnapshot, PredictionSignal, utc_now
from bitbat_v2.storage import RuntimeStore
from tests.api.client import SyncASGIClient

AUTH_HEADERS = {"X-BitBat-Operator-Token": "test-token"}
AUTH_TOKEN = AUTH_HEADERS["X-BitBat-Operator-Token"]


def test_v2_app_module_imports_without_repo_data_dir(tmp_path) -> None:
    working_dir = tmp_path / "isolated-cwd"
    working_dir.mkdir()

    result = subprocess.run(  # noqa: S603 - fixed interpreter and import target
        [sys.executable, "-c", "import bitbat_v2.api.app"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
    assert health.json()["signal_source"] == "heuristic"
    assert health.json()["signal_model_name"] is None

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
    assert body["signal"]["expected_value_return"] > 0
    assert body["signal"]["expected_cost_return"] > 0
    assert "confirmed positive trend and score above threshold" in body["decision"]["reason"]

    signal = client.get("/v1/signals/latest", headers=AUTH_HEADERS)
    assert signal.status_code == 200
    assert signal.json()["product_id"] == "BTC-USD"
    assert signal.json()["expected_value_return"] > 0

    orders = client.get("/v1/orders", headers=AUTH_HEADERS)
    assert orders.status_code == 200
    assert "orders" in orders.json()

    portfolio = client.get("/v1/portfolio", headers=AUTH_HEADERS)
    assert portfolio.status_code == 200
    assert portfolio.json()["equity"] > 0

    paper = client.get("/v1/paper", headers=AUTH_HEADERS)
    assert paper.status_code == 200
    assert paper.json()["performance"]["trade_count"] >= 1
    assert "equity_curve" in paper.json()

    performance = client.get("/v1/performance", headers=AUTH_HEADERS)
    assert performance.status_code == 200
    assert "net_pnl" in performance.json()


def test_v2_health_exposes_promotion_evidence(tmp_path, monkeypatch) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "cv_summary.json").write_text(
        json.dumps({
            "champion_decision": {
                "winner": "xgb",
                "promote_candidate": False,
                "reason": "replay-gate-failed",
                "promotion_gate": {
                    "pass": False,
                    "reasons": ["replay_hold_rate_above_threshold"],
                    "replay_gate": {"runtime_compatible": True},
                    "replay_summary": {
                        "trade_count": 1,
                        "hold_rate": 0.99,
                        "action_rate": 0.01,
                        "calibration_brier": 0.9,
                        "mean_expected_value_return": 0.0,
                        "net_pnl_pct": -0.01,
                        "abstain_breakdown": {"legacy model predicted flat": 4},
                        "replay_start": "2026-04-25T10:00:00+00:00",
                        "replay_end": "2026-04-25T12:00:00+00:00",
                    },
                },
                "artifact_metadata": {
                    "family": "xgb",
                    "label_mode": "direction",
                    "version": "test-model-v1",
                },
                "artifact_path": "models/5m_30m/xgb.json",
            },
            "runtime_replay": {"generated_at": "2026-04-25T12:05:00+00:00", "families": {}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("bitbat_v2.api.app.resolve_metrics_dir", lambda: metrics_dir)

    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    health = client.get("/v1/health", headers=AUTH_HEADERS)

    assert health.status_code == 200
    payload = health.json()
    assert payload["promotion"]["verdict"] == "blocked"
    assert payload["promotion"]["label_mode"] == "direction"
    assert payload["promotion"]["model_family"] == "xgb"
    assert payload["promotion"]["replay_hold_rate"] == 0.99
    assert payload["promotion"]["reasons"] == ["replay_hold_rate_above_threshold"]


def test_v2_paper_exposes_hold_rate_and_abstain_breakdown(tmp_path, monkeypatch) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("bitbat_v2.api.app.resolve_metrics_dir", lambda: metrics_dir)

    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    signal = PredictionSignal(
        signal_id="sig-hold",
        generated_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        product_id="BTC-USD",
        venue="coinbase",
        model_name="legacy_xgb_5m_30m",
        direction="hold",
        confidence=0.0,
        predicted_return=0.0,
        predicted_price=100_000.0,
        reasons=["signal_source=legacy_ml"],
        abstain_reason="legacy model predicted flat",
    )
    app.state.store.save_latest_signal(signal)
    app.state.store.append_event(
        "signal.generated",
        signal.to_dict(),
        occurred_at=signal.generated_at,
    )
    app.state.store.append_event(
        "decision.made",
        {
            "decision_id": "dec-hold",
            "signal_id": signal.signal_id,
            "decided_at": signal.generated_at.isoformat(),
            "action": "hold",
            "quantity_btc": 0.0,
            "reason": "no valid spot action",
            "stale_data": False,
            "trading_paused": False,
        },
        occurred_at=signal.generated_at,
    )

    client = SyncASGIClient(app)
    paper = client.get("/v1/paper", headers=AUTH_HEADERS)

    assert paper.status_code == 200
    payload = paper.json()
    assert payload["performance"]["hold_rate"] == 1.0
    assert payload["performance"]["action_rate"] == 0.0
    assert payload["performance"]["abstain_breakdown"]["legacy model predicted flat"] == 1


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


@pytest.mark.parametrize("origin", ALLOWED_BROWSER_ORIGINS)
def test_v2_api_allows_configured_local_browser_origin_preflight(tmp_path, origin: str) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.request(
        "OPTIONS",
        "/v1/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-BitBat-Operator-Token",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    allow_headers = response.headers["access-control-allow-headers"].lower()
    assert "x-bitbat-operator-token" in allow_headers


def test_v2_api_rejects_unknown_origin_preflight(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.request(
        "OPTIONS",
        "/v1/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-BitBat-Operator-Token",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_v2_config_from_env_rejects_unknown_signal_source(monkeypatch) -> None:
    monkeypatch.setenv("BITBAT_V2_SIGNAL_SOURCE", "typo-mode")

    with pytest.raises(ValueError, match="Unsupported BITBAT_V2_SIGNAL_SOURCE"):
        BitBatV2Config.from_env()


def test_v2_control_pause_resume_reset_and_acknowledge(tmp_path) -> None:
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

    acknowledged = client.request(
        "POST",
        "/v1/control/acknowledge",
        headers=AUTH_HEADERS,
        json={"message": "operator cleared ritual alert"},
    )
    assert acknowledged.status_code == 200
    assert (
        acknowledged.json()["control"]["last_acknowledged_alert"] == "operator cleared ritual alert"
    )

    reset = client.request("POST", "/v1/control/reset-paper", headers=AUTH_HEADERS)
    assert reset.status_code == 200
    assert reset.json()["portfolio"]["cash"] == 10_000.0


def test_v2_retrain_endpoint_reports_not_wired(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    client = SyncASGIClient(app)

    response = client.request("POST", "/v1/control/retrain", headers=AUTH_HEADERS)

    assert response.status_code == 501
    assert "not wired yet" in response.json()["detail"]


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


def test_v2_paper_summary_uses_full_order_history_beyond_recent_table_limit(tmp_path) -> None:
    app = create_app(
        database_url=f"sqlite:///{tmp_path / 'bitbat_v2.db'}",
        demo_mode=False,
        operator_token=AUTH_TOKEN,
    )
    store = RuntimeStore(app.state.config.database_url)
    store.save_latest_signal(
        PredictionSignal(
            signal_id="sig-many",
            generated_at=utc_now(),
            product_id="BTC-USD",
            venue="coinbase",
            model_name="ritual-momentum-v1",
            direction="buy",
            confidence=0.61,
            predicted_return=0.003,
            predicted_price=101_000.0,
            reasons=["score=0.003000"],
        )
    )
    store.save_portfolio(
        PortfolioSnapshot(
            as_of=utc_now(),
            cash=10_050.0,
            position_qty=0.0,
            avg_entry_price=0.0,
            mark_price=100_900.0,
            realized_pnl=50.0,
            unrealized_pnl=0.0,
            equity=10_050.0,
            status="paper",
        )
    )
    for idx in range(600):
        app.state.store.save_order(
            app.state.runtime._fill_order(  # noqa: SLF001
                decision=type(
                    "Decision",
                    (),
                    {
                        "decision_id": f"dec-{idx}",
                        "signal_id": "sig-many",
                        "decided_at": utc_now(),
                        "action": "buy" if idx % 2 == 0 else "sell",
                        "quantity_btc": 0.01,
                    },
                )(),
                price=100_000.0 + idx,
            )
        )
    app.state.store.append_event(
        "portfolio.updated",
        {
            "equity": 10_050.0,
            "cash": 10_050.0,
            "position_qty": 0.0,
            "mark_price": 100_900.0,
            "realized_pnl": 50.0,
            "unrealized_pnl": 0.0,
        },
    )
    client = SyncASGIClient(app)

    paper = client.get("/v1/paper", headers=AUTH_HEADERS)
    performance = client.get("/v1/performance", headers=AUTH_HEADERS)

    assert paper.status_code == 200
    assert performance.status_code == 200
    assert paper.json()["performance"]["trade_count"] == 600
    assert performance.json()["trade_count"] == 600
    assert len(paper.json()["recent_orders"]) == 50
