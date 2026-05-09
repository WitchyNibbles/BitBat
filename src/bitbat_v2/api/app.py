"""Standalone FastAPI app for the BitBat v2 clean-room runtime."""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from bitbat.api.cors import ALLOWED_BROWSER_ORIGINS
from bitbat.config.loader import resolve_metrics_dir
from bitbat_v2.api.schemas import (
    AcknowledgeAlertRequest,
    AutorunStatusResponse,
    ControlActionResponse,
    ControlStateResponse,
    DecisionResponse,
    HealthResponse,
    OrderResponse,
    OrdersResponse,
    PaperAlertResponse,
    PaperCockpitResponse,
    PaperOrderResponse,
    PaperPerformancePointResponse,
    PaperPerformanceResponse,
    PaperTradeResponse,
    PortfolioResponse,
    PromotionEvidenceResponse,
    ResetPaperResponse,
    SignalResponse,
    SimulateCandleRequest,
    SimulateCandleResponse,
)
from bitbat_v2.autorun import AutonomousPaperTrader
from bitbat_v2.coinbase import CoinbaseMarketDataClient
from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
from bitbat_v2.paper import (
    build_paper_cockpit_snapshot,
    build_paper_performance_summary,
    closed_trades_from_orders,
)
from bitbat_v2.runtime import BitBatRuntime, EventBroker, format_sse
from bitbat_v2.storage import RuntimeStore


def _control_response(status: str, control: ControlStateResponse) -> ControlActionResponse:
    return ControlActionResponse(status=status, control=control)


def _resolve_operator_token(config: BitBatV2Config, override: str | None) -> str | None:
    if override is not None:
        return override
    if config.operator_token:
        return config.operator_token
    if config.demo_mode:
        return "bitbat-local-dev-token"
    return None


def _load_json_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = pd.Timestamp(value)
    parsed = parsed.tz_localize(UTC) if parsed.tzinfo is None else parsed.tz_convert(UTC)
    return parsed.to_pydatetime()


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _winner_runtime_payload(runtime_replay: dict[str, Any], winner: str | None) -> dict[str, Any]:
    family_payloads = _as_mapping(runtime_replay.get("families", {}))
    if winner is None:
        return {}
    return _as_mapping(family_payloads.get(winner, {}))


def _promotion_reasons(champion: dict[str, Any], promotion_gate: dict[str, Any]) -> list[str]:
    reasons = promotion_gate.get("reasons", [])
    if isinstance(reasons, list) and reasons:
        return [str(reason) for reason in reasons]
    if champion.get("reason"):
        return [str(champion["reason"])]
    return []


def _replay_generated_at(metrics_dir: Path, runtime_replay: dict[str, Any]) -> datetime | None:
    generated_at = _coerce_datetime(runtime_replay.get("generated_at"))
    if generated_at is not None:
        return generated_at
    replay_path = metrics_dir / "runtime_replay_summary.json"
    if replay_path.exists():
        return datetime.fromtimestamp(replay_path.stat().st_mtime, tz=UTC)
    return None


def _load_promotion_evidence() -> dict[str, Any] | None:
    metrics_dir = resolve_metrics_dir()
    cv_summary = _load_json_payload(metrics_dir / "cv_summary.json")
    if cv_summary is None:
        return None

    champion = _as_mapping(cv_summary.get("champion_decision", {}))
    if not champion:
        return None
    winner = champion.get("winner") if isinstance(champion.get("winner"), str) else None
    promotion_gate = _as_mapping(champion.get("promotion_gate", {}))
    runtime_replay = _as_mapping(cv_summary.get("runtime_replay", {}))
    winner_runtime = _winner_runtime_payload(runtime_replay, winner)
    replay_gate = _as_mapping(promotion_gate.get("replay_gate", {})) or _as_mapping(
        winner_runtime.get("replay_gate", {})
    )
    replay_summary = _as_mapping(promotion_gate.get("replay_summary", {})) or _as_mapping(
        winner_runtime.get("replay_summary", {})
    )
    artifact_metadata = _as_mapping(champion.get("artifact_metadata", {})) or _as_mapping(
        winner_runtime.get("artifact_metadata", {})
    )

    return {
        "verdict": (
            "promotable"
            if bool(champion.get("promote_candidate")) and bool(promotion_gate.get("pass", False))
            else "blocked"
        ),
        "winner": winner,
        "reasons": _promotion_reasons(champion, promotion_gate),
        "runtime_compatible": replay_gate.get("runtime_compatible"),
        "model_family": artifact_metadata.get("family"),
        "label_mode": artifact_metadata.get("label_mode"),
        "model_version": artifact_metadata.get("version"),
        "artifact_path": champion.get("artifact_path") or winner_runtime.get("artifact_path"),
        "replay_generated_at": _replay_generated_at(metrics_dir, runtime_replay),
        "replay_start": _coerce_datetime(replay_summary.get("replay_start")),
        "replay_end": _coerce_datetime(replay_summary.get("replay_end")),
        "replay_trade_count": replay_summary.get("trade_count"),
        "replay_hold_rate": replay_summary.get("hold_rate"),
        "replay_action_rate": replay_summary.get("action_rate"),
        "replay_calibration_brier": replay_summary.get("calibration_brier"),
        "replay_mean_expected_value_return": replay_summary.get("mean_expected_value_return"),
        "replay_net_pnl_pct": replay_summary.get("net_pnl_pct"),
        "replay_abstain_breakdown": replay_summary.get("abstain_breakdown", {}),
    }


def create_app(  # noqa: C901
    database_url: str | None = None,
    demo_mode: bool | None = None,
    market_data_client: CoinbaseMarketDataClient | None = None,
    operator_token: str | None = None,
    autorun_enabled: bool | None = None,
) -> FastAPI:
    config = BitBatV2Config.from_env()
    if database_url is not None:
        config = BitBatV2Config(**{**config.__dict__, "database_url": database_url})
    if demo_mode is not None:
        config = BitBatV2Config(**{**config.__dict__, "demo_mode": demo_mode})
    if autorun_enabled is not None:
        config = BitBatV2Config(**{**config.__dict__, "autorun_enabled": autorun_enabled})

    store = RuntimeStore(config.database_url)
    broker = EventBroker()
    runtime = BitBatRuntime(store=store, config=config, broker=broker)
    coinbase = market_data_client or CoinbaseMarketDataClient()
    autorun = AutonomousPaperTrader(
        runtime=runtime,
        config=config,
        market_data_client=coinbase,
    )
    resolved_operator_token = _resolve_operator_token(config, operator_token)
    runtime.initialize()
    if config.demo_mode:
        runtime.seed_demo_state()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if config.autorun_enabled:
            autorun.start()
        try:
            yield
        finally:
            await autorun.stop()

    app = FastAPI(
        title="BitBat v2 API",
        version="0.2.0",
        description="Clean-room operator API for BitBat paper trading.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_BROWSER_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.runtime = runtime
    app.state.store = store
    app.state.broker = broker
    app.state.market_data_client = coinbase
    app.state.autorun = autorun
    app.state.operator_token = resolved_operator_token

    def require_operator_auth(
        x_bitbat_operator_token: str | None = Header(
            default=None,
            alias="X-BitBat-Operator-Token",
        ),
    ) -> None:
        if resolved_operator_token is None:
            raise HTTPException(status_code=503, detail="Operator token is not configured.")
        if not x_bitbat_operator_token or not hmac.compare_digest(
            x_bitbat_operator_token,
            resolved_operator_token,
        ):
            raise HTTPException(status_code=401, detail="Invalid operator token.")

    @app.get("/v1/health", response_model=HealthResponse)
    def health(_: None = Depends(require_operator_auth)) -> HealthResponse:
        control = store.get_control_state()
        latest_signal = store.get_latest_signal()
        promotion = _load_promotion_evidence()
        return HealthResponse(
            status="ok",
            venue=config.venue,
            product_id=config.product_id,
            trading_paused=control.trading_paused,
            event_count=store.count_events(),
            signal_source=runtime.signal_source,
            signal_model_name=latest_signal.model_name if latest_signal is not None else None,
            last_signal_at=latest_signal.generated_at if latest_signal is not None else None,
            last_event_at=store.get_last_event_at(),
            promotion=(
                PromotionEvidenceResponse.model_validate(promotion)
                if promotion is not None
                else None
            ),
            autorun=AutorunStatusResponse.model_validate(autorun.snapshot().__dict__),
        )

    @app.get("/v1/portfolio", response_model=PortfolioResponse)
    def portfolio(_: None = Depends(require_operator_auth)) -> PortfolioResponse:
        return PortfolioResponse.model_validate(store.get_portfolio().to_dict())

    @app.get("/v1/signals/latest", response_model=SignalResponse)
    def latest_signal(_: None = Depends(require_operator_auth)) -> SignalResponse:
        signal = store.get_latest_signal()
        if signal is None:
            raise HTTPException(status_code=404, detail="No v2 signal has been generated yet.")
        return SignalResponse.model_validate(signal.to_dict())

    @app.get("/v1/orders", response_model=OrdersResponse)
    def orders(
        limit: int = Query(default=20, ge=0, le=100),
        _: None = Depends(require_operator_auth),
    ) -> OrdersResponse:
        return OrdersResponse(
            orders=[
                OrderResponse.model_validate(order.to_dict())
                for order in store.get_orders(limit=limit)
            ]
        )

    @app.get("/v1/paper", response_model=PaperCockpitResponse)
    def paper(_: None = Depends(require_operator_auth)) -> PaperCockpitResponse:
        portfolio = store.get_portfolio()
        latest_signal = store.get_latest_signal()
        orders = store.get_orders(limit=None)
        portfolio_events = store.list_events_by_type("portfolio.updated", limit=None)
        alert_events = store.list_events_by_type("alert.raised", limit=20)
        decision_events = store.list_events_by_type("decision.made", limit=None)
        signal_events = store.list_events_by_type("signal.generated", limit=None)
        snapshot = build_paper_cockpit_snapshot(
            config=config,
            portfolio=portfolio,
            latest_signal=latest_signal,
            last_event_at=store.get_last_event_at(),
            orders=orders,
            portfolio_events=portfolio_events,
            alert_events=alert_events,
            decision_events=decision_events,
            signal_events=signal_events,
        )
        promotion = _load_promotion_evidence()
        closed_trades = [
            PaperTradeResponse.model_validate(trade.to_dict())
            for trade in closed_trades_from_orders(orders, fee_bps=config.fee_bps)
        ]
        return PaperCockpitResponse(
            portfolio=PortfolioResponse.model_validate(snapshot.portfolio.to_dict()),
            performance=PaperPerformanceResponse.model_validate(snapshot.performance.to_dict()),
            promotion=(
                PromotionEvidenceResponse.model_validate(promotion)
                if promotion is not None
                else None
            ),
            latest_signal=(
                SignalResponse.model_validate(snapshot.latest_signal.to_dict())
                if snapshot.latest_signal is not None
                else None
            ),
            recent_orders=[
                PaperOrderResponse.model_validate(order) for order in snapshot.recent_orders
            ],
            recent_alerts=[
                PaperAlertResponse.model_validate(alert.to_dict())
                for alert in snapshot.recent_alerts
            ],
            equity_curve=[
                PaperPerformancePointResponse.model_validate(point.to_dict())
                for point in snapshot.equity_curve
            ],
            closed_trades=closed_trades,
        )

    @app.get("/v1/performance", response_model=PaperPerformanceResponse)
    def performance(_: None = Depends(require_operator_auth)) -> PaperPerformanceResponse:
        orders = store.get_orders(limit=None)
        summary = build_paper_performance_summary(
            config=config,
            portfolio=store.get_portfolio(),
            latest_signal=store.get_latest_signal(),
            last_event_at=store.get_last_event_at(),
            orders=orders,
            portfolio_events=store.list_events_by_type("portfolio.updated", limit=None),
            decision_events=store.list_events_by_type("decision.made", limit=None),
            signal_events=store.list_events_by_type("signal.generated", limit=None),
        )
        return PaperPerformanceResponse.model_validate(summary.to_dict())

    @app.post("/v1/control/pause", response_model=ControlActionResponse)
    def pause(_: None = Depends(require_operator_auth)) -> ControlActionResponse:
        control = runtime.pause_trading()
        return _control_response(
            "paused",
            ControlStateResponse.model_validate(control.to_dict()),
        )

    @app.post("/v1/control/resume", response_model=ControlActionResponse)
    def resume(_: None = Depends(require_operator_auth)) -> ControlActionResponse:
        control = runtime.resume_trading()
        return _control_response(
            "running",
            ControlStateResponse.model_validate(control.to_dict()),
        )

    @app.post("/v1/control/retrain")
    def retrain(_: None = Depends(require_operator_auth)) -> None:
        raise HTTPException(
            status_code=501,
            detail=(
                "BitBat v2 retraining is not wired yet. Use the legacy monitor pipeline for model "
                "retraining during migration."
            ),
        )

    @app.post("/v1/control/acknowledge", response_model=ControlActionResponse)
    def acknowledge(
        request: AcknowledgeAlertRequest,
        _: None = Depends(require_operator_auth),
    ) -> ControlActionResponse:
        control = runtime.acknowledge_alert(request.message)
        return ControlActionResponse(
            status="acknowledged",
            control=ControlStateResponse.model_validate(control.to_dict()),
        )

    @app.post("/v1/control/reset-paper", response_model=ResetPaperResponse)
    def reset_paper(_: None = Depends(require_operator_auth)) -> ResetPaperResponse:
        portfolio_snapshot = runtime.reset_paper_account()
        return ResetPaperResponse(
            status="reset",
            portfolio=PortfolioResponse.model_validate(portfolio_snapshot.to_dict()),
        )

    @app.post("/v1/control/simulate-candle", response_model=SimulateCandleResponse)
    def simulate_candle(
        request: SimulateCandleRequest,
        _: None = Depends(require_operator_auth),
    ) -> SimulateCandleResponse:
        now = datetime.now(tz=UTC)
        fresh_offset_seconds = min(
            config.granularity_seconds,
            max(1, config.stale_after_seconds // 2),
        )
        candle = Candle(
            product_id=config.product_id,
            granularity_seconds=config.granularity_seconds,
            start=now - timedelta(seconds=fresh_offset_seconds),
            open=request.open,
            high=request.high,
            low=request.low,
            close=request.close,
            volume=request.volume,
        )
        outcome = runtime.process_candle(candle)
        return SimulateCandleResponse(
            signal=SignalResponse.model_validate(outcome.signal.to_dict()),
            decision=DecisionResponse.model_validate(outcome.decision.to_dict()),
            portfolio=PortfolioResponse.model_validate(outcome.portfolio.to_dict()),
            order=(
                OrderResponse.model_validate(outcome.order.to_dict()) if outcome.order else None
            ),
        )

    @app.post("/v1/control/sync-market", response_model=SimulateCandleResponse)
    def sync_market(_: None = Depends(require_operator_auth)) -> SimulateCandleResponse:
        now = datetime.now(tz=UTC)
        candles = coinbase.fetch_candles(
            product_id=config.product_id,
            granularity_seconds=config.granularity_seconds,
            start=now - timedelta(seconds=config.granularity_seconds * 3),
            end=now,
        )
        if not candles:
            raise RuntimeError("Coinbase returned no candles for the requested window")
        outcome = runtime.process_candle(candles[-1])
        return SimulateCandleResponse(
            signal=SignalResponse.model_validate(outcome.signal.to_dict()),
            decision=DecisionResponse.model_validate(outcome.decision.to_dict()),
            portfolio=PortfolioResponse.model_validate(outcome.portfolio.to_dict()),
            order=(
                OrderResponse.model_validate(outcome.order.to_dict()) if outcome.order else None
            ),
        )

    @app.get("/v1/stream/events")
    async def stream_events(
        limit: int = Query(default=20, ge=0, le=100),
        once: bool = False,
        token: str | None = Query(default=None),
    ) -> StreamingResponse:
        if resolved_operator_token is None:
            raise HTTPException(status_code=503, detail="Operator token is not configured.")
        if not token or not hmac.compare_digest(token, resolved_operator_token):
            raise HTTPException(status_code=401, detail="Invalid operator token.")

        async def event_generator() -> AsyncIterator[str]:
            for event in store.list_events(limit=limit):
                yield format_sse(event)
            if once:
                return
            queue = broker.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield format_sse(event)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                broker.unsubscribe(queue)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


app = create_app()
