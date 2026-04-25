"""Standalone FastAPI app for the BitBat v2 clean-room runtime."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from bitbat_v2.api.schemas import (
    AcknowledgeAlertRequest,
    AutorunStatusResponse,
    ControlActionResponse,
    ControlStateResponse,
    DecisionResponse,
    HealthResponse,
    OrderResponse,
    OrdersResponse,
    PortfolioResponse,
    ResetPaperResponse,
    SignalResponse,
    SimulateCandleRequest,
    SimulateCandleResponse,
)
from bitbat_v2.autorun import AutonomousPaperTrader
from bitbat_v2.coinbase import CoinbaseMarketDataClient
from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
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
    async def lifespan(_: FastAPI):
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
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
        return HealthResponse(
            status="ok",
            venue=config.venue,
            product_id=config.product_id,
            trading_paused=control.trading_paused,
            event_count=store.count_events(),
            last_event_at=store.get_last_event_at(),
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

    @app.post("/v1/control/retrain", response_model=ControlActionResponse)
    def retrain(_: None = Depends(require_operator_auth)) -> ControlActionResponse:
        control = runtime.request_retrain()
        return ControlActionResponse(
            status="retrain_requested",
            control=ControlStateResponse.model_validate(control.to_dict()),
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
                OrderResponse.model_validate(outcome.order.to_dict())
                if outcome.order
                else None
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
                OrderResponse.model_validate(outcome.order.to_dict())
                if outcome.order
                else None
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
