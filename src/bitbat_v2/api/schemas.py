"""Pydantic schemas for the BitBat v2 API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str
    venue: str
    product_id: str
    trading_paused: bool
    event_count: int
    last_event_at: datetime | None = None
    autorun: "AutorunStatusResponse"


class AutorunStatusResponse(BaseModel):
    enabled: bool
    interval_seconds: int
    running: bool
    last_cycle_status: str | None = None
    last_cycle_started_at: datetime | None = None
    last_cycle_completed_at: datetime | None = None
    last_error: str | None = None
    last_processed_candle_start: datetime | None = None
    last_action: str | None = None


class SignalResponse(BaseModel):
    signal_id: str
    generated_at: datetime
    product_id: str
    venue: str
    model_name: str
    direction: str
    confidence: float
    predicted_return: float
    predicted_price: float
    reasons: list[str] = Field(default_factory=list)


class PortfolioResponse(BaseModel):
    as_of: datetime
    cash: float
    position_qty: float
    avg_entry_price: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    status: str


class OrderResponse(BaseModel):
    order_id: str
    decision_id: str
    signal_id: str
    created_at: datetime
    side: str
    quantity_btc: float
    fill_price: float
    status: str
    filled_at: datetime | None = None


class OrdersResponse(BaseModel):
    orders: list[OrderResponse] = Field(default_factory=list)


class ControlStateResponse(BaseModel):
    trading_paused: bool
    retrain_requested: bool
    last_acknowledged_alert: str | None = None
    updated_at: datetime


class ControlActionResponse(BaseModel):
    status: str
    control: ControlStateResponse


class ResetPaperResponse(BaseModel):
    status: str
    portfolio: PortfolioResponse


class SimulateCandleRequest(BaseModel):
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=10.0, ge=0)

    @model_validator(mode="after")
    def validate_ohlc_order(self) -> SimulateCandleRequest:
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must fall within the candle range")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must fall within the candle range")
        return self


class DecisionResponse(BaseModel):
    decision_id: str
    signal_id: str
    decided_at: datetime
    action: str
    quantity_btc: float
    reason: str
    stale_data: bool = False
    trading_paused: bool = False


class SimulateCandleResponse(BaseModel):
    signal: SignalResponse
    decision: DecisionResponse
    portfolio: PortfolioResponse
    order: OrderResponse | None = None


class AcknowledgeAlertRequest(BaseModel):
    message: str
