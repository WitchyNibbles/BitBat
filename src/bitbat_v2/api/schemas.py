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
    signal_source: str
    signal_model_name: str | None = None
    last_signal_at: datetime | None = None
    last_event_at: datetime | None = None
    promotion: PromotionEvidenceResponse | None = None
    autorun: AutorunStatusResponse


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
    p_up: float = 0.0
    p_down: float = 0.0
    p_flat: float = 0.0
    expected_move_return: float = 0.0
    expected_cost_return: float = 0.0
    expected_value_return: float = 0.0
    abstain_reason: str | None = None


class PromotionEvidenceResponse(BaseModel):
    verdict: str
    winner: str | None = None
    reasons: list[str] = Field(default_factory=list)
    runtime_compatible: bool | None = None
    model_family: str | None = None
    label_mode: str | None = None
    model_version: str | None = None
    artifact_path: str | None = None
    replay_generated_at: datetime | None = None
    replay_start: datetime | None = None
    replay_end: datetime | None = None
    replay_trade_count: int | None = None
    replay_hold_rate: float | None = None
    replay_action_rate: float | None = None
    replay_calibration_brier: float | None = None
    replay_mean_expected_value_return: float | None = None
    replay_net_pnl_pct: float | None = None
    replay_abstain_breakdown: dict[str, int] = Field(default_factory=dict)


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


class PaperAlertResponse(BaseModel):
    occurred_at: datetime
    code: str
    message: str


class PaperPerformancePointResponse(BaseModel):
    occurred_at: datetime
    equity: float
    cash: float
    position_qty: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float


class PaperTradeResponse(BaseModel):
    closed_at: datetime
    quantity_btc: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    return_pct: float


class PaperPerformanceResponse(BaseModel):
    as_of: datetime
    starting_cash: float
    equity: float
    cash: float
    position_qty: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float
    net_pnl: float
    net_pnl_pct: float
    fees_paid: float
    turnover_usd: float
    trade_count: int
    closed_trade_count: int
    win_rate: float
    expectancy_per_trade: float
    max_drawdown_pct: float
    exposure_pct: float
    benchmark_equity: float
    benchmark_return_pct: float
    alpha_vs_buy_hold: float
    hold_rate: float = 0.0
    action_rate: float = 0.0
    abstain_breakdown: dict[str, int] = Field(default_factory=dict)
    last_signal_at: datetime | None = None
    last_signal_direction: str | None = None
    signal_confidence: float | None = None
    last_event_at: datetime | None = None


class PaperOrderResponse(BaseModel):
    order_id: str
    created_at: datetime
    filled_at: datetime | None = None
    side: str
    quantity_btc: float
    fill_price: float
    status: str
    notional_usd: float


class PaperCockpitResponse(BaseModel):
    portfolio: PortfolioResponse
    performance: PaperPerformanceResponse
    promotion: PromotionEvidenceResponse | None = None
    latest_signal: SignalResponse | None = None
    recent_orders: list[PaperOrderResponse] = Field(default_factory=list)
    recent_alerts: list[PaperAlertResponse] = Field(default_factory=list)
    equity_curve: list[PaperPerformancePointResponse] = Field(default_factory=list)
    closed_trades: list[PaperTradeResponse] = Field(default_factory=list)


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
