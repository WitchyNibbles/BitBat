"""Domain objects for BitBat v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class Candle:
    product_id: str
    granularity_seconds: int
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "granularity_seconds": self.granularity_seconds,
            "start": _iso(self.start),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candle:
        return cls(
            product_id=str(data["product_id"]),
            granularity_seconds=int(data["granularity_seconds"]),
            start=_parse_dt(str(data["start"])) or utc_now(),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
        )


@dataclass(frozen=True)
class FeatureSnapshot:
    generated_at: datetime
    product_id: str
    close: float
    open_to_close_return: float
    momentum_return: float
    range_ratio: float
    short_trend_return: float = 0.0
    trend_return: float = 0.0
    body_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": _iso(self.generated_at),
            "product_id": self.product_id,
            "close": self.close,
            "open_to_close_return": self.open_to_close_return,
            "momentum_return": self.momentum_return,
            "range_ratio": self.range_ratio,
            "short_trend_return": self.short_trend_return,
            "trend_return": self.trend_return,
            "body_strength": self.body_strength,
        }


@dataclass(frozen=True)
class PredictionSignal:
    signal_id: str
    generated_at: datetime
    product_id: str
    venue: str
    model_name: str
    direction: str
    confidence: float
    predicted_return: float
    predicted_price: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "generated_at": _iso(self.generated_at),
            "product_id": self.product_id,
            "venue": self.venue,
            "model_name": self.model_name,
            "direction": self.direction,
            "confidence": self.confidence,
            "predicted_return": self.predicted_return,
            "predicted_price": self.predicted_price,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionSignal:
        return cls(
            signal_id=str(data["signal_id"]),
            generated_at=_parse_dt(str(data["generated_at"])) or utc_now(),
            product_id=str(data["product_id"]),
            venue=str(data["venue"]),
            model_name=str(data["model_name"]),
            direction=str(data["direction"]),
            confidence=float(data["confidence"]),
            predicted_return=float(data["predicted_return"]),
            predicted_price=float(data["predicted_price"]),
            reasons=[str(reason) for reason in data.get("reasons", [])],
        )


@dataclass(frozen=True)
class StrategyDecision:
    decision_id: str
    signal_id: str
    decided_at: datetime
    action: str
    quantity_btc: float
    reason: str
    stale_data: bool = False
    trading_paused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "decided_at": _iso(self.decided_at),
            "action": self.action,
            "quantity_btc": self.quantity_btc,
            "reason": self.reason,
            "stale_data": self.stale_data,
            "trading_paused": self.trading_paused,
        }


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    decision_id: str
    signal_id: str
    created_at: datetime
    side: str
    quantity_btc: float
    fill_price: float
    status: str
    filled_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "created_at": _iso(self.created_at),
            "side": self.side,
            "quantity_btc": self.quantity_btc,
            "fill_price": self.fill_price,
            "status": self.status,
            "filled_at": _iso(self.filled_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperOrder:
        return cls(
            order_id=str(data["order_id"]),
            decision_id=str(data["decision_id"]),
            signal_id=str(data["signal_id"]),
            created_at=_parse_dt(str(data["created_at"])) or utc_now(),
            side=str(data["side"]),
            quantity_btc=float(data["quantity_btc"]),
            fill_price=float(data["fill_price"]),
            status=str(data["status"]),
            filled_at=_parse_dt(data.get("filled_at")),
        )


@dataclass(frozen=True)
class PortfolioSnapshot:
    as_of: datetime
    cash: float
    position_qty: float
    avg_entry_price: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": _iso(self.as_of),
            "cash": self.cash,
            "position_qty": self.position_qty,
            "avg_entry_price": self.avg_entry_price,
            "mark_price": self.mark_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "equity": self.equity,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortfolioSnapshot:
        return cls(
            as_of=_parse_dt(str(data["as_of"])) or utc_now(),
            cash=float(data["cash"]),
            position_qty=float(data["position_qty"]),
            avg_entry_price=float(data["avg_entry_price"]),
            mark_price=float(data["mark_price"]),
            realized_pnl=float(data["realized_pnl"]),
            unrealized_pnl=float(data["unrealized_pnl"]),
            equity=float(data["equity"]),
            status=str(data["status"]),
        )


@dataclass(frozen=True)
class ControlState:
    trading_paused: bool
    retrain_requested: bool
    last_acknowledged_alert: str | None
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_paused": self.trading_paused,
            "retrain_requested": self.retrain_requested,
            "last_acknowledged_alert": self.last_acknowledged_alert,
            "updated_at": _iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlState:
        return cls(
            trading_paused=bool(data["trading_paused"]),
            retrain_requested=bool(data["retrain_requested"]),
            last_acknowledged_alert=data.get("last_acknowledged_alert"),
            updated_at=_parse_dt(str(data["updated_at"])) or utc_now(),
        )


@dataclass(frozen=True)
class RuntimeEvent:
    id: int
    event_type: str
    occurred_at: datetime
    payload: dict[str, Any]
    stream_key: str = "runtime"

    def to_sse_payload(self) -> str:
        return (
            f"id: {self.id}\n"
            f"event: {self.event_type}\n"
            f"data: {self.payload}\n\n"
        )


@dataclass(frozen=True)
class RuntimeOutcome:
    candle: Candle
    features: FeatureSnapshot
    signal: PredictionSignal
    decision: StrategyDecision
    order: PaperOrder | None
    portfolio: PortfolioSnapshot
