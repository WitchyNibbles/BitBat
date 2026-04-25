"""Persistence for BitBat v2 runtime state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine, Row

from .domain import (
    Candle,
    ControlState,
    PaperOrder,
    PortfolioSnapshot,
    PredictionSignal,
    RuntimeEvent,
    utc_now,
)


class RuntimeStore:
    """SQL-backed event store plus read models for BitBat v2."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine: Engine = create_engine(database_url, future=True)
        self.metadata = MetaData()
        self.events = Table(
            "v2_runtime_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("occurred_at", DateTime(timezone=True), nullable=False),
            Column("event_type", String(80), nullable=False),
            Column("stream_key", String(80), nullable=False, default="runtime"),
            Column("payload", Text, nullable=False),
        )
        self.latest_signal = Table(
            "v2_latest_signal",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("generated_at", DateTime(timezone=True), nullable=False),
            Column("signal_id", String(80), nullable=False),
            Column("product_id", String(40), nullable=False),
            Column("venue", String(40), nullable=False),
            Column("model_name", String(80), nullable=False),
            Column("direction", String(16), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("predicted_return", Float, nullable=False),
            Column("predicted_price", Float, nullable=False),
            Column("reasons_json", Text, nullable=False),
        )
        self.portfolio = Table(
            "v2_portfolio",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("as_of", DateTime(timezone=True), nullable=False),
            Column("cash", Float, nullable=False),
            Column("position_qty", Float, nullable=False),
            Column("avg_entry_price", Float, nullable=False),
            Column("mark_price", Float, nullable=False),
            Column("realized_pnl", Float, nullable=False),
            Column("unrealized_pnl", Float, nullable=False),
            Column("equity", Float, nullable=False),
            Column("status", String(24), nullable=False),
        )
        self.orders = Table(
            "v2_paper_orders",
            self.metadata,
            Column("order_id", String(80), primary_key=True),
            Column("decision_id", String(80), nullable=False),
            Column("signal_id", String(80), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("side", String(16), nullable=False),
            Column("quantity_btc", Float, nullable=False),
            Column("fill_price", Float, nullable=False),
            Column("status", String(24), nullable=False),
            Column("filled_at", DateTime(timezone=True), nullable=True),
        )
        self.controls = Table(
            "v2_control_state",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("trading_paused", Boolean, nullable=False),
            Column("retrain_requested", Boolean, nullable=False),
            Column("last_acknowledged_alert", String(255), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

    def create_schema(self) -> None:
        self.metadata.create_all(self.engine)

    def ensure_seed_state(self, starting_cash_usd: float, mark_price: float = 0.0) -> None:
        now = utc_now()
        with self.engine.begin() as conn:
            if conn.execute(select(self.portfolio.c.id)).first() is None:
                conn.execute(
                    insert(self.portfolio).values(
                        id=1,
                        as_of=now,
                        cash=starting_cash_usd,
                        position_qty=0.0,
                        avg_entry_price=0.0,
                        mark_price=mark_price,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        equity=starting_cash_usd,
                        status="paper",
                    )
                )
            if conn.execute(select(self.controls.c.id)).first() is None:
                conn.execute(
                    insert(self.controls).values(
                        id=1,
                        trading_paused=False,
                        retrain_requested=False,
                        last_acknowledged_alert=None,
                        updated_at=now,
                    )
                )

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        occurred_at: datetime | None = None,
        stream_key: str = "runtime",
    ) -> RuntimeEvent:
        stamp = (occurred_at or utc_now()).astimezone(UTC)
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.events).values(
                    occurred_at=stamp,
                    event_type=event_type,
                    stream_key=stream_key,
                    payload=json.dumps(payload, sort_keys=True),
                )
            )
            event_id = int(result.inserted_primary_key[0])
        return RuntimeEvent(
            id=event_id,
            event_type=event_type,
            occurred_at=stamp,
            payload=payload,
            stream_key=stream_key,
        )

    def list_events(self, limit: int = 50) -> list[RuntimeEvent]:
        safe_limit = max(0, min(limit, 100))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.events)
                .order_by(self.events.c.id.desc())
                .limit(safe_limit)
            ).all()
        return [self._event_from_row(row) for row in reversed(rows)]

    def count_events(self) -> int:
        with self.engine.begin() as conn:
            value = conn.execute(select(func.count()).select_from(self.events)).scalar_one()
        return int(value)

    def get_last_event_at(self) -> datetime | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.events.c.occurred_at).order_by(self.events.c.id.desc()).limit(1)
            ).first()
        return self._normalize_dt(row[0]) if row is not None else None

    def get_last_candle(self) -> Candle | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.events)
                .where(self.events.c.event_type == "candle.closed")
                .order_by(self.events.c.id.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        payload = json.loads(row._mapping["payload"])
        return Candle.from_dict(payload)

    def get_recent_candles(self, limit: int) -> list[Candle]:
        safe_limit = max(0, min(limit, 200))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.events)
                .where(self.events.c.event_type == "candle.closed")
                .order_by(self.events.c.id.desc())
                .limit(safe_limit)
            ).all()
        candles = [Candle.from_dict(json.loads(row._mapping["payload"])) for row in rows]
        return list(reversed(candles))

    def save_latest_signal(self, signal: PredictionSignal) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(select(self.latest_signal.c.id)).first()
            values = {
                "id": 1,
                "generated_at": signal.generated_at,
                "signal_id": signal.signal_id,
                "product_id": signal.product_id,
                "venue": signal.venue,
                "model_name": signal.model_name,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "predicted_return": signal.predicted_return,
                "predicted_price": signal.predicted_price,
                "reasons_json": json.dumps(signal.reasons),
            }
            if existing is None:
                conn.execute(insert(self.latest_signal).values(**values))
            else:
                conn.execute(
                    update(self.latest_signal)
                    .where(self.latest_signal.c.id == 1)
                    .values(**values)
                )

    def get_latest_signal(self) -> PredictionSignal | None:
        with self.engine.begin() as conn:
            row = conn.execute(select(self.latest_signal).limit(1)).first()
        if row is None:
            return None
        mapping = row._mapping
        return PredictionSignal.from_dict(
            {
                "signal_id": mapping["signal_id"],
                "generated_at": self._normalize_dt(mapping["generated_at"]).isoformat(),
                "product_id": mapping["product_id"],
                "venue": mapping["venue"],
                "model_name": mapping["model_name"],
                "direction": mapping["direction"],
                "confidence": mapping["confidence"],
                "predicted_return": mapping["predicted_return"],
                "predicted_price": mapping["predicted_price"],
                "reasons": json.loads(mapping["reasons_json"]),
            }
        )

    def save_portfolio(self, portfolio: PortfolioSnapshot) -> None:
        values = {
            "id": 1,
            "as_of": portfolio.as_of,
            "cash": portfolio.cash,
            "position_qty": portfolio.position_qty,
            "avg_entry_price": portfolio.avg_entry_price,
            "mark_price": portfolio.mark_price,
            "realized_pnl": portfolio.realized_pnl,
            "unrealized_pnl": portfolio.unrealized_pnl,
            "equity": portfolio.equity,
            "status": portfolio.status,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(select(self.portfolio.c.id)).first()
            if existing is None:
                conn.execute(insert(self.portfolio).values(**values))
            else:
                conn.execute(
                    update(self.portfolio)
                    .where(self.portfolio.c.id == 1)
                    .values(**values)
                )

    def get_portfolio(self) -> PortfolioSnapshot:
        with self.engine.begin() as conn:
            row = conn.execute(select(self.portfolio).limit(1)).first()
        if row is None:
            raise RuntimeError("portfolio state has not been initialized")
        return self._portfolio_from_row(row)

    def save_order(self, order: PaperOrder) -> None:
        values = {
            "order_id": order.order_id,
            "decision_id": order.decision_id,
            "signal_id": order.signal_id,
            "created_at": order.created_at,
            "side": order.side,
            "quantity_btc": order.quantity_btc,
            "fill_price": order.fill_price,
            "status": order.status,
            "filled_at": order.filled_at,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(self.orders.c.order_id).where(self.orders.c.order_id == order.order_id)
            ).first()
            if existing is None:
                conn.execute(insert(self.orders).values(**values))
            else:
                conn.execute(
                    update(self.orders)
                    .where(self.orders.c.order_id == order.order_id)
                    .values(**values)
                )

    def get_orders(self, limit: int = 20) -> list[PaperOrder]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.orders).order_by(self.orders.c.created_at.desc()).limit(limit)
            ).all()
        return [self._order_from_row(row) for row in rows]

    def clear_orders(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(self.orders))

    def save_control_state(self, control: ControlState) -> None:
        values = {
            "id": 1,
            "trading_paused": control.trading_paused,
            "retrain_requested": control.retrain_requested,
            "last_acknowledged_alert": control.last_acknowledged_alert,
            "updated_at": control.updated_at,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(select(self.controls.c.id)).first()
            if existing is None:
                conn.execute(insert(self.controls).values(**values))
            else:
                conn.execute(update(self.controls).where(self.controls.c.id == 1).values(**values))

    def get_control_state(self) -> ControlState:
        with self.engine.begin() as conn:
            row = conn.execute(select(self.controls).limit(1)).first()
        if row is None:
            raise RuntimeError("control state has not been initialized")
        mapping = row._mapping
        return ControlState(
            trading_paused=bool(mapping["trading_paused"]),
            retrain_requested=bool(mapping["retrain_requested"]),
            last_acknowledged_alert=mapping["last_acknowledged_alert"],
            updated_at=self._normalize_dt(mapping["updated_at"]),
        )

    def _event_from_row(self, row: Row[Any]) -> RuntimeEvent:
        mapping = row._mapping
        return RuntimeEvent(
            id=int(mapping["id"]),
            event_type=str(mapping["event_type"]),
            occurred_at=self._normalize_dt(mapping["occurred_at"]),
            payload=json.loads(mapping["payload"]),
            stream_key=str(mapping["stream_key"]),
        )

    def _order_from_row(self, row: Row[Any]) -> PaperOrder:
        mapping = row._mapping
        return PaperOrder(
            order_id=str(mapping["order_id"]),
            decision_id=str(mapping["decision_id"]),
            signal_id=str(mapping["signal_id"]),
            created_at=self._normalize_dt(mapping["created_at"]),
            side=str(mapping["side"]),
            quantity_btc=float(mapping["quantity_btc"]),
            fill_price=float(mapping["fill_price"]),
            status=str(mapping["status"]),
            filled_at=self._normalize_dt(mapping["filled_at"]) if mapping["filled_at"] else None,
        )

    def _portfolio_from_row(self, row: Row[Any]) -> PortfolioSnapshot:
        mapping = row._mapping
        return PortfolioSnapshot(
            as_of=self._normalize_dt(mapping["as_of"]),
            cash=float(mapping["cash"]),
            position_qty=float(mapping["position_qty"]),
            avg_entry_price=float(mapping["avg_entry_price"]),
            mark_price=float(mapping["mark_price"]),
            realized_pnl=float(mapping["realized_pnl"]),
            unrealized_pnl=float(mapping["unrealized_pnl"]),
            equity=float(mapping["equity"]),
            status=str(mapping["status"]),
        )

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
