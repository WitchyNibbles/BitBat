"""Paper trading projections for BitBat v2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import BitBatV2Config
from .domain import PortfolioSnapshot, PredictionSignal, RuntimeEvent


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class PaperPerformancePoint:
    occurred_at: datetime
    equity: float
    cash: float
    position_qty: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurred_at": _iso(self.occurred_at),
            "equity": self.equity,
            "cash": self.cash,
            "position_qty": self.position_qty,
            "mark_price": self.mark_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass(frozen=True)
class PaperTrade:
    closed_at: datetime
    quantity_btc: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    return_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed_at": _iso(self.closed_at),
            "quantity_btc": self.quantity_btc,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "fees_paid": self.fees_paid,
            "return_pct": self.return_pct,
        }


@dataclass(frozen=True)
class PaperAlert:
    occurred_at: datetime
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurred_at": _iso(self.occurred_at),
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class PaperPerformanceSummary:
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
    abstain_breakdown: dict[str, int] = field(default_factory=dict)
    last_signal_at: datetime | None = None
    last_signal_direction: str | None = None
    signal_confidence: float | None = None
    last_event_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": _iso(self.as_of),
            "starting_cash": self.starting_cash,
            "equity": self.equity,
            "cash": self.cash,
            "position_qty": self.position_qty,
            "mark_price": self.mark_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "net_pnl": self.net_pnl,
            "net_pnl_pct": self.net_pnl_pct,
            "fees_paid": self.fees_paid,
            "turnover_usd": self.turnover_usd,
            "trade_count": self.trade_count,
            "closed_trade_count": self.closed_trade_count,
            "win_rate": self.win_rate,
            "expectancy_per_trade": self.expectancy_per_trade,
            "max_drawdown_pct": self.max_drawdown_pct,
            "exposure_pct": self.exposure_pct,
            "benchmark_equity": self.benchmark_equity,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_vs_buy_hold": self.alpha_vs_buy_hold,
            "hold_rate": self.hold_rate,
            "action_rate": self.action_rate,
            "abstain_breakdown": dict(self.abstain_breakdown),
            "last_signal_at": _iso(self.last_signal_at),
            "last_signal_direction": self.last_signal_direction,
            "signal_confidence": self.signal_confidence,
            "last_event_at": _iso(self.last_event_at),
        }


@dataclass(frozen=True)
class PaperCockpitSnapshot:
    portfolio: PortfolioSnapshot
    performance: PaperPerformanceSummary
    latest_signal: PredictionSignal | None
    recent_alerts: list[PaperAlert] = field(default_factory=list)
    recent_orders: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[PaperPerformancePoint] = field(default_factory=list)


def portfolio_points_from_events(events: list[RuntimeEvent]) -> list[PaperPerformancePoint]:
    points: list[PaperPerformancePoint] = []
    for event in events:
        if event.event_type != "portfolio.updated":
            continue
        payload = event.payload
        points.append(
            PaperPerformancePoint(
                occurred_at=event.occurred_at,
                equity=float(payload.get("equity", 0.0)),
                cash=float(payload.get("cash", 0.0)),
                position_qty=float(payload.get("position_qty", 0.0)),
                mark_price=float(payload.get("mark_price", 0.0)),
                realized_pnl=float(payload.get("realized_pnl", 0.0)),
                unrealized_pnl=float(payload.get("unrealized_pnl", 0.0)),
            )
        )
    return points


def alerts_from_events(events: list[RuntimeEvent]) -> list[PaperAlert]:
    alerts: list[PaperAlert] = []
    for event in events:
        if event.event_type != "alert.raised":
            continue
        payload = event.payload
        alerts.append(
            PaperAlert(
                occurred_at=event.occurred_at,
                code=str(payload.get("code", "unknown")),
                message=str(payload.get("message", "")),
            )
        )
    return alerts


def closed_trades_from_orders(
    orders: Sequence[dict[str, Any] | Any],
    *,
    fee_bps: float,
) -> list[PaperTrade]:
    fee_rate = max(float(fee_bps), 0.0) / 10_000.0
    avg_entry_fill_price = 0.0
    avg_entry_cost_price = 0.0
    position_qty = 0.0
    trades: list[PaperTrade] = []

    def _attr(order: dict[str, Any] | Any, name: str) -> Any:
        if isinstance(order, dict):
            return order[name]
        return getattr(order, name)

    filled_orders = [order for order in orders if _attr(order, "status") == "filled"]
    sorted_orders = sorted(
        filled_orders,
        key=lambda order: _attr(order, "filled_at") or _attr(order, "created_at"),
    )
    for order in sorted_orders:
        side = str(_attr(order, "side"))
        quantity = float(_attr(order, "quantity_btc"))
        fill_price = float(_attr(order, "fill_price"))
        fee_paid = quantity * fill_price * fee_rate
        if side == "buy":
            fill_notional = quantity * fill_price
            fill_cost = fill_notional + fee_paid
            new_position = position_qty + quantity
            avg_entry_fill_price = (
                ((position_qty * avg_entry_fill_price) + fill_notional) / new_position
                if new_position
                else 0.0
            )
            avg_entry_cost_price = (
                ((position_qty * avg_entry_cost_price) + fill_cost) / new_position
                if new_position
                else 0.0
            )
            position_qty = new_position
            continue
        if side != "sell" or position_qty <= 0:
            continue
        quantity = min(quantity, position_qty)
        gross_pnl = (fill_price - avg_entry_fill_price) * quantity
        entry_fee_share = max(avg_entry_cost_price - avg_entry_fill_price, 0.0) * quantity
        total_fees = entry_fee_share + fee_paid
        net_pnl = gross_pnl - total_fees
        return_pct = (
            (net_pnl / (avg_entry_cost_price * quantity)) if avg_entry_cost_price > 0 else 0.0
        )
        trades.append(
            PaperTrade(
                closed_at=_attr(order, "filled_at") or _attr(order, "created_at"),
                quantity_btc=round(quantity, 8),
                entry_price=round(avg_entry_fill_price, 2),
                exit_price=round(fill_price, 2),
                gross_pnl=round(gross_pnl, 2),
                net_pnl=round(net_pnl, 2),
                fees_paid=round(total_fees, 2),
                return_pct=round(return_pct, 6),
            )
        )
        position_qty = round(max(position_qty - quantity, 0.0), 8)
        if position_qty == 0.0:
            avg_entry_fill_price = 0.0
            avg_entry_cost_price = 0.0
    return trades


def signal_activity_summary(
    *,
    decision_events: list[RuntimeEvent],
    signal_events: list[RuntimeEvent],
) -> tuple[float, float, dict[str, int]]:
    signal_by_id = {
        str(event.payload.get("signal_id")): event.payload
        for event in signal_events
        if event.event_type == "signal.generated"
    }
    hold_count = 0
    action_count = 0
    abstain_breakdown: dict[str, int] = {}

    for event in decision_events:
        if event.event_type != "decision.made":
            continue
        action = str(event.payload.get("action", "hold"))
        if action in {"buy", "sell"}:
            action_count += 1
            continue
        hold_count += 1
        signal_payload = signal_by_id.get(str(event.payload.get("signal_id")), {})
        reason = str(
            signal_payload.get("abstain_reason") or event.payload.get("reason") or "unspecified"
        )
        abstain_breakdown[reason] = abstain_breakdown.get(reason, 0) + 1

    total = hold_count + action_count
    if total == 0:
        return 0.0, 0.0, abstain_breakdown
    hold_rate = round(hold_count / total, 6)
    action_rate = round(action_count / total, 6)
    return hold_rate, action_rate, abstain_breakdown


def build_paper_performance_summary(
    *,
    config: BitBatV2Config,
    portfolio: PortfolioSnapshot,
    latest_signal: PredictionSignal | None,
    last_event_at: datetime | None,
    orders: list[Any],
    portfolio_events: list[RuntimeEvent],
    decision_events: list[RuntimeEvent],
    signal_events: list[RuntimeEvent],
) -> PaperPerformanceSummary:
    points = portfolio_points_from_events(portfolio_events)
    equity_series = pd.Series([point.equity for point in points], dtype="float64")
    if not equity_series.empty:
        drawdown = equity_series / equity_series.cummax() - 1.0
        max_drawdown_pct = round(float(drawdown.min()), 4)
    else:
        max_drawdown_pct = 0.0

    starting_cash = round(float(config.starting_cash_usd), 2)
    current_equity = round(float(portfolio.equity), 2)
    net_pnl = round(current_equity - starting_cash, 2)
    net_pnl_pct = round((net_pnl / starting_cash) if starting_cash else 0.0, 6)
    turnover_usd = round(
        sum(
            float(order.quantity_btc) * float(order.fill_price)
            for order in orders
            if order.status == "filled"
        ),
        2,
    )
    fee_rate = max(float(config.fee_bps), 0.0) / 10_000.0
    fees_paid = round(
        sum(
            float(order.quantity_btc) * float(order.fill_price) * fee_rate
            for order in orders
            if order.status == "filled"
        ),
        2,
    )
    trades = closed_trades_from_orders(orders, fee_bps=config.fee_bps)
    closed_trade_count = len(trades)
    win_rate = round(
        (sum(1 for trade in trades if trade.net_pnl > 0) / closed_trade_count)
        if closed_trade_count
        else 0.0,
        4,
    )
    expectancy = round(
        (sum(trade.net_pnl for trade in trades) / closed_trade_count)
        if closed_trade_count
        else 0.0,
        2,
    )
    first_mark = next(
        (point.mark_price for point in points if point.mark_price > 0), portfolio.mark_price
    )
    benchmark_equity = round(
        (starting_cash * (portfolio.mark_price / first_mark))
        if first_mark and portfolio.mark_price > 0
        else starting_cash,
        2,
    )
    benchmark_return_pct = round(
        ((benchmark_equity - starting_cash) / starting_cash) if starting_cash else 0.0,
        6,
    )
    exposure_pct = round(
        ((portfolio.position_qty * portfolio.mark_price) / current_equity)
        if current_equity
        else 0.0,
        4,
    )
    alpha_vs_buy_hold = round(net_pnl_pct - benchmark_return_pct, 6)
    hold_rate, action_rate, abstain_breakdown = signal_activity_summary(
        decision_events=decision_events,
        signal_events=signal_events,
    )
    return PaperPerformanceSummary(
        as_of=portfolio.as_of,
        starting_cash=starting_cash,
        equity=current_equity,
        cash=round(float(portfolio.cash), 2),
        position_qty=round(float(portfolio.position_qty), 8),
        mark_price=round(float(portfolio.mark_price), 2),
        realized_pnl=round(float(portfolio.realized_pnl), 2),
        unrealized_pnl=round(float(portfolio.unrealized_pnl), 2),
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl_pct,
        fees_paid=fees_paid,
        turnover_usd=turnover_usd,
        trade_count=sum(1 for order in orders if order.status == "filled"),
        closed_trade_count=closed_trade_count,
        win_rate=win_rate,
        expectancy_per_trade=expectancy,
        max_drawdown_pct=max_drawdown_pct,
        exposure_pct=exposure_pct,
        benchmark_equity=benchmark_equity,
        benchmark_return_pct=benchmark_return_pct,
        alpha_vs_buy_hold=alpha_vs_buy_hold,
        hold_rate=hold_rate,
        action_rate=action_rate,
        abstain_breakdown=abstain_breakdown,
        last_signal_at=latest_signal.generated_at if latest_signal else None,
        last_signal_direction=latest_signal.direction if latest_signal else None,
        signal_confidence=latest_signal.confidence if latest_signal else None,
        last_event_at=last_event_at,
    )


def build_paper_cockpit_snapshot(
    *,
    config: BitBatV2Config,
    portfolio: PortfolioSnapshot,
    latest_signal: PredictionSignal | None,
    last_event_at: datetime | None,
    orders: list[Any],
    portfolio_events: list[RuntimeEvent],
    alert_events: list[RuntimeEvent],
    decision_events: list[RuntimeEvent],
    signal_events: list[RuntimeEvent],
) -> PaperCockpitSnapshot:
    performance = build_paper_performance_summary(
        config=config,
        portfolio=portfolio,
        latest_signal=latest_signal,
        last_event_at=last_event_at,
        orders=orders,
        portfolio_events=portfolio_events,
        decision_events=decision_events,
        signal_events=signal_events,
    )
    recent_orders = [
        {
            "order_id": str(order.order_id),
            "created_at": _iso(order.created_at),
            "filled_at": _iso(order.filled_at),
            "side": str(order.side),
            "quantity_btc": round(float(order.quantity_btc), 8),
            "fill_price": round(float(order.fill_price), 2),
            "status": str(order.status),
            "notional_usd": round(float(order.quantity_btc) * float(order.fill_price), 2),
        }
        for order in orders[:50]
    ]
    return PaperCockpitSnapshot(
        portfolio=portfolio,
        performance=performance,
        latest_signal=latest_signal,
        recent_alerts=alerts_from_events(alert_events),
        recent_orders=recent_orders,
        equity_curve=portfolio_points_from_events(portfolio_events),
    )
