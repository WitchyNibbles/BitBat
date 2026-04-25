"""Runtime orchestration for BitBat v2."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .config import BitBatV2Config
from .domain import (
    Candle,
    ControlState,
    FeatureSnapshot,
    PaperOrder,
    PortfolioSnapshot,
    PredictionSignal,
    RuntimeEvent,
    RuntimeOutcome,
    StrategyDecision,
    utc_now,
)
from .storage import RuntimeStore
from .strategy import StrategyEvaluation, StrategyContext, compute_metrics, get_strategy


@dataclass
class EventBroker:
    """Lightweight in-process pubsub for SSE consumers."""

    subscribers: set[asyncio.Queue[RuntimeEvent]]

    def __init__(self) -> None:
        self.subscribers = set()

    def subscribe(self) -> asyncio.Queue[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=100)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[RuntimeEvent]) -> None:
        self.subscribers.discard(queue)

    def publish(self, event: RuntimeEvent) -> None:
        for queue in list(self.subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)


class BitBatRuntime:
    """Stateful runtime that turns candles into signals, decisions, and paper fills."""

    def __init__(
        self,
        store: RuntimeStore,
        config: BitBatV2Config,
        broker: EventBroker | None = None,
        now_fn: Callable[[], datetime] | None = None,
        strategy_name: str = "filtered_momentum_v2",
    ) -> None:
        self.store = store
        self.config = config
        self.broker = broker or EventBroker()
        self.now_fn = now_fn or utc_now
        self.strategy = get_strategy(strategy_name)

    def initialize(self) -> None:
        self.store.create_schema()
        self.store.ensure_seed_state(self.config.starting_cash_usd)

    def seed_demo_state(self) -> None:
        if self.store.count_events() > 0:
            return
        now = self.now_fn().astimezone(UTC)
        candle = Candle(
            product_id=self.config.product_id,
            granularity_seconds=self.config.granularity_seconds,
            start=now - timedelta(seconds=self.config.granularity_seconds),
            open=100_000.0,
            high=100_850.0,
            low=99_700.0,
            close=100_600.0,
            volume=14.2,
        )
        self.process_candle(candle)

    def process_candle(self, candle: Candle) -> RuntimeOutcome:
        self._validate_candle(candle)
        latest_candle = self.store.get_last_candle()
        if latest_candle is not None and latest_candle.start == candle.start:
            latest_signal = self.store.get_latest_signal()
            history = self.store.get_recent_candles(limit=self._strategy_history_limit() + 1)
            previous_candle = history[-2] if len(history) >= 2 else None
            context = StrategyContext(
                config=self.config,
                candle=candle,
                previous_candle=previous_candle,
                history=history[:-1] if history else [],
            )
            evaluation = self.strategy.evaluate(context)
            features = self._compute_features(evaluation)
            signal = latest_signal or self._build_signal(evaluation)
            return RuntimeOutcome(
                candle=candle,
                features=features,
                signal=signal,
                decision=StrategyDecision(
                    decision_id=f"dec-{uuid4().hex[:12]}",
                    signal_id=signal.signal_id,
                    decided_at=self.now_fn(),
                    action="hold",
                    quantity_btc=0.0,
                    reason="duplicate candle",
                ),
                order=None,
                portfolio=self.store.get_portfolio(),
            )
        recent_history = self.store.get_recent_candles(limit=self._strategy_history_limit())
        previous_candle = recent_history[-1] if recent_history else None
        context = StrategyContext(
            config=self.config,
            candle=candle,
            previous_candle=previous_candle,
            history=recent_history,
        )
        evaluation = self.strategy.evaluate(context)
        candle_event = self.store.append_event(
            "candle.closed",
            candle.to_dict(),
            occurred_at=candle.start,
        )
        self.broker.publish(candle_event)

        features = self._compute_features(evaluation)
        feature_event = self.store.append_event(
            "features.computed",
            features.to_dict(),
            occurred_at=features.generated_at,
        )
        self.broker.publish(feature_event)

        signal = self._build_signal(evaluation)
        self.store.save_latest_signal(signal)
        signal_event = self.store.append_event(
            "signal.generated",
            signal.to_dict(),
            occurred_at=signal.generated_at,
        )
        self.broker.publish(signal_event)

        decision = self._make_decision(signal, candle, evaluation)
        decision_event = self.store.append_event(
            "decision.made",
            decision.to_dict(),
            occurred_at=decision.decided_at,
        )
        self.broker.publish(decision_event)

        order: PaperOrder | None = None
        portfolio = self.store.get_portfolio()
        if decision.action in {"buy", "sell"}:
            order = self._fill_order(decision, candle.close)
            self.store.save_order(order)
            order_event = self.store.append_event(
                "order.paper_filled",
                order.to_dict(),
                occurred_at=order.filled_at or order.created_at,
            )
            self.broker.publish(order_event)
            portfolio = self._apply_order(order, portfolio, candle.close)
        else:
            portfolio = self._mark_portfolio(portfolio, candle.close)

        self.store.save_portfolio(portfolio)
        portfolio_event = self.store.append_event(
            "portfolio.updated",
            portfolio.to_dict(),
            occurred_at=portfolio.as_of,
        )
        self.broker.publish(portfolio_event)

        return RuntimeOutcome(
            candle=candle,
            features=features,
            signal=signal,
            decision=decision,
            order=order,
            portfolio=portfolio,
        )

    def pause_trading(self) -> ControlState:
        current = self.store.get_control_state()
        updated = ControlState(
            trading_paused=True,
            retrain_requested=current.retrain_requested,
            last_acknowledged_alert=current.last_acknowledged_alert,
            updated_at=self.now_fn(),
        )
        self.store.save_control_state(updated)
        return updated

    def resume_trading(self) -> ControlState:
        current = self.store.get_control_state()
        updated = ControlState(
            trading_paused=False,
            retrain_requested=current.retrain_requested,
            last_acknowledged_alert=current.last_acknowledged_alert,
            updated_at=self.now_fn(),
        )
        self.store.save_control_state(updated)
        return updated

    def request_retrain(self) -> ControlState:
        current = self.store.get_control_state()
        updated = ControlState(
            trading_paused=current.trading_paused,
            retrain_requested=True,
            last_acknowledged_alert=current.last_acknowledged_alert,
            updated_at=self.now_fn(),
        )
        self.store.save_control_state(updated)
        return updated

    def acknowledge_alert(self, message: str) -> ControlState:
        current = self.store.get_control_state()
        updated = ControlState(
            trading_paused=current.trading_paused,
            retrain_requested=current.retrain_requested,
            last_acknowledged_alert=message,
            updated_at=self.now_fn(),
        )
        self.store.save_control_state(updated)
        return updated

    def reset_paper_account(self) -> PortfolioSnapshot:
        self.store.clear_orders()
        portfolio = PortfolioSnapshot(
            as_of=self.now_fn(),
            cash=self.config.starting_cash_usd,
            position_qty=0.0,
            avg_entry_price=0.0,
            mark_price=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            equity=self.config.starting_cash_usd,
            status="paper",
        )
        self.store.save_portfolio(portfolio)
        event = self.store.append_event(
            "portfolio.updated",
            portfolio.to_dict(),
            occurred_at=portfolio.as_of,
        )
        self.broker.publish(event)
        return portfolio

    def _compute_features(self, evaluation: StrategyEvaluation) -> FeatureSnapshot:
        metrics = evaluation.metrics
        return FeatureSnapshot(
            generated_at=self.now_fn(),
            product_id=self.config.product_id,
            close=metrics.close,
            open_to_close_return=metrics.open_to_close_return,
            momentum_return=metrics.momentum_return,
            range_ratio=metrics.range_ratio,
            short_trend_return=metrics.short_trend_return,
            trend_return=metrics.trend_return,
            body_strength=metrics.body_strength,
        )

    def _build_signal(self, evaluation: StrategyEvaluation) -> PredictionSignal:
        return PredictionSignal(
            signal_id=f"sig-{uuid4().hex[:12]}",
            generated_at=self.now_fn(),
            product_id=self.config.product_id,
            venue=self.config.venue,
            model_name=self.config.model_name,
            direction=evaluation.direction,
            confidence=evaluation.confidence,
            predicted_return=evaluation.predicted_return,
            predicted_price=evaluation.predicted_price,
            reasons=evaluation.reasons,
        )

    def _make_decision(
        self,
        signal: PredictionSignal,
        candle: Candle,
        evaluation: StrategyEvaluation,
    ) -> StrategyDecision:
        control = self.store.get_control_state()
        portfolio = self.store.get_portfolio()
        candle_closed_at = candle.start + timedelta(seconds=candle.granularity_seconds)
        stale = (self.now_fn() - candle_closed_at).total_seconds() > self.config.stale_after_seconds
        if stale:
            self._raise_alert("stale_market_data", "Candle was too old for execution.")
            return StrategyDecision(
                decision_id=f"dec-{uuid4().hex[:12]}",
                signal_id=signal.signal_id,
                decided_at=self.now_fn(),
                action="hold",
                quantity_btc=0.0,
                reason="stale data kill switch",
                stale_data=True,
                trading_paused=control.trading_paused,
            )
        if control.trading_paused:
            self._raise_alert("trading_paused", "Operator pause prevented execution.")
            return StrategyDecision(
                decision_id=f"dec-{uuid4().hex[:12]}",
                signal_id=signal.signal_id,
                decided_at=self.now_fn(),
                action="hold",
                quantity_btc=0.0,
                reason="operator pause",
                stale_data=False,
                trading_paused=True,
            )
        if signal.direction == "buy":
            if (
                portfolio.position_qty + self.config.order_size_btc
                > self.config.max_position_size_btc
            ):
                self._raise_alert("risk_cap", "Position cap prevented a new buy order.")
                return StrategyDecision(
                    decision_id=f"dec-{uuid4().hex[:12]}",
                    signal_id=signal.signal_id,
                    decided_at=self.now_fn(),
                    action="hold",
                    quantity_btc=0.0,
                    reason="risk cap reached",
                )
            return StrategyDecision(
                decision_id=f"dec-{uuid4().hex[:12]}",
                signal_id=signal.signal_id,
                decided_at=self.now_fn(),
                action="buy",
                quantity_btc=self.config.order_size_btc,
                reason="confirmed positive trend and score above threshold",
            )
        if signal.direction == "sell" and portfolio.position_qty >= self.config.order_size_btc:
            return StrategyDecision(
                decision_id=f"dec-{uuid4().hex[:12]}",
                signal_id=signal.signal_id,
                decided_at=self.now_fn(),
                action="sell",
                quantity_btc=self.config.order_size_btc,
                reason="confirmed downside trend and score below sell threshold",
            )
        return StrategyDecision(
            decision_id=f"dec-{uuid4().hex[:12]}",
            signal_id=signal.signal_id,
            decided_at=self.now_fn(),
            action="hold",
            quantity_btc=0.0,
            reason=evaluation.block_reason or "no valid spot action",
        )

    def _fill_order(self, decision: StrategyDecision, price: float) -> PaperOrder:
        filled_at = self.now_fn()
        return PaperOrder(
            order_id=f"ord-{uuid4().hex[:12]}",
            decision_id=decision.decision_id,
            signal_id=decision.signal_id,
            created_at=decision.decided_at,
            side=decision.action,
            quantity_btc=decision.quantity_btc,
            fill_price=price,
            status="filled",
            filled_at=filled_at,
        )

    def _apply_order(
        self,
        order: PaperOrder,
        portfolio: PortfolioSnapshot,
        mark_price: float,
    ) -> PortfolioSnapshot:
        cash = portfolio.cash
        position_qty = portfolio.position_qty
        avg_entry_price = portfolio.avg_entry_price
        realized_pnl = portfolio.realized_pnl
        if order.side == "buy":
            fill_cost = order.quantity_btc * order.fill_price
            new_position = position_qty + order.quantity_btc
            avg_entry_price = (
                ((position_qty * avg_entry_price) + fill_cost) / new_position
                if new_position
                else 0.0
            )
            cash -= fill_cost
            position_qty = new_position
        elif order.side == "sell" and position_qty >= order.quantity_btc:
            cash += order.quantity_btc * order.fill_price
            realized_pnl += order.quantity_btc * (order.fill_price - avg_entry_price)
            position_qty -= order.quantity_btc
            if position_qty == 0.0:
                avg_entry_price = 0.0
        unrealized_pnl = (mark_price - avg_entry_price) * position_qty if position_qty else 0.0
        equity = cash + (position_qty * mark_price)
        return PortfolioSnapshot(
            as_of=self.now_fn(),
            cash=round(cash, 2),
            position_qty=round(position_qty, 8),
            avg_entry_price=round(avg_entry_price, 2),
            mark_price=round(mark_price, 2),
            realized_pnl=round(realized_pnl, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            equity=round(equity, 2),
            status="paper",
        )

    def _mark_portfolio(self, portfolio: PortfolioSnapshot, mark_price: float) -> PortfolioSnapshot:
        unrealized_pnl = (
            (mark_price - portfolio.avg_entry_price) * portfolio.position_qty
            if portfolio.position_qty
            else 0.0
        )
        equity = portfolio.cash + (portfolio.position_qty * mark_price)
        return PortfolioSnapshot(
            as_of=self.now_fn(),
            cash=portfolio.cash,
            position_qty=portfolio.position_qty,
            avg_entry_price=portfolio.avg_entry_price,
            mark_price=round(mark_price, 2),
            realized_pnl=portfolio.realized_pnl,
            unrealized_pnl=round(unrealized_pnl, 2),
            equity=round(equity, 2),
            status=portfolio.status,
        )

    def _raise_alert(self, code: str, message: str) -> RuntimeEvent:
        payload = {
            "code": code,
            "message": message,
            "raised_at": self.now_fn().astimezone(UTC).isoformat(),
        }
        event = self.store.append_event("alert.raised", payload, occurred_at=self.now_fn())
        self.broker.publish(event)
        return event

    def _validate_candle(self, candle: Candle) -> None:
        if candle.product_id != self.config.product_id:
            raise ValueError("unexpected product for v2 runtime")
        if candle.granularity_seconds != self.config.granularity_seconds:
            raise ValueError("unexpected candle granularity for v2 runtime")
        if min(candle.open, candle.high, candle.low, candle.close) <= 0:
            raise ValueError("candle prices must be positive")
        if candle.low > candle.high:
            raise ValueError("candle low cannot exceed high")
        if not (candle.low <= candle.open <= candle.high):
            raise ValueError("candle open must be inside the candle range")
        if not (candle.low <= candle.close <= candle.high):
            raise ValueError("candle close must be inside the candle range")
        if candle.volume < 0:
            raise ValueError("candle volume cannot be negative")
        latest_allowed = self.now_fn() + timedelta(seconds=self.config.granularity_seconds)
        if candle.start > latest_allowed:
            raise ValueError("future candles are not accepted")

    def _strategy_history_limit(self) -> int:
        return max(
            int(self.config.trend_lookback_candles),
            int(self.config.short_trend_lookback_candles),
        )


def format_sse(event: RuntimeEvent) -> str:
    return (
        f"id: {event.id}\n"
        f"event: {event.event_type}\n"
        f"data: {json.dumps(event.payload)}\n\n"
    )
