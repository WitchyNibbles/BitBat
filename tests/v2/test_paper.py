from __future__ import annotations

from datetime import UTC, datetime

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import PortfolioSnapshot, PredictionSignal, RuntimeEvent
from bitbat_v2.paper import (
    build_paper_cockpit_snapshot,
    build_paper_performance_summary,
    closed_trades_from_orders,
)


class _Order:
    def __init__(
        self,
        *,
        side: str,
        quantity_btc: float,
        fill_price: float,
        filled_at: datetime,
    ) -> None:
        self.order_id = f"ord-{side}-{filled_at.timestamp()}"
        self.decision_id = "dec-1"
        self.signal_id = "sig-1"
        self.created_at = filled_at
        self.side = side
        self.quantity_btc = quantity_btc
        self.fill_price = fill_price
        self.status = "filled"
        self.filled_at = filled_at


def test_closed_trades_from_orders_replays_net_pnl() -> None:
    orders = [
        _Order(
            side="buy",
            quantity_btc=0.01,
            fill_price=100_000.0,
            filled_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        ),
        _Order(
            side="sell",
            quantity_btc=0.01,
            fill_price=101_000.0,
            filled_at=datetime(2026, 4, 25, 10, 5, tzinfo=UTC),
        ),
    ]

    trades = closed_trades_from_orders(orders, fee_bps=10.0)

    assert len(trades) == 1
    assert trades[0].entry_price == 100_000.0
    assert trades[0].gross_pnl == 10.0
    assert trades[0].fees_paid == 2.01
    assert trades[0].net_pnl == 7.99


def test_build_paper_performance_summary_adds_buy_hold_and_drawdown() -> None:
    config = BitBatV2Config(starting_cash_usd=10_000.0, fee_bps=10.0)
    portfolio = PortfolioSnapshot(
        as_of=datetime(2026, 4, 25, 10, 10, tzinfo=UTC),
        cash=8_950.0,
        position_qty=0.01,
        avg_entry_price=100_100.0,
        mark_price=101_200.0,
        realized_pnl=0.0,
        unrealized_pnl=8.0,
        equity=9_960.0,
        status="paper",
    )
    signal = PredictionSignal(
        signal_id="sig-1",
        generated_at=datetime(2026, 4, 25, 10, 10, tzinfo=UTC),
        product_id="BTC-USD",
        venue="coinbase",
        model_name="ritual-momentum-v1",
        direction="buy",
        confidence=0.62,
        predicted_return=0.004,
        predicted_price=101_500.0,
        reasons=["score=0.004000"],
    )
    orders = [
        _Order(
            side="buy",
            quantity_btc=0.01,
            fill_price=100_100.0,
            filled_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        )
    ]
    portfolio_events = [
        RuntimeEvent(
            id=1,
            event_type="portfolio.updated",
            occurred_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            payload={
                "equity": 9_998.0,
                "cash": 8_998.0,
                "position_qty": 0.01,
                "mark_price": 100_100.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": -2.0,
            },
        ),
        RuntimeEvent(
            id=2,
            event_type="portfolio.updated",
            occurred_at=datetime(2026, 4, 25, 10, 10, tzinfo=UTC),
            payload={
                "equity": 9_960.0,
                "cash": 8_950.0,
                "position_qty": 0.01,
                "mark_price": 101_200.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 8.0,
            },
        ),
    ]
    decision_events = [
        RuntimeEvent(
            id=3,
            event_type="decision.made",
            occurred_at=portfolio.as_of,
            payload={"signal_id": "sig-1", "action": "buy", "reason": "positive edge"},
        )
    ]
    signal_events = [
        RuntimeEvent(
            id=4,
            event_type="signal.generated",
            occurred_at=portfolio.as_of,
            payload={**signal.to_dict(), "abstain_reason": None},
        )
    ]

    summary = build_paper_performance_summary(
        config=config,
        portfolio=portfolio,
        latest_signal=signal,
        last_event_at=portfolio.as_of,
        orders=orders,
        portfolio_events=portfolio_events,
        decision_events=decision_events,
        signal_events=signal_events,
    )

    assert summary.net_pnl == -40.0
    assert summary.benchmark_equity > 0
    assert summary.max_drawdown_pct <= 0.0
    assert summary.last_signal_direction == "buy"
    assert summary.action_rate == 1.0


def test_paper_cockpit_summary_uses_full_order_history_but_caps_recent_rows() -> None:
    config = BitBatV2Config(starting_cash_usd=10_000.0)
    portfolio = PortfolioSnapshot(
        as_of=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        cash=10_020.0,
        position_qty=0.0,
        avg_entry_price=0.0,
        mark_price=100_500.0,
        realized_pnl=20.0,
        unrealized_pnl=0.0,
        equity=10_020.0,
        status="paper",
    )
    orders = [
        _Order(
            side="buy" if idx % 2 == 0 else "sell",
            quantity_btc=0.01,
            fill_price=100_000.0 + idx,
            filled_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        )
        for idx in range(60)
    ]
    portfolio_events = [
        RuntimeEvent(
            id=1,
            event_type="portfolio.updated",
            occurred_at=portfolio.as_of,
            payload={
                "equity": portfolio.equity,
                "cash": portfolio.cash,
                "position_qty": portfolio.position_qty,
                "mark_price": portfolio.mark_price,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": portfolio.unrealized_pnl,
            },
        )
    ]
    decision_events = [
        RuntimeEvent(
            id=2,
            event_type="decision.made",
            occurred_at=portfolio.as_of,
            payload={"signal_id": "sig-hold", "action": "hold", "reason": "no edge"},
        )
    ]
    signal_events = [
        RuntimeEvent(
            id=3,
            event_type="signal.generated",
            occurred_at=portfolio.as_of,
            payload={
                "signal_id": "sig-hold",
                "generated_at": portfolio.as_of.isoformat(),
                "product_id": "BTC-USD",
                "venue": "coinbase",
                "model_name": "legacy_xgb_5m_30m",
                "direction": "hold",
                "confidence": 0.0,
                "predicted_return": 0.0,
                "predicted_price": 100_500.0,
                "reasons": ["signal_source=legacy_ml"],
                "abstain_reason": "replay gate blocked entry",
            },
        )
    ]

    snapshot = build_paper_cockpit_snapshot(
        config=config,
        portfolio=portfolio,
        latest_signal=None,
        last_event_at=portfolio.as_of,
        orders=orders,
        portfolio_events=portfolio_events,
        alert_events=[],
        decision_events=decision_events,
        signal_events=signal_events,
    )

    assert snapshot.performance.trade_count == 60
    assert len(snapshot.recent_orders) == 50
    assert snapshot.performance.hold_rate == 1.0
    assert snapshot.performance.abstain_breakdown["replay gate blocked entry"] == 1
