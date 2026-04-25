"""Offline evaluation helpers for BitBat v2 strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import pandas as pd

from .config import BitBatV2Config
from .domain import Candle
from .strategy import StrategyContext, get_strategy

REQUIRED_PRICE_COLUMNS = {"timestamp_utc", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class SimulationSummary:
    strategy_name: str
    trade_count: int
    buy_count: int
    sell_count: int
    hold_rate: float
    final_equity: float
    realized_pnl: float
    unrealized_pnl: float
    max_drawdown_pct: float
    ending_position_qty: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "strategy_name": self.strategy_name,
            "trade_count": self.trade_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_rate": self.hold_rate,
            "final_equity": self.final_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "ending_position_qty": self.ending_position_qty,
        }


def load_candles_from_parquet(path: str | Path, config: BitBatV2Config) -> list[Candle]:
    parquet_path = Path(path)
    frame = pd.read_parquet(parquet_path)
    missing = sorted(REQUIRED_PRICE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    candles = [
        Candle(
            product_id=config.product_id,
            granularity_seconds=config.granularity_seconds,
            start=timestamps.iloc[idx].to_pydatetime().astimezone(UTC),
            open=float(frame.iloc[idx]["open"]),
            high=float(frame.iloc[idx]["high"]),
            low=float(frame.iloc[idx]["low"]),
            close=float(frame.iloc[idx]["close"]),
            volume=float(frame.iloc[idx]["volume"]),
        )
        for idx in range(len(frame))
    ]
    return candles


def simulate_strategy(
    candles: list[Candle],
    config: BitBatV2Config,
    strategy_name: str,
) -> SimulationSummary:
    strategy = get_strategy(strategy_name)
    cash = config.starting_cash_usd
    position_qty = 0.0
    avg_entry_price = 0.0
    realized_pnl = 0.0
    history: list[Candle] = []
    buy_count = 0
    sell_count = 0
    hold_count = 0
    equity_curve: list[float] = []

    for candle in candles:
        previous_candle = history[-1] if history else None
        evaluation = strategy.evaluate(
            StrategyContext(
                config=config,
                candle=candle,
                previous_candle=previous_candle,
                history=history,
            )
        )
        action = "hold"
        if evaluation.direction == "buy":
            if position_qty + config.order_size_btc <= config.max_position_size_btc:
                fill_cost = config.order_size_btc * candle.close
                new_position = position_qty + config.order_size_btc
                avg_entry_price = (
                    ((position_qty * avg_entry_price) + fill_cost) / new_position
                    if new_position
                    else 0.0
                )
                cash -= fill_cost
                position_qty = new_position
                buy_count += 1
                action = "buy"
        elif evaluation.direction == "sell" and position_qty >= config.order_size_btc:
            cash += config.order_size_btc * candle.close
            realized_pnl += config.order_size_btc * (candle.close - avg_entry_price)
            position_qty -= config.order_size_btc
            if position_qty == 0.0:
                avg_entry_price = 0.0
            sell_count += 1
            action = "sell"
        if action == "hold":
            hold_count += 1
        equity_curve.append(cash + (position_qty * candle.close))
        history.append(candle)

    equity_series = pd.Series(equity_curve, dtype="float64")
    drawdown = equity_series / equity_series.cummax() - 1.0 if len(equity_series) else pd.Series()
    final_mark = candles[-1].close if candles else 0.0
    unrealized_pnl = (final_mark - avg_entry_price) * position_qty if position_qty else 0.0
    return SimulationSummary(
        strategy_name=strategy_name,
        trade_count=buy_count + sell_count,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_rate=round(hold_count / len(candles), 4) if candles else 0.0,
        final_equity=round(float(equity_series.iloc[-1]), 2) if len(equity_series) else 0.0,
        realized_pnl=round(realized_pnl, 2),
        unrealized_pnl=round(unrealized_pnl, 2),
        max_drawdown_pct=round(float(drawdown.min()), 4) if len(drawdown) else 0.0,
        ending_position_qty=round(position_qty, 8),
    )


def compare_strategies(
    candles: list[Candle],
    config: BitBatV2Config,
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    baseline = simulate_strategy(candles, config, "baseline_v1")
    improved = simulate_strategy(candles, config, "filtered_momentum_v2")
    return {
        "strategies": {
            "baseline_v1": baseline.to_dict(),
            "filtered_momentum_v2": improved.to_dict(),
        },
        "delta": {
            "final_equity_delta": round(improved.final_equity - baseline.final_equity, 2),
            "realized_pnl_delta": round(improved.realized_pnl - baseline.realized_pnl, 2),
            "unrealized_pnl_delta": round(improved.unrealized_pnl - baseline.unrealized_pnl, 2),
            "max_drawdown_pct_delta": round(
                improved.max_drawdown_pct - baseline.max_drawdown_pct,
                4,
            ),
            "trade_count_delta": improved.trade_count - baseline.trade_count,
            "hold_rate_delta": round(improved.hold_rate - baseline.hold_rate, 4),
        },
    }
