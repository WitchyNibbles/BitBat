"""Deterministic strategy definitions for BitBat v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import BitBatV2Config
from .domain import Candle

EPSILON = 1e-9


@dataclass(frozen=True)
class StrategyMetrics:
    close: float
    open_to_close_return: float
    momentum_return: float
    range_ratio: float
    short_trend_return: float
    trend_return: float
    body_strength: float


@dataclass(frozen=True)
class StrategyContext:
    config: BitBatV2Config
    candle: Candle
    previous_candle: Candle | None
    history: list[Candle]


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy_name: str
    direction: str
    predicted_return: float
    predicted_price: float
    confidence: float
    reasons: list[str]
    block_reason: str | None
    metrics: StrategyMetrics


class Strategy(Protocol):
    name: str

    def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        ...


def _anchor_close(history: list[Candle], lookback: int, fallback: float) -> float:
    if len(history) >= lookback:
        return history[-lookback].close
    return fallback


def compute_metrics(context: StrategyContext) -> StrategyMetrics:
    candle = context.candle
    previous_close = context.previous_candle.close if context.previous_candle is not None else candle.open
    open_to_close_return = (candle.close - candle.open) / candle.open if candle.open else 0.0
    momentum_return = (
        (candle.close - previous_close) / previous_close
        if previous_close
        else open_to_close_return
    )
    range_ratio = (candle.high - candle.low) / candle.close if candle.close else 0.0
    body_strength = abs(candle.close - candle.open) / max(candle.high - candle.low, EPSILON)
    short_anchor = _anchor_close(
        context.history,
        context.config.short_trend_lookback_candles,
        previous_close,
    )
    trend_anchor = _anchor_close(
        context.history,
        context.config.trend_lookback_candles,
        previous_close,
    )
    short_trend_return = (
        (candle.close - short_anchor) / short_anchor if short_anchor else momentum_return
    )
    trend_return = (candle.close - trend_anchor) / trend_anchor if trend_anchor else momentum_return
    return StrategyMetrics(
        close=candle.close,
        open_to_close_return=open_to_close_return,
        momentum_return=momentum_return,
        range_ratio=range_ratio,
        short_trend_return=short_trend_return,
        trend_return=trend_return,
        body_strength=body_strength,
    )


def _confidence(predicted_return: float) -> float:
    return min(0.99, max(0.05, abs(predicted_return) * 250.0))


class BaselineV1Strategy:
    name = "baseline_v1"

    def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        metrics = compute_metrics(context)
        predicted_return = round(
            (metrics.momentum_return * 0.85)
            + (metrics.open_to_close_return * 0.35)
            - (metrics.range_ratio * 0.1),
            6,
        )
        if predicted_return >= context.config.signal_threshold:
            direction = "buy"
        elif predicted_return <= -context.config.signal_threshold:
            direction = "sell"
        else:
            direction = "hold"
        reasons = [
            f"momentum={metrics.momentum_return:.5f}",
            f"body={metrics.open_to_close_return:.5f}",
            f"range={metrics.range_ratio:.5f}",
        ]
        return StrategyEvaluation(
            strategy_name=self.name,
            direction=direction,
            predicted_return=predicted_return,
            predicted_price=round(metrics.close * (1.0 + predicted_return), 2),
            confidence=_confidence(predicted_return),
            reasons=reasons,
            block_reason=None,
            metrics=metrics,
        )


class FilteredMomentumV2Strategy:
    name = "filtered_momentum_v2"

    def evaluate(self, context: StrategyContext) -> StrategyEvaluation:
        metrics = compute_metrics(context)
        score = round(
            (metrics.momentum_return * 0.55)
            + (metrics.open_to_close_return * 0.25)
            + (metrics.short_trend_return * 0.25)
            + (metrics.trend_return * 0.2)
            - (metrics.range_ratio * 0.35),
            6,
        )
        reasons = [
            f"score={score:.6f}",
            f"momentum={metrics.momentum_return:.6f}",
            f"body={metrics.open_to_close_return:.6f}",
            f"short_trend={metrics.short_trend_return:.6f}",
            f"trend={metrics.trend_return:.6f}",
            f"range={metrics.range_ratio:.6f}",
            f"body_strength={metrics.body_strength:.6f}",
        ]

        buy_range_ok = metrics.range_ratio < context.config.max_range_ratio
        buy_body_ok = metrics.body_strength >= context.config.min_body_strength
        buy_trend_ok = metrics.trend_return > 0
        buy_short_ok = metrics.short_trend_return > -0.0005
        sell_trend_ok = metrics.trend_return < 0
        sell_short_ok = metrics.short_trend_return < 0.0005

        reasons.extend(
            [
                f"trend_gate={'pass' if buy_trend_ok or sell_trend_ok else 'fail'}",
                f"short_trend_gate={'pass' if buy_short_ok or sell_short_ok else 'fail'}",
                f"range_gate={'pass' if buy_range_ok else 'fail'}",
                f"body_gate={'pass' if buy_body_ok else 'fail'}",
            ]
        )

        if score >= context.config.signal_threshold:
            if buy_trend_ok and buy_short_ok and buy_range_ok and buy_body_ok:
                direction = "buy"
                block_reason = None
            else:
                direction = "hold"
                if not buy_range_ok:
                    block_reason = "volatility/range filter blocked entry"
                elif not buy_body_ok:
                    block_reason = "weak candle body blocked entry"
                elif not buy_trend_ok:
                    block_reason = "trend confirmation blocked entry"
                else:
                    block_reason = "short-term momentum filter blocked entry"
        elif score <= -context.config.sell_signal_threshold:
            if sell_trend_ok and sell_short_ok:
                direction = "sell"
                block_reason = None
            else:
                direction = "hold"
                if not sell_trend_ok:
                    block_reason = "trend confirmation blocked sell"
                else:
                    block_reason = "short-term momentum filter blocked sell"
        else:
            direction = "hold"
            block_reason = None

        return StrategyEvaluation(
            strategy_name=self.name,
            direction=direction,
            predicted_return=score,
            predicted_price=round(metrics.close * (1.0 + score), 2),
            confidence=_confidence(score),
            reasons=reasons,
            block_reason=block_reason,
            metrics=metrics,
        )


STRATEGIES: dict[str, Strategy] = {
    BaselineV1Strategy.name: BaselineV1Strategy(),
    FilteredMomentumV2Strategy.name: FilteredMomentumV2Strategy(),
}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown v2 strategy: {name}") from exc
