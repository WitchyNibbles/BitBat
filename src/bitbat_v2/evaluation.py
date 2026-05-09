"""Offline evaluation helpers for BitBat v2 strategies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BitBatV2Config
from .domain import Candle
from .paper import build_paper_performance_summary
from .runtime import BitBatRuntime
from .storage import RuntimeStore

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
    fees_paid: float
    turnover_usd: float
    benchmark_equity: float
    benchmark_return_pct: float
    alpha_vs_buy_hold: float

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
            "fees_paid": self.fees_paid,
            "turnover_usd": self.turnover_usd,
            "benchmark_equity": self.benchmark_equity,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_vs_buy_hold": self.alpha_vs_buy_hold,
        }


@dataclass(frozen=True)
class RuntimeReplaySummary:
    signal_source: str
    model_name: str
    trade_count: int
    buy_count: int
    sell_count: int
    hold_rate: float
    action_rate: float
    final_equity: float
    net_pnl_pct: float
    max_drawdown_pct: float
    alpha_vs_buy_hold: float
    mean_expected_value_return: float
    calibration_brier: float | None
    abstain_breakdown: dict[str, int]
    replay_start: str | None
    replay_end: str | None
    runtime_compatible: bool = True
    compatibility_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_source": self.signal_source,
            "model_name": self.model_name,
            "trade_count": self.trade_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_rate": self.hold_rate,
            "action_rate": self.action_rate,
            "final_equity": self.final_equity,
            "net_pnl_pct": self.net_pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "alpha_vs_buy_hold": self.alpha_vs_buy_hold,
            "mean_expected_value_return": self.mean_expected_value_return,
            "calibration_brier": self.calibration_brier,
            "abstain_breakdown": dict(self.abstain_breakdown),
            "replay_start": self.replay_start,
            "replay_end": self.replay_end,
            "runtime_compatible": self.runtime_compatible,
            "compatibility_reason": self.compatibility_reason,
        }


def load_candles_from_parquet(path: str | Path, config: BitBatV2Config) -> list[Candle]:
    parquet_path = Path(path)
    frame = pd.read_parquet(parquet_path)
    missing = sorted(REQUIRED_PRICE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return [
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


def simulate_strategy(
    candles: list[Candle],
    config: BitBatV2Config,
    strategy_name: str,
) -> SimulationSummary:
    runtime_config = BitBatV2Config(**{
        **config.__dict__,
        "signal_source": "heuristic",
        "model_name": strategy_name,
        "database_url": "sqlite:///:memory:",
        "demo_mode": False,
        "autorun_enabled": False,
    })
    clock = {"now": candles[0].start if candles else pd.Timestamp.now(tz="UTC").to_pydatetime()}
    runtime = BitBatRuntime(
        store=RuntimeStore(runtime_config.database_url),
        config=runtime_config,
        now_fn=lambda: clock["now"],
    )
    runtime.initialize()

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for candle in candles:
        clock["now"] = candle.start + timedelta(seconds=candle.granularity_seconds)
        outcome = runtime.process_candle(candle)
        if outcome.decision.action == "buy":
            buy_count += 1
        elif outcome.decision.action == "sell":
            sell_count += 1
        else:
            hold_count += 1

    orders = runtime.store.get_orders(limit=None)
    portfolio = runtime.store.get_portfolio()
    summary = build_paper_performance_summary(
        config=runtime_config,
        portfolio=portfolio,
        latest_signal=runtime.store.get_latest_signal(),
        last_event_at=runtime.store.get_last_event_at(),
        orders=orders,
        portfolio_events=runtime.store.list_events_by_type("portfolio.updated", limit=None),
        decision_events=runtime.store.list_events_by_type("decision.made", limit=None),
        signal_events=runtime.store.list_events_by_type("signal.generated", limit=None),
    )
    return SimulationSummary(
        strategy_name=strategy_name,
        trade_count=buy_count + sell_count,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_rate=round(hold_count / len(candles), 4) if candles else 0.0,
        final_equity=summary.equity,
        realized_pnl=summary.realized_pnl,
        unrealized_pnl=summary.unrealized_pnl,
        max_drawdown_pct=summary.max_drawdown_pct,
        ending_position_qty=summary.position_qty,
        fees_paid=summary.fees_paid,
        turnover_usd=summary.turnover_usd,
        benchmark_equity=summary.benchmark_equity,
        benchmark_return_pct=summary.benchmark_return_pct,
        alpha_vs_buy_hold=summary.alpha_vs_buy_hold,
    )


def unsupported_runtime_replay_summary(
    *,
    signal_source: str,
    model_name: str,
    compatibility_reason: str,
) -> RuntimeReplaySummary:
    return RuntimeReplaySummary(
        signal_source=signal_source,
        model_name=model_name,
        trade_count=0,
        buy_count=0,
        sell_count=0,
        hold_rate=1.0,
        action_rate=0.0,
        final_equity=0.0,
        net_pnl_pct=0.0,
        max_drawdown_pct=0.0,
        alpha_vs_buy_hold=0.0,
        mean_expected_value_return=0.0,
        calibration_brier=None,
        abstain_breakdown={},
        replay_start=None,
        replay_end=None,
        runtime_compatible=False,
        compatibility_reason=compatibility_reason,
    )


def _resolve_horizon_bars(
    *,
    granularity_seconds: int,
    horizon: str,
) -> int:
    horizon_seconds = int(pd.Timedelta(horizon).total_seconds())
    return max(int(round(horizon_seconds / granularity_seconds)), 1)


def _realized_direction(
    current_close: float,
    future_close: float,
    *,
    tau: float,
) -> str:
    realized_return = (future_close / current_close) - 1.0
    if realized_return >= tau:
        return "up"
    if realized_return <= -tau:
        return "down"
    return "flat"


def _multiclass_brier_score(
    outcomes: list[tuple[dict[str, float], str]],
) -> float | None:
    if not outcomes:
        return None
    total = 0.0
    for probabilities, actual in outcomes:
        for label in ("up", "down", "flat"):
            target = 1.0 if label == actual else 0.0
            total += (float(probabilities.get(label, 0.0)) - target) ** 2
    return round(total / len(outcomes), 6)


def simulate_legacy_model_replay(
    candles: list[Candle],
    config: BitBatV2Config,
    *,
    tau: float,
) -> RuntimeReplaySummary:
    runtime_config = BitBatV2Config(**{
        **config.__dict__,
        "signal_source": "legacy_ml",
        "database_url": "sqlite:///:memory:",
        "demo_mode": False,
        "autorun_enabled": False,
    })
    clock = {"now": candles[0].start if candles else pd.Timestamp.now(tz="UTC").to_pydatetime()}
    runtime = BitBatRuntime(
        store=RuntimeStore(runtime_config.database_url),
        config=runtime_config,
        now_fn=lambda: clock["now"],
    )
    runtime.initialize()

    buy_count = 0
    sell_count = 0
    hold_count = 0
    acted_expected_values: list[float] = []
    abstain_counter: Counter[str] = Counter()
    probabilities_with_realized: list[tuple[dict[str, float], str]] = []
    horizon_bars = _resolve_horizon_bars(
        granularity_seconds=runtime_config.granularity_seconds,
        horizon=runtime_config.legacy_signal_horizon,
    )

    for index, candle in enumerate(candles):
        clock["now"] = candle.start + timedelta(seconds=candle.granularity_seconds)
        outcome = runtime.process_candle(candle)
        if outcome.decision.action == "buy":
            buy_count += 1
            acted_expected_values.append(float(outcome.signal.expected_value_return))
        elif outcome.decision.action == "sell":
            sell_count += 1
            acted_expected_values.append(float(outcome.signal.expected_value_return))
        else:
            hold_count += 1
            reason = outcome.signal.abstain_reason or outcome.decision.reason
            abstain_counter[str(reason or "unspecified")] += 1

        future_index = index + horizon_bars
        if future_index >= len(candles):
            continue
        realized = _realized_direction(
            candle.close,
            candles[future_index].close,
            tau=tau,
        )
        probabilities_with_realized.append((
            {
                "up": float(outcome.signal.p_up),
                "down": float(outcome.signal.p_down),
                "flat": float(outcome.signal.p_flat),
            },
            realized,
        ))

    orders = runtime.store.get_orders(limit=None)
    portfolio = runtime.store.get_portfolio()
    summary = build_paper_performance_summary(
        config=runtime_config,
        portfolio=portfolio,
        latest_signal=runtime.store.get_latest_signal(),
        last_event_at=runtime.store.get_last_event_at(),
        orders=orders,
        portfolio_events=runtime.store.list_events_by_type("portfolio.updated", limit=None),
        decision_events=runtime.store.list_events_by_type("decision.made", limit=None),
        signal_events=runtime.store.list_events_by_type("signal.generated", limit=None),
    )
    latest_signal = runtime.store.get_latest_signal()
    return RuntimeReplaySummary(
        signal_source="legacy_ml",
        model_name=(
            latest_signal.model_name if latest_signal is not None else "legacy_xgb_unavailable"
        ),
        trade_count=buy_count + sell_count,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_rate=round(hold_count / len(candles), 6) if candles else 1.0,
        action_rate=round((buy_count + sell_count) / len(candles), 6) if candles else 0.0,
        final_equity=summary.equity,
        net_pnl_pct=summary.net_pnl_pct,
        max_drawdown_pct=summary.max_drawdown_pct,
        alpha_vs_buy_hold=summary.alpha_vs_buy_hold,
        mean_expected_value_return=(
            round(sum(acted_expected_values) / len(acted_expected_values), 6)
            if acted_expected_values
            else 0.0
        ),
        calibration_brier=_multiclass_brier_score(probabilities_with_realized),
        abstain_breakdown=dict(abstain_counter),
        replay_start=candles[0].start.astimezone(UTC).isoformat() if candles else None,
        replay_end=candles[-1].start.astimezone(UTC).isoformat() if candles else None,
    )


def compare_strategies(
    candles: list[Candle],
    config: BitBatV2Config,
) -> dict[str, Any]:
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
            "fees_paid_delta": round(improved.fees_paid - baseline.fees_paid, 2),
            "max_drawdown_pct_delta": round(
                improved.max_drawdown_pct - baseline.max_drawdown_pct,
                4,
            ),
            "alpha_vs_buy_hold_delta": round(
                improved.alpha_vs_buy_hold - baseline.alpha_vs_buy_hold,
                6,
            ),
            "trade_count_delta": improved.trade_count - baseline.trade_count,
            "hold_rate_delta": round(improved.hold_rate - baseline.hold_rate, 4),
        },
    }
