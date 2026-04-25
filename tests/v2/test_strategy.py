from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
from bitbat_v2.strategy import StrategyContext, get_strategy


def _make_context() -> StrategyContext:
    config = BitBatV2Config(signal_threshold=0.0014)
    current = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )
    previous = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=current.start - timedelta(seconds=300),
        open=99_500.0,
        high=100_100.0,
        low=99_300.0,
        close=99_900.0,
        volume=10.0,
    )
    history = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=current.start - timedelta(seconds=300 * offset),
            open=99_000.0 + (offset * 50.0),
            high=99_200.0 + (offset * 50.0),
            low=98_900.0 + (offset * 50.0),
            close=99_100.0 + (offset * 50.0),
            volume=8.0,
        )
        for offset in range(config.trend_lookback_candles, 0, -1)
    ]
    return StrategyContext(config=config, candle=current, previous_candle=previous, history=history)


def test_baseline_strategy_reproduces_old_heuristic_signal() -> None:
    context = _make_context()

    result = get_strategy("baseline_v1").evaluate(context)

    assert result.direction == "buy"
    assert result.predicted_return == 0.008164
    assert result.reasons[:3] == [
        "momentum=0.00801",
        "body=0.00700",
        "range=0.01092",
    ]


def test_filtered_strategy_buys_on_trend_aligned_continuation() -> None:
    context = _make_context()

    result = get_strategy("filtered_momentum_v2").evaluate(context)

    assert result.direction == "buy"
    assert any(reason.startswith("score=") for reason in result.reasons)
    assert any(reason == "trend_gate=pass" for reason in result.reasons)


def test_filtered_strategy_holds_on_noisy_weak_body_candle() -> None:
    config = BitBatV2Config()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
        open=100_000.0,
        high=102_000.0,
        low=99_000.0,
        close=100_100.0,
        volume=20.0,
    )
    history = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=candle.start - timedelta(seconds=300 * offset),
            open=99_000.0 + (offset * 40.0),
            high=99_200.0 + (offset * 40.0),
            low=98_900.0 + (offset * 40.0),
            close=99_150.0 + (offset * 40.0),
            volume=7.0,
        )
        for offset in range(config.trend_lookback_candles, 0, -1)
    ]
    context = StrategyContext(
        config=config,
        candle=candle,
        previous_candle=history[-1],
        history=history,
    )

    result = get_strategy("filtered_momentum_v2").evaluate(context)

    assert result.direction == "hold"
    assert "range_gate=fail" in result.reasons
    assert "body_gate=fail" in result.reasons


def test_filtered_strategy_sells_only_on_confirmed_downside() -> None:
    config = BitBatV2Config()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        open=99_400.0,
        high=99_450.0,
        low=98_500.0,
        close=98_700.0,
        volume=14.0,
    )
    history = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=candle.start - timedelta(seconds=300 * offset),
            open=100_500.0 - (offset * 70.0),
            high=100_550.0 - (offset * 70.0),
            low=100_250.0 - (offset * 70.0),
            close=100_300.0 - (offset * 70.0),
            volume=9.0,
        )
        for offset in range(config.trend_lookback_candles, 0, -1)
    ]
    context = StrategyContext(
        config=config,
        candle=candle,
        previous_candle=history[-1],
        history=history,
    )

    result = get_strategy("filtered_momentum_v2").evaluate(context)

    assert result.direction == "sell"
    assert "trend_gate=pass" in result.reasons
    assert any(reason.startswith("short_trend=") for reason in result.reasons)


def test_filtered_strategy_reasons_are_deterministic() -> None:
    result = get_strategy("filtered_momentum_v2").evaluate(_make_context())

    assert result.reasons[:7] == [
        "score=0.007990",
        "momentum=0.008008",
        "body=0.007000",
        "short_trend=0.014610",
        "trend=0.010030",
        "range=0.010924",
        "body_strength=0.636364",
    ]
