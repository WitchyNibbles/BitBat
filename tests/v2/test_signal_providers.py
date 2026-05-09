from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.domain import Candle
from bitbat_v2.signals import LegacyModelSignalProvider, build_signal_provider
from bitbat_v2.strategy import StrategyContext


def _context() -> StrategyContext:
    config = BitBatV2Config()
    candle = Candle(
        product_id="BTC-USD",
        granularity_seconds=300,
        start=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        open=100_000.0,
        high=100_900.0,
        low=99_800.0,
        close=100_700.0,
        volume=12.5,
    )
    history = [
        Candle(
            product_id="BTC-USD",
            granularity_seconds=300,
            start=candle.start - timedelta(seconds=300 * offset),
            open=99_000.0 + (offset * 50.0),
            high=99_200.0 + (offset * 50.0),
            low=98_900.0 + (offset * 50.0),
            close=99_100.0 + (offset * 50.0),
            volume=8.0,
        )
        for offset in range(config.trend_lookback_candles, 0, -1)
    ]
    return StrategyContext(
        config=config,
        candle=candle,
        previous_candle=history[-1],
        history=history,
    )


def test_build_signal_provider_uses_heuristic_by_default() -> None:
    provider = build_signal_provider(BitBatV2Config())

    result = provider.evaluate(_context())

    assert result.model_name == "filtered_momentum_v2"
    assert result.direction in {"buy", "sell", "hold"}


def test_build_signal_provider_rejects_unknown_signal_source() -> None:
    with pytest.raises(ValueError, match="Unsupported BITBAT_V2_SIGNAL_SOURCE"):
        build_signal_provider(BitBatV2Config(signal_source="typo-mode"))


def test_legacy_model_provider_returns_hold_when_model_missing(tmp_path: Path) -> None:
    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
        )
    )

    from bitbat_v2 import signals as signal_module

    signal_module.get_runtime_config = lambda: {"data_dir": str(tmp_path)}
    signal_module.load_config = lambda: {"data_dir": str(tmp_path)}
    signal_module.resolve_models_dir = lambda cfg: tmp_path

    result = provider.evaluate(_context())

    assert result.direction == "hold"
    assert result.model_name == "legacy_xgb_unavailable"
    assert any("legacy model artifact missing" in reason for reason in result.reasons)


def test_legacy_model_provider_maps_classifier_probabilities_to_trade_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.json").write_text("fake-model", encoding="utf-8")
    (model_dir / "xgb.meta.json").write_text(
        json.dumps({
            "family": "xgb",
            "label_mode": "direction",
            "version": "test-model-v1",
            "freq": "5m",
            "horizon": "30m",
        }),
        encoding="utf-8",
    )

    price_index = pd.date_range("2026-04-25 06:00:00", periods=40, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100_000.0 + idx for idx in range(40)],
            "high": [100_050.0 + idx for idx in range(40)],
            "low": [99_950.0 + idx for idx in range(40)],
            "close": [100_020.0 + idx for idx in range(40)],
            "volume": [10.0] * 40,
        },
        index=price_index,
    )
    prices.index.name = "timestamp_utc"

    def _fake_generate_price_features(
        prices: pd.DataFrame,
        enable_garch: bool,
        freq: str,
    ) -> pd.DataFrame:
        aligned_index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        features = pd.DataFrame({"feat_stub": [0.5] * len(prices)}, index=aligned_index)
        features.index.name = "timestamp_utc"
        return features

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(
        signal_module,
        "get_runtime_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(
        signal_module,
        "load_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")
    monkeypatch.setattr(signal_module, "_load_ingested_prices", lambda data_dir, freq: prices)
    monkeypatch.setattr(signal_module, "generate_price_features", _fake_generate_price_features)
    monkeypatch.setattr(
        signal_module,
        "load_model",
        lambda path, expected_label_mode=None: type(
            "FakeBooster",
            (),
            {"feature_names": ["feat_stub"]},
        )(),
    )
    monkeypatch.setattr(
        signal_module,
        "predict_bar",
        lambda booster, feature_row, timestamp, current_price, tau: {
            "predicted_direction": "up",
            "predicted_price": current_price * 1.01,
            "p_up": 0.72,
            "p_down": 0.18,
            "p_flat": 0.10,
            "confidence": 0.72,
        },
    )

    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
        )
    )

    result = provider.evaluate(_context())

    assert result.model_name == "legacy_xgb_5m_30m"
    assert result.direction == "buy"
    assert result.predicted_return > 0
    assert result.confidence == 0.72
    assert "signal_source=legacy_ml" in result.reasons
    assert "artifact_label_mode=direction" in result.reasons


def test_legacy_model_provider_emits_ev_evidence_from_classifier_probs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.json").write_text("fake-model", encoding="utf-8")
    (model_dir / "xgb.meta.json").write_text(
        json.dumps({
            "family": "xgb",
            "label_mode": "direction",
            "version": "test-model-v1",
            "freq": "5m",
            "horizon": "30m",
        }),
        encoding="utf-8",
    )

    price_index = pd.date_range("2026-04-25 06:00:00", periods=40, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100_000.0 + idx for idx in range(40)],
            "high": [100_050.0 + idx for idx in range(40)],
            "low": [99_950.0 + idx for idx in range(40)],
            "close": [100_020.0 + idx for idx in range(40)],
            "volume": [10.0] * 40,
        },
        index=price_index,
    )
    prices.index.name = "timestamp_utc"

    def _fake_generate_price_features(
        prices: pd.DataFrame,
        enable_garch: bool,
        freq: str,
    ) -> pd.DataFrame:
        aligned_index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        features = pd.DataFrame({"feat_stub": [0.5] * len(prices)}, index=aligned_index)
        features.index.name = "timestamp_utc"
        return features

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(
        signal_module,
        "get_runtime_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(
        signal_module,
        "load_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")
    monkeypatch.setattr(signal_module, "_load_ingested_prices", lambda data_dir, freq: prices)
    monkeypatch.setattr(signal_module, "generate_price_features", _fake_generate_price_features)
    monkeypatch.setattr(
        signal_module,
        "load_model",
        lambda path, expected_label_mode=None: type(
            "FakeBooster",
            (),
            {"feature_names": ["feat_stub"]},
        )(),
    )
    monkeypatch.setattr(
        signal_module,
        "predict_bar",
        lambda booster, feature_row, timestamp, current_price, tau: {
            "predicted_direction": "up",
            "predicted_price": current_price * 1.01,
            "p_up": 0.72,
            "p_down": 0.18,
            "p_flat": 0.10,
            "confidence": 0.72,
        },
    )

    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
            fee_bps=4.0,
            slippage_bps=1.0,
        )
    )

    result = provider.evaluate(_context())

    assert result.p_up == pytest.approx(0.72)
    assert result.p_down == pytest.approx(0.18)
    assert result.p_flat == pytest.approx(0.10)
    assert result.expected_move_return == pytest.approx(result.predicted_return)
    assert result.expected_cost_return == pytest.approx(0.001)
    assert result.expected_value_return == pytest.approx(0.0044)
    assert result.abstain_reason is None
    assert "artifact_version=test-model-v1" in result.reasons


def test_legacy_model_provider_uses_meta_label_action_artifact_to_abstain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.side.json").write_text("fake-side", encoding="utf-8")
    (model_dir / "xgb.side.meta.json").write_text(
        json.dumps({
            "family": "xgb",
            "label_mode": "direction",
            "artifact_role": "side",
            "version": "side-v1",
            "freq": "5m",
            "horizon": "30m",
        }),
        encoding="utf-8",
    )
    (model_dir / "xgb.action.meta_label.json").write_text("fake-action", encoding="utf-8")
    (model_dir / "xgb.action.meta_label.meta.json").write_text(
        json.dumps({
            "family": "xgb",
            "label_mode": "meta_label",
            "artifact_role": "action",
            "version": "action-v1",
            "freq": "5m",
            "horizon": "30m",
        }),
        encoding="utf-8",
    )

    price_index = pd.date_range("2026-04-25 06:00:00", periods=40, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100_000.0 + idx for idx in range(40)],
            "high": [100_050.0 + idx for idx in range(40)],
            "low": [99_950.0 + idx for idx in range(40)],
            "close": [100_020.0 + idx for idx in range(40)],
            "volume": [10.0] * 40,
        },
        index=price_index,
    )
    prices.index.name = "timestamp_utc"

    def _fake_generate_price_features(
        prices: pd.DataFrame,
        enable_garch: bool,
        freq: str,
    ) -> pd.DataFrame:
        aligned_index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        features = pd.DataFrame({"feat_stub": [0.5] * len(prices)}, index=aligned_index)
        features.index.name = "timestamp_utc"
        return features

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(
        signal_module,
        "get_runtime_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(
        signal_module,
        "load_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")
    monkeypatch.setattr(signal_module, "_load_ingested_prices", lambda data_dir, freq: prices)
    monkeypatch.setattr(signal_module, "generate_price_features", _fake_generate_price_features)
    monkeypatch.setattr(
        signal_module,
        "load_model",
        lambda path, expected_label_mode=None: type(
            "FakeBooster",
            (),
            {"feature_names": ["feat_stub"]},
        )(),
    )
    monkeypatch.setattr(
        signal_module,
        "predict_bar",
        lambda booster, feature_row, timestamp, current_price, tau: {
            "predicted_direction": "up",
            "predicted_price": current_price * 1.01,
            "p_up": 0.72,
            "p_down": 0.18,
            "p_flat": 0.10,
            "confidence": 0.72,
        },
    )
    monkeypatch.setattr(
        signal_module,
        "predict_classification",
        lambda booster, feature_row, timestamp=None: {
            "label_mode": "meta_label",
            "predicted_label": "pass",
            "confidence": 0.91,
            "probabilities": {"pass": 0.91, "act": 0.09},
        },
    )

    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
        )
    )

    result = provider.evaluate(_context())

    assert result.direction == "hold"
    assert result.abstain_reason == "meta-label predicted pass"
    assert "p_act=0.090000" in result.reasons
    assert "action_artifact_artifact_role=action" in result.reasons


def test_legacy_model_provider_holds_for_non_direction_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.json").write_text("fake-model", encoding="utf-8")
    (model_dir / "xgb.meta.json").write_text(
        json.dumps({"family": "xgb", "label_mode": "triple_barrier"}),
        encoding="utf-8",
    )

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(signal_module, "get_runtime_config", lambda: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(signal_module, "load_config", lambda: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")

    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
        )
    )

    result = provider.evaluate(_context())

    assert result.direction == "hold"
    assert result.abstain_reason is not None
    assert "not runtime-tradable" in result.abstain_reason


def test_legacy_model_provider_predicts_for_current_candle_timestamp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "5m_30m"
    model_dir.mkdir(parents=True)
    (model_dir / "xgb.json").write_text("fake-model", encoding="utf-8")

    candle = _context().candle
    price_index = pd.date_range("2026-04-25 06:00:00", periods=60, freq="5min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": [100_000.0 + idx for idx in range(60)],
            "high": [100_050.0 + idx for idx in range(60)],
            "low": [99_950.0 + idx for idx in range(60)],
            "close": [100_020.0 + idx for idx in range(60)],
            "volume": [10.0] * 60,
        },
        index=price_index,
    )
    prices.index.name = "timestamp_utc"
    observed_timestamp: dict[str, pd.Timestamp | None] = {"value": None}

    def _fake_generate_price_features(
        prices: pd.DataFrame,
        enable_garch: bool,
        freq: str,
    ) -> pd.DataFrame:
        aligned_index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
        return pd.DataFrame({"feat_stub": [0.5] * len(prices)}, index=aligned_index)

    from bitbat_v2 import signals as signal_module

    monkeypatch.setattr(
        signal_module,
        "get_runtime_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(
        signal_module,
        "load_config",
        lambda: {"data_dir": str(tmp_path), "tau": 0.01},
    )
    monkeypatch.setattr(signal_module, "resolve_models_dir", lambda cfg: tmp_path / "models")
    monkeypatch.setattr(signal_module, "_load_ingested_prices", lambda data_dir, freq: prices)
    monkeypatch.setattr(signal_module, "generate_price_features", _fake_generate_price_features)
    monkeypatch.setattr(
        signal_module,
        "load_model",
        lambda path, expected_label_mode=None: type(
            "FakeBooster",
            (),
            {"feature_names": ["feat_stub"]},
        )(),
    )
    monkeypatch.setattr(
        signal_module,
        "predict_bar",
        lambda booster, feature_row, timestamp, current_price, tau: (
            observed_timestamp.__setitem__("value", pd.Timestamp(timestamp)),
            {
                "predicted_direction": "flat",
                "predicted_price": current_price,
                "p_up": 0.2,
                "p_down": 0.2,
                "p_flat": 0.6,
                "confidence": 0.6,
            },
        )[1],
    )

    provider = LegacyModelSignalProvider(
        runtime_config=BitBatV2Config(
            signal_source="legacy_ml",
            legacy_signal_freq="5m",
            legacy_signal_horizon="30m",
        )
    )

    provider.evaluate(_context())

    assert observed_timestamp["value"] == pd.Timestamp(candle.start).tz_localize(None)
