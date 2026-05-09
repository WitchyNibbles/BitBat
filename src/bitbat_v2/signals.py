"""Signal providers for BitBat v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from bitbat.autonomous.predictor import (
    _derive_classifier_predicted_price,
    _fill_optional_feature_defaults,
    _load_ingested_prices,
)
from bitbat.config.loader import get_runtime_config, load_config, resolve_models_dir
from bitbat.dataset.build import generate_price_features, join_auxiliary_features
from bitbat.model.infer import predict_bar, predict_classification
from bitbat.model.persist import load as load_model
from bitbat.model.persist import load_metadata as load_model_metadata
from bitbat.model.persist import normalize_label_mode

from .config import BitBatV2Config, resolve_signal_source
from .domain import Candle
from .strategy import StrategyContext, compute_metrics, get_strategy

MIN_BARS_REQUIRED = 30


@dataclass(frozen=True)
class SignalEvaluation:
    model_name: str
    direction: str
    predicted_return: float
    predicted_price: float
    confidence: float
    reasons: list[str]
    block_reason: str | None
    p_up: float = 0.0
    p_down: float = 0.0
    p_flat: float = 0.0
    expected_move_return: float = 0.0
    expected_cost_return: float = 0.0
    expected_value_return: float = 0.0
    abstain_reason: str | None = None


class SignalProvider(Protocol):
    def evaluate(self, context: StrategyContext) -> SignalEvaluation: ...


def _round_trip_cost_return(config: BitBatV2Config) -> float:
    return round(
        ((max(float(config.fee_bps), 0.0) * 2.0) + (max(float(config.slippage_bps), 0.0) * 2.0))
        / 10_000.0,
        6,
    )


def _signed_expected_value(expected_move_return: float, expected_cost_return: float) -> float:
    if abs(expected_move_return) <= 1e-12:
        return 0.0
    return round(
        expected_move_return - math.copysign(expected_cost_return, expected_move_return),
        6,
    )


def _heuristic_probabilities(direction: str, confidence: float) -> tuple[float, float, float]:
    clamped_confidence = min(0.99, max(0.0, float(confidence)))
    if direction == "buy":
        residual = max(0.0, 1.0 - clamped_confidence)
        p_down = round(residual * 0.25, 6)
        p_flat = round(max(0.0, residual - p_down), 6)
        return round(clamped_confidence, 6), p_down, p_flat
    if direction == "sell":
        residual = max(0.0, 1.0 - clamped_confidence)
        p_up = round(residual * 0.25, 6)
        p_flat = round(max(0.0, residual - p_up), 6)
        return p_up, round(clamped_confidence, 6), p_flat
    return 0.0, 0.0, 1.0


class HeuristicSignalProvider:
    def __init__(self, strategy_name: str, runtime_config: BitBatV2Config) -> None:
        self.strategy = get_strategy(strategy_name)
        self.runtime_config = runtime_config

    def evaluate(self, context: StrategyContext) -> SignalEvaluation:
        result = self.strategy.evaluate(context)
        expected_move_return = float(result.predicted_return)
        expected_cost_return = _round_trip_cost_return(self.runtime_config)
        expected_value_return = _signed_expected_value(expected_move_return, expected_cost_return)
        p_up, p_down, p_flat = _heuristic_probabilities(result.direction, result.confidence)
        return SignalEvaluation(
            model_name=result.strategy_name,
            direction=result.direction,
            predicted_return=result.predicted_return,
            predicted_price=result.predicted_price,
            confidence=result.confidence,
            reasons=list(result.reasons),
            block_reason=result.block_reason,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            expected_move_return=expected_move_return,
            expected_cost_return=expected_cost_return,
            expected_value_return=expected_value_return,
            abstain_reason=result.block_reason,
        )


class LegacyModelSignalProvider:
    """Generate v2 trade signals from the legacy trained-model path."""

    def __init__(
        self,
        *,
        runtime_config: BitBatV2Config,
    ) -> None:
        self.runtime_config = runtime_config
        self._cached_legacy_config: dict[str, object] | None = None
        self._cached_data_dir: Path | None = None
        self._cached_model_dir: Path | None = None
        self._cached_boosters: dict[Path, Any] = {}
        self._cached_metadata_by_path: dict[Path, dict[str, Any]] = {}
        self._cached_prices: pd.DataFrame | None = None
        self._cached_aligned_features: pd.DataFrame | None = None

    def evaluate(self, context: StrategyContext) -> SignalEvaluation:
        prepared = self._prepare_prediction_inputs(context)
        if isinstance(prepared, SignalEvaluation):
            return prepared

        prediction = prepared["prediction"]
        current_price = float(prepared["current_price"])
        metadata = prepared["metadata"]
        tau = float(prepared["tau"])
        freq = str(prepared["freq"])
        horizon = str(prepared["horizon"])
        action_prediction = prepared.get("action_prediction")
        action_metadata = prepared.get("action_metadata", {})

        predicted_direction = str(prediction["predicted_direction"]).lower()
        p_up = float(prediction.get("p_up", 0.0))
        p_down = float(prediction.get("p_down", 0.0))
        p_flat = float(prediction.get("p_flat", 0.0))
        side_confidence = float(prediction.get("confidence") or max(p_up, p_down, p_flat))
        confidence = side_confidence
        signed_score = round(tau * (p_up - p_down), 6)
        direction = {
            "up": "buy",
            "down": "sell",
            "flat": "hold",
        }.get(predicted_direction, "hold")
        predicted_price = prediction.get("predicted_price")
        if predicted_price is None:
            predicted_price = _derive_classifier_predicted_price(
                predicted_direction,
                current_price=current_price,
                tau=tau,
            )

        reasons = [
            "signal_source=legacy_ml",
            f"legacy_pair={freq}/{horizon}",
            f"class={predicted_direction}",
            f"p_up={p_up:.6f}",
            f"p_down={p_down:.6f}",
            f"p_flat={p_flat:.6f}",
        ]
        action_confidence = 1.0
        if isinstance(action_prediction, dict):
            action_probabilities = action_prediction.get("probabilities", {})
            p_act = float(action_probabilities.get("act", 0.0))
            p_pass = float(action_probabilities.get("pass", 0.0))
            action_confidence = float(action_prediction.get("confidence", max(p_act, p_pass)))
            reasons.extend([
                f"p_act={p_act:.6f}",
                f"p_pass={p_pass:.6f}",
            ])
            for key in ("family", "label_mode", "version", "freq", "horizon", "artifact_role"):
                value = action_metadata.get(key)
                if value not in (None, ""):
                    reasons.append(f"action_artifact_{key}={value}")
            confidence = round((side_confidence + p_act) / 2.0, 6)
            if str(action_prediction.get("predicted_label", "")).lower() != "act":
                direction = "hold"
        expected_move_return = (
            0.0 if direction == "hold" else round(signed_score * action_confidence, 6)
        )
        expected_cost_return = _round_trip_cost_return(self.runtime_config)
        expected_value_return = _signed_expected_value(expected_move_return, expected_cost_return)
        reasons.extend([
            f"expected_move_return={expected_move_return:.6f}",
            f"expected_cost_return={expected_cost_return:.6f}",
            f"expected_value_return={expected_value_return:.6f}",
        ])
        for key in ("family", "label_mode", "version", "freq", "horizon", "artifact_role"):
            value = metadata.get(key)
            if value not in (None, ""):
                reasons.append(f"artifact_{key}={value}")
        block_reason: str | None
        if direction == "hold" and isinstance(action_prediction, dict):
            block_reason = "meta-label predicted pass"
        else:
            block_reason = "legacy model predicted flat" if direction == "hold" else None
        return SignalEvaluation(
            model_name=f"legacy_xgb_{freq}_{horizon}",
            direction=direction,
            predicted_return=expected_move_return,
            predicted_price=float(predicted_price),
            confidence=confidence,
            reasons=reasons,
            block_reason=block_reason,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            expected_move_return=expected_move_return,
            expected_cost_return=expected_cost_return,
            expected_value_return=expected_value_return,
            abstain_reason=block_reason,
        )

    def _prepare_prediction_inputs(  # noqa: C901
        self,
        context: StrategyContext,
    ) -> dict[str, Any] | SignalEvaluation:
        legacy_config = self._legacy_config()
        data_dir = self._data_dir(legacy_config)
        model_dir = self._model_dir(legacy_config)
        freq = self.runtime_config.legacy_signal_freq
        horizon = self.runtime_config.legacy_signal_horizon
        model_path = model_dir / f"{freq}_{horizon}" / "xgb.json"
        side_model_path = model_dir / f"{freq}_{horizon}" / "xgb.side.json"
        action_model_path = model_dir / f"{freq}_{horizon}" / "xgb.action.meta_label.json"
        use_meta_policy = side_model_path.exists() and action_model_path.exists()

        if not model_path.exists() and not use_meta_policy:
            return self._hold(
                f"legacy model artifact missing: {model_path}",
                current_price=context.candle.close,
            )

        try:
            if use_meta_policy:
                metadata = self._metadata(side_model_path)
                action_metadata = self._metadata(action_model_path)
                artifact_label_mode = normalize_label_mode(
                    str(metadata.get("label_mode", "direction"))
                )
                action_label_mode = normalize_label_mode(
                    str(action_metadata.get("label_mode", "meta_label"))
                )
                if artifact_label_mode != "direction" or action_label_mode != "meta_label":
                    return self._hold(
                        "legacy ML meta policy artifacts are not runtime-tradable",
                        current_price=context.candle.close,
                    )
                booster = self._booster(side_model_path)
                action_booster = self._booster(action_model_path)
            else:
                metadata = self._metadata(model_path)
                action_metadata = {}
                artifact_label_mode = normalize_label_mode(
                    str(metadata.get("label_mode", "direction"))
                )
                if artifact_label_mode != "direction":
                    return self._hold(
                        "legacy ML artifact is not runtime-tradable: "
                        f"label_mode={artifact_label_mode}",
                        current_price=context.candle.close,
                    )
                booster = self._booster(model_path)
                action_booster = None
            prices = self._prices(data_dir, freq, context.candle)
        except Exception as exc:
            return self._hold(
                f"legacy ML setup failed: {exc}",
                current_price=context.candle.close,
            )

        if len(prices) < MIN_BARS_REQUIRED:
            return self._hold(
                f"insufficient price history for legacy ML: {len(prices)} bars",
                current_price=context.candle.close,
            )

        try:
            aligned = self._aligned_features(
                booster=booster,
                prices=prices,
                data_dir=data_dir,
                legacy_config=legacy_config,
                freq=freq,
            )
        except Exception as exc:
            return self._hold(
                f"legacy ML feature preparation failed: {exc}",
                current_price=context.candle.close,
            )

        if aligned.empty:
            return self._hold(
                "legacy ML produced no aligned feature rows",
                current_price=context.candle.close,
            )

        current_ts = pd.Timestamp(context.candle.start).tz_convert("UTC").tz_localize(None)
        if current_ts not in aligned.index:
            return self._hold(
                "legacy ML has no aligned features for current candle",
                current_price=context.candle.close,
            )

        feature_row = aligned.loc[current_ts]
        current_price = float(prices.loc[current_ts, "close"])
        raw_tau = legacy_config.get("tau", 0.01)
        tau = float(raw_tau if isinstance(raw_tau, int | float | str) else 0.01)

        try:
            prediction = predict_bar(
                booster,
                feature_row,
                timestamp=current_ts,
                current_price=current_price,
                tau=tau,
            )
            action_prediction = (
                predict_classification(
                    action_booster,
                    feature_row,
                    timestamp=current_ts,
                )
                if action_booster is not None
                else None
            )
        except Exception as exc:
            return self._hold(
                f"legacy ML inference failed: {exc}",
                current_price=current_price,
            )
        return {
            "metadata": metadata,
            "prediction": prediction,
            "current_price": current_price,
            "tau": tau,
            "freq": freq,
            "horizon": horizon,
            "action_prediction": action_prediction,
            "action_metadata": action_metadata,
        }

    def _aligned_features(
        self,
        *,
        booster: Any,
        prices: pd.DataFrame,
        data_dir: Path,
        legacy_config: dict[str, object],
        freq: str,
    ) -> pd.DataFrame:
        if self._cached_aligned_features is not None:
            return self._cached_aligned_features

        enable_garch = bool(legacy_config.get("enable_garch", False))
        features = generate_price_features(prices, enable_garch=enable_garch, freq=freq)

        enable_sentiment = bool(legacy_config.get("enable_sentiment", True))
        if enable_sentiment:
            from bitbat.features.sentiment import aggregate as aggregate_sentiment

            news_candidates = [
                data_dir / "raw" / "news" / "rss_1h" / "rss_crypto_1h.parquet",
                data_dir / "raw" / "news" / "gdelt_1h" / "gdelt_crypto_1h.parquet",
                data_dir
                / "raw"
                / "news"
                / f"cryptocompare_{freq}"
                / f"cryptocompare_btc_{freq}.parquet",
            ]
            for candidate in news_candidates:
                if not candidate.exists():
                    continue
                news_df = pd.read_parquet(candidate)
                news_df["published_utc"] = pd.to_datetime(
                    news_df["published_utc"],
                    utc=True,
                ).dt.tz_localize(None)
                news_df = news_df.sort_values("published_utc")
                bar_df = prices.reset_index()[["timestamp_utc"]]
                sentiment_features = aggregate_sentiment(
                    news_df=news_df,
                    bar_df=bar_df,
                    freq=freq,
                )
                features = features.join(sentiment_features, how="left")
                break

        enable_macro = bool(legacy_config.get("enable_macro", False))
        enable_onchain = bool(legacy_config.get("enable_onchain", False))
        if enable_macro or enable_onchain:
            macro_path = data_dir / "raw" / "macro" / "fred.parquet"
            onchain_path = data_dir / "raw" / "onchain" / "blockchain_info.parquet"
            features = join_auxiliary_features(
                features,
                macro_parquet=macro_path if enable_macro and macro_path.exists() else None,
                onchain_parquet=onchain_path if enable_onchain and onchain_path.exists() else None,
                freq=freq,
            )

        rename_mapping = {
            col: col if col.startswith("feat_") else f"feat_{col}" for col in features.columns
        }
        features = features.rename(columns=rename_mapping)
        expected_features = list(booster.feature_names or [])
        if not expected_features:
            raise ValueError("legacy model artifact missing feature names")

        missing = sorted(set(expected_features) - set(features.columns))
        if missing:
            raise ValueError(f"missing feature columns: {missing}")

        aligned = _fill_optional_feature_defaults(features[expected_features]).dropna()
        self._cached_aligned_features = aligned
        return self._cached_aligned_features

    @staticmethod
    def _merge_context_candle(prices: pd.DataFrame, candle: Candle) -> pd.DataFrame:
        frame = prices.reset_index().copy()
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
        ).dt.tz_localize(None)
        candle_ts = pd.Timestamp(candle.start).tz_convert("UTC").tz_localize(None)
        current_row = {
            "timestamp_utc": candle_ts,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        frame = frame[frame["timestamp_utc"] != candle_ts]
        frame = pd.concat([frame, pd.DataFrame([current_row])], ignore_index=True)
        return (
            frame.sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc"], keep="last")
            .set_index("timestamp_utc")
        )

    def _legacy_config(self) -> dict[str, object]:
        if self._cached_legacy_config is None:
            self._cached_legacy_config = get_runtime_config() or load_config()
        return self._cached_legacy_config

    def _data_dir(self, legacy_config: dict[str, object]) -> Path:
        if self._cached_data_dir is None:
            self._cached_data_dir = Path(str(legacy_config.get("data_dir", "data"))).expanduser()
        return self._cached_data_dir

    def _model_dir(self, legacy_config: dict[str, object]) -> Path:
        if self._cached_model_dir is None:
            self._cached_model_dir = resolve_models_dir(legacy_config)
        return self._cached_model_dir

    def _booster(self, model_path: Path) -> Any:
        if model_path not in self._cached_boosters:
            expected_label_mode = "meta_label" if "meta_label" in model_path.name else "direction"
            self._cached_boosters[model_path] = load_model(
                model_path,
                expected_label_mode=expected_label_mode,
            )
        return self._cached_boosters[model_path]

    def _metadata(self, model_path: Path) -> dict[str, Any]:
        if model_path not in self._cached_metadata_by_path:
            self._cached_metadata_by_path[model_path] = load_model_metadata(model_path)
        return self._cached_metadata_by_path[model_path]

    def _prices(self, data_dir: Path, freq: str, candle: Candle) -> pd.DataFrame:
        if self._cached_prices is None:
            self._cached_prices = _load_ingested_prices(data_dir, freq)
            self._cached_prices.index = pd.to_datetime(
                self._cached_prices.index,
                utc=True,
            ).tz_localize(None)
            self._cached_prices.index.name = "timestamp_utc"
        candle_ts = pd.Timestamp(candle.start).tz_convert("UTC").tz_localize(None)
        normalized_index = pd.Index(self._cached_prices.index)
        needs_upsert = candle_ts not in normalized_index
        if not needs_upsert:
            existing = self._cached_prices.loc[normalized_index == candle_ts].iloc[-1]
            needs_upsert = any(
                float(existing[field]) != float(getattr(candle, field))
                for field in ("open", "high", "low", "close", "volume")
            )
        if needs_upsert:
            self._cached_prices = self._merge_context_candle(self._cached_prices, candle)
            self._cached_aligned_features = None
        return self._cached_prices

    @staticmethod
    def _hold(reason: str, *, current_price: float) -> SignalEvaluation:
        return SignalEvaluation(
            model_name="legacy_xgb_unavailable",
            direction="hold",
            predicted_return=0.0,
            predicted_price=float(current_price),
            confidence=0.0,
            reasons=["signal_source=legacy_ml", reason],
            block_reason=reason,
            p_up=0.0,
            p_down=0.0,
            p_flat=1.0,
            expected_move_return=0.0,
            expected_cost_return=0.0,
            expected_value_return=0.0,
            abstain_reason=reason,
        )


def build_signal_provider(config: BitBatV2Config) -> SignalProvider:
    if resolve_signal_source(config.signal_source) == "legacy_ml":
        return LegacyModelSignalProvider(runtime_config=config)
    return HeuristicSignalProvider(config.model_name, runtime_config=config)


def build_feature_snapshot(context: StrategyContext) -> dict[str, float]:
    metrics = compute_metrics(context)
    return {
        "close": metrics.close,
        "open_to_close_return": metrics.open_to_close_return,
        "momentum_return": metrics.momentum_return,
        "range_ratio": metrics.range_ratio,
        "short_trend_return": metrics.short_trend_return,
        "trend_return": metrics.trend_return,
        "body_strength": metrics.body_strength,
    }
