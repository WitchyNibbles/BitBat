"""Live prediction generator for the autonomous monitoring pipeline.

Loads the latest ingested price data, generates features, runs model
inference, and stores the prediction in the autonomous database so the
dashboard and validation feedback loop can use it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from bitbat.autonomous.db import AutonomousDB, classify_monitor_db_error
from bitbat.autonomous.models import PredictionOutcome
from bitbat.config.loader import get_runtime_config, load_config, resolve_models_dir
from bitbat.dataset.build import generate_price_features
from bitbat.model.infer import predict_bar
from bitbat.model.persist import load as load_model

logger = logging.getLogger(__name__)
MIN_BARS_REQUIRED = 30


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _normalize_timestamp(value: datetime | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.to_pydatetime()


def _fill_optional_feature_defaults(features: pd.DataFrame) -> pd.DataFrame:
    """Fill runtime-safe defaults for optional feature families."""
    if features.empty:
        return features

    filled = features.copy()
    for column in filled.columns:
        if not column.startswith("feat_sent_"):
            continue
        if column.endswith(("mean", "median", "decay")):
            filled[column] = filled[column].fillna(0.0)
        elif column.endswith(("pos", "neg", "neu", "count")):
            filled[column] = filled[column].fillna(0)
    return filled


def _derive_classifier_predicted_price(
    direction: str,
    *,
    current_price: float,
    tau: float,
) -> float:
    """Derive a concrete target price when the classifier only emits a direction label."""
    normalized = direction.strip().lower()
    if normalized == "up":
        return current_price * (1.0 + tau)
    if normalized == "down":
        return current_price * (1.0 - tau)
    return current_price


def _load_ingested_prices(data_dir: Path, freq: str) -> pd.DataFrame:
    """Load price bars from the date-partitioned ingestion directory.

    Delegates to the shared :func:`bitbat.io.prices.load_prices` implementation
    which scans ``data_dir / "raw" / "prices"`` for all ``**/*.parquet`` files
    (date-partitioned and legacy flat files).
    """
    from bitbat.io.prices import load_prices

    return load_prices(data_dir, freq)


class LivePredictor:
    """Generate a prediction from the latest ingested data and store it in the DB."""

    def __init__(
        self,
        db: AutonomousDB,
        freq: str = "5m",
        horizon: str = "30m",
    ) -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        config = get_runtime_config() or load_config()
        self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        self.model_dir = resolve_models_dir(config)

    def _model_path(self) -> Path:
        return self.model_dir / f"{self.freq}_{self.horizon}" / "xgb.json"

    def _load_model(self) -> xgb.Booster:
        model_path = self._model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        return load_model(model_path)

    def _active_model_version(self) -> str:
        try:
            with self.db.session() as session:
                active = self.db.get_active_model(session, self.freq, self.horizon)
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="predict.get_active_model",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc
        if active is not None:
            return active.version
        # Fall back to a default version tag when no model is registered in DB
        from bitbat import __version__

        return __version__

    @staticmethod
    def _result(
        *,
        status: str,
        reason: str,
        message: str,
        details: dict[str, Any] | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "message": message,
            "diagnostic_reason": reason,
            "diagnostic_message": message,
        }
        if details:
            result["details"] = details
        result.update(payload)
        return result

    def predict_latest(self) -> dict[str, Any]:  # noqa: C901
        """Generate a prediction for the most recent bar and store it.

        Returns a structured result that always includes:
        - ``status``: ``generated`` or ``no_prediction``
        - ``reason``: stable reason code for operator diagnostics
        - ``message``: concise human-readable explanation
        """
        try:
            with self.db.session() as session:
                pending_predictions = self.db.get_unrealized_predictions(
                    session=session,
                    freq=self.freq,
                    horizon=self.horizon,
                )
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="predict.fetch_unrealized_predictions",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc
        if pending_predictions:
            oldest_pending = min(
                _normalize_timestamp(prediction.timestamp_utc) for prediction in pending_predictions
            )
            logger.info(
                "Skipping prediction for %s/%s because %d prediction(s) remain unrealized",
                self.freq,
                self.horizon,
                len(pending_predictions),
            )
            return self._result(
                status="no_prediction",
                reason="pending_realization",
                message="Existing prediction must be realized before starting a new one",
                details={
                    "pending_predictions": int(len(pending_predictions)),
                    "oldest_pending_timestamp": oldest_pending.isoformat(),
                },
            )

        # Load model
        try:
            booster = self._load_model()
        except FileNotFoundError:
            logger.error("No model artifact found at %s — skipping prediction", self._model_path())
            return self._result(
                status="no_prediction",
                reason="missing_model",
                message="Model artifact not found for runtime pair",
                details={"model_path": str(self._model_path())},
            )

        # Load ingested prices
        try:
            prices = _load_ingested_prices(self.data_dir, self.freq)
        except RuntimeError as exc:
            logger.error("Cannot generate prediction: %s", exc)
            return self._result(
                status="no_prediction",
                reason="missing_prices",
                message="No price data available for prediction",
                details={"error": str(exc)},
            )

        if len(prices) < MIN_BARS_REQUIRED:
            logger.warning(
                "Only %d price bars available — need at least %d for features",
                len(prices),
                MIN_BARS_REQUIRED,
            )
            return self._result(
                status="no_prediction",
                reason="insufficient_data",
                message="Insufficient price bars for feature generation",
                details={
                    "available_bars": int(len(prices)),
                    "required_bars": MIN_BARS_REQUIRED,
                },
            )

        # Generate features
        try:
            config = get_runtime_config() or load_config()
            enable_garch = bool(config.get("enable_garch", False))
            features = generate_price_features(prices, enable_garch=enable_garch, freq=self.freq)

            # Join sentiment features if news data exists
            enable_sentiment = bool(config.get("enable_sentiment", True))
            if enable_sentiment:
                # Try RSS (free, default) first, then GDELT, then CryptoCompare
                news_candidates = [
                    self.data_dir / "raw" / "news" / "rss_1h" / "rss_crypto_1h.parquet",
                    self.data_dir / "raw" / "news" / "gdelt_1h" / "gdelt_crypto_1h.parquet",
                    self.data_dir
                    / "raw"
                    / "news"
                    / f"cryptocompare_{self.freq}"
                    / f"cryptocompare_btc_{self.freq}.parquet",
                ]
                news_path = None
                for candidate in news_candidates:
                    if candidate.exists():
                        news_path = candidate
                        break

                if news_path is not None:
                    try:
                        from bitbat.features.sentiment import aggregate as aggregate_sentiment

                        news_df = pd.read_parquet(news_path)
                        news_df["published_utc"] = pd.to_datetime(
                            news_df["published_utc"], utc=True
                        ).dt.tz_localize(None)
                        news_df = news_df.sort_values("published_utc")
                        bar_df = prices.reset_index()[["timestamp_utc"]]
                        sentiment_features = aggregate_sentiment(
                            news_df=news_df,
                            bar_df=bar_df,
                            freq=self.freq,
                        )
                        features = features.join(sentiment_features, how="left")
                    except Exception as exc:
                        logger.warning("Sentiment feature generation failed: %s", exc)
                else:
                    logger.info(
                        "No news data found (searched %s) — skipping sentiment features",
                        [str(c) for c in news_candidates],
                    )

            # Join auxiliary features if enabled and data exists
            enable_macro = bool(config.get("enable_macro", False))
            enable_onchain = bool(config.get("enable_onchain", False))
            if enable_macro or enable_onchain:
                from bitbat.dataset.build import join_auxiliary_features

                macro_path = self.data_dir / "raw" / "macro" / "fred.parquet"
                onchain_path = self.data_dir / "raw" / "onchain" / "blockchain_info.parquet"
                features = join_auxiliary_features(
                    features,
                    macro_parquet=macro_path if enable_macro and macro_path.exists() else None,
                    onchain_parquet=(
                        onchain_path if enable_onchain and onchain_path.exists() else None
                    ),
                    freq=self.freq,
                )

            rename_mapping = {
                col: col if col.startswith("feat_") else f"feat_{col}" for col in features.columns
            }
            features = features.rename(columns=rename_mapping)
        except Exception as exc:
            logger.error("Feature generation failed: %s", exc)
            return self._result(
                status="no_prediction",
                reason="feature_generation_failed",
                message="Feature generation failed",
                details={"error": str(exc)},
            )

        if features.empty:
            logger.warning("Feature generation produced no usable rows")
            return self._result(
                status="no_prediction",
                reason="no_features",
                message="Feature generation produced no usable rows",
            )

        # Align features to model's expected columns
        expected_features = list(booster.feature_names or [])
        if not expected_features:
            logger.error("Model artifact missing feature names — cannot align features")
            return self._result(
                status="no_prediction",
                reason="missing_feature_names",
                message="Model artifact missing feature names",
            )

        available = set(features.columns)
        missing = sorted(set(expected_features) - available)
        if missing:
            logger.error("Features missing columns expected by model: %s", missing)
            return self._result(
                status="no_prediction",
                reason="feature_mismatch",
                message="Model feature set does not match generated features",
                details={"missing_columns": missing},
            )

        aligned = _fill_optional_feature_defaults(features[expected_features])
        aligned = aligned.dropna()
        if aligned.empty:
            logger.warning("Aligned feature set produced no usable rows")
            return self._result(
                status="no_prediction",
                reason="no_features",
                message="Feature generation produced no usable rows",
            )

        latest_ts = aligned.index.max()
        latest_feature_ts = _normalize_timestamp(latest_ts)
        latest_price_ts = _normalize_timestamp(prices.index.max())

        if latest_feature_ts < latest_price_ts:
            logger.warning(
                "Latest usable feature row lags latest price bar (feature=%s price=%s)",
                latest_feature_ts,
                latest_price_ts,
            )
            return self._result(
                status="no_prediction",
                reason="stale_features",
                message="Latest feature row lags latest price bar",
                details={
                    "latest_feature_timestamp": latest_feature_ts.isoformat(),
                    "latest_price_timestamp": latest_price_ts.isoformat(),
                },
            )

        # Check for duplicate prediction — don't re-predict the same bar
        try:
            with self.db.session() as session:
                any_existing = (
                    session.query(PredictionOutcome.id)
                    .filter(
                        PredictionOutcome.freq == self.freq,
                        PredictionOutcome.horizon == self.horizon,
                        PredictionOutcome.timestamp_utc == latest_feature_ts,
                    )
                    .first()
                )
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="predict.fetch_unrealized_predictions",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc
        if any_existing is not None:
            logger.info("Prediction for %s already exists in history — skipping", latest_ts)
            return self._result(
                status="no_prediction",
                reason="duplicate_bar",
                message="Prediction for latest bar already exists",
                details={"timestamp_utc": str(latest_ts)},
            )

        # Run inference
        feature_row = aligned.loc[latest_ts]
        current_price = float(prices.loc[latest_ts, "close"])
        tau = float(config.get("tau", 0.01) or 0.01)
        prediction = predict_bar(
            booster,
            feature_row,
            timestamp=latest_ts,
            current_price=current_price,
            tau=tau,
        )

        predicted_return = prediction.get("predicted_return")
        if predicted_return is not None:
            predicted_return = float(predicted_return)
        direction = str(prediction["predicted_direction"])
        predicted_price = prediction.get("predicted_price")
        if predicted_price is not None:
            predicted_price = float(predicted_price)
        elif predicted_return is None:
            predicted_price = _derive_classifier_predicted_price(
                direction,
                current_price=current_price,
                tau=tau,
            )

        model_version = self._active_model_version()

        timestamp_utc = pd.Timestamp(latest_ts)
        if hasattr(timestamp_utc, "tz_localize"):
            timestamp_utc = timestamp_utc.tz_localize(None)
        timestamp_py = timestamp_utc.to_pydatetime()

        # Store in autonomous.db
        try:
            with self.db.session() as session:
                self.db.store_prediction(
                    session=session,
                    timestamp_utc=timestamp_py,
                    predicted_direction=direction,
                    model_version=model_version,
                    freq=self.freq,
                    horizon=self.horizon,
                    predicted_return=predicted_return,
                    predicted_price=predicted_price,
                    p_up=float(prediction.get("p_up", 0.0)),
                    p_down=float(prediction.get("p_down", 0.0)),
                    p_flat=float(prediction.get("p_flat", 0.0)),
                )
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="predict.store_prediction",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc

        logger.info(
            "Prediction stored: ts=%s direction=%s predicted_return=%.6f"
            " predicted_price=%.2f model=%s",
            timestamp_py,
            direction,
            predicted_return if predicted_return is not None else 0.0,
            predicted_price if predicted_price is not None else 0.0,
            model_version,
        )

        return self._result(
            status="generated",
            reason="prediction_generated",
            message="Prediction generated and stored",
            timestamp_utc=timestamp_py,
            predicted_direction=direction,
            predicted_return=predicted_return,
            predicted_price=predicted_price,
            model_version=model_version,
            p_up=float(prediction.get("p_up", 0.0)),
            p_down=float(prediction.get("p_down", 0.0)),
            p_flat=float(prediction.get("p_flat", 0.0)),
            confidence=prediction.get("confidence"),
        )
