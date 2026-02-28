"""Monitoring agent that orchestrates validation, drift checks, and retraining."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from bitbat.autonomous.alerting import send_alert
from bitbat.autonomous.continuous_trainer import ContinuousTrainer
from bitbat.autonomous.db import (
    AutonomousDB,
    MonitorDatabaseError,
    classify_monitor_db_error,
)
from bitbat.autonomous.drift import DriftDetector
from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.autonomous.predictor import LivePredictor
from bitbat.autonomous.schema_compat import ensure_schema_compatibility
from bitbat.autonomous.validator import PredictionValidator
from bitbat.config.loader import get_runtime_config, load_config

logger = logging.getLogger(__name__)


class MonitoringAgent:
    """Coordinate autonomous monitoring pipeline steps."""

    def __init__(self, db: AutonomousDB, freq: str = "5m", horizon: str = "30m") -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        self._validate_schema_preflight()
        self._validate_model_preflight()

        config = get_runtime_config() or load_config()

        self.predictor = LivePredictor(db, freq=freq, horizon=horizon)
        self.validator = PredictionValidator(db, freq=freq, horizon=horizon)
        self.drift_detector = DriftDetector(db, freq=freq, horizon=horizon)
        self.continuous_trainer = ContinuousTrainer(db, freq=freq, horizon=horizon, config=config)

    def _validate_schema_preflight(self) -> None:
        """Fail fast when runtime-required schema columns are missing."""
        ensure_schema_compatibility(
            database_url=self.db.database_url,
            engine=self.db.engine,
            auto_upgrade=False,
            raise_on_error=True,
        )

    def _validate_model_preflight(self) -> None:
        """Fail fast when the runtime pair has no model artifact."""
        model_path = Path("models") / f"{self.freq}_{self.horizon}" / "xgb.json"
        if model_path.exists():
            return
        raise FileNotFoundError(
            "Missing monitor model artifact for resolved runtime pair "
            f"{self.freq}/{self.horizon}: {model_path}. "
            "Use --config or BITBAT_CONFIG to select the intended pair, or train/copy "
            "the required artifact before starting monitor commands. "
            "You can bootstrap the runtime artifact with "
            "`python scripts/bootstrap_monitor_model.py --config <path>`."
        )

    def _active_model_version(self) -> str:
        try:
            with self.db.session() as session:
                active = self.db.get_active_model(session, self.freq, self.horizon)
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="monitor.get_active_model",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc
        return active.version if active is not None else "unknown"

    def _store_performance_snapshot(self, metrics: dict[str, Any], model_version: str) -> None:
        try:
            with self.db.session() as session:
                self.db.store_performance_snapshot(
                    session=session,
                    model_version=model_version,
                    freq=self.freq,
                    horizon=self.horizon,
                    window_days=int(self.drift_detector.window_days),
                    metrics={
                        "total_predictions": metrics.get("total_predictions", 0),
                        "realized_predictions": metrics.get("realized_predictions", 0),
                        "hit_rate": metrics.get("hit_rate"),
                        "sharpe_ratio": metrics.get("sharpe_ratio"),
                        "avg_return": metrics.get("average_return"),
                        "max_drawdown": metrics.get("max_drawdown"),
                        "win_streak": metrics.get("win_streak"),
                        "lose_streak": metrics.get("lose_streak"),
                        "calibration_score": metrics.get("calibration_score"),
                    },
                )
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="monitor.store_performance_snapshot",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc

    def _ingest_prices(self) -> None:
        """Fetch the latest price bars so the predictor sees fresh data."""
        try:
            from pathlib import Path

            from bitbat.autonomous.price_ingestion import PriceIngestionService
            from bitbat.config.loader import get_runtime_config

            config = get_runtime_config()
            data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
            svc = PriceIngestionService(
                symbol="BTC-USD", interval=self.freq, data_dir=data_dir
            )
            n = svc.fetch_with_retry()
            if n > 0:
                logger.info("Ingested %d new price bars", n)
        except Exception:
            logger.warning("Price ingestion failed", exc_info=True)

    def _ingest_news(self) -> None:
        """Fetch the latest news articles so sentiment features stay fresh."""
        try:
            from bitbat.config.loader import get_runtime_config

            config = get_runtime_config()
            if not config.get("enable_sentiment", True):
                return

            from datetime import UTC, datetime, timedelta

            from bitbat.ingest import news_cryptocompare as news_cc

            news_cc.fetch(
                from_dt=datetime.now(UTC) - timedelta(days=2),
                to_dt=datetime.now(UTC),
                throttle_seconds=0.5,
            )
            logger.info("News data refreshed")
        except Exception:
            logger.warning("News ingestion failed", exc_info=True)

    def _ingest_auxiliary_data(self) -> None:
        """Refresh macro and on-chain data if enabled in config."""
        try:
            from bitbat.config.loader import get_runtime_config

            config = get_runtime_config()
        except Exception:
            return

        if config.get("enable_macro"):
            try:
                from pathlib import Path

                from bitbat.autonomous.macro_ingestion import MacroIngestionService

                data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
                MacroIngestionService(data_dir=data_dir).fetch_with_retry()
            except Exception:
                logger.warning("Macro data ingestion failed", exc_info=True)

        if config.get("enable_onchain"):
            try:
                from pathlib import Path

                from bitbat.autonomous.onchain_ingestion import OnchainIngestionService

                data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
                OnchainIngestionService(data_dir=data_dir).fetch_with_retry()
            except Exception:
                logger.warning("On-chain data ingestion failed", exc_info=True)

    def run_once(self) -> dict[str, Any]:
        """Run one monitoring cycle: predict, validate, assess drift, retrain."""
        logger.info("Monitoring cycle started (%s/%s)", self.freq, self.horizon)

        # Step 0a: Refresh price data so predictor sees the latest bars.
        self._ingest_prices()

        # Step 0b: Refresh news data so sentiment features stay current.
        self._ingest_news()

        # Step 0c: Refresh auxiliary data sources (macro, on-chain).
        self._ingest_auxiliary_data()

        # Step 1: Validate old predictions whose horizon has elapsed.
        validation_summary = self.validator.validate_all()

        # Step 2: Generate a new prediction for the latest bar.
        prediction_result: dict[str, Any]
        try:
            raw_prediction = self.predictor.predict_latest()
            if raw_prediction is None:
                prediction_result = {
                    "status": "no_prediction",
                    "reason": "unknown",
                    "message": "Predictor returned no payload",
                }
            else:
                prediction_result = dict(raw_prediction)
            if prediction_result.get("status") == "generated":
                logger.info("New prediction: %s", prediction_result)
            else:
                logger.info(
                    "No new prediction generated this cycle (%s)",
                    prediction_result.get("reason", "unknown"),
                )
        except MonitorDatabaseError:
            raise
        except Exception as exc:
            logger.exception("Prediction generation failed")
            prediction_result = {
                "status": "no_prediction",
                "reason": "prediction_exception",
                "message": "Prediction generation failed",
                "details": {"error": str(exc)},
            }

        try:
            with self.db.session() as session:
                pending_predictions = self.db.get_unrealized_predictions(
                    session=session,
                    freq=self.freq,
                    horizon=self.horizon,
                )
                recent_predictions = self.db.get_recent_predictions(
                    session=session,
                    freq=self.freq,
                    horizon=self.horizon,
                    days=int(self.drift_detector.window_days),
                    realized_only=True,
                )
        except Exception as exc:
            raise classify_monitor_db_error(
                exc,
                step="monitor.fetch_recent_predictions",
                database_url=self.db.database_url,
                engine=self.db.engine,
            ) from exc

        metrics = PerformanceMetrics(recent_predictions).to_dict()
        pending_count = int(len(pending_predictions))
        realized_count = int(metrics.get("realized_predictions", 0))
        prediction_state = (
            "generated" if prediction_result.get("status") == "generated" else "none"
        )
        prediction_reason = str(prediction_result.get("reason", "unknown"))
        prediction_message = str(prediction_result.get("message", ""))
        if prediction_state == "generated":
            cycle_diagnostic = "prediction_generated"
        elif prediction_message:
            cycle_diagnostic = f"{prediction_reason}: {prediction_message}"
        else:
            cycle_diagnostic = prediction_reason

        if realized_count > 0:
            realization_state = "realized"
        elif pending_count > 0:
            realization_state = "pending"
        else:
            realization_state = "none"

        model_version = self._active_model_version()
        if realized_count > 0:
            self._store_performance_snapshot(metrics, model_version=model_version)

        drift_detected, drift_reason, drift_metrics = self.drift_detector.check_drift()
        retraining_triggered = False
        retraining_result: dict[str, Any] | None = None

        if drift_detected:
            send_alert(
                "WARNING",
                f"Drift detected for {self.freq}/{self.horizon}",
                {"reason": drift_reason, "metrics": drift_metrics},
            )

        # Continuous retraining: retrain on schedule when enough new samples exist
        if self.continuous_trainer.should_retrain():
            retraining_triggered = True
            retraining_result = self.continuous_trainer.retrain()
            if retraining_result.get("status") == "completed":
                send_alert("SUCCESS", "Continuous retraining completed", retraining_result)
            else:
                send_alert("ERROR", "Continuous retraining failed", retraining_result)

        result = {
            "prediction": prediction_result,
            "prediction_state": prediction_state,
            "prediction_reason": prediction_reason,
            "prediction_message": prediction_message,
            "realization_state": realization_state,
            "pending_validations": pending_count,
            "cycle_diagnostic": cycle_diagnostic,
            "cycle_state": {
                "prediction_state": prediction_state,
                "prediction_reason": prediction_reason,
                "prediction_message": prediction_message,
                "realization_state": realization_state,
                "cycle_diagnostic": cycle_diagnostic,
            },
            "validations": int(validation_summary.get("validated_count", 0)),
            "correct": int(validation_summary.get("correct_count", 0)),
            "hit_rate": float(validation_summary.get("hit_rate", 0.0)),
            "validation_errors": list(validation_summary.get("errors", [])),
            "drift_detected": drift_detected,
            "drift_reason": drift_reason,
            "retraining_triggered": retraining_triggered,
            "retraining_result": retraining_result,
            "metrics": metrics,
        }

        logger.info(
            "Cycle state: prediction_state=%s reason=%s realization_state=%s pending=%d",
            prediction_state,
            prediction_reason,
            realization_state,
            pending_count,
        )
        logger.info("Cycle diagnostic: %s", cycle_diagnostic)
        logger.info("Monitoring cycle complete: %s", result)
        return result

    def run_forever(self, interval_seconds: int = 300) -> None:
        """Run monitoring loop continuously."""
        logger.info("Starting monitoring loop with interval=%ss", interval_seconds)
        while True:
            try:
                self.run_once()
            except MonitorDatabaseError as exc:
                logger.error(
                    "Monitoring cycle database failure at %s: %s",
                    exc.step,
                    exc.detail,
                )
                send_alert("ERROR", "Monitoring cycle database failure", exc.to_dict())
            except Exception as exc:
                logger.exception("Monitoring cycle failed: %s", exc)
                send_alert("ERROR", "Monitoring cycle failed", {"error": str(exc)})
            time.sleep(interval_seconds)
