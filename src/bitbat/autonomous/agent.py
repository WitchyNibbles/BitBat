"""Monitoring agent that orchestrates validation, drift checks, and retraining."""

from __future__ import annotations

import logging
import time
from typing import Any

from bitbat.autonomous.alerting import send_alert
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.drift import DriftDetector
from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.autonomous.predictor import LivePredictor
from bitbat.autonomous.retrainer import AutoRetrainer
from bitbat.autonomous.validator import PredictionValidator

logger = logging.getLogger(__name__)


class MonitoringAgent:
    """Coordinate autonomous monitoring pipeline steps."""

    def __init__(self, db: AutonomousDB, freq: str = "1h", horizon: str = "4h") -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        self.predictor = LivePredictor(db, freq=freq, horizon=horizon)
        self.validator = PredictionValidator(db, freq=freq, horizon=horizon)
        self.drift_detector = DriftDetector(db, freq=freq, horizon=horizon)
        self.retrainer = AutoRetrainer(db, freq=freq, horizon=horizon)

    def _active_model_version(self) -> str:
        with self.db.session() as session:
            active = self.db.get_active_model(session, self.freq, self.horizon)
        return active.version if active is not None else "unknown"

    def _store_performance_snapshot(self, metrics: dict[str, Any], model_version: str) -> None:
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

        # Step 0: Refresh auxiliary data sources (macro, on-chain).
        self._ingest_auxiliary_data()

        # Step 1: Validate old predictions whose horizon has elapsed.
        validation_summary = self.validator.validate_all()

        # Step 2: Generate a new prediction for the latest bar.
        prediction_result = None
        try:
            prediction_result = self.predictor.predict_latest()
            if prediction_result is not None:
                logger.info("New prediction: %s", prediction_result)
            else:
                logger.info("No new prediction generated this cycle")
        except Exception:
            logger.exception("Prediction generation failed")

        with self.db.session() as session:
            recent_predictions = self.db.get_recent_predictions(
                session=session,
                freq=self.freq,
                horizon=self.horizon,
                days=int(self.drift_detector.window_days),
                realized_only=True,
            )

        metrics = PerformanceMetrics(recent_predictions).to_dict()
        model_version = self._active_model_version()
        if int(metrics["realized_predictions"]) > 0:
            self._store_performance_snapshot(metrics, model_version=model_version)

        drift_detected, drift_reason, drift_metrics = self.drift_detector.check_drift()
        retraining_triggered = False
        retraining_result: dict[str, Any] | None = None

        if drift_detected:
            if self.drift_detector.is_in_cooldown():
                warning_message = (
                    f"Drift detected but retraining in cooldown for {self.freq}/{self.horizon}"
                )
                logger.warning("%s: %s", warning_message, drift_reason)
                send_alert(
                    "WARNING",
                    warning_message,
                    {"reason": drift_reason, "metrics": drift_metrics},
                )
            else:
                retraining_triggered = True
                send_alert(
                    "CRITICAL",
                    "Drift detected - starting retraining",
                    {"reason": drift_reason, "metrics": drift_metrics},
                )
                retraining_result = self.retrainer.retrain(
                    trigger_reason="drift_detected",
                    trigger_metrics=drift_metrics,
                )

                if retraining_result.get("status") == "completed":
                    send_alert("SUCCESS", "Auto-retraining completed", retraining_result)
                else:
                    send_alert("ERROR", "Auto-retraining failed", retraining_result)

        result = {
            "prediction": prediction_result,
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

        logger.info("Monitoring cycle complete: %s", result)
        return result

    def run_forever(self, interval_seconds: int = 3600) -> None:
        """Run monitoring loop continuously."""
        logger.info("Starting monitoring loop with interval=%ss", interval_seconds)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.exception("Monitoring cycle failed: %s", exc)
                send_alert("ERROR", "Monitoring cycle failed", {"error": str(exc)})
            time.sleep(interval_seconds)
