"""Drift detection logic for autonomous monitoring."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.autonomous.models import RetrainingEvent
from bitbat.config.loader import get_runtime_config, load_config

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DriftDetector:
    """Evaluate recent prediction quality and detect performance drift."""

    def __init__(self, db: AutonomousDB, freq: str = "5m", horizon: str = "30m") -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        config = get_runtime_config() or load_config()
        autonomous = config.get("autonomous", {})
        drift_cfg = autonomous.get("drift_detection", {})
        retrain_cfg = autonomous.get("retraining", {})

        self.window_days = int(drift_cfg.get("window_days", 30))
        self.min_predictions_required = int(drift_cfg.get("min_predictions_required", 30))
        self.mae_threshold = float(drift_cfg.get("mae_threshold", 0.01))
        self.directional_accuracy_threshold = float(
            drift_cfg.get("directional_accuracy_threshold", 0.50)
        )
        self.rmse_increase_threshold = float(drift_cfg.get("rmse_increase_threshold", 1.5))
        self.cooldown_hours = int(retrain_cfg.get("cooldown_hours", 24))

    def get_baseline_metrics(self) -> dict[str, Any]:
        """Return baseline metrics from active model metadata."""
        baseline_hit_rate = 0.55
        model_version: str | None = None
        cv_score: float | None = None

        with self.db.session() as session:
            active_model = self.db.get_active_model(session, self.freq, self.horizon)
            if active_model is not None:
                model_version = active_model.version
                cv_score = (
                    float(active_model.cv_score) if active_model.cv_score is not None else None
                )
                if cv_score is not None:
                    baseline_hit_rate = cv_score

        return {
            "model_version": model_version,
            "cv_score": cv_score,
            "baseline_hit_rate": baseline_hit_rate,
        }

    def is_in_cooldown(self) -> bool:
        """Return True when retraining is within cooldown window."""
        cutoff = _utcnow() - timedelta(hours=self.cooldown_hours)
        with self.db.session() as session:
            last_event = (
                session.query(RetrainingEvent)
                .filter(RetrainingEvent.started_at >= cutoff)
                .order_by(RetrainingEvent.started_at.desc())
                .first()
            )
        return last_event is not None

    def check_drift(self) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate drift conditions and return `(detected, reason, metrics)`."""
        with self.db.session() as session:
            predictions = self.db.get_recent_predictions(
                session=session,
                freq=self.freq,
                horizon=self.horizon,
                days=self.window_days,
                realized_only=True,
            )

        metrics = PerformanceMetrics(predictions).to_dict()
        baseline = self.get_baseline_metrics()
        metrics["baseline_hit_rate"] = baseline["baseline_hit_rate"]
        metrics["baseline_model_version"] = baseline["model_version"]

        realized_predictions = int(metrics["realized_predictions"])
        if realized_predictions < self.min_predictions_required:
            reason = (
                f"Insufficient realized predictions: {realized_predictions}/"
                f"{self.min_predictions_required}"
            )
            logger.info(reason)
            return (False, reason, metrics)

        mae = float(metrics.get("mae", 0.0))
        directional_accuracy = float(metrics.get("directional_accuracy", 1.0))
        lose_streak = int(metrics["lose_streak"])

        reasons: list[str] = []

        if mae > self.mae_threshold:
            reasons.append(
                f"MAE exceeds threshold: {mae:.6f} > {self.mae_threshold:.6f}"
            )

        if directional_accuracy < self.directional_accuracy_threshold:
            reasons.append(
                f"Directional accuracy below threshold: "
                f"{directional_accuracy:.2%} < {self.directional_accuracy_threshold:.2%}"
            )

        if lose_streak >= 10:
            reasons.append(f"Losing streak too long: {lose_streak}")

        if not reasons:
            return (False, "No drift detected", metrics)

        return (True, "; ".join(reasons), metrics)
