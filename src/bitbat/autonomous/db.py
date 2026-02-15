"""Repository-style database access helpers for autonomous monitoring."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    ModelVersion,
    PerformanceSnapshot,
    PredictionOutcome,
    RetrainingEvent,
    SystemLog,
    create_database_engine,
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AutonomousDB:
    """High-level interface for autonomous system database operations."""

    def __init__(self, database_url: str = "sqlite:///data/autonomous.db") -> None:
        self.database_url = database_url
        self.engine = create_database_engine(database_url)
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=True,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )

    @contextmanager
    def session(self) -> Any:
        """Yield a managed session with automatic commit/rollback behavior."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def store_prediction(
        self,
        session: Session,
        timestamp_utc: datetime,
        predicted_direction: str,
        p_up: float,
        p_down: float,
        model_version: str,
        freq: str,
        horizon: str,
        predicted_return: float | None = None,
        features_used: dict[str, Any] | None = None,
    ) -> PredictionOutcome:
        """Insert a new prediction row."""
        if not (0.0 <= p_up <= 1.0 and 0.0 <= p_down <= 1.0):
            raise ValueError("Probabilities must be in [0, 1].")
        if p_up + p_down > 1.0 + 1e-9:
            raise ValueError("p_up + p_down must be <= 1.")

        prediction = PredictionOutcome(
            timestamp_utc=timestamp_utc,
            prediction_timestamp=_utcnow(),
            predicted_direction=predicted_direction,
            p_up=p_up,
            p_down=p_down,
            p_flat=max(0.0, 1.0 - p_up - p_down),
            predicted_return=predicted_return,
            model_version=model_version,
            freq=freq,
            horizon=horizon,
            features_used=features_used,
        )
        session.add(prediction)
        session.flush()
        return prediction

    def get_unrealized_predictions(
        self,
        session: Session,
        freq: str,
        horizon: str,
        cutoff_time: datetime | None = None,
    ) -> list[PredictionOutcome]:
        """Return predictions without realized returns for the requested config."""
        query = session.query(PredictionOutcome).filter(
            PredictionOutcome.actual_return.is_(None),
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
        )
        if cutoff_time is not None:
            query = query.filter(PredictionOutcome.timestamp_utc < cutoff_time)
        return list(query.order_by(PredictionOutcome.timestamp_utc.asc()).all())

    def realize_prediction(
        self,
        session: Session,
        prediction_id: int,
        actual_return: float,
        actual_direction: str,
    ) -> PredictionOutcome:
        """Fill realized fields for an existing prediction row."""
        prediction = session.get(PredictionOutcome, prediction_id)
        if prediction is None:
            raise ValueError(f"Prediction {prediction_id} not found.")

        prediction.actual_return = actual_return
        prediction.actual_direction = actual_direction
        prediction.correct = prediction.predicted_direction == actual_direction
        prediction.realized_at = _utcnow()
        session.flush()
        return prediction

    def get_recent_predictions(
        self,
        session: Session,
        freq: str,
        horizon: str,
        days: int = 30,
        realized_only: bool = True,
    ) -> list[PredictionOutcome]:
        """Return recent predictions for performance calculations."""
        cutoff = _utcnow() - timedelta(days=days)
        query = session.query(PredictionOutcome).filter(
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
            PredictionOutcome.created_at >= cutoff,
        )
        if realized_only:
            query = query.filter(PredictionOutcome.actual_return.is_not(None))
        return list(query.order_by(desc(PredictionOutcome.timestamp_utc)).all())

    def store_model_version(
        self,
        session: Session,
        version: str,
        freq: str,
        horizon: str,
        training_start: datetime,
        training_end: datetime,
        training_samples: int,
        cv_score: float | None,
        features: list[str] | None,
        hyperparameters: dict[str, Any] | None,
        training_metadata: dict[str, Any] | None,
        is_active: bool = True,
    ) -> ModelVersion:
        """Insert model metadata for a newly trained model."""
        model = ModelVersion(
            version=version,
            freq=freq,
            horizon=horizon,
            training_start=training_start,
            training_end=training_end,
            training_samples=training_samples,
            cv_score=cv_score,
            features=features,
            hyperparameters=hyperparameters,
            is_active=is_active,
            training_metadata=training_metadata,
        )
        session.add(model)
        session.flush()
        return model

    def get_active_model(self, session: Session, freq: str, horizon: str) -> ModelVersion | None:
        """Fetch the active model for a frequency and horizon pair."""
        return (
            session.query(ModelVersion)
            .filter(
                ModelVersion.freq == freq,
                ModelVersion.horizon == horizon,
                ModelVersion.is_active.is_(True),
            )
            .first()
        )

    def deactivate_old_models(self, session: Session, freq: str, horizon: str) -> int:
        """Deactivate all models for the requested frequency/horizon pair."""
        updated_count = (
            session.query(ModelVersion)
            .filter(ModelVersion.freq == freq, ModelVersion.horizon == horizon)
            .update(
                {
                    "is_active": False,
                    "replaced_at": _utcnow(),
                },
                synchronize_session=False,
            )
        )
        session.flush()
        return int(updated_count or 0)

    def create_retraining_event(
        self,
        session: Session,
        trigger_reason: str,
        trigger_metrics: dict[str, Any] | None,
        old_model_version: str | None = None,
    ) -> RetrainingEvent:
        """Create a retraining event row with status `started`."""
        event = RetrainingEvent(
            trigger_reason=trigger_reason,
            trigger_metrics=trigger_metrics,
            old_model_version=old_model_version,
            status="started",
            started_at=_utcnow(),
        )
        session.add(event)
        session.flush()
        return event

    def complete_retraining_event(
        self,
        session: Session,
        event_id: int,
        new_model_version: str,
        cv_improvement: float,
        training_duration_seconds: float,
    ) -> RetrainingEvent:
        """Update an event row with success metadata."""
        event = session.get(RetrainingEvent, event_id)
        if event is None:
            raise ValueError(f"Retraining event {event_id} not found.")
        event.status = "completed"
        event.new_model_version = new_model_version
        event.cv_improvement = cv_improvement
        event.training_duration_seconds = training_duration_seconds
        event.completed_at = _utcnow()
        session.flush()
        return event

    def fail_retraining_event(
        self,
        session: Session,
        event_id: int,
        error_message: str,
    ) -> RetrainingEvent:
        """Update an event row with failure metadata."""
        event = session.get(RetrainingEvent, event_id)
        if event is None:
            raise ValueError(f"Retraining event {event_id} not found.")
        event.status = "failed"
        event.error_message = error_message
        event.completed_at = _utcnow()
        session.flush()
        return event

    def store_performance_snapshot(
        self,
        session: Session,
        model_version: str,
        freq: str,
        horizon: str,
        window_days: int,
        metrics: dict[str, Any],
    ) -> PerformanceSnapshot:
        """Insert a performance snapshot row from a metrics dictionary."""
        snapshot = PerformanceSnapshot(
            model_version=model_version,
            freq=freq,
            horizon=horizon,
            snapshot_time=_utcnow(),
            window_days=window_days,
            total_predictions=int(metrics.get("total_predictions", 0)),
            realized_predictions=int(metrics.get("realized_predictions", 0)),
            hit_rate=metrics.get("hit_rate"),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            avg_return=metrics.get("avg_return"),
            max_drawdown=metrics.get("max_drawdown"),
            win_streak=metrics.get("win_streak"),
            lose_streak=metrics.get("lose_streak"),
            calibration_score=metrics.get("calibration_score"),
        )
        session.add(snapshot)
        session.flush()
        return snapshot

    def log(
        self,
        session: Session,
        level: str,
        service: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> SystemLog:
        """Insert a system log row."""
        log_entry = SystemLog(
            level=level,
            service=service,
            message=message,
            details=details,
        )
        session.add(log_entry)
        session.flush()
        return log_entry
