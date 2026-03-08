"""Repository-style database access helpers for autonomous monitoring."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
    init_database,
)
from .schema_compat import (
    SchemaAuditReport,
    SchemaCompatibilityError,
    audit_schema_compatibility,
    ensure_schema_compatibility,
    format_missing_columns,
    upgrade_schema_compatibility,
)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _schema_remediation_text(database_url: str, *, can_auto_upgrade: bool) -> str:
    audit_cmd = (
        "poetry run python scripts/init_autonomous_db.py "
        f'--database-url "{database_url}" --audit'
    )
    if can_auto_upgrade:
        upgrade_cmd = (
            "poetry run python scripts/init_autonomous_db.py "
            f'--database-url "{database_url}" --upgrade'
        )
        return f"Run `{audit_cmd}` then `{upgrade_cmd}`."
    return (
        f"Run `{audit_cmd}`. Blocking non-additive incompatibilities were detected; "
        "use `--force` only if table recreation is acceptable."
    )


def _schema_detail_from_report(report: SchemaAuditReport) -> str:
    missing = format_missing_columns(report) or "unknown"
    return f"Schema incompatible: missing {missing}"


@dataclass(slots=True)
class MonitorDatabaseError(RuntimeError):
    """Structured monitor DB failure with actionable diagnostics."""

    step: str
    detail: str
    remediation: str
    error_class: str
    database_url: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"[{self.step}] {self.error_class}: {self.detail} Remediation: {self.remediation}",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "step": self.step,
            "detail": self.detail,
            "remediation": self.remediation,
            "error_class": self.error_class,
            "database_url": self.database_url,
        }


def classify_monitor_db_error(
    exc: Exception,
    *,
    step: str,
    database_url: str,
    engine: Any | None = None,
) -> MonitorDatabaseError:
    """Map raw DB exceptions to actionable monitor diagnostics."""
    error_class = type(exc).__name__
    raw_detail = str(exc).strip() or repr(exc)
    detail = raw_detail
    remediation = (
        f"Check database availability at `{database_url}` and inspect monitor logs for context."
    )

    if isinstance(exc, SchemaCompatibilityError):
        detail = _schema_detail_from_report(exc.report)
        remediation = _schema_remediation_text(
            database_url,
            can_auto_upgrade=exc.report.can_auto_upgrade,
        )
    else:
        lower = raw_detail.lower()
        report: SchemaAuditReport | None = None
        if (
            "no such column" in lower
            or "prediction_outcomes" in lower
            or "performance_snapshots" in lower
        ):
            try:
                report = audit_schema_compatibility(database_url=database_url, engine=engine)
            except Exception:
                report = None

            if report is not None and not report.is_compatible:
                detail = _schema_detail_from_report(report)
                remediation = _schema_remediation_text(
                    database_url,
                    can_auto_upgrade=report.can_auto_upgrade,
                )
            elif "no such column" in lower:
                detail = f"Runtime query failed: {raw_detail}"
                remediation = _schema_remediation_text(database_url, can_auto_upgrade=True)

    return MonitorDatabaseError(
        step=step,
        detail=detail,
        remediation=remediation,
        error_class=error_class,
        database_url=database_url,
    )


class AutonomousDB:
    """High-level interface for autonomous system database operations."""

    def __init__(
        self,
        database_url: str = "sqlite:///data/autonomous.db",
        *,
        auto_upgrade_schema: bool = True,
    ) -> None:
        self.database_url = database_url
        self.engine = create_database_engine(database_url)
        init_database(database_url, engine=self.engine)
        self.schema_compatibility_status: dict[str, str | int] = {}
        if auto_upgrade_schema:
            upgrade_result = upgrade_schema_compatibility(
                database_url=database_url,
                engine=self.engine,
            )
            self.schema_compatibility_status = dict(upgrade_result.status)
            if not upgrade_result.is_compatible:
                raise SchemaCompatibilityError(
                    report=upgrade_result.report_after,
                    database_url=database_url,
                )
        else:
            report = ensure_schema_compatibility(
                database_url=database_url,
                engine=self.engine,
                auto_upgrade=False,
                raise_on_error=True,
            )
            self.schema_compatibility_status = {
                "upgrade_state": "already_compatible",
                "operations_applied": 0,
                "missing_columns_before": report.missing_column_count,
                "missing_columns_after": report.missing_column_count,
            }
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
        model_version: str,
        freq: str,
        horizon: str,
        predicted_return: float | None = None,
        predicted_price: float | None = None,
        p_up: float = 0.0,
        p_down: float = 0.0,
        features_used: dict[str, Any] | None = None,
    ) -> PredictionOutcome:
        """Insert a new prediction row."""
        prediction = PredictionOutcome(
            timestamp_utc=timestamp_utc,
            prediction_timestamp=_utcnow(),
            predicted_direction=predicted_direction,
            p_up=p_up,
            p_down=p_down,
            p_flat=0.0,
            predicted_return=predicted_return,
            predicted_price=predicted_price,
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

    def get_prediction_counts(self, session: Session, freq: str, horizon: str) -> dict[str, int | str]:  # noqa: E501
        """Return pair-scoped total/unrealized/realized prediction counts."""
        base_query = session.query(PredictionOutcome).filter(
            PredictionOutcome.freq == freq,
            PredictionOutcome.horizon == horizon,
        )
        total_predictions = int(base_query.count())
        realized_predictions = int(
            base_query.filter(PredictionOutcome.actual_return.is_not(None)).count()
        )
        unrealized_predictions = total_predictions - realized_predictions
        return {
            "freq": freq,
            "horizon": horizon,
            "total_predictions": total_predictions,
            "unrealized_predictions": unrealized_predictions,
            "realized_predictions": realized_predictions,
        }

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
