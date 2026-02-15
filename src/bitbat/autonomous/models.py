"""SQLAlchemy models and helpers for the autonomous monitoring database."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

_ENGINE_CACHE: dict[str, Engine] = {}


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Base declarative class for autonomous schema models."""


class PredictionOutcome(Base):
    """Stores generated predictions and realized outcomes."""

    __tablename__ = "prediction_outcomes"
    __table_args__ = (
        CheckConstraint(
            "predicted_direction IN ('up', 'down', 'flat')",
            name="ck_predicted_direction",
        ),
        CheckConstraint(
            "actual_direction IS NULL OR actual_direction IN ('up', 'down', 'flat')",
            name="ck_actual_direction",
        ),
        Index("idx_timestamp", "timestamp_utc"),
        Index("idx_model_version", "model_version"),
        Index("idx_freq_horizon", "freq", "horizon"),
        Index("idx_unrealized", "actual_return", sqlite_where=text("actual_return IS NULL")),
        Index("idx_created_at", "created_at"),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = mapped_column(DateTime, nullable=False)
    prediction_timestamp = mapped_column(DateTime, nullable=False)
    predicted_direction = mapped_column(String(10), nullable=False)
    p_up = mapped_column(Float, nullable=False)
    p_down = mapped_column(Float, nullable=False)
    p_flat = mapped_column(Float, nullable=True)
    predicted_return = mapped_column(Float, nullable=True)
    actual_return = mapped_column(Float, nullable=True)
    actual_direction = mapped_column(String(10), nullable=True)
    correct = mapped_column(Boolean, nullable=True)
    model_version = mapped_column(String(64), nullable=False)
    freq = mapped_column(String(16), nullable=False)
    horizon = mapped_column(String(16), nullable=False)
    features_used = mapped_column(JSON, nullable=True)
    created_at = mapped_column(DateTime, default=_utcnow, nullable=False)
    realized_at = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize prediction outcome to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "prediction_timestamp": (
                self.prediction_timestamp.isoformat() if self.prediction_timestamp else None
            ),
            "predicted_direction": self.predicted_direction,
            "p_up": self.p_up,
            "p_down": self.p_down,
            "p_flat": self.p_flat,
            "predicted_return": self.predicted_return,
            "actual_return": self.actual_return,
            "actual_direction": self.actual_direction,
            "correct": self.correct,
            "model_version": self.model_version,
            "freq": self.freq,
            "horizon": self.horizon,
            "features_used": self.features_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "realized_at": self.realized_at.isoformat() if self.realized_at else None,
        }

    @property
    def is_realized(self) -> bool:
        """Return True when actual values have been filled."""
        return self.actual_return is not None


class ModelVersion(Base):
    """Tracks model versions and their training metadata."""

    __tablename__ = "model_versions"
    __table_args__ = (
        Index("idx_version", "version", unique=True),
        Index("idx_active", "is_active"),
        Index("idx_freq_horizon_mv", "freq", "horizon"),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    version = mapped_column(String(64), nullable=False)
    freq = mapped_column(String(16), nullable=False)
    horizon = mapped_column(String(16), nullable=False)
    training_start = mapped_column(DateTime, nullable=False)
    training_end = mapped_column(DateTime, nullable=False)
    training_samples = mapped_column(Integer, nullable=False)
    cv_score = mapped_column(Float, nullable=True)
    features = mapped_column(JSON, nullable=True)
    hyperparameters = mapped_column(JSON, nullable=True)
    deployed_at = mapped_column(DateTime, nullable=True)
    replaced_at = mapped_column(DateTime, nullable=True)
    is_active = mapped_column(Boolean, default=True, nullable=False)
    training_metadata = mapped_column(JSON, nullable=True)
    created_at = mapped_column(DateTime, default=_utcnow, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model version metadata."""
        return {
            "id": self.id,
            "version": self.version,
            "freq": self.freq,
            "horizon": self.horizon,
            "training_start": self.training_start.isoformat() if self.training_start else None,
            "training_end": self.training_end.isoformat() if self.training_end else None,
            "training_samples": self.training_samples,
            "cv_score": self.cv_score,
            "features": self.features,
            "hyperparameters": self.hyperparameters,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "replaced_at": self.replaced_at.isoformat() if self.replaced_at else None,
            "is_active": self.is_active,
            "training_metadata": self.training_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RetrainingEvent(Base):
    """Logs retraining triggers and run outcomes."""

    __tablename__ = "retraining_events"
    __table_args__ = (
        CheckConstraint(
            "trigger_reason IN ('drift_detected', 'scheduled', 'manual', 'poor_performance')",
            name="ck_trigger_reason",
        ),
        CheckConstraint(
            "status IN ('started', 'completed', 'failed')",
            name="ck_retraining_status",
        ),
        Index("idx_started_at", "started_at"),
        Index("idx_status", "status"),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    trigger_reason = mapped_column(String(32), nullable=False)
    trigger_metrics = mapped_column(JSON, nullable=True)
    old_model_version = mapped_column(String(64), nullable=True)
    new_model_version = mapped_column(String(64), nullable=True)
    cv_improvement = mapped_column(Float, nullable=True)
    training_duration_seconds = mapped_column(Float, nullable=True)
    status = mapped_column(String(16), nullable=False)
    error_message = mapped_column(Text, nullable=True)
    started_at = mapped_column(DateTime, nullable=False)
    completed_at = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize retraining event to dictionary."""
        return {
            "id": self.id,
            "trigger_reason": self.trigger_reason,
            "trigger_metrics": self.trigger_metrics,
            "old_model_version": self.old_model_version,
            "new_model_version": self.new_model_version,
            "cv_improvement": self.cv_improvement,
            "training_duration_seconds": self.training_duration_seconds,
            "status": self.status,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class PerformanceSnapshot(Base):
    """Stores periodic summary metrics for model performance."""

    __tablename__ = "performance_snapshots"
    __table_args__ = (
        Index("idx_snapshot_time", "snapshot_time"),
        Index("idx_model_version_ps", "model_version"),
        Index("idx_freq_horizon_ps", "freq", "horizon"),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version = mapped_column(String(64), nullable=False)
    freq = mapped_column(String(16), nullable=False)
    horizon = mapped_column(String(16), nullable=False)
    snapshot_time = mapped_column(DateTime, nullable=False)
    window_days = mapped_column(Integer, nullable=False)
    total_predictions = mapped_column(Integer, nullable=False)
    realized_predictions = mapped_column(Integer, nullable=False)
    hit_rate = mapped_column(Float, nullable=True)
    sharpe_ratio = mapped_column(Float, nullable=True)
    avg_return = mapped_column(Float, nullable=True)
    max_drawdown = mapped_column(Float, nullable=True)
    win_streak = mapped_column(Integer, nullable=True)
    lose_streak = mapped_column(Integer, nullable=True)
    calibration_score = mapped_column(Float, nullable=True)
    created_at = mapped_column(DateTime, default=_utcnow, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "id": self.id,
            "model_version": self.model_version,
            "freq": self.freq,
            "horizon": self.horizon,
            "snapshot_time": self.snapshot_time.isoformat() if self.snapshot_time else None,
            "window_days": self.window_days,
            "total_predictions": self.total_predictions,
            "realized_predictions": self.realized_predictions,
            "hit_rate": self.hit_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_return": self.avg_return,
            "max_drawdown": self.max_drawdown,
            "win_streak": self.win_streak,
            "lose_streak": self.lose_streak,
            "calibration_score": self.calibration_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SystemLog(Base):
    """General service log table for autonomous services."""

    __tablename__ = "system_logs"
    __table_args__ = (
        CheckConstraint(
            "level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')",
            name="ck_log_level",
        ),
        Index("idx_timestamp_sl", "timestamp"),
        Index("idx_level", "level"),
        Index("idx_service", "service"),
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    level = mapped_column(String(16), nullable=False)
    service = mapped_column(String(64), nullable=False)
    message = mapped_column(Text, nullable=False)
    details = mapped_column(JSON, nullable=True)
    timestamp = mapped_column(DateTime, default=_utcnow, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize system log row."""
        return {
            "id": self.id,
            "level": self.level,
            "service": self.service,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


def create_database_engine(database_url: str = "sqlite:///data/autonomous.db") -> Engine:
    """Create a database engine configured for SQLite and future DB backends."""
    if ":memory:" in database_url and database_url in _ENGINE_CACHE:
        return _ENGINE_CACHE[database_url]

    kwargs: dict[str, Any] = {"future": True}
    if database_url.startswith("sqlite"):
        if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
            sqlite_path = Path(database_url.replace("sqlite:///", "", 1))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(database_url, **kwargs)
    if ":memory:" in database_url:
        _ENGINE_CACHE[database_url] = engine
    return engine


def init_database(
    database_url: str = "sqlite:///data/autonomous.db",
    *,
    engine: Engine | None = None,
) -> Engine:
    """Create all autonomous tables and return the engine in use."""
    db_engine = engine if engine is not None else create_database_engine(database_url)
    Base.metadata.create_all(db_engine)
    return db_engine


def get_session(engine: Engine) -> Session:
    """Create and return a new SQLAlchemy session bound to the given engine."""
    session_factory = sessionmaker(bind=engine, autoflush=True, autocommit=False, future=True)
    return session_factory()
