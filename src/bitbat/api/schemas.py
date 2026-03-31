"""Pydantic response/request schemas for the BitBat REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(description="Overall system status: ok | degraded | error")
    version: str = Field(description="Application version string")
    uptime_seconds: float = Field(description="Seconds since the API process started")


class ServiceStatus(BaseModel):
    name: str
    status: str = Field(description="ok | degraded | unavailable | error")
    detail: str | None = None


class SchemaReadinessDetails(BaseModel):
    compatibility_state: str = Field(description="compatible | incompatible | unavailable | error")
    is_compatible: bool
    can_auto_upgrade: bool = False
    missing_columns: dict[str, list[str]] = Field(default_factory=dict)
    missing_columns_text: str | None = None
    detail: str | None = None


class DetailedHealthResponse(HealthResponse):
    services: list[ServiceStatus] = Field(default_factory=list)
    schema_readiness: SchemaReadinessDetails | None = None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    id: int
    timestamp_utc: datetime
    predicted_direction: str
    predicted_return: float | None = None
    predicted_price: float | None = None
    p_up: float | None = None
    p_down: float | None = None
    p_flat: float | None = None
    confidence: float | None = None
    actual_direction: str | None = None
    actual_return: float | None = None
    correct: bool | None = None
    model_version: str
    freq: str
    horizon: str


class PredictionListResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    freq: str
    horizon: str


class PredictionTimelinePoint(BaseModel):
    timestamp_utc: datetime
    actual_price: float | None = None
    predicted_price: float | None = None
    predicted_direction: str
    confidence: float | None = None
    correct: bool | None = None
    is_realized: bool = False


class PriceTimelinePoint(BaseModel):
    timestamp_utc: datetime
    actual_price: float


class PredictionTimelineResponse(BaseModel):
    points: list[PredictionTimelinePoint]
    price_points: list[PriceTimelinePoint] = Field(default_factory=list)
    total: int
    freq: str
    horizon: str


# ---------------------------------------------------------------------------
# Analytics / Performance
# ---------------------------------------------------------------------------


class PerformanceResponse(BaseModel):
    model_version: str | None = None
    freq: str
    horizon: str
    window_days: int
    total_predictions: int
    realized_predictions: int
    hit_rate: float | None = None
    avg_return: float | None = None
    win_streak: int = 0
    lose_streak: int = 0
    mae: float | None = None
    rmse: float | None = None
    directional_accuracy: float | None = None


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    model_path: str
    features: list[FeatureImportanceItem]


# ---------------------------------------------------------------------------
# System / Status
# ---------------------------------------------------------------------------


class SystemStatusResponse(BaseModel):
    database_ok: bool
    database_present: bool = False
    model_exists: bool
    dataset_exists: bool
    schema_readiness: SchemaReadinessDetails | None = None
    active_model_version: str | None = None
    total_predictions: int = 0
    last_prediction_time: datetime | None = None


# ---------------------------------------------------------------------------
# System logs, retraining events, performance snapshots
# ---------------------------------------------------------------------------


class SystemLogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    service: str | None = None


class SystemLogsResponse(BaseModel):
    logs: list[SystemLogEntry]
    total: int


class RetrainingEventEntry(BaseModel):
    id: int
    started_at: datetime
    trigger_reason: str
    status: str
    old_model_version: str | None = None
    new_model_version: str | None = None
    cv_improvement: float | None = None
    training_duration_seconds: float | None = None


class RetrainingEventsResponse(BaseModel):
    events: list[RetrainingEventEntry]
    total: int


class PerformanceSnapshotEntry(BaseModel):
    snapshot_time: datetime
    model_version: str
    hit_rate: float | None = None
    total_predictions: int = 0
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None


class PerformanceSnapshotsResponse(BaseModel):
    snapshots: list[PerformanceSnapshotEntry]


# ---------------------------------------------------------------------------
# Training & Settings
# ---------------------------------------------------------------------------


class TrainingRequest(BaseModel):
    preset: str = Field(description="Preset name: conservative, balanced, or aggressive")


class TrainingResponse(BaseModel):
    status: str
    model_version: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class SettingsResponse(BaseModel):
    preset: str = "custom"
    freq: str
    horizon: str
    tau: float
    enter_threshold: float
    valid_freqs: list[str] = Field(default_factory=list, description="Accepted frequency values")
    valid_horizons: list[str] = Field(default_factory=list, description="Accepted horizon values")


class SettingsUpdateRequest(BaseModel):
    preset: str | None = None
    freq: str | None = None
    horizon: str | None = None
    tau: float | None = None
    enter_threshold: float | None = None
