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
