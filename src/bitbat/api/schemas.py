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
    status: str = Field(description="ok | unavailable | error")
    detail: str | None = None


class DetailedHealthResponse(HealthResponse):
    services: list[ServiceStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    id: int
    timestamp_utc: datetime
    predicted_direction: str
    p_up: float
    p_down: float
    p_flat: float
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
    model_exists: bool
    dataset_exists: bool
    active_model_version: str | None = None
    total_predictions: int = 0
    last_prediction_time: datetime | None = None
