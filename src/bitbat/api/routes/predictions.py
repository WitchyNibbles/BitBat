"""Prediction endpoints — latest, history, and performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.api.schemas import (
    PerformanceResponse,
    PredictionListResponse,
    PredictionResponse,
    PredictionTimelinePoint,
    PredictionTimelineResponse,
    PriceTimelinePoint,
)
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.model.infer import prediction_confidence

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Compute once at import time from config
_FREQ = _default_freq()
_HORIZON = _default_horizon()


def _get_db() -> AutonomousDB:
    """Return a database handle, raising 503 if unavailable."""
    db_path = Path("data/autonomous.db")
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Database not available")
    return AutonomousDB(f"sqlite:///{db_path}")


def _prediction_to_response(p) -> PredictionResponse:  # type: ignore[no-untyped-def]
    from bitbat.config.loader import get_runtime_config, load_config

    config = get_runtime_config() or load_config()
    tau = float(config.get("tau", 0.01) or 0.01)
    p_up = getattr(p, "p_up", None)
    p_down = getattr(p, "p_down", None)
    p_flat = getattr(p, "p_flat", None)
    direction = str(getattr(p, "predicted_direction", "")).lower()
    if (
        direction == "flat"
        and p_up is not None
        and p_down is not None
        and (p_flat is None or float(p_flat) == 0.0)
    ):
        remaining_mass = max(0.0, 1.0 - float(p_up) - float(p_down))
        if remaining_mass > 0.0:
            p_flat = remaining_mass
    confidence = prediction_confidence(
        direction,
        p_up=p_up,
        p_down=p_down,
        p_flat=p_flat,
    )
    actual_return = getattr(p, "actual_return", None)
    actual_direction = getattr(p, "actual_direction", None)
    correct = getattr(p, "correct", None)
    if actual_return is not None:
        if float(actual_return) > tau:
            actual_direction = "up"
        elif float(actual_return) < -tau:
            actual_direction = "down"
        else:
            actual_direction = "flat"
        predicted_return = getattr(p, "predicted_return", None)
        if predicted_return is not None:
            if float(predicted_return) > tau:
                predicted_direction = "up"
            elif float(predicted_return) < -tau:
                predicted_direction = "down"
            else:
                predicted_direction = "flat"
        else:
            predicted_direction = direction if direction in {"up", "down", "flat"} else "flat"
        correct = predicted_direction == actual_direction

    return PredictionResponse(
        id=p.id,
        timestamp_utc=p.timestamp_utc,
        predicted_direction=p.predicted_direction,
        predicted_return=p.predicted_return,
        predicted_price=p.predicted_price,
        p_up=p_up,
        p_down=p_down,
        p_flat=p_flat,
        confidence=confidence,
        start_price=getattr(p, "start_price", None),
        end_price=getattr(p, "end_price", None),
        actual_direction=actual_direction,
        actual_return=actual_return,
        correct=correct,
        model_version=p.model_version,
        freq=p.freq,
        horizon=p.horizon,
    )


def _timeline_confidence(p) -> float | None:  # type: ignore[no-untyped-def]
    return prediction_confidence(
        str(getattr(p, "predicted_direction", "")),
        p_up=getattr(p, "p_up", None),
        p_down=getattr(p, "p_down", None),
        p_flat=getattr(p, "p_flat", None),
    )


def _load_price_points(freq: str) -> pd.DataFrame:
    from bitbat.config.loader import get_runtime_config, load_config
    from bitbat.io.prices import load_prices

    config = get_runtime_config() or load_config()
    data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
    prices = load_prices(data_dir, freq)
    frame = prices.reset_index()[["timestamp_utc", "close"]].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_localize(None)
    frame = frame.rename(columns={"close": "actual_price"})
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def _timeline_tolerance(freq: str) -> pd.Timedelta | None:
    try:
        return pd.to_timedelta(freq) / 2
    except Exception:
        return None


def _target_timestamps(timestamps: pd.Series, horizon: str) -> pd.Series:
    try:
        delta = pd.to_timedelta(horizon)
    except Exception:
        return timestamps
    return timestamps + delta


@router.get("/latest", response_model=PredictionResponse)
async def latest_prediction(
    freq: str = Query(_FREQ, description="Bar frequency"),
    horizon: str = Query(_HORIZON, description="Prediction horizon"),
) -> PredictionResponse:
    """Return the most recent prediction for the requested config."""
    db = _get_db()
    with db.session() as session:
        rows = db.get_pair_predictions(session, freq, horizon, realized_only=False)
        if not rows:
            raise HTTPException(status_code=404, detail="No predictions found")
        return _prediction_to_response(rows[0])


@router.get("/history", response_model=PredictionListResponse)
async def prediction_history(
    freq: str = Query(_FREQ),
    horizon: str = Query(_HORIZON),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(100, ge=1, le=1000),
) -> PredictionListResponse:
    """Return recent prediction history."""
    db = _get_db()
    with db.session() as session:
        rows = db.get_pair_predictions(session, freq, horizon, realized_only=False)
        limited = rows[:limit]
        return PredictionListResponse(
            predictions=[_prediction_to_response(r) for r in limited],
            total=len(rows),
            freq=freq,
            horizon=horizon,
        )


@router.get("/timeline", response_model=PredictionTimelineResponse)
async def prediction_timeline(
    freq: str = Query(_FREQ),
    horizon: str = Query(_HORIZON),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(48, ge=1, le=1000),
) -> PredictionTimelineResponse:
    """Return chart-friendly price/prediction points for the Home view."""
    db = _get_db()
    with db.session() as session:
        rows = db.get_pair_predictions(session, freq, horizon, realized_only=False)

    limited = rows[:limit]
    if not limited:
        return PredictionTimelineResponse(
            points=[],
            price_points=[],
            total=0,
            freq=freq,
            horizon=horizon,
        )

    points_frame = pd.DataFrame([
        {
            "timestamp_utc": pd.Timestamp(row.timestamp_utc),
            "predicted_price": row.predicted_price,
            "predicted_direction": row.predicted_direction,
            "confidence": _timeline_confidence(row),
            "correct": _prediction_to_response(row).correct,
            "is_realized": bool(
                row.correct is not None
                or row.actual_direction is not None
                or row.actual_return is not None
            ),
        }
        for row in limited
    ])
    points_frame["timestamp_utc"] = pd.to_datetime(
        points_frame["timestamp_utc"], utc=True
    ).dt.tz_localize(None)
    points_frame = points_frame.sort_values("timestamp_utc").reset_index(drop=True)

    try:
        price_frame = _load_price_points(freq)
    except RuntimeError:
        price_frame = pd.DataFrame(columns=["timestamp_utc", "actual_price"])

    dense_price_points: list[PriceTimelinePoint] = []
    if not price_frame.empty:
        start_time = points_frame["timestamp_utc"].min()
        end_time = points_frame["timestamp_utc"].max()
        price_window = price_frame.loc[
            price_frame["timestamp_utc"].between(start_time, end_time, inclusive="both")
        ].copy()
        dense_price_points = [
            PriceTimelinePoint(
                timestamp_utc=record["timestamp_utc"].to_pydatetime(),
                actual_price=float(record["actual_price"]),
            )
            for record in price_window.to_dict(orient="records")
            if pd.notna(record.get("actual_price"))
        ]

        tolerance = _timeline_tolerance(freq)
        target_frame = points_frame.assign(
            target_timestamp_utc=_target_timestamps(points_frame["timestamp_utc"], horizon)
        )
        target_prices = pd.merge_asof(
            target_frame.sort_values("target_timestamp_utc"),
            price_frame.rename(
                columns={
                    "timestamp_utc": "target_timestamp_utc",
                    "actual_price": "actual_price_target",
                }
            ).sort_values("target_timestamp_utc"),
            on="target_timestamp_utc",
            direction="nearest",
            tolerance=tolerance,
        ).sort_values("timestamp_utc")
        points_frame = points_frame.merge(
            target_prices.loc[:, ["timestamp_utc", "actual_price_target"]],
            on="timestamp_utc",
            how="left",
        )
        fallback_prices = pd.merge_asof(
            points_frame.sort_values("timestamp_utc"),
            price_frame,
            on="timestamp_utc",
            direction="nearest",
            tolerance=tolerance,
        ).sort_values("timestamp_utc")
        points_frame["actual_price"] = points_frame["actual_price_target"].where(
            points_frame["actual_price_target"].notna(),
            fallback_prices["actual_price"],
        )
        points_frame = points_frame.drop(columns=["actual_price_target"])
    else:
        points_frame["actual_price"] = pd.NA

    points = [
        PredictionTimelinePoint(
            timestamp_utc=record["timestamp_utc"].to_pydatetime(),
            actual_price=(
                float(record["actual_price"]) if pd.notna(record.get("actual_price")) else None
            ),
            predicted_price=(
                float(record["predicted_price"])
                if pd.notna(record.get("predicted_price"))
                else None
            ),
            predicted_direction=str(record["predicted_direction"]),
            confidence=(
                float(record["confidence"]) if pd.notna(record.get("confidence")) else None
            ),
            correct=record["correct"],
            is_realized=bool(record["is_realized"]),
        )
        for record in points_frame.to_dict(orient="records")
    ]

    return PredictionTimelineResponse(
        points=points,
        price_points=dense_price_points,
        total=len(rows),
        freq=freq,
        horizon=horizon,
    )


@router.get("/performance", response_model=PerformanceResponse)
async def prediction_performance(
    freq: str = Query(_FREQ),
    horizon: str = Query(_HORIZON),
    days: int = Query(30, ge=1, le=365),
) -> PerformanceResponse:
    """Return aggregate performance metrics for realized predictions."""
    db = _get_db()
    with db.session() as session:
        rows = db.get_pair_predictions(session, freq, horizon, realized_only=True)
        total = len(rows)
        if total == 0:
            return PerformanceResponse(
                freq=freq,
                horizon=horizon,
                window_days=days,
                total_predictions=0,
                realized_predictions=0,
            )
        metrics = PerformanceMetrics(rows).to_dict()

        model_ver = rows[0].model_version if rows else None

        return PerformanceResponse(
            model_version=model_ver,
            freq=freq,
            horizon=horizon,
            window_days=days,
            total_predictions=total,
            realized_predictions=total,
            hit_rate=float(metrics["hit_rate"]),
            avg_return=float(metrics["average_return"]),
            win_streak=int(metrics["win_streak"]),
            lose_streak=int(metrics["lose_streak"]),
            mae=float(metrics["mae"]),
            rmse=float(metrics["rmse"]),
            directional_accuracy=float(metrics["directional_accuracy"]),
        )
