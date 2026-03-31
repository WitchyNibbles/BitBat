"""Prediction endpoints — latest, history, and performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.api.schemas import (
    PerformanceResponse,
    PriceTimelinePoint,
    PredictionListResponse,
    PredictionResponse,
    PredictionTimelinePoint,
    PredictionTimelineResponse,
)
from bitbat.autonomous.db import AutonomousDB
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
        actual_direction=p.actual_direction,
        actual_return=p.actual_return,
        correct=p.correct,
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


@router.get("/latest", response_model=PredictionResponse)
async def latest_prediction(
    freq: str = Query(_FREQ, description="Bar frequency"),
    horizon: str = Query(_HORIZON, description="Prediction horizon"),
) -> PredictionResponse:
    """Return the most recent prediction for the requested config."""
    db = _get_db()
    with db.session() as session:
        rows = db.get_recent_predictions(session, freq, horizon, days=7, realized_only=False)
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
        rows = db.get_recent_predictions(session, freq, horizon, days=days, realized_only=False)
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
        rows = db.get_recent_predictions(session, freq, horizon, days=days, realized_only=False)

    limited = rows[:limit]
    if not limited:
        return PredictionTimelineResponse(
            points=[],
            price_points=[],
            total=0,
            freq=freq,
            horizon=horizon,
        )

    points_frame = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(row.timestamp_utc),
                "predicted_price": row.predicted_price,
                "predicted_direction": row.predicted_direction,
                "confidence": _timeline_confidence(row),
                "correct": row.correct,
                "is_realized": bool(
                    row.correct is not None
                    or row.actual_direction is not None
                    or row.actual_return is not None
                ),
            }
            for row in limited
        ]
    )
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
        points_frame = pd.merge_asof(
            points_frame,
            price_frame,
            on="timestamp_utc",
            direction="nearest",
            tolerance=tolerance,
        )
    else:
        points_frame["actual_price"] = pd.NA

    points = [
        PredictionTimelinePoint(
            timestamp_utc=record["timestamp_utc"].to_pydatetime(),
            actual_price=(
                float(record["actual_price"])
                if pd.notna(record.get("actual_price"))
                else None
            ),
            predicted_price=(
                float(record["predicted_price"])
                if pd.notna(record.get("predicted_price"))
                else None
            ),
            predicted_direction=str(record["predicted_direction"]),
            confidence=(
                float(record["confidence"])
                if pd.notna(record.get("confidence"))
                else None
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
        rows = db.get_recent_predictions(session, freq, horizon, days=days, realized_only=True)
        total = len(rows)
        if total == 0:
            return PerformanceResponse(
                freq=freq,
                horizon=horizon,
                window_days=days,
                total_predictions=0,
                realized_predictions=0,
            )

        correct = sum(1 for r in rows if r.correct)
        hit_rate = correct / total if total else None
        returns = [r.actual_return for r in rows if r.actual_return is not None]
        avg_ret = sum(returns) / len(returns) if returns else None

        # Directional accuracy, MAE, RMSE from predicted_return vs actual_return.
        # Newer classifier rows do not carry predicted_return, so directional
        # accuracy must fall back to the stored direction labels.
        dir_correct = 0
        dir_total = 0
        errors: list[float] = []
        for r in rows:
            pr = getattr(r, "predicted_return", None)
            ar = r.actual_return
            if pr is not None and ar is not None:
                errors.append(pr - ar)
                # Directional accuracy: both same sign (or both zero)
                if (pr >= 0 and ar >= 0) or (pr < 0 and ar < 0):
                    dir_correct += 1
                dir_total += 1
                continue

            predicted_direction = getattr(r, "predicted_direction", None)
            actual_direction = getattr(r, "actual_direction", None)
            if predicted_direction is not None and actual_direction is not None:
                dir_correct += int(predicted_direction == actual_direction)
                dir_total += 1

        mae = sum(abs(e) for e in errors) / len(errors) if errors else None
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5 if errors else None
        directional_accuracy = dir_correct / dir_total if dir_total else None

        # Streaks
        win_streak = lose_streak = cur_win = cur_lose = 0
        for r in reversed(rows):
            if r.correct:
                cur_win += 1
                cur_lose = 0
            else:
                cur_lose += 1
                cur_win = 0
            win_streak = max(win_streak, cur_win)
            lose_streak = max(lose_streak, cur_lose)

        model_ver = rows[0].model_version if rows else None

        return PerformanceResponse(
            model_version=model_ver,
            freq=freq,
            horizon=horizon,
            window_days=days,
            total_predictions=total,
            realized_predictions=total,
            hit_rate=hit_rate,
            avg_return=avg_ret,
            win_streak=win_streak,
            lose_streak=lose_streak,
            mae=mae,
            rmse=rmse,
            directional_accuracy=directional_accuracy,
        )
