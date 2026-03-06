"""Prediction endpoints — latest, history, and performance."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.api.schemas import (
    PerformanceResponse,
    PredictionListResponse,
    PredictionResponse,
)
from bitbat.autonomous.db import AutonomousDB

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
    return PredictionResponse(
        id=p.id,
        timestamp_utc=p.timestamp_utc,
        predicted_direction=p.predicted_direction,
        predicted_return=p.predicted_return,
        predicted_price=p.predicted_price,
        actual_direction=p.actual_direction,
        actual_return=p.actual_return,
        correct=p.correct,
        model_version=p.model_version,
        freq=p.freq,
        horizon=p.horizon,
    )


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

        # Directional accuracy, MAE, RMSE from predicted_return vs actual_return
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
