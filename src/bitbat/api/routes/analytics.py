"""Analytics endpoints — feature importance and system status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from bitbat.api.schemas import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    SystemStatusResponse,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance(
    freq: str = Query("1h"),
    horizon: str = Query("4h"),
    top_n: int = Query(20, ge=1, le=100),
) -> FeatureImportanceResponse:
    """Return model feature importance (gain-based) from the trained booster."""
    import xgboost as xgb

    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"No model at {model_path}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    raw = booster.get_score(importance_type="gain")
    # Normalise: flatten any list values
    importance = {
        k: float(v[0] if isinstance(v, list) else v) for k, v in raw.items()
    }
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return FeatureImportanceResponse(
        model_path=str(model_path),
        features=[FeatureImportanceItem(feature=k, importance=v) for k, v in sorted_items],
    )


@router.get("/status", response_model=SystemStatusResponse)
def system_status(
    freq: str = Query("1h"),
    horizon: str = Query("4h"),
) -> SystemStatusResponse:
    """Return a summary of system readiness (DB, model, dataset, predictions)."""
    db_path = Path("data/autonomous.db")
    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    dataset_path = Path("data/features") / f"{freq}_{horizon}" / "dataset.parquet"

    database_ok = db_path.exists()
    model_exists = model_path.exists()
    dataset_exists = dataset_path.exists()

    active_version = None
    total_predictions = 0
    last_prediction_time = None

    if database_ok:
        try:
            from bitbat.autonomous.db import AutonomousDB

            db = AutonomousDB(f"sqlite:///{db_path}")
            with db.session() as session:
                active = db.get_active_model(session, freq, horizon)
                if active:
                    active_version = active.version
                rows = db.get_recent_predictions(
                    session, freq, horizon, days=365, realized_only=False
                )
                total_predictions = len(rows)
                if rows:
                    last_prediction_time = rows[0].timestamp_utc
        except Exception:
            database_ok = False

    return SystemStatusResponse(
        database_ok=database_ok,
        model_exists=model_exists,
        dataset_exists=dataset_exists,
        active_model_version=active_version,
        total_predictions=total_predictions,
        last_prediction_time=last_prediction_time,
    )
