"""Analytics endpoints — feature importance and system status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from bitbat.api.defaults import _default_freq, _default_horizon
from bitbat.api.schemas import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    SchemaReadinessDetails,
    SystemStatusResponse,
)
from bitbat.autonomous.schema_compat import audit_schema_compatibility, format_missing_columns

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Compute once at import time from config
_FREQ = _default_freq()
_HORIZON = _default_horizon()


def _schema_readiness(db_path: Path) -> SchemaReadinessDetails:
    """Audit schema compatibility for readiness without mutating DB state."""
    if not db_path.exists():
        return SchemaReadinessDetails(
            compatibility_state="unavailable",
            is_compatible=False,
            detail="autonomous.db not found",
        )

    try:
        report = audit_schema_compatibility(database_url=f"sqlite:///{db_path}")
    except Exception as exc:  # noqa: BLE001
        return SchemaReadinessDetails(
            compatibility_state="error",
            is_compatible=False,
            detail=f"schema audit failed: {exc}",
        )

    if report.is_compatible:
        return SchemaReadinessDetails(
            compatibility_state="compatible",
            is_compatible=True,
            can_auto_upgrade=report.can_auto_upgrade,
        )

    missing_columns = {
        table_name: list(columns) for table_name, columns in report.missing_columns.items()
    }
    missing_text = format_missing_columns(report)
    return SchemaReadinessDetails(
        compatibility_state="incompatible",
        is_compatible=False,
        can_auto_upgrade=report.can_auto_upgrade,
        missing_columns=missing_columns,
        missing_columns_text=missing_text,
        detail=f"missing required columns: {missing_text}",
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def feature_importance(
    freq: str = Query(_FREQ),
    horizon: str = Query(_HORIZON),
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
    importance = {k: float(v[0] if isinstance(v, list) else v) for k, v in raw.items()}
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return FeatureImportanceResponse(
        model_path=str(model_path),
        features=[FeatureImportanceItem(feature=k, importance=v) for k, v in sorted_items],
    )


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(
    freq: str = Query(_FREQ),
    horizon: str = Query(_HORIZON),
) -> SystemStatusResponse:
    """Return a summary of system readiness (DB, model, dataset, predictions)."""
    db_path = Path("data/autonomous.db")
    model_path = Path("models") / f"{freq}_{horizon}" / "xgb.json"
    dataset_path = Path("data/features") / f"{freq}_{horizon}" / "dataset.parquet"

    database_present = db_path.exists()
    schema_readiness = _schema_readiness(db_path)
    database_ok = database_present and schema_readiness.is_compatible
    model_exists = model_path.exists()
    dataset_exists = dataset_path.exists()

    active_version = None
    total_predictions = 0
    last_prediction_time = None

    if database_ok:
        try:
            from bitbat.autonomous.db import AutonomousDB

            db = AutonomousDB(f"sqlite:///{db_path}", auto_upgrade_schema=False)
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
        except Exception:  # noqa: BLE001
            database_ok = False

    return SystemStatusResponse(
        database_ok=database_ok,
        database_present=database_present,
        model_exists=model_exists,
        dataset_exists=dataset_exists,
        schema_readiness=schema_readiness,
        active_model_version=active_version,
        total_predictions=total_predictions,
        last_prediction_time=last_prediction_time,
    )
