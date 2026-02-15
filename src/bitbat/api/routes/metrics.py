"""Prometheus-compatible metrics endpoint."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["metrics"])

_START_TIME = time.time()


def _gauge(name: str, help_text: str, value: float | int) -> str:
    """Format a single Prometheus gauge metric."""
    return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n{name} {value}\n"


def _collect_metrics() -> str:
    """Gather system metrics and return Prometheus exposition text."""
    lines: list[str] = []

    # Uptime
    lines.append(_gauge("bitbat_uptime_seconds", "Seconds since API start", round(time.time() - _START_TIME, 1)))

    # Database
    db_path = Path("data/autonomous.db")
    lines.append(_gauge("bitbat_database_available", "1 if autonomous.db exists", int(db_path.exists())))

    # Model availability
    model_path = Path("models/1h_4h/xgb.json")
    lines.append(_gauge("bitbat_model_available", "1 if default model exists", int(model_path.exists())))

    # Dataset availability
    ds_path = Path("data/features/1h_4h/dataset.parquet")
    lines.append(_gauge("bitbat_dataset_available", "1 if default dataset exists", int(ds_path.exists())))

    # Prediction counts (from DB if available)
    if db_path.exists():
        try:
            from bitbat.autonomous.db import AutonomousDB

            db = AutonomousDB(f"sqlite:///{db_path}")
            with db.session() as session:
                all_preds = db.get_recent_predictions(
                    session, "1h", "4h", days=30, realized_only=False
                )
                realized = [p for p in all_preds if p.actual_return is not None]
                correct = [p for p in realized if p.correct]

                lines.append(
                    _gauge("bitbat_predictions_total_30d", "Total predictions in last 30 days", len(all_preds))
                )
                lines.append(
                    _gauge("bitbat_predictions_realized_30d", "Realized predictions in last 30 days", len(realized))
                )
                lines.append(
                    _gauge("bitbat_predictions_correct_30d", "Correct predictions in last 30 days", len(correct))
                )
                hit_rate = len(correct) / len(realized) if realized else 0.0
                lines.append(
                    _gauge("bitbat_hit_rate_30d", "Hit rate over last 30 days", round(hit_rate, 4))
                )
        except Exception:
            pass

    return "\n".join(lines)


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """Prometheus-compatible metrics endpoint."""
    return _collect_metrics()
