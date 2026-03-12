"""Helpers for building fresh post-reset recovery evidence in a sandbox."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func

from bitbat.config.loader import (
    get_runtime_config,
    load_config,
    resolve_metrics_dir,
    resolve_models_dir,
)
from bitbat.model.infer import predict_bar
from bitbat.model.persist import load as load_model

from .db import AutonomousDB
from .models import PredictionOutcome
from .validator import PredictionValidator


@dataclass(slots=True)
class StagedRecoveryDataset:
    """Paths and row counts for a staged recovery train/eval split."""

    freq: str
    horizon: str
    training_rows: int
    evaluation_rows: int
    training_dataset_path: Path
    evaluation_dataset_path: Path


@dataclass(slots=True)
class RecoveryEvidenceSummary:
    """Saved summary for fresh recovery evidence."""

    freq: str
    horizon: str
    accuracy: float
    correct_count: int
    realized_count: int
    zero_return_count: int
    direction_counts: dict[str, int]
    validation_result: dict[str, Any]
    model_path: str
    database_url: str
    evaluation_dataset_path: str
    price_dataset_path: str
    evidence_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config() -> dict[str, Any]:
    return get_runtime_config() or load_config()


def _resolve_pair(
    *,
    freq: str | None = None,
    horizon: str | None = None,
) -> tuple[str, str]:
    cfg = _config()
    resolved_freq = str(freq or cfg.get("freq", "5m"))
    resolved_horizon = str(horizon or cfg.get("horizon", "30m"))
    return resolved_freq, resolved_horizon


def _data_dir() -> Path:
    return Path(str(_config().get("data_dir", "data"))).expanduser()


def _database_url() -> str:
    cfg = _config()
    return str(cfg.get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db"))


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Recovery evidence only supports sqlite URLs, got: {database_url}")
    return Path(database_url.removeprefix(prefix))


def _timedelta_from_duration(value: str) -> timedelta:
    text = value.strip().lower()
    if len(text) < 2:
        raise ValueError(f"Invalid duration: {value}")
    amount = int(text[:-1])
    unit = text[-1]
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    raise ValueError(f"Invalid duration: {value}")


def _evaluation_dataset_path(freq: str, horizon: str) -> Path:
    return resolve_metrics_dir() / f"recovery_eval_{freq}_{horizon}.parquet"


def _evidence_output_path() -> Path:
    return resolve_metrics_dir() / "recovery_evidence.json"


def stage_recovery_dataset(
    source_dataset: Path,
    *,
    evaluation_rows: int,
    freq: str | None = None,
    horizon: str | None = None,
) -> StagedRecoveryDataset:
    """Write train/eval datasets into the configured sandbox layout."""
    if evaluation_rows <= 0:
        raise ValueError("evaluation_rows must be > 0")
    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")

    resolved_freq, resolved_horizon = _resolve_pair(freq=freq, horizon=horizon)
    dataset = pd.read_parquet(source_dataset).sort_values("timestamp_utc").reset_index(drop=True)
    if len(dataset) <= evaluation_rows:
        raise ValueError(
            "Dataset has "
            f"{len(dataset)} rows but evaluation_rows={evaluation_rows} leaves no train split"
        )

    training = dataset.iloc[:-evaluation_rows].copy()
    evaluation = dataset.iloc[-evaluation_rows:].copy()

    training_path = (
        _data_dir() / "features" / f"{resolved_freq}_{resolved_horizon}" / "dataset.parquet"
    )
    evaluation_path = _evaluation_dataset_path(resolved_freq, resolved_horizon)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_parquet(training_path, index=False)
    evaluation.to_parquet(evaluation_path, index=False)

    return StagedRecoveryDataset(
        freq=resolved_freq,
        horizon=resolved_horizon,
        training_rows=len(training),
        evaluation_rows=len(evaluation),
        training_dataset_path=training_path,
        evaluation_dataset_path=evaluation_path,
    )


def _synthetic_price_history(dataset: pd.DataFrame, horizon: str) -> pd.DataFrame:
    ordered = dataset.sort_values("timestamp_utc").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp_utc"], utc=True, errors="coerce").dt.tz_localize(
        None
    )
    returns = pd.to_numeric(ordered["r_forward"], errors="coerce")
    if timestamps.isna().any() or returns.isna().any():
        raise ValueError("Evaluation dataset contains invalid timestamps or r_forward values")

    step = _timedelta_from_duration(horizon)
    close = 100.0
    rows: list[dict[str, Any]] = []
    for timestamp, forward_return in zip(timestamps, returns, strict=True):
        rows.append({
            "timestamp_utc": timestamp,
            "close": close,
        })
        close = close * (1.0 + float(forward_return))
    rows.append({
        "timestamp_utc": timestamps.iloc[-1] + step,
        "close": close,
    })
    return pd.DataFrame(rows)


def build_recovery_evidence(
    evaluation_dataset_path: Path | None = None,
    *,
    freq: str | None = None,
    horizon: str | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
    model_version: str = "recovery-evidence",
) -> RecoveryEvidenceSummary:
    """Create fresh realized prediction evidence for the configured pair."""
    resolved_freq, resolved_horizon = _resolve_pair(freq=freq, horizon=horizon)
    dataset_path = evaluation_dataset_path or _evaluation_dataset_path(
        resolved_freq,
        resolved_horizon,
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    resolved_model_path = (
        model_path or resolve_models_dir() / f"{resolved_freq}_{resolved_horizon}" / "xgb.json"
    )
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {resolved_model_path}")

    database_url = _database_url()
    db_path = _sqlite_path(database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path = output_path or _evidence_output_path()
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation = pd.read_parquet(dataset_path).sort_values("timestamp_utc").reset_index(drop=True)
    price_history = _synthetic_price_history(evaluation, resolved_horizon)
    price_path = _data_dir() / "raw" / "prices" / f"recovery_{resolved_freq}.parquet"
    price_path.parent.mkdir(parents=True, exist_ok=True)
    price_history.to_parquet(price_path, index=False)

    db = AutonomousDB(database_url)
    with db.session() as session:
        existing = db.get_prediction_counts(session, resolved_freq, resolved_horizon)
        if int(existing["total_predictions"]) > 0:
            raise RuntimeError(
                "Recovery evidence requires an empty prediction_outcomes set for the target pair."
            )

    booster = load_model(resolved_model_path, family="xgb")
    feature_cols = [column for column in evaluation.columns if column.startswith("feat_")]
    for row in evaluation.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp_utc).to_pydatetime()
        features = pd.Series(
            {column: getattr(row, column) for column in feature_cols},
            dtype="float64",
        )
        prediction = predict_bar(
            booster,
            features,
            timestamp=timestamp,
            tau=float(_config().get("tau", 0.01)),
        )
        with db.session() as session:
            db.store_prediction(
                session=session,
                timestamp_utc=timestamp,
                predicted_direction=str(prediction["predicted_direction"]),
                model_version=model_version,
                freq=resolved_freq,
                horizon=resolved_horizon,
                predicted_return=prediction.get("predicted_return"),
                predicted_price=prediction.get("predicted_price"),
                p_up=float(prediction.get("p_up", 0.0)),
                p_down=float(prediction.get("p_down", 0.0)),
                p_flat=float(prediction.get("p_flat", 0.0)),
            )

    validation = PredictionValidator(
        db,
        freq=resolved_freq,
        horizon=resolved_horizon,
        tau=float(_config().get("tau", 0.01)),
    ).validate_all()

    with db.session() as session:
        base_query = session.query(PredictionOutcome).filter(
            PredictionOutcome.freq == resolved_freq,
            PredictionOutcome.horizon == resolved_horizon,
        )
        realized_query = base_query.filter(PredictionOutcome.actual_return.is_not(None))
        realized_count = int(realized_query.count())
        correct_count = int(realized_query.filter(PredictionOutcome.correct.is_(True)).count())
        zero_return_count = int(
            realized_query.filter(PredictionOutcome.actual_return == 0.0).count()
        )
        direction_rows = (
            session.query(
                PredictionOutcome.predicted_direction,
                func.count(PredictionOutcome.id),
            )
            .filter(
                PredictionOutcome.freq == resolved_freq,
                PredictionOutcome.horizon == resolved_horizon,
            )
            .group_by(PredictionOutcome.predicted_direction)
            .all()
        )

    accuracy = float(correct_count / realized_count) if realized_count else 0.0
    direction_counts = {str(direction): int(count) for direction, count in direction_rows}
    summary = RecoveryEvidenceSummary(
        freq=resolved_freq,
        horizon=resolved_horizon,
        accuracy=accuracy,
        correct_count=correct_count,
        realized_count=realized_count,
        zero_return_count=zero_return_count,
        direction_counts=direction_counts,
        validation_result=validation,
        model_path=str(resolved_model_path),
        database_url=database_url,
        evaluation_dataset_path=str(dataset_path),
        price_dataset_path=str(price_path),
        evidence_path=str(metrics_output_path),
    )
    metrics_output_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary
