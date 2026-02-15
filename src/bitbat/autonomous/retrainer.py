"""Automatic retraining workflow for degraded model performance."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bitbat import __version__
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import ModelVersion
from bitbat.config.loader import get_runtime_config, load_config

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AutoRetrainer:
    """Run retraining and model deployment when drift is detected."""

    def __init__(self, db: AutonomousDB, freq: str = "1h", horizon: str = "4h") -> None:
        self.db = db
        self.freq = freq
        self.horizon = horizon

        config = get_runtime_config() or load_config()
        self.tau = float(config.get("tau", 0.01))
        autonomous = config.get("autonomous", {})
        retraining_cfg = autonomous.get("retraining", {})
        self.cv_improvement_threshold = float(retraining_cfg.get("cv_improvement_threshold", 0.02))
        self.max_training_time_seconds = int(retraining_cfg.get("max_training_time_seconds", 3600))
        self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Run an external command and raise on non-zero exit status."""
        logger.info("Running command: %s", " ".join(command))
        return subprocess.run(  # noqa: S603,S607
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.max_training_time_seconds,
        )

    def _cv_summary_path(self) -> Path:
        return Path("metrics") / "cv_summary.json"

    def _read_cv_score(self) -> float:
        path = self._cv_summary_path()
        if not path.exists():
            logger.warning("CV summary not found at %s; defaulting score to 0.0", path)
            return 0.0

        payload = json.loads(path.read_text(encoding="utf-8"))
        score = payload.get("average_balanced_accuracy")
        if score is None:
            return 0.0
        return float(score)

    def _training_sample_count(self) -> int:
        dataset_path = (
            self.data_dir / "features" / f"{self.freq}_{self.horizon}" / "dataset.parquet"
        )
        if not dataset_path.exists():
            return 0
        frame = pd.read_parquet(dataset_path)
        return int(len(frame))

    def _next_model_version(self) -> str:
        timestamp = _utcnow().strftime("%Y%m%d%H%M%S")
        return f"{__version__}-{timestamp}"

    def should_deploy(
        self,
        new_model: dict[str, Any],
        old_model: dict[str, Any] | None,
    ) -> bool:
        """Return True when deployment thresholds are satisfied."""
        old_cv = float(old_model["cv_score"]) if old_model and old_model.get("cv_score") else 0.0
        new_cv = float(new_model.get("cv_score", 0.0))
        holdout_hit_rate = float(new_model.get("holdout_hit_rate", new_cv))

        improvement = new_cv - old_cv
        return improvement >= self.cv_improvement_threshold and holdout_hit_rate >= 0.55

    def deploy_model(self, model_version: str) -> None:
        """Mark new model active and deactivate older models for same freq/horizon."""
        with self.db.session() as session:
            self.db.deactivate_old_models(session, self.freq, self.horizon)
            candidate = (
                session.query(ModelVersion).filter(ModelVersion.version == model_version).first()
            )
            if candidate is None:
                raise ValueError(f"Model version not found for deployment: {model_version}")
            candidate.is_active = True
            candidate.deployed_at = _utcnow()

    def retrain(
        self,
        trigger_reason: str = "drift_detected",
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run retraining pipeline and deploy model if improvement threshold is met."""
        start_monotonic = time.monotonic()
        retraining_event_id: int | None = None
        old_version: str | None = None
        old_cv_score = 0.0

        with self.db.session() as session:
            active_model = self.db.get_active_model(session, self.freq, self.horizon)
            if active_model is not None:
                old_version = active_model.version
                old_cv_score = float(active_model.cv_score or 0.0)

            event = self.db.create_retraining_event(
                session=session,
                trigger_reason=trigger_reason,
                trigger_metrics=trigger_metrics or {},
                old_model_version=old_version,
            )
            retraining_event_id = int(event.id)

        try:
            end_dt = _utcnow()
            start_dt = end_dt - timedelta(days=365)
            start_iso = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_iso = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            self._run_command([
                "poetry",
                "run",
                "bitbat",
                "features",
                "build",
                "--tau",
                str(self.tau),
            ])
            self._run_command([
                "poetry",
                "run",
                "bitbat",
                "model",
                "cv",
                "--freq",
                self.freq,
                "--horizon",
                self.horizon,
                "--start",
                start_iso,
                "--end",
                end_iso,
            ])
            self._run_command([
                "poetry",
                "run",
                "bitbat",
                "model",
                "train",
                "--freq",
                self.freq,
                "--horizon",
                self.horizon,
            ])

            new_cv_score = self._read_cv_score()
            new_version = self._next_model_version()
            training_samples = self._training_sample_count()

            with self.db.session() as session:
                self.db.store_model_version(
                    session=session,
                    version=new_version,
                    freq=self.freq,
                    horizon=self.horizon,
                    training_start=start_dt,
                    training_end=end_dt,
                    training_samples=training_samples,
                    cv_score=new_cv_score,
                    features=[],
                    hyperparameters={},
                    training_metadata={"tau": self.tau, "source": "auto_retrainer"},
                    is_active=False,
                )

            deploy = self.should_deploy(
                new_model={"cv_score": new_cv_score, "holdout_hit_rate": new_cv_score},
                old_model={"cv_score": old_cv_score} if old_version else None,
            )
            if deploy:
                self.deploy_model(new_version)

            duration = time.monotonic() - start_monotonic
            if retraining_event_id is None:
                raise ValueError("Retraining event missing.")

            with self.db.session() as session:
                self.db.complete_retraining_event(
                    session=session,
                    event_id=retraining_event_id,
                    new_model_version=new_version,
                    cv_improvement=new_cv_score - old_cv_score,
                    training_duration_seconds=duration,
                )

            return {
                "status": "completed",
                "event_id": retraining_event_id,
                "old_model_version": old_version,
                "new_model_version": new_version,
                "old_cv_score": old_cv_score,
                "new_cv_score": new_cv_score,
                "cv_improvement": new_cv_score - old_cv_score,
                "deployed": deploy,
                "duration_seconds": duration,
            }

        except Exception as exc:
            duration = time.monotonic() - start_monotonic
            if retraining_event_id is not None:
                with self.db.session() as session:
                    self.db.fail_retraining_event(
                        session=session,
                        event_id=retraining_event_id,
                        error_message=str(exc),
                    )
            logger.exception("Auto-retraining failed")
            return {
                "status": "failed",
                "event_id": retraining_event_id,
                "old_model_version": old_version,
                "error": str(exc),
                "duration_seconds": duration,
            }
