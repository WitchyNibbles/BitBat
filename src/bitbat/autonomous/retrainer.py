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
from bitbat.config.loader import get_runtime_config, load_config, resolve_metrics_dir

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
        self.train_window_days = int(retraining_cfg.get("train_window_days", 365))
        self.backtest_window_days = int(retraining_cfg.get("backtest_window_days", 90))
        self.window_step_days = int(retraining_cfg.get("window_step_days", 90))
        self.cv_window_count = int(retraining_cfg.get("cv_windows", 3))
        self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
        self._last_cv_summary: dict[str, Any] = {}

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
        return resolve_metrics_dir() / "cv_summary.json"

    def _read_cv_score(self) -> float:
        path = self._cv_summary_path()
        if not path.exists():
            logger.warning("CV summary not found at %s; defaulting score to 0.0", path)
            self._last_cv_summary = {}
            return 0.0

        payload = json.loads(path.read_text(encoding="utf-8"))
        self._last_cv_summary = payload
        score = payload.get("mean_directional_accuracy")
        if score is not None:
            return float(score)

        champion = payload.get("champion_decision", {})
        winner = champion.get("winner") if isinstance(champion, dict) else None
        reports = payload.get("candidate_reports", {})
        if winner and isinstance(reports, dict):
            winner_report = reports.get(winner, {})
            directional = winner_report.get("metrics", {}).get("directional", {})
            winner_score = directional.get("mean_directional_accuracy")
            if winner_score is not None:
                return float(winner_score)

        fallback_rmse = payload.get("average_rmse")
        if fallback_rmse is not None:
            return max(0.0, 1.0 - float(fallback_rmse))

        return 0.0

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

    def _build_cv_windows(
        self, anchor_end: datetime
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:  # noqa: E501
        """Build rolling train/backtest windows for the CV command."""
        train_delta = timedelta(days=self.train_window_days)
        backtest_delta = timedelta(days=self.backtest_window_days)
        step_delta = timedelta(days=max(self.window_step_days, 1))
        window_count = max(self.cv_window_count, 1)

        windows: list[tuple[datetime, datetime, datetime, datetime]] = []
        for offset in range(window_count - 1, -1, -1):
            test_start = anchor_end - backtest_delta - (step_delta * offset)
            test_end = test_start + backtest_delta
            train_end = test_start
            train_start = train_end - train_delta
            windows.append((train_start, train_end, test_start, test_end))
        return windows

    def should_deploy(
        self,
        new_model: dict[str, Any],
        old_model: dict[str, Any] | None,
    ) -> bool:
        """Return True when deployment thresholds are satisfied."""
        old_cv = float(old_model["cv_score"]) if old_model and old_model.get("cv_score") else 0.0
        new_cv = float(new_model.get("cv_score", 0.0))
        holdout_hit_rate = float(new_model.get("holdout_hit_rate", new_cv))
        champion_decision = new_model.get("champion_decision", {})
        if (
            isinstance(champion_decision, dict)
            and champion_decision.get("promote_candidate") is False
        ):
            return False
        promotion_gate = (
            champion_decision.get("promotion_gate", {})
            if isinstance(champion_decision, dict)
            else new_model.get("promotion_gate", {})
        )
        if isinstance(promotion_gate, dict) and promotion_gate.get("pass") is False:
            return False

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
            windows = self._build_cv_windows(end_dt)
            start_dt = windows[0][0]
            cv_end_dt = windows[-1][3]
            start_iso = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_iso = cv_end_dt.strftime("%Y-%m-%d %H:%M:%S")

            self._run_command([
                "poetry",
                "run",
                "bitbat",
                "features",
                "build",
            ])
            self._run_command(
                [
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
                ]
                + [
                    arg
                    for train_start, train_end, test_start, test_end in windows
                    for arg in (
                        "--windows",
                        train_start.strftime("%Y-%m-%d %H:%M:%S"),
                        train_end.strftime("%Y-%m-%d %H:%M:%S"),
                        test_start.strftime("%Y-%m-%d %H:%M:%S"),
                        test_end.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                ]
            )
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
            champion_decision_payload = self._last_cv_summary.get("champion_decision", {})
            promotion_gate_payload = (
                champion_decision_payload.get("promotion_gate", {})
                if isinstance(champion_decision_payload, dict)
                else {}
            )

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
                    training_metadata={
                        "tau": self.tau,
                        "source": "auto_retrainer",
                        "window_config": {
                            "train_window_days": self.train_window_days,
                            "backtest_window_days": self.backtest_window_days,
                            "window_step_days": self.window_step_days,
                            "cv_windows": self.cv_window_count,
                        },
                        "champion_decision": champion_decision_payload,
                        "promotion_gate": promotion_gate_payload,
                        "candidate_reports": self._last_cv_summary.get("candidate_reports"),
                    },
                    is_active=False,
                )

            deploy = self.should_deploy(
                new_model={
                    "cv_score": new_cv_score,
                    "holdout_hit_rate": new_cv_score,
                    "champion_decision": champion_decision_payload,
                    "promotion_gate": promotion_gate_payload,
                },
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
