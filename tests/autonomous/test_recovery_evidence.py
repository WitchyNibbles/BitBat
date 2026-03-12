from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bitbat.config.loader import reset_runtime_config, resolve_metrics_dir, set_runtime_config
from bitbat.model.persist import save_baseline_artifact
from bitbat.model.train import fit_xgb


def _write_recovery_config(path: Path, root: Path) -> Path:
    data_dir = root / "data"
    models_dir = root / "models"
    metrics_dir = root / "metrics"
    db_path = data_dir / "autonomous.db"
    path.write_text(
        "\n".join([
            f'data_dir: "{data_dir}"',
            f'models_dir: "{models_dir}"',
            f'metrics_dir: "{metrics_dir}"',
            'freq: "1h"',
            'horizon: "1h"',
            "tau: 0.01",
            "seed: 42",
            "enable_sentiment: false",
            "autonomous:",
            f'  database_url: "sqlite:///{db_path}"',
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.integration
def test_build_recovery_evidence_generates_fresh_realized_accuracy(tmp_path: Path) -> None:
    source_dataset = Path("data/features/1h_1h/dataset.parquet")
    if not source_dataset.exists():
        pytest.skip("recovery source dataset not present")

    from bitbat.autonomous.recovery_evidence import build_recovery_evidence, stage_recovery_dataset

    sandbox_root = tmp_path / "sandbox"
    config_path = _write_recovery_config(tmp_path / "recovery.yaml", sandbox_root)

    set_runtime_config(config_path)
    try:
        staged = stage_recovery_dataset(source_dataset=source_dataset, evaluation_rows=240)
        train_df = pd.read_parquet(staged.training_dataset_path).sort_values("timestamp_utc")
        feature_cols = [col for col in train_df.columns if col.startswith("feat_")]
        X_train = train_df[feature_cols].set_index(train_df["timestamp_utc"])
        X_train.attrs["freq"] = "1h"
        X_train.attrs["horizon"] = "1h"
        booster, _ = fit_xgb(
            X_train,
            train_df["label"].astype(str),
            seed=42,
            persist=False,
        )
        save_baseline_artifact(
            booster,
            family="xgb",
            freq="1h",
            horizon="1h",
            metadata={"source": "test-recovery-evidence"},
        )

        summary = build_recovery_evidence(staged.evaluation_dataset_path)
        evidence_payload = json.loads(
            (resolve_metrics_dir() / "recovery_evidence.json").read_text(encoding="utf-8")
        )
    finally:
        reset_runtime_config()

    assert summary.realized_count == 240
    assert summary.accuracy > 0.33
    assert summary.direction_counts["flat"] > 0
    assert summary.zero_return_count < 50
    assert evidence_payload["accuracy"] == pytest.approx(summary.accuracy)
    assert evidence_payload["realized_count"] == 240
