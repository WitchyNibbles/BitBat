from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bitbat.autonomous.recovery_evidence import build_recovery_evidence, stage_recovery_dataset
from bitbat.config.loader import reset_runtime_config, set_runtime_config
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


@pytest.fixture(scope="session", autouse=True)
def diagnosis_runtime_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    existing_config = os.environ.get("BITBAT_CONFIG")
    if existing_config:
        set_runtime_config(existing_config)
        try:
            yield Path(existing_config)
        finally:
            reset_runtime_config()
        return

    source_dataset = Path("data/features/1h_1h/dataset.parquet")
    if not source_dataset.exists():
        pytest.skip("diagnosis recovery source dataset not present")

    sandbox_root = tmp_path_factory.mktemp("diagnosis-runtime")
    config_path = _write_recovery_config(sandbox_root / "recovery.yaml", sandbox_root)
    os.environ["BITBAT_CONFIG"] = str(config_path)
    set_runtime_config(config_path)
    try:
        staged = stage_recovery_dataset(source_dataset=source_dataset, evaluation_rows=300)
        train_df = pd.read_parquet(staged.training_dataset_path).sort_values("timestamp_utc")
        feature_cols = [column for column in train_df.columns if column.startswith("feat_")]
        X_train = train_df[feature_cols].set_index(train_df["timestamp_utc"])
        X_train.attrs["freq"] = staged.freq
        X_train.attrs["horizon"] = staged.horizon
        booster, _ = fit_xgb(
            X_train,
            train_df["label"].astype(str),
            seed=42,
            persist=False,
        )
        save_baseline_artifact(
            booster,
            family="xgb",
            freq=staged.freq,
            horizon=staged.horizon,
            metadata={"source": "tests.diagnosis.conftest"},
        )
        build_recovery_evidence(staged.evaluation_dataset_path)
        yield config_path
    finally:
        reset_runtime_config()
        os.environ.pop("BITBAT_CONFIG", None)
