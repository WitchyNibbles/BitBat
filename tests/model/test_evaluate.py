from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if pytest.importorskip("matplotlib"):
    import matplotlib.pyplot as plt  # noqa: F401

from alpha.model.evaluate import classification_metrics


def test_classification_metrics_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(42)
    proba = rng.random((100, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = np.array(["down", "flat", "up"])
    y_true = pd.Series(labels[rng.integers(0, 3, size=100)])

    monkeypatch.chdir(tmp_path)
    metrics = classification_metrics(y_true, proba, threshold=0.0, class_labels=list(labels))

    assert "balanced_accuracy" in metrics
    assert "pr_curves" in metrics

    metrics_path = Path("metrics") / "classification_metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mcc" in data

    png_path = Path("metrics") / "confusion_matrix.png"
    assert png_path.exists()


def test_classification_metrics_warns_on_suspicious_jump(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, size=200).astype(str))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)

    # Baseline run
    baseline_proba = np.zeros((200, 2))
    baseline_proba[:, 1] = np.where(y_true == "1", 0.01, 0.99)
    baseline_proba[:, 0] = 1.0 - baseline_proba[:, 1]
    classification_metrics(y_true, baseline_proba, threshold=0.0, class_labels=["0", "1"])
    capsys.readouterr()

    # Second run with inflated PR-AUC but same class balance
    confident = np.zeros_like(baseline_proba)
    confident[:, 1] = np.where(y_true == "1", 0.99, 0.01)
    confident[:, 0] = 1.0 - confident[:, 1]

    classification_metrics(y_true, confident, threshold=0.0, class_labels=["0", "1"])
    out = capsys.readouterr().out
    assert "metrics-warning" in out
    monkeypatch.undo()
