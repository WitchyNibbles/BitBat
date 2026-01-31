"""Model evaluation routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    threshold: float = 0.6,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics from probabilities."""
    if proba.ndim != 2:
        raise ValueError(
            "Probability array must be 2-dimensional with shape (n_samples, n_classes)."
        )

    classes = class_labels or list(y_true.unique())
    classes = sorted(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    y_encoded = y_true.map(class_to_idx)
    if y_encoded.isna().any():
        raise ValueError("y_true contains labels not present in class_labels.")

    top_class = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    preds = np.where(confidence >= threshold, top_class, -1)

    valid_mask = preds != -1
    filtered_preds = preds[valid_mask]
    filtered_true = y_encoded.to_numpy()[valid_mask]

    if filtered_preds.size == 0:
        raise ValueError("All predictions below threshold; adjust threshold or check model output.")

    balanced_acc = balanced_accuracy_score(filtered_true, filtered_preds)
    mcc = matthews_corrcoef(filtered_true, filtered_preds)

    pr_curves = {}
    pr_aucs: list[float] = []
    for idx, label in enumerate(classes):
        precision, recall, _ = precision_recall_curve((y_encoded == idx).astype(int), proba[:, idx])
        if recall[0] > recall[-1]:
            precision = precision[::-1]
            recall = recall[::-1]
        pr_auc = np.trapezoid(precision, recall)
        pr_curves[label] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc": float(pr_auc),
        }
        pr_aucs.append(float(pr_auc))

    precision, recall, _, _ = precision_recall_fscore_support(
        filtered_true,
        filtered_preds,
        labels=list(range(len(classes))),
        zero_division=0,
    )

    class_counts = y_true.value_counts(normalize=True).to_dict()
    class_balance = {str(label): float(class_counts.get(label, 0.0)) for label in classes}
    average_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    metrics = {
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
            }
            for idx, label in enumerate(classes)
        },
        "pr_curves": pr_curves,
        "threshold": threshold,
        "average_pr_auc": average_pr_auc,
        "class_balance": class_balance,
    }

    cm = confusion_matrix(filtered_true, filtered_preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(int(value)), ha="center", va="center", color="black")

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "classification_metrics.json"

    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        prev_auc = previous.get("average_pr_auc")
        prev_balance = previous.get("class_balance") or {}
        if isinstance(prev_auc, int | float):
            auc_delta = average_pr_auc - float(prev_auc)
            if auc_delta > 0.10:
                balance_similar = True
                for label in classes:
                    prev_rate = float(prev_balance.get(str(label), 0.0))
                    new_rate = class_balance.get(str(label), 0.0)
                    if abs(prev_rate - new_rate) > 0.01:
                        balance_similar = False
                        break
                if balance_similar:
                    print(
                        "[metrics-warning] PR-AUC jumped by more than 10 points while "
                        "class balance remained stable. Double-check for leakage.",
                        flush=True,
                    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    fig_path = metrics_dir / "confusion_matrix.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return metrics


def evaluate_model(model: Any, validation_data: Any) -> dict[str, Any]:  # pragma: no cover - stub
    """Evaluate a trained model and return a metrics mapping."""
    raise NotImplementedError("evaluate_model is not implemented yet.")
