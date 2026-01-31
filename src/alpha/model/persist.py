"""Model persistence helpers."""

from __future__ import annotations

from pathlib import Path

import xgboost as xgb


def save(model: xgb.Booster, path: str | Path) -> None:
    """Persist the trained model artifact to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(target))


def load(path: str | Path) -> xgb.Booster:
    """Load a model artifact from disk."""
    target = Path(path)
    booster = xgb.Booster()
    booster.load_model(str(target))
    return booster
