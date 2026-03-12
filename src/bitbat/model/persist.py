"""Model persistence helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from bitbat.config.loader import resolve_models_dir

BaselineFamily = Literal["xgb", "random_forest"]
TreeBaselineModel = xgb.Booster | RandomForestRegressor


def _family_filename(family: BaselineFamily) -> str:
    return "xgb.json" if family == "xgb" else "random_forest.pkl"


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _infer_family(path: Path, family: BaselineFamily | None = None) -> BaselineFamily:
    if family is not None:
        return family
    if path.suffix == ".json":
        return "xgb"
    if path.suffix == ".pkl":
        return "random_forest"
    metadata_path = _metadata_path(path)
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = str(payload.get("family", "")).strip().lower()
        if value in {"xgb", "random_forest"}:
            return value  # type: ignore[return-value]
    raise ValueError(f"Unable to infer model family for artifact: {path}")


def default_model_artifact_path(
    freq: str,
    horizon: str,
    *,
    family: BaselineFamily,
    root: str | Path | None = None,
) -> Path:
    """Return the canonical artifact path for a baseline family."""
    resolved_root = Path(root) if root is not None else resolve_models_dir()
    return resolved_root / f"{freq}_{horizon}" / _family_filename(family)


def save(
    model: TreeBaselineModel,
    path: str | Path,
    *,
    family: BaselineFamily | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the trained model artifact to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    resolved_family = _infer_family(target, family)
    if resolved_family == "xgb":
        if not isinstance(model, xgb.Booster):
            raise TypeError("Expected xgboost.Booster for family='xgb'.")
        model.save_model(str(target))
    else:
        if not isinstance(model, RandomForestRegressor):
            raise TypeError("Expected RandomForestRegressor for family='random_forest'.")
        with target.open("wb") as artifact:
            pickle.dump(model, artifact)

    payload: dict[str, Any] = {
        "family": resolved_family,
        "artifact": target.name,
        "path": str(target),
    }
    if metadata:
        payload.update(metadata)
    _metadata_path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(
    path: str | Path,
    *,
    family: BaselineFamily | None = None,
) -> TreeBaselineModel:
    """Load a baseline model artifact from disk."""
    target = Path(path)
    resolved_family = _infer_family(target, family)

    if resolved_family == "xgb":
        booster = xgb.Booster()
        booster.load_model(str(target))
        return booster

    with target.open("rb") as artifact:
        model = pickle.load(artifact)  # noqa: S301
    if not isinstance(model, RandomForestRegressor):
        raise TypeError(f"Expected RandomForestRegressor artifact at {target}")
    return model


def save_baseline_artifact(
    model: TreeBaselineModel,
    *,
    family: BaselineFamily,
    freq: str,
    horizon: str,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a baseline artifact using stable family-aware paths."""
    target = default_model_artifact_path(freq, horizon, family=family, root=root)
    payload = {
        "freq": freq,
        "horizon": horizon,
        "family": family,
    }
    if metadata:
        payload.update(metadata)
    save(model, target, family=family, metadata=payload)
    return target


def load_baseline_artifact(
    freq: str,
    horizon: str,
    *,
    family: BaselineFamily,
    root: str | Path | None = None,
) -> TreeBaselineModel:
    """Load a baseline artifact using stable family-aware paths."""
    target = default_model_artifact_path(freq, horizon, family=family, root=root)
    return load(target, family=family)
