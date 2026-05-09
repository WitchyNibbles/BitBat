"""Model persistence helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal, TypeAlias

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from bitbat.config.loader import resolve_models_dir

BaselineFamily = Literal["xgb", "random_forest"]
LabelMode = Literal["direction", "triple_barrier", "meta_label"]
ArtifactRole = Literal["primary", "side", "action"]
TreeBaselineModel: TypeAlias = xgb.Booster | RandomForestRegressor


def normalize_label_mode(label_mode: str | None) -> LabelMode:
    """Normalize artifact label regimes to a small supported set."""
    resolved = str(label_mode or "direction").strip().lower()
    if resolved in {"direction", "return_direction"}:
        return "direction"
    if resolved == "triple_barrier":
        return "triple_barrier"
    if resolved in {"meta_label", "meta-label", "meta"}:
        return "meta_label"
    raise ValueError(
        "Unsupported label_mode "
        f"'{label_mode}'. Expected 'direction', 'triple_barrier', or 'meta_label'."
    )


def normalize_artifact_role(role: str | None) -> ArtifactRole:
    resolved = str(role or "primary").strip().lower()
    if resolved in {"", "primary"}:
        return "primary"
    if resolved == "side":
        return "side"
    if resolved == "action":
        return "action"
    raise ValueError(
        f"Unsupported artifact role '{role}'. Expected 'primary', 'side', or 'action'."
    )


def _family_filename(
    family: BaselineFamily,
    *,
    label_mode: LabelMode = "direction",
    artifact_role: ArtifactRole = "primary",
) -> str:
    suffix = ""
    if artifact_role != "primary":
        suffix = f".{artifact_role}"
    if family == "xgb":
        if label_mode == "direction":
            return f"xgb{suffix}.json"
        return f"xgb{suffix}.{label_mode}.json"
    if label_mode == "direction":
        return f"random_forest{suffix}.pkl"
    return f"random_forest{suffix}.{label_mode}.pkl"


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load artifact metadata when available, otherwise infer a minimal payload."""
    target = Path(path)
    metadata_path = _metadata_path(target)
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload.setdefault("family", _infer_family(target))
        payload.setdefault("artifact", target.name)
        payload.setdefault("path", str(target))
        return payload
    try:
        family = _infer_family(target)
    except ValueError:
        return {}
    return {
        "family": family,
        "artifact": target.name,
        "path": str(target),
    }


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
    label_mode: str | None = "direction",
    artifact_role: str | None = "primary",
    root: str | Path | None = None,
) -> Path:
    """Return the canonical artifact path for a baseline family."""
    resolved_root = Path(root) if root is not None else resolve_models_dir()
    resolved_label_mode = normalize_label_mode(label_mode)
    return (
        resolved_root
        / f"{freq}_{horizon}"
        / _family_filename(
            family,
            label_mode=resolved_label_mode,
            artifact_role=normalize_artifact_role(artifact_role),
        )
    )


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
    expected_label_mode: str | None = None,
) -> TreeBaselineModel:
    """Load a baseline model artifact from disk."""
    target = Path(path)
    resolved_family = _infer_family(target, family)
    metadata = load_metadata(target)
    if expected_label_mode is not None and metadata:
        resolved_expected = normalize_label_mode(expected_label_mode)
        artifact_label_mode = normalize_label_mode(str(metadata.get("label_mode", "direction")))
        if artifact_label_mode != resolved_expected:
            raise ValueError(
                "Artifact label_mode mismatch: "
                f"expected '{resolved_expected}' but found '{artifact_label_mode}' at {target}"
            )

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
    label_mode: str | None = "direction",
    artifact_role: str | None = "primary",
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a baseline artifact using stable family-aware paths."""
    resolved_label_mode = normalize_label_mode(label_mode)
    target = default_model_artifact_path(
        freq,
        horizon,
        family=family,
        label_mode=resolved_label_mode,
        artifact_role=artifact_role,
        root=root,
    )
    payload = {
        "freq": freq,
        "horizon": horizon,
        "family": family,
        "label_mode": resolved_label_mode,
        "artifact_role": normalize_artifact_role(artifact_role),
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
    label_mode: str | None = "direction",
    artifact_role: str | None = "primary",
    root: str | Path | None = None,
) -> TreeBaselineModel:
    """Load a baseline artifact using stable family-aware paths."""
    resolved_label_mode = normalize_label_mode(label_mode)
    target = default_model_artifact_path(
        freq,
        horizon,
        family=family,
        label_mode=resolved_label_mode,
        artifact_role=artifact_role,
        root=root,
    )
    return load(target, family=family, expected_label_mode=resolved_label_mode)
