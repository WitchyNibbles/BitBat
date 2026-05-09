"""Mode-specific training profiles for autonomous trading presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bitbat.common.presets import Preset, get_preset

BaselineFamily = Literal["xgb", "random_forest"]
TargetKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class ModeModelProfile:
    """Training contract for a user-facing preset."""

    preset_name: str
    freq: str
    horizon: str
    family: BaselineFamily
    label_mode: str
    target_kind: TargetKind
    candidate_families: tuple[BaselineFamily, ...] = ()
    class_labels: tuple[str, ...] = ()
    barrier_take_profit: float | None = None
    barrier_stop_loss: float | None = None

    def artifact_metadata(self, preset: Preset) -> dict[str, object]:
        """Return stable artifact metadata for the trained preset model."""
        payload: dict[str, object] = {
            "preset_name": self.preset_name,
            "freq": self.freq,
            "horizon": self.horizon,
            "family": self.family,
            "label_mode": self.label_mode,
            "target_kind": self.target_kind,
            "tau": float(preset.tau),
            "enter_threshold": float(preset.enter_threshold),
            "candidate_families": list(self.candidate_families or (self.family,)),
            "action_policy": {
                "min_confidence": float(preset.enter_threshold),
                "min_expected_value_return": 0.0,
            },
        }
        if self.class_labels:
            payload["class_labels"] = list(self.class_labels)
        if self.barrier_take_profit is not None:
            payload["barrier_take_profit"] = float(self.barrier_take_profit)
        if self.barrier_stop_loss is not None:
            payload["barrier_stop_loss"] = float(self.barrier_stop_loss)
        return payload


MODE_MODEL_PROFILES: dict[str, ModeModelProfile] = {
    "scalper": ModeModelProfile(
        preset_name="scalper",
        freq="5m",
        horizon="30m",
        family="xgb",
        label_mode="triple_barrier",
        target_kind="classification",
        candidate_families=("xgb", "random_forest"),
        class_labels=("take_profit", "stop_loss", "timeout"),
        barrier_take_profit=0.0045,
        barrier_stop_loss=0.0030,
    ),
    "conservative": ModeModelProfile(
        preset_name="conservative",
        freq="1h",
        horizon="24h",
        family="random_forest",
        label_mode="return_direction",
        target_kind="regression",
        candidate_families=("random_forest", "xgb"),
    ),
    "balanced": ModeModelProfile(
        preset_name="balanced",
        freq="1h",
        horizon="4h",
        family="xgb",
        label_mode="return_direction",
        target_kind="classification",
        candidate_families=("xgb", "random_forest"),
        class_labels=("up", "down", "flat"),
    ),
    "aggressive": ModeModelProfile(
        preset_name="aggressive",
        freq="1h",
        horizon="1h",
        family="xgb",
        label_mode="return_direction",
        target_kind="classification",
        candidate_families=("xgb", "random_forest"),
        class_labels=("up", "down", "flat"),
    ),
    "swing": ModeModelProfile(
        preset_name="swing",
        freq="15m",
        horizon="1h",
        family="xgb",
        label_mode="triple_barrier",
        target_kind="classification",
        candidate_families=("xgb", "random_forest"),
        class_labels=("take_profit", "stop_loss", "timeout"),
        barrier_take_profit=0.0105,
        barrier_stop_loss=0.0070,
    ),
}


def get_mode_model_profile(preset_name: str) -> ModeModelProfile:
    """Return the training profile for a preset, defaulting to balanced."""
    resolved_name = str(preset_name).strip().lower()
    return MODE_MODEL_PROFILES.get(resolved_name, MODE_MODEL_PROFILES["balanced"])


def get_mode_model_profile_for_pair(freq: str, horizon: str) -> ModeModelProfile | None:
    """Resolve the preset profile associated with a freq/horizon runtime pair."""
    resolved_freq = str(freq).strip().lower()
    resolved_horizon = str(horizon).strip().lower()
    for profile in MODE_MODEL_PROFILES.values():
        if profile.freq == resolved_freq and profile.horizon == resolved_horizon:
            return profile
    return None


def get_profile_preset(preset_name: str) -> Preset:
    """Return the user-facing preset for a training profile."""
    return get_preset(preset_name)
