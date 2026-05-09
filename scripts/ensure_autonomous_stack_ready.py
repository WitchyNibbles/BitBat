#!/usr/bin/env python
"""Ensure the autonomous stack has a trained runtime model before services start."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitbat.autonomous.orchestrator import one_click_train
from bitbat.config.loader import (
    get_runtime_config,
    reset_runtime_config,
    resolve_models_dir,
    set_runtime_config,
)
from bitbat.model.mode_profiles import get_mode_model_profile


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the active autonomous runtime model if it is missing."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path. Defaults to the active runtime config resolution.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Override the preset used for bootstrapping. Defaults to config preset or balanced.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if a compatible model artifact already exists.",
    )
    return parser.parse_args(argv)


def _candidate_artifacts(preset_name: str, *, config: dict[str, object]) -> list[Path]:
    profile = get_mode_model_profile(preset_name)
    pair_dir = resolve_models_dir(config) / f"{profile.freq}_{profile.horizon}"
    candidates: list[Path] = []
    for family in profile.candidate_families or (profile.family,):
        if family == "xgb":
            candidates.append(pair_dir / "xgb.json")
        elif family == "random_forest":
            candidates.append(pair_dir / "random_forest.pkl")
    return candidates


def _resolve_preset(config: dict[str, object], *, explicit_preset: str | None) -> str:
    resolved = explicit_preset or str(config.get("preset", "balanced"))
    return resolved.strip().lower() or "balanced"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reset_runtime_config()
    config = set_runtime_config(args.config)
    preset_name = _resolve_preset(config, explicit_preset=args.preset)
    artifacts = _candidate_artifacts(preset_name, config=config)

    if not args.force_train and any(path.exists() for path in artifacts):
        existing = next(path for path in artifacts if path.exists())
        print(
            "Autonomous bootstrap: existing artifact found for "
            f"preset '{preset_name}': {existing}"
        )
        return 0

    print(f"Autonomous bootstrap: training preset '{preset_name}'...")
    result = one_click_train(preset_name=preset_name)
    if result.get("status") != "success":
        print(f"Autonomous bootstrap failed: {result}", file=sys.stderr)
        return 1

    refreshed = get_runtime_config()
    final_artifacts = _candidate_artifacts(preset_name, config=refreshed)
    existing_after = next((path for path in final_artifacts if path.exists()), None)
    if existing_after is None:
        print(
            "Autonomous bootstrap finished without a runtime artifact in the expected model pair.",
            file=sys.stderr,
        )
        return 1

    print(
        "Autonomous bootstrap complete: "
        f"preset={preset_name} artifact={existing_after} "
        f"model_version={result.get('model_version')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
