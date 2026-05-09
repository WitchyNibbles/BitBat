#!/usr/bin/env python
"""Bootstrap monitor model artifacts for the configured runtime pair."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Allow direct script execution without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitbat.config.loader import load_config, resolve_models_dir
from bitbat.model.mode_profiles import get_mode_model_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _default_start_date() -> str:
    return (datetime.now(UTC) - timedelta(days=120)).date().isoformat()


def _python_env(*, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["BITBAT_CONFIG"] = str(config_path)

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC_ROOT)
    return env


def _run_bitbat(
    args: list[str],
    *,
    config_path: Path,
    root_dir: Path,
    runner: Runner,
) -> None:
    cmd = [sys.executable, "-m", "bitbat.cli", *args]
    result = runner(
        cmd,
        cwd=root_dir,
        env=_python_env(config_path=config_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{details}".strip())


def _resolve_preset(config: dict[str, object]) -> str:
    resolved = str(config.get("preset", "balanced")).strip().lower()
    return resolved or "balanced"


def _candidate_artifacts(*, config: dict[str, object], root_dir: Path) -> list[Path]:
    preset_name = _resolve_preset(config)
    profile = get_mode_model_profile(preset_name)
    freq = str(config.get("freq", profile.freq))
    horizon = str(config.get("horizon", profile.horizon))
    pair_dir = root_dir / resolve_models_dir(config) / f"{freq}_{horizon}"
    candidates: list[Path] = []
    for family in profile.candidate_families or (profile.family,):
        if family == "xgb":
            candidates.append(pair_dir / "xgb.json")
        elif family == "random_forest":
            candidates.append(pair_dir / "random_forest.pkl")
    return candidates


def bootstrap_runtime_model(
    *,
    config_path: Path,
    start_date: str | None,
    symbol: str,
    root_dir: Path = REPO_ROOT,
    runner: Runner = subprocess.run,
) -> Path:
    cfg = load_config(config_path)
    freq = str(cfg.get("freq", "5m"))
    horizon = str(cfg.get("horizon", "30m"))
    start = start_date or _default_start_date()

    _run_bitbat(
        [
            "prices",
            "pull",
            "--symbol",
            symbol,
            "--interval",
            freq,
            "--start",
            start,
        ],
        config_path=config_path,
        root_dir=root_dir,
        runner=runner,
    )
    _run_bitbat(
        ["features", "build"],
        config_path=config_path,
        root_dir=root_dir,
        runner=runner,
    )
    _run_bitbat(
        ["model", "train", "--freq", freq, "--horizon", horizon],
        config_path=config_path,
        root_dir=root_dir,
        runner=runner,
    )

    artifacts = _candidate_artifacts(config=cfg, root_dir=root_dir)
    artifact = next((path for path in artifacts if path.exists()), None)
    if artifact is None:
        raise FileNotFoundError(
            "Bootstrap completed, but no compatible model artifact was created for "
            f"{freq}_{horizon}: {', '.join(str(path.relative_to(root_dir)) for path in artifacts)}"
        )
    return artifact


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build price/features/model artifacts for the runtime monitor pair."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to runtime config YAML (must include freq/horizon).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date for price backfill (YYYY-MM-DD). Defaults to 120 days ago.",
    )
    parser.add_argument(
        "--symbol",
        default="BTC-USD",
        help="Price symbol for bootstrap ingestion.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()

    artifact = bootstrap_runtime_model(
        config_path=config_path,
        start_date=args.start,
        symbol=args.symbol,
    )
    print(f"Bootstrap complete: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
