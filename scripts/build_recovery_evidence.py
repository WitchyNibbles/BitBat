"""Stage and realize sandboxed recovery evidence for Phase 36."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bitbat.autonomous.recovery_evidence import build_recovery_evidence, stage_recovery_dataset
from bitbat.config.loader import set_runtime_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="Write sandbox train/eval recovery datasets.")
    stage.add_argument("--config", type=Path, required=True, help="Sandbox config path.")
    stage.add_argument(
        "--source-dataset",
        type=Path,
        required=True,
        help="Source dataset used to create train/eval recovery splits.",
    )
    stage.add_argument(
        "--evaluation-rows",
        type=int,
        default=300,
        help="How many trailing rows to reserve for evaluation evidence.",
    )
    stage.add_argument("--freq", type=str, default=None, help="Override runtime freq.")
    stage.add_argument("--horizon", type=str, default=None, help="Override runtime horizon.")

    realize = subparsers.add_parser("realize", help="Create fresh realized recovery evidence.")
    realize.add_argument("--config", type=Path, required=True, help="Sandbox config path.")
    realize.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=None,
        help="Optional explicit evaluation dataset path.",
    )
    realize.add_argument("--freq", type=str, default=None, help="Override runtime freq.")
    realize.add_argument("--horizon", type=str, default=None, help="Override runtime horizon.")

    return parser


def main() -> None:
    args = _parser().parse_args()
    set_runtime_config(args.config)

    if args.command == "stage":
        staged = stage_recovery_dataset(
            args.source_dataset,
            evaluation_rows=args.evaluation_rows,
            freq=args.freq,
            horizon=args.horizon,
        )
        print(json.dumps({
            "freq": staged.freq,
            "horizon": staged.horizon,
            "training_rows": staged.training_rows,
            "evaluation_rows": staged.evaluation_rows,
            "training_dataset_path": str(staged.training_dataset_path),
            "evaluation_dataset_path": str(staged.evaluation_dataset_path),
        }, indent=2))
        return

    summary = build_recovery_evidence(
        args.evaluation_dataset,
        freq=args.freq,
        horizon=args.horizon,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
