"""Candidate manifest helpers for promotable model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bitbat.config.loader import resolve_metrics_dir


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_candidate_manifest(
    *,
    candidate_id: str,
    freq: str,
    horizon: str,
    dataset_meta: dict[str, Any],
    candidate_report: dict[str, Any],
    artifact_bundle: dict[str, Any],
    champion_decision: dict[str, Any],
    cost_assumptions: dict[str, float],
) -> dict[str, Any]:
    winner_id = champion_decision.get("winner")
    winner = candidate_id == winner_id
    winner_gate = (
        champion_decision.get("promotion_gate", {})
        if winner
        else candidate_report.get("replay_gate", {})
    )
    return {
        "candidate_id": candidate_id,
        "pair": {
            "freq": freq,
            "horizon": horizon,
        },
        "dataset": _json_ready(dataset_meta),
        "artifacts": _json_ready(artifact_bundle),
        "candidate_report": _json_ready(candidate_report),
        "overfit_evidence": _json_ready(candidate_report.get("safeguards", {})),
        "replay_evidence": _json_ready(candidate_report.get("replay_summary", {})),
        "promotion_evidence": _json_ready(winner_gate),
        "champion_context": {
            "winner": winner,
            "winner_id": winner_id,
            "promote_candidate": bool(champion_decision.get("promote_candidate", False))
            if winner
            else False,
            "reason": (
                champion_decision.get("reason", "not-winner")
                if winner
                else candidate_report.get("replay_gate", {}).get("reasons", [])
            ),
        },
        "cost_assumptions": _json_ready(cost_assumptions),
    }


def write_candidate_manifests(
    *,
    freq: str,
    horizon: str,
    dataset_meta: dict[str, Any],
    candidate_reports: dict[str, dict[str, Any]],
    artifact_registry: dict[str, dict[str, Any]],
    champion_decision: dict[str, Any],
    cost_assumptions: dict[str, float],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    manifest_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_metrics_dir() / "candidate_manifests"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, str] = {}
    for candidate_id, candidate_report in sorted(candidate_reports.items()):
        artifact_bundle = artifact_registry.get(candidate_id, {})
        payload = build_candidate_manifest(
            candidate_id=candidate_id,
            freq=freq,
            horizon=horizon,
            dataset_meta=dataset_meta,
            candidate_report=candidate_report,
            artifact_bundle=artifact_bundle,
            champion_decision=champion_decision,
            cost_assumptions=cost_assumptions,
        )
        path = manifest_dir / f"{candidate_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest_paths[candidate_id] = str(path)
    return manifest_paths
