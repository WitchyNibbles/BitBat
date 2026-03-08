"""Round-trip consistency tests for cv_summary.json write/read contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database
from bitbat.autonomous.retrainer import AutoRetrainer

pytestmark = pytest.mark.behavioral

_METRIC_KEY = "mean_directional_accuracy"


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'roundtrip.db'}"


def _make_cv_summary(mean_da: float) -> dict:
    """Build a cv_summary.json payload matching the structure cli.py writes."""
    return {
        "primary_family": "xgboost",
        "selected_families": ["xgboost"],
        "family_metrics": {},
        "folds": 3,
        "average_rmse": 0.05,
        "average_mae": 0.03,
        _METRIC_KEY: mean_da,
        "candidate_reports": {
            "xgboost": {
                "metrics": {
                    "directional": {
                        "mean_directional_accuracy": mean_da,
                    }
                }
            }
        },
        "champion_decision": {
            "winner": "xgboost",
            "promote_candidate": True,
        },
    }


def test_cv_summary_roundtrip_consistency(tmp_path: Path) -> None:
    """Write cv_summary.json, then read via AutoRetrainer._read_cv_score() -- must match."""
    expected_score = 0.7321

    summary_dir = tmp_path / "metrics"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "cv_summary.json"
    summary_path.write_text(
        json.dumps(_make_cv_summary(expected_score), indent=2),
        encoding="utf-8",
    )

    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    retrainer = AutoRetrainer(db, "1h", "4h")

    # Point the retrainer at the tmp cv_summary.json
    retrainer._cv_summary_path = lambda: summary_path  # type: ignore[assignment]

    actual_score = retrainer._read_cv_score()
    assert actual_score == pytest.approx(
        expected_score
    ), f"Round-trip mismatch: wrote {expected_score}, read {actual_score}"


def test_cv_summary_key_names_match_between_writer_and_reader() -> None:
    """The writer (cli.py) and reader (retrainer.py) must use the same metric key name."""
    import bitbat.cli as cli_module

    cli_source = inspect.getsource(cli_module)
    retrainer_source = inspect.getsource(AutoRetrainer._read_cv_score)

    # The writer should write "mean_directional_accuracy" as the top-level key
    assert f'"{_METRIC_KEY}"' in cli_source, (
        f'cli.py does not contain key "{_METRIC_KEY}" -- '
        "the cv_summary.json writer is using a different key name"
    )

    # The reader should look up the same key
    assert f'"{_METRIC_KEY}"' in retrainer_source, (
        f'retrainer.py _read_cv_score does not look up "{_METRIC_KEY}" -- '
        "the reader is using a different key name than the writer"
    )

    # The old confusing alias should NOT be in the writer
    old_key = "average_balanced_accuracy"
    # Check specifically that the old key is not used as a dict key in the aggregate dict
    # (it may still appear in comments or elsewhere, so we check the cv_summary writing section)
    cv_summary_writer_lines = []
    in_aggregate = False
    for line in cli_source.splitlines():
        if "aggregate" in line and "{" in line:
            in_aggregate = True
        if in_aggregate:
            cv_summary_writer_lines.append(line)
            if "cv_summary.json" in line:
                break

    aggregate_block = "\n".join(cv_summary_writer_lines)
    assert (
        f'"{old_key}"' not in aggregate_block
    ), f'cli.py still writes the old key "{old_key}" in the cv_summary aggregate dict'
