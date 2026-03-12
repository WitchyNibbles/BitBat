"""Pipeline stage trace tests for Phase 30 fix validation.

These tests confirm that the three bugs from ROOT_CAUSE.md ARE FIXED.
They skip when live artifacts (model file or autonomous.db) are absent.
They are the inverse of the Phase 29 diagnosis tests.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from bitbat.config.loader import get_runtime_config, resolve_models_dir

xgb = pytest.importorskip("xgboost")


def _runtime_pair() -> tuple[str, str]:
    cfg = get_runtime_config()
    return str(cfg.get("freq", "5m")), str(cfg.get("horizon", "30m"))


def _db_path() -> Path:
    cfg = get_runtime_config()
    db_url = str(cfg.get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db"))
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        pytest.skip(f"Diagnosis tests only support sqlite URLs, got: {db_url}")
    return Path(db_url.removeprefix(prefix))


def _model_path() -> Path:
    freq, horizon = _runtime_pair()
    return resolve_models_dir() / f"{freq}_{horizon}" / "xgb.json"


def test_model_objective_is_classification():
    """Bug 1 fixed (PRIMARY): Model trained with classification objective, not regression.

    This test confirms Bug 1 is fixed. The objective should be 'multi:softprob'
    after the Phase 30 fix (was 'reg:squarederror' before fix).
    """
    model_path = _model_path()
    if not model_path.exists():
        pytest.skip(f"{model_path} not present — run pipeline first")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    cfg = json.loads(booster.save_config())
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "multi:softprob", (
        f"Expected classification objective 'multi:softprob' (Bug 1 fixed), "
        f"got: {objective}. "
        "If this fails, Bug 1 is still present — check model training config."
    )


def test_serving_direction_is_balanced():
    """Bug 2 fixed: Classification model outputs all three classes (up/down/flat).

    After fix, flat class must appear and predictions should no longer collapse into
    an overwhelmingly down-only stream.
    This is the inverse of the Phase 29 direction-bias test.
    """
    freq, horizon = _runtime_pair()
    db_path = _db_path()
    if not db_path.exists():
        pytest.skip(f"{db_path} not present — run autonomous monitor first")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT predicted_direction, COUNT(*) FROM prediction_outcomes"
            " WHERE freq = ? AND horizon = ?"
            " GROUP BY predicted_direction",
            (freq, horizon),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        pytest.skip("No prediction_outcomes rows in autonomous.db")

    counts = {row[0]: row[1] for row in rows}
    down_count = counts.get("down", 0)
    up_count = counts.get("up", 0)
    flat_count = counts.get("flat", 0)
    assert flat_count > 0, "Flat class must appear in predictions after Bug 2 fix"
    non_down_count = up_count + flat_count
    assert not (
        down_count > non_down_count * 2
    ), (
        "Down-only collapse should be eliminated after fix. "
        f"down={down_count}, up={up_count}, flat={flat_count}"
    )


def test_validation_zero_return_eliminated():
    """Bug 3 fixed: actual_return = 0.0 corruption eliminated from price lookup.

    After fix, zero-return count should drop below 50 (was >= 100 before fix).
    This is the inverse of the Phase 29 zero-return-corruption test.
    """
    freq, horizon = _runtime_pair()
    db_path = _db_path()
    if not db_path.exists():
        pytest.skip(f"{db_path} not present — run autonomous monitor first")
    conn = sqlite3.connect(str(db_path))
    try:
        (zero_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
            " WHERE actual_return = 0.0 AND actual_return IS NOT NULL"
            " AND freq = ? AND horizon = ?",
            (freq, horizon),
        ).fetchone()
    finally:
        conn.close()

    assert zero_count < 50, (
        f"Expected < 50 zero-return rows after Bug 3 fix (price lookup fixed), "
        f"got: {zero_count}. "
        "If this fails, Bug 3 is still present — check validator tau and price lookup."
    )


def test_accuracy_exceeds_random_baseline():
    """Accuracy recovered: model correct > 33% of realized predictions.

    After fix, accuracy should exceed 0.33 (was 14.3% = 38/266 before fix).
    This is the inverse of the Phase 29 accuracy-collapse test.
    """
    freq, horizon = _runtime_pair()
    db_path = _db_path()
    if not db_path.exists():
        pytest.skip(f"{db_path} not present — run autonomous monitor first")
    conn = sqlite3.connect(str(db_path))
    try:
        (correct_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
            " WHERE correct = 1 AND actual_return IS NOT NULL"
            " AND freq = ? AND horizon = ?",
            (freq, horizon),
        ).fetchone()
        (total_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
            " WHERE actual_return IS NOT NULL AND freq = ? AND horizon = ?",
            (freq, horizon),
        ).fetchone()
    finally:
        conn.close()

    if total_count < 50:
        pytest.skip(f"Fewer than 50 realized rows ({total_count}) — insufficient sample")

    accuracy = correct_count / total_count
    assert accuracy > 0.33, (
        f"Expected accuracy > 0.33 after accuracy recovery, "
        f"got: {accuracy:.3f} ({correct_count}/{total_count}). "
        "If this fails, accuracy has not recovered — check model objective and inference."
    )
