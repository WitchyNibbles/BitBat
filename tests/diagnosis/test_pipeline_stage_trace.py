"""Pipeline stage trace tests for Phase 30 fix validation.

These tests confirm that the three bugs from ROOT_CAUSE.md ARE FIXED.
They skip when live artifacts (model file or autonomous.db) are absent.
They are the inverse of the Phase 29 diagnosis tests.
"""

import json
import sqlite3
from pathlib import Path

import pytest

xgb = pytest.importorskip("xgboost")

DB_PATH = Path("data/autonomous.db")
MODEL_PATH_5M = Path("models/5m_30m/xgb.json")


def test_model_objective_is_classification():
    """Bug 1 fixed (PRIMARY): Model trained with classification objective, not regression.

    This test confirms Bug 1 is fixed. The objective should be 'multi:softprob'
    after the Phase 30 fix (was 'reg:squarederror' before fix).
    """
    if not MODEL_PATH_5M.exists():
        pytest.skip("models/5m_30m/xgb.json not present — run pipeline first")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH_5M))
    cfg = json.loads(booster.save_config())
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "multi:softprob", (
        f"Expected classification objective 'multi:softprob' (Bug 1 fixed), "
        f"got: {objective}. "
        "If this fails, Bug 1 is still present — check model training config."
    )


def test_serving_direction_is_balanced():
    """Bug 2 fixed: Classification model outputs all three classes (up/down/flat).

    After fix, flat class must appear and 'down' should not dominate by 2x.
    This is the inverse of the Phase 29 direction-bias test.
    """
    if not DB_PATH.exists():
        pytest.skip("data/autonomous.db not present — run autonomous monitor first")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT predicted_direction, COUNT(*) FROM prediction_outcomes"
            " GROUP BY predicted_direction"
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
    assert not (
        down_count > up_count * 2
    ), f"Down bias should be eliminated after fix. down={down_count}, up={up_count}"


def test_validation_zero_return_eliminated():
    """Bug 3 fixed: actual_return = 0.0 corruption eliminated from price lookup.

    After fix, zero-return count should drop below 50 (was >= 100 before fix).
    This is the inverse of the Phase 29 zero-return-corruption test.
    """
    if not DB_PATH.exists():
        pytest.skip("data/autonomous.db not present — run autonomous monitor first")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        (zero_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
            " WHERE actual_return = 0.0 AND actual_return IS NOT NULL"
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
    if not DB_PATH.exists():
        pytest.skip("data/autonomous.db not present — run autonomous monitor first")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        (correct_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
            " WHERE correct = 1 AND actual_return IS NOT NULL"
        ).fetchone()
        (total_count,) = conn.execute(
            "SELECT COUNT(*) FROM prediction_outcomes" " WHERE actual_return IS NOT NULL"
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
