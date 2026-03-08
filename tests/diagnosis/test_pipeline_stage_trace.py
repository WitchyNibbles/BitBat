"""Pipeline stage trace tests for Phase 29 diagnosis.

These tests confirm that the known bugs ARE present. They are expected to
PASS now (confirming the diagnosis) and should FAIL after Phase 30 fixes
them, at which point they become regression markers.
"""

import json
import sqlite3
from pathlib import Path

import pytest

xgb = pytest.importorskip("xgboost")

DB_PATH = Path("data/autonomous.db")
MODEL_PATH_5M = Path("models/5m_30m/xgb.json")


def test_model_objective_is_regression():
    """Bug 1 (PRIMARY): Model trained with regression objective, not classification.

    This test confirms Bug 1 exists. After Phase 30 fix, this assertion should
    be inverted — objective should become 'multi:softprob'.
    """
    if not MODEL_PATH_5M.exists():
        pytest.skip("models/5m_30m/xgb.json not present — run pipeline first")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH_5M))
    cfg = json.loads(booster.save_config())
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "reg:squarederror", (
        f"Expected regression objective (confirming bug 1), got: {objective}. "
        "If this fails, Bug 1 has been fixed — update this test for Phase 30."
    )


def test_serving_direction_bias():
    """Bug 2: Regression model output mapped to binary up/down with no tau threshold.

    Negative predicted returns dominate, causing 'down' direction bias >= 2x.
    This test confirms the bias exists. After Phase 30 fix, direction counts
    should be more balanced (this assertion should be inverted).
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

    assert down_count > up_count * 2, (
        f"Expected 'down' bias >= 2x 'up' (confirming regression model bias), "
        f"got down={down_count}, up={up_count}. "
        "If this fails, Bug 2 has been fixed — update this test for Phase 30."
    )


def test_validation_zero_return_corruption():
    """Bug 3: actual_return = 0.0 corruption from price lookup failures.

    Expected ~179 of 266 rows have actual_return = 0.0 exactly. This test
    confirms the corruption exists. After Phase 30 fix, zero-return count
    should drop below 100 (this assertion should be inverted).
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

    assert zero_count >= 100, (
        f"Expected >= 100 zero-return rows (confirming price lookup corruption), "
        f"got: {zero_count}. "
        "If this fails, Bug 3 has been fixed — update this test for Phase 30."
    )


def test_accuracy_below_random_baseline():
    """Accuracy collapse: model correct < 33% of realized predictions.

    38/266 correct = 14.3% in live DB (well below 33% random baseline).
    This test confirms accuracy collapse exists. After Phase 30 fix, accuracy
    should exceed 0.33 (this assertion should be inverted).
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
    assert accuracy < 0.33, (
        f"Expected accuracy < 0.33 (confirming accuracy collapse), "
        f"got: {accuracy:.3f} ({correct_count}/{total_count}). "
        "If this fails, accuracy has recovered — update this test for Phase 30."
    )
