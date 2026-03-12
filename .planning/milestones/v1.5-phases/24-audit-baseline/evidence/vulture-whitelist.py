# vulture whitelist -- curated false positives for bitbat codebase
# Generated: 2026-03-04
# Confidence threshold: 80%
# Scope: src/bitbat (target) + tests/ (usage context)
#
# Each entry below is a confirmed false positive that vulture flags
# due to limitations of static analysis. These are NOT dead code.

# --- Pytest fixture parameters (dependency injection) ---
# Vulture cannot trace pytest's runtime fixture injection. These
# parameters appear "unused" in test function signatures but are
# required by pytest to trigger fixture setup/teardown side effects.

db_with_data  # pytest fixture: seeds SQLite DB with sample data (tests/api/test_metrics.py)
incompatible_schema_db  # pytest fixture: seeds DB with wrong schema for negative testing (tests/api/test_metrics.py)
db_with_predictions  # pytest fixture: seeds DB with prediction records (tests/api/test_predictions.py)
model_on_disk  # pytest fixture: writes XGBoost model artifact to tmp_path (tests/api/test_predictions.py)

# --- Lambda stub parameters ---
# Used in monkeypatch stubs to absorb keyword arguments that the real
# function accepts but the stub ignores. Standard Python pattern.

kw  # lambda **kw absorbs kwargs in monkeypatch stub (tests/autonomous/test_orchestrator.py:30)
