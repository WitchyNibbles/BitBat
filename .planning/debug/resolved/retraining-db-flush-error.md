---
status: resolved
trigger: "SQLAlchemy session.flush() fails during create_retraining_event() in the monitoring agent's retraining flow"
created: 2026-02-28T00:00:00Z
updated: 2026-02-28T00:10:00Z
---

## Current Focus

hypothesis: CONFIRMED - Old CHECK constraint on retraining_events table rejects 'continuous' trigger_reason
test: Reproduced by creating DB with old schema and inserting trigger_reason='continuous'
expecting: IntegrityError: CHECK constraint failed: ck_trigger_reason
next_action: Done - fix applied and verified

## Symptoms

expected: make test passes all tests; monitoring agent completes retraining cycle without errors
actual: SQLAlchemy flush error during create_retraining_event() call
errors: Traceback shows session.flush() failing in db.py:379, called from continuous_trainer.py:89, called from agent.py:290. Actual error: IntegrityError: CHECK constraint failed: ck_trigger_reason
reproduction: Run monitoring agent retraining cycle or run monitoring agent directly
started: Just noticed, tests not run recently

## Eliminated

## Evidence

- timestamp: 2026-02-28T00:01:00Z
  checked: RetrainingEvent ORM model CHECK constraint
  found: Model defines trigger_reason CHECK with 'continuous' included (added in commit 384ab0f)
  implication: ORM model is correct, but existing DB tables may have stale CHECK

- timestamp: 2026-02-28T00:02:00Z
  checked: Existing data/autonomous.db retraining_events schema
  found: CHECK constraint is old version WITHOUT 'continuous': IN ('drift_detected', 'scheduled', 'manual', 'poor_performance')
  implication: DB was created before 384ab0f and create_all() does not update existing CHECK constraints

- timestamp: 2026-02-28T00:03:00Z
  checked: Reproduction test with old-schema DB
  found: IntegrityError: (sqlite3.IntegrityError) CHECK constraint failed: ck_trigger_reason when inserting trigger_reason='continuous'
  implication: Root cause confirmed - stale CHECK constraint rejects new enum value

- timestamp: 2026-02-28T00:04:00Z
  checked: Pre-existing test failure (test_init_script_upgrade_is_repeat_safe_and_reports_status)
  found: This test was already failing before our changes due to upgrade_schema_compatibility trying ALTER TABLE on non-existent tables
  implication: Pre-existing bug in upgrade flow when DB has only some tables

## Resolution

root_cause: The retraining_events table in existing DBs has a stale CHECK constraint that does not include 'continuous' as a valid trigger_reason. The ORM model was updated in commit 384ab0f to add 'continuous', but SQLite's CREATE TABLE IF NOT EXISTS (from create_all()) does not modify existing table constraints. When create_retraining_event(trigger_reason='continuous') is called, SQLite raises IntegrityError: CHECK constraint failed: ck_trigger_reason.

fix: |
  Three changes:
  1. schema_compat.py: Added CHECK_CONSTRAINT_CONTRACT defining required enum values, plus
     _get_check_constraint_sql(), _check_constraints_are_current(), and
     _rebuild_table_with_current_schema() to detect stale CHECK constraints and rebuild
     the table using SQLite's rename-create-copy-drop pattern. Integrated into
     upgrade_schema_compatibility(). Also fixed pre-existing bug where ALTER TABLE was
     attempted on non-existent tables by checking existing_tables before upgrading.
  2. scripts/init_autonomous_db.py: Fixed --upgrade flow to create missing tables
     (not just when NO tables exist, but when some are missing).
  3. tests/autonomous/test_schema_compat.py: Added regression test that creates a DB
     with old CHECK constraint, verifies AutonomousDB auto-upgrade rebuilds the table,
     and confirms trigger_reason='continuous' inserts succeed while preserving old data.

verification: |
  - Reproduced original error: IntegrityError on INSERT with trigger_reason='continuous' against old-schema DB
  - After fix: AutonomousDB auto-upgrade detects stale CHECK, rebuilds table, INSERT succeeds
  - Old data preserved across rebuild (verified in regression test)
  - Idempotent: no-op when constraints already current
  - All 85 autonomous tests pass (including new regression test)
  - Full suite: 595 tests pass, 0 failures
  - Existing data/autonomous.db upgraded on disk

files_changed:
  - src/bitbat/autonomous/schema_compat.py
  - scripts/init_autonomous_db.py
  - tests/autonomous/test_schema_compat.py
