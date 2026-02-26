---
phase: 17-runtime-pair-alignment-and-startup-guardrails
plan: "03"
subsystem: monitor-schema-compatibility
tags: [schema-compat, performance-snapshots, monitor-status, monitor-snapshots]
requires:
  - phase: 08-regression-gates-release-verification
    provides: Existing runtime schema compatibility framework and additive upgrade path
provides:
  - Runtime schema contract coverage for `performance_snapshots` monitor-required columns
  - Compatibility upgrade detection/remediation for legacy snapshot schemas
  - Regression coverage for schema remediation across DB classification and CLI monitor status paths
affects: [schema-compat, autonomous-db, monitor-cli, autonomous-tests, v1.3-phase17]
tech-stack:
  added: []
  patterns:
    - Runtime schema contract must include every ORM-mapped monitor-critical table/column
    - Schema remediation messages should be actionable (`--audit` then `--upgrade`) for status/snapshot failures
key-files:
  created: []
  modified:
    - src/bitbat/autonomous/schema_compat.py
    - src/bitbat/autonomous/db.py
    - tests/autonomous/test_schema_compat.py
    - tests/autonomous/test_db.py
key-decisions:
  - Added a dedicated `PERFORMANCE_SNAPSHOTS_CONTRACT` with additive classification for late-added columns.
  - Extended DB error classifier to run schema audit when snapshot-column failures are observed.
  - Locked compatibility expectations in tests for audit, idempotent upgrade, and operator remediation text.
patterns-established:
  - Schema compatibility coverage now spans both prediction ingestion and monitor status/snapshot persistence paths.
  - Legacy schema fixtures in tests must preserve baseline table creation before targeted downgrades.
requirements-completed: [SCHE-04]
duration: 19 min
completed: 2026-02-26
---

# Phase 17 Plan 03: Performance Snapshot Schema Contract Summary

**Runtime schema compatibility now protects monitor status/snapshot flows from legacy `performance_snapshots` drift.**

## Performance

- **Duration:** 19 min
- **Completed:** 2026-02-26T12:56:58Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `performance_snapshots` columns to runtime compatibility contract with additive migration semantics.
- Extended monitor DB error classification to treat snapshot schema issues as actionable compatibility failures.
- Added schema tests covering detection and idempotent upgrade of legacy snapshot columns.

## Task Commits

Implementation was committed as a single atomic plan commit:

1. **Plan 17-03 implementation** - `40d7abc` (feat)

## Files Created/Modified

- `src/bitbat/autonomous/schema_compat.py` - added `PERFORMANCE_SNAPSHOTS_CONTRACT` and runtime contract wiring.
- `src/bitbat/autonomous/db.py` - extended schema-aware DB error classification for `performance_snapshots`.
- `tests/autonomous/test_schema_compat.py` - added legacy snapshot audit/upgrade tests and contract assertions.
- `tests/autonomous/test_db.py` - added snapshot schema remediation classification test.

## Decisions Made

- Treated core snapshot table columns as non-additive requirements and derived metrics columns as additive for legacy upgrades.
- Preserved deterministic audit output by keeping contract keys sorted and explicit.

## Deviations from Plan

- CLI schema messaging regression coverage was included in the broader Phase 17 monitor CLI update commit to avoid redundant file churn; schema contract and DB classifier changes remained isolated in this commit.

## Issues Encountered

- Existing schema tests assumed single-table contract ordering. Fixed by selecting tables by name instead of positional index.

## User Setup Required

None.

## Next Phase Readiness

- Monitor status/snapshot schema compatibility is now contract-enforced and test-locked.
- Heartbeat metadata work can proceed without risking hidden snapshot-table drift.

## Self-Check: PASSED

- `poetry run pytest tests/autonomous/test_schema_compat.py -q -k "performance_snapshots or required_contract"` -> 3 passed, 6 deselected
- `poetry run pytest tests/autonomous/test_schema_compat.py tests/autonomous/test_db.py -q -k "performance_snapshots or schema_remediation"` -> 3 passed, 11 deselected
- `poetry run pytest tests/test_cli.py -q -k "monitor and (status or snapshots) and schema"` -> 1 passed, 27 deselected

---
*Phase: 17-runtime-pair-alignment-and-startup-guardrails*
*Completed: 2026-02-26*
