---
phase: 24-audit-baseline
plan: 01
subsystem: testing
tags: [pytest, markers, test-classification, coverage-gaps]

# Dependency graph
requires: []
provides:
  - Pytest marker registration (behavioral, integration, structural) in pyproject.toml
  - All 76 test files classified with markers (603 tests total)
  - Pure source-reader milestone-marker test deleted (5 tests removed)
  - Coverage gap matrix mapping 14 v1.5 requirements to test status
  - evidence/test-classification.md with full classification report
affects: [25-critical-correctness, 26-architecture, 27-ci-gates]

# Tech tracking
tech-stack:
  added: []
  patterns: [pytestmark-module-level-markers, behavioral-integration-structural-test-taxonomy]

key-files:
  created:
    - .planning/phases/24-audit-baseline/evidence/test-classification.md
  modified:
    - pyproject.toml
    - Makefile
    - 76 test files under tests/

key-decisions:
  - "Module-level pytestmark used for all files (no per-function markers) for consistency and simplicity"
  - "test_phase19_d1_monitor_alignment_complete.py deleted as pure source-reader (5 tests, all Path.read_text assertions)"
  - "16 *_complete.py files retained and reclassified because they exercise real production code"
  - "All 14 v1.5 requirements confirmed as coverage gaps (expected: requirements were defined to target known issues)"

patterns-established:
  - "pytestmark = pytest.mark.<type> at module level after imports for all test files"
  - "Three-way test taxonomy: behavioral (unit, isolated), integration (multi-component, real I/O), structural (source/config validation)"

requirements-completed: [AUDT-01]

# Metrics
duration: 9min
completed: 2026-03-04
---

# Phase 24 Plan 01: Test Classification Summary

**Classified 76 test files (603 tests) into behavioral/integration/structural markers, deleted 1 pure source-reader, and produced coverage gap matrix showing all 14 v1.5 requirements lack test coverage**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-04T13:12:32Z
- **Completed:** 2026-03-04T13:21:58Z
- **Tasks:** 1
- **Files modified:** 80 (76 test files + pyproject.toml + Makefile + 1 deleted + 1 evidence created)

## Accomplishments

- Registered behavioral, integration, and structural pytest markers in pyproject.toml
- Classified all 76 test files: 32 behavioral (269 tests), 37 integration (308 tests), 7 structural (26 tests)
- Deleted 1 pure source-reader milestone-marker test (5 tests removed, 608 -> 603)
- Reclassified 16 *_complete.py files that exercise real production code (kept, not deleted)
- Created comprehensive evidence/test-classification.md with deletion log, reclassification log, full classification, and coverage gap matrix
- Verified pytest collection: 0 warnings, all 603 tests have exactly one marker, marker-filtered collection totals match

## Task Commits

Each task was committed atomically:

1. **Task 1: Register pytest markers and classify all test files** - `6f0dedc` (feat)

**Plan metadata:** (pending final commit)

## Files Created/Modified

- `pyproject.toml` - Added behavioral/integration/structural marker registration
- `Makefile` - Removed reference to deleted test file from test-release target
- `tests/autonomous/test_phase19_d1_monitor_alignment_complete.py` - DELETED (pure source-reader)
- `tests/autonomous/test_phase8_d1_monitor_schema_complete.py` - Removed deleted file from D1_CANONICAL_SUITE, added integration marker
- `tests/gui/test_phase8_release_verification_complete.py` - Removed deleted file from REQUIRED_GATE_FILES and assertions, added structural marker
- 74 other test files - Added pytestmark with appropriate marker
- `.planning/phases/24-audit-baseline/evidence/test-classification.md` - Full classification report with coverage gap matrix

## Decisions Made

- **Module-level pytestmark for all files:** Used `pytestmark = pytest.mark.<type>` at module level rather than per-function decorators. Simpler, consistent, and correctly propagates to all functions in the file.
- **16 *_complete.py files kept:** All milestone-completion test files that import from `bitbat.*` and exercise production code were reclassified (mostly as integration), not deleted. Only the 1 pure source-reader file was deleted.
- **Coverage gap matrix confirms all 14 requirements are gaps:** This is expected and validates the v1.5 requirements were well-targeted at real missing coverage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed broken marker insertions from previous agent**
- **Found during:** Task 1 (initial verification of existing work)
- **Issue:** A previous execution attempt had inserted `pytestmark` lines inside multi-line import statements, try/except blocks, and function bodies, breaking syntax in 26 of 77 test files
- **Fix:** Restored all test files to HEAD, then re-applied markers correctly using AST-guided insertion positioning
- **Files modified:** All 76 test files (clean re-application)
- **Verification:** `python -c "py_compile.compile(f, doraise=True)"` passed for all files; `poetry run pytest --co -q` collected 603 tests with 0 warnings
- **Committed in:** 6f0dedc

**2. [Rule 2 - Missing Critical] Removed references to deleted test from dependent files**
- **Found during:** Task 1 (after deleting pure source-reader)
- **Issue:** The deleted file `test_phase19_d1_monitor_alignment_complete.py` was referenced in `D1_CANONICAL_SUITE`, `REQUIRED_GATE_FILES`, assertion checks, and Makefile assertions in 2 other test files
- **Fix:** Removed all references to the deleted file from `test_phase8_d1_monitor_schema_complete.py` and `test_phase8_release_verification_complete.py`
- **Files modified:** `tests/autonomous/test_phase8_d1_monitor_schema_complete.py`, `tests/gui/test_phase8_release_verification_complete.py`
- **Verification:** Both files compile and all assertions reference only existing files
- **Committed in:** 6f0dedc

**3. [Rule 2 - Missing Critical] Added markers to 3 previously untouched gui test files**
- **Found during:** Task 1 (comprehensive coverage check)
- **Issue:** `test_phase11_runtime_stability_complete.py`, `test_phase7_streamlit_compat_complete.py`, and `test_phase8_d2_timeline_complete.py` were not in the original plan's file list but needed markers for complete coverage
- **Fix:** Added appropriate pytestmark (integration) to all 3 files
- **Files modified:** 3 gui test files
- **Committed in:** 6f0dedc

---

**Total deviations:** 3 auto-fixed (1 bug fix, 2 missing critical)
**Impact on plan:** All auto-fixes were necessary for correctness and completeness. No scope creep.

## Issues Encountered

- Previous agent's work was completely unusable due to syntax errors in 26 files. Required full reset and re-implementation from scratch.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Test classification is complete; marker-based test selection works (`-m behavioral`, `-m integration`, `-m structural`)
- Coverage gap matrix is ready for use by subsequent plans (AUDT-02 through AUDT-05) and remediation phases (25-27)
- No blockers identified for remaining audit baseline plans

---
*Phase: 24-audit-baseline*
*Completed: 2026-03-04*
