---
phase: 20-api-config-alignment
plan: "01"
subsystem: api
tags: [fastapi, yaml-config, settings, bucket-validation, pydantic]

# Dependency graph
requires: []
provides:
  - "GET /system/settings with default.yaml fallback (freq=5m, horizon=30m)"
  - "PUT /system/settings with bucket.py frequency validation"
  - "SettingsResponse with valid_freqs and valid_horizons lists"
  - "6 regression tests for APIC-01 and APIC-02"
affects: [21-settings-ui-expansion, 22-sub-hourly-presets, 23-configuration-test-coverage]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Settings fallback chain: user_config.yaml -> default.yaml (not presets)"
    - "Frequency validation against bucket.py _SUPPORTED_FREQUENCIES canonical set"
    - "Response includes valid option lists for downstream UI consumption"

key-files:
  created:
    - tests/api/test_settings.py
  modified:
    - src/bitbat/api/schemas.py
    - src/bitbat/api/routes/system.py

key-decisions:
  - "Default fallback uses load_config() (default.yaml) instead of get_preset('balanced')"
  - "valid_horizons excludes 1m (impractical as horizon); valid_freqs includes full set"
  - "Frequencies sorted by actual duration via pandas Timedelta for correct ordering"
  - "Preset resolution kept in PUT handler for backward compatibility but GET uses config loader"

patterns-established:
  - "Settings responses always include valid_freqs and valid_horizons lists"
  - "Frequency/horizon validation against bucket.py canonical set in PUT handler"

requirements-completed: [APIC-01, APIC-02]

# Metrics
duration: 10min
completed: 2026-02-28
---

# Phase 20 Plan 01: Settings Endpoint Default.yaml Fallback Summary

**Settings GET/PUT wired to default.yaml (5m/30m) with bucket.py validation and valid option lists for Phase 21 dropdowns**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-28T13:53:12Z
- **Completed:** 2026-02-28T14:03:28Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- GET /system/settings returns freq=5m, horizon=30m from default.yaml when no user config exists (was 1h/4h from balanced preset)
- PUT /system/settings validates freq/horizon against bucket.py _SUPPORTED_FREQUENCIES; rejects invalid values with 422
- SettingsResponse includes valid_freqs and valid_horizons lists for Phase 21 UI dropdown consumption
- Partial PUT updates merge with existing config without losing unspecified fields
- 6 regression tests covering both APIC-01 (default fallback) and APIC-02 (sub-hourly persistence)
- Full test suite (601 tests) passes with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for settings default fallback and sub-hourly validation** - `2d99c42` (test)
2. **Task 2: Implement settings handlers with default.yaml fallback and bucket.py validation** - `0bc6fb1` (feat)
3. **Task 3: Verify non-regression and clean up** - `7034fb6` (chore)

## Files Created/Modified
- `tests/api/test_settings.py` - 6 test cases for APIC-01 default fallback and APIC-02 sub-hourly persistence
- `src/bitbat/api/schemas.py` - SettingsResponse gains valid_freqs and valid_horizons fields; preset defaults to "custom"
- `src/bitbat/api/routes/system.py` - GET/PUT handlers rewritten: default.yaml fallback via load_config(), bucket.py validation, sorted option lists

## Decisions Made
- Used load_config() from bitbat.config.loader for default fallback instead of get_preset("balanced") -- ensures defaults match default.yaml (5m/30m)
- valid_horizons excludes "1m" (a 1-minute prediction horizon is impractical) but valid_freqs includes the full _SUPPORTED_FREQUENCIES set
- Frequency sorting uses pandas Timedelta conversion for correct duration ordering (not lexicographic)
- Kept get_preset import in PUT handler for backward compatibility when a preset name is provided, but GET handler no longer depends on presets

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Settings endpoint returns valid_freqs and valid_horizons lists ready for Phase 21 React dropdown consumption
- All sub-hourly frequencies (5m, 15m, 30m) accepted and validated
- Backward compatibility maintained for preset-based PUT requests

## Self-Check: PASSED

All files exist. All commits verified.

---
*Phase: 20-api-config-alignment*
*Completed: 2026-02-28*
