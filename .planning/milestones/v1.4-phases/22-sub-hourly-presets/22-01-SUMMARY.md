---
phase: 22-sub-hourly-presets
plan: "01"
subsystem: ui
tags: [presets, streamlit, sub-hourly, gui]

requires:
  - phase: 21-settings-ui-expansion
    provides: "API-driven dynamic dropdowns in Streamlit Settings"
provides:
  - "Scalper (5m/30m) and Swing (15m/1h) preset definitions"
  - "Sub-hourly format helpers in _format_freq/_format_horizon"
  - "5-preset Streamlit Settings page with sub-hourly dropdown options"
affects: [22-02, dashboard, presets]

tech-stack:
  added: []
  patterns: ["preset ordering fastest-to-slowest: scalper→conservative→balanced→aggressive→swing"]

key-files:
  created: []
  modified:
    - src/bitbat/gui/presets.py
    - streamlit/pages/1_⚙️_Settings.py

key-decisions:
  - "Scalper uses ⚡ icon, Swing uses 🌊 icon — match existing emoji style"
  - "Preset ordering: scalper, conservative, balanced, aggressive, swing (fastest to most methodical)"
  - "Sub-hourly format: 'Every 5 min', 'Every 15 min', 'Every 30 min' for freq; '15 min ahead', '30 min ahead' for horizon"

patterns-established:
  - "Sub-hourly format mapping: 5m→'Every 5 min', 15m→'Every 15 min', 30m→'Every 30 min'"

requirements-completed: [PRES-01, PRES-02, PRES-03]

duration: 15min
completed: 2026-03-01
---

# Plan 22-01: Sub-Hourly Presets Summary

**Scalper (5m/30m) and Swing (15m/1h) presets added to backend registry with sub-hourly format helpers; Streamlit Settings expanded to 5 presets with sub-hourly dropdown options**

## Performance

- **Duration:** 15 min
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added SCALPER preset (freq=5m, horizon=30m, tau=0.003, threshold=0.55)
- Added SWING preset (freq=15m, horizon=1h, tau=0.007, threshold=0.60)
- Sub-hourly format helpers: _format_freq covers 5m/15m/30m; _format_horizon covers 15m/30m
- Streamlit Settings expanded from 3 to 5 preset cards with correct ordering
- Advanced dropdowns now include 5m, 15m, 30m frequency and 15m, 30m horizon options

## Task Commits

1. **Task 1: Add Scalper and Swing presets with sub-hourly format helpers** - `e6634d9` (feat)
2. **Task 2: Update Streamlit Settings for 5 presets and sub-hourly options** - `b6c5e7c` (feat)

## Files Created/Modified
- `src/bitbat/gui/presets.py` - Added SCALPER and SWING preset definitions; extended _format_freq and _format_horizon mappings for sub-hourly values
- `streamlit/pages/1_⚙️_Settings.py` - 5 preset columns (scalper→swing ordering); sub-hourly freq/horizon dropdown options

## Decisions Made
- Scalper uses ⚡ (lightning) icon, Swing uses 🌊 (wave) icon to match existing emoji style
- Preset ordering from fastest to most methodical: scalper, conservative, balanced, aggressive, swing

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend presets ready for React dashboard to reference via API
- get_preset("scalper") and get_preset("swing") available for API PUT handler
- 22-02 can now build React PresetSelector with all 5 presets

---
*Phase: 22-sub-hourly-presets*
*Completed: 2026-03-01*
