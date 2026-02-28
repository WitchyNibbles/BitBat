# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- ✅ **v1.1 UI-First Simplification** — Phases 10-12 shipped on 2026-02-25 ([roadmap archive](milestones/v1.1-ROADMAP.md), [requirements archive](milestones/v1.1-REQUIREMENTS.md), [audit archive](milestones/v1.1-MILESTONE-AUDIT.md)).
- ✅ **v1.2 BTC Prediction Accuracy Evolution** — Phases 13-16 shipped on 2026-02-26 ([roadmap archive](milestones/v1.2-ROADMAP.md), [requirements archive](milestones/v1.2-REQUIREMENTS.md)).
- ✅ **v1.3 Autonomous Monitor Alignment and Metrics Integrity** — Phases 17-19 shipped on 2026-02-26 ([roadmap archive](milestones/v1.3-ROADMAP.md), [requirements archive](milestones/v1.3-REQUIREMENTS.md)).

## v1.4 Configuration Alignment

**Milestone Goal:** Make the UI settings, presets, and API defaults reflect the actual runtime configuration (5m freq) instead of hardcoded 1h-only options.

## Phases

- [x] **Phase 20: API Config Alignment** - API settings endpoint defaults to default.yaml and accepts sub-hourly freq/horizon values (completed 2026-02-28)
- [ ] **Phase 21: Settings UI Expansion** - React dashboard dropdowns expose all supported frequencies and horizons with correct defaults
- [ ] **Phase 22: Sub-Hourly Presets** - Scalper and Swing presets available in both GUI and React dashboard with human-readable labels
- [ ] **Phase 23: Configuration Test Coverage** - Automated tests validate preset parameters and settings round-trip for the full frequency range

## Phase Details

### Phase 20: API Config Alignment
**Goal**: Operators get correct default configuration from the API without manual overrides, and can persist sub-hourly freq/horizon selections
**Depends on**: Nothing (first phase in v1.4)
**Requirements**: APIC-01, APIC-02
**Success Criteria** (what must be TRUE):
  1. When no user config exists, the API settings endpoint returns freq=5m and horizon=30m (matching default.yaml)
  2. Operator can POST a settings update with freq=15m/horizon=1h and GET it back unchanged on the next request
  3. All sub-hourly freq values (5m, 15m, 30m) are accepted by the API settings endpoint without validation errors
**Plans**: 1 plan
Plans:
- [ ] 20-01-PLAN.md — Settings endpoint default.yaml fallback, bucket.py validation, and regression tests (TDD)

### Phase 21: Settings UI Expansion
**Goal**: Operators see the full range of supported frequencies and horizons in the React dashboard, with defaults that match the actual runtime configuration
**Depends on**: Phase 20
**Requirements**: SETT-01, SETT-02, SETT-03
**Success Criteria** (what must be TRUE):
  1. React dashboard frequency dropdown shows 5m, 15m, 30m, 1h, 4h, 1d as selectable options
  2. React dashboard horizon dropdown shows 15m, 30m, 1h, 4h, 24h as selectable options
  3. On first load with no saved preferences, the React dashboard shows 5m frequency and 30m horizon (matching default.yaml)
  4. Selecting a sub-hourly frequency in the dropdown persists through page navigation without reverting to 1h
**Plans**: TBD

### Phase 22: Sub-Hourly Presets
**Goal**: Operators can choose named trading presets that configure sub-hourly freq/horizon pairs in a single click
**Depends on**: Phase 21
**Requirements**: PRES-01, PRES-02, PRES-03
**Success Criteria** (what must be TRUE):
  1. Operator can select "Scalper" preset in both Streamlit GUI and React dashboard, which sets freq=5m and horizon=30m
  2. Operator can select "Swing" preset in both Streamlit GUI and React dashboard, which sets freq=15m and horizon=1h
  3. Format helpers display "5 min", "15 min", "30 min" (or equivalent human-readable labels) instead of raw "5m", "15m", "30m" strings
  4. Preset selection updates both the frequency and horizon dropdowns to the preset values
**Plans**: TBD

### Phase 23: Configuration Test Coverage
**Goal**: Automated tests guarantee that presets and settings behave correctly across the full supported frequency range, preventing regressions
**Depends on**: Phase 22
**Requirements**: TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. Running the test suite exercises both Scalper (5m/30m) and Swing (15m/1h) presets and asserts correct parameter values
  2. A settings round-trip test saves a sub-hourly freq/horizon via the API, reloads, and verifies the values match
  3. All new tests pass in `make test-release` alongside existing D1/D2/D3 gates
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 20 -> 21 -> 22 -> 23

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 20. API Config Alignment | 1/1 | Complete    | 2026-02-28 |
| 21. Settings UI Expansion | 0/0 | Not started | - |
| 22. Sub-Hourly Presets | 0/0 | Not started | - |
| 23. Configuration Test Coverage | 0/0 | Not started | - |
