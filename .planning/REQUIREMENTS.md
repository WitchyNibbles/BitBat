# Requirements: BitBat Configuration Alignment

**Defined:** 2026-02-28
**Core Value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.

## v1.4 Requirements

Requirements for configuration alignment. Each maps to roadmap phases.

### Presets

- [x] **PRES-01**: Operator can select a Scalper preset (5m freq, 30m horizon) in both GUI and React dashboard
- [x] **PRES-02**: Operator can select a Swing preset (15m freq, 1h horizon) in both GUI and React dashboard
- [x] **PRES-03**: Preset format helpers display human-readable labels for all sub-hourly frequencies (5m, 15m, 30m)

### Settings UI

- [x] **SETT-01**: React dashboard frequency dropdown includes 5m, 15m, 30m alongside 1h, 4h, 1d
- [x] **SETT-02**: React dashboard horizon dropdown includes 15m, 30m alongside 1h, 4h, 24h
- [x] **SETT-03**: React dashboard defaults reflect default.yaml values (5m/30m) instead of hardcoded 1h/4h

### API Config

- [x] **APIC-01**: API settings endpoint falls back to default.yaml values when no user config exists
- [x] **APIC-02**: API settings endpoint accepts and persists sub-hourly freq/horizon values

### Test Coverage

- [ ] **TEST-01**: Preset tests cover Scalper and Swing presets with correct parameter values
- [ ] **TEST-02**: Settings/API tests verify sub-hourly freq/horizon round-trip through save and load

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Config

- **ADVC-01**: Operator can define custom presets with arbitrary freq/horizon combinations
- **ADVC-02**: Settings UI validates that horizon >= freq to prevent nonsensical configurations

## Out of Scope

| Feature | Reason |
|---------|--------|
| Streamlit settings page redesign | Streamlit dashboard is being replaced by React dashboard |
| New freq values beyond supported set | bucket.py already defines the canonical set (1m-24h) |
| Changing default.yaml values | The actual defaults are correct; the UI needs to reflect them |
| yfinance data source changes | Data ingestion already supports sub-hourly via existing freq map |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRES-01 | Phase 22 | Complete |
| PRES-02 | Phase 22 | Complete |
| PRES-03 | Phase 22 | Complete |
| SETT-01 | Phase 21 | Complete |
| SETT-02 | Phase 21 | Complete |
| SETT-03 | Phase 21 | Complete |
| APIC-01 | Phase 20 | Complete |
| APIC-02 | Phase 20 | Complete |
| TEST-01 | Phase 23 | Pending |
| TEST-02 | Phase 23 | Pending |

**Coverage:**
- v1.4 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-02-28*
*Last updated: 2026-02-28 after roadmap creation*
