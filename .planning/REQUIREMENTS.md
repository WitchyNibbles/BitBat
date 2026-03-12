# Requirements: BitBat v1.6 Accuracy Recovery & Technical Debt Remediation

**Defined:** 2026-03-08
**Core Value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.

## v1 Requirements

### Diagnosis

- [x] **DIAG-01**: Operator can identify which pipeline stage caused the live accuracy collapse (ingestion → features → labels → model → serving)
- [x] **DIAG-02**: Root cause is documented with a reproducible trace (CLI run or test) before any fix is applied

### Fix & Reset

- [x] **FIXR-01**: Root cause of live accuracy ~1% is fixed in code
- [x] **FIXR-02**: A clean reset procedure (data/ + models/ + autonomous.db) is executable via CLI command(s) and documented
- [x] **FIXR-03**: After reset + retrain, live directional accuracy on realized predictions exceeds random baseline (>33%)
- [x] **FIXR-04**: Monitor agent alerts when realized accuracy falls below a configurable threshold (default: 40%)

### Tech Debt

- [x] **DEBT-01**: cli.py monolith decomposed — 53 functions and 1802+ lines split into focused modules with no behavioral change
- [x] **DEBT-02**: Hardcoded `Path("models")` / `Path("metrics")` centralized — all 15+ occurrences replaced with config-driven path resolution
- [x] **DEBT-03**: Dual DB access unified — SQLAlchemy ORM + raw sqlite3 consolidated into a single consistent approach
- [x] **DEBT-04**: XGBoost objective mismatch fixed — `reg:squarederror` replaced with `multi:softprob` (or equivalent classification objective), model retrained

## Future Requirements

*(Moved from Active after v1.6 scoping)*

- **MICR-01**: Optional microstructure/LOB feature pipeline for short-horizon experiments
- **PORT-01**: Multi-asset portfolio-level forecasting and allocation workflow
- **EA-01**: EA-driven policy optimization under strict anti-overfitting controls

## Out of Scope

| Feature | Reason |
|---------|--------|
| Dashboard redesign | Monitor/runtime reliability still top priority |
| New model architectures | Stability before capability expansion |
| Bypassing walk-forward discipline | Core anti-overfitting constraint |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | Phase 29 | Complete |
| DIAG-02 | Phase 29 | Complete |
| FIXR-01 | Phase 30 | Complete |
| FIXR-02 | Phase 30 | Complete |
| FIXR-03 | Phase 30 | Complete |
| FIXR-04 | Phase 31 | Complete |
| DEBT-01 | Phase 32 | Complete |
| DEBT-02 | Phase 33 | Complete |
| DEBT-03 | Phase 34 | Complete |
| DEBT-04 | Phase 35 | Complete |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-08*
*Last updated: 2026-03-12 — DEBT-02/03/04 completed through Phases 33-35*
