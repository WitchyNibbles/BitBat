# Requirements: BitBat v1.5 Codebase Health Audit & Critical Remediation

**Defined:** 2026-03-04
**Core Value:** A reliable prediction system where operators can trust that monitoring outputs correspond to real, active prediction flows for the configured runtime pair.

## v1.5 Requirements

Requirements for the audit and remediation milestone. Each maps to roadmap phases.

### Audit Baseline

- [x] **AUDT-01**: All existing tests classified by type (behavioral unit, integration, structural conformance, milestone-marker) with coverage gap report
- [ ] **AUDT-02**: Dead code audit completed with vulture at 80% confidence (findings triaged, false-positive whitelist created)
- [ ] **AUDT-03**: Branch coverage report generated with pytest-cov identifying lowest-coverage modules
- [ ] **AUDT-04**: Complexity audit completed with radon identifying high-complexity functions for remediation candidates
- [ ] **AUDT-05**: E2E pipeline smoke test executed (ingest → features → train → batch → monitor) documenting which sequential steps pass and which fail

### Critical Correctness

- [ ] **CORR-01**: Retrainer subprocess CLI contract fixed (--tau argument removed, command arguments match actual CLI interface)
- [ ] **CORR-02**: CV metric key mismatch fixed (writer and reader use same key names for walk-forward scores)
- [ ] **CORR-03**: `regression_metrics()` refactored to separate computation from file I/O side effects
- [ ] **CORR-04**: `assert isinstance` replaced with proper runtime guards in production code paths
- [ ] **CORR-05**: `test_leakage.py` created with PR-AUC guardrail as documented in CLAUDE.md
- [ ] **CORR-06**: API route freq/horizon defaults aligned with default.yaml instead of hardcoded 1h/4h

### Architecture

- [ ] **ARCH-01**: Private feature pipeline functions (`_generate_price_features`, `_join_auxiliary_features`) promoted to public API with stable interface
- [ ] **ARCH-02**: Price loading logic consolidated into single shared function replacing 3 divergent implementations
- [ ] **ARCH-03**: Config reset function added to `config/loader.py` and used in test fixtures for isolation
- [ ] **ARCH-04**: API→GUI cross-layer import eliminated (`api/routes/system.py` no longer imports from `gui/`)
- [ ] **ARCH-05**: import-linter contracts added to CI preventing future cross-layer import drift
- [ ] **ARCH-06**: ruff C901 complexity gate added to CI preventing new high-complexity functions

### OBV Leakage

- [ ] **LEAK-01**: OBV fold-boundary leakage impact assessed with empirical walk-forward comparison (with vs without OBV)
- [ ] **LEAK-02**: OBV cumsum leakage fixed with fold-aware computation (conditional on LEAK-01 confirming feasibility and scoped fix)

## Deferred Requirements

Acknowledged but not in current roadmap. Tracked for future milestones.

### Architecture Debt (v1.6+)

- **DEBT-01**: CLI monolith decomposition (cli.py at 1802 lines, 53 functions)
- **DEBT-02**: Full path centralization (15+ hardcoded `Path("models")` / `Path("metrics")` sites)
- **DEBT-03**: Dual DB access unification (SQLAlchemy ORM + raw sqlite3 accessing same database)
- **DEBT-04**: XGBoost objective mismatch assessment (reg:squarederror for directional classification task)

### Feature Development (v1.6+)

- **MICR-01**: Add optional microstructure/LOB feature pipeline for short-horizon experiments
- **PORT-01**: Add multi-asset portfolio-level forecasting and allocation workflow
- **EA-01**: Add EA-driven policy optimization under strict anti-overfitting controls

## Out of Scope

| Feature | Reason |
|---------|--------|
| Style-only fixes (naming, docstrings, formatting) | Creates audit noise; consumes milestone budget without correctness gains |
| XGBoost objective change (reg:squarederror → multi:softprob) | Cascading downstream impact on inference, backtest, monitoring, API schemas; needs dedicated impact analysis |
| CLI monolith decomposition | High fix cost, medium severity; defer to dedicated refactoring milestone |
| Full path centralization | Medium severity, requires config propagation through all callers; defer |
| Broad exception narrowing | Many `except Exception` blocks are architecturally intentional (agent keep-cycling semantics) |
| New feature development | Audit milestone; no new capabilities added |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUDT-01 | Phase 24 | Complete |
| AUDT-02 | Phase 24 | Pending |
| AUDT-03 | Phase 24 | Pending |
| AUDT-04 | Phase 24 | Pending |
| AUDT-05 | Phase 24 | Pending |
| CORR-01 | Phase 25 | Pending |
| CORR-02 | Phase 25 | Pending |
| CORR-03 | Phase 25 | Pending |
| CORR-04 | Phase 25 | Pending |
| CORR-05 | Phase 25 | Pending |
| CORR-06 | Phase 25 | Pending |
| LEAK-01 | Phase 25 | Pending |
| LEAK-02 | Phase 25 | Pending |
| ARCH-01 | Phase 26 | Pending |
| ARCH-02 | Phase 26 | Pending |
| ARCH-03 | Phase 26 | Pending |
| ARCH-04 | Phase 26 | Pending |
| ARCH-05 | Phase 27 | Pending |
| ARCH-06 | Phase 27 | Pending |

**Coverage:**
- v1.5 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-04*
*Last updated: 2026-03-04 after initial definition*
