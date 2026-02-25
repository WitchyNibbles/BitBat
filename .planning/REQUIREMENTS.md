# Requirements: BitBat Prediction Accuracy Evolution

**Defined:** 2026-02-25
**Status:** v1.2 planned
**Core Value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## v1.2 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Data and Label Integrity

- [x] **DATA-01**: BTC training features are built with strict as-of timestamp alignment so no feature row can reference future information.
- [x] **DATA-02**: Default prediction targets are return-based (configurable horizons) with direction labels derived from the same horizon definitions.
- [ ] **LABL-01**: Triple-barrier event labels can be generated as an optional dataset mode for trading-aligned experiments.

### Model Baselines and Retraining

- [ ] **MODL-01**: Tree-ensemble baselines (XGBoost and RandomForest) train and produce comparable prediction artifacts from the same dataset contract.
- [ ] **MODL-02**: Retraining/backtest windows are configurable and run as rolling walk-forward cycles (train window + backtest window) without manual intervention.
- [ ] **MODL-03**: Regime/drift diagnostics are computed per retrain window and surfaced in model evaluation outputs.

### Evaluation and Selection Rigor

- [ ] **EVAL-01**: Walk-forward evaluation enforces time-ordered splits and supports purge/embargo-style controls for overlapping label horizons.
- [ ] **EVAL-02**: Backtest evaluation reports net metrics with transaction fees and slippage, not only gross predictive performance.
- [ ] **EVAL-03**: Candidate model reports include regression, directional, and risk-aware metrics with an explicit champion-selection rule.
- [ ] **EVAL-04**: Hyperparameter/threshold optimization uses nested validation and records multiple-testing safeguards before candidate promotion.

### Promotion and Operations Safety

- [ ] **OPER-02**: A model is promoted only when it beats the incumbent across consecutive out-of-sample windows without violating drawdown guardrails.

## v1.3+ Requirements (Deferred)

Deferred to a later milestone after v1.2 pipeline accuracy controls are stable.

### Advanced Modeling

- **MICR-01**: Add optional microstructure/LOB feature pipeline for minute-level short-horizon signal experiments.
- **PORT-01**: Add multi-asset portfolio-level forecasting and allocation workflow beyond single-BTC prediction.
- **EA-01**: Add EA-driven policy optimization layer for threshold/policy search under strict anti-overfitting controls.

## Out of Scope

Explicitly excluded for v1.2 to keep scope focused on measurable accuracy gains.

| Feature | Reason |
|---------|--------|
| Rebuilding all retired advanced Streamlit pages | v1.2 target is prediction accuracy and evaluation rigor, not UI surface expansion |
| Raw next-price forecasting as primary objective | Return/direction labels are more stable and better aligned with trading decisions |
| Immediate Transformer/GNN-first migration | Research favors strong baselines and pipeline discipline before high-complexity architectures |
| Live capital deployment changes | Milestone focuses on offline and simulated evaluation reliability first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 13 | Complete |
| DATA-02 | Phase 13 | Complete |
| LABL-01 | Phase 13 | Pending |
| MODL-01 | Phase 14 | Pending |
| MODL-02 | Phase 14 | Pending |
| MODL-03 | Phase 14 | Pending |
| EVAL-01 | Phase 15 | Pending |
| EVAL-02 | Phase 15 | Pending |
| EVAL-03 | Phase 15 | Pending |
| EVAL-04 | Phase 16 | Pending |
| OPER-02 | Phase 16 | Pending |

**Coverage:**
- v1.2 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-25*
*Last updated: 2026-02-25 after v1.2 milestone initialization*
