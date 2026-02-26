# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- ✅ **v1.1 UI-First Simplification** — Phases 10-12 shipped on 2026-02-25 ([roadmap archive](milestones/v1.1-ROADMAP.md), [requirements archive](milestones/v1.1-REQUIREMENTS.md), [audit archive](milestones/v1.1-MILESTONE-AUDIT.md)).
- 🚧 **v1.2 BTC Prediction Accuracy Evolution** — Phases 13-16 planned (improve prediction accuracy through leakage-safe data contracts, retraining discipline, and cost-aware model selection).

## v1.2 Planned Phases

### Phase 13: Data and Label Contract Upgrade
**Goal:** Move BitBat prediction inputs/targets to leakage-safe as-of features and return-first labels.
**Depends on:** v1.1 verified baseline
**Requirements:** [DATA-01, DATA-02, LABL-01]
**Plans:** 3/3 plans complete
**Status:** Complete

Success criteria:
1. Dataset assembly enforces as-of timestamp alignment across all feature sources and rejects future leakage.
2. Return and direction labels are generated for configured horizons from one shared label contract.
3. Triple-barrier labeling mode exists as optional output without breaking baseline datasets.
4. Automated tests verify timestamp ordering and horizon/label consistency.

### Phase 14: Baseline Models and Retraining Cadence
**Goal:** Establish robust baseline predictors and repeatable retraining windows before complex model expansion.
**Depends on:** Phase 13
**Requirements:** [MODL-01, MODL-02, MODL-03]
**Plans:** 3/3 plans complete
**Status:** Complete

Success criteria:
1. XGBoost and RandomForest baselines train from the v1.2 dataset contract and produce versioned artifacts.
2. Train/backtest windows run as rolling walk-forward cycles with configurable durations.
3. Regime/drift diagnostics are emitted for each retraining window and tied to model artifacts.
4. Baseline performance summaries are reproducible across repeated runs.

### Phase 15: Cost-Aware Walk-Forward Evaluation
**Goal:** Replace optimistic evaluation with realistic, leakage-resistant, and cost-aware model assessment.
**Depends on:** Phase 14
**Requirements:** [EVAL-01, EVAL-02, EVAL-03]
**Plans:** 3/3 plans complete
**Status:** Complete

Success criteria:
1. Evaluation engine supports walk-forward splits with purge/embargo controls for overlapping horizons.
2. Backtest outputs include fee/slippage-adjusted net metrics alongside gross predictive metrics.
3. Model comparison outputs include regression, directional, and risk-aware metrics in one report.
4. Champion selection rule is explicit, automated, and persisted with evaluation artifacts.

### Phase 16: Promotion Guardrails and Optimization Safety
**Goal:** Prevent overfit model promotion through statistically safer optimization and multi-window acceptance gates.
**Depends on:** Phase 15
**Requirements:** [EVAL-04, OPER-02]
**Plans:** 2/3 plans executed
**Status:** In Progress

Success criteria:
1. Hyperparameter/threshold optimization runs nested validation and records search provenance.
2. Multiple-testing safeguards are computed and used to reject unstable candidates.
3. Promotion gate requires incumbent-beating out-of-sample performance across consecutive windows and drawdown constraints.
4. Regression tests enforce that unsafe or leakage-prone candidates cannot be promoted.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 13. Data and Label Contract Upgrade | v1.2 | 3/3 | Complete | 2026-02-25 |
| 14. Baseline Models and Retraining Cadence | v1.2 | 3/3 | Complete | 2026-02-26 |
| 15. Cost-Aware Walk-Forward Evaluation | v1.2 | 3/3 | Complete | 2026-02-26 |
| 16. Promotion Guardrails and Optimization Safety | v1.2 | 2/3 | In Progress | - |

## Next

- Execute Phase 16 plans with `$gsd-execute-phase 16`.
