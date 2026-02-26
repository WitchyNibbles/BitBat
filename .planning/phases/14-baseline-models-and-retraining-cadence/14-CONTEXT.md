# Phase 14: Baseline Models and Retraining Cadence - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning
**Source:** Roadmap requirements + `deep-research-report.md`

<domain>
## Phase Boundary

Phase 14 establishes reproducible baseline model training and retraining loops on top of
Phase 13's leakage-safe dataset contract. The goal is to make model comparison and
retraining cadence operationally reliable before adding cost-aware selection or promotion gates.

</domain>

<decisions>
## Implementation Decisions

### Dual Tree-Ensemble Baselines Are Required
- XGBoost and RandomForest must train from the same dataset contract and comparable split logic.
- Baseline artifacts must be versioned and queryable so downstream phases can evaluate candidates fairly.
- Default model flows should remain deterministic under fixed seeds and identical folds.

### Retraining/Backtest Windows Must Be Explicit and Repeatable
- Retraining cadence must be window-driven (train window + backtest window), not ad-hoc one-shot runs.
- Window definitions should be configurable from CLI/config, then applied consistently in CV/retraining flows.
- Walk-forward style rolling execution should be the default benchmark path for medium-horizon BTC forecasts.

### Regime/Drift Diagnostics Must Be Emitted Per Window
- Each retraining window should output regime/drift diagnostics linked to the same window metadata.
- Diagnostics should be persisted in machine-readable artifacts so comparisons are automatable.
- Drift diagnostics should complement, not replace, baseline regression/directional metrics.

### Scope Boundaries
- Phase 14 does not introduce promotion gates (Phase 16) or full cost-aware model selection rules (Phase 15).
- Phase 14 focuses on baseline training comparability and retraining diagnostics, not UI expansion.

### Claude's Discretion
- Choose module boundaries for baseline registry/training orchestration as long as MODL-01/02/03 remain test-locked.
- Choose concrete regime signal set (volatility/trend/stability indicators) if outputs are deterministic and documented.

</decisions>

<specifics>
## Specific Ideas

- Build from current model and retraining entrypoints:
  - `src/bitbat/model/train.py`
  - `src/bitbat/model/walk_forward.py`
  - `src/bitbat/cli.py`
  - `src/bitbat/autonomous/continuous_trainer.py`
  - `src/bitbat/autonomous/retrainer.py`
  - `src/bitbat/autonomous/drift.py`
- Add/extend tests in:
  - `tests/model/test_train.py`
  - `tests/model/test_walk_forward.py`
  - `tests/autonomous/test_retrainer.py`
  - `tests/autonomous/test_drift.py`
  - `tests/test_cli.py`
- Preserve compatibility with existing `features build`, `model cv`, and `model train` workflows.

</specifics>

<deferred>
## Deferred Ideas

- Purge/embargo policy expansion and multi-testing safeguards (Phase 15/16).
- Promotion policy enforcement across consecutive out-of-sample windows (Phase 16).
- Transformer/GNN-first model expansion.

</deferred>

---

*Phase: 14-baseline-models-and-retraining-cadence*
*Context gathered: 2026-02-26*
