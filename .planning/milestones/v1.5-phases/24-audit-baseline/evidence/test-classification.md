# Test Classification Report

**Date:** 2026-03-04
**Plan:** 24-01 (Audit Baseline)
**Tool:** pytest markers + manual classification

## Summary

| Metric | Count |
|--------|-------|
| Total test files (before cleanup) | 77 |
| Total test files (after cleanup) | 76 |
| Files deleted (pure source-reader) | 1 |
| Files reclassified (*_complete.py kept) | 16 |
| Tests before cleanup | 608 |
| Tests after cleanup | 603 |
| Tests removed | 5 |

### Classification by Marker

| Marker | Files | Tests |
|--------|-------|-------|
| behavioral | 32 | 269 |
| integration | 37 | 308 |
| structural | 7 | 26 |
| **Total** | **76** | **603** |

## Deletion Log

### Deleted Files (Pure Source-Reader Milestone Markers)

| File | Tests Removed | Reason |
|------|--------------|--------|
| `tests/autonomous/test_phase19_d1_monitor_alignment_complete.py` | 5 | All 5 tests are pure source-readers using `Path.read_text()` and string-presence assertions. No production code imported or exercised. |

**Additional cleanup:** References to the deleted file were removed from:
- `tests/autonomous/test_phase8_d1_monitor_schema_complete.py` (D1_CANONICAL_SUITE list)
- `tests/gui/test_phase8_release_verification_complete.py` (REQUIRED_GATE_FILES list, assertion, and Makefile check assertion)
- `Makefile` (test-release target)

## Reclassification Log

### *_complete.py Files Retained (Exercise Real Production Code)

| File | New Marker | Rationale |
|------|-----------|-----------|
| `tests/analytics/test_phase3_complete.py` | integration | Imports bitbat.analytics modules, exercises backtest report and explainer with real data |
| `tests/api/test_phase4_complete.py` | integration | Creates ASGI client against bitbat.api.app, exercises API endpoints with real DB fixtures |
| `tests/autonomous/test_phase8_d1_monitor_schema_complete.py` | integration | Seeds SQLAlchemy DB via bitbat.autonomous.models, exercises schema compat and agent startup |
| `tests/autonomous/test_session3_complete.py` | integration | Exercises autonomous agent components with real DB and monkeypatched configs |
| `tests/autonomous/test_session4_complete.py` | integration | Integration tests for ingestion components (rate limiter, price ingestion, news ingestion) |
| `tests/gui/test_complete_gui.py` | integration | Exercises GUI widgets, timeline, presets with real SQLite fixtures |
| `tests/gui/test_phase5_timeline_complete.py` | integration | Exercises bitbat.gui.timeline functions with real DB data |
| `tests/gui/test_phase6_timeline_ux_complete.py` | integration | Exercises timeline UX functions (filters, insights, empty state) with real DB |
| `tests/gui/test_phase7_streamlit_compat_complete.py` | integration | Tests widget functions + AST validation of Streamlit source for width compat |
| `tests/gui/test_phase8_d2_timeline_complete.py` | integration | Exercises timeline data/rendering functions with real DB fixtures |
| `tests/gui/test_phase8_release_verification_complete.py` | structural | Checks file existence, AST parsing of Streamlit source, Makefile content verification |
| `tests/gui/test_phase9_timeline_readability_complete.py` | integration | Exercises timeline readability and comparison features with real DB |
| `tests/gui/test_phase10_supported_surface_complete.py` | structural | Validates page directory structure and navigation constants against source |
| `tests/gui/test_phase11_runtime_stability_complete.py` | integration | Exercises widget functions with legacy DB schema, validates retired page structure |
| `tests/gui/test_phase12_simplified_ui_regression_complete.py` | structural | Validates page count, directory structure, and navigation references |
| `tests/model/test_phase5_complete.py` | integration | Exercises Monte Carlo, backtest, walk-forward, and ensemble with real model objects |

## Full Classification

### behavioral (32 files, 269 tests)

Tests that exercise a single function or method in isolation, using mocks/monkeypatch for dependencies.

| File | Module |
|------|--------|
| `tests/analytics/test_backtest_report.py` | analytics |
| `tests/analytics/test_explainer.py` | analytics |
| `tests/analytics/test_feature_analysis.py` | analytics |
| `tests/analytics/test_monte_carlo.py` | analytics |
| `tests/autonomous/test_drift.py` | autonomous |
| `tests/autonomous/test_metrics.py` | autonomous |
| `tests/autonomous/test_orchestrator.py` | autonomous |
| `tests/autonomous/test_retrainer.py` | autonomous |
| `tests/backtest/test_engine.py` | backtest |
| `tests/backtest/test_metrics.py` | backtest |
| `tests/contracts/test_contracts.py` | contracts |
| `tests/dataset/test_splits.py` | dataset |
| `tests/features/test_macro_features.py` | features |
| `tests/features/test_onchain_features.py` | features |
| `tests/features/test_price_features.py` | features |
| `tests/features/test_sentiment.py` | features |
| `tests/features/test_sentiment_aggregate.py` | features |
| `tests/features/test_volatility.py` | features |
| `tests/gui/test_presets.py` | gui |
| `tests/io/test_io_helpers.py` | io |
| `tests/labeling/test_returns.py` | labeling |
| `tests/labeling/test_targets.py` | labeling |
| `tests/labeling/test_triple_barrier.py` | labeling |
| `tests/model/test_evaluate.py` | model |
| `tests/model/test_infer.py` | model |
| `tests/model/test_optimize.py` | model |
| `tests/model/test_train.py` | model |
| `tests/model/test_walk_forward.py` | model |
| `tests/test_config_loader.py` | config |
| `tests/test_run_monitoring_agent.py` | autonomous |
| `tests/timealign/test_asof_join.py` | timealign |
| `tests/timealign/test_bucket_calendar.py` | timealign |
| `tests/timealign/test_purging.py` | timealign |

### integration (37 files, 308 tests)

Tests that exercise multiple real components together, using tmp_path for real I/O, seeding databases, or calling CLI commands.

| File | Module |
|------|--------|
| `tests/analytics/test_phase3_complete.py` | analytics |
| `tests/api/test_health.py` | api |
| `tests/api/test_metrics.py` | api |
| `tests/api/test_phase4_complete.py` | api |
| `tests/api/test_predictions.py` | api |
| `tests/api/test_settings.py` | api |
| `tests/autonomous/test_agent_integration.py` | autonomous |
| `tests/autonomous/test_db.py` | autonomous |
| `tests/autonomous/test_ingestion.py` | autonomous |
| `tests/autonomous/test_init_script.py` | autonomous |
| `tests/autonomous/test_models.py` | autonomous |
| `tests/autonomous/test_phase8_d1_monitor_schema_complete.py` | autonomous |
| `tests/autonomous/test_schema_compat.py` | autonomous |
| `tests/autonomous/test_session3_complete.py` | autonomous |
| `tests/autonomous/test_session4_complete.py` | autonomous |
| `tests/autonomous/test_validator.py` | autonomous |
| `tests/dataset/test_build_xy.py` | dataset |
| `tests/gui/test_complete_gui.py` | gui |
| `tests/gui/test_phase11_runtime_stability_complete.py` | gui |
| `tests/gui/test_phase5_timeline_complete.py` | gui |
| `tests/gui/test_phase6_timeline_ux_complete.py` | gui |
| `tests/gui/test_phase7_streamlit_compat_complete.py` | gui |
| `tests/gui/test_phase8_d2_timeline_complete.py` | gui |
| `tests/gui/test_phase9_timeline_readability_complete.py` | gui |
| `tests/gui/test_timeline.py` | gui |
| `tests/gui/test_widgets.py` | gui |
| `tests/ingest/test_macro_fred.py` | ingest |
| `tests/ingest/test_news_cryptocompare.py` | ingest |
| `tests/ingest/test_news_gdelt.py` | ingest |
| `tests/ingest/test_onchain.py` | ingest |
| `tests/ingest/test_yfinance_prices.py` | ingest |
| `tests/model/test_ensemble.py` | model |
| `tests/model/test_persist.py` | model |
| `tests/model/test_phase5_complete.py` | model |
| `tests/test_bootstrap_monitor_model.py` | autonomous |
| `tests/test_cli.py` | cli |

### structural (7 files, 26 tests)

Tests that check file existence, read source code, verify docstrings, check CLI help text, or validate schemas without exercising production logic.

| File | Module |
|------|--------|
| `tests/docs/test_persistence_docs.py` | docs |
| `tests/gui/test_phase10_supported_surface_complete.py` | gui |
| `tests/gui/test_phase12_simplified_ui_regression_complete.py` | gui |
| `tests/gui/test_phase12_supported_views_smoke.py` | gui |
| `tests/gui/test_phase8_release_verification_complete.py` | gui |
| `tests/gui/test_streamlit_width_compat.py` | gui |
| `tests/test_monitor_runbook_contract.py` | docs |

## Coverage Gap Matrix

Maps all v1.5 requirements to existing test coverage status.

| Requirement | Description | Existing Test Coverage | Gap? |
|-------------|-------------|----------------------|------|
| **CORR-01** | Retrainer subprocess CLI contract (--tau removal) | `tests/autonomous/test_retrainer.py` tests retrainer logic but does NOT test the subprocess CLI invocation with `--tau`. No test catches the broken `--tau` argument. | **YES - GAP** |
| **CORR-02** | CV metric key mismatch | `tests/model/test_walk_forward.py` tests walk-forward CV but does not verify metric key consistency between writer and reader. | **YES - GAP** |
| **CORR-03** | `regression_metrics()` side effects | `tests/model/test_evaluate.py` tests evaluation functions but does not isolate computation from I/O side effects. | **YES - GAP** |
| **CORR-04** | `assert isinstance` in production code | No test validates that production code uses proper runtime guards instead of assert. | **YES - GAP** |
| **CORR-05** | `test_leakage.py` with PR-AUC guardrail | `tests/features/test_price_features.py` has `_assert_no_leakage` helper but the dedicated `test_leakage.py` referenced in CLAUDE.md does not exist. | **YES - GAP** |
| **CORR-06** | API route freq/horizon defaults | `tests/api/test_predictions.py` and `tests/api/test_health.py` test API routes but do not verify default freq/horizon alignment with config. | **YES - GAP** |
| **LEAK-01** | OBV fold-boundary leakage assessment | No existing test compares model performance with/without OBV to assess leakage impact. | **YES - GAP** |
| **LEAK-02** | OBV cumsum leakage fix | No fold-aware OBV computation exists. `tests/features/test_price_features.py` tests OBV output shape but not fold-boundary behavior. | **YES - GAP** |
| **ARCH-01** | Private feature pipeline promotion | `tests/features/` tests individual feature functions but `_generate_price_features` and `_join_auxiliary_features` are private and untested directly. | **YES - GAP** |
| **ARCH-02** | Price loading consolidation | `tests/ingest/test_yfinance_prices.py` tests ingestion but 3 divergent price-loading implementations are not tested for consistency. | **YES - GAP** |
| **ARCH-03** | Config reset function | `tests/test_config_loader.py` tests config loading but no `reset()` function exists yet, so no test coverage. | **YES - GAP** |
| **ARCH-04** | API-to-GUI cross-layer import | No test enforces import boundaries. `tests/api/` tests do not check for GUI imports. | **YES - GAP** |
| **ARCH-05** | import-linter CI contracts | No import-linter configuration or tests exist. | **YES - GAP** |
| **ARCH-06** | ruff C901 complexity gate | No complexity gate exists in CI configuration. | **YES - GAP** |

### Coverage Gap Summary

| Category | Total | Covered | Gap |
|----------|-------|---------|-----|
| CORR (Correctness) | 6 | 0 | 6 |
| LEAK (Leakage) | 2 | 0 | 2 |
| ARCH (Architecture) | 6 | 0 | 6 |
| **Total** | **14** | **0** | **14** |

All 14 v1.5 remediation requirements have coverage gaps. This is expected: the requirements were defined specifically to address known issues, and the existing test suite does not exercise these specific code paths or behaviors.

## Marker Registration

Markers registered in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "slow: marks tests as slow and may require network access",
    "behavioral: behavioral unit test exercising production logic with mocks/isolation",
    "integration: integration test exercising multiple real components together",
    "structural: structural conformance test checking file existence, schemas, CLI contracts",
]
```

## Verification

```
$ poetry run pytest --co -q
603 tests collected in 2.07s

$ poetry run pytest -m behavioral --co -q
269/603 tests collected (334 deselected) in 2.01s

$ poetry run pytest -m integration --co -q
308/603 tests collected (295 deselected) in 2.03s

$ poetry run pytest -m structural --co -q
26/603 tests collected (577 deselected) in 2.06s

$ poetry run pytest -m "not behavioral and not integration and not structural" --co -q
no tests collected (603 deselected) in 2.09s
```

No `PytestUnknownMarkWarning` warnings. All 603 tests accounted for.

---
*Generated: 2026-03-04*
*Plan: 24-01 (Audit Baseline - Test Classification)*
