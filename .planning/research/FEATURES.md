# Feature Research

**Domain:** Code quality audit — Python ML prediction pipeline (BitBat v1.5)
**Researched:** 2026-03-04
**Confidence:** HIGH

## Overview

This document maps the "feature landscape" for the v1.5 codebase health audit. In this
context "features" are the audit checks themselves: what the audit must find, categorized
by severity, with grep-searchable patterns for each issue and a description of what "good"
looks like. Sources are academic ML code-smell research (MLScent, arxiv 2502.18466),
scikit-learn official pitfall documentation, and direct inspection of the BitBat codebase.

---

## Feature Landscape

### Table Stakes (Audit Must Catch These)

Issues that any credible ML audit is expected to surface. Missing these = audit is
incomplete.

| Check | Severity | Why Expected | Grep Pattern |
|-------|----------|--------------|--------------|
| Hardcoded hyperparameters in training code | CRITICAL | Reproducibility and tuning discipline | `num_boost_round=100`, `max_depth=6`, `eta=0.05` |
| OBV cumsum leakage across train/test boundary | HIGH | OBV accumulates globally; not sliced per fold | `\.cumsum()` in `features/` |
| `regression_metrics` writes files as side effect | HIGH | Functions computing metrics should not write to disk | `fig.savefig`, `metrics_path.write_text` inside `regression_metrics` |
| `assert` in production code (not tests) | HIGH | Stripped by `-O` optimization; not a runtime guard | `assert isinstance` in `src/bitbat/model/train.py` |
| Broad `except Exception` silencing | HIGH | Hides correctness failures; masks pipeline corruption | `except Exception:` followed by `logger.warning` with no re-raise |
| `iterrows()` in feature pipeline hot paths | MEDIUM | 10-100x slower than vectorized pandas; sentiment aggregate is O(n*m) | `\.iterrows()` in `src/bitbat/` |
| Module-level global config state | MEDIUM | Makes parallel test isolation impossible; test order dependency | `global _ACTIVE_CONFIG` in `loader.py` |
| Hardcoded relative paths (`Path("models")`, `Path("metrics")`) | MEDIUM | Breaks when CWD changes; untestable without filesystem fixtures | `Path\("models"\)\|Path\("data"\)\|Path\("metrics"\)` in `src/` |
| Repeated `from pathlib import Path` inside function bodies | LOW | Import inside function is an anti-pattern; should be module-level | `from pathlib import Path` inside `def ` in `agent.py` |
| Stray parquet files committed to repo root | LOW | Data artifacts in source tree; confusing and git-bloated | `n.parquet`, `p.parquet`, `test_preds.parquet` in root |

### Differentiators (Deeper Audit Value)

Checks that go beyond surface linting and add genuine ML pipeline integrity value.

| Check | Value Proposition | Severity | Grep / Detection Method |
|-------|-------------------|----------|------------------------|
| OBV feature not reset at fold boundary | Catches leakage invisible to simple look-ahead checks | CRITICAL | Walk-forward fold code; OBV carries cumulative history from training into test windows |
| `regression_metrics` called from `continuous_trainer` during autonomous retraining writes to fixed path | Silent file clobbering; concurrent retrain cycles corrupt metrics | HIGH | `regression_metrics(` in `continuous_trainer.py` + hardcoded `Path("metrics")` |
| XGBoost objective is `reg:squarederror` but pipeline goal is directional classification | Objective mismatch degrades calibration; regression loss does not optimize directional accuracy | HIGH | `"objective": "reg:squarederror"` vs `direction_from_prices` label path |
| Baseline hit rate hardcoded to 0.55 in `DriftDetector.get_baseline_metrics` | Drift threshold is arbitrary and disconnected from actual model CV score | HIGH | `baseline_hit_rate = 0.55` in `drift.py` |
| `regression_metrics` called by both CLI and `ContinuousTrainer` — always writes `metrics/regression_metrics.json` | Two concurrent callers overwrite the same file | HIGH | Single hardcoded output path, multiple call sites |
| VADER sentiment `aggregate()` uses `bars.iterrows()` for O(n*m) loop with news window slicing | At 5m frequency with 1-year history this is ~100k iterations | MEDIUM | `for _, bar in bars.iterrows()` in `sentiment.py` |
| No early stopping in XGBoost training (`num_boost_round=100` fixed) | Fixed rounds risk overfitting on smaller fold windows in walk-forward | MEDIUM | `num_boost_round=100` (fixed literal, no `early_stopping_rounds`) |
| Config global state (`_ACTIVE_CONFIG`) makes test isolation fragile | Tests that call `load_config` or `get_runtime_config` share process state | MEDIUM | `global _ACTIVE_CONFIG` — confirmed 22/84 tests use mocking but isolation is per-function |
| `verbose_eval=False` deprecated XGBoost parameter | Causes warnings or is silently ignored in XGBoost >= 2.x | MEDIUM | `verbose_eval=False` in `walk_forward.py`, `optimize.py` |
| `evaluate_promotion_gate` function is 200+ lines with nested branching | High cyclomatic complexity; promotion logic is hard to reason about | MEDIUM | `evaluate_promotion_gate` + `select_champion_report` in `evaluate.py` |
| Missing test for `features/test_leakage.py` — referenced in CLAUDE.md but does not exist | Documented guardrail is absent from test suite | HIGH | `find tests/features/test_leakage.py` returns empty |
| CORS `allow_origins` permits only hardcoded localhost ports | Does not fail loudly in production; silent misconfig | MEDIUM | `allow_origins=["http://localhost:5173", "http://localhost:3000"]` |
| `pickle` used for `RandomForest` model persistence | Pickle is not version-stable; model load may fail across Python versions | MEDIUM | `import pickle` + `pickle.dump` in `model/train.py`, `model/persist.py` |

### Anti-Features (Checks That Seem Useful But Mislead)

Audit activities to explicitly avoid because they produce noise or false confidence.

| Anti-Feature | Why Requested | Why Problematic | What to Do Instead |
|--------------|---------------|-----------------|-------------------|
| 100% test coverage as a success metric | Easy to report, looks good | Coverage only measures execution, not correctness; ML pipeline can run to completion with wrong results | Require behavioral tests (directional expectations, invariance checks) and leakage guardrails |
| Linting-only audit (ruff/mypy pass = done) | Fast, automated, objective | Static analysis misses ML-specific correctness issues: leakage, objective mismatch, fold boundary errors | Combine linting with domain-specific pattern checks |
| Flag every `except Exception` as critical | Broad catches look dangerous | Some `except Exception` patterns in ingestion retry loops are intentional and safe; blanket flags create noise | Distinguish "swallowed silently" (bad) from "logged + continue" (acceptable in ingestion) from "suppressed critical path" (critical) |
| Demand sklearn Pipeline wrapping everywhere | sklearn Pipelines prevent leakage | BitBat uses XGBoost's native DMatrix API, not sklearn transformers; Pipeline wrapping is not applicable | Verify fold boundary correctness manually in the walk-forward implementation |
| Flag all hardcoded strings as magic numbers | Good principle in general | Not all literals are magic numbers; `"BTC-USD"` and `"xgb.json"` have clear domain meaning | Flag literals that control model behavior (hyperparameters, thresholds) not identifiers |

---

## Audit Checks by Category

### Category 1: Data Leakage (Beyond Simple Look-Ahead)

These are leakage patterns that are invisible to timestamp comparisons alone.

**1.1 OBV Cumulative State (CRITICAL)**
- What goes wrong: `obv()` in `features/price.py` calls `.cumsum()` on the full price series before any train/test split. When `WalkForwardValidator` creates fold windows, the OBV values in the test set carry cumulative state from the entire training history. This is global-index leakage.
- Grep: `\.cumsum()` in `src/bitbat/features/`
- Good: OBV should be reset to zero at each fold boundary, or computed only on the training window and then re-anchored for the test window.

**1.2 Global Normalization Before Splitting (MEDIUM)**
- What goes wrong: `rolling_z()` computes z-scores using rolling windows. If `dropna()` is called on the full dataset in `build_xy` (`features = features.dropna()`) before splitting, the rolling window parameters are implicitly derived from the full dataset's structure.
- Grep: `\.dropna()` in `dataset/build.py`
- Good: Drop NaN rows after splitting, or at minimum confirm that rolling windows are backward-looking only.

**1.3 Missing `test_leakage.py` Guardrail (HIGH)**
- What goes wrong: `CLAUDE.md` documents a `tests/features/test_leakage.py` with a PR-AUC guardrail. This file does not exist. The documented defense is absent.
- Grep: `find tests/ -name "test_leakage.py"` — returns empty
- Good: A test that trains on synthetic data where future leakage is injected and asserts that PR-AUC does NOT exceed a random baseline when the model is evaluated on a fresh fold.

**1.4 Sentiment `aggregate()` Uses Bar Timestamp as Exclusive Upper Bound (MEDIUM)**
- What goes wrong: The mask in `sentiment.py` is `published_utc <= end` (line 66). For a bar at timestamp T, this includes news published at exactly T. In production, bar T is the bar that *closes* at T, so news at timestamp T may be contemporaneous with the prediction, not prior to it.
- Grep: `published_utc <= end` in `features/sentiment.py`
- Good: Use `< end` (strict less than) to ensure only pre-bar news is included.

---

### Category 2: Feature Engineering Code Smells

**2.1 `iterrows()` in O(n * m) Sentiment Aggregation (HIGH)**
- What goes wrong: `sentiment.py:aggregate()` loops over every price bar with `for _, bar in bars.iterrows()` and for each bar performs a pandas mask operation over all news. At 5m frequency with 1 year of data this is ~105k iterations, each with a pandas boolean indexing operation.
- Grep: `\.iterrows()` in `src/bitbat/features/sentiment.py`
- Good: Use `pd.merge_asof` or a vectorized rolling window with `pd.Grouper` to aggregate sentiment per bar.

**2.2 MACD Spans Are Hardcoded Magic Numbers (MEDIUM)**
- What goes wrong: `macd()` in `features/price.py` uses spans `(12, 26, 9)` as literals. These are standard daily MACD parameters; at 5m frequency they are 12 5-minute bars, not 12 days. The feature has different semantics at different frequencies without any adaptation.
- Grep: `span=12`, `span=26`, `span=9` in `features/price.py`
- Good: Pass spans as parameters derived from `freq`-aware duration calculation (like `bars_for_duration` already used elsewhere).

**2.3 OBV Accumulates Across Full History (CRITICAL — see 1.1)**
Already documented above under leakage. Also a feature engineering smell: the feature depends on the start of the entire price history, making it non-stationary and sensitive to data range.

**2.4 GARCH Feature Generation Silently Skipped (MEDIUM)**
- What goes wrong: `_generate_price_features` wraps GARCH generation in a bare `except Exception` with `logger.warning`. GARCH failures are silently dropped; the dataset proceeds without that feature column. Training and inference may see different feature sets if GARCH fails intermittently.
- Grep: `except Exception:` followed by `logger.warning("GARCH` in `dataset/build.py`
- Good: Propagate GARCH failures unless `enable_garch=False` is explicitly set. Silent drops create training/inference schema divergence.

---

### Category 3: Model Training Anti-Patterns

**3.1 Objective Mismatch: Regression Loss for Directional Classification (HIGH)**
- What goes wrong: All XGBoost training uses `"objective": "reg:squarederror"`. The pipeline labels are directional (`up`/`down`/`flat`) and the actual trading signal is directional accuracy. RMSE optimization does not optimize for directional accuracy; a model can have good RMSE while being worse than random at direction.
- Grep: `"objective": "reg:squarederror"` in `src/bitbat/model/`
- Good: For a 3-class classification task, use `"objective": "multi:softprob"` with `"num_class": 3`. For a regression-then-threshold pipeline, document explicitly why RMSE optimization produces acceptable directional accuracy (no such documentation exists).

**3.2 Fixed `num_boost_round=100` Without Early Stopping (MEDIUM)**
- What goes wrong: `train.py` and `walk_forward.py` both use `num_boost_round=100` as a fixed literal. Walk-forward folds vary in size; small folds will overfit with 100 rounds while large folds may underfit. No `early_stopping_rounds` is configured.
- Grep: `num_boost_round=100` in `src/bitbat/model/`
- Good: Pass an evaluation set to `xgb.train` with `evals=[(dtest, "eval")]` and set `early_stopping_rounds=10` to let training length adapt to fold size.

**3.3 `assert isinstance` Used as Runtime Type Guard (HIGH)**
- What goes wrong: `train.py` uses `assert isinstance(model, xgb.Booster)` after a conditional branch. Python's `-O` optimization strips `assert` statements. In production with `python -O`, these become no-ops and the subsequent code is unguarded.
- Grep: `assert isinstance` in `src/bitbat/model/train.py`
- Good: Replace with `if not isinstance(model, xgb.Booster): raise TypeError(...)`.

**3.4 `pickle` for RandomForest Persistence (MEDIUM)**
- What goes wrong: `train.py` and `persist.py` use `pickle.dump` to save `RandomForestRegressor`. Pickle is not stable across Python versions or sklearn minor versions. A model pickled under Python 3.11/sklearn 1.4 may not load under 3.12/sklearn 1.5.
- Grep: `pickle.dump` in `src/bitbat/model/`
- Good: Use `joblib.dump` (sklearn's recommended persistence format) or export to ONNX for cross-version stability. At minimum, pin sklearn version in `pyproject.toml` and document the constraint.

**3.5 `verbose_eval=False` Deprecated Parameter (MEDIUM)**
- What goes wrong: XGBoost >= 2.0 deprecated `verbose_eval` in favor of `callbacks`. Passing `verbose_eval=False` to `xgb.train` produces a deprecation warning in XGBoost 2.1 (the version pinned in `pyproject.toml`).
- Grep: `verbose_eval=False` in `src/bitbat/model/`
- Good: Remove `verbose_eval` and use `callbacks=[xgb.callback.EvaluationMonitor(period=0)]` or simply do not pass evaluation sets if verbose output is not wanted.

---

### Category 4: API and UI Code Quality

**4.1 `regression_metrics()` Has File Write Side Effects (CRITICAL)**
- What goes wrong: `model/evaluate.py:regression_metrics()` unconditionally writes two files — `metrics/regression_metrics.json` and `metrics/prediction_scatter.png` — as side effects. It is called from both the CLI and `ContinuousTrainer`. Two concurrent autonomous retraining cycles (if triggered simultaneously) will overwrite the same files, producing corrupted or inconsistent metrics.
- Grep: `fig.savefig` + `metrics_path.write_text` inside `regression_metrics` in `evaluate.py`
- Good: Separate the computation from I/O. Return the metrics dict from a pure function. Let the caller decide whether and where to write. Pass an optional `output_path` parameter.

**4.2 CORS Configuration Hardcodes Localhost Only (MEDIUM)**
- What goes wrong: `api/app.py` sets `allow_origins=["http://localhost:5173", "http://localhost:3000"]`. This cannot be overridden at runtime via environment variable. Deploying via docker-compose with a custom port or domain silently breaks cross-origin requests.
- Grep: `allow_origins=\[` in `src/bitbat/api/app.py`
- Good: Read CORS origins from an environment variable (e.g., `BITBAT_CORS_ORIGINS`) with localhost defaults for development.

**4.3 CLI File Is 1802 Lines with 53 Functions (HIGH)**
- What goes wrong: `cli.py` is a 1802-line monolith. It mixes Click command definitions, business logic, file I/O, model training orchestration, and output formatting. This violates single-responsibility and makes individual commands untestable in isolation.
- Detection: `wc -l src/bitbat/cli.py` → 1802; `grep -c "def " src/bitbat/cli.py` → 53
- Good: CLI commands should be thin wrappers that delegate to service functions. Business logic belongs in domain modules (`dataset/`, `model/`, etc.), not in `cli.py`.

**4.4 Repeated `from pathlib import Path` Inside Function Bodies (LOW)**
- What goes wrong: `agent.py` imports `from pathlib import Path` at module level (line 7) and again inside `_ingest_prices()` (line 113) and `_ingest_auxiliary_data()` (line 162, 173). Imports inside function bodies are an anti-pattern: they re-execute on every function call and obscure the module's actual dependencies.
- Grep: `from pathlib import Path` inside `def ` in `autonomous/agent.py`
- Good: All imports at module top level, except for genuinely optional/heavy dependencies that should be guarded with `try/except ImportError`.

**4.5 Streamlit `iterrows()` in Timeline Rendering (MEDIUM)**
- What goes wrong: `gui/timeline.py` uses `for _, row in predictions.iterrows()` at line 271. Streamlit reruns on every user interaction; this O(n) loop runs on every rerun.
- Grep: `\.iterrows()` in `src/bitbat/gui/`
- Good: Use vectorized DataFrame operations or `.itertuples()` (faster than `iterrows`) for read-only iteration.

---

### Category 5: Configuration Management Anti-Patterns

**5.1 Module-Level Global Config State (HIGH)**
- What goes wrong: `config/loader.py` uses three module-level globals (`_ACTIVE_CONFIG`, `_ACTIVE_PATH`, `_ACTIVE_SOURCE`) managed via `global` statements. Process-level state means: (a) tests that call `load_config` or `get_runtime_config` share state across test functions unless explicitly reset; (b) parallel test execution can produce non-deterministic failures.
- Grep: `global _ACTIVE_CONFIG` in `src/bitbat/config/loader.py`
- Good: Use a class-based `ConfigRegistry` or pass config explicitly. If module-level state is required, expose a `reset_runtime_config()` function and call it in test teardown.

**5.2 Hardcoded Relative Paths Scattered Across Modules (HIGH)**
- What goes wrong: `Path("models")`, `Path("data")`, and `Path("metrics")` appear in at least 15 locations across `src/bitbat/`: `agent.py`, `autonomous/predictor.py`, `retrainer.py`, `continuous_trainer.py`, `backtest/metrics.py`, `model/evaluate.py`, `api/routes/health.py`, and others. All of these assume the process CWD is the project root. If the application is invoked from a different directory, all paths silently resolve to wrong locations.
- Grep: `Path\("models"\)\|Path\("metrics"\)` in `src/` (excluding tests)
- Good: Resolve base paths from a centralized config (`data_dir`, `models_dir`, `metrics_dir`) derived from the loaded YAML. Pass resolved paths as parameters rather than computing them inline.

**5.3 `baseline_hit_rate` Hardcoded to 0.55 (HIGH)**
- What goes wrong: `DriftDetector.get_baseline_metrics()` defaults to `baseline_hit_rate = 0.55` when no active model CV score is available. 0.55 is an arbitrary threshold with no documented basis. Drift detection fires (or doesn't fire) based on this magic number.
- Grep: `baseline_hit_rate = 0.55` in `autonomous/drift.py`
- Good: Make this configurable via `default.yaml` under `autonomous.drift_detection.baseline_hit_rate`. Document the value's origin (e.g., "coin-flip + small edge" or "historical market directional rate").

**5.4 Config Loader `get_runtime_config()` Returns Shallow Copy (MEDIUM)**
- What goes wrong: The comment in `loader.py` says "Return a shallow copy to prevent accidental mutation." A shallow copy of a nested dict (e.g., `config["autonomous"]`) still returns the same nested dict object. Callers who mutate `config["autonomous"]["drift_detection"]` are mutating the global singleton.
- Grep: `return dict(_ACTIVE_CONFIG or {})` in `loader.py`
- Good: Use `copy.deepcopy(_ACTIVE_CONFIG)` or use a frozen/immutable config representation (Pydantic model with `model_config = ConfigDict(frozen=True)`).

---

### Category 6: Testing Anti-Patterns in ML Projects

**6.1 Referenced Test Does Not Exist (CRITICAL)**
- What goes wrong: `CLAUDE.md` documents `tests/features/test_leakage.py` as a "failing PR-AUC guardrail test that signals potential leakage." This file does not exist in the test suite. The documented defense layer is missing.
- Detection: `find tests/ -name "test_leakage*"` returns nothing
- Good: Create `tests/features/test_leakage.py` that: (1) generates a synthetic dataset with injected future leakage, (2) trains a model, (3) asserts PR-AUC is NOT significantly above the no-information rate. A passing test means no detected leakage.

**6.2 Behavioral Tests Missing (HIGH)**
- What goes wrong: 84 tests exist but they are predominantly structural (schema contract enforcement, CLI invocation, ingestion format checks). There are no invariance tests (prediction should not change when irrelevant features are perturbed) and no directional expectation tests (model should predict higher returns when momentum features are strongly positive).
- Detection: `grep -r "invariance\|directional_expectation\|perturb" tests/` returns nothing
- Good: Add at least one invariance test and one directional expectation test per major feature category (price, sentiment, macro, on-chain).

**6.3 Tests Relying on Process-Level Config Singleton (MEDIUM)**
- What goes wrong: 22 of 84 tests use mocking, but the config loader global is not reset between tests by default. Tests that call `get_runtime_config()` without mocking may pick up config state left by a prior test, causing non-deterministic failures when test order changes.
- Detection: `grep -r "get_runtime_config\|load_config" tests/` — matches that don't mock the function
- Good: Add a pytest fixture that resets `_ACTIVE_CONFIG = None` before each test that calls config-dependent code.

**6.4 No End-to-End Pipeline Test (HIGH)**
- What goes wrong: There is no test that exercises the full pipeline: ingest → features → dataset build → walk-forward CV → promotion gate → prediction. The test suite covers each layer in isolation but not their composition. Pipeline-level bugs (schema mismatch across stages, feature count drift between training and inference) are not caught.
- Detection: No test file matches `test_pipeline*`, `test_e2e*`, or `test_full_*`
- Good: A single slow integration test (marked `@pytest.mark.slow`) that runs the pipeline end-to-end on a small synthetic dataset (100 bars, 2 folds) and asserts: (a) no ContractError, (b) predictions have the right schema, (c) fold count matches expectation.

**6.5 Flaky Tests From Non-Deterministic Random State (MEDIUM)**
- What goes wrong: `fit_baseline` accepts `seed: int = 42`. Tests that rely on numerical proximity of model outputs (e.g., asserting RMSE < X) may be fragile if the seed or data changes. XGBoost's walk-forward uses `self.xgb_params` which may or may not include a seed.
- Grep: `approx\|pytest.approx` in `tests/model/` — check coverage
- Good: Use `pytest.approx` with tolerances for float comparisons, not exact equality. Pin seeds explicitly in all test training calls.

---

## Feature Dependencies

```
[Hardcoded path remediation]
    └──enables──> [Testable path resolution]
                      └──enables──> [E2E pipeline test]

[Config global state refactor]
    └──enables──> [Reliable test isolation]
                      └──enables──> [Parallel test execution]

[OBV fold-boundary fix]
    └──addresses──> [Data leakage at fold boundary]

[regression_metrics side-effect removal]
    └──addresses──> [Concurrent retraining file corruption]
                    [Testability of evaluate functions]

[Leakage test creation]
    └──gates──> [Audit AUDIT-01 completion]

[Objective mismatch fix]
    └──requires──> [Label encoding update]
               └──requires──> [Inference pipeline update]
                              [Promotion gate metric update]
```

### Dependency Notes

- **OBV fix requires walk-forward fold integration:** Not a standalone feature change; requires WalkForwardValidator to slice and reset cumulative features per fold.
- **Objective mismatch fix has broad downstream impact:** Changing from regression to classification XGBoost changes the prediction output format (probabilities per class instead of a continuous return), which cascades to inference, backtest, monitoring, and API schemas. This is a high-risk change that requires phased validation.
- **Config singleton refactor is a prerequisite for safe parallel testing:** Until module-level globals are eliminated, test isolation is fragile.

---

## MVP Definition (Audit Deliverables)

### Must Find and Fix (AUDIT-04 Critical Remediation)

- [ ] `assert isinstance` in production code — replace with proper runtime guards
- [ ] `regression_metrics()` side effects — separate computation from I/O
- [ ] Missing `test_leakage.py` — create the documented guardrail
- [ ] OBV cumsum leakage at fold boundary — document and either fix or formally accept with rationale
- [ ] `baseline_hit_rate = 0.55` magic number — move to config

### Must Find, Catalog, Defer (AUDIT-01/02/03)

- [ ] All hardcoded relative paths — catalog locations, defer refactor
- [ ] CLI monolith (1802 lines) — catalog, defer decomposition
- [ ] Objective mismatch (regression vs classification) — catalog with impact analysis, defer change
- [ ] `verbose_eval=False` deprecation — catalog, trivial fix
- [ ] `pickle` for RandomForest — catalog, defer to model v2

### Informational Only (Document, Do Not Fix Now)

- [ ] MACD hardcoded spans — domain-appropriate for short-horizon, document
- [ ] CORS localhost hardcoding — acceptable for local-first app, document
- [ ] Repeated imports inside functions — cosmetic, low risk

---

## Feature Prioritization Matrix

| Audit Check | Pipeline Risk | Fix Cost | Priority |
|-------------|--------------|----------|----------|
| Missing leakage test (`test_leakage.py`) | CRITICAL | LOW | P1 |
| `regression_metrics()` I/O side effects | HIGH | LOW | P1 |
| `assert isinstance` in prod code | HIGH | LOW | P1 |
| OBV cumsum fold boundary leakage | CRITICAL | MEDIUM | P1 |
| `baseline_hit_rate = 0.55` magic constant | HIGH | LOW | P1 |
| Objective mismatch (reg vs clf) | HIGH | HIGH | P2 (catalog, assess) |
| Hardcoded relative paths | MEDIUM | MEDIUM | P2 |
| Config global state | MEDIUM | MEDIUM | P2 |
| CLI monolith | MEDIUM | HIGH | P3 |
| `iterrows()` in sentiment pipeline | MEDIUM | MEDIUM | P2 |
| `pickle` for RandomForest | MEDIUM | LOW | P2 |
| `verbose_eval=False` deprecation | LOW | LOW | P1 (trivial) |
| MACD hardcoded spans | LOW | LOW | P3 |

**Priority key:**
- P1: Fix in AUDIT-04 critical remediation sprint
- P2: Fix or formally defer with documented rationale
- P3: Catalog only; revisit in later milestone

---

## Sources

- [MLScent: Anti-pattern detection in ML projects](https://arxiv.org/html/2502.18466v1) — 76 anti-pattern detectors across ML frameworks
- [scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — data leakage, preprocessing inconsistency, randomness control
- [Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932v1) — pre-split sequence generation and temporal leakage
- [Testing Machine Learning Systems](https://eugeneyan.com/writing/testing-ml/) — behavioral, invariance, and directional expectation tests
- [FastAPI Anti-Patterns](https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/) — side effects, CORS, path hardcoding
- [Code Smells for Machine Learning Applications](https://ar5iv.labs.arxiv.org/html/2203.13746) — ML-specific smell taxonomy
- Direct codebase inspection: `src/bitbat/model/train.py`, `src/bitbat/model/evaluate.py`, `src/bitbat/features/price.py`, `src/bitbat/features/sentiment.py`, `src/bitbat/autonomous/drift.py`, `src/bitbat/autonomous/agent.py`, `src/bitbat/dataset/build.py`, `src/bitbat/config/loader.py`

---
*Feature research for: BitBat v1.5 Codebase Health Audit*
*Researched: 2026-03-04*
