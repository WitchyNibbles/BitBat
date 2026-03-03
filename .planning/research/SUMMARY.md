# Project Research Summary

**Project:** BitBat v1.5 — Codebase Health Audit & Critical Remediation
**Domain:** Static analysis, ML pipeline integrity audit, and targeted correctness remediation for a brownfield Python ML prediction system
**Researched:** 2026-03-04
**Confidence:** HIGH

## Executive Summary

BitBat v1.5 is a codebase health audit and critical remediation milestone for a 14,400-LOC Python ML pipeline. The research makes clear this is not a feature build — it is a correctness-first forensic audit of a codebase that has accumulated significant technical debt across four previous milestones. The primary finding is that the codebase has two distinct problem classes: (1) correctness breaks that are already silently failing in production (retrainer subprocess contract mismatch, CV metric key mismatch, OBV fold-boundary leakage, absent `test_leakage.py` guardrail) and (2) architectural drift that degrades maintainability but does not cause immediate failures (duplicated price loading logic, private function imports across module boundaries, hardcoded paths, god-module `autonomous/`, dual DB access patterns). These two classes must be addressed in strict order: correctness first, architecture second, style never-in-this-milestone.

The recommended approach is a four-phase structure: (1) pre-work test classification to establish a trustworthy behavioral baseline, (2) correctness audit to surface and file all breaking bugs before any fix is written, (3) targeted remediation of critical and high-severity findings with a one-fix-one-test discipline, and (4) catalog of deferred items with documented rationale. The existing toolchain (ruff, mypy, pytest) needs targeted additions — pytest-cov for branch coverage measurement, vulture for dead code detection, import-linter for architecture boundary enforcement, and pip-audit for CVE scanning — but these are audit instrumentation, not new runtime dependencies.

The most dangerous audit risk is false confidence from the test suite. Of the approximately 169 tests in the repository, a significant fraction are milestone-marker tests (asserting file existence, checking source text of other test files) and structural conformance tests, not behavioral tests. The audit must begin by classifying every test file before treating any test passage as evidence of correctness. The retrainer's subprocess call to `bitbat features build --tau <value>` — where no `--tau` option exists on that CLI command — is the most actionable correctness bug: it means autonomous retraining has been silently broken and every drift-triggered retrain has been failing since the CLI was changed.

## Key Findings

### Recommended Stack

The audit toolchain additions are zero-runtime-impact additions to the `dev` dependency group. The existing ruff, mypy, and pytest infrastructure is well-configured and should be extended, not replaced. The key principle is to use tools already in the toolchain wherever possible (ruff C901 for complexity gating rather than xenon or radon-in-CI; mypy in strict mode for the audit pass rather than switching to pyright).

**Core audit technologies:**
- **pytest-cov + coverage.py 7.x:** Branch-level coverage measurement — identifies untested pipeline stages; `--branch` mode is critical for ML feature toggle paths
- **vulture 2.14:** Dead code detection at `--min-confidence 80` — reduces false positives from CLI callbacks and FastAPI route decorators that appear unused statically
- **import-linter 2.9:** Import architecture enforcement — the 12-module pipeline has a clear intended layering (`api, autonomous, gui` above `model, backtest, analytics` above `features, labeling, dataset` above `ingest, io, timealign, config`) that should be contractually enforced
- **pip-audit 2.10.0:** CVE scanning against the GitHub Python Advisory Database — free, PyPA-maintained, no paid tier needed
- **ruff C901 + PLR rules (extend existing config):** Complexity gating in CI via `max-complexity = 10`; zero new dependencies since ruff already runs in CI
- **radon 6.0.1:** One-time audit report for cyclomatic complexity and Maintainability Index; not added to CI (ruff C901 covers ongoing enforcement)

**Tools explicitly rejected:** bandit standalone (ruff S rules already cover it), pylint (noise-to-signal ratio too high on an existing ruff codebase), pyright (switching type checkers mid-audit floods output with unrelated errors), SonarQube (operational overhead exceeds signal value for 14,400 LOC).

### Expected Features (Audit Deliverables)

The "features" of this milestone are the audit checks and fixes themselves, organized by must-fix vs. must-catalog vs. informational-only.

**Must fix in v1.5 critical remediation (AUDIT-04):**
- Missing `tests/features/test_leakage.py` — documented guardrail does not exist; create it
- `regression_metrics()` has file write side effects — separating computation from I/O is required to prevent concurrent retraining file corruption
- `assert isinstance` in production model code — `assert` is stripped by `python -O`; replace with `if not isinstance: raise TypeError`
- OBV cumsum fold-boundary leakage — OBV carries cumulative state across walk-forward folds; must document, fix, or formally accept with rationale
- `baseline_hit_rate = 0.55` hardcoded magic constant in drift detector — move to config

**Must catalog with documented rationale, defer fix:**
- Retrainer subprocess contract mismatch (`--tau` flag that does not exist on the CLI) — CRITICAL correctness bug, HIGH fix priority
- CV metric key mismatch (`_read_cv_score` reads `average_balanced_accuracy` but `model cv` writes `average_rmse`) — CRITICAL correctness bug
- Hardcoded `freq="1h"` / `horizon="4h"` defaults in API routes and service constructors while config default is `5m/30m`
- XGBoost objective `reg:squarederror` for a directional classification task — HIGH impact, HIGH change cost; requires cascading updates to inference, backtest, monitoring, API schemas
- Hardcoded relative paths (`Path("models")`, `Path("metrics")`) scattered across 15+ call sites — MEDIUM severity, MEDIUM fix cost
- Config global state (`_ACTIVE_CONFIG`) breaking test isolation — MEDIUM severity, MEDIUM fix cost
- CLI monolith `cli.py` at 1802 lines with 53 functions — MEDIUM severity, HIGH fix cost; defer decomposition

**Informational only (document, do not fix now):**
- MACD hardcoded spans `(12, 26, 9)` — domain-appropriate semantics, document the frequency-adaptation concern
- CORS localhost hardcoding — acceptable for local-first app, document explicitly
- Duplicate `from pathlib import Path` imports inside function bodies in `agent.py` — cosmetic, low risk

**Anti-features to avoid:** treating 100% test coverage as success (coverage measures execution, not correctness), linting-only audit (static analysis misses ML-specific correctness issues), demanding sklearn Pipeline wrapping (BitBat uses XGBoost's native DMatrix API), flagging every broad exception as critical without reading the surrounding context.

### Architecture Approach

The codebase structure is logically sound — the module boundaries are correct in principle. The problems are boundary violations within the existing structure, not structural misdesign. The audit and remediation should repair boundary violations without restructuring the overall layout.

**Major components and their current state:**

1. **`autonomous/` package** — god module containing ingestion, training, inference, monitoring, orchestration, and DB access; should own only monitoring loop logic; ingestion belongs in `ingest/`, training belongs in `model/`
2. **`dataset/build.py:_generate_price_features`** — private function imported by 3 external callers (`cli.py`, `autonomous/predictor.py`, `autonomous/continuous_trainer.py`); should be promoted to a public API in `features/pipeline.py`
3. **`config/loader.py`** — process-singleton pattern with module-level globals; correct for single-process use but breaks test isolation; needs `reset_runtime_config()` and `with_config()` context manager for tests
4. **Price loading logic** — duplicated in 3 separate implementations (`cli.py`, `autonomous/predictor.py`, `autonomous/continuous_trainer.py`) with divergent behavior; consolidate into `ingest/prices.py:load_prices(data_dir, freq)`
5. **`api/routes/system.py` → `gui/widgets.py` import** — layering violation; API layer depends on GUI layer; shared DB utilities should live in `autonomous/db.py` or `io/sqlite.py`
6. **Dual DB access** — `autonomous/db.py` uses SQLAlchemy ORM while `api/routes/system.py`, `gui/widgets.py`, and `gui/timeline.py` use raw `sqlite3`; ORM schema and raw queries can silently drift

**Key patterns to preserve:**
- Contract enforcement at pipeline stage boundaries via `contracts.py` — the codebase's most important correctness mechanism; must be extended to macro/on-chain Parquet (currently uncovered)
- `{freq}_{horizon}` compound path key — consistent and effective; centralize construction in a shared helper rather than duplicating the string interpolation in 6+ places
- Walk-forward CV with embargo bars — the leakage prevention mechanism is architecturally correct; the OBV fold-boundary issue is a gap in its implementation, not a flaw in the design

### Critical Pitfalls

1. **Test suite theater** — A significant portion of the ~169 tests are milestone-marker tests (asserting file existence, checking source text of other test files) and structural conformance tests, not behavioral tests. `make test-release` runs only a `-k "schema or monitor"` filtered subset. Before any remediation begins, classify every test file as behavioral, structural, or milestone-marker. Only behavioral tests count as safety nets.

2. **Correctness-before-style ordering** — The retrainer's `--tau` CLI mismatch and CV metric key mismatch are CRITICAL correctness bugs documented in `CONCERNS.md` but still open as of v1.4. Static analysis tools (ruff, mypy) surface style and type issues automatically and noisily; correctness bugs require reading intent, not pattern matching. Structure the audit as two separate passes: correctness first, style second. Do not open fix PRs until the correctness findings list is complete.

3. **Remediation regressions from broad exception narrowing** — `autonomous/agent.py` contains `except Exception` blocks in alerting and metric-write paths where continue-on-failure is architecturally intentional (the agent should keep cycling even if Slack is down). Replacing these wholesale without reading context inverts the intended behavior. Classify each exception-handling site before touching it.

4. **Scope creep during remediation** — Enforce one-issue, one-fix, one-test per PR. The `--tau` mismatch fix is removing one argument from a subprocess call and adding a test that asserts the argument list matches the CLI signature — not introducing a `PipelineCommandBuilder` abstraction. Any "noticed while fixing" improvements must be filed as separate issues.

5. **Missing end-to-end view** — No test exercises more than one pipeline stage in sequence. AUDIT-02 (end-to-end pipeline usability validation) must run before remediation is declared complete. The full sequence `bitbat prices ingest → features build → model train → batch run → monitor run-once` has not been verified since v1.2 pipeline changes.

## Implications for Roadmap

Based on combined research, the recommended phase structure for v1.5 is four phases, not a feature-delivery sequence.

### Phase 1: Audit Baseline — Test Classification and Correctness Mapping

**Rationale:** The most dangerous starting position is false confidence. Before any code is changed, establish what the test suite actually validates (behavioral vs. structural vs. milestone-marker) and produce a complete inventory of correctness breaks vs. architecture drift vs. style issues. This is the prerequisite that all subsequent work depends on.

**Delivers:** (a) classified test inventory with behavioral coverage gaps identified; (b) prioritized audit findings list with CRITICAL/HIGH/MEDIUM/DEFER severity; (c) end-to-end pipeline smoke run log showing which sequential steps pass and which fail; (d) specific documentation of the retrainer subprocess contract mismatch, CV metric key mismatch, and hardcoded freq/horizon default drift.

**Addresses:** AUDIT-01 (static analysis), AUDIT-02 (E2E usability validation), and the pre-work needed for AUDIT-03/04.

**Avoids:** Pitfall 1 (test theater), Pitfall 2 (correctness vs. style ordering), Pitfall 4 (findings inflation without severity rubric).

**Tools to use:** ruff extended config (C901, PLR), pytest-cov with `--branch`, vulture at 80% confidence, import-linter, pip-audit, radon for one-time report, mypy in `--strict` mode captured to file (not in CI).

### Phase 2: Critical Correctness Remediation

**Rationale:** Fix only what is silently broken in production. Every fix must have a test that would have caught the original bug. Fixes are scoped to the minimal change that makes behavior correct — no new abstractions, no refactors.

**Delivers:** (a) retrainer subprocess contract fixed (`--tau` argument removed, CLI argument list asserted in test); (b) CV metric key agreement between writer and reader; (c) `regression_metrics()` computation separated from I/O; (d) `assert isinstance` replaced with proper runtime guards; (e) `baseline_hit_rate = 0.55` moved to config; (f) missing `tests/features/test_leakage.py` created; (g) API route `freq`/`horizon` default drift corrected to match `default.yaml` or documented with explicit rationale.

**Addresses:** AUDIT-04 (critical remediation), the CONCERNS.md HIGH items that survived v1.4.

**Avoids:** Pitfall 3 (remediation regressions), Pitfall 6 (scope creep), Pitfall 9 (over-engineering fixes).

**Constraint:** OBV fold-boundary leakage is included here for decision, but if the fix scope is large (requires WalkForwardValidator integration), it may be formally accepted with documented rationale rather than fixed in this phase.

### Phase 3: Architecture Debt Catalog and Targeted High-Value Fixes

**Rationale:** After correctness is restored, address the highest-cost architecture violations that actively impede future development. The catalog from Phase 1 drives scope; only items where fix cost is justified by ongoing friction should be fixed here.

**Delivers:** (a) `_generate_price_features` promoted to public API in `features/pipeline.py` and private import removed from 3 callers; (b) price loading logic consolidated into `ingest/prices.py:load_prices()`; (c) `reset_runtime_config()` added to config loader for test isolation; (d) `get_runtime_config() or load_config()` dead-code fallback removed from 8 call sites; (e) `api/routes/system.py` → `gui/widgets.py` cross-layer import eliminated; (f) macro/on-chain contracts added to `contracts.py`; (g) catalog document of remaining deferred items (CLI monolith decomposition, full path centralization, dual DB access unification, XGBoost objective mismatch) with rationale for deferral.

**Addresses:** AUDIT-03 (architecture and code quality findings).

**Avoids:** Pitfall 6 (scope creep) — the CLI monolith decomposition, full path centralization, and XGBoost objective change are explicitly deferred with documented rationale, not fixed here.

### Phase 4: Verification and Guardrail Hardening

**Rationale:** Prevent recurrence. The audit findings become CI gates and documented architecture contracts that outlast this milestone.

**Delivers:** (a) import-linter contracts committed to `.importlinter` and added to CI; (b) pip-audit added to CI dependency vulnerability gate; (c) ruff C901 complexity rule added to CI with `max-complexity = 10`; (d) at least one end-to-end behavioral integration test (marked `@pytest.mark.slow`) that exercises ingest → features → train → predict sequentially with synthetic data; (e) deferred findings documented in `.planning/codebase/CONCERNS.md` with explicit "not fixing in v1.5" rationale so they do not resurface as surprises in v1.6.

**Addresses:** Ongoing prevention of Pitfall 5 (pipeline stage integration breaks), Pitfall 7 (hardcoded default drift), Pitfall 8 (missing end-to-end view).

### Phase Ordering Rationale

- **Classify before fixing:** The test classification (Phase 1) must precede all fix work because the test suite is not a reliable safety net until its behavioral coverage is understood.
- **Correctness before architecture:** The CRITICAL correctness bugs (retrainer mismatch, CV key mismatch) are low fix-cost but have been open for multiple milestones. They must be closed before architecture work begins; otherwise the audit declares success while the codebase is still silently broken in its autonomous path.
- **Architecture before guardrails:** The import-linter contracts (Phase 4) must be written against the corrected architecture (Phase 3), not the current state. Defining contracts against the current violation-ridden state would codify the violations.
- **No style phase:** Style findings (naming, docstrings, formatting) are deliberately excluded from this roadmap. They create audit noise and consume milestone budget that should go to correctness and behavioral coverage.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 2 (OBV fold-boundary fix):** The fix requires integrating with `WalkForwardValidator` to reset cumulative state at fold boundaries. The interaction between `features/price.py:obv()`, `dataset/splits.py`, and the walk-forward fold windowing needs architectural mapping before fix scope can be committed.
- **Phase 3 (XGBoost objective mismatch — decision only):** The research identifies that changing from `reg:squarederror` to `multi:softprob` has cascading downstream impact on inference output format, backtest input, monitoring thresholds, and API prediction schemas. If this is included in v1.5 scope, it needs a dedicated impact analysis pass first. Current recommendation is to catalog and defer.

Phases with standard patterns (skip research-phase):

- **Phase 1 (audit tooling setup):** All tools are well-documented with clear installation and usage patterns. ruff, pytest-cov, vulture, import-linter, and pip-audit have stable APIs and pyproject.toml configuration.
- **Phase 2 (retrainer subprocess fix, metric key fix, assert replacement, regression_metrics I/O separation):** All are small, targeted, well-understood fixes. No architectural research needed.
- **Phase 4 (CI gate additions):** Standard CI integration patterns for ruff, pip-audit, and import-linter; no research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All audit tools verified via PyPI with confirmed versions; existing toolchain analysis based on direct inspection of `pyproject.toml` and source files |
| Features | HIGH | Audit findings derived from direct codebase inspection with grep-verifiable patterns; backed by MLScent academic research and scikit-learn official pitfall documentation |
| Architecture | HIGH | All architectural findings based on direct source inspection with specific line references; no inference or speculation |
| Pitfalls | HIGH | Pitfalls are grounded in specific, verifiable evidence from the codebase (named files, line numbers, exact CLI argument mismatches); cross-referenced against `.planning/codebase/CONCERNS.md` and `.planning/RETROSPECTIVE.md` |

**Overall confidence:** HIGH

### Gaps to Address

- **OBV fold-boundary leakage fix scope:** Research identifies the problem clearly but the fix requires understanding how `WalkForwardValidator` creates fold windows and whether resetting cumulative features is feasible without rewriting the fold logic. This needs a targeted code-reading session at Phase 2 planning time.
- **XGBoost objective mismatch downstream impact:** Research identifies the problem and flags it as high-change-cost, but the full cascade (inference output format change, backtest input shape, monitoring threshold recalibration, API schema update) has not been mapped. If this is included in v1.5, it needs a full impact analysis before any code is touched.
- **Test classification count:** Research estimates the 169-test suite contains a significant proportion of non-behavioral tests but does not provide an exact breakdown. Phase 1 begins with running `poetry run pytest` (not `make test-release`) and building a complete classification map.
- **`autonomous.db` migration state in production:** The dual DB access pattern (ORM + raw sqlite3) means the ORM schema and raw queries may have diverged in any deployed instance. Phase 3 architecture work must verify this before eliminating raw sqlite3 access.

## Sources

### Primary (HIGH confidence)

- Direct codebase inspection: all files under `src/bitbat/` (2026-03-04) — architecture findings, anti-pattern identification, specific line references
- `.planning/codebase/CONCERNS.md` — documented HIGH priority items including retrainer CLI mismatch, freq/horizon default drift, duplicate ingestion implementations
- `.planning/RETROSPECTIVE.md` — cross-milestone lessons, integration checker findings, tau tech debt discovery
- PyPI package registry — vulture 2.14, import-linter 2.9, pip-audit 2.10.0, pytest-cov 7.0.0, coverage.py 7.13.4 (all versions verified)
- Ruff 0.15.4 release notes — C901 and PLR rule documentation (astral.sh/blog, 2026-02-26)
- mypy 1.19.1 official documentation — strict mode guidance (existing_code.html)

### Secondary (MEDIUM confidence)

- [MLScent: Anti-pattern detection in ML projects](https://arxiv.org/html/2502.18466v1) — 76 anti-pattern detectors across ML frameworks; informed the feature engineering smell taxonomy
- [Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932v1) — pre-split sequence generation and temporal leakage patterns
- [Testing Machine Learning Systems — Eugene Yan](https://eugeneyan.com/writing/testing-ml/) — behavioral, invariance, and directional expectation test patterns
- [FastAPI Anti-Patterns](https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/) — side effects, CORS, path hardcoding patterns
- [scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — data leakage, preprocessing inconsistency, randomness control
- David Seddon — import-linter `acyclic_siblings` pattern (2025, single source)
- ML Technical Debt patterns — Carnegie Mellon MLIP — glue code, pipeline jungles, dead experimental code

### Tertiary (LOW confidence)

- None used for decision-critical claims in this research

---
*Research completed: 2026-03-04*
*Ready for roadmap: yes*
