# Pitfalls Research

**Domain:** Auditing and remediating a rapidly-built Python ML pipeline (BitBat v1.5)
**Researched:** 2026-03-04
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Test Suite Theater — Tests That Pass Without Validating Anything

**What goes wrong:**
The audit declares "169 tests passing" and concludes the codebase is healthy, but the test suite
contains a significant proportion of meta-tests, structural-conformance tests, and milestone-marker
tests that verify documentation/file existence rather than runtime behavior. When these inflate the
count, real behavioral gaps hide behind a misleadingly large number.

**Why it happens:**
Milestone-driven development under time pressure incentivizes shipping tests that make the gate
pass quickly. Tests like `test_phase19_gate_required_upstream_suites_exist` (which asserts only
that a list of files exists) and `test_phase8_release_required_gate_files_exist` (same pattern)
accumulate as each phase adds its own confidence-marker. Nobody deliberately writes fake tests —
they write "did we do the work" tests that structurally confirm artifacts exist, not that they
function correctly.

**Specific evidence in BitBat:**
- `tests/autonomous/test_phase19_d1_monitor_alignment_complete.py` asserts file existence and
  string presence in source code (`assert "with pytest.raises(FileNotFoundError" in source`) — it
  tests that another test file contains certain text, not that the behavior is correct.
- `make test-release` runs only a `-k "schema or monitor"` filtered subset of the D1 suites,
  meaning many tests pass silently by not running at all.
- The "169 tests" total includes gui, autonomous, and milestone-complete files that primarily
  check structural conformance (file inventory, AST parsability, page navigation regex matches).

**How to avoid:**
Before treating any "test X passes" as evidence of correctness, classify each test file by type:
(a) behavioral — inputs in, assertions on outputs; (b) structural — file/source presence checks;
(c) milestone-marker — "did we write the phase N test?" checks. Only category (a) counts as
coverage for an audit claim. When auditing, run the full `poetry run pytest` (not just
`make test-release`) and inspect what the failing tests actually assert.

**Warning signs:**
- Test file names containing `_complete`, `_phase_N_`, `_session_N_` patterns.
- Tests that `read_text()` another test file and assert string membership.
- `make test-release` filters tests with `-k` flags, leaving most tests unexercised.
- Tests that assert `result["drift_detected"] in [True, False]` — vacuously true assertions.
- Coverage number rises with each milestone even though no new behavioral paths are exercised.

**Severity triage:** CRITICAL — a false sense of test coverage is the most dangerous audit
starting position. Mis-categorizing theater tests as real coverage causes auditors to skip re-
testing paths that have never been validated.

**Phase to address:** Phase 1 of the audit — test suite classification before any remediation work
begins. Do not start fixing code until you know which test failures are real.

---

### Pitfall 2: Auditing Style While Correctness Breaks Are Hiding

**What goes wrong:**
The audit produces a long list of code smell findings (broad `except Exception`, hardcoded string
literals, missing type annotations, naming inconsistencies) and the team spends the milestone
fixing those. Meanwhile, the one correctness bug that breaks autonomous retraining at runtime
(`bitbat features build --tau` being passed to a CLI that has no `--tau` option) ships unfixed
because it was not visually prominent in the linter output.

**Why it happens:**
Static analysis tools (ruff, mypy) surface style and type issues automatically and noisily. They
are easy to triage because each finding has a file/line/rule. Correctness bugs — especially
subprocess contract mismatches, cross-module behavioral drift, and silent failure paths — require
understanding intent, not just pattern matching. Auditors follow the path of least resistance.

**Specific evidence in BitBat:**
- `src/bitbat/autonomous/retrainer.py` calls `bitbat features build --tau <value>` (line 200),
  but `src/bitbat/cli.py` `features build` command accepts only `--start` and `--end` (line 376).
  This subprocess invocation will silently fail or raise on every autonomous retraining trigger.
- `_read_cv_score()` in retrainer looks for `average_balanced_accuracy` in `cv_summary.json`, but
  the CLI CV command writes `average_rmse`/`average_mae`. The fallback chain may resolve to 0.0,
  causing every retrained model to be rejected as failing the improvement threshold.
- These correctness bugs were documented in `.planning/codebase/CONCERNS.md` as HIGH priority in
  February 2026 but remain unfixed as of v1.4.

**How to avoid:**
Structure the audit in two separate passes: (1) correctness pass — trace every subprocess
invocation, every CLI command call chain, every cross-module contract; (2) style pass — run linter,
address suppressible issues. Do not start the style pass until the correctness pass is complete and
all critical bugs are filed. Use severity triage criteria (see below) to enforce this ordering.

**Warning signs:**
- PR descriptions mention "cleaned up exception handling" and "added type hints" but no behavioral
  test changes.
- CONCERNS.md items marked HIGH remain open while LOW items get closed.
- The milestone closes with "zero linter warnings" but autonomous retraining has not been exercised
  end-to-end since the CLI change that broke it.

**Severity triage:**
- CRITICAL (fix now): subprocess command contract mismatches, silent metric key mismatches that
  control promotion/deployment decisions, leakage controls that can be bypassed.
- DEFER: exception broadness in non-critical logging paths, naming inconsistencies, missing
  docstrings, `# type: ignore` comments on optional dependencies.

**Phase to address:** Phase 1 — correctness findings must precede style remediation. File a
prioritized issue list before writing any fix code.

---

### Pitfall 3: Remediation Regressions — Breaking Working Code While "Fixing" Smells

**What goes wrong:**
An auditor sees `except Exception` in the monitoring agent and refactors it to raise on all
exception types. The monitor agent was intentionally swallowing non-critical alerting failures
(Telegram send errors) to keep the main cycle alive. After the "fix", a misconfigured Telegram
token kills every monitoring cycle.

**Why it happens:**
Rapid remediation without reading the surrounding context. Exception handling in a long-running
service loop is often intentionally broad on non-critical branches. Replacing all 73 instances of
`except Exception` in the codebase with narrow exception types without understanding each call site
introduces new failure modes.

**Specific evidence in BitBat:**
- `src/bitbat/autonomous/agent.py` has `except Exception` at lines 126, 148, 157, 168, 179 — some
  of these are in alerting and metric-write paths where continue-on-failure is architecturally
  correct (the agent should keep cycling even if Slack is down).
- The monitoring agent already has a deliberate policy: schema/model preflight failures raise
  (lines 45-66), but cycle-level alerting failures are caught and logged.
- Changing this without understanding the policy inverts the intended behavior.

**How to avoid:**
For each exception-handling site, classify the intent before touching it:
(a) intentional defensive (non-critical branch, continue on failure) — do not change;
(b) masking a real error (critical path swallowed without surfacing) — narrow and re-raise;
(c) ambiguous — add a comment and defer to the phase that owns that module.
Never refactor exception handling wholesale. Fix one site, add a test that exercises the failure
path, verify it, then proceed.

**Warning signs:**
- A PR that touches `except Exception` in more than 5 files simultaneously.
- CI passes but no new tests were added to cover the changed failure paths.
- "Fixed all broad exceptions" in the commit message.

**Severity triage:**
- CRITICAL to fix: silent swallowing on correctness paths (model deployment, label generation,
  metric writes to the DB that drive promotion decisions).
- DEFER: broad catches in optional feature paths (macro/on-chain ingestion when disabled, alerting
  channels when unconfigured).

**Phase to address:** Phase 2 — after correctness audit, per-site exception classification should
be a gate for any exception-narrowing PR.

---

### Pitfall 4: False Positive Noise Overwhelming Real Issues (Audit Findings Inflation)

**What goes wrong:**
The audit produces 200 findings. 150 are minor/deferred (naming, formatting, optional type
annotations, `# noqa` suppressions). The team triages for a week, the milestone budget is consumed,
and the 5 critical correctness findings get scheduled for "next milestone." The audit is declared
complete.

**Why it happens:**
Automated tools (ruff, mypy, bandit) are indiscriminate — they flag everything at the same
priority level. Without an explicit triage protocol before the audit runs, every finding looks like
a peer of every other finding.

**How to avoid:**
Before running any audit tooling, define the severity criteria that separate "must fix in this
milestone" from "catalog and defer":

CRITICAL (fix in v1.5):
- Any runtime path that silently returns wrong data (mislabeled predictions, wrong freq/horizon
  routing, metric key mismatches that affect promotion).
- Any subprocess or CLI contract mismatch where retrainer/ingest calls a CLI command with
  arguments that do not exist on that command.
- Any pipeline stage that can produce a dataset with future leakage without surfacing an error.

HIGH (strong preference to fix in v1.5):
- Hardcoded freq/horizon values in route defaults that contradict the "API as source of truth"
  contract established in v1.4 (e.g., `Query("1h")` on analytics/predictions/health routes while
  config default is `5m`).
- Duplicate ingestion implementations where behavioral divergence is already documented.

DEFER (catalog for future milestone):
- Style: naming, docstrings, formatting inconsistencies.
- `# type: ignore` on optional imports (praw, requests guard pattern is standard).
- Minor exception broadness in non-critical paths (alerting channels, optional feature toggles).

**Warning signs:**
- Audit report has more than 50 findings with no severity labels.
- Team is discussing lint rule configuration during audit week.
- Milestone completion percentage is measured by number of closed findings, not by severity class.

**Phase to address:** Pre-audit scoping (before Phase 1 begins) — establish the triage rubric so
work flows to real issues first.

---

### Pitfall 5: Pipeline Stage Integration Breaks — Contracts Assumed, Not Verified

**What goes wrong:**
Individual modules pass their unit tests in isolation. The end-to-end pipeline (ingest → feature
build → dataset assemble → train → batch inference → monitoring) is never run sequentially in a
test environment. Integration breaks — like the retrainer calling a CLI command with wrong flags —
survive indefinitely because unit tests mock the subprocess call.

**Why it happens:**
The pipeline has strong internal contracts (`contracts.py` enforcement, `ContractError` on schema
violation), but the boundaries between CLI commands invoked by the retrainer via subprocess are not
contractually tested. `subprocess.run()` calls are opaque to the test suite's monkeypatching.

**Specific evidence in BitBat:**
- Retrainer integration tests in `tests/autonomous/test_session3_complete.py` monkeypatch
  `agent.continuous_trainer.retrain` to return a hardcoded dict — the actual subprocess commands
  that `retrain()` runs are never exercised.
- `tests/autonomous/test_retrainer.py` existence needs verification that it exercises the
  `_run_command` paths rather than mocking them away.
- The dual ingestion implementations (batch `src/bitbat/ingest/` vs autonomous
  `src/bitbat/autonomous/`) are not tested together for output compatibility.

**How to avoid:**
Add an end-to-end pipeline integration test (marked `@pytest.mark.slow`) that exercises the full
sequence with synthetic data against real subprocess calls in a temp directory. This is the only
test that catches CLI contract drift. Separately, document which integration seams are "contract
boundaries" (where `ContractError` is the safety net) vs "convention boundaries" (where only
convention, not enforcement, keeps them aligned).

**Warning signs:**
- Retrainer tests use `monkeypatch.setattr(agent.continuous_trainer, "retrain", lambda: ...)`.
- No test file runs `poetry run bitbat features build` in a subprocess with real arguments.
- `CONCERNS.md` HIGH items are documented but have no associated failing tests.

**Phase to address:** Phase 1 (audit) — map all inter-stage seams; Phase 2 (remediation) — fix
broken contracts and add at least one integration-level regression for each fixed seam.

---

### Pitfall 6: Scope Creep During Remediation — Fixing More Than the Audit Found

**What goes wrong:**
While fixing the `--tau` CLI mismatch in the retrainer, the developer notices the `_build_cv_windows`
logic is complex and decides to refactor it. While refactoring, they notice the `AutoRetrainer`
class has too many responsibilities and extracts subclasses. By the end of the PR, the original
`--tau` fix is buried in a large refactor, and the PR introduces three new untested code paths.

**Why it happens:**
Audit milestones touch many files and create a psychological sense that "everything is being
fixed." The developer is already context-loaded and sees adjacent improvements. The "while we're
here" instinct compounds with each touch.

**How to avoid:**
Enforce the rule: one issue, one fix, one test. Every remediation PR must have a corresponding
entry in the audit findings list. Any "noticed while fixing" improvements must be filed as separate
issues, not included in the current PR. Gate reviews specifically ask: "Does this PR touch anything
not in the audit finding it references?"

**Warning signs:**
- PR description says "while fixing X, also improved Y and Z."
- Changed file count exceeds 5 files for a single audit finding fix.
- PR introduces new abstractions (new classes, new modules) that were not in the audit.

**Phase to address:** Phase 2 (remediation) — each fix PR must be tagged to an audit finding ID.

---

### Pitfall 7: Hardcoded Default Drift — The v1.4 Problem Recurring

**What goes wrong:**
v1.4 fixed the API settings endpoint to read from `default.yaml`. But the analytics, predictions,
and health routes still have `Query("1h")` and `Query("4h")` as their default parameter values
while the config default is `5m`/`30m`. This means any caller that omits the freq/horizon
parameters gets inconsistent results depending on which endpoint they hit. The v1.4 audit caught
the settings endpoint; the other routes were not in scope.

**Why it happens:**
Scope was narrowed to what was explicitly broken in v1.4 requirements (APIC-01/02, SETT-01/02/03).
The surrounding routes were "working" (not crashing) so they were not touched. Over 4 milestones,
each milestone touched different subsystems without a cross-cutting audit of the canonical value
set contract.

**Specific evidence in BitBat:**
- `src/bitbat/api/routes/predictions.py` lines 45-46: `Query("1h")`, `Query("4h")`.
- `src/bitbat/api/routes/analytics.py` lines 61-62: `Query("1h")`, `Query("4h")`.
- `src/bitbat/api/routes/health.py` line 88: `_check_model(freq="1h", horizon="4h")`.
- `src/bitbat/api/routes/metrics.py` line 76: `Path("models/1h_4h/xgb.json")` hardcoded.
- `src/bitbat/autonomous/retrainer.py` line 30: `freq="1h", horizon="4h"` in constructor.
- The `.planning/codebase/CONCERNS.md` HIGH item #2 documents this but it remained open through v1.4.

**How to avoid:**
The v1.4 architectural decision "API as single source of truth for config values" must be applied
uniformly. During the v1.5 audit, grep for every occurrence of `"1h"` and `"4h"` as string
literals in route defaults, service constructors, and path constructions. Each occurrence is either
intentional (historical data compatibility) or a drift bug. Document intent explicitly.

**Warning signs:**
- `grep -rn '"1h"\|"4h"' src/bitbat` returns hits in route Query defaults, constructor defaults,
  or path strings not in `_SUPPORTED_FREQUENCIES`.
- A route's `--freq` default disagrees with `default.yaml`'s `freq` value.
- Tests for a route pass `freq="1h"` explicitly — masking the wrong default.

**Phase to address:** Phase 1 (audit) — catalog all occurrences. Phase 2 (remediation) — fix
route defaults to read from config or at minimum agree with `default.yaml`. Flag intentional
exceptions with comments.

---

### Pitfall 8: Missing the Forest — Fixing Individual Smells While the End-to-End Flow Is Broken

**What goes wrong:**
The audit produces a clean bill of health for each module in isolation (contracts pass, features
have no leakage, model trains correctly). Nobody runs `bitbat prices ingest && bitbat features
build && bitbat model train && bitbat batch run && bitbat monitor run-once` sequentially on a
fresh clone. The end-to-end clone-to-predict-to-monitor path has never been exercised since the
v1.2 pipeline changes.

**Why it happens:**
Module-level testing is fast, hermetic, and repeatable. End-to-end testing is slow, requires real
data sources (or realistic synthetic substitutes), and touches filesystem paths that vary by
environment. Teams skip it under time pressure and rely on the sum of unit tests as a proxy.

**Specific evidence in BitBat:**
- AUDIT-02 ("End-to-end pipeline usability validation") is listed as a v1.5 requirement but has
  no existing test coverage. The test suite has no file that runs a simulated full pipeline
  sequence, even with mocked external data.
- `scripts/run_ingestion_service.py` hardcodes `1h` price interval — never caught because no
  end-to-end test exercises the ingestion service entrypoint.
- The `bitbat ingest` CLI command group is not represented in `make test-release`.

**How to avoid:**
Dedicate one audit phase specifically to end-to-end usability. Use a "fresh clone" protocol: start
from a clean state, run each step in sequence, document what fails. This is separate from the
module-level audit. A minimal E2E smoke test that synthesizes data and runs each CLI command is
acceptable for CI; the goal is to catch integration breaks, not validate ML correctness.

**Warning signs:**
- No test file exercises more than one pipeline stage in sequence.
- `scripts/` directory has entrypoints that are not exercised by any test.
- The README "Getting Started" steps have not been verified since a pipeline-changing milestone.

**Phase to address:** A dedicated E2E usability phase (AUDIT-02) run before any remediation begins,
so the remediation work is scoped to what the E2E test actually breaks.

---

### Pitfall 9: Over-Engineering Fixes — Replacing Simple Bugs with Complex Abstractions

**What goes wrong:**
The `--tau` CLI mismatch in the retrainer is fixed by introducing a new `PipelineCommandBuilder`
abstraction that dynamically constructs CLI command lists from config schema definitions. This is
more correct in principle but adds 400 lines of new code, requires new tests, and is harder to
understand than the original. It ships with three subtle bugs of its own.

**Why it happens:**
Audit milestones attract architectural ambition. Developers who have read the full codebase want to
"do it right." The fix for a specific bug becomes the foundation for a general solution that nobody
asked for.

**How to avoid:**
For every fix, ask: "What is the minimal change that makes the behavior correct and adds a test
that would have caught this?" The answer to the `--tau` mismatch is: remove `--tau` from the
subprocess call in `retrainer.py` (one line), verify the CLI accepts `--start`/`--end`, and add a
test that runs `retrainer._build_features_command()` and asserts the returned argument list matches
the CLI signature. Not a new abstraction.

**Warning signs:**
- Fix PR introduces a new class that did not exist before.
- Fix PR is larger than the bug it addresses (measured by diff lines).
- Fix description references "architectural improvement" for a bug in a single function.

**Phase to address:** Phase 2 (remediation) — fixes must be reviewed for minimal scope before
merging.

---

### Pitfall 10: Test Suite Trust Assumed from Count — Structural vs Behavioral Coverage

**What goes wrong:**
The audit starts with "we have 169 tests and they pass" as a baseline assumption of safety. This
creates false confidence that any regression introduced by a fix will be caught. In practice, many
of the 169 tests check static structure (file presence, source text patterns, AST parsability),
not runtime behavior. A fix that breaks autonomous retraining at runtime would not fail any of the
current `make test-release` tests.

**Why it happens:**
169 is a large number. It is hard to read 169 test functions to understand what they actually
validate. The instinct is to trust the count.

**How to avoid:**
Classify the 169 tests before the audit begins. Build a map:

| Test category | Example file | Validates |
|---|---|---|
| Behavioral unit | `tests/features/test_price_features.py` | actual computation results |
| Integration behavioral | `tests/autonomous/test_session3_complete.py` | agent cycle with real DB |
| Structural conformance | `tests/gui/test_phase12_supported_views_smoke.py` | file inventory, AST parse |
| Milestone-marker meta | `tests/autonomous/test_phase19_d1_monitor_alignment_complete.py` | other test files exist |
| Release gate | `tests/gui/test_phase8_release_verification_complete.py` | gate file existence |

Only behavioral unit and integration behavioral tests count as safety nets for remediation. During
the audit, identify which critical pipeline paths have zero behavioral test coverage.

**Warning signs:**
- A test file reads another test file's source and asserts string membership.
- A test fixture writes an empty `{}` JSON as a model artifact (not a real model).
- A test passes even when the module it ostensibly tests is deleted.

**Phase to address:** Phase 1 (audit) — test classification is prerequisite to remediation safety.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| subprocess calls to CLI commands as integration mechanism | Simple reuse of existing CLI logic | CLI contract drift breaks caller silently; subprocess is opaque to test monkeypatching | Never for critical paths like autonomous retraining |
| Milestone-marker tests that assert other test files exist | Rapid phase-gate closure | Inflates test count without behavioral coverage; creates false audit baseline | Only as a lightweight supplementary check, never as primary coverage |
| Hardcoded `freq="1h"` defaults in route/constructor signatures | Works for original use case | Breaks when config default changes; contradicts "API as source of truth" contract | Only with explicit comment documenting why this specific value is intentional |
| `monkeypatch.setattr` on the exact method under test | Fast, isolated unit test | Prevents detection of subprocess command contract drift; mocks the actual behavior being tested | Acceptable only for external I/O (network, filesystem), not for internal business logic |
| Broad `except Exception` in service loops | Service stays alive | Masks correctness failures in critical paths; makes root-cause diagnosis hard | Acceptable only in explicitly non-critical branches (alerting channels, optional features) |
| Duplicating ingestion logic in batch and autonomous modules | Faster per-milestone delivery | Behavioral divergence accumulates; partition schemes, dedupe logic, error handling drift | Never without an explicit shared utility for the shared logic |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Retrainer → CLI via subprocess | Pass flags that don't exist on the target CLI command | Before any subprocess call, assert the command can be constructed by importing the CLI module and inspecting its parameters |
| Retrainer → cv_summary.json | Read a key (`average_balanced_accuracy`) that the writer (`model cv`) never writes | Use a shared constant or schema dict for cv_summary keys; test that writer and reader agree |
| API routes → config default | Hardcode `Query("1h")` while `default.yaml` says `freq: 5m` | Read route defaults from `load_config()` or accept that the default is "no default — caller must provide" |
| Autonomous ingestion → batch parquet | Write to different partition schemes (`date=YYYY-MM-DD` vs flat) | Consolidate partition logic into `io/fs.py`; both ingestion paths call the same writer |
| Test fixtures → model artifact | Write `{}` as a fake `xgb.json` | Use `xgb.train()` with minimal params; a real model file prevents false positives in downstream tests |
| Monitor agent → active model lookup | Assume model exists after training; skip model artifact verification in tests | `_validate_model_preflight()` is correct — ensure every test that exercises the agent seeds a real model path |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full dataset read on every feature build | Feature build is slow; retrainer timeout hits `max_training_time_seconds` | Add incremental/windowed feature build for retraining cycles | When `train_window_days=365` with 5m bars (105,120 rows) |
| SQLite write lock contention under Docker Compose | API reads fail intermittently while monitor writes | Use WAL mode for SQLite; separate read/write connection pools | When all 4 Docker services run simultaneously |
| Walk-forward CV recomputes all windows on each retrain trigger | Retrain takes longer than the monitoring cycle interval | Cache CV window results; skip redundant windows | When `cv_window_count=3` with large datasets |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| API routes exposed without auth | Predictions and analytics readable without credentials | Accepted risk for local-first; document explicitly and add auth when deploying beyond localhost |
| `FRED_API_KEY` and alerting credentials in plain config | Key leakage in repo or logs | Ensure `default.yaml` has no real keys; verify `.gitignore` covers local config overrides |
| Subprocess calls with `S603`/`S607` noqa suppressions | Shell injection if any argument comes from user input | All subprocess argument construction in `retrainer.py` uses internal config values — acceptable; document this constraint |

---

## "Looks Done But Isn't" Checklist

- [ ] **Retrainer subprocess contract:** `bitbat features build` is called without `--tau` — verify
  that the retrainer's `_run_command` list exactly matches the CLI command's actual options.
- [ ] **CV metric key agreement:** `_read_cv_score()` reads `average_balanced_accuracy` but
  `model cv` writes `average_rmse`/`average_mae` — verify what the fallback chain actually returns
  and whether it causes every retrain to be rejected.
- [ ] **Autonomous retraining constructor default:** `AutoRetrainer(db, freq="1h", horizon="4h")`
  — verify that the monitoring agent passes its own `freq`/`horizon` rather than relying on the
  class default.
- [ ] **API route defaults:** All `Query("1h")` and `Query("4h")` in route signatures — verify
  each is either intentional (with comment) or updated to match config.
- [ ] **metrics.py hardcoded path:** `Path("models/1h_4h/xgb.json")` — verify this is
  intentional or updated to use the runtime pair.
- [ ] **End-to-end sequence:** Clone → `bitbat prices ingest` → `bitbat features build` →
  `bitbat model train` → `bitbat batch run` → `bitbat monitor run-once` — verify each step
  succeeds sequentially without manual intervention.
- [ ] **scripts/ entrypoints:** `scripts/run_ingestion_service.py` hardcodes `1h` — verify against
  config default, or document as intentional with rationale.
- [ ] **Test classification:** Run `poetry run pytest` (not `make test-release`) and verify that
  failures are in behavioral tests, not in milestone-marker tests.
- [ ] **Retraining cooldown test:** Verify that the test for `should_retrain`/cooldown logic uses
  real time-comparison logic, not a monkeypatched `is_in_cooldown` that always returns a fixed
  value.
- [ ] **Backup file:** `streamlit/app_pipeline_backup.py` identified in CONCERNS.md — verify it is
  not imported anywhere active before deleting.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Retrainer subprocess contract mismatch | LOW | Remove `--tau` from the `features build` call; add an integration test that constructs the command and asserts its argument list |
| CV metric key mismatch in promotion gate | LOW-MEDIUM | Align the key read by `_read_cv_score` to what `model cv` actually writes; add a test that writes a fixture `cv_summary.json` and asserts the score returned |
| Test suite theater discovered late | MEDIUM | Re-run full `poetry run pytest`, classify failures by type, add behavioral tests for each critical path with zero behavioral coverage |
| End-to-end pipeline breaks discovered after remediation | HIGH | Each fix PR must be followed immediately by an E2E smoke run; build a minimal synthetic-data E2E test that can run in CI |
| Remediation regression introduced | MEDIUM | Enforce: every fix adds a test that would have caught the original bug; revert the fix, add the test (watch it fail), re-apply the fix (watch it pass) |
| Over-engineered fix introducing new bugs | MEDIUM-HIGH | Revert to minimal fix; file a separate issue for the architectural improvement; keep the two concerns separated across different PRs |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Test suite theater | Phase 1 pre-work: test classification | A written map of all 169 tests by category exists; behavioral coverage gaps are documented |
| Correctness vs style ordering | Phase 1: audit scoping with severity rubric | CRITICAL findings list exists before any fix PRs are opened |
| Remediation regressions | Phase 2: one-fix-one-test policy | Every remediation PR includes a test that catches the original bug |
| Audit findings inflation | Pre-audit: severity triage protocol | Fewer than 10 CRITICAL items; remaining items are DEFERRED with rationale |
| Pipeline stage integration breaks | Phase 1 audit + Phase 2 per-seam regression | At least one integration test exercises the retrainer-to-CLI boundary with real command construction |
| Remediation scope creep | Phase 2: PR review policy | No remediation PR touches modules not referenced in the audit finding it closes |
| Hardcoded default drift | Phase 1 audit: grep for literal freq/horizon in routes | Zero `Query("1h")` defaults in routes where the runtime default is not `1h` |
| Missing end-to-end view | AUDIT-02 usability validation phase | A documented "fresh clone" run log exists with each step's success/failure status |
| Over-engineering fixes | Phase 2: minimal-fix principle | Fix PR diff is smaller than the surrounding code; no new classes introduced |
| Test count trusted over test content | Phase 1 pre-work: classification | `make test-release` is understood to be a subset filter; full suite is baselined |

---

## Sources

- `.planning/PROJECT.md` — milestone history, v1.5 audit requirements (AUDIT-01/02/03/04)
- `.planning/RETROSPECTIVE.md` — cross-milestone lessons, integration checker findings, tau tech debt discovery
- `.planning/codebase/CONCERNS.md` — HIGH priority items: retrainer CLI mismatch, freq/horizon defaults, duplicate ingestion
- `.planning/codebase/TESTING.md` — testing risks: no shared conftest, broad exception fallback behavior
- `.planning/codebase/ARCHITECTURE.md` — pipeline stage map, persistence model, notable couplings
- `src/bitbat/autonomous/retrainer.py` — subprocess command construction (lines 194-202), `_read_cv_score` key lookup
- `src/bitbat/cli.py` — `features build` command signature (line 376), no `--tau` option
- `src/bitbat/api/routes/` — hardcoded `Query("1h")` defaults in predictions, analytics, health, metrics routes
- `tests/autonomous/test_phase19_d1_monitor_alignment_complete.py` — milestone-marker test pattern
- `tests/gui/test_phase8_release_verification_complete.py` — release gate as file-existence check
- `tests/autonomous/test_session3_complete.py` — monkeypatched retrain in integration test
- `Makefile` — `make test-release` uses `-k "schema or monitor"` filter, runs only 3 of 4 test gate groups
- [ML Technical Debt patterns — Carnegie Mellon MLIP](https://mlip-cmu.github.io/book/22-technical-debt.html) — glue code, pipeline jungles, dead experimental code (MEDIUM confidence)
- [Refactoring at Scale — scope creep prevention](https://understandlegacycode.com/blog/key-points-of-refactoring-at-scale/) — time-boxing, focused fixes, regression safety nets (MEDIUM confidence)
- [ML Pipeline Testing best practices — Evidently AI](https://learn.evidentlyai.com/ml-observability-course/module-5-ml-pipelines-validation-and-testing/introduction-data-ml-pipeline-testing) — data validation, behavioral vs structural tests (MEDIUM confidence)

---
*Pitfalls research for: BitBat v1.5 codebase health audit and critical remediation*
*Researched: 2026-03-04*
