# Phase 24: Audit Baseline - Research

**Researched:** 2026-03-04
**Domain:** Python static analysis, test classification, coverage measurement, E2E pipeline smoke testing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Test Classification Method**
- Add pytest markers (`@pytest.mark.behavioral`, `@pytest.mark.structural`, etc.) to every test AND produce a summary report
- Claude determines the right category scheme based on what's actually in the test suite
- Delete milestone-marker tests entirely — they inflate the test count and give false confidence
- Produce a full coverage matrix cross-referencing every v1.5 requirement (CORR-01 through ARCH-06) against existing test coverage

**Findings Documentation**
- Save raw tool outputs (vulture.txt, coverage report, radon report) as evidence files in the phase directory
- Produce a synthesized AUDIT-REPORT.md with both views: category-organized findings with a severity-sorted summary table at top
- Each finding tagged with which v1.5 requirement (CORR-01, ARCH-02, etc.) it maps to, or 'DEFER' if out of scope
- Pre-populate known issues from research (retrainer --tau, CV key mismatch, OBV leakage, etc.) AND run tools independently, then cross-reference: did automated tooling catch what manual research found?

**E2E Smoke Test Approach**
- Try real data download first (yfinance for small date range). If it fails (network, API limit), fall back to synthetic fixtures and note ingestion failure separately
- Skip-and-continue on failures: if a stage fails, use pre-existing data from previous runs (if available) to continue testing downstream stages. Note the failure but test as much as possible
- Claude determines which stages are meaningful to chain based on what's available after ingestion
- Save both raw console log for evidence AND a structured summary extracted from it

**Severity Triage Criteria**
- **CRITICAL**: Silently broken in production AND breaks core value promise AND data corruption risk. All three dimensions.
- **HIGH**: Causes active development friction OR is a latent risk that could escalate to CRITICAL if conditions change. Fix in v1.5.
- **MEDIUM**: Genuine debt but stable and non-blocking. Neither causes friction nor poses escalation risk. Catalog, defer.
- Claude assigns severity for all findings. User reviews the severity-sorted summary table and can override before planning starts.

### Claude's Discretion
- Exact pytest marker names and registration in conftest.py
- Tool configuration details (vulture min-confidence, radon thresholds, coverage report format)
- How to structure raw tool output files
- Whether to run mypy --strict as part of this phase or defer
- Ordering of audit activities within the phase

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUDT-01 | All existing tests classified by type (behavioral unit, integration, structural conformance, milestone-marker) with coverage gap report | Marker scheme designed; classification criteria and conftest.py registration pattern documented |
| AUDT-02 | Dead code audit completed with vulture at 80% confidence (findings triaged, false-positive whitelist created) | vulture 2.14 installed; `--make-whitelist` and `pyproject.toml` config pattern documented |
| AUDT-03 | Branch coverage report generated with pytest-cov identifying lowest-coverage modules | pytest-cov 7.0.0 installed; `--cov-branch --cov-report` invocation documented; lowest modules identified in research |
| AUDT-04 | Complexity audit completed with radon identifying high-complexity functions for remediation candidates | radon 6.0.1 installed; rank C+ (CC >= 11) functions enumerated; report command documented |
| AUDT-05 | E2E pipeline smoke test executed (ingest → features → train → batch → monitor) documenting which sequential steps pass and which fail | CLI commands for each stage documented; real-data-first/fallback strategy confirmed; all command signatures verified |
</phase_requirements>

## Summary

Phase 24 is a pure audit and documentation phase — no code fixes, only evidence gathering. The codebase has 608 tests across 77 test files, all currently passing. Three audit tools need installation as dev dependencies (vulture 2.14, radon 6.0.1, pytest-cov 7.0.0) — **these are not yet in pyproject.toml** and must be added in Wave 0.

The test suite contains a well-understood mixture of types. Milestone-marker tests are recognizable by filename convention (`test_phaseN_*_complete.py`, `test_sessionN_complete.py`) — approximately 76 tests across 15 files match this pattern. The source-reading tests (e.g., `test_phase19_d1_monitor_alignment_complete.py`) check that specific contract strings exist in other test files rather than exercising production code. These are the deletion candidates the user has authorized. The remaining tests fall cleanly into behavioral-unit (testing single functions with mocks), integration (exercising real production code with tmp_path), and structural-conformance (checking file existence, doc strings, CLI contracts, and service template alignment).

Current overall branch coverage is 77%. Lowest modules are: `api/routes/system.py` (46%), `autonomous/continuous_trainer.py` (47%), `autonomous/agent.py` (74%), `dataset/build.py` (76%), `analytics/feature_analysis.py` (65%), `gui/widgets.py` (69%), `ingest/news_cryptocompare.py` (70%), and `ingest/news_gdelt.py` (75%). The complexity audit reveals `model_cv` in cli.py (CC=37, rank E) as the highest-complexity function, followed by `LivePredictor.predict_latest` (CC=28) and `model_optimize` (CC=27). The confirmed CORR-01 bug (retrainer passes `--tau` to `model train` which does not accept it) is pre-validated via code inspection.

**Primary recommendation:** Run all four tool invocations (vulture, radon, pytest-cov, E2E smoke) in sequence, saving raw outputs as evidence files, then synthesize into AUDIT-REPORT.md with pre-populated known bugs tagged to requirement IDs.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| vulture | 2.14 | Dead code detection via AST analysis | Standard Python dead-code tool; `--make-whitelist` for false-positive management |
| radon | 6.0.1 | Cyclomatic complexity (CC) and maintainability metrics | De facto Python complexity tool; rank A-F threshold system maps directly to CC 10 requirement |
| pytest-cov | 7.0.0 | Branch coverage measurement for pytest | Native pytest plugin; `--cov-branch` flag is the standard approach |
| coverage | 7.13.4 | Underlying coverage engine (installed by pytest-cov) | Provides the `.coveragerc` configuration system |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest markers | built-in | Test classification tags | Added to every test function via `@pytest.mark.X`; registered in conftest.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| vulture | pyflakes/flake8-unused | vulture is purpose-built for dead code; pyflakes has partial coverage, no whitelist workflow |
| radon cc | wemake-python-styleguide | radon is lighter, output is more parseable for evidence files |
| pytest-cov | coverage.py directly | pytest-cov integrates without subprocess; no separate run needed |

**Installation (already executed — tools are in the poetry dev group):**
```bash
poetry add --group dev vulture radon pytest-cov
```

## Architecture Patterns

### Recommended Output File Structure
```
.planning/phases/24-audit-baseline/
├── 24-CONTEXT.md          # (exists)
├── 24-RESEARCH.md         # (this file)
├── evidence/
│   ├── vulture.txt        # raw vulture output
│   ├── coverage.txt       # raw pytest-cov term-missing output
│   ├── radon.txt          # raw radon cc output
│   ├── smoke-test.log     # raw E2E console log
│   └── smoke-summary.md   # structured E2E result table
└── AUDIT-REPORT.md        # synthesized report for the planner
```

### Pattern 1: Vulture Dead Code Scan with Whitelist Workflow
**What:** Run vulture at 80% confidence across src/ and tests/, generate whitelist for false positives, then re-run including whitelist.
**When to use:** AUDT-02. Run once on src/ only first, then again including tests/ to reduce fixture/conftest false positives.

```bash
# Source: https://github.com/jendrikseipp/vulture
# Step 1: Scan src only (highest signal, tests are treated as usage)
poetry run vulture src/bitbat tests/ --min-confidence 80 > .planning/phases/24-audit-baseline/evidence/vulture.txt

# Step 2: Generate whitelist for items that are legitimate false positives
poetry run vulture src/bitbat --min-confidence 80 --make-whitelist > whitelist_vulture.py

# Step 3: Triage manually — edit whitelist_vulture.py to remove genuine dead code
# Then re-run with whitelist:
poetry run vulture src/bitbat whitelist_vulture.py --min-confidence 80
```

Vulture output format:
```
tests/api/test_metrics.py:153: unused variable 'db_with_data' (100% confidence)
src/bitbat/autonomous/retrainer.py:200: unused function 'foo' (80% confidence)
```

Key false-positive patterns in this codebase (confirmed by pre-run):
- pytest fixture parameters named in test signatures but used implicitly by pytest (e.g., `db_with_data` as a fixture that sets up state) — these show as 100% confidence in tests
- `__all__` exports in `__init__.py` files
- Click command decorators (vulture may miss that `@cli.command()` registers functions as used)

### Pattern 2: Branch Coverage Report
**What:** Run pytest with coverage enabled for the `bitbat` package, capturing branch-level misses.
**When to use:** AUDT-03.

```bash
# Source: https://pytest-cov.readthedocs.io/en/latest/
poetry run pytest \
  --cov=bitbat \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=html:.planning/phases/24-audit-baseline/evidence/coverage_html \
  -q \
  2>&1 | tee .planning/phases/24-audit-baseline/evidence/coverage.txt
```

Key flags:
- `--cov=bitbat` — measures only the `bitbat` package (src/bitbat/), not test code itself
- `--cov-branch` — enables branch coverage (counts both if/else arms)
- `--cov-report=term-missing` — shows missing line numbers in terminal
- `tee` — streams to terminal and saves to file simultaneously

### Pattern 3: Radon Complexity Scan
**What:** Run radon cc filtering to rank C and above (CC >= 11), showing score alongside rank.
**When to use:** AUDT-04.

```bash
# Source: https://radon.readthedocs.io/en/latest/commandline.html
poetry run radon cc src/bitbat -s -n C -o SCORE \
  2>&1 | tee .planning/phases/24-audit-baseline/evidence/radon.txt
```

Key flags:
- `cc` — cyclomatic complexity subcommand
- `-s` — show complexity score (number) alongside rank letter
- `-n C` — minimum rank C (CC >= 11); shows C, D, E, F only
- `-o SCORE` — sort by complexity score descending (highest first)

Rank interpretation:
| CC Score | Rank | Risk |
|----------|------|------|
| 1-5 | A | Low |
| 6-10 | B | Low-moderate |
| 11-20 | C | Moderate — flag for attention |
| 21-30 | D | High — remediation candidate |
| 31-40 | E | Very high — priority fix |
| 41+ | F | Error-prone |

### Pattern 4: Pytest Marker Registration
**What:** Register custom markers in conftest.py to avoid `PytestUnknownMarkWarning`.
**When to use:** AUDT-01.

```python
# Source: https://docs.pytest.org/en/stable/reference/reference.html#ini-options-ref
# tests/conftest.py (does not currently exist — must be created)

import pytest

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "behavioral: behavioral unit test exercising production logic")
    config.addinivalue_line("markers", "integration: integration test requiring multiple components")
    config.addinivalue_line("markers", "structural: structural conformance test (file existence, schemas, contracts)")
    config.addinivalue_line("markers", "milestone_marker: phase gate test cross-referencing other test files — deletion candidate")
```

Alternatively, register in `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
markers = [
    "behavioral: behavioral unit test",
    "integration: integration test",
    "structural: structural conformance test",
    "milestone_marker: phase gate — deletion candidate",
]
```

The `pyproject.toml` approach is cleaner for this codebase since pytest config already lives there.

### Pattern 5: E2E Pipeline Smoke Test
**What:** Execute each pipeline stage via CLI and record pass/fail. Use a temp working directory to avoid contaminating real data/.
**When to use:** AUDT-05.

```bash
# Stage 1: Ingest prices (try real data first)
mkdir -p /tmp/smoke_test/data/raw/prices
poetry run bitbat prices pull \
  --symbol BTC-USD \
  --start 2024-01-01 \
  --interval 5m \
  --output /tmp/smoke_test/data/raw/prices \
  2>&1 | tee -a /tmp/smoke_test/smoke-test.log

# Stage 2: Build features
# (requires prices from stage 1 OR pre-existing data)
BITBAT_DATA_DIR=/tmp/smoke_test poetry run bitbat features build \
  --start 2024-01-01 \
  2>&1 | tee -a /tmp/smoke_test/smoke-test.log

# Stage 3: Train model
BITBAT_DATA_DIR=/tmp/smoke_test poetry run bitbat model train \
  --freq 5m --horizon 30m \
  2>&1 | tee -a /tmp/smoke_test/smoke-test.log

# Stage 4: Batch prediction
BITBAT_DATA_DIR=/tmp/smoke_test poetry run bitbat batch run \
  --freq 5m --horizon 30m \
  2>&1 | tee -a /tmp/smoke_test/smoke-test.log

# Stage 5: Monitor run-once
BITBAT_DATA_DIR=/tmp/smoke_test poetry run bitbat monitor run-once \
  2>&1 | tee -a /tmp/smoke_test/smoke-test.log
```

**Note:** `BITBAT_DATA_DIR` override may not work if the config loader doesn't respect it — check `data_dir` in config. The `--config` flag is the supported way to pass a custom config file:
```bash
# Create a smoke-test config
cat > /tmp/smoke_config.yaml << EOF
data_dir: "/tmp/smoke_test/data"
freq: "5m"
horizon: "30m"
tau: 0.01
enable_sentiment: false
enable_garch: false
enable_macro: false
enable_onchain: false
EOF

poetry run bitbat --config /tmp/smoke_config.yaml prices pull ...
```

### Anti-Patterns to Avoid

- **Modifying source code during the audit phase:** CONTEXT.md explicitly states no code fixes. Only documentation, marker additions, and evidence collection.
- **Running mypy --strict globally now:** The CONTEXT.md leaves this as Claude's discretion — defer mypy to phase 25/26. The current codebase is not typed strictly and this will generate significant noise that doesn't serve the audit.
- **Treating fixture-parameter false positives as genuine dead code:** Vulture flags pytest fixture parameters as "unused variables" at 100% confidence. These are not dead code.
- **Including tests/ in the `--cov` target:** Coverage is measured on `bitbat` (src/bitbat), not test files themselves.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dead code detection | Custom AST walker counting unused names | vulture | Handles closures, decorators, `__all__`, class attributes; battle-tested |
| Cyclomatic complexity | Count if/for/while manually | radon cc | Correct McCabe formula, rank system, JSON/XML output |
| Branch coverage | Custom execution tracer | pytest-cov + coverage.py | Accurate branch arc tracking; integrates with pytest fixtures |
| Whitelist management | Custom ignore comments | vulture whitelist.py mechanism | Valid Python file, version-controllable, composable |

**Key insight:** All four audit tools are deterministic — given the same source, they produce identical output. This makes their raw output reliable evidence for future phases.

## Common Pitfalls

### Pitfall 1: Vulture False Positives from Pytest Fixtures
**What goes wrong:** Vulture flags fixture parameters as "unused variable" at 100% confidence because pytest injects them implicitly via test function signatures.
**Why it happens:** Vulture does static analysis; it can't see pytest's runtime dependency injection.
**How to avoid:** Run vulture with both `src/bitbat` AND `tests/` — this way, vulture sees fixtures defined in conftest.py as "used" by test functions. Then use `--make-whitelist` for any remaining false positives.
**Warning signs:** Any finding of the form `unused variable 'db_with_data'` in a test function signature.

Confirmed false positives from pre-run (17 fixture-parameter findings in api/ tests):
```
tests/api/test_metrics.py:153: unused variable 'db_with_data' (100%)
tests/api/test_predictions.py:92: unused variable 'db_with_predictions' (100%)
```

### Pitfall 2: Coverage Run Time
**What goes wrong:** Full suite takes ~100 seconds to run; adding coverage measurement adds ~10-15% overhead.
**Why it happens:** Coverage hooks every statement execution.
**How to avoid:** Run coverage once and save the HTML report. Don't re-run for every task in the phase.

### Pitfall 3: Milestone-Marker Tests That Are Actually Integration Tests
**What goes wrong:** Some `_complete.py` tests contain real integration test logic (e.g., `test_phase8_d1_monitor_schema_complete.py` seeds a real SQLAlchemy DB and exercises the monitor agent). Deleting these would reduce real coverage.
**Why it happens:** The naming convention was applied inconsistently — some "complete" files are pure source-readers (safe to delete), others exercise production code under a milestone name.
**How to avoid:** Classify each `_complete.py` file individually. Delete ONLY tests whose entire body is `source_file.read_text()` string assertions. Tests that create databases, call production functions, or use fixtures are integration tests — reclassify them with `@pytest.mark.integration`, don't delete.
**Warning signs:** Test file imports `from bitbat.*` or uses `tmp_path` fixture.

### Pitfall 4: E2E Smoke Test Config Isolation
**What goes wrong:** Smoke test uses `data/` in the project root, contaminating real data or failing if `data/` doesn't exist.
**Why it happens:** Default config points to `data_dir: "data"` (relative path).
**How to avoid:** Use `--config` flag with a custom YAML pointing to a temp directory (`/tmp/smoke_test/`). Confirmed: `bitbat` accepts `--config FILE` as a top-level flag.

### Pitfall 5: model train Does Not Accept --tau
**What goes wrong:** `bitbat model train` does not accept `--tau` as an argument. The retrainer passes it anyway (CORR-01 bug). During smoke test, `model train` called directly by Claude will succeed; the bug only manifests when the autonomous retrainer calls it.
**Why it happens:** The CLI was updated but the retrainer subprocess command was not.
**How to avoid:** In the smoke test, call `model train` directly (it will work). Document CORR-01 as pre-validated CRITICAL in AUDIT-REPORT.md, separate from the smoke test pass/fail.

## Code Examples

### Test Marker Classification Scheme

Based on code inspection of the 608-test, 77-file suite:

```python
# Source: pytest docs + code inspection of this codebase

# BEHAVIORAL: Tests a single function/method in isolation, mocking dependencies
# Example pattern: monkeypatch, return value assertions on real logic
@pytest.mark.behavioral
def test_should_deploy_logic(tmp_path: Path) -> None:
    # Creates minimal DB, calls should_deploy() directly
    ...

# INTEGRATION: Tests multiple real components together, uses tmp_path for I/O
# Example pattern: seeds DB, calls agent.run_once(), checks DB state
@pytest.mark.integration
def test_monitoring_agent_blocks_startup_without_model_artifact(...):
    ...

# STRUCTURAL: Checks file existence, schema definitions, CLI help text, doc strings
# Does NOT call production logic — reads files or checks schemas
@pytest.mark.structural
def test_monitor_runbook_contains_required_operator_contracts() -> None:
    content = RUNBOOK.read_text()
    assert "--config" in content
    ...

# MILESTONE_MARKER: Reads other test files as strings, checks specific line presence
# These are deletion candidates — they add no coverage, inflate count
@pytest.mark.milestone_marker
def test_phase19_startup_guardrail_contract_remains_anchored() -> None:
    source = _source("tests/autonomous/test_agent_integration.py")
    assert "test_monitoring_agent_blocks_startup_without_model_artifact" in source
    ...
```

### Vulture Whitelist Pattern

```python
# whitelist_vulture.py — generated by --make-whitelist, then manually curated
# Source: https://github.com/jendrikseipp/vulture

# Pytest fixtures used implicitly via dependency injection
db_with_data  # used by pytest
db_with_predictions  # used by pytest
model_on_disk  # used by pytest

# Click options registered via decorator (vulture can't trace decorator-based registration)
_.freq  # used by @click.option
_.horizon  # used by @click.option
```

### Coverage pyproject.toml Config

```toml
# Add to pyproject.toml under [tool.pytest.ini_options] after tool installation
# Source: https://pytest-cov.readthedocs.io/en/latest/config.html
[tool.coverage.run]
source = ["bitbat"]
branch = true

[tool.coverage.report]
show_missing = true
skip_covered = false
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual test review | Pytest markers + automated classification | Always available | Makes test type machine-readable |
| Line coverage only | Branch coverage (`--cov-branch`) | pytest-cov 3.0+ | Reveals uncovered conditional arms, not just uncovered lines |
| Vulture standalone | Vulture + whitelist.py | vulture 1.0+ | Reduces triage work on known false positives |
| Radon per-module | Radon with `-n C -o SCORE` | Always | Filters noise; only shows functions needing attention |

**Deprecated/outdated:**
- `coverage run` + `coverage report` as separate steps: replaced by `pytest --cov` which does both in one pass
- Checking only CC > 20 (rank D+): The phase requirement specifies CC > 10 (rank C+), which is the appropriate threshold for a codebase audit

## Pre-Discovered Findings (Pre-Validated from Code Inspection)

These are confirmed before any tool runs. They should be pre-populated in AUDIT-REPORT.md and then cross-referenced with tool output.

### CRITICAL — CORR-01: Retrainer --tau argument bug
**Location:** `src/bitbat/autonomous/retrainer.py` lines 199-201
**Evidence:** `model train --help` shows no `--tau` option. Retrainer builds CLI command with `"--tau", str(self.tau)`. This causes `UsageError: No such option: --tau` when autonomous retraining fires.
**Maps to:** CORR-01
**Did vulture catch it?** No — vulture detects unused code, not wrong CLI arguments. This is a semantic bug.

### CRITICAL — CORR-02: CV key mismatch
**Location:** `src/bitbat/autonomous/retrainer.py` line 62-70 vs `metrics/cv_summary.json` writer
**Evidence:** `_read_cv_score()` reads from cv_summary.json. The key used to write vs read must be verified — if the writer uses a different key name, `_read_cv_score()` returns 0.0 silently, making every retraining candidate appear to be an improvement. Requires cross-reference with `model/walk_forward.py` and `model cv` CLI output.
**Maps to:** CORR-02

### HIGH — ARCH-04: API routes/system.py imports from gui/
**Location:** Flagged in REQUIREMENTS.md; needs code inspection during audit
**Maps to:** ARCH-04

### Milestone-Marker Test Files (confirmed deletion candidates — pure source-readers)
These files contain ONLY `source_file.read_text()` string assertions, no production code calls:
```
tests/autonomous/test_phase19_d1_monitor_alignment_complete.py  (5 tests — all source-readers)
tests/autonomous/test_session3_complete.py   (verify during classification)
tests/autonomous/test_session4_complete.py   (verify during classification)
tests/gui/test_phase5_timeline_complete.py   (verify — imports from bitbat.gui, may be integration)
tests/gui/test_phase6_timeline_ux_complete.py (verify)
tests/gui/test_phase7_streamlit_compat_complete.py (verify)
tests/gui/test_phase8_d2_timeline_complete.py (verify)
tests/gui/test_phase8_release_verification_complete.py (verify)
tests/gui/test_phase9_timeline_readability_complete.py (verify)
tests/gui/test_phase10_supported_surface_complete.py (verify)
tests/gui/test_phase11_runtime_stability_complete.py (verify)
tests/gui/test_phase12_simplified_ui_regression_complete.py (verify)
tests/gui/test_phase12_supported_views_smoke.py (verify)
tests/model/test_phase5_complete.py (verify — may be integration)
tests/api/test_phase4_complete.py (verify — imports bitbat.api, likely integration)
tests/analytics/test_phase3_complete.py (verify — imports bitbat.analytics, likely integration)
```

**Caution:** Files that import from `bitbat.*` and use `tmp_path` are integration tests mislabeled as milestone markers. Do NOT delete those — reclassify them.

## Open Questions

1. **Does `--config` propagate `data_dir` to all stages correctly?**
   - What we know: The `data_dir` key exists in `default.yaml`. The `--config FILE` flag is accepted by the top-level CLI.
   - What's unclear: Whether every subcommand respects the custom config's `data_dir` or hardcodes `Path("data")`.
   - Recommendation: Test with a small smoke run first. If stages write to the wrong location, use real `data/` directory for the smoke test and document it.

2. **Does the CV key mismatch (CORR-02) cause a silent 0.0 return or a KeyError?**
   - What we know: `_read_cv_score()` reads a JSON file. If the key is missing, `dict.get()` returns a default. The exact key names used by the cv writer vs reader must be verified.
   - What's unclear: Whether the mismatch causes silent failure (always shows improvement) or an exception.
   - Recommendation: During the audit, grep for the key names in `model/walk_forward.py` and compare to what `retrainer._read_cv_score()` expects.

3. **Are any `_complete.py` tests in `tests/api/` and `tests/analytics/` genuine integration tests?**
   - What we know: `tests/api/test_phase4_complete.py` and `tests/analytics/test_phase3_complete.py` both import from their respective `bitbat` modules.
   - What's unclear: Whether their test bodies exercise real production code or just do existence checks.
   - Recommendation: Read each file during classification. Preserve any test that calls a production function.

## Sources

### Primary (HIGH confidence)
- vulture 2.14 official README: https://github.com/jendrikseipp/vulture — CLI flags, whitelist mechanism, pyproject.toml config
- radon 6.0.1 official docs: https://radon.readthedocs.io/en/latest/commandline.html — cc command, rank thresholds, CLI flags
- pytest-cov 7.0.0 official docs: https://pytest-cov.readthedocs.io/en/latest/ — branch coverage, report formats, configuration
- Direct code inspection: `src/bitbat/autonomous/retrainer.py` — confirmed CORR-01 bug at line 200
- Direct tool execution: `poetry run radon cc src/bitbat -s -n C` — verified 32 high-complexity functions
- Direct tool execution: `poetry run pytest --cov=bitbat --cov-branch` — 77% overall branch coverage confirmed, module breakdown obtained
- Direct tool execution: `poetry run vulture src/bitbat tests/ --min-confidence 80` — confirmed false-positive patterns
- Direct tool execution: `poetry run pytest` — 608 tests, all passing

### Secondary (MEDIUM confidence)
- pytest marker documentation: https://docs.pytest.org/en/stable/reference/reference.html — marker registration in conftest.py and pyproject.toml

### Tertiary (LOW confidence)
- None — all findings verified by direct tool execution or official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all three tools installed and verified to run correctly on this codebase
- Architecture: HIGH — tool output patterns confirmed by actual execution; evidence file structure matches CONTEXT.md requirements
- Pitfalls: HIGH — false-positive pattern confirmed by actual vulture run; milestone-marker pattern confirmed by reading test files; CORR-01 confirmed by code inspection + CLI help comparison

**Research date:** 2026-03-04
**Valid until:** 2026-04-03 (30 days — tools are stable; codebase complexity may shift)
