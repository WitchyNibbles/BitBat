# Phase 27: Verification & Guardrail Hardening — Research

**Researched:** 2026-03-07
**Domain:** CI static-analysis gates — import-linter layer contracts + ruff C901 complexity gate
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ARCH-05 | import-linter contracts added to CI preventing future cross-layer import drift | import-linter forbidden/layers contract types, pyproject.toml config syntax, `lint-imports` CLI command, CI step pattern |
| ARCH-06 | ruff C901 complexity gate added to CI with max-complexity = 10, adding a function > 10 causes CI to fail | ruff `[tool.ruff.lint.mccabe] max-complexity = 10`, C901 rule, existing violations inventory, `# noqa: C901` suppression strategy |
</phase_requirements>

---

## Summary

Phase 27 adds two static-analysis gates to the existing GitHub Actions CI pipeline (`.github/workflows/ci.yml`). Neither tool is currently installed or configured in this project. The existing `lint` job runs `ruff check` and `ruff format` but does NOT include C901 (it is absent from `[tool.ruff.lint] select`). There is no `lint-imports` step anywhere in CI.

The most important pre-planning fact: **11 functions already violate C901 at max-complexity = 10**. These are pre-existing legacy violations in `autonomous/`, `backtest/`, `cli.py`, `contracts.py`, and `ingest/`. Naively adding C901 to the select list will immediately break CI until those lines receive `# noqa: C901` suppressions (or are refactored, which is explicitly out of scope per REQUIREMENTS.md — see DEBT-01 deferred items). The planner must account for a Wave 0 task that installs suppressions on the 11 existing violations before enabling the gate.

The import-linter configuration needs to encode the architectural decisions already enforced by hand in Phase 26: specifically that `api` must not import from `gui`. The cleanest approach is a `forbidden` contract (not a full `layers` contract) targeting `bitbat.api` → `bitbat.gui` since the project's layer hierarchy is not uniformly clean enough for a strict top-down layers contract without many ignore entries.

**Primary recommendation:** Add `import-linter` as a dev dependency, write a `forbidden` contract for `api→gui`, add `C901` to ruff select with `max-complexity = 10`, suppress the 11 existing violations with `# noqa: C901`, and add a `import-linter` CI step after the existing lint step.

---

## Standard Stack

### Core

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| `import-linter` | 2.7+ (latest on PyPI) | Enforce layer contracts by analysing import graph | Only Python tool that does graph-based import architecture enforcement; used by Django, attrs projects |
| `ruff` | 0.5.7 (already installed) | C901 McCabe complexity gate | Already in project; C901 support built-in via `mccabe` plugin |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| `pytest` (structural mark) | 8.4.2 (already installed) | Pytest-based structural guard as belt-and-suspenders | Catches regressions in test runs, not just CI lint step |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `import-linter` | `flake8-import-order` / `isort` | Only check import ordering, not cross-layer dependencies — wrong problem |
| `import-linter` | Custom AST test (already done for api→gui in test_no_gui_import.py) | Already exists as pytest test; import-linter adds a lint-phase gate that fails faster and covers indirect imports too |
| `import-linter` forbidden contract | `import-linter` layers contract | Full layers contract requires exhaustive layer ordering; the bitbat codebase has too many cross-cutting modules (contracts.py, common, io, config) for a clean strict hierarchy; forbidden contract is narrowly correct |

**Installation:**
```bash
poetry add --group dev import-linter
```

---

## Architecture Patterns

### Recommended Structure for Phase 27

```
.importlinter                    # import-linter config (alternative: pyproject.toml)
pyproject.toml                   # add C901 to [tool.ruff.lint] select
                                 # add [tool.ruff.lint.mccabe] max-complexity = 10
.github/workflows/ci.yml         # add lint-imports step to lint job
src/bitbat/autonomous/agent.py   # add # noqa: C901 to run_once
src/bitbat/autonomous/orchestrator.py  # add # noqa: C901 to one_click_train
src/bitbat/autonomous/predictor.py    # add # noqa: C901 to predict_latest
src/bitbat/backtest/metrics.py        # add # noqa: C901 to summary
src/bitbat/cli.py                     # add # noqa: C901 to model_cv
src/bitbat/contracts.py               # add # noqa: C901 to ensure_feature_contract
src/bitbat/ingest/news_cryptocompare.py  # add # noqa: C901 to _fetch_page, fetch
src/bitbat/ingest/news_gdelt.py          # add # noqa: C901 to _fetch_chunk, fetch
src/bitbat/ingest/prices.py              # add # noqa: C901 to fetch_yf
```

### Pattern 1: ruff C901 Gate

**What:** Add `C901` to `[tool.ruff.lint] select` and configure `[tool.ruff.lint.mccabe] max-complexity = 10`. Suppress pre-existing violations with `# noqa: C901`.

**When to use:** Preventing new high-complexity functions while acknowledging legacy debt.

**Example (pyproject.toml):**
```toml
# Source: https://docs.astral.sh/ruff/settings/#lint_mccabe_max-complexity

[tool.ruff.lint]
# Add C901 to existing select list:
select = ["E", "F", "B", "I", "UP", "S", "C4", "RET", "SIM", "C901"]
ignore = ["E203", "B008"]

[tool.ruff.lint.mccabe]
max-complexity = 10
```

**Inline suppression for existing violations:**
```python
# Source: https://docs.astral.sh/ruff/rules/complex-structure/
def run_once(self) -> dict[str, Any]:  # noqa: C901
    ...
```

### Pattern 2: import-linter Forbidden Contract

**What:** Define a `forbidden` contract in pyproject.toml declaring that `bitbat.api` must not import from `bitbat.gui`.

**When to use:** Narrow architectural constraint that prevents a specific cross-layer pattern from being reintroduced, without requiring exhaustive layer ordering.

**Example (pyproject.toml):**
```toml
# Source: https://import-linter.readthedocs.io/en/v2.9/contract_types/forbidden/

[tool.importlinter]
root_package = "bitbat"

[[tool.importlinter.contracts]]
name = "API layer must not import from GUI layer"
type = "forbidden"
source_modules = ["bitbat.api"]
forbidden_modules = ["bitbat.gui"]
```

**Running it:**
```bash
poetry run lint-imports
```

### Pattern 3: CI Integration

**What:** Add `lint-imports` as a step in the existing `lint` job in `.github/workflows/ci.yml`.

**When to use:** Gate must run on every push/PR to main, blocking merge if violated.

**Example:**
```yaml
# In the existing lint job, after the ruff steps:
- name: Import architecture contracts
  run: poetry run lint-imports
```

### Anti-Patterns to Avoid

- **Adding C901 without noqa on existing violations:** Immediately breaks CI on 11 functions. The gate must be added together with suppressions in a single atomic commit.
- **Using a layers contract instead of forbidden:** A layers contract for bitbat would require fully ordering all modules including contracts.py, common, io, timealign, labeling — too many cross-cutting concerns; will produce spurious failures.
- **Putting import-linter config in `.importlinter` file:** Prefer pyproject.toml for consistency with project's existing tool config style.
- **Adding import-linter step before installing it as a dev dependency:** `lint-imports` will not be available in CI without `poetry install` picking it up.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Layer enforcement | Custom AST import scanner in tests | `import-linter` forbidden contract | import-linter follows the full import graph including transitive/indirect imports; AST test only catches direct top-level imports in scanned files |
| Complexity gate | radon + custom CI script | ruff C901 | ruff is already the project's linter; C901 is built-in; zero extra tooling needed |

**Key insight:** The project already has AST-based structural guards (`tests/api/test_no_gui_import.py`) that catch direct `from bitbat.gui` imports in api source files. import-linter adds a *lint-phase* gate (not test-phase) and catches *transitive* imports too — if api imports from X which imports from gui, import-linter catches it; the AST test does not.

---

## Common Pitfalls

### Pitfall 1: C901 Gate Breaks CI Immediately on 11 Existing Functions

**What goes wrong:** Adding `"C901"` to ruff select causes `poetry run ruff check src/` to fail with 11 violations. CI blocks all PRs.

**Why it happens:** The audit (Phase 24) identified these functions as high-complexity (CC 11–19). They were catalogued as DEBT-01 (deferred to v1.6+). The C901 gate must be added with simultaneous suppressions.

**How to avoid:** In the same commit that enables C901, add `# noqa: C901` to each function definition line for all 11 pre-existing violations.

**Exact list of violations (verified 2026-03-07 with `ruff check --select C901 src/`):**
```
src/bitbat/autonomous/agent.py:182     run_once (14 > 10)
src/bitbat/autonomous/orchestrator.py:29  one_click_train (17 > 10)
src/bitbat/autonomous/predictor.py:109   predict_latest (18 > 10)
src/bitbat/backtest/metrics.py:20        summary (12 > 10)
src/bitbat/cli.py:544                    model_cv (14 > 10)
src/bitbat/contracts.py:79              ensure_feature_contract (19 > 10)
src/bitbat/ingest/news_cryptocompare.py:80   _fetch_page (14 > 10)
src/bitbat/ingest/news_cryptocompare.py:259  fetch (18 > 10)
src/bitbat/ingest/news_gdelt.py:95      _fetch_chunk (11 > 10)
src/bitbat/ingest/news_gdelt.py:244     fetch (15 > 10)
src/bitbat/ingest/prices.py:94          fetch_yf (11 > 10)
```

**Warning signs:** CI lint job fails immediately after adding C901 to select.

### Pitfall 2: import-linter Root Package Configuration

**What goes wrong:** `lint-imports` fails with "root_package is not importable" because the package src layout requires `src` to be on the path.

**Why it happens:** `bitbat` is under `src/bitbat/` — Poetry installs it as a package, so `import bitbat` works in the virtual env; `lint-imports` uses the virtual env's import mechanism, so it works correctly as long as `poetry run lint-imports` is used.

**How to avoid:** Always run `lint-imports` via `poetry run lint-imports`. Do not run it with bare Python.

**Warning signs:** "ModuleNotFoundError: No module named 'bitbat'" when running lint-imports outside the poetry environment.

### Pitfall 3: Forgetting to Add import-linter to Dev Dependencies

**What goes wrong:** CI fails with `lint-imports: command not found` even though the step is in ci.yml.

**Why it happens:** `lint-imports` is not installed unless `import-linter` is in `pyproject.toml` dev dependencies and picked up by `poetry install`.

**How to avoid:** Run `poetry add --group dev import-linter` and commit the updated `pyproject.toml` and `poetry.lock` together.

### Pitfall 4: Skipping Verification That the Gate Actually Blocks Violations

**What goes wrong:** Gate is configured but doesn't actually block violations due to misconfiguration (wrong contract type, wrong module path, etc.).

**Why it happens:** Configuration errors are silent — `lint-imports` exits 0 even if contracts are malformed.

**How to avoid:** After configuring, test that the gate works by temporarily introducing a violation and confirming `lint-imports` fails. The success criteria in the phase explicitly requires this: "introducing a cross-layer import causes CI to fail."

---

## Code Examples

Verified patterns from official sources:

### Full pyproject.toml additions for C901 + import-linter

```toml
# Source: https://docs.astral.sh/ruff/settings/#lint_mccabe_max-complexity
# and: https://import-linter.readthedocs.io/en/v2.9/contract_types/forbidden/

[tool.ruff.lint]
select = ["E", "F", "B", "I", "UP", "S", "C4", "RET", "SIM", "C901"]
ignore = ["E203", "B008"]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.importlinter]
root_package = "bitbat"

[[tool.importlinter.contracts]]
name = "API layer must not import from GUI layer"
type = "forbidden"
source_modules = ["bitbat.api"]
forbidden_modules = ["bitbat.gui"]
```

### CI step addition

```yaml
# Source: import-linter docs — lint-imports is the CLI entry point
# In .github/workflows/ci.yml, lint job, after existing ruff steps:
- name: Import architecture contracts
  run: poetry run lint-imports
```

### Noqa suppression for existing C901 violations

```python
# Source: https://docs.astral.sh/ruff/rules/complex-structure/
# Pattern: add # noqa: C901 to the def line only
def run_once(self) -> dict[str, Any]:  # noqa: C901
    """Run one monitoring cycle: predict, validate, assess drift, retrain."""
```

### Verifying the gate actually blocks violations (test pattern)

```python
# Temporarily add to a test file to confirm gate works, then revert:
# Method: Use subprocess to run lint-imports and check exit code
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "importlinter"],
    capture_output=True
)
assert result.returncode == 0, result.stderr.decode()
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual code review for cross-layer imports | import-linter forbidden contracts in CI | import-linter 1.x circa 2019; 2.x in 2022+ | Architectural violations caught in CI, not post-merge review |
| flake8-mccabe plugin | ruff C901 built-in | ruff 0.1+ (2023) | Single linter for all static analysis; no separate flake8 needed |
| AST test guards (Phase 26 approach) | import-linter (Phase 27) | This phase | AST tests run at test time; import-linter runs at lint time (faster feedback); also catches transitive imports |

**Deprecated/outdated:**
- `flake8-mccabe`: Replaced by ruff's C901 for projects already using ruff
- `.importlinter` file: Still supported but pyproject.toml format is preferred for projects using pyproject.toml for all tool config

---

## Open Questions

1. **Should the forbidden contract also prevent `autonomous` from importing `gui`?**
   - What we know: The audit (AUDIT-REPORT.md in Phase 24) flagged `api→gui` as the specific violation. The `autonomous` layer was not called out.
   - What's unclear: Whether `autonomous` currently imports from `gui` and whether this would be intended.
   - Recommendation: Start with `source_modules = ["bitbat.api"]` as the ARCH-05 requirement specifies. Do not expand scope beyond what ARCH-05 requires.

2. **Should cli.py be in the forbidden source_modules for gui imports?**
   - What we know: cli.py is not in `bitbat.api`; it may legitimately coordinate with GUI in some contexts.
   - What's unclear: Whether any such imports exist currently.
   - Recommendation: Scope the contract to `bitbat.api` only per ARCH-05. If broader enforcement is desired, that is a separate requirements scope.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `poetry run pytest tests/api/ tests/config/ -m structural -q --tb=short` |
| Full suite command | `poetry run pytest tests/ -q --tb=short` |
| Estimated runtime | ~5–10 seconds (structural tests only); ~60–90 seconds (full suite) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ARCH-05 | `lint-imports` exits non-zero when api imports gui | structural | `poetry run lint-imports` | ❌ Wave 0 gap (config to be added) |
| ARCH-05 | Contract is syntactically valid and lint-imports exits 0 on clean code | structural | `poetry run lint-imports` | ❌ Wave 0 gap |
| ARCH-05 | Existing pytest guard still passes after import-linter config added | structural | `poetry run pytest tests/api/test_no_gui_import.py -v` | ✅ exists |
| ARCH-06 | `ruff check src/` exits non-zero when a new function exceeds CC=10 | structural | `poetry run ruff check src/ --select C901` | ❌ Wave 0 gap (C901 not in select yet) |
| ARCH-06 | All 11 pre-existing violations are suppressed with `# noqa: C901` | structural | `poetry run ruff check src/` (must exit 0) | ❌ Wave 0 gap (noqa not added yet) |

### Nyquist Sampling Rate

- **Minimum sample interval:** After every committed task → run: `poetry run ruff check src/ tests/ && poetry run lint-imports`
- **Full suite trigger:** Before merging final task → `poetry run pytest tests/ -q`
- **Phase-complete gate:** Full suite green before `/gsd:verify-work` runs
- **Estimated feedback latency per task:** ~5–15 seconds

### Wave 0 Gaps

- [ ] `pyproject.toml` — add `C901` to `[tool.ruff.lint] select` and add `[tool.ruff.lint.mccabe] max-complexity = 10`
- [ ] `pyproject.toml` — add `[tool.importlinter]` and `[[tool.importlinter.contracts]]` sections
- [ ] `pyproject.toml` and `poetry.lock` — add `import-linter` to dev dependencies (`poetry add --group dev import-linter`)
- [ ] 11 source files — add `# noqa: C901` to pre-existing violation function definitions (must be done simultaneously with C901 gate enablement)
- [ ] `.github/workflows/ci.yml` — add `lint-imports` step to the `lint` job

*(No new test files are needed; validation is done via the tools themselves. The existing `tests/api/test_no_gui_import.py` covers belt-and-suspenders for ARCH-05.)*

---

## Sources

### Primary (HIGH confidence)

- https://docs.astral.sh/ruff/rules/complex-structure/ — C901 rule documentation, noqa suppression pattern
- https://docs.astral.sh/ruff/settings/ — `[tool.ruff.lint.mccabe] max-complexity` configuration syntax
- https://import-linter.readthedocs.io/en/v2.9/contract_types/forbidden/ — forbidden contract configuration syntax (pyproject.toml format)
- https://import-linter.readthedocs.io/en/v2.9/contract_types/layers/ — layers contract reference
- https://import-linter.readthedocs.io/en/stable/get_started/configure/ — root_package and pyproject.toml configuration
- Direct ruff execution: `poetry run ruff check --select C901 src/` — produced exact list of 11 violations (verified 2026-03-07)
- Direct inspection: `cat .github/workflows/ci.yml` — confirmed no C901 or lint-imports steps exist
- Direct inspection: `cat pyproject.toml` — confirmed C901 absent from select, import-linter not installed

### Secondary (MEDIUM confidence)

- https://import-linter.readthedocs.io/en/stable/ — CLI entry point (`lint-imports`) confirmed
- https://pypi.org/project/import-linter/ — PyPI page confirms current release is 2.7+

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — both tools verified against official docs; ruff already installed and tested live; import-linter docs verified via WebFetch
- Architecture: HIGH — CI file structure confirmed by direct read; exact violation list confirmed by live ruff execution
- Pitfalls: HIGH — Pitfall 1 (11 violations) confirmed by live tool run; others follow directly from docs

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (ruff is stable; import-linter is stable; 30-day horizon appropriate)
