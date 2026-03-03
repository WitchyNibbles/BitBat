# Stack Research

**Domain:** Static analysis and code quality audit tooling for Python ML pipeline
**Researched:** 2026-03-04
**Confidence:** HIGH

## Context

This research covers audit tooling for the v1.5 Codebase Health Audit & Critical Remediation
milestone. The codebase is ~14,400 lines across 79 source files and 84 test files. The existing
toolchain already includes: ruff (0.5+), mypy (1.11+), black, pytest (8.x). This research
identifies what to ADD for a comprehensive audit pass, not what already exists.

Existing ruff rule set: `E, F, B, I, UP, S, C4, RET, SIM` — security rules (S) already enabled
through ruff's bandit integration. Missing from the current config: complexity checks (C901),
dead code detection, import graph analysis, test coverage measurement.

---

## Recommended Stack

### Core Audit Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ruff (extend config) | 0.15.4 (current) | Add C901 McCabe complexity + PLR rules | Already in toolchain; adding `C` and `PLR` rule families catches complexity hotspots and refactor candidates with zero new dependencies |
| pytest-cov | 7.0.0 | Branch-level test coverage measurement | Identifies which pipeline stages (ingest, features, labeling, autonomous) have zero test coverage; `--branch` flag catches partial branch coverage the 84 existing tests miss |
| coverage.py | 7.13.4 | Coverage driver used by pytest-cov | Required by pytest-cov 7.x; `--branch` mode critical for ML pipeline conditionals (config toggles, feature flags, error paths) |
| vulture | 2.14 | Dead code detection | Finds unused functions, classes, variables across 79 source files; confidence scoring (60–100%) reduces false-positive noise; `--min-confidence 80` is practical for a brownfield codebase |
| import-linter | 2.9 | Import graph / circular dependency enforcement | BitBat's 12-module architecture (cli → contracts → features → model) makes layered import contracts feasible; catches architecture drift where lower layers (io, timealign) start importing from upper layers (api, autonomous) |
| pip-audit | 2.10.0 | Dependency vulnerability scanning | Scans all 20 runtime dependencies against GitHub Python Advisory Database; free, Apache 2.0, PyPA-maintained — no paid tier needed |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| radon | 6.0.1 | Cyclomatic complexity + Maintainability Index (MI) | Use for the one-time audit report (`radon cc src/ -a -nc` for average complexity, `radon mi src/` for MI scores); not needed in CI — ruff C901 covers ongoing enforcement |
| mypy (extend config) | 1.11+ (current) | Increase strictness for audit pass | Add `--strict` flag or `warn_return_any = true`, `disallow_any_generics = true` to surface type gaps in the 14,400 LOC that current config (non-strict) allows through |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff (C901 extension) | Complexity gate in CI | Add `"C"` to `select` in `pyproject.toml`; set `[tool.ruff.mccabe] max-complexity = 10` — flags functions exceeding threshold without requiring radon in CI |
| vulture (whitelist pattern) | Dead code audit | Run `vulture src/ --min-confidence 80 > vulture_findings.txt`; review findings manually; create `whitelist.py` for false positives (CLI callbacks, FastAPI route handlers, contract validators used via string name) |
| import-linter (`.importlinter`) | Architecture enforcement | Define `layers` contract: `api, autonomous, gui` above `model, backtest, analytics` above `features, labeling, dataset` above `ingest, io, timealign, config`; run `lint-imports` in CI |
| pip-audit (CI integration) | Dependency CVE gate | `pip-audit --requirement <(poetry export -f requirements.txt)` in CI; flags before new vulnerabilities ship |

---

## Installation

```bash
# Audit-specific additions (dev group only — no runtime impact)
poetry add --group dev pytest-cov coverage radon vulture import-linter pip-audit

# Verify versions
poetry run pytest-cov --version   # expect 7.0.x
poetry run coverage --version      # expect 7.13.x
poetry run vulture --version       # expect 2.14
poetry run lint-imports --version  # expect 2.9.x
pip-audit --version                # expect 2.10.x
radon --version                    # expect 6.0.1
```

---

## Audit Commands

```bash
# 1. Coverage — identify untested pipeline stages
poetry run pytest --cov=src/bitbat --cov-branch --cov-report=term-missing --cov-report=html

# 2. Dead code — find unused functions/classes across 79 source files
poetry run vulture src/ --min-confidence 80 --sort-by-size

# 3. Complexity — one-time audit report (not CI)
poetry run radon cc src/ -a -nc -s         # cyclomatic complexity, average, show complexity
poetry run radon mi src/ -s                # maintainability index with scores

# 4. Complexity — CI gate via ruff (add to existing check)
# (requires pyproject.toml change: add "C" to select, add [tool.ruff.mccabe])
poetry run ruff check src/ --select C901

# 5. Import architecture — enforce layered boundaries
poetry run lint-imports

# 6. Dependency CVEs
pip-audit --requirement <(poetry export --without-hashes -f requirements.txt)

# 7. Type strictness delta (audit pass only)
poetry run mypy src/ --strict --ignore-missing-imports 2>&1 | grep "error:" | wc -l
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| vulture | deadcode (newer tool) | deadcode has fewer false positives but lacks confidence scoring; use if vulture whitelist becomes unwieldy after initial cleanup |
| import-linter | pycycle | pycycle only detects cycles; import-linter enforces layered architecture AND cycles — better fit for a 12-module pipeline with intentional layering |
| pip-audit | safety | safety requires paid tier for full vulnerability database; pip-audit covers GitHub Advisory DB for free, sufficient for this codebase's dependency surface |
| pytest-cov + coverage.py | codecov (SaaS) | codecov adds SaaS dependency and token management; coverage.py HTML reports are sufficient for a local-first single-operator platform audit |
| ruff C901 for CI | radon in CI | radon is slower and not already in toolchain; ruff C901 is zero-overhead since ruff already runs in CI |
| mypy extended config | pyright | pyright is faster and has better inference, but mypy is already in the toolchain with existing suppressions; switching type checkers mid-audit would surface thousands of new errors, obscuring real findings |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| bandit (standalone) | ruff already implements bandit rules via the `S` rule family, which is already enabled in `pyproject.toml`; running standalone bandit duplicates findings with worse DX | ruff with existing S rules |
| pylint | Enormous noise-to-signal ratio on a codebase already using ruff; pylint and ruff rules overlap significantly, creating duplicate findings that slow audit triage | ruff (already configured) |
| prospector / flake8 | Both are multi-tool wrappers that conflict with ruff and produce overlapping reports; adds configuration management overhead with no additional signal | ruff with targeted rule expansion |
| xenon (complexity CI gate) | xenon is a radon wrapper for CI; ruff C901 covers the same gate with zero additional tooling | ruff C901 rule |
| SonarQube | Requires server infrastructure and produces findings that overlap entirely with ruff + mypy + vulture for a codebase this size (14,400 LOC); operational cost exceeds signal value | ruff + mypy + vulture combination |
| pyright (replacing mypy) | Switching type checkers mid-audit floods output with new errors unrelated to real issues being hunted; audit goal is delta measurement, not switching tools | mypy with strictness flags added temporarily |

---

## Stack Patterns by Audit Domain

**Dead code in ML pipeline:**
- Use vulture at `--min-confidence 80` not 100; ML code commonly has functions used via dynamic dispatch (contract validators called by string, CLI callbacks registered by decorator, FastAPI routes registered by decorator) that vulture misidentifies at 100%
- Create `whitelist.py` listing false positives: contract check methods, Click command functions, route handler functions

**Type coverage gaps:**
- Run `mypy src/ --strict` once to capture the full error delta vs. current config
- Triage: functions with no annotations at all (easy wins), `Any` propagation chains (architecture risk), third-party stub gaps (ignore)
- Do NOT use `--strict` in CI during audit — it will break CI; capture output to file for manual review

**Import architecture validation:**
- Define contracts matching existing CLAUDE.md module map: `cli` at top, `timealign/io/config` at bottom
- Run `lint-imports --verbose` first time to understand actual import graph before defining contracts
- The `acyclic_siblings` contract type is the minimal starting point if full layering contract is too strict initially

**Test coverage gaps:**
- Focus on `--cov-branch` not just line coverage; ML conditionals (feature toggle branches, horizon/frequency branches) are the highest-risk untested paths
- Sort HTML coverage report by coverage % ascending to find lowest-covered modules first
- `bitbat/autonomous/` and `bitbat/ingest/` are highest-risk candidates based on architecture complexity

**Dependency vulnerabilities:**
- `pip-audit` against `poetry export` output; re-run after any `poetry update`
- Check results against severity filter: `pip-audit --severity high` to focus on exploitable CVEs first

---

## pyproject.toml Changes Required

```toml
# Add to [tool.ruff.lint] select:
# "C" enables C901 McCabe complexity
select = ["E", "F", "B", "I", "UP", "S", "C4", "RET", "SIM", "C", "PLR"]
#                                                                 ^    ^
#                                         McCabe complexity ----+    |
#                                         Pylint refactor rules -----+

# Add new section for complexity threshold:
[tool.ruff.mccabe]
max-complexity = 10

# Add coverage config:
[tool.coverage.run]
branch = true
source = ["src/bitbat"]

[tool.coverage.report]
show_missing = true
skip_covered = false

# Add vulture config:
[tool.vulture]
min_confidence = 80
paths = ["src/bitbat"]
sort_by_size = true
```

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| pytest-cov 7.0.0 | coverage.py >= 7.10.6 | pytest-cov 7 dropped subprocess measurement; install coverage 7.13.4 |
| pytest-cov 7.0.0 | pytest 8.x | Compatible; tested combination |
| vulture 2.14 | Python 3.11 | Full support; pyproject.toml config supported |
| import-linter 2.9 | Python 3.11 | Current; `acyclic_siblings` contract type added in 2.x series |
| pip-audit 2.10.0 | Python 3.10+ (required) | Python 3.11 in this project meets requirement |
| radon 6.0.1 | Python 3.11 | Stable; no known incompatibilities |
| ruff 0.15.4 | Python 3.11, pyproject.toml | Current version; C901 and PLR rules stable |

---

## Sources

- PyPI vulture — version 2.14, released December 8, 2024 (HIGH confidence, verified via pypi.org)
- PyPI import-linter — version 2.9, released February 6, 2026 (HIGH confidence, verified via libraries.io)
- PyPI pip-audit — version 2.10.0, released December 1, 2025 (HIGH confidence, verified via pypi.org)
- PyPI pytest-cov — version 7.0.0, released September 9, 2025 (HIGH confidence, verified via pypi.org)
- PyPI coverage.py — version 7.13.4, released February 9, 2026 (HIGH confidence, verified via coverage.readthedocs.io)
- PyPI radon — version 6.0.1, released March 26, 2023 (HIGH confidence; stable, low churn tool)
- Ruff 0.15.4 release — 2026-02-26 (HIGH confidence, verified via astral.sh/blog)
- Ruff C901 docs — https://docs.astral.sh/ruff/rules/complex-structure/ (HIGH confidence)
- David Seddon — import-linter acyclic_siblings pattern, 2025 (MEDIUM confidence, single source)
- Bandit 1.9.3 review 2026 — confirmed ruff S rules are equivalent subset (MEDIUM confidence)
- mypy 1.19.1 docs — existing_code.html, strict mode guidance (HIGH confidence, official docs)

---
*Stack research for: v1.5 Codebase Health Audit & Critical Remediation — static analysis tooling*
*Researched: 2026-03-04*
