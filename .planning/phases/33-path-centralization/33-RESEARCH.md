# Phase 33: Path Centralization - Research

**Researched:** 2026-03-12
**Domain:** Python path management, config-driven artifact location
**Confidence:** HIGH

## Summary

Phase 33 addresses DEBT-02: 15+ hardcoded `Path("models")` and `Path("metrics")` literals scattered across `src/bitbat/`. These literals are relative to the process working directory, so they silently break whenever an operator runs a command from a different directory or wants to relocate artifacts. The fix is a single canonical path-resolution helper in `bitbat/config/loader.py` (or a new `bitbat/config/paths.py`) that reads `models_dir` and `metrics_dir` from the config and returns absolute `Path` objects. Every call site is updated to call this helper instead of constructing `Path("models")` or `Path("metrics")` directly.

The existing config infrastructure is already well-suited: `get_runtime_config()` is called pervasively across the codebase, the config dict already contains `data_dir` (handled correctly by most modules), and `default.yaml` is the single place to add new keys. The missing piece is that `models_dir` and `metrics_dir` were never added to the config — they kept their hardcoded defaults.

A key complication is test compatibility: many tests use `monkeypatch.chdir(tmp_path)` and then assert that `Path("metrics")/...` or `Path("models")/...` exist. After centralization these tests will no longer work via chdir — they must either (a) set a config override pointing paths into `tmp_path`, or (b) monkeypatch the path-resolution helper. The recommended approach is to use `monkeypatch.chdir(tmp_path)` and set default values in the config helper to `Path("models")` / `Path("metrics")` relative to cwd — meaning behavior is unchanged when no config override is provided, and tests only need updating if they explicitly test the new config-driven behavior.

An alternative that avoids test churn: keep defaults as `Path("models")` and `Path("metrics")` (cwd-relative), but resolve them against the config when `models_dir`/`metrics_dir` keys are set. This satisfies the success criterion "changing config YAML redirects all artifact reads and writes" while being backward-compatible with every existing test.

**Primary recommendation:** Add `models_dir` and `metrics_dir` keys to `default.yaml` (defaulting to `"models"` and `"metrics"`). Add `resolve_models_dir(config=None)` and `resolve_metrics_dir(config=None)` to `bitbat/config/loader.py`. Replace all 15+ call sites with these helpers. Existing tests using `monkeypatch.chdir(tmp_path)` continue to work because the defaults are relative strings — resolved relative to cwd, same as before.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEBT-02 | Hardcoded `Path("models")` / `Path("metrics")` centralized — all 15+ occurrences replaced with config-driven path resolution | Config loader already has `get_runtime_config()`; adding two resolution helpers and two YAML keys is sufficient; defaults preserve existing test behavior |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pathlib.Path | stdlib | Path construction and resolution | Already used everywhere |
| PyYAML | (existing) | Config file loading | Already in use via `bitbat/config/loader.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| bitbat.config.loader | (internal) | Config cache and resolution | All modules already import from here |

**Installation:** No new dependencies — pure internal refactor.

## Architecture Patterns

### Recommended Project Structure

No directory changes needed. Two new functions are added to the existing config loader:

```
src/bitbat/config/
├── __init__.py          # unchanged (empty)
├── loader.py            # add resolve_models_dir(), resolve_metrics_dir()
└── default.yaml         # add models_dir: "models", metrics_dir: "metrics"
```

### Pattern 1: Config-Driven Path Resolution Helper

**What:** Two thin functions in `loader.py` that read the config and return resolved `Path` objects.

**When to use:** Any module that needs to construct a path under `models/` or `metrics/`.

**Example:**
```python
# In bitbat/config/loader.py

def resolve_models_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the canonical models directory as an absolute Path.

    Reads ``models_dir`` from the active config (default: ``"models"``).
    Relative values are resolved against the process working directory,
    preserving backward compatibility for all existing tests that use
    ``monkeypatch.chdir(tmp_path)``.
    """
    cfg = config if config is not None else get_runtime_config()
    raw = str(cfg.get("models_dir", "models"))
    return Path(raw).expanduser()


def resolve_metrics_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the canonical metrics directory as an absolute Path.

    Reads ``metrics_dir`` from the active config (default: ``"metrics"``).
    """
    cfg = config if config is not None else get_runtime_config()
    raw = str(cfg.get("metrics_dir", "metrics"))
    return Path(raw).expanduser()
```

**Call-site replacement pattern:**
```python
# BEFORE (16 occurrences across 8 files):
Path("models") / f"{freq}_{horizon}" / "xgb.json"
Path("metrics") / "cv_summary.json"

# AFTER:
from bitbat.config.loader import resolve_models_dir, resolve_metrics_dir
resolve_models_dir() / f"{freq}_{horizon}" / "xgb.json"
resolve_metrics_dir() / "cv_summary.json"
```

### Pattern 2: Instance-Level Config Injection (Classes)

Classes that already receive config in `__init__` (e.g., `LivePredictor`, `ContinuousTrainer`, `AutoRetrainer`) should resolve the dirs once in `__init__` and store them:

```python
# In LivePredictor.__init__:
self.model_dir = resolve_models_dir(config)

# Then _model_path() becomes:
def _model_path(self) -> Path:
    return self.model_dir / f"{self.freq}_{self.horizon}" / "xgb.json"
```

This is already the existing pattern for `self.data_dir` — `LivePredictor` and `AutoRetrainer` both do:
```python
self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
```
The new helpers follow the same convention.

### Pattern 3: Default-Argument Functions with Optional Config

Functions with `output_dir: str | Path = Path("metrics")` as a default parameter (e.g., `write_regression_metrics`, `write_window_diagnostics`) need careful treatment. The default at function definition time is evaluated once — changing the function signature to accept `None` and resolve lazily is the correct approach:

```python
# BEFORE:
def write_regression_metrics(
    metrics: dict[str, Any],
    y_true: ...,
    y_pred: ...,
    output_dir: str | Path = Path("metrics"),
) -> Path:

# AFTER:
def write_regression_metrics(
    metrics: dict[str, Any],
    y_true: ...,
    y_pred: ...,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is None:
        from bitbat.config.loader import resolve_metrics_dir
        output_dir = resolve_metrics_dir()
    ...
```

This change is backward-compatible: callers that pass an explicit `output_dir` are unaffected; callers that relied on the default now get the config-resolved path.

### Anti-Patterns to Avoid

- **Resolving paths at module import time:** If `resolve_models_dir()` is called at the module level (outside a function), the config may not be loaded yet, or may be the wrong config in test scenarios. Always call inside a function or `__init__`.
- **Absolute path baking in default.yaml:** Keep defaults as relative strings (`"models"`, `"metrics"`) so that `monkeypatch.chdir(tmp_path)` continues to work in unit tests. An absolute path default would break all existing `chdir`-based tests.
- **Duplicating the resolution logic:** Do not copy `Path(config.get("models_dir", "models"))` inline at call sites. Always use `resolve_models_dir()`. Otherwise the config key name diverges from usage.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Path resolution with config fallback | Custom resolver per module | `resolve_models_dir()` / `resolve_metrics_dir()` in loader.py | Single definition, single change point |
| Config-aware default args | Evaluating `Path(config.get(...))` inline | Call the helper inside the function body | Avoids module-import-time evaluation |

**Key insight:** This phase is a pure extraction refactor — there is no new logic to invent, only to centralize what each site already does.

## Common Pitfalls

### Pitfall 1: Module-level Path Resolution
**What goes wrong:** `models_dir = resolve_models_dir()` at the top of a module is evaluated when the module is first imported. If the config is not yet loaded (or a test overrides it later), the path is stale.
**Why it happens:** Python evaluates module-level statements at import time.
**How to avoid:** Resolve inside functions or `__init__` methods, never at module scope.
**Warning signs:** Tests that override config after import still see old path.

### Pitfall 2: Default Argument Evaluated at Definition Time
**What goes wrong:** `def f(path=Path("metrics")):` bakes `Path("metrics")` at function definition time regardless of config.
**Why it happens:** Python default argument evaluation is eager.
**How to avoid:** Use `None` as default, resolve inside the function body.
**Warning signs:** Config change in YAML has no effect on the function even after reload.

### Pitfall 3: Tests Using monkeypatch.chdir
**What goes wrong:** After centralization, if the config returns an absolute path (e.g., from a test override), `monkeypatch.chdir(tmp_path)` no longer controls where artifacts land.
**Why it happens:** `chdir` only affects relative path resolution; absolute paths ignore it.
**How to avoid:** Keep defaults as relative strings in `default.yaml` and in the helper fallbacks. Tests that need absolute overrides should use `monkeypatch.setattr` on the loader, not chdir.
**Warning signs:** Tests that used `chdir` now write to unexpected locations.

### Pitfall 4: Incomplete Sweep — Missed Occurrences
**What goes wrong:** One hardcoded literal survives, and the structural grep test introduced in success criterion 2 fails.
**Why it happens:** Manual find-and-replace misses edge cases (string concatenation, f-strings, non-obvious patterns).
**How to avoid:** The structural test (`grep -r 'Path("models")\|Path("metrics")' src/`) must be run as part of the phase gate. Run it before declaring done.
**Warning signs:** Any `Path("models"` or `Path("metrics"` in grep output after the sweep.

### Pitfall 5: API Routes That Don't Have Config Context
**What goes wrong:** `bitbat/api/routes/health.py` and `bitbat/api/routes/analytics.py` have no config passed in — they call `Path("models")` inline in endpoint functions. The API server may start before `set_runtime_config()` is called.
**Why it happens:** FastAPI endpoint functions are called per-request; `get_runtime_config()` returns the cached config (or loads default if not set).
**How to avoid:** `resolve_models_dir()` inside the endpoint function is safe — it calls `get_runtime_config()` which lazy-loads. This is the same pattern as `bitbat.api.defaults._default_freq()`.

## Code Examples

Verified patterns from the actual codebase:

### All 15 Hardcoded Occurrences (complete inventory)

```
src/bitbat/model/evaluate.py:71      Path("metrics") / "window_diagnostics.json"   — default arg
src/bitbat/model/evaluate.py:131     Path("metrics")                                — default arg
src/bitbat/model/train.py:30         Path("models") / f"{freq}_{horizon}"           — _default_model_path()
src/bitbat/autonomous/predictor.py:59   Path("models")                              — __init__
src/bitbat/autonomous/continuous_trainer.py:235  Path("metrics") / f"continuous_diagnostics_..."  — inline
src/bitbat/autonomous/continuous_trainer.py:239  Path("models") / f"{...}" / "xgb.json"            — inline
src/bitbat/cli/commands/monitor.py:74    Path("metrics")                            — inline
src/bitbat/cli/commands/system.py:34     Path("models")                             — inline
src/bitbat/cli/commands/model.py:294     Path("metrics")                            — inline
src/bitbat/cli/commands/model.py:626     Path("metrics")                            — inline
src/bitbat/autonomous/agent.py:99        Path("models") / f"{...}" / "xgb.json"     — _validate_model_preflight
src/bitbat/autonomous/retrainer.py:60    Path("metrics") / "cv_summary.json"        — _cv_summary_path()
src/bitbat/api/routes/health.py:95      Path("models") / f"{...}" / "xgb.json"     — _check_model()
src/bitbat/api/routes/analytics.py:73   Path("models") / f"{...}" / "xgb.json"     — feature_importance()
src/bitbat/api/routes/analytics.py:98   Path("models") / f"{...}" / "xgb.json"     — system_status()
src/bitbat/backtest/metrics.py:105      Path("metrics")                             — summary()
```

Total: 10 `Path("models")` + 9 `Path("metrics")` = **16 occurrences** across 10 files.
(Note: grep returned 16 lines, not 15 as stated in the requirement — all must be replaced.)

### Existing Analogous Pattern: data_dir (already config-driven)

```python
# LivePredictor.__init__ (predictor.py line 58):
self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()

# AutoRetrainer.__init__ (retrainer.py line 45):
self.data_dir = Path(str(config.get("data_dir", "data"))).expanduser()
```

The new `resolve_models_dir()` / `resolve_metrics_dir()` functions follow this exact convention.

### Structural Grep Test (success criterion 2 gating test)

```python
# tests/test_path_centralization.py
import subprocess
import pytest

def test_no_hardcoded_models_path() -> None:
    """Ensure no literal Path('models') exists in src/."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "Path(\"models\")", "src/"],
        capture_output=True, text=True
    )
    assert result.stdout == "", f"Hardcoded Path('models') found:\n{result.stdout}"

def test_no_hardcoded_metrics_path() -> None:
    """Ensure no literal Path('metrics') exists in src/."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "Path(\"metrics\")", "src/"],
        capture_output=True, text=True
    )
    assert result.stdout == "", f"Hardcoded Path('metrics') found:\n{result.stdout}"
```

### Config Redirect Test (success criterion 3 gating test)

```python
def test_config_redirect_models_dir(tmp_path, monkeypatch):
    """Changing models_dir in config redirects resolve_models_dir."""
    from bitbat.config import loader
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", {"models_dir": str(tmp_path / "custom_models")})
    result = loader.resolve_models_dir()
    assert result == tmp_path / "custom_models"

def test_config_redirect_metrics_dir(tmp_path, monkeypatch):
    """Changing metrics_dir in config redirects resolve_metrics_dir."""
    from bitbat.config import loader
    monkeypatch.setattr(loader, "_ACTIVE_CONFIG", {"metrics_dir": str(tmp_path / "custom_metrics")})
    result = loader.resolve_metrics_dir()
    assert result == tmp_path / "custom_metrics"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Path("models")` inline | `resolve_models_dir()` | Phase 33 | Operators can set `models_dir` in YAML |
| `Path("metrics")` inline | `resolve_metrics_dir()` | Phase 33 | Operators can set `metrics_dir` in YAML |
| `data_dir` also hardcoded | `config.get("data_dir", "data")` | Pre-existing (done correctly) | Existing pattern to follow |

**Deprecated/outdated:**
- Direct `Path("models")` construction in business logic — replaced by `resolve_models_dir()`
- Direct `Path("metrics")` construction in business logic — replaced by `resolve_metrics_dir()`

## Open Questions

1. **Should `resolve_models_dir()` return an absolute path or a relative Path?**
   - What we know: `data_dir` is kept as a relative string and resolved via `Path(...).expanduser()` (not `.resolve()`), so it remains relative unless prefixed with `~`.
   - What's unclear: Whether the success criterion "verified by a structural grep test or linter rule" requires a ruff rule (requires custom plugin) or a pytest structural test.
   - Recommendation: Use a pytest structural test (simpler, no plugin needed). Ruff custom rules require significant tooling investment not warranted for a single check.

2. **`model/evaluate.py` default arguments with `Path("metrics")`**
   - What we know: Two functions use `Path("metrics")` as a default arg value, which is evaluated at import time.
   - What's unclear: Whether any external caller (outside `src/bitbat/`) passes these defaults explicitly.
   - Recommendation: Change default to `None`, resolve inside the function. Run `grep -r "write_window_diagnostics\|write_regression_metrics"` in tests to confirm no test passes an explicit path relying on the old default.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `poetry run pytest tests/config/ tests/model/test_train.py tests/backtest/test_metrics.py -x` |
| Full suite command | `poetry run pytest` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEBT-02 | `resolve_models_dir()` returns config-driven path | unit | `poetry run pytest tests/config/test_path_resolution.py -x` | ❌ Wave 0 |
| DEBT-02 | `resolve_metrics_dir()` returns config-driven path | unit | `poetry run pytest tests/config/test_path_resolution.py -x` | ❌ Wave 0 |
| DEBT-02 | No `Path("models")` literal in `src/` | structural | `poetry run pytest tests/config/test_path_resolution.py::test_no_hardcoded_models_path -x` | ❌ Wave 0 |
| DEBT-02 | No `Path("metrics")` literal in `src/` | structural | `poetry run pytest tests/config/test_path_resolution.py::test_no_hardcoded_metrics_path -x` | ❌ Wave 0 |
| DEBT-02 | Existing model training still saves to expected location | behavioral | `poetry run pytest tests/model/test_train.py -x` | ✅ exists |
| DEBT-02 | Existing backtest metrics still writes to expected location | behavioral | `poetry run pytest tests/backtest/test_metrics.py -x` | ✅ exists |

### Sampling Rate
- **Per task commit:** `poetry run pytest tests/config/ tests/model/test_train.py tests/backtest/test_metrics.py -x`
- **Per wave merge:** `poetry run pytest`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/config/test_path_resolution.py` — covers DEBT-02 (structural grep tests + config redirect unit tests)

*(Existing tests in `tests/model/test_train.py` and `tests/backtest/test_metrics.py` use `monkeypatch.chdir(tmp_path)` and will continue to pass if defaults remain relative — no changes needed there.)*

## Sources

### Primary (HIGH confidence)
- Grep scan of `src/bitbat/` — all 16 hardcoded path occurrences inventoried directly
- `src/bitbat/config/loader.py` — existing config infrastructure read directly
- `src/bitbat/config/default.yaml` — current config keys verified directly
- `tests/backtest/test_metrics.py`, `tests/model/test_train.py` — test patterns verified directly

### Secondary (MEDIUM confidence)
- Python docs on default argument evaluation — well-established language behavior

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pure internal refactor, no new dependencies
- Architecture: HIGH — direct code inspection of all 16 call sites and existing config loader
- Pitfalls: HIGH — identified from direct code analysis and established Python patterns

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable internal codebase)
