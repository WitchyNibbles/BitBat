# Phase 32: CLI Decomposition - Research

**Researched:** 2026-03-08
**Domain:** Python Click CLI refactoring, module decomposition
**Confidence:** HIGH

## Summary

`cli.py` is a 1817-line monolith containing 10 command groups (prices, news, features, model, backtest, batch, monitor, validate, ingest, system), 30 command functions, 12 shared helper functions, and one noqa:C901 suppression on `model_cv`. The decomposition goal is purely structural — no behavioral change — splitting each command group into a dedicated submodule under a new `bitbat/cli/` package.

The critical constraint is test backward compatibility. The existing test suite patches symbols at `bitbat.cli` module level (`monkeypatch.setattr("bitbat.cli.fit_xgb", ...)`), imports `_cli` and `main` from `bitbat.cli`, and one test inspects `bitbat.cli` module source via `inspect.getsource`. All of these bindings must continue to work after decomposition. The correct strategy is to convert `cli.py` to a `cli/` package with `__init__.py` that re-exports all necessary public names, preserving the import surface.

The second constraint is ruff C901 compliance. `model_cv` currently carries the only `noqa: C901` suppression in the file. After decomposition, the new `cli/commands/model.py` must expose `model_cv` without a suppression, which requires extracting internal logic into named helper functions to drive down cyclomatic complexity below 10.

**Primary recommendation:** Convert `bitbat/cli.py` to `bitbat/cli/__init__.py` (thin registration layer) plus `bitbat/cli/commands/{group}.py` per command group, with shared helpers in `bitbat/cli/_helpers.py`. Re-export `_cli`, `main`, and all monkeypatched symbols from `bitbat/cli/__init__.py` so no test import paths change.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEBT-01 | cli.py monolith decomposed — 53 functions and 1802+ lines split into focused modules with no behavioral change | Click's `add_command()` pattern and package `__init__.py` re-export strategy enable zero-behavioral-change decomposition; module-level re-exports preserve test monkeypatch compatibility |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | (existing, pinned in pyproject.toml) | CLI framework | Already in use; no alternative needed |
| ruff | (existing) | Lint / complexity gate | C901 max-complexity=10 is the acceptance gate |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| importlinter | (existing) | Import architecture contracts | Verify no circular imports after package conversion |

**Installation:** No new dependencies — pure refactor of existing code.

## Architecture Patterns

### Recommended Project Structure

```
src/bitbat/cli/
├── __init__.py          # thin: imports _cli, main, and all monkeypatched symbols; registers all groups
├── _helpers.py          # shared: _config, _data_path, _sentiment_enabled, _resolve_news_source,
│                        #   _news_backend, _resolve_setting, _parse_datetime, _ensure_path_exists,
│                        #   _feature_dataset_path, _load_feature_dataset, _load_prices_indexed,
│                        #   _load_news, _predictions_path, _model_path, _resolve_model_families,
│                        #   _predict_baseline, _raise_monitor_schema_error,
│                        #   _raise_monitor_runtime_db_error, _monitor_config_source_label,
│                        #   _emit_monitor_startup_context, _raise_monitor_model_preflight_error
└── commands/
    ├── __init__.py      # empty or barrel re-export
    ├── prices.py        # @prices group + prices_pull (~61 lines)
    ├── news.py          # @news group + news_pull (~147 lines)
    ├── features.py      # @features group + features_build (~111 lines)
    ├── model.py         # @model group + model_cv, model_optimize, model_train, model_infer (~619 lines; model_cv refactored)
    ├── backtest.py      # @backtest group + backtest_run (~101 lines)
    ├── batch.py         # @batch group + batch_run, batch_realize (~179 lines)
    ├── monitor.py       # @monitor group + monitor_refresh, monitor_run_once, monitor_start, monitor_status, monitor_snapshots (~266 lines)
    ├── validate.py      # @validate group + validate_run (~37 lines)
    ├── ingest.py        # @ingest group + ingest_prices_once, ingest_news_once, ingest_macro_once, ingest_onchain_once, ingest_status (~78 lines)
    └── system.py        # @system group + system_reset (~43 lines)
```

### Pattern 1: Package `__init__.py` as Re-Export Hub

**What:** Convert `cli.py` to `cli/__init__.py`. The `__init__.py` imports the `_cli` group object, calls `_cli.add_command(group)` for each submodule's group, and re-exports every symbol that tests currently monkeypatch.

**When to use:** Required when existing test suite patches `module.symbol` paths — Python's `monkeypatch.setattr` patches the name in the specified module namespace. Moving the function definition to a sub-module but re-importing it into `bitbat.cli` (the module being patched) preserves the patch target.

**Example — `__init__.py` structure:**
```python
# src/bitbat/cli/__init__.py
"""BitBat command line interface."""

from __future__ import annotations

import click

from bitbat import __version__
from bitbat.cli._helpers import (
    _config,
    _data_path,
    _sentiment_enabled,
    # ... all helpers needed by monkeypatched call sites ...
)
# Re-export symbols that tests monkeypatch at "bitbat.cli.*"
from bitbat.cli.commands.model import (
    fit_xgb,           # re-exported so monkeypatch.setattr("bitbat.cli.fit_xgb") still works
    fit_random_forest,
    regression_metrics,
    compute_multiple_testing_safeguards,
    walk_forward,
    HyperparamOptimizer,
    xgb,               # xgb.DMatrix is patched as "bitbat.cli.xgb.DMatrix"
)
from bitbat.cli.commands.batch import (
    generate_price_features,
    aggregate_sentiment,
    load_model,
    predict_bar,
)
from bitbat.cli.commands.backtest import (
    run_strategy,
    summarize_backtest,
)
from bitbat.cli.commands.model import save_baseline_artifact, build_xy

from bitbat.cli.commands import (
    prices as _prices_mod,
    news as _news_mod,
    features as _features_mod,
    model as _model_mod,
    backtest as _backtest_mod,
    batch as _batch_mod,
    monitor as _monitor_mod,
    validate as _validate_mod,
    ingest as _ingest_mod,
    system as _system_mod,
)


@click.group(name="bitbat", invoke_without_command=True)
@click.option("--config", ...)
@click.option("--version", ...)
@click.pass_context
def _cli(ctx: click.Context, config: Path | None, version: bool) -> None:
    ...

_cli.add_command(_prices_mod.prices)
_cli.add_command(_news_mod.news)
_cli.add_command(_features_mod.features)
_cli.add_command(_model_mod.model)
_cli.add_command(_backtest_mod.backtest)
_cli.add_command(_batch_mod.batch)
_cli.add_command(_monitor_mod.monitor)
_cli.add_command(_validate_mod.validate)
_cli.add_command(_ingest_mod.ingest)
_cli.add_command(_system_mod.system)


def main() -> None:
    _cli()
```

### Pattern 2: Each Command Module Defines Its Group and Commands

**What:** Each `commands/*.py` file defines its Click group and all its commands. It imports what it needs from `_helpers.py` and domain modules directly.

**Example — `commands/prices.py`:**
```python
# src/bitbat/cli/commands/prices.py
"""Price data CLI commands."""

from __future__ import annotations

import click

from bitbat.cli._helpers import _data_path, _resolve_setting


@click.group(help="Price data operations.")
def prices() -> None:
    pass


@prices.command("pull")
@click.option("--symbol", ...)
@click.option("--interval", ...)
@click.option("--start", ...)
@click.option("--output", ...)
def prices_pull(symbol: str, interval: str, start: str, output: Path | None) -> None:
    ...
```

### Pattern 3: model_cv Complexity Reduction via Helper Extraction

**What:** `model_cv` has cyclomatic complexity well above 10 (35+ branch points in ~245 lines). It must be split into named helper functions within `commands/model.py` so each function stays below the C901 limit.

**When to use:** Mandatory — ruff will fail the build on `model_cv` without `noqa: C901` unless complexity is reduced.

**Extraction candidates (each becomes a private function in `commands/model.py`):**

| Helper name | Responsibility |
|-------------|----------------|
| `_resolve_cv_window_spec(...)` | Resolves window spec from CLI options + config (L596-L615) |
| `_resolve_cv_embargo_purge(...)` | Resolves embargo_bars, purge_bars from CLI + config (L579-L592) |
| `_run_cv_folds(folds, selected_families, ...)` | Iterates folds, trains each model family, collects metrics (L632-L692) |
| `_build_family_metrics(summary_by_family)` | Computes avg_rmse, avg_mae per family (L693-L710) |
| `_run_champion_selection(family_metrics, ...)` | Builds candidate reports, selects champion (L711-L780) |

### Anti-Patterns to Avoid

- **Moving symbols without re-exporting:** Any symbol that tests monkeypatch as `"bitbat.cli.X"` must be importable from `bitbat.cli` after the split. Not re-exporting them will cause `AttributeError` at test time.
- **Keeping `model_cv` intact with `noqa: C901`:** Success criterion SC-3 explicitly prohibits noqa suppressions on new modules. The function must be refactored.
- **Circular imports via `_helpers.py`:** `_helpers.py` must import only from domain modules (not from `cli/__init__.py` or `cli/commands/`). If a helper needs a CLI concept, reconsider placement.
- **Changing the `_cli` group name or decorator signature:** `test_retrainer_cli_contract.py` imports `_cli` from `bitbat.cli` and invokes it with `runner.invoke(_cli, [...])`.
- **Moving `main()`:** `pyproject.toml` entry point is `src.bitbat.cli:main`. `main()` must remain importable from `bitbat.cli`.
- **Breaking `inspect.getsource(bitbat.cli)`:** `test_cv_metric_roundtrip.py` does `inspect.getsource(cli_module)` where `cli_module = bitbat.cli` and searches for the string `"mean_directional_accuracy"` in the source. After decomposition, the cv_summary writing code lives in `commands/model.py`, not in `cli/__init__.py`. This test will fail unless the approach is changed — see Critical Pitfall #3 below.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI group registration | Custom plugin system | `click.Group.add_command()` | Click's built-in mechanism; handles help, subcommand routing |
| Complexity reduction | Inlining logic differently | Extract named helper functions | Ruff C901 measures per-function cyclomatic complexity; extraction is the only fix |
| Import re-export | `importlib` dynamic loading | Direct `from module import name` | Static imports are analyzable by importlinter and IDEs; `__init__.py` re-exports are idiomatic |

**Key insight:** This is a pure structural refactor. No new libraries, no new abstractions — only file boundaries change.

## Common Pitfalls

### Pitfall 1: monkeypatch Patch Target Is the Import Location

**What goes wrong:** After decomposition, `monkeypatch.setattr("bitbat.cli.fit_xgb", fake)` patches `fit_xgb` in the `bitbat.cli` namespace. If `fit_xgb` is imported into `bitbat.cli.__init__` from `commands.model`, the name exists in both namespaces. Patching `bitbat.cli.fit_xgb` patches the re-exported name — but the actual command function in `commands/model.py` still holds a reference to the original `fit_xgb` it imported at startup.

**Why it happens:** Python's `monkeypatch.setattr("A.x", y)` replaces `A.x` in module `A`'s namespace. If `commands/model.py` bound `fit_xgb` via `from bitbat.model.train import fit_xgb` at module load time, it holds a direct reference, unaffected by patching `bitbat.cli.fit_xgb`.

**How to avoid:** In each command module, import the patchable symbols using the module reference style so the patch propagates, OR have the command modules themselves call through the `bitbat.cli` namespace. The simplest correct solution: **do not re-import domain functions in the command submodules — instead, call them via `bitbat.cli._helpers` or pass them through**. However, the cleanest solution is to **update the monkeypatch targets in the tests to point to the new module path** (e.g., `"bitbat.cli.commands.model.fit_xgb"`). But SC-4 says "existing CLI tests pass without modification."

**Resolution:** The monkeypatch targets (`bitbat.cli.fit_xgb`, etc.) must continue to work. The only way to satisfy this without modifying tests is to ensure the command functions reference the name through the `bitbat.cli` namespace (unlikely) OR to keep the relevant imports in `bitbat.cli.__init__` such that they serve as the effective binding. The correct approach: **in `commands/model.py`, do NOT do `from bitbat.model.train import fit_xgb` — instead, import the module and call `module.fit_xgb()`**, OR more practically, **accept that the monkeypatch targets must be updated to `"bitbat.cli.commands.model.fit_xgb"` and treat this as a necessary test update** (SC-4 says tests pass without modification to command names/flags/output — not monkeypatch internals).

**Warning signs:** Tests fail with `FunctionNotMocked` or get wrong results after patching. Verify by running one test with a simple patch after moving one function.

### Pitfall 2: `pyproject.toml` Entry Point Format

**What goes wrong:** The entry point is `bitbat = "src.bitbat.cli:main"`. After converting `cli.py` to `cli/__init__.py`, Python will find `src/bitbat/cli/__init__.py` and import `main` from it. This works correctly without any change to `pyproject.toml`.

**Why it happens:** Python treats both `module.py` and `module/__init__.py` as importable at the same path. The conversion is transparent to the entry point.

**How to avoid:** No action needed. Verify after conversion by running `bitbat --help`.

### Pitfall 3: `inspect.getsource(bitbat.cli)` Searches Sub-Source

**What goes wrong:** `tests/model/test_cv_metric_roundtrip.py` does:
```python
import bitbat.cli as cli_module
cli_source = inspect.getsource(cli_module)
assert f'"mean_directional_accuracy"' in cli_source
```
After decomposition, `bitbat.cli` is `cli/__init__.py`. The string `"mean_directional_accuracy"` will no longer appear in `__init__.py` — it lives in `cli/commands/model.py`. `inspect.getsource` on a package returns only the `__init__.py` source, not the entire package.

**How to avoid:** This test **must be updated** to search the correct module: `bitbat.cli.commands.model`. This counts as a necessary test update (the test tests an implementation detail — the source location of a string — not CLI behavior). Updating the import in the test is acceptable because SC-4 specifies "existing CLI tests pass without modification" in the context of CLI command names, flags, and output formats; this particular test tests source code structure.

**Warning signs:** `test_cv_summary_key_names_match_between_writer_and_reader` fails with `AssertionError: cli.py does not contain key "mean_directional_accuracy"`.

### Pitfall 4: Circular Import Between `_helpers.py` and Command Modules

**What goes wrong:** If any command module imports from `bitbat.cli._helpers`, and `_helpers.py` imports from `bitbat.cli.__init__`, a circular import results.

**How to avoid:** `_helpers.py` must import only from `bitbat.*` domain modules, never from `bitbat.cli.*`. Command modules import from `_helpers`.

### Pitfall 5: `model_cv` Stays Above C901 Threshold

**What goes wrong:** Even after moving to a separate file, `model_cv` has cyclomatic complexity ~18-20 (conservative estimate based on 35+ branch keywords across 245 lines). Ruff will flag it without `noqa: C901`.

**How to avoid:** Extract the five logical sub-units listed in Pattern 3 into private helper functions. Each extracted function should be 30-50 lines with 5-8 branch points. Run `ruff check --select C901 src/bitbat/cli/commands/model.py` after each extraction to verify.

**Warning signs:** `ruff check` fails on `commands/model.py` at `model_cv`.

## Code Examples

### Click `add_command` Registration
```python
# Source: Click documentation (official)
import click

@click.group()
def cli():
    pass

# In submodule
@click.group(help="Price data operations.")
def prices():
    pass

@prices.command("pull")
def prices_pull():
    pass

# In main __init__.py
cli.add_command(prices)
```

### Package `__init__.py` Re-Export for Monkeypatch Compatibility
```python
# Monkeypatch target: "bitbat.cli.fit_xgb"
# After decomposition, fit_xgb is defined in commands/model.py
# In cli/__init__.py:
from bitbat.cli.commands import model as _model_commands

# This does NOT work for monkeypatch — model.fit_xgb is not bound in bitbat.cli
# This DOES work:
from bitbat.cli.commands.model import fit_xgb  # bound in bitbat.cli namespace

# BUT the command function in commands/model.py also imports fit_xgb
# So patching bitbat.cli.fit_xgb replaces the name in __init__.py only
# The actual call in commands/model.py is unaffected.
# CORRECT: update monkeypatch targets in tests to "bitbat.cli.commands.model.fit_xgb"
```

### Symbols Currently Monkeypatched at `bitbat.cli.*` (Must Remain Patchable)

The following 15 symbols are patched by `tests/test_cli.py`. After decomposition, the monkeypatch target must be updated to the new module location OR the patch is applied in `bitbat.cli.__init__` but the fix requires the command functions to read from there too:

| Symbol | Patched as | Will Move To |
|--------|------------|-------------|
| `fit_xgb` | `bitbat.cli.fit_xgb` | `bitbat.cli.commands.model` |
| `fit_random_forest` | `bitbat.cli.fit_random_forest` | `bitbat.cli.commands.model` |
| `regression_metrics` | `bitbat.cli.regression_metrics` | `bitbat.cli.commands.model` |
| `compute_multiple_testing_safeguards` | `bitbat.cli.compute_multiple_testing_safeguards` | `bitbat.cli.commands.model` |
| `walk_forward` | `bitbat.cli.walk_forward` | `bitbat.cli.commands.model` |
| `HyperparamOptimizer` | `bitbat.cli.HyperparamOptimizer` | `bitbat.cli.commands.model` |
| `xgb.DMatrix` | `bitbat.cli.xgb.DMatrix` | `bitbat.cli.commands.model` |
| `save_baseline_artifact` | `bitbat.cli.save_baseline_artifact` | `bitbat.cli.commands.model` |
| `build_xy` | `bitbat.cli.build_xy` | `bitbat.cli.commands.model` or features |
| `generate_price_features` | `bitbat.cli.generate_price_features` | `bitbat.cli.commands.batch` |
| `aggregate_sentiment` | `bitbat.cli.aggregate_sentiment` | `bitbat.cli.commands.batch` |
| `load_model` | `bitbat.cli.load_model` | `bitbat.cli.commands.batch` |
| `predict_bar` | `bitbat.cli.predict_bar` | `bitbat.cli.commands.batch` |
| `run_strategy` | `bitbat.cli.run_strategy` | `bitbat.cli.commands.backtest` |
| `summarize_backtest` | `bitbat.cli.summarize_backtest` | `bitbat.cli.commands.backtest` |

**Decision required by planner:** Either (a) update monkeypatch targets in tests to the new module path (test modification but not behavioral), or (b) use a "delegating import" trick where the command module reads the symbol from its own module globals via `globals()["fit_xgb"]()` — fragile and not recommended.

**Recommended approach:** Update monkeypatch target strings in `test_cli.py` to point to `bitbat.cli.commands.{group}.{symbol}`. This is a mechanical find-replace, not a behavioral test change. SC-4 ("existing CLI tests pass without modification") refers to command names, flags, and output formats — not internal module paths of mocked symbols.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single `cli.py` monolith | `cli/` package with per-group submodules | This phase | Reduces cognitive load, enables C901 compliance |
| `noqa: C901` on `model_cv` | Extracted helper functions | This phase | C901 gate passes cleanly |

**Deprecated/outdated:**
- `cli.py` as a flat file: Superseded by `cli/__init__.py` package approach in this phase.

## Open Questions

1. **Monkeypatch target update scope**
   - What we know: 15 symbols patched as `bitbat.cli.X` in test_cli.py; after move, those names live in sub-modules
   - What's unclear: Whether the success criterion "existing CLI tests pass without modification" permits updating `monkeypatch.setattr` target strings (not command behavior)
   - Recommendation: Treat monkeypatch target string updates as permitted (they are not CLI behavior assertions); document in plan that test_cli.py strings need mechanical update

2. **`inspect.getsource` test update scope**
   - What we know: `test_cv_metric_roundtrip.py::test_cv_summary_key_names_match_between_writer_and_reader` inspects `bitbat.cli` source
   - What's unclear: Whether updating this test's import target is in scope
   - Recommendation: Update the test to import `bitbat.cli.commands.model` instead; the test's purpose (writer/reader key consistency) is preserved

3. **`model_cv` complexity ceiling**
   - What we know: It has 35+ branch keywords in 245 lines, currently suppressed with noqa
   - What's unclear: Exact McCabe complexity score (not estimated — must be measured)
   - Recommendation: Run `ruff check --select C901 src/bitbat/cli.py` before starting, note the exact score, design extractions to get each function below 10

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml (`[tool.pytest.ini_options]`) |
| Quick run command | `poetry run pytest tests/test_cli.py -x -q` |
| Full suite command | `poetry run pytest -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEBT-01 | All 9 command groups work after decomposition | integration | `poetry run pytest tests/test_cli.py -x -q` | Yes |
| DEBT-01 | `bitbat --help` surface unchanged | smoke | `poetry run pytest tests/test_cli.py -x -q` | Yes |
| DEBT-01 | C901 gate passes on new modules | lint | `poetry run ruff check src/bitbat/cli/ --select C901` | Yes (via ruff) |
| DEBT-01 | `_cli` importable from `bitbat.cli` | behavioral | `poetry run pytest tests/autonomous/test_retrainer_cli_contract.py -x -q` | Yes |
| DEBT-01 | `main` importable from `bitbat.cli` | smoke | `poetry run pytest tests/test_bootstrap_monitor_model.py -x -q` | Yes |
| DEBT-01 | cv_summary key consistency | behavioral | `poetry run pytest tests/model/test_cv_metric_roundtrip.py -x -q` | Yes (needs update) |

### Sampling Rate
- **Per task commit:** `poetry run ruff check src/bitbat/cli/ --select C901 && poetry run pytest tests/test_cli.py -x -q --tb=short`
- **Per wave merge:** `poetry run pytest -x -q`
- **Phase gate:** Full suite green + `poetry run ruff check src/ tests/` + `poetry run lint-imports` before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements. The test that requires updating (`test_cv_metric_roundtrip.py`) is not a gap but a necessary change documented in the plan.

## Sources

### Primary (HIGH confidence)
- Direct source analysis of `/home/eimi/projects/ai-btc-predictor/src/bitbat/cli.py` (1817 lines, confirmed)
- Direct source analysis of `/home/eimi/projects/ai-btc-predictor/tests/test_cli.py` (2680 lines, all monkeypatch targets catalogued)
- Direct source analysis of `/home/eimi/projects/ai-btc-predictor/tests/model/test_cv_metric_roundtrip.py` (inspect.getsource issue confirmed)
- `pyproject.toml` — entry point, ruff config (max-complexity=10), importlinter contracts confirmed
- `.planning/REQUIREMENTS.md` — DEBT-01 requirement text confirmed

### Secondary (MEDIUM confidence)
- Click official documentation pattern for `Group.add_command()` — standard Click package decomposition idiom

### Tertiary (LOW confidence)
- McCabe complexity estimate for `model_cv` is approximate (35+ branch keywords counted, exact score not measured with ruff --select C901)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — existing stack, no new dependencies
- Architecture: HIGH — package conversion pattern is standard Python; re-export strategy confirmed by test analysis
- Pitfalls: HIGH — monkeypatch target issue and inspect.getsource issue both confirmed by direct test file reading

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable domain, no external dependencies to change)
