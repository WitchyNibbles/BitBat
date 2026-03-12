# Coding Conventions

**Analysis Date:** 2026-02-24

## Naming Patterns

**Files:**
- Snake case: `src/bitbat/features/sentiment.py`, `src/bitbat/ingest/news_gdelt.py`
- Module files match function/class exports: `train.py` exports `fit_xgb()`
- Contracts: `contracts.py` contains schema validators
- Tests: `test_*.py` or `*_test.py` naming

**Functions:**
- Snake case with verbs: `fit_xgb()`, `ensure_prices_contract()`, `forward_return()`, `run_strategy()`
- Private functions start with underscore: `_ensure_datetime()`, `_config()`, `_resolve_news_source()`
- Type hints required on all functions (mypy strict mode)

**Variables:**
- Snake case: `X_train`, `y_train`, `output_dir`, `raw_importance`
- DataFrame columns follow domain conventions: `timestamp_utc`, `feat_*` (feature prefix), `label`, `r_forward`
- Constants in UPPER_CASE: `seed`, configuration keys are lowercase in YAML

**Types:**
- PEP 484 type hints on all parameters and returns: `def fit_xgb(X_train: pd.DataFrame, y_train: pd.Series, *, seed: int = 42) -> tuple[xgb.Booster, dict[str, float]]`
- Use `from __future__ import annotations` for forward references
- Generic types: `dict[str, float]`, `tuple[str, ...]`, `Iterable[T]`
- Type guards via contracts: `ContractError` raised for schema violations

## Code Style

**Formatting:**
- Line length: 100 characters (Black + Ruff configured in `pyproject.toml`)
- Target: Python 3.11+
- Black with `preview = true` for modern formatting
- Ruff format enabled with `preview = true`

**Linting:**
- Ruff rules: `["E", "F", "B", "I", "UP", "S", "C4", "RET", "SIM"]`
- Ignored: `["E203", "B008"]` (Black compatibility)
- Test files exempt: `S101` (assert), `S608` (hardcoded SQL)
- Check before commit: `poetry run ruff check src tests`
- Format: `poetry run ruff format src tests && poetry run black src tests`

**Type Checking:**
- mypy strict: `disallow_incomplete_defs = true`, `disallow_untyped_defs = true`, `strict_optional = true`
- Config at `pyproject.toml` section `[tool.mypy]`
- Run: `poetry run mypy src tests`

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations`
2. Standard library: `from pathlib import Path`, `import json`
3. Third-party: `import pandas as pd`, `import xgboost as xgb`
4. Local: `from bitbat.contracts import ContractError`, `from bitbat.config.loader import get_runtime_config`

**Path Aliases:**
- None configured; use absolute imports from `src/`: `from bitbat.features.sentiment import aggregate`
- Paths must be added to PYTHONPATH in `pyproject.toml`: `pythonpath = ["src"]`

**Import Style:**
- Explicit imports preferred: `from bitbat.ingest import prices as prices_module`
- Dynamic imports for optional backends: See `_news_backend()` in `src/bitbat/cli.py` (checks available newsfeeds)

## Error Handling

**Patterns:**
- Custom exception class for schema violations: `class ContractError(ValueError)` in `src/bitbat/contracts.py`
- Raise early on validation: `missing = set(expected) - set(frame.columns); if missing: raise ContractError(...)`
- Type checking: `if not isinstance(X_train, pd.DataFrame): raise TypeError(...)`
- Click CLI errors: `raise click.ClickException("message")`
- Use try/except for optional dependencies: `try: import xgboost; except ImportError: pytest.skip()`

**Example from `src/bitbat/contracts.py`:**
```python
def ensure_prices_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the prices parquet contract."""
    expected = ["timestamp_utc", "open", "high", "low", "close", "volume", "source"]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ContractError(f"Prices frame missing columns: {sorted(missing)}")
    # ... normalize and return
```

## Logging

**Framework:** Standard `logging` module (no explicit setup visible)

**Patterns:**
- Not explicitly visible; codebase appears to use print/Click for CLI feedback
- Status messages via Click: `click.echo()`, `click.secho()` for colors
- Silent operation by default; verbosity controlled via CLI flags

## Comments

**When to Comment:**
- Docstrings required on all public functions and classes
- Explain WHY, not WHAT (code is self-documenting)
- Mark exclusions/workarounds: `# pragma: no cover` for optional dependencies
- Mark linting exemptions where necessary

**JSDoc/TSDoc:**
- Use triple-quote docstrings with parameter descriptions
- Example from `src/bitbat/model/train.py`:
```python
def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train an XGBoost regression model and persist it to disk.

    Persists the trained booster to `models/{freq}_{horizon}/xgb.json` when
    `X_train` includes the `freq` and `horizon` attributes on `.attrs`.

    Args:
        X_train: Feature matrix for training, using numeric columns.
        y_train: Continuous forward returns (float64) aligned to `X_train`.
        seed: Random seed for model training.

    Returns:
        The trained booster and a gain-based importance mapping keyed by feature name.
    """
```

## Function Design

**Size:** Functions are typically 20-50 lines with clear responsibilities

**Parameters:**
- Keyword-only parameters after *: `fit_xgb(X_train, y_train, *, seed: int = 42)`
- DataFrame/Series as first parameters for transformation functions
- Required parameters before optional

**Return Values:**
- Explicit tuple returns: `tuple[xgb.Booster, dict[str, float]]`
- Modified DataFrame returned from contract validators
- None for side-effect functions (e.g., model persistence)

## Module Design

**Exports:**
- Single responsibility per module (e.g., `train.py` = training, `ingest/prices.py` = price fetching)
- Public functions at module level
- Private utilities prefixed with underscore
- Import contracts at module entry (validate immediately)

**Barrel Files:**
- Minimal use; `__init__.py` files typically empty or import exceptions
- Example: `src/bitbat/ingest/__init__.py` may export common functions

---

*Convention analysis: 2026-02-24*
