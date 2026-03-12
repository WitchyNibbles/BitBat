# Testing Patterns

**Analysis Date:** 2026-02-24

## Test Framework

**Runner:**
- pytest 8.2.0+
- Config: `pyproject.toml` section `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest native assertions with type hints

**Run Commands:**
```bash
poetry run pytest                           # Run all tests in tests/
poetry run pytest tests/model/test_train.py # Run single test file
poetry run pytest -k test_name              # Run by name pattern
poetry run pytest --co                      # List all tests
poetry run pytest -m slow                   # Run marked tests
```

**Watch/Coverage:**
- No watch mode configured
- Coverage tracking: pytest can generate via plugins (not explicitly configured)

## Test File Organization

**Location:**
- Mirror source structure: `src/bitbat/model/train.py` → `tests/model/test_train.py`
- Separate `tests/` directory at project root
- Co-located fixtures in same test file

**Naming:**
- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_<function_or_feature>()`
- Test classes: `Test<Feature>` for grouped tests

**Structure:**
```
tests/
├── analytics/          # Feature analysis, Monte Carlo, SHAP
├── autonomous/         # Monitoring agent tests
├── backtest/          # Strategy engine tests
├── contracts/         # Schema validation tests
├── dataset/           # Dataset assembly and splits
├── features/          # Feature engineering (leakage checks critical)
├── gui/               # Streamlit widget tests
├── model/             # Training, inference, evaluation
├── timealign/         # Calendar, bucketing, leakage prevention
└── conftest.py        # Pytest fixtures (if present)
```

## Test Structure

**Suite Organization:**
```python
# From tests/model/test_train.py
def test_fit_xgb_trains_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(50, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    X.attrs["freq"] = "1h"
    X.attrs["horizon"] = "2h"
    y = pd.Series(rng.normal(0.0, 0.01, size=50))

    monkeypatch.chdir(tmp_path)
    booster, importance = fit_xgb(X, y, seed=0)

    assert booster is not None
    assert set(importance.keys()) == set(X.columns)
    model_path = Path("models") / "1h_2h" / "xgb.json"
    assert model_path.exists()
```

**Patterns:**
- Fixtures via function parameters: `tmp_path: Path` (pytest built-in), `monkeypatch: pytest.MonkeyPatch`
- Arrange-Act-Assert structure (implicit in naming)
- Type hints on test functions: `-> None`
- Random seed pinning: `rng = np.random.default_rng(0)` for reproducibility

**Setup/Teardown:**
- Use pytest fixtures for setup: `@pytest.fixture`
- Use `monkeypatch` for environment changes
- Use `tmp_path` for temporary directories (auto-cleanup)
- No explicit teardown needed (pytest handles cleanup)

## Mocking

**Framework:**
- `unittest.mock` (standard library)
- `pytest-mock` may be available (not explicitly in deps, but standard)
- Optional dependency guards: `try: import xgboost; except ImportError: pytest.skip()`

**Patterns:**
```python
# From tests/model/test_train.py (implicit mocking via fixtures)
def test_fit_xgb_trains_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch.chdir() replaces actual filesystem access
    monkeypatch.chdir(tmp_path)
```

**What to Mock:**
- External API calls (yfinance, FRED, blockchain.info) - see `tests/ingest/`
- File system via `tmp_path` fixture
- Configuration via `monkeypatch`
- Clock/time dependencies

**What NOT to Mock:**
- pandas operations (too broad; test real DataFrames)
- Contract validators (must test actual validation logic)
- Model persistence (test actual file writes to temp dirs)

## Fixtures and Factories

**Test Data:**
```python
# From tests/contracts/test_contracts.py
def test_feature_contract_happy_path() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="1h"),
        "feat_a": [0.1, 0.2],
        "label": ["up", "down"],
        "r_forward": [0.05, -0.02],
    })

    validated = ensure_feature_contract(
        frame,
        require_label=True,
        require_forward_return=True,
        require_features_full=False,
    )
```

**Location:**
- Inline construction in test functions (small datasets)
- Reusable fixtures in `conftest.py` if shared
- Factory functions for complex objects (not yet visible in codebase)

**Markers:**
- `@pytest.mark.slow` for network/long-running tests
- Configured in `pyproject.toml`: `markers = ["slow: marks tests as slow and may require network access"]`

## Coverage

**Requirements:**
- No explicit coverage requirement enforced
- Target: Cover all contract validators, feature engineering, model training/inference

**View Coverage:**
- Use `pytest --cov=bitbat` (requires pytest-cov plugin)
- No coverage config in `pyproject.toml` yet

## Test Types

**Unit Tests:**
- Scope: Single function or small class
- Approach: Fast, deterministic, use fixtures for dependencies
- Example: `test_fit_xgb_trains_and_saves()` in `tests/model/test_train.py`
- Assertions: Check return values, side effects (file writes), data shapes

**Integration Tests:**
- Scope: Multiple modules working together (e.g., feature engineering → dataset assembly)
- Approach: Use temporary directories, real pandas operations, contract validation
- Example: Tests in `tests/dataset/` validating walk-forward splits with embargo bars
- Assertions: Check output Parquet structure, label correctness, no leakage

**Contract/Schema Tests:**
- Location: `tests/contracts/test_contracts.py`
- Scope: Validate schema enforcement via `ContractError`
- Example: `test_feature_contract_requires_feat_prefix()` ensures required columns exist
- Pattern: Test both happy path (valid data) and error cases (missing columns, wrong types)

**E2E Tests:**
- Framework: Not explicitly present
- Approach: CLI tests may serve E2E role
- Tests in `tests/test_cli.py` validate command execution

**Leakage Detection Tests:**
- Critical file: `tests/features/test_leakage.py`
- Scope: Ensure no future data in features
- Approach: Train/test on walk-forward splits, check PR-AUC guardrail
- Pattern: Feature engineering → dataset → model → check metrics don't exceed threshold

## Common Patterns

**Async Testing:**
- Not applicable (synchronous codebase)

**Error Testing:**
```python
# From tests/contracts/test_contracts.py
def test_feature_contract_requires_feat_prefix() -> None:
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="1h"),
        "value": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ContractError):
        ensure_feature_contract(
            frame, require_label=False, require_forward_return=False
        )
```

**Class-Based Tests:**
```python
# From tests/analytics/test_monte_carlo.py (inferred structure)
class TestMonteCarloSimulator:
    def test_accepts_series(self) -> None:
        # test implementation

    def test_accepts_array(self, returns: np.ndarray) -> None:
        # test with fixture parameter
```

**Fixture Usage:**
- Built-in: `tmp_path` (temporary directory), `monkeypatch` (environment changes)
- Custom fixtures: Define in `conftest.py` or inline
- Parametrized fixtures: `@pytest.mark.parametrize()` for multiple inputs

## Test Execution

**Configuration:**
- `pythonpath = ["src"]` in `pyproject.toml` enables imports
- `testpaths = ["tests"]` tells pytest where to find tests
- `minversion = "8.0"` requires pytest 8+
- `addopts = "-ra"` shows test result summary

**Skip/Mark Examples:**
```python
# Skip optional dependencies
try:
    import xgboost
except ImportError:
    pytest.skip("xgboost not installed", allow_module_level=True)

# Skip slow tests
@pytest.mark.slow
def test_expensive_operation() -> None:
    pass
```

**Total Tests:** 63 test functions across 9 modules
- Located: `tests/analytics/`, `tests/autonomous/`, `tests/backtest/`, `tests/contracts/`, `tests/dataset/`, `tests/features/`, `tests/gui/`, `tests/model/`, `tests/timealign/`

---

*Testing analysis: 2026-02-24*
