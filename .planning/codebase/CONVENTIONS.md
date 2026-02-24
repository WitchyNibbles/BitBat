# Conventions

**Analysis Date:** 2026-02-24

## Language and Formatting

- Python style baseline:
  - 4-space indentation.
  - Max line length 100 (`[tool.black]`, `[tool.ruff]` in `pyproject.toml`).
- Formatting tools:
  - `ruff format` and `black` run via `make fmt`.
- Lint profile:
  - Ruff rules include `E,F,B,I,UP,S,C4,RET,SIM`.
  - Per-test ignores allow `assert` and subprocess patterns in tests.

## Typing and Static Analysis

- MyPy strict typed-def style is enabled:
  - `disallow_untyped_defs = true`
  - `disallow_incomplete_defs = true`
- New/changed functions generally include explicit type hints.
- Type-check command: `make lint` (includes `poetry run mypy src tests`).

## Config Access Pattern

- Runtime configuration is loaded through `src/bitbat/config/loader.py`.
- Typical usage:
  - `set_runtime_config(...)` near process startup.
  - `get_runtime_config()` inside modules for read-only access.
- Default config source of truth: `src/bitbat/config/default.yaml`.

## Data Contract Pattern

- Contract-first validation is central in `src/bitbat/contracts.py`.
- Expected practice:
  - Validate dataframes at boundaries before downstream use.
  - Keep canonical column names (`timestamp_utc`, `feat_*`, etc.).
  - Normalize timestamps to tz-naive UTC.

## Storage and Path Conventions

- Local-first persistence under `data/`, `models/`, `metrics/`.
- Frequency/horizon scoped outputs use `{freq}_{horizon}` folder naming.
- Parquet IO should use helpers from `src/bitbat/io/fs.py` where possible.

## CLI and API Conventions

- CLI uses Click command groups in `src/bitbat/cli.py` (`prices`, `news`, `features`, `model`, `backtest`, `batch`, `monitor`, `validate`, `ingest`).
- API uses FastAPI router-per-domain layout in `src/bitbat/api/routes/`.
- API route defaults commonly assume `freq="1h"`, `horizon="4h"`.

## Error Handling Conventions

- Mixed pattern in current codebase:
  - Contract/CLI/API boundaries often fail fast with explicit exceptions.
  - Operational and UI code frequently catches broad exceptions and logs/falls back.
- Examples:
  - Broad catch in monitoring loops (`scripts/run_monitoring_agent.py`, `src/bitbat/autonomous/agent.py`).
  - Graceful fallback in Streamlit pages and widgets (`streamlit/`, `src/bitbat/gui/widgets.py`).

## Testing Conventions

- Pytest is the default framework; tests live under `tests/` with mirrored domains.
- `tmp_path` and `monkeypatch` are common for filesystem and runtime isolation.
- Slow/network marker exists (`@pytest.mark.slow`) in pytest config.

## CI and Quality Gates

- CI workflow (`.github/workflows/ci.yml`) runs:
  - Ruff lint + format check,
  - pytest,
  - Docker build and health probe.
- Pre-commit hooks mirror these standards locally.

