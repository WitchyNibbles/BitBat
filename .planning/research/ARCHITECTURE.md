# Architecture Research

**Domain:** ML pipeline codebase health audit — anti-patterns, integration breakpoints, audit order
**Researched:** 2026-03-04
**Confidence:** HIGH (direct codebase inspection, no speculation)

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  CLI (cli.py)    │  │  FastAPI (api/)   │  │  Streamlit (gui/)│   │
│  └────────┬─────────┘  └────────┬──────────┘  └────────┬─────────┘   │
│           │                     │                       │            │
├───────────┴─────────────────────┴───────────────────────┴────────────┤
│                      Autonomous / Service Layer                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ autonomous/: agent, predictor, validator, drift, retrainer,    │  │
│  │ continuous_trainer, orchestrator, price_ingestion, news_ingest │  │
│  └─────────────────────────────┬──────────────────────────────────┘  │
│                                 │ (calls down into)                   │
├─────────────────────────────────┴────────────────────────────────────┤
│                        Domain / Pipeline Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ ingest/  │ │features/ │ │labeling/ │ │ dataset/ │ │  model/  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│         │           │           │              │             │        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │timealign/│ │contracts │ │config/   │ │backtest/ │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                        Persistence Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ SQLite (ops DB) │  │ Parquet (data/) │  │ JSON (models/)  │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Observed Implementation |
|-----------|----------------|-------------------------|
| `cli.py` | 9-command-group operator interface | 2600+ line single file; imports private functions from domain modules |
| `contracts.py` | Schema enforcement at pipeline boundaries | Single file; covers prices, news, features, predictions — no macro/onchain contracts |
| `config/loader.py` | Global config state via module-level cache | Process-singleton pattern; `get_runtime_config() or load_config()` idiom repeated across 8+ callers |
| `autonomous/agent.py` | Monitoring cycle orchestration | Imports `ingest/*` directly inside methods; mixes ingestion concerns with monitoring |
| `autonomous/predictor.py` | Live inference from ingested data | Duplicates price loading, feature generation, and rename logic from `dataset/build.py` |
| `autonomous/continuous_trainer.py` | In-process retraining | Third copy of price loading and feature generation; writes model path without abstraction |
| `autonomous/retrainer.py` | Drift-triggered retraining via subprocess | Shells out to `poetry run bitbat` — creates process boundary with no error contract |
| `api/routes/system.py` | System logs, settings, training trigger | Imports from `gui/widgets.py` (ingestion_status); duplicates `_table_columns` from gui/widgets |
| `gui/widgets.py` | Shared Streamlit helpers | Contains `_table_columns` + raw sqlite3 queries that are also in `api/routes/system.py` |
| `dataset/build.py` | Feature matrix assembly | `_generate_price_features` and `_join_auxiliary_features` are private but used by 3 other modules |
| `model/train.py` | XGBoost/RF training | Reads `freq`/`horizon` from DataFrame attrs — implicit side-channel contract |

---

## Recommended Project Structure

The current structure is logically sound. The problems are not structural — they are boundary violations within the existing structure:

```
src/bitbat/
├── config/            # GOOD: isolated loader; PROBLEM: singleton state leaks into tests
├── contracts.py       # GOOD: enforcement layer; GAP: no macro/onchain contracts
├── io/                # GOOD: isolated I/O; PROBLEM: bypassed by sqlite3 in gui/ and api/
├── ingest/            # GOOD: data fetchers; PROBLEM: duplicated in autonomous/
├── timealign/         # GOOD: leakage guardrails
├── features/          # GOOD: feature implementations
├── labeling/          # GOOD: target computation
├── dataset/           # PROBLEM: private functions (_generate_price_features) used by 3 modules
├── model/             # GOOD: training/inference; PROBLEM: path hardcoded as Path("models")
├── backtest/          # GOOD: isolated strategy engine
├── analytics/         # GOOD: isolated analysis
├── autonomous/        # PROBLEM: god package — owns ingestion, training, inference, monitoring
│   ├── agent.py       # PROBLEM: mixes ingestion calls with monitoring orchestration
│   ├── predictor.py   # PROBLEM: duplicates feature pipeline from dataset/build.py
│   ├── continuous_trainer.py  # PROBLEM: third copy of price loading + feature generation
│   ├── retrainer.py   # PROBLEM: subprocess shell-out to own CLI breaks error contract
│   ├── orchestrator.py  # PROBLEM: hardcodes legacy file path btcusd_yf_{freq}.parquet
│   ├── price_ingestion.py  # OK: clean service; but parallel to ingest/prices.py
│   ├── news_ingestion.py   # OK: clean service; but agent.py directly calls ingest/ instead
│   ├── macro_ingestion.py  # OK: clean service
│   └── onchain_ingestion.py  # OK: clean service
├── api/
│   └── routes/
│       └── system.py  # PROBLEM: imports gui/widgets.py; duplicates _table_columns
└── gui/
    └── widgets.py     # PROBLEM: contains logic reused in api/routes/system.py
```

---

## Architectural Patterns

### Pattern 1: Contract Enforcement at Stage Boundaries

**What:** Every Parquet write/read passes through `contracts.py` validators before use.
**When to use:** All pipeline ingestion and feature output writes.
**Trade-offs:** Catches column drift early; adds one DataFrame copy per boundary.
**Current state:** Prices, news, features, predictions are covered. Macro and on-chain Parquet are NOT covered — they are read directly from disk without contract validation.

### Pattern 2: Config as Process Singleton

**What:** `config/loader.py` maintains a module-level `_ACTIVE_CONFIG` dict. `get_runtime_config()` returns a shallow copy.
**When to use:** Appropriate for a single-process CLI or API server.
**Trade-offs:** Correct for single-process use. Breaks in test suites if config is mutated between tests without reset. The `get_runtime_config() or load_config()` pattern (8+ occurrences) means the fallback silently loads defaults instead of raising — making config misconfiguration invisible.

### Pattern 3: Freq/Horizon as Path Key

**What:** All artifacts use `{freq}_{horizon}` as a compound path key (e.g. `models/5m_30m/xgb.json`, `data/features/5m_30m/dataset.parquet`).
**When to use:** Consistent across all pipeline stages.
**Trade-offs:** Simple and effective. Problem: the path construction is done independently in 6+ places with no central `artifact_path(freq, horizon)` helper — making it fragile to any future path scheme change.

---

## Data Flow

### Pipeline Execution Flow

```
[CLI: bitbat prices fetch]
    → ingest/prices.py → ensure_prices_contract → data/raw/prices/btcusd_yf_{freq}.parquet

[CLI: bitbat features build]
    → dataset/build.py:build_xy
        → _generate_price_features (price.py)
        → aggregate_sentiment (sentiment.py) [optional]
        → _join_auxiliary_features (macro.py, onchain.py) [optional]
        → ensure_feature_contract
        → data/features/{freq}_{horizon}/dataset.parquet + meta.json

[CLI: bitbat model train]
    → model/train.py:fit_xgb
        → reads DataFrame.attrs["freq"], DataFrame.attrs["horizon"]
        → writes models/{freq}_{horizon}/xgb.json

[CLI: bitbat batch run]
    → model/infer.py:predict_bar
        → writes data/predictions/{freq}_{horizon}.parquet (ensure_predictions_contract)

[Monitor: bitbat monitor start]
    → autonomous/agent.py:MonitoringAgent
        → _ingest_prices() → autonomous/price_ingestion.py
        → _ingest_news() → ingest/news_cryptocompare.py [DIRECT — bypasses news_ingestion.py]
        → validator.validate_all()
        → predictor.predict_latest() → autonomous/predictor.py
            → _load_ingested_prices (LOCAL duplicate of price loading)
            → _generate_price_features (imported from dataset/build.py — private function)
        → continuous_trainer.retrain() → autonomous/continuous_trainer.py
            → _load_prices (LOCAL duplicate — third copy)
            → _generate_price_features (imported from dataset/build.py — private function)
```

### Config Initialization Flow

```
[Any module calling get_runtime_config()]
    → if _ACTIVE_CONFIG is None → load_config(cache=True) [implicit default load]
    → return shallow copy of _ACTIVE_CONFIG

[Problem: get_runtime_config() or load_config() pattern]
    → get_runtime_config() NEVER returns falsy (returns {} at worst)
    → load_config() fallback is unreachable but creates confusion and masks intent
```

### Key Integration Point: autonomous/ → dataset/build.py

This is the most fragile integration in the codebase:

```
autonomous/predictor.py  ──imports──▶  dataset/build._generate_price_features (private)
autonomous/continuous_trainer.py ──imports──▶  dataset/build._generate_price_features (private)
cli.py ──imports──▶  dataset/build._generate_price_features (private)
```

Three callers depend on a private implementation detail of the dataset builder. Any refactoring of `_generate_price_features` requires coordinated changes in 3 locations.

---

## Anti-Patterns

### Anti-Pattern 1: Private Function Imported Across Module Boundaries

**What people do:** Import underscore-prefixed functions (e.g., `_generate_price_features`, `_join_auxiliary_features`) from another module's implementation file.

**In this codebase:**
- `autonomous/predictor.py` line 20: `from bitbat.dataset.build import _generate_price_features`
- `autonomous/continuous_trainer.py` line 147 (inside method): `from bitbat.dataset.build import _generate_price_features`
- `cli.py` line 27: `from bitbat.dataset.build import _generate_price_features, build_xy`

**Why it's wrong:** The underscore prefix signals "not part of the public API." Cross-module imports of private functions mean any internal refactoring in `dataset/build.py` breaks callers that had no indication they were depending on it. The test suite cannot mock these dependencies cleanly.

**Do this instead:** Promote `_generate_price_features` and `_join_auxiliary_features` to a public API in `features/` (e.g., `features/pipeline.py:generate_price_features()`). The function already belongs in `features/` conceptually — it ended up in `dataset/build.py` as a build step convenience.

---

### Anti-Pattern 2: Price Loading Logic Duplicated in Three Places

**What people do:** Each component that needs prices writes its own loading logic.

**In this codebase:**
- `cli.py:_load_prices_indexed()` — flat file path `btcusd_yf_{freq}.parquet`
- `autonomous/predictor.py:_load_ingested_prices()` — glob over `data/raw/prices/**/*.parquet` + legacy flat file fallback
- `autonomous/continuous_trainer.py:_load_prices()` — same glob logic, different error handling

**Why it's wrong:** The three implementations differ in how they handle the legacy flat file, timezone normalization, and deduplication. A bug fix in one does not fix the others. `orchestrator.py` also hardcodes the legacy path directly. If the storage layout changes, 4 locations must be updated.

**Do this instead:** Consolidate into `ingest/prices.py:load_prices(data_dir, freq)` — a public loader that encapsulates both the date-partitioned format and the legacy flat file fallback. All callers use one function.

---

### Anti-Pattern 3: `get_runtime_config() or load_config()` Idiom

**What people do:** Call `get_runtime_config() or load_config()` as a "safe" config fetch.

**In this codebase:** 8 occurrences across `autonomous/` modules (agent, validator, alerting, drift, predictor, retrainer, orchestrator).

**Why it's wrong:** `get_runtime_config()` is documented to load and cache defaults if no config has been set — it never returns `None` or an empty falsy value. The `or load_config()` branch is unreachable dead code that creates the false impression that two separate config paths are possible. Worse, if `get_runtime_config()` ever returned `{}` (empty config), the `or` branch would trigger `load_config()` without caching, silently creating a second config load.

**Do this instead:** Use `get_runtime_config()` alone. If the intent is "load if not yet loaded," the function already handles that internally. Remove all `or load_config()` fallbacks.

---

### Anti-Pattern 4: Hardcoded Relative Paths Scattered Across Modules

**What people do:** Construct artifact paths inline with `Path("models")`, `Path("metrics")`, `Path("data")` at every call site.

**In this codebase:**
- `Path("models") / f"{freq}_{horizon}" / "xgb.json"` appears in: `autonomous/agent.py`, `autonomous/predictor.py`, `autonomous/continuous_trainer.py`, `autonomous/retrainer.py` (via subprocess), `api/routes/analytics.py`, `api/routes/health.py`
- `Path("metrics") / "cv_summary.json"` appears in: `autonomous/retrainer.py`, `cli.py`
- `Path("data") / "autonomous.db"` appears in: `api/routes/predictions.py`, `api/routes/system.py`

**Why it's wrong:** The `data_dir` config key exists but is ignored in half the call sites. If the runtime is configured with `data_dir: /mnt/storage`, the model persistence and DB paths do not follow. This creates a split-brain between config-aware paths (ingest services) and hardcoded paths (API, monitor agent).

**Do this instead:** Centralize path construction in `config/paths.py` (or `io/paths.py`) with functions: `model_path(data_dir, freq, horizon)`, `metrics_path(data_dir, name)`, `db_path(data_dir)`. Every call site uses these functions, driven by `config.data_dir`.

---

### Anti-Pattern 5: `autonomous/` Package as God Module

**What people do:** Bundle ingestion, training, inference, monitoring, orchestration, and DB access into one package.

**In this codebase:** `autonomous/` contains 14 modules covering: price/news/macro/onchain ingestion services, live prediction, drift detection, model retraining (two implementations), monitoring orchestration, DB access, alerting, rate limiting, and schema compatibility.

**Why it's wrong:** The package boundary no longer signals anything to a reader. A developer adding a new feature cannot determine which sub-concerns belong in `autonomous/` vs elsewhere. The duplicate ingestion services (`autonomous/price_ingestion.py` vs `ingest/prices.py`) exist because the boundary was ambiguous when they were written.

**Do this instead:** `autonomous/` should own only the monitoring loop logic: agent, validator, drift, alerting, DB, and models. Ingestion services belong in `ingest/`. Training services belong in `model/`. The orchestrator (`orchestrator.py`) is a cross-cutting coordinator and should be explicit about its dependencies.

---

### Anti-Pattern 6: API Route Importing from GUI Module

**What people do:** Reuse a utility function by importing it from whichever module defined it first.

**In this codebase:**
- `api/routes/system.py` line 219: `from bitbat.gui.widgets import get_ingestion_status`
- `api/routes/system.py` defines its own `_table_columns(con, table)` which duplicates `gui/widgets.py:_table_columns(db_path, table)` with a different signature

**Why it's wrong:** `api/` importing from `gui/` creates a layering violation — the API layer depends on the GUI layer. If the Streamlit dependency is ever decoupled or the GUI module refactored, the API breaks unexpectedly. The duplicate `_table_columns` implementations (one takes a `Connection`, one takes a `Path`) diverge silently.

**Do this instead:** Extract shared DB utilities into `autonomous/db.py` or a new `io/sqlite.py` module. Both `gui/` and `api/` import from there. `get_ingestion_status` should be in `ingest/` or `io/` — it reads filesystem state, not GUI state.

---

### Anti-Pattern 7: Subprocess Shell-Out for Retraining

**What people do:** Invoke the application's own CLI as a subprocess to trigger pipeline stages.

**In this codebase:** `autonomous/retrainer.py:_run_command()` calls `poetry run bitbat features build` and `poetry run bitbat model cv` as subprocesses. This is the only trigger path for drift-based retraining.

**Why it's wrong:**
1. Error handling is limited to exit code — structured exceptions from the pipeline are lost
2. The subprocess inherits no runtime config context (it will load defaults unless the env var is set)
3. Testing requires mocking subprocess calls, not the actual pipeline logic
4. The `poetry run` prefix is environment-specific — breaks in venv-only or Docker deployments
5. `continuous_trainer.py` (the other retraining path) calls pipeline functions directly, creating two incompatible retraining code paths

**Do this instead:** Call pipeline functions directly, the same way `continuous_trainer._do_retrain()` already does. The `retrainer.py` and `continuous_trainer.py` retraining paths should be unified into one implementation.

---

### Anti-Pattern 8: DataFrame.attrs as Implicit Side-Channel

**What people do:** Pass configuration to downstream functions via DataFrame metadata attributes.

**In this codebase:** `model/train.py:_extract_freq_horizon()` reads `X_train.attrs["freq"]` and `X_train.attrs["horizon"]` to determine the model save path. This is set in `cli.py` (`X.attrs["freq"] = freq`) and `autonomous/orchestrator.py` before calling `fit_xgb()`.

**Why it's wrong:** The `freq` and `horizon` parameters are required for correct behavior but are not in the function signature. A caller that forgets to set attrs gets silently wrong model paths (`models/unknown_unknown/xgb.json`). This is an invisible precondition.

**Do this instead:** Add `freq` and `horizon` as explicit keyword parameters to `fit_xgb(X_train, y_train, *, freq, horizon, seed, persist)`. The DataFrame.attrs channel should be documented as legacy or removed.

---

### Anti-Pattern 9: Config Singleton Not Reset Between Tests

**What people do:** Use a process-level singleton for config without providing a reset mechanism.

**In this codebase:** `config/loader.py` uses module-level globals `_ACTIVE_CONFIG`, `_ACTIVE_PATH`, `_ACTIVE_SOURCE`. Tests that call `set_runtime_config()` or `load_config(cache=True)` modify shared state that bleeds into subsequent tests in the same process.

**Why it's wrong:** Test isolation breaks. A test that loads a custom config can corrupt all subsequent test runs in the same pytest session. This likely already affects test suite reliability.

**Do this instead:** Add a `reset_runtime_config()` function (or a context manager `with_config(path)`) that resets the module-level globals. Test fixtures use it as teardown. This is a 3-line addition to `loader.py`.

---

### Anti-Pattern 10: Dual Database Access Patterns (ORM + Raw sqlite3)

**What people do:** Access the same SQLite database through two different mechanisms in the same application.

**In this codebase:**
- `autonomous/db.py` uses SQLAlchemy ORM (sessions, models)
- `api/routes/system.py` uses raw `sqlite3.connect()` directly
- `gui/widgets.py` uses raw `sqlite3.connect()` via `db_query()`
- `gui/timeline.py` uses `sqlite3.connect()` directly (lines 377, 399)

**Why it's wrong:** The ORM schema and the raw SQL queries can drift. If a column is added via SQLAlchemy's model but the migration adds it with a different default, the raw queries return inconsistent data. The `_table_columns` + `_first_available` defensive column-sniffing pattern in `system.py` and `widgets.py` exists precisely because the two access patterns diverged.

**Do this instead:** All reads from `autonomous.db` go through `AutonomousDB` methods (or new read-model methods). The ORM is the single access layer. Raw `sqlite3` is eliminated outside of `autonomous/db.py`.

---

## Integration Points

### Integration Points That Commonly Break

| Boundary | Mechanism | Break Pattern | Audit Priority |
|----------|-----------|---------------|----------------|
| `autonomous/predictor.py` → `dataset/build._generate_price_features` | Private function import | Refactoring dataset/build breaks predictor silently | HIGH |
| `autonomous/retrainer.py` → CLI via subprocess | Shell-out | Config not propagated; exit code swallows structured errors | HIGH |
| `api/routes/system.py` → `gui/widgets.get_ingestion_status` | Cross-layer import | Streamlit import chain pulled into API process | HIGH |
| `autonomous/agent._ingest_news()` → `ingest/news_cryptocompare` directly | Bypasses `news_ingestion.py` | Two news-ingest call paths diverge in error handling | MEDIUM |
| `model/train._extract_freq_horizon(X)` → `X.attrs["freq"]` | DataFrame attrs side-channel | Silent wrong model path on missing attrs | HIGH |
| `config/loader._ACTIVE_CONFIG` → shared test state | Module global | Config pollution between tests | MEDIUM |
| ORM schema (autonomous/models.py) → raw sqlite3 queries (api/, gui/) | Dual DB access | Schema migrations only update ORM, not raw queries | MEDIUM |
| Hardcoded `Path("models")` → actual `data_dir` | Path construction | Config `data_dir` ignored in model paths | MEDIUM |

### Internal Boundaries — Current State

| Boundary | Communication | Correct? |
|----------|---------------|----------|
| `cli.py` ↔ `dataset/build.py` | Direct public + private function calls | NO — private function import |
| `autonomous/` ↔ `dataset/build.py` | Private function import | NO — predictor, continuous_trainer |
| `autonomous/` ↔ `ingest/` | Direct module import (agent._ingest_news) | NO — bypasses autonomous ingestion layer |
| `api/` ↔ `gui/` | `get_ingestion_status` import | NO — layering violation |
| `api/` ↔ `autonomous/db.py` | SQLAlchemy ORM for predictions, raw sqlite3 for system routes | MIXED — dual access patterns |
| `contracts.py` ↔ all pipeline stages | Explicit calls at every write | YES — correct pattern |
| `timealign/` ↔ `dataset/build.py` | Public function imports | YES — correct |
| `model/` ↔ `backtest/` | Clean separation | YES — correct |
| `model/` ↔ `analytics/` | Clean separation | YES — correct |

---

## Audit Order

Based on the dependency graph and break severity, audit in this order:

### Tier 1: Highest Impact — Breaks Silently

1. **`autonomous/predictor.py`** — private function import from `dataset/build.py`, duplicate price loading, feature rename duplication. This is the live prediction path — bugs here produce wrong predictions silently.

2. **`autonomous/retrainer.py`** — subprocess shell-out pattern, hardcoded paths, no config propagation. This is the drift-response path — failures here are opaque.

3. **`cli.py`** — imports private `_generate_price_features`; hardcoded paths mixed with config-aware paths; 2600+ lines with embedded business logic that belongs in domain modules. Start with the imports and path patterns, not the full file.

4. **`config/loader.py`** — the `get_runtime_config() or load_config()` idiom and test isolation gap. Small file, high impact across all callers.

### Tier 2: Architecture Drift — Breaks Over Time

5. **`autonomous/agent.py`** — mixed ingestion + monitoring concerns; inconsistent use of `price_ingestion.py` vs direct `ingest/` calls.

6. **`autonomous/continuous_trainer.py`** — third copy of price loading logic; trains differently from `retrainer.py` (in-process vs subprocess) with no documented rationale.

7. **`autonomous/orchestrator.py`** — hardcoded legacy file path `btcusd_yf_{freq}.parquet`; loads prices via the old flat-file path while the monitor uses date-partitioned format.

### Tier 3: Layering Violations — Breaks on Dependency Changes

8. **`api/routes/system.py`** — cross-layer import from `gui/`, duplicate `_table_columns`, raw sqlite3 bypassing ORM.

9. **`gui/widgets.py` + `gui/timeline.py`** — raw sqlite3 access duplicating ORM path; `_table_columns` duplication with system.py.

### Tier 4: Contract Gaps — Breaks on Data Shape Changes

10. **`contracts.py`** — verify coverage: macro and on-chain Parquet have no contracts. Verify that `ensure_feature_contract` is called after all feature join paths in `dataset/build.py` and `autonomous/continuous_trainer.py`.

11. **`model/train.py`** — DataFrame.attrs side-channel for freq/horizon. Verify all callers set attrs before calling `fit_xgb`.

---

## Scaling Considerations

| Concern | Current State | Risk Level |
|---------|---------------|------------|
| Multiple monitor cycles in parallel | SQLite write contention via both ORM and raw connections | HIGH |
| Config change mid-cycle | Process singleton not thread-safe | MEDIUM |
| New freq/horizon pair | 6+ hardcoded path construction sites to update | MEDIUM |
| New feature source | Must duplicate feature pipeline in predictor.py and continuous_trainer.py separately | HIGH |
| Test isolation | Config singleton bleeds between tests | MEDIUM |

---

## Sources

- Direct codebase inspection: all files under `src/bitbat/` (2026-03-04)
- `src/bitbat/autonomous/predictor.py` — private import cross-reference
- `src/bitbat/autonomous/continuous_trainer.py` — duplicate price loading
- `src/bitbat/autonomous/retrainer.py` — subprocess pattern
- `src/bitbat/autonomous/agent.py` — mixed ingestion concerns
- `src/bitbat/cli.py` — private function import, hardcoded paths
- `src/bitbat/api/routes/system.py` — cross-layer import, dual DB access
- `src/bitbat/gui/widgets.py` — raw sqlite3, duplicate _table_columns
- `src/bitbat/config/loader.py` — singleton pattern, test isolation gap
- `.planning/PROJECT.md` — milestone context

---
*Architecture research for: BitBat v1.5 Codebase Health Audit*
*Researched: 2026-03-04*
