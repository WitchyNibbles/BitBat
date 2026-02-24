# Architecture Research

**Domain:** Brownfield stabilization architecture for monitoring + timeline UX
**Researched:** 2026-02-24
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Streamlit UI │   │ FastAPI API  │   │ CLI Commands   │  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬─────────┘  │
│         │                  │                  │            │
├─────────┴──────────────────┴──────────────────┴────────────┤
│                 Service / Domain Layer                      │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Monitor Agent + Predictor + Validator + Drift/Train  │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Persistence / Artifacts Layer               │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ SQLite (ops)   │  │ Parquet data   │  │ Model files  │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Migration boundary | Keep schema and code in sync | Versioned migration + startup compatibility check |
| Monitor service | Generate/validate/store predictions | `MonitoringAgent` orchestration with explicit failure surfaces |
| Timeline read model | Provide chart-ready prediction history | Query adapter that normalizes missing optional columns |
| UI renderer | Present filters, confidence, outcomes | Streamlit page modules with stable width API usage |

## Recommended Project Structure

```
src/
├── bitbat/
│   ├── autonomous/         # Monitor, DB models, drift/retraining
│   ├── api/                # External endpoints
│   ├── gui/                # Reusable dashboard widgets/timeline helpers
│   ├── model/              # Training/inference logic
│   └── ...
├── migrations/             # NEW: schema migrations (Alembic)
└── tests/
    ├── autonomous/         # DB + monitor regressions
    ├── gui/                # Timeline rendering/data contracts
    └── api/                # Service contract coverage
```

### Structure Rationale

- **`migrations/`** introduces deterministic DB evolution for brownfield stability.
- **`autonomous/` + `gui/` separation** keeps data producers and timeline presentation decoupled.
- **Targeted regression tests** stop recurrence of schema/UI compatibility breaks.

## Architectural Patterns

### Pattern 1: Schema-Version Handshake

**What:** Validate expected columns/version before monitor cycles and timeline reads.
**When to use:** Any startup path touching `prediction_outcomes`.
**Trade-offs:** Small startup overhead, major reliability gain.

### Pattern 2: Read-Model Adapter for Timeline

**What:** Build timeline DTOs from DB rows with defaulting/normalization.
**When to use:** Timeline page queries prediction history.
**Trade-offs:** Extra adapter layer, but isolates UI from raw schema churn.

### Pattern 3: Progressive UI Modernization

**What:** Replace deprecated widget arguments in one consistency pass.
**When to use:** Brownfield Streamlit upgrades.
**Trade-offs:** Broad edit set, but one-time cleanup lowers future maintenance cost.

## Data Flow

### Request Flow

```
[Monitoring cycle]
    ↓
[Ingestion refresh] → [Predict latest] → [Store prediction in SQLite]
    ↓
[Validator realizes outcomes] → [Drift metrics] → [Retraining decision]
```

### Timeline Flow

```
[User opens timeline page]
    ↓
[Filter params] → [Timeline query adapter] → [SQLite prediction rows]
    ↓
[Normalize + derive overlays] → [Chart/table render]
```

### Key Data Flows

1. **Prediction write flow:** Predictor -> DB repository -> `prediction_outcomes`.
2. **Timeline read flow:** UI -> query adapter -> normalized records -> chart.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single-user local | SQLite + filesystem artifacts are sufficient |
| Multi-process local services | Add startup migration lock + retry strategy |
| High-concurrency/shared deployment | Move DB to managed server store, keep repository interfaces stable |

### Scaling Priorities

1. **First bottleneck:** SQLite write/read contention under concurrent monitor/UI/API load.
2. **Second bottleneck:** timeline query latency on unindexed large prediction tables.

## Anti-Patterns

### Anti-Pattern 1: Runtime Schema Guessing

**What people do:** Assume columns exist and patch ad-hoc on failure.
**Why it's wrong:** Causes recurring outages and inconsistent local states.
**Do this instead:** Migrate schema explicitly and enforce preflight checks.

### Anti-Pattern 2: UI-Coupled DB Queries

**What people do:** Query/transform inline inside Streamlit page code.
**Why it's wrong:** Hard to test, fragile during schema evolution.
**Do this instead:** Centralize query/transform logic in timeline helper modules.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Streamlit | UI runtime | Replace deprecated width API usage globally |
| SQLAlchemy/SQLite | ORM + repository | Add migration/version discipline |
| Prometheus/Grafana | Metrics scrape + dashboards | Useful for monitor-health visibility |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `autonomous` ↔ `gui` | DB read models/helpers | Keep field contracts stable for timeline |
| `autonomous` ↔ `api` | shared repository/models | Schema mismatch impacts both surfaces |
| `cli` ↔ `autonomous` | direct function calls + DB writes | Keep prediction write semantics consistent |

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/CONCERNS.md`
- `src/bitbat/autonomous/models.py`, `src/bitbat/autonomous/db.py`, `src/bitbat/gui/timeline.py`

---
*Architecture research for: BitBat stabilization domain*
*Researched: 2026-02-24*
