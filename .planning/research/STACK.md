# Stack Research

**Domain:** Brownfield ML prediction platform stabilization (Streamlit + FastAPI + SQLite)
**Researched:** 2026-02-24
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11+ | Runtime across CLI/API/UI/services | Already the deployed project baseline with mature ecosystem support |
| SQLAlchemy | 2.x | DB model/query layer for monitor + API + UI readers | Existing codebase standard; supports clear migration and schema introspection patterns |
| Streamlit | 1.38+ (current in repo) | Dashboard and operator controls | Existing surface; targeted remediation (`width=` API migration) is low-risk and high-impact |
| FastAPI + Uvicorn | FastAPI 0.129+, Uvicorn 0.40+ | Service endpoints for health/predictions/metrics | Already integrated with monitoring stack and compose deployment |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Alembic | latest compatible with SQLAlchemy 2.x | Explicit schema migration/versioning | Required for safe DB evolution in brownfield monitor workflows |
| pytest | 8.x | Regression + integration test coverage | Use for DB migration tests, timeline rendering contracts, warning regression checks |
| pyarrow | 17.x | Parquet IO for model artifacts and datasets | Keep as default artifact format in existing data pipeline |
| pydantic (FastAPI schemas) | bundled with FastAPI | Strict API payload contracts | Keep API/DB contract boundaries explicit while schema evolves |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Ruff + Black | Lint/format consistency | Already configured in `pyproject.toml` and CI |
| MyPy | Typed interface safety | Helpful for DB/model/API interface refactors |
| Pre-commit | Local quality gate | Prevents style/type drift during stabilization sprint |

## Installation

```bash
# Existing project bootstrap
poetry install

# Suggested addition for schema migrations
poetry add alembic
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Alembic-based migrations | Manual ad-hoc SQL DDL in scripts | Only for one-off local experiments; not for shared stable workflows |
| SQLite retained | PostgreSQL migration | If concurrency/load requirements exceed SQLite operational envelope |
| Streamlit retained | Rewrite UI to React or other frontend stack | Only if product goals require richer web-app interactions beyond Streamlit model |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Destructive DB reset as default fix | Can hide migration bugs and destroy operational history | Versioned migrations + backward-compatible schema guards |
| Silent broad exception swallowing in critical DB paths | Masks production failures until data integrity is already degraded | Targeted exceptions + structured logging + failure visibility |
| Mixed widget APIs (`use_container_width` + `width`) | Guarantees repeated deprecation churn and noisy UX | Standardize to `width='stretch'` / `width='content'` consistently |

## Stack Patterns by Variant

**If running local single-operator workflows:**
- Keep SQLite + filesystem artifacts.
- Add schema-version checks at startup.

**If multi-process monitor/API/UI concurrency increases:**
- Introduce DB locking/retry strategy and evaluate migration to server DB.
- Keep SQLAlchemy repository boundaries to limit migration blast radius.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| SQLAlchemy 2.x | Alembic latest compatible | Standard migration path for declarative models |
| Streamlit 1.38+ | `width=` API | Required replacement for deprecated `use_container_width` |
| FastAPI 0.129+ | Pydantic-compatible schemas in project | Keep schemas aligned with DB model changes |

## Sources

- `.planning/codebase/STACK.md` — existing runtime stack and versions
- `.planning/codebase/ARCHITECTURE.md` — current subsystem boundaries and operational couplings
- `pyproject.toml` — dependency truth source
- `src/bitbat/autonomous/models.py`, `db.py`, `agent.py` — DB/runtime behavior
- `streamlit/` pages and app modules — UI API usage patterns

---
*Stack research for: BitBat stabilization + timeline enhancement*
*Researched: 2026-02-24*
