# BitBat Reliability and Timeline Evolution

## What This Is

BitBat is an existing local-first BTC prediction application with CLI, API, autonomous monitoring, and a Streamlit dashboard. This project focuses on stabilizing the current broken behavior and extending the timeline experience so the GUI becomes reliable and decision-useful again. The immediate targets are monitor DB runtime failures, broken prediction timeline behavior, and Streamlit deprecation cleanup.

## Core Value

A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## Requirements

### Validated

- [x] Ingest BTC market/news data and persist datasets to local parquet storage.
- [x] Build features and train/persist XGBoost models for directional prediction workflows.
- [x] Serve prediction/system data through FastAPI and Streamlit surfaces.
- [x] Persist autonomous monitoring state in SQLite (`data/autonomous.db`) with model/prediction/performance tables.

### Active

- [ ] Fix monitor/runtime DB errors caused by schema mismatch around `prediction_outcomes.predicted_price`.
- [ ] Restore and improve prediction timeline behavior to include predictions, realized outcomes, confidence context, and practical filtering.
- [ ] Remove Streamlit deprecation usage by replacing all `use_container_width` calls with `width='stretch'` or `width='content'` as appropriate.
- [ ] Deliver timeline enhancements beyond basic restore (scope C) without regressing existing monitoring and prediction workflows.
- [ ] Define and verify completion gates: D1 no monitor DB runtime errors, D2 timeline renders/updates correctly, D3 no `use_container_width` warnings.

### Out of Scope

- Full UI redesign across all pages - not required to solve current reliability and timeline goals.
- Major model strategy replacement (for example changing core model family/objective) - not part of this stabilization cycle.
- Infrastructure migration away from SQLite - defer unless blocked by hard technical limits.

## Context

The codebase already implements full ingestion, modeling, API, autonomous monitoring, and dashboard layers. Current operational breakage is concentrated in monitor DB interactions (`no such column: prediction_outcomes.predicted_price`) and timeline/UI reliability. Streamlit deprecation warnings indicate outdated widget API usage (`use_container_width`) and must be remediated across the GUI.

## Constraints

- **Tech stack**: Keep Python + Streamlit + FastAPI + SQLAlchemy/SQLite architecture - align with existing deployed/runtime surfaces.
- **Compatibility**: Preserve existing local artifact layout (`data/`, `models/`, `metrics/`) - avoid breaking current workflows and scripts.
- **Safety**: Prefer additive/compatible DB fixes and deterministic migrations - avoid risky destructive schema operations.
- **Scope discipline**: Prioritize D1/D2/D3 completion before optional broader enhancements.
- **Performance**: Timeline improvements must remain responsive on typical local datasets.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app already has production-like surfaces but is currently broken in critical paths | - Pending |
| Timeline target is T2 (improve, not just restore) | User explicitly requested richer timeline behavior | - Pending |
| Done criteria are D1/D2/D3 | Provides explicit technical acceptance gates | - Pending |
| Prioritize DB/runtime correctness before visual polish | Monitor failures block trust in system operation | - Pending |

---
*Last updated: 2026-02-24 after initialization*
