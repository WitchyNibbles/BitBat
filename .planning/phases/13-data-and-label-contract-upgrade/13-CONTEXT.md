# Phase 13: Data and Label Contract Upgrade - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning
**Source:** User request + `deep-research-report.md`

<domain>
## Phase Boundary

Phase 13 establishes trustworthy BTC training data and targets before model-complexity work.
The deliverable is a leakage-safe data/label contract: as-of feature joins, return-first targets,
and optional triple-barrier labels that can be enabled for trading-aligned experiments.

</domain>

<decisions>
## Implementation Decisions

### Leakage-Safe as-of Data Is Mandatory
- Feature construction must enforce strict timestamp ordering across all joined sources.
- Any feature row that cannot satisfy as-of semantics must be dropped or explicitly rejected.
- Regression tests must assert no future information can influence row features/targets.

### Return-First Label Contract
- Primary target remains forward return, with direction labels derived from the same horizon.
- Horizon handling must be explicit and shared so return/direction labels cannot drift.
- Label generation behavior must be deterministic and testable across gap/missing-time cases.

### Optional Triple-Barrier Labeling
- Triple-barrier labels are in scope only as an optional dataset mode.
- Baseline dataset output and existing regression paths must remain intact when barrier mode is off.
- Barrier labeling API should be composable with current dataset builder interfaces.

### Scope Boundaries
- Phase 13 focuses on data and label contracts only.
- No model architecture upgrades, walk-forward engine changes, or promotion policy changes in this phase.

### Claude's Discretion
- Choose the cleanest module boundaries (dataset builder, labeling package, contract layer) as long as
  DATA-01, DATA-02, and LABL-01 are each test-locked.
- Choose precise barrier parameter defaults and naming conventions if they are configurable and documented.

</decisions>

<specifics>
## Specific Ideas

- Use current dataset pipeline entrypoints:
  - `src/bitbat/dataset/build.py`
  - `src/bitbat/labeling/returns.py`
  - `src/bitbat/labeling/targets.py`
  - `src/bitbat/contracts.py`
- Add/extend tests in:
  - `tests/dataset/test_build_xy.py`
  - `tests/labeling/test_returns.py`
  - New triple-barrier and leakage guard suites as needed.
- Keep contract compatibility with current `features.build` CLI flow in `src/bitbat/cli.py`.

</specifics>

<deferred>
## Deferred Ideas

- Walk-forward purge/embargo evaluation engine upgrades (Phase 15).
- Nested optimization and multiple-testing safeguards (Phase 16).
- Transformer/GNN/microstructure model expansion.

</deferred>

---

*Phase: 13-data-and-label-contract-upgrade*
*Context gathered: 2026-02-25*
