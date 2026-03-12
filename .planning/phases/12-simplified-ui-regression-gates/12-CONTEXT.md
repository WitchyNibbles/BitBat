# Phase 12: Simplified UI Regression Gates - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Codify the simplified five-view UI contract and Phase 11 crash hardening outcomes as
durable regression gates. This phase focuses on automated verification quality and
release-contract durability, not new product features.

</domain>

<decisions>
## Implementation Decisions

### Lock the Simplified Surface Contract
- Supported runtime views remain: Quick Start, Settings, Performance, About, and System.
- Core UI tests must fail if retired-page links reappear in active navigation surfaces.

### Preserve Runtime Stability Fixes
- Regression coverage must explicitly defend against the reported failure signatures:
  missing-confidence home crash, pipeline import crash, and backtest indexing crash.
- Guard behavior is acceptable if it prevents traceback-prone paths in normal usage.

### Add Practical Smoke Confidence
- Include deterministic smoke checks that assert all five supported pages are present,
  importable, and mapped in navigation paths without runtime exceptions.

### Scope Boundaries
- Phase 12 validates contracts and verification depth only.
- No reintroduction of advanced/retired pages or broader UX redesign in this phase.

### Claude's Discretion
- Choose test architecture split (phase gate vs smoke suite) as long as QUAL-04/05/06
  are each test-locked and release-wired.
- Prefer source-based and module-import smoke checks over brittle end-to-end browser flow.

</decisions>

<specifics>
## Specific Ideas

- Existing relevant gates:
  - `tests/gui/test_phase10_supported_surface_complete.py`
  - `tests/gui/test_phase11_runtime_stability_complete.py`
  - `tests/gui/test_phase8_release_verification_complete.py`
- Existing canonical command:
  - `make test-release` is the acceptance contract and should include Phase 12 gate coverage.

</specifics>

<deferred>
## Deferred Ideas

- Full browser-driven integration smoke harness for all pages.
- Reintroducing advanced Analytics/Backtest/Pipeline functionality.

</deferred>

---

*Phase: 12-simplified-ui-regression-gates*
*Context gathered: 2026-02-25*
