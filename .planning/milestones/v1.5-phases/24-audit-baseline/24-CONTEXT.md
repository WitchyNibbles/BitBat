# Phase 24: Audit Baseline - Context

**Gathered:** 2026-03-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish evidence-based codebase health understanding before any remediation. Classify all tests by type, run static analysis tooling (vulture, pytest-cov, radon), and execute an E2E pipeline smoke test. No code fixes in this phase — output is audit evidence that drives phases 25-27.

</domain>

<decisions>
## Implementation Decisions

### Test Classification Method
- Add pytest markers (`@pytest.mark.behavioral`, `@pytest.mark.structural`, etc.) to every test AND produce a summary report
- Claude determines the right category scheme based on what's actually in the test suite
- Delete milestone-marker tests entirely — they inflate the test count and give false confidence
- Produce a full coverage matrix cross-referencing every v1.5 requirement (CORR-01 through ARCH-06) against existing test coverage

### Findings Documentation
- Save raw tool outputs (vulture.txt, coverage report, radon report) as evidence files in the phase directory
- Produce a synthesized AUDIT-REPORT.md with both views: category-organized findings with a severity-sorted summary table at top
- Each finding tagged with which v1.5 requirement (CORR-01, ARCH-02, etc.) it maps to, or 'DEFER' if out of scope
- Pre-populate known issues from research (retrainer --tau, CV key mismatch, OBV leakage, etc.) AND run tools independently, then cross-reference: did automated tooling catch what manual research found?

### E2E Smoke Test Approach
- Try real data download first (yfinance for small date range). If it fails (network, API limit), fall back to synthetic fixtures and note ingestion failure separately
- Skip-and-continue on failures: if a stage fails, use pre-existing data from previous runs (if available) to continue testing downstream stages. Note the failure but test as much as possible
- Claude determines which stages are meaningful to chain based on what's available after ingestion
- Save both raw console log for evidence AND a structured summary extracted from it

### Severity Triage Criteria
- **CRITICAL**: Silently broken in production AND breaks core value promise AND data corruption risk. All three dimensions.
- **HIGH**: Causes active development friction OR is a latent risk that could escalate to CRITICAL if conditions change. Fix in v1.5.
- **MEDIUM**: Genuine debt but stable and non-blocking. Neither causes friction nor poses escalation risk. Catalog, defer.
- Claude assigns severity for all findings. User reviews the severity-sorted summary table and can override before planning starts.

### Claude's Discretion
- Exact pytest marker names and registration in conftest.py
- Tool configuration details (vulture min-confidence, radon thresholds, coverage report format)
- How to structure raw tool output files
- Whether to run mypy --strict as part of this phase or defer
- Ordering of audit activities within the phase

</decisions>

<specifics>
## Specific Ideas

- Cross-reference automated tool findings against manual research findings — the gap itself (what tools miss vs what humans catch) is a useful audit meta-finding
- The full coverage matrix should make it immediately obvious which pipeline stages have zero behavioral test coverage
- Severity triage should explicitly note research-confirmed bugs (retrainer --tau, CV key mismatch) as pre-validated CRITICAL items

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 24-audit-baseline*
*Context gathered: 2026-03-04*
