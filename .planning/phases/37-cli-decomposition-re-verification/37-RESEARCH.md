# Phase 37: CLI Decomposition Re-Verification - Research

**Researched:** 2026-03-12
**Domain:** formal closure of DEBT-01 verification evidence
**Confidence:** HIGH

## Summary

Phase 32's implementation is already complete, but its saved verification artifact is stale. The current `32-VERIFICATION.md` has contradictory metadata:

- frontmatter says `status: passed`
- body says `Status: gaps_found`
- the gap text still describes the old `features build` monkeypatch-target failure

The underlying codebase no longer matches that old report:

- `tests/test_cli.py` now patches `bitbat.cli.commands.features.build_xy` on the relevant `features build` tests
- the previously failing tests now pass
- `poetry run bitbat --help` still shows the expected 10 command groups

This means Phase 37 is a verification-and-traceability closure phase, not a new implementation phase unless fresh evidence uncovers a regression.

## Current Evidence

- `tests/test_cli.py::test_cli_features_build_label_mode_default_compatibility` passes
- `tests/test_cli.py::test_cli_features_build_triple_barrier_label_mode` passes
- `bitbat --help` shows all command groups
- the stale narrative in `32-VERIFICATION.md` is the remaining audit blocker

## Recommended scope

1. Re-run the authoritative CLI regression suites.
2. Re-run the C901 check over `src/bitbat/cli/`.
3. Re-run `bitbat --help`.
4. Rewrite `32-VERIFICATION.md` as a clean passed artifact that matches the current codebase.
5. Add a Phase 37 summary/verification pair and update `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, and `STATE.md` so DEBT-01 is formally closed.

## Risks

- Because this phase is mostly evidence repair, the main risk is under-verifying and merely changing docs. The solution is to rerun the actual CLI regression suites before touching the stale artifact.
- A new CLI regression discovered here would expand the phase from docs-only to a real code fix. Current spot checks suggest that will not be necessary.
