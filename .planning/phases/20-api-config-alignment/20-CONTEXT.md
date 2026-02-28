# Phase 20: API Config Alignment - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

API settings endpoint defaults to default.yaml values (freq=5m, horizon=30m) when no user config exists, and accepts/persists sub-hourly freq/horizon values. Covers APIC-01 and APIC-02. UI changes, presets, and test coverage are separate phases (21-23).

</domain>

<decisions>
## Implementation Decisions

### Config Persistence
- Operator settings should persist across API server restarts
- Single global config (not multi-user/profile)
- Partial updates merge with existing config — operator can update just freq without losing horizon

### Validation & Errors
- Accept only freq/horizon values defined in the existing bucket.py canonical set
- horizon >= freq enforcement is deferred (ADVC-02 is explicitly a future requirement)
- Error format should match existing API error patterns in bitbat/api/
- Defaults read from default.yaml at startup — changing the file and restarting updates defaults

### Response Shape
- GET /settings returns freq/horizon with available options (valid_freqs, valid_horizons) to help Phase 21 build dropdowns dynamically
- POST /settings returns the updated merged config so the client sees results immediately

### Default Fallback
- Missing user-config fields fall back to default.yaml values (merge strategy, not all-or-nothing)
- Extend the existing config loader in bitbat/config/ rather than creating a separate API settings layer

### Claude's Discretion
- Exact persistence mechanism (YAML file vs SQLite vs other — researcher should evaluate what fits best)
- Internal-only config keys (seed, tau) visibility in GET response — expose or hide based on downstream phase needs
- DELETE/reset endpoint — include if simple, defer if complex
- Error response structure details beyond matching existing patterns

</decisions>

<specifics>
## Specific Ideas

- Success criteria are explicit: GET returns freq=5m/horizon=30m when no user config exists; POST freq=15m/horizon=1h persists and returns unchanged on next GET
- All sub-hourly freq values (5m, 15m, 30m) must be accepted without validation errors
- Phase 21 will consume the settings endpoint to populate React dashboard dropdowns — including valid options in the response avoids hardcoding

</specifics>

<deferred>
## Deferred Ideas

- Custom operator presets with arbitrary freq/horizon combinations (ADVC-01 — future requirement)
- horizon >= freq validation (ADVC-02 — future requirement)
- Multi-user/profile config support

</deferred>

---

*Phase: 20-api-config-alignment*
*Context gathered: 2026-02-28*
