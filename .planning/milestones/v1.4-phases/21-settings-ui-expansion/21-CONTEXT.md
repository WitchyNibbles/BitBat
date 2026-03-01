# Phase 21: Settings UI Expansion - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

React dashboard frequency and horizon dropdowns expose all supported values (5m, 15m, 30m, 1h, 4h, 1d for freq; 15m, 30m, 1h, 4h, 24h for horizon) with defaults matching default.yaml (5m/30m). Covers SETT-01, SETT-02, SETT-03. Presets and human-readable format helpers are Phase 22. Test coverage is Phase 23.

</domain>

<decisions>
## Implementation Decisions

### Dropdown data source
- Pull option lists from API: use `valid_freqs` and `valid_horizons` from GET /system/settings (Phase 20 already provides these)
- Single source of truth — if bucket.py canonical set changes, UI updates automatically without frontend changes
- Place dropdowns on the existing Settings page/section within the React dashboard (where settings currently live)

### Selection persistence
- API is the source of truth for settings — every settings page load fetches current values from GET /system/settings
- Changing a dropdown PUTs to /system/settings — whether auto-save or explicit save follows existing dashboard patterns
- Selections persist across page navigation because the API stores them (Phase 20's YAML persistence)

### Display format
- Show frequency/horizon values matching the dashboard's existing label conventions
- Dropdowns are independent — both show all valid options regardless of each other's selection (horizon >= freq filtering is deferred per ADVC-02)
- No descriptions/tooltips per option in this phase — Phase 22 presets will add trading context ("Scalper", "Swing")

### Default loading
- On first load with no saved preferences, fetch defaults from API (GET /system/settings returns 5m/30m from default.yaml)
- Loading state while fetching follows existing dashboard loading patterns

### Claude's Discretion
- Auto-save on dropdown change vs explicit Save button — follow existing dashboard save patterns
- Loading state presentation (spinner vs disabled dropdowns) — match existing dashboard conventions
- Whether to show "modified from defaults" indicator or reset button — keep UI minimal unless it fits naturally
- Error handling on API failure — follow existing dashboard error patterns
- Whether other dashboard views update immediately on settings change or on next load
- URL parameter support for freq/horizon — follow existing routing patterns

</decisions>

<specifics>
## Specific Ideas

- Phase 20 already returns `valid_freqs` and `valid_horizons` in the GET /system/settings response — the React dashboard should consume these directly
- Success criteria #4 specifies selections must persist through page navigation without reverting to 1h — this confirms API-backed persistence is required
- The existing React dashboard already has a settings section/page — this phase updates the dropdowns within it, not a redesign

</specifics>

<deferred>
## Deferred Ideas

- Human-readable labels ("5 min" instead of "5m") — Phase 22 (PRES-03)
- Trading context descriptions per option ("best for scalping") — Phase 22 presets cover this
- Horizon filtering based on selected frequency (ADVC-02) — future requirement
- Grouped dropdown sections (sub-hourly vs hourly+) — nice-to-have, not required

</deferred>

---

*Phase: 21-settings-ui-expansion*
*Context gathered: 2026-02-28*
