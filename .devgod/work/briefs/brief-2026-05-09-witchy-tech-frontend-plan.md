# Intake Brief

## Task ID

`2026-05-09-witchy-tech-frontend-plan`

## Brief ID

`brief-2026-05-09-witchy-tech-frontend-plan`

## Request

Original user ask:
Make a plan to refactor the front end, look for front-end references, theme it witchy tech, make it informative so the user can know for sure what is happening, add the paper trade view, research properly, and research skills that can make devgod produce excellent frontends.

## Goal

Produce a source-backed frontend refactor plan for this repository that:
- inventories the current frontend surface and nearby operator/dashboard flows
- defines a distinctive "witchy tech" visual and interaction direction
- improves operator clarity so runtime state, model state, strategy state, and execution state are explicit
- adds a paper trading view to the target UI architecture
- identifies repo-local skills and workflow support that best improve frontend quality under devgod

## Audience

- repository maintainers
- operators using the dashboard or future web frontend
- future implementation agents working from the plan

## Constraints

- planning and research only in this task
- must align with repo-local devgod workflow artifacts
- must use repo evidence plus external reference research
- frontend direction should preserve informative, trustworthy operator UX over decorative styling

## Risks

- current frontend surface may be split across legacy and v2 modules
- "witchy tech" styling can drift into novelty and hurt operational clarity
- paper trading state can be under-specified in the UI if backend events are not mapped cleanly
- frontend skill recommendations can become generic if not tied to this repo's actual workflow

## Unknowns

- whether the long-term frontend target is Streamlit refresh, React app, or hybrid
- which backend endpoints and event streams are stable enough for a dedicated paper trade cockpit
- what level of live refresh and audit history already exists in the current UI layer

## Success criteria

- current frontend references are inventoried with file pointers
- external references are documented with rationale tied to this product
- final plan includes IA, view model, theming system, phased execution, and verification gates
- paper trading view requirements are explicit
- recommended devgod/frontend skills are concrete and prioritized

## Out of scope

- implementing the refactor
- shipping UI code
- backend API changes beyond planning notes

## Trust boundaries

- frontend must treat model output, market data, and broker execution state as external inputs requiring explicit validation and status labeling
- operator-facing claims must distinguish prediction, signal, order intent, submitted order, and filled order

## Stop/go

`go`

## Next step

Planner action required:
Run architecture, product/UX, repo inventory, external design research, and frontend-skill research; then consolidate into a plan artifact for implementation.
