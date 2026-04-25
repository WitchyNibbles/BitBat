# Patterns

## Good Patterns

- use narrow specialist agents instead of one giant "expert"
- compress agent chatter with the caveman schema
- plan first, then build the thinnest vertical slice
- run security and QA review before calling work done
- capture only durable lessons in memory
- keep repo policy in markdown and operational state in the shared core
- use explicit task packets with write scope, tests, and rollback notes
- prefer one writer per overlapping write scope
- after shadow-run evidence is captured, shut down residual sessions unless a named soak owner and
  time window are recorded
- when a host dependency is missing for browser verification, document the machine-level package fix
  instead of normalizing a `/tmp` workaround
- when market polling is automated, make duplicate-candle suppression part of the runtime contract
  so manual and background paths cannot double-trade the same exchange candle

## Anti-Patterns

- vague scope with no acceptance criteria
- pretending "self-improvement" exists without stored evidence
- huge roadmaps before a working thin slice
- storing secrets in project memory
- direct worker writes to shared state outside the service layer
- broad file ownership without lock discipline
