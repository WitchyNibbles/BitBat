# Lessons Learned

## 2026-04-24 - Be honest about platform limits

Issue:

- built-in memory cannot be assumed available everywhere

Fix:

- use repo-local memory as the primary learning layer

Prevention:

- never promise persistent learning without confirming the actual storage path

## 2026-04-24 - Keep the first version operational, not mythical

Issue:

- "agents that improve themselves" can drift into hand-wavy claims

Fix:

- define improvement as reviewed memory, better prompts, safer defaults, and stronger workflows

Prevention:

- require concrete artifacts for every claimed improvement

## 2026-04-25 - Policy and runtime must not be the same thing

Issue:

- repo instructions alone do not create a reliable multi-project operating system

Fix:

- add an explicit shared-core runtime, task packets, locks, reviews, and active work artifacts

Prevention:

- when a workflow claim depends on state or enforcement, back it with code or durable artifacts

## 2026-04-25 - Temporary browser hacks are not a baseline

Issue:

- headless UI verification worked only by extracting a user-space audio library into `/tmp`

Fix:

- capture the host-class dependency explicitly and prefer the machine-level package install path

Prevention:

- for Ubuntu 24.04-class hosts, treat `libasound2t64` as the expected runtime dependency and rerun
  verification without `/tmp` extraction before calling the environment reproducible
