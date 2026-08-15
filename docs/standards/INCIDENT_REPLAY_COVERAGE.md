# Incident-Replay Coverage

<!-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc. -->
<!-- SPDX-License-Identifier: MIT -->

> **The enforcing definition is `scripts/ci/check_incident_replay_coverage.py`, not this
> page.** This page exists so the other repos have something to cite while they adopt it.
> If the two ever disagree, the script wins — a convention that lives only in prose is the
> thing this convention exists to replace.

## The rule

**Every enforcement guard carries at least one regression case sourced from a real
incident.** A guard that has never been run against the real thing it exists to catch is
decorative.

On 2026-07-30 that failed three times in one day, with an identical shape every time —
the guard was tested against a synthetic input that *could not exhibit the failure*:

| Guard | Synthetic input | What it missed |
|---|---|---|
| workflow-pin validators (5 repos) | any `[0-9a-f]{40}` | `879d6fc6`, the pin that wedged every open `omnibase_infra` PR for ~2.5h (OMN-15536). Re-running the validator against it returns `2 passed`. |
| `.201` system-health alert | a 63-byte `HEALTHY_BODY` | a 180-byte pre-parse truncation. Real bodies are 2079–2644 bytes, so `jq` died on every lane and the alert paged CRITICAL against a healthy fleet (OMN-15525). |
| merge-hold falsifier | validated under `bash -c` / `sh -c` | the runner reads *command position*, so a probe wrapped in `python3 -c "…"` classified `NOT_EXECUTED` — the check never ran at all (OMN-15484). |

In all three the guard was green, the enforcement was zero, and nothing in CI could tell
the difference.

## What counts as "real"

A fixture is real only if it **came from an actual failure**, not from a hand-typed
approximation of one. That distinction has to survive contact with a machine, so it is
five checkable rules rather than an intention:

| Rule | Requirement | Why a machine can apply it |
|---|---|---|
| **R1** | `artifact.fixture` is a committed FILE **ending in `.captured`**, and `sha256(bytes)` matches `artifact.sha256` | A literal inside a test cannot be hashed or diffed against an origin. Editing a capture breaks the claim, so it must break the build. The suffix is load-bearing, not cosmetic: these very fixtures were rewritten by `end-of-file-fixer` on their first commit because they still ended in `.json`, and R1 is what caught it. |
| **R2** | `capture.source` matches a locator grammar: `gh-api:`, `git-object:`, `host-file:`, `live-http:`, `ci-artifact:` | **This is the whole discriminator.** An invented payload has no locator that resolves. Free-text provenance ("same shape as prod") is rejected on purpose — that prose is exactly what the OMN-15525 repair wrote above a fixture it had typed by hand. |
| **R3** | `incident` is `OMN-<n>` or `<owner>/<repo>#<n>` | A replay case has to name the failure it replays. |
| **R4** | `test` exists and references the fixture path | A registry entry nobody reads is paperwork. |
| **R5** | `regression_class` (`false_green` → verdict `reject`; `false_red` → verdict `accept` **plus** a `discriminator` test) | Pins the verdict the buggy guard got *wrong*. A `false_red` proof alone cannot tell a working guard from one stuck open, so it must be paired. |

## Coverage is ratcheted, and new guards default-deny

- `scope.required_guards` — each must have a valid case. **Append-only.** An entry whose
  file does not exist yet reports `PENDING`: that is how a requirement is armed *before*
  the guard lands (`check_pin_reachability.py` is pre-registered against OMN-15538).
- `scope.debt_baseline` — the wired guards with no case yet, enumerated so the debt is
  countable. **May only shrink.** Being on it is a debt record, not permission.
- **DEFAULT-DENY** — a newly wired guard in neither list fails. This is the load-bearing
  property: it is what stops the detection shelf growing faster than the proof behind it.

The lint fired on itself the first time it was wired, and the case that resolved it
(`omn15547-handtyped-fixture-passed-as-proof`) replays the verbatim dev blob that shipped
a hand-typed fixture as proof. That is the intended experience.

## Adding a case

1. **Get the real bytes.** `gh api` the run/PR, `git cat-file` the object, `curl` the live
   endpoint, `scp` the host file. Do not retype them from a report.
2. **Commit them verbatim** under `tests/fixtures/<incident>/`, with a `.captured` infix so
   header and format hooks leave the bytes alone — a reformatted artifact is no longer the
   artifact that failed.
3. **Record** the sha256 and a re-fetchable `capture.source`.
4. **Write the test** that drives the real guard with those bytes and asserts the verdict
   the buggy guard got wrong.

## Status

Adopted in `omnibase_infra` (OMN-15547): 99 wired guards, 2 covered, 97 baselined.
Rollout to `omnibase_core`, `omnimarket`, `omniclaude` and `onex_change_control` is
registry + wiring only — the lint takes `--repo-root`.

Fleet-wide starting point (2026-07-31 audit): of **361** wired guards across the five CI
repos, **14 (3.9%)** replay anything real; 206 are synthetic-only and 105 have no test at
all. The closest pre-existing exemplar is OCC's
`tests/fixtures/evidence_admissibility_cases.yaml` (OMN-15309) — 21 cases, each pinned to
a named defect class, missing only the locator and byte-parity this convention adds.
