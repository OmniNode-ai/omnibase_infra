# Fresh-deploy fitness gates (Wave E) — dod_evidence

Wire 7 fresh-deploy fitness validators as enforcement (required CI + pre-commit),
not detection. Each broken input fails CI with a non-zero exit BEFORE merge.

## Gate inventory and wiring status

| # | Gate | Logic | CI wiring | Pre-commit |
|---|------|-------|-----------|------------|
| 1 | sibling-pin recurrence ratchet | `scripts/runtime_build/check_sibling_lock_pins.py` | `fresh-deploy-fitness.yml` → `sibling-lock-pins` (ratchet logic regression gate) + live compare inside Dockerfile.runtime provenance step | n/a (build-time direction; CI test gate) |
| 2 | build-provenance version-skew | `scripts/check-pinned-wheels.py` (A6) | `fresh-deploy-fitness.yml` → `pinned-wheel-skew` (lifts the build-time assertion into CI; version-skew fails the build not the deploy) | n/a (needs `gh` auth) |
| 3 | scratch-Postgres cold-apply | `scripts/run-migrations.py` | `ci.yml` → `migration-integration` (blank PG service, applies 0001→N, asserts 24 tables, fails on any error) — pre-existing required job | n/a |
| 4 | vendored-tree byte-equality | `scripts/sync-node-migrations.sh --check` | `node-migration-sync.yml` (pre-existing required) | `onex-check-node-migration-sync` (pre-existing) |
| 5 | terminal cost completeness | `scripts/check_terminal_cost_completeness.py` (NEW) | `fresh-deploy-fitness.yml` → `terminal-cost-completeness` | `check-terminal-cost-completeness` |
| 6 | context-ROI field presence | `scripts/check_context_field_presence.py` (NEW) | `fresh-deploy-fitness.yml` → `context-field-presence` | `check-context-field-presence` |
| 7 | release identity | `scripts/check_release_identity.py` (NEW) | `fresh-deploy-fitness.yml` → `release-identity` | `check-release-identity` |

Items 3 and 4 were ALREADY wired as required gates before this PR (verified
against `origin/dev` — `ci.yml:migration-integration` in the
`required-status-check` aggregator at line ~1369; `node-migration-sync.yml`
required). This PR adds items 1, 2, 5, 6, 7.

## Required-status-check registration (admin action)

`fresh-deploy-fitness.yml` jobs become blocking the same way `node-migration-sync`
does — branch protection required contexts (configured out-of-band by repo admin,
not in repo source). Add these contexts to `dev` (and `main`) branch protection on
**omnibase_infra** (and `release-identity` + `terminal-cost-completeness` +
`context-field-presence` on **omnimarket** where applicable):

- `release-identity`
- `terminal-cost-completeness`
- `context-field-presence`
- `sibling-lock-pins`
- `pinned-wheel-skew`

## DoD proof — deliberately broken input fails CI (non-zero exit)

```
--- [item 5] hardcoded cost_usd=0.0 (un-annotated) ---
exit=1   # check_terminal_cost_completeness.py .dod_bad_cost.py

--- [item 6] context-ROI claim missing context_pack_hash ---
exit=1   # check_context_field_presence.py .dod_c/contract.yaml

--- [item 7] version-skew: code on already-published version, no bump ---
exit=1   # check_release_identity.py --changed-file src/...  (version pinned to latest tag)
```

Each correct input passes (exit 0):

```
terminal-cost   exit=0  (annotated legitimate zero paths in service_auto_eval_runner.py)
context-field   exit=0  (no contract makes an unpinned ROI claim — clean ratchet)
release-identity exit=0 (pyproject 0.38.4 ahead of latest published v0.38.3)
sibling-lock-pins exit=0 (clone-ahead descendant note, non-fatal)
```

## The release train disarms gate 7 itself (OMN-13912)

Gate 7 (`release-identity`) is **armed by the release train**: publishing
`X.Y.Z` from dev HEAD leaves dev's `[project].version` exactly equal to the
highest published tag, so the next dev PR touching `src/**` goes red — and
stays red for every PR after it — until someone unrelated bumps.

Measured twice in the v0.38 series:

| Tag | Tagged at dev HEAD | Dev bumped by | Armed window |
|-----|--------------------|---------------|--------------|
| `v0.38.10` (`5d3f77792`) | 2026-08-26T01:38:23Z | `a07fefde4` (OMN-16536, unrelated) 2026-08-26T03:44:31Z | ~2h06m |
| `v0.38.11` (`4529c3486`) | 2026-08-28T00:49:31Z | `93c42ada4` (OMN-16769, unrelated) 2026-08-28T02:27:16Z | ~1h38m |

The disarm is now part of the same flow that arms it: `release.yml` job
`post-release-dev-bump` runs `scripts/ci/post_release_dev_bump.py`, which bumps
`[project].version` to `published + 1 patch` (and re-locks `uv.lock`), then
opens an auto-merging PR against `dev`. It is a no-op when dev is already
ahead, so a re-dispatched tag or a hand-bump does not produce a second PR.

Two deliberate design points, both proven in
`tests/ci/test_post_release_dev_bump_workflow.py`:

- The job is gated on `needs.release.outputs.version != ''` (publish
  succeeded), **not** on `needs.release.result == 'success'`. Both windows
  above published cleanly and then went red on the unrelated `Sync main to
  release tag` GH006 (OMN-16343); a success-gated disarm would have been
  skipped in exactly the two cases that needed it.
- The bump lands via a PR against `dev`, never a direct push — a direct push
  would bypass every required check on the branch.

`release-identity` being *advisory* on the PR path (not aggregated by
`CI Summary`, not a required context) is a separate defect tracked as
OMN-16819; this change removes the recurring arming event, it does not make the
gate blocking.

## Local gate results (in worktree)

- `ruff format` + `ruff check --fix`: clean
- `mypy --strict` on 3 new scripts: `Success: no issues found in 3 source files`
- `pytest tests/scripts/test_fresh_deploy_fitness_gates.py`: 8 passed
- `pytest tests/scripts/test_check_sibling_lock_pins.py`: 8 passed

## Guardrail compliance

- Did NOT modify cost-path computation. The terminal-cost gate is a
  static lint; the two annotated `cost_usd=0.0` sites in
  `service_auto_eval_runner.py` are the budget-rejection and exception paths
  (no LLM call / no tokens), annotated `# cost-zero-ok:` — no cost-computation
  logic touched.
- Did NOT touch delegation-telemetry ratchet files.
- No skip tokens, no `--no-verify`.
