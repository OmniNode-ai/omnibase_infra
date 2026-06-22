# OMN-13412 — Fresh-deploy fitness gates (Wave E) — dod_evidence

Wire 7 fresh-deploy fitness validators as enforcement (required CI + pre-commit),
not detection. Each broken input fails CI with a non-zero exit BEFORE merge.

## Gate inventory and wiring status

| # | Gate | Logic | CI wiring | Pre-commit |
|---|------|-------|-----------|------------|
| 1 | sibling-pin recurrence ratchet | `scripts/runtime_build/check_sibling_lock_pins.py` (OMN-12977, A4) | `fresh-deploy-fitness.yml` → `sibling-lock-pins` (ratchet logic regression gate) + live compare inside Dockerfile.runtime provenance step | n/a (build-time direction; CI test gate) |
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
sibling-lock-pins exit=0 (clone-ahead descendant note, non-fatal — OMN-13403)
```

## Local gate results (in worktree)

- `ruff format` + `ruff check --fix`: clean
- `mypy --strict` on 3 new scripts: `Success: no issues found in 3 source files`
- `pytest tests/scripts/test_fresh_deploy_fitness_gates.py`: 8 passed
- `pytest tests/scripts/test_check_sibling_lock_pins.py`: 8 passed

## Guardrail compliance

- Did NOT modify OMN-13408 cost-path computation. The terminal-cost gate is a
  static lint; the two annotated `cost_usd=0.0` sites in
  `service_auto_eval_runner.py` are the budget-rejection and exception paths
  (no LLM call / no tokens), annotated `# cost-zero-ok:` — no cost-computation
  logic touched.
- Did NOT touch OMN-13472 ratchet files.
- No skip tokens, no `--no-verify`.
