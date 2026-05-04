---
ticket_id: OMN-10089
---

# Seed-Keycloak Orchestrator Split

> **For Claude:** small two-file change; no sub-skill required.

**Goal:** Move the Keycloak seed orchestration (env sourcing, KC_URL polling, localhost-gated `--reset-bootstrap-admin`, `seed-keycloak-clients.py` invocation) out of the top-level `omnibase` Makefile and into `omnibase_infra` where the Docker/Keycloak knowledge belongs. Make Keycloak start on a default `docker compose up -d` so the omnibase distribution Makefile no longer needs to pass `--profile auth`.

**Why:** The top-level `omnibase` repo is the user-facing distribution point — users clone it to run the platform but are not guaranteed to be running Docker. Anything Docker-specific belongs in `omnibase_infra`. Reviewer (jonahgabriel) called this out during PR review of `harsh-omni/omnibase#2`. OMN-10089's acceptance criterion explicitly offers two paths to enable Keycloak by default; this plan picks **"move keycloak service into default profile"** since it preserves the architectural boundary.

**Scope:** Two files in `omnibase_infra`. No application code; no contract changes; no migrations.

---

## Files

- **Create:** `scripts/seed-keycloak.sh` — orchestration shell wrapper. Owns:
  - Sourcing `${OMNIBASE_ENV_FILE:-$HOME/.omnibase/.env}` (skips silently if absent so callers can pre-populate the env vars another way)
  - Validating required vars (`KEYCLOAK_ADMIN_USERNAME`, `KEYCLOAK_ADMIN_PASSWORD`) with fail-fast `${VAR:?}` syntax
  - Polling `${KC_URL:-http://localhost:28080}/realms/${KC_REALM:-omninode}/.well-known/openid-configuration` with a configurable timeout (default 90s)
  - Emitting `docker ps` + `docker logs --tail 40` diagnostics on timeout
  - Caller-side localhost gate for `--reset-bootstrap-admin` (defense in depth alongside the callee gate already in `seed-keycloak-clients.py`)
  - `cd` into repo root and `exec uv run python scripts/seed-keycloak-clients.py ...`
- **Modify:** `docker/docker-compose.infra.yml` — remove `profiles: ["auth", "full"]` from the `keycloak` service so `docker compose up -d` brings it up by default. Update the explanatory comments at the top of the file to reflect the change. The `auth` profile name itself remains valid for selective service startup (e.g., to bring up *only* keycloak); removing the profile gate makes keycloak default-on, which is the OMN-10089 acceptance criterion.

## Acceptance criteria

- `bash scripts/seed-keycloak.sh` from a fresh checkout produces all-`unchanged` output on a re-run against an already-seeded Keycloak (idempotent).
- A clean `docker compose -f docker/docker-compose.infra.yml up -d` (no `--profile` flag) brings up Postgres, Redpanda, Valkey, **and** Keycloak.
- Running `bash scripts/seed-keycloak.sh` with `OMNIBASE_ENV_FILE` pointing at a missing file fails with a clear "must be set" error from the `${VAR:?}` guard, not a confusing shell error.
- Running with `KC_URL=https://staging.example.com/auth` does NOT pass `--reset-bootstrap-admin` to the Python script (caller-side localhost gate works).
- On Keycloak-not-ready timeout, the script prints the last 40 lines of the keycloak container logs before exiting nonzero.

## Out of scope

- Changing `seed-keycloak-clients.py` (already merged in PR #1432, OMN-10088).
- Changing the catalog/`bundles.yaml` system. Keycloak's `auth` bundle membership stays intact for users who want bundle-driven startup; this plan only touches the legacy direct-compose path.
- Wiring this script into a top-level Makefile in omnibase_infra (no such Makefile exists today; not creating one in this PR).

## Cross-repo coordination

This PR must merge before [`harsh-omni/omnibase#2`](https://github.com/harsh-omni/omnibase/pull/2) (OMN-10089 Makefile-side) can be updated to call `bash $(REPOS_DIR)/omnibase_infra/scripts/seed-keycloak.sh` as a thin pass-through and to drop its own `--profile auth` additions. The omnibase PR will be rebased + the body updated to link this PR's merge SHA after merge.

## Test plan

- [ ] Local clean run: `docker compose -f docker/docker-compose.infra.yml down -v && docker compose -f docker/docker-compose.infra.yml up -d` — verify keycloak container is in the resulting `docker ps`.
- [ ] `bash scripts/seed-keycloak.sh` against the freshly-started Keycloak — verify exit 0 and JSON `op` lines.
- [ ] Re-run `bash scripts/seed-keycloak.sh` — verify all `op=unchanged`.
- [ ] `OMNIBASE_ENV_FILE=/nonexistent bash scripts/seed-keycloak.sh` — verify clear failure message naming the missing env vars.
- [ ] `KC_URL=https://example.invalid bash scripts/seed-keycloak.sh` (without admin vars set) — verify the `--reset-bootstrap-admin` flag is NOT in the final `uv run python` invocation (`bash -x` to inspect).
