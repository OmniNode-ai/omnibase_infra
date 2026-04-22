# Emergency Runtime Refresh Runbook

## When to Run This

A runtime-only refresh is required when the `omninode-runtime` and/or
`runtime-effects` containers need to be recreated without disturbing the
core infrastructure services (`postgres`, `redpanda`, `valkey`,
`omnibase-infra-infisical`, `phoenix`).

Typical triggers:

- A new `omnibase_infra` release pinning a runtime image needs to take
  effect on `.201`.
- The runtime container is crash-looping or stuck in an unhealthy state
  but core infra is healthy and in use by other services.
- The normal deploy-agent path is unavailable and you need to restore
  the runtime quickly.

## Why This Runbook Exists (OMN-9455)

On 2026-04-22, a runtime-only rebuild request was accepted by the
deploy-agent on `.201`, but the daemon ran

```bash
docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
  --profile runtime up -d --force-recreate --pull always
```

without `--no-deps`. `docker compose` therefore recreated every service
in the runtime profile's dependency graph — including the
`omnibase-infra-infisical` container that was already running — and
collided with it, leaving Redpanda, Postgres, Valkey, and Phoenix
partially broken.

Manual recovery succeeded only after narrowing the rollout to an
explicit service list with `--no-deps`:

```bash
docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
  up -d --no-deps --force-recreate \
  omninode-runtime runtime-effects
```

The deploy-agent path has since been fixed to always use `--no-deps`
plus an explicit service list for `Scope.RUNTIME` rebuilds (see
`scripts/deploy-agent/deploy_agent/executor.py::_compose_up`). This
runbook codifies the manual fallback path for when the daemon is
unavailable.

## Prerequisites

- SSH access to `.201` (`ssh jonah@192.168.86.201`).
- Working directory is the `omnibase_infra` checkout on the host
  (default: `/data/omninode/omnibase_infra`).
- Core infra (`postgres`, `redpanda`, `valkey`, `infisical`) must
  already be running and healthy. If they are not, run the core bundle
  first — see `~/.claude/CLAUDE.md` for `infra-up` / `onex up core`.

## Step 1 — Confirm Core Infra Is Healthy Before Touching Runtime

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | \
  grep -E 'omnibase-infra-(postgres|redpanda|valkey|infisical)'
```

All four must report `Up ... (healthy)`. If any are unhealthy, recover
core first — `--no-deps` below will NOT pull them up for you by design.

## Step 2 — Run the Targeted Runtime Refresh

From the `omnibase_infra` repo root on `.201`:

```bash
docker compose \
  -f docker/docker-compose.infra.yml \
  -p omnibase-infra \
  up -d --no-deps --force-recreate --pull always \
  omninode-runtime runtime-effects
```

Flags explained:

- `-f docker/docker-compose.infra.yml` — the compose file that the
  deploy-agent also uses, so the project namespace lines up.
- `-p omnibase-infra` — the project name that matches running
  containers; without this, `docker compose` will create a parallel
  project and leave your live runtime untouched.
- `--no-deps` — **critical.** Prevents compose from recreating
  dependency-graph services (postgres, redpanda, valkey, infisical,
  phoenix). This is the exact flag the daemon now emits for
  `Scope.RUNTIME`.
- `--force-recreate` — recreate the container even when the image
  digest did not change (useful when pulling a new tag or refreshing
  environment variables).
- `--pull always` — always fetch the latest image tag before
  recreating, matching the daemon's behavior.
- Explicit service list (`omninode-runtime runtime-effects`) —
  compose will act only on the services you name. If you need to
  refresh a different runtime service, add it by name; do **not** omit
  the list and let compose fan out across the whole runtime profile.

## Step 3 — Verify Only the Targeted Services Came Up

```bash
for svc in omninode-runtime runtime-effects; do
  docker ps --filter "name=${svc}" \
    --format '{{.Names}}\t{{.Status}}'
done
```

Both must report `Up ... (healthy)` or `Up ... (starting)` within the
compose `start_period` (currently 600s for `omninode-runtime`, see
`docker/docker-compose.infra.yml`).

Then confirm core infra is still the same instance it was before —
container IDs must not have changed:

```bash
docker ps --format '{{.Names}} {{.ID}}' | \
  grep -E 'omnibase-infra-(postgres|redpanda|valkey|infisical)'
```

If any core container ID changed while you were running the runtime
refresh, the `--no-deps` flag was not applied — stop and escalate.

## Step 4 — Tail Runtime Logs for a Minute

```bash
docker logs -f --tail 200 omninode-runtime
```

Look for `registration_projections` populated, no crash-loop
`RestartCount`, and no handler wiring errors.

## What NOT To Do

- **Do not** run `docker compose ... up -d` without `--no-deps` to
  "refresh runtime." That is the exact command shape that caused the
  2026-04-22 outage.
- **Do not** pass `--profile runtime` without `--no-deps` and an
  explicit service list. The profile selects services; the dependency
  graph still reaches into core.
- **Do not** restart core infra as a side effect. If core needs to
  cycle, do that deliberately via `onex up core` — never as collateral
  of a runtime rebuild.

## Related

- Code: `scripts/deploy-agent/deploy_agent/executor.py` (`_compose_up`)
- Tests: `scripts/deploy-agent/tests/unit/test_executor_runtime_scope.py`
- Ticket: [OMN-9455](https://linear.app/omninode/issue/OMN-9455)
- Parent: [OMN-9235](https://linear.app/omninode/issue/OMN-9235)
