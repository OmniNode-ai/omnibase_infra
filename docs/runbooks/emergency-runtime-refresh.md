# Emergency runtime refresh

This runbook captures the **service-targeted runtime refresh** that must be
used any time the automated deploy-agent pipeline is unavailable or the
runtime containers need a surgical rebuild that **must not touch core infra**
(Postgres, Redpanda, Valkey, Infisical, Phoenix).

## When to use this runbook

Use this procedure when any of the following is true:

- The deploy-agent on `.201` is unavailable or wedged.
- The runtime containers must be rebuilt urgently and there is no
  corresponding change to core infra.
- A prior runtime rebuild took core infra offline and the
  operator needs to bring runtime back up without re-recreating core infra.

Do **not** use this runbook for full-stack rebuilds. Use `onex up core runtime`
(or the `infra-up` + `infra-up-runtime` shell wrappers) for those.

## Why it is separate from the core-infra path

`docker compose up -d --force-recreate` walks the `depends_on` graph unless
`--no-deps` is supplied. A runtime-only rebuild without `--no-deps` therefore
collides with the live core infra containers (Postgres/Redpanda/Valkey/
Infisical) and can leave them half-recreated.

The deploy-agent executor now enforces this separation automatically for
`Scope.RUNTIME` (see `scripts/deploy-agent/deploy_agent/executor.py`
`_compose_up`). This runbook is the manual equivalent for the rare cases
where the deploy-agent is not in the loop.

## Procedure

Run on the host owning the compose project (currently `.201`):

```bash
cd /data/omninode/omnibase_infra

# 1. Rebuild the runtime image(s) only — no core-infra rebuild.
docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
  --profile runtime build

# 2. Recreate ONLY the runtime services. --no-deps prevents compose from
#    touching the core infra containers via depends_on.
docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
  up -d --no-deps --force-recreate \
  omninode-runtime \
  runtime-effects \
  runtime-worker \
  agent-actions-consumer \
  skill-lifecycle-consumer \
  context-audit-consumer \
  intelligence-migration \
  intelligence-api \
  omninode-contract-resolver \
  autoheal

# 3. Verify the runtime services reached running state without disturbing
#    core infra.
docker ps --format '{{.Names}}\t{{.State}}\t{{.Status}}' | \
  grep -E 'omninode-runtime|runtime-effects|runtime-worker|agent-actions|skill-lifecycle|context-audit|intelligence-migration|intelligence-api|contract-resolver|autoheal'

# 4. Confirm core infra is still the original containers (no restart/recreate).
docker ps --format '{{.Names}}\t{{.RunningFor}}' | \
  grep -E 'omnibase-infra-postgres|omnibase-infra-redpanda|omnibase-infra-valkey|omnibase-infra-infisical|omnibase-infra-phoenix'
```

Core infra container `RunningFor` values in step 4 should be older than the
start of this procedure. If any core infra container has a fresh `RunningFor`
value, the procedure recreated core infra (bug) — check that `--no-deps` was
actually passed and report a regression against .

## Subset refresh

To refresh a specific runtime subset (for example, just the consumer pods),
pass the subset to step 2 instead of the full runtime list:

```bash
docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
  up -d --no-deps --force-recreate \
  omninode-runtime runtime-effects
```

## Recovery after an accidental full runtime rebuild

If a runtime rebuild was performed without `--no-deps` and broke core infra:

1. Stop the half-recreated core infra containers:
   ```bash
   docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
     stop postgres redpanda valkey infisical phoenix
   ```
2. Bring core infra back cleanly:
   ```bash
   docker compose -f docker/docker-compose.infra.yml -p omnibase-infra \
     --profile core up -d --force-recreate \
     postgres redpanda valkey infisical phoenix
   ```
3. Re-run the runtime refresh procedure above.
4. File an internal follow-up issue with the compose command that was actually
   executed.
