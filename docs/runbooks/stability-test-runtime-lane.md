# Stability-Test Runtime Lane

Ticket: OMN-10281, extended by OMN-10345

This runbook defines the first non-deploying prep slice for a separate
stability-test runtime lane. It does not start, restart, deploy, or mutate any
runtime host. In particular, it must not mutate `.201`.

## Purpose

The stability-test lane exists so runtime proof can be prepared without sharing
production runtime names, state paths, or consumer groups. It is the bridge from
the two-phase runtime build plan into a concrete runtime lane:

- `BUILD_SOURCE=workspace` is declared in the overlay so selector-aware builds
  use local workspace contents once OMN-9471 or equivalent selector support is
  on main.
- `ONEX_ENVIRONMENT=stability-test` separates derived node consumer groups from
  the production/local `ONEX_ENVIRONMENT=local` group prefix.
- Explicit `ONEX_GROUP_ID` values use the `onex-stability-test-` prefix for the
  coarse runtime group settings.
- `ONEX_RUNTIME_ADDRESS` gives each runtime a routable address instead of
  treating stability-test as a special runtime flag.
- `ONEX_RUNTIME_ID`, `ONEX_BOX_ID`, and `ONEX_RUNTIME_CAPABILITIES` describe
  the concrete runtime target that orchestration can select later.
- `KAFKA_INSTANCE_ID` values are stability-test specific, so selector-aware
  event-bus consumers can append a lane-specific instance discriminator.
- Runtime state is mounted on stability-test volumes and uses
  `/app/data/.onex_state_stability_test`.

## What Is Defined

- Compose overlay: `docker/docker-compose.stability-test.yml`
- Compose project name: `omnibase-infra-stability-test`
- Runtime image tag: `runtime:stability-test-workspace`
- Runtime containers:
  - `omninode-stability-test-runtime`
  - `omninode-stability-test-runtime-effects`
  - `omninode-stability-test-runtime-worker`
- Runtime addresses:
  - `runtime://omninode-pc/stability-test/main`
  - `runtime://omninode-pc/stability-test/effects`
  - `runtime://omninode-pc/stability-test/worker`
- Runtime IDs:
  - `stability-test-main`
  - `stability-test-effects`
  - `stability-test-worker`
- Runtime box ID: `omninode-pc`
- Runtime group IDs:
  - `onex-stability-test-runtime-main`
  - `onex-stability-test-runtime-effects`
  - `onex-stability-test-runtime-workers`
- Runtime state root: `/app/data/.onex_state_stability_test`

## Validation Only

Render the config only:

```bash
docker compose \
  -f docker/docker-compose.infra.yml \
  -f docker/docker-compose.stability-test.yml \
  --profile runtime \
  config
```

List the rendered services and confirm the stability-test runtime services are
present:

```bash
docker compose \
  -f docker/docker-compose.infra.yml \
  -f docker/docker-compose.stability-test.yml \
  --profile runtime \
  config --services
```

Run static validation:

```bash
uv run pytest tests/unit/infra/test_stability_test_runtime_lane.py -q
uv run pytest tests/integration/infra/test_stability_test_runtime_compose_render.py -q
```

## Explicit Non-Goals For This PR

- It does not run `docker compose up`.
- It does not deploy, restart, or change `.201`.
- It does not install launchd, cron, systemd, or autoheal hooks.
- It does not prove runtime health, Redpanda membership, or build-loop to
  ticket-pipeline processing.
- It does not prove workspace image provenance until `BUILD_SOURCE` selector
  support is available on this branch.

## Operator Gate

Any command that starts containers, restarts services, deploys to a host, or
changes `.201` requires explicit operator approval and must produce a separate
runtime proof record.
