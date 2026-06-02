# Stability-Test Runtime Lane

Ticket: OMN-10281, extended by OMN-10345

This runbook defines the first non-deploying prep slice for a separate
stability-test runtime lane. It does not start, restart, deploy, or mutate any
runtime host. In particular, it must not mutate `.201`.

## Purpose

The stability-test lane exists so runtime proof can be prepared without sharing
production runtime names, state paths, or consumer groups. It is the bridge from
the two-phase runtime build plan into a concrete runtime lane:

- The overlay inherits the release runtime image build from the base compose
  file instead of declaring a lane-specific workspace image.
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
- The overlay does not expose inherited production host ports in the rendered
  stability-test config.
- The overlay does not render inherited out-of-lane runtime services such as
  observability consumers, contract resolver, Phoenix, autoheal, or Infisical.

## What Is Defined

- Compose overlay: `docker/docker-compose.stability-test.yml`
- Compose project name: `omnibase-infra-stability-test`
- Runtime image: inherited from the base runtime build.
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
- Published host ports:
  - Postgres: `15436`
  - Redpanda Kafka: `39092`
  - Redpanda admin: `29644`
  - Valkey: `26379`
  - Runtime main: `18085`
  - Runtime effects: `18086`
- Build args: inherited from the base runtime build.
- Networks:
  - `omnibase-infra-stability-test-network`
  - `omnibase-infra-stability-test-omnimemory-network`

## Connected-Network Kafka Access

The stability-test Redpanda external listener must advertise one address that is
reachable by every authorized stability-test operator. Do not advertise
`localhost` or a LAN-only address such as `192.168.86.201` for this lane: Kafka
clients bootstrap on the supplied broker, then reconnect to the broker address
returned in metadata.

On `.201`, use the Tailscale address or a stable MagicDNS name as the
stability-test advertise host:

```bash
export STABILITY_TEST_REDPANDA_ADVERTISE_HOST=100.109.203.94
export STABILITY_TEST_REDPANDA_EXTERNAL_PORT=39092
```

Operators should then bootstrap with the same connected-network endpoint:

```bash
export KAFKA_BOOTSTRAP_SERVERS=100.109.203.94:39092
kcat -L -b "$KAFKA_BOOTSTRAP_SERVERS" | sed -n '1,5p'
```

The metadata must report broker `0` at the connected-network host and port. If
it reports `localhost` or `192.168.86.201`, off-LAN clients can open the TCP port
but fail when Kafka redirects them to the advertised broker address.

## Validation Only

Render the config only:

```bash
STABILITY_TEST_REDPANDA_ADVERTISE_HOST=100.109.203.94 \
docker compose \
  -f docker/docker-compose.infra.yml \
  -f docker/docker-compose.stability-test.yml \
  --profile runtime \
  config
```

The overlay requires Docker Compose v2.24.4 or later because it uses Compose
`!override` merge semantics to replace inherited port and profile lists.

List the rendered services and confirm the stability-test runtime services are
present:

```bash
STABILITY_TEST_REDPANDA_ADVERTISE_HOST=100.109.203.94 \
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
- It does not prove workspace image provenance; this lane intentionally follows
  the release runtime build until a separately approved selector change lands.

## Operator Gate

Any command that starts containers, restarts services, deploys to a host, or
changes `.201` requires explicit operator approval and must produce a separate
runtime proof record.
