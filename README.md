<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/omninode-inline-white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/omninode-inline-full-color.svg">
    <img alt="omninode" src="docs/assets/brand/omninode-inline-full-color.svg" width="420">
  </picture>
</p>

# omnibase_infra

Production infrastructure runtime for ONEX (OmniNode eXecution).

[![CI](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/ci.yml/badge.svg)](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`omnibase_infra` owns the infrastructure implementation layer for ONEX:
runtime hosting, Kafka event transport, contract-driven handler loading,
registration workflows, Infisical-backed configuration, operational runbooks,
and infrastructure nodes that perform external I/O.

It depends on `omnibase_core` for shared models and validation primitives, and
on `omnibase_spi` for protocol boundaries. Core must not import this package.

## What This Repo Owns

| Area | Current source |
|------|----------------|
| Runtime host process and service kernel | `src/omnibase_infra/runtime/` |
| Kafka event bus and DLQ support | `src/omnibase_infra/event_bus/` |
| Contract-driven handler discovery | `src/omnibase_infra/runtime/handler_plugin_loader.py` |
| Registration orchestration and storage effects | `src/omnibase_infra/nodes/node_registration_*` |
| Infrastructure handlers for DB, HTTP, Consul, secrets, Kafka, LLM, graph, vector, and filesystem integrations | `src/omnibase_infra/handlers/` |
| Config discovery and prefetch | `src/omnibase_infra/runtime/config_discovery/` |
| Operational scripts and CLIs | `scripts/` |

Prose for every row above lives in the knowledge base — see [Documentation](#documentation).

## Documentation

**This repository holds code, not prose.** Every architecture note, pattern,
decision record, guide, reference page, and runbook that used to live under
`docs/` now lives in one of the two knowledge bases. There are no pointer stubs
in the tree — this section is the pointer, and the `kb-doc-gate` check
(`.kb-doc-gate.yaml`, `mode: strict`) keeps it that way.

| Home | What is there |
|------|---------------|
| [`OmniNode-ai/knowledge-base`](https://github.com/OmniNode-ai/knowledge-base) (public) | Platform documentation anyone can read: `architecture/`, `reference/`, `guides/`, `runbooks/`, and the ADR ledger. This repository's pages are prefixed `omnibase-infra-`. |
| [`OmniNode-ai/knowledge-base-internal`](https://github.com/OmniNode-ai/knowledge-base-internal) (private, teammates) | Documentation that names real internal topology, the lab and CI fleet, deploy lanes, or the secrets manager — same `omnibase-infra-` prefix under `reference/` and `runbooks/`. |

| Need | Where |
|------|-------|
| Run locally / first-time bootstrap | `runbooks/omnibase-infra-quickstart.md`, `runbooks/omnibase-infra-full-platform-setup.md` (internal) |
| Understand the handler architecture | `reference/omnibase-infra-handler-protocol-driven-architecture.md` (internal) |
| Work on contracts | `reference/omnibase-infra-contract-yaml-reference.md` (public) |
| Node archetypes and registration | `reference/omnibase-infra-node-archetypes.md` (internal), `reference/omnibase-infra-node-registration-orchestrator.md` (public) |
| Operate Kafka / DLQ / the event bus | `runbooks/omnibase-infra-event-bus-operations.md`, `runbooks/omnibase-infra-dlq-replay.md` (public) |
| Validate changes | `reference/omnibase-infra-validation-framework.md` (internal) |
| Decision records for this repo | `reference/omnibase-infra-adr-*.md` (public) |

## Install

Library or CLI use:

```bash
uv add omnibase-infra
```

Repository development:

```bash
git clone https://github.com/OmniNode-ai/omnibase_infra.git
cd omnibase_infra
uv sync
```

The packaged distribution includes the Python package and console scripts. The
repo-local `scripts/` directory is for development, operational bootstrap, and
CI support, and requires a clone.

## Common Commands

### Local infrastructure lifecycle

The repo ships a top-level `Makefile` with user-facing infra entrypoints.
All Docker/Compose/Keycloak orchestration lives here, never in
the public `omnibase` shell.

```bash
# Start core infra bundle (postgres, redpanda, valkey, infisical)
make up

# Add Keycloak (auth bundle) on top of core
make up-auth

# Reconcile Keycloak clients from desired-clients.json
make seed-keycloak

# Seed Infisical from ONEX contracts (writes with --execute)
make seed-infisical

# Show running containers
make status

# Stop the core bundle ONLY (auth/runtime stay running)
make down

# Stop the auth bundle (keycloak)
make down-auth

# Stop everything (runtime + auth + core, in safe teardown order)
make down-all

# List all targets
make help
```

`make` targets detect a missing/stopped Docker daemon and emit an actionable
error before doing anything destructive. They also detect a missing
`~/.omnibase/.env` and point at remediation rather than failing with a stack
trace. The full first-time bootstrap sequence is
`runbooks/omnibase-infra-full-platform-setup.md` in the internal knowledge base.

### Development and testing

```bash
# Run unit tests
uv run pytest tests/unit

# Run the infra validation suite
uv run python scripts/validate.py all --verbose

# Start the runtime CLI entrypoint
uv run onex-runtime

# Inspect operational status
uv run onex-status
```

Some operational flows require Docker, Kafka/Redpanda, PostgreSQL, Valkey, and
the secrets manager. See `runbooks/omnibase-infra-full-platform-setup.md` and
`runbooks/omnibase-infra-infisical-secrets.md` in the internal knowledge base.

## Runtime Shape

ONEX infra uses four node archetypes:

| Archetype | Role |
|-----------|------|
| `ORCHESTRATOR` | Coordinates workflows and publishes events |
| `REDUCER` | Owns pure FSM state transitions and emits intents |
| `COMPUTE` | Performs deterministic transformations with no side effects |
| `EFFECT` | Performs external I/O through infrastructure handlers |

Node behavior is declared in `contract.yaml`. Node classes are intentionally
thin; runtime behavior comes from contracts, handlers, registries, and the
runtime host.

## Documentation Policy

Documentation does not live in this repository. The allowed markdown set is
this `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `SECURITY.md`, anything under
`.claude/` or `.github/`, and test fixtures a test actually opens as data. The
`kb-doc-gate` required check enforces that list in `strict` mode, so a new
document under `docs/` fails CI rather than quietly re-growing the tree.

New prose goes to [the public knowledge base](https://github.com/OmniNode-ai/knowledge-base) by default, and to
[the internal knowledge base](https://github.com/OmniNode-ai/knowledge-base-internal) when it names real internal topology, the
lab or CI fleet, deploy lanes, or the secrets manager. Dated point-in-time
artifacts — evidence bundles, audit snapshots, run transcripts — are not
documentation and are not migrated; definition-of-done evidence is tracked in
the change-control evidence system.

## License

[MIT](LICENSE)
