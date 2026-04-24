# omnibase_infra

Production infrastructure runtime for ONEX.

[![CI](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/test.yml/badge.svg)](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/test.yml)
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
| Kafka event bus and DLQ support | `src/omnibase_infra/event_bus/`, [event bus guide](docs/architecture/EVENT_BUS_INTEGRATION_GUIDE.md) |
| Contract-driven handler discovery | `src/omnibase_infra/runtime/handler_plugin_loader.py`, [handler architecture](docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) |
| Registration orchestration and storage effects | `src/omnibase_infra/nodes/node_registration_*`, [registration workflow](docs/architecture/REGISTRATION_WORKFLOW.md) |
| Infrastructure handlers for DB, HTTP, Consul, Infisical, Kafka, LLM, graph, vector, and filesystem integrations | `src/omnibase_infra/handlers/`, [handler guide](docs/guides/HANDLER_AUTHORING_GUIDE.md) |
| Infisical config discovery and prefetch | `src/omnibase_infra/runtime/config_discovery/`, [config discovery](docs/architecture/CONFIG_DISCOVERY.md) |
| Operational scripts and CLIs | `scripts/`, [operations index](docs/operations/README.md) |

## Start Here

| Need | Document |
|------|----------|
| Understand the repo boundary | [Documentation index](docs/index.md) |
| Run locally | [Quick start](docs/getting-started/quickstart.md) |
| Understand the architecture | [Architecture overview](docs/architecture/overview.md) |
| Work on nodes | [Current node architecture](docs/architecture/CURRENT_NODE_ARCHITECTURE.md) |
| Work on contracts | [Contract reference](docs/reference/contracts.md) |
| Operate Kafka/DLQ/runtime services | [Operations index](docs/operations/README.md) |
| Validate changes | [Validation framework](docs/validation/README.md) |

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

```bash
# Run unit tests
uv run pytest tests/unit

# Run the infra validation suite
uv run python scripts/validate.py all --verbose

# Run markdown link validation
uv run python scripts/validation/validate_markdown_links.py

# Start the runtime CLI entrypoint
uv run onex-runtime

# Inspect operational status
uv run onex-status
```

Some operational flows require Docker, Kafka/Redpanda, PostgreSQL, Valkey, and
Infisical. See [full platform setup](docs/getting-started/full-platform.md) and
[Infisical secrets guide](docs/guides/INFISICAL_SECRETS_GUIDE.md).

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

Current runtime, architecture, operations, and reference guidance belongs in
this repository. Historical plans, one-off verification reports, and design
investigations are not primary docs and are preserved outside this public docs
tree when they still need retention.

Definition-of-done evidence for documentation refresh work is tracked in the
change-control evidence system, not in this repository.

## License

[MIT](LICENSE)
