# ONEX Infrastructure Documentation

`omnibase_infra` owns ONEX infrastructure runtime behavior: runtime hosting,
Kafka transport, registration flows, handler loading, config discovery,
infrastructure handlers, operational scripts, and validation checks for the
infrastructure layer.

## Current Truth Boundary

Stable docs in this tree describe current runtime, architecture, operations, and
developer workflows. Historical plans, point-in-time verification reports,
design investigations, and old POC writeups are not primary documentation and
are kept outside this repo's public docs tree when retention is needed.

Definition-of-done evidence for documentation refresh work is tracked in the
change-control evidence system, not in this repository.

## Start Here

| Need | Document |
|------|----------|
| Run the repo locally | [getting-started/quickstart.md](getting-started/quickstart.md) |
| Understand the runtime shape | [architecture/overview.md](architecture/overview.md) |
| Understand node inventory and ownership | [architecture/CURRENT_NODE_ARCHITECTURE.md](architecture/CURRENT_NODE_ARCHITECTURE.md) |
| Understand registration | [architecture/REGISTRATION_WORKFLOW.md](architecture/REGISTRATION_WORKFLOW.md) |
| Understand Kafka and event topics | [architecture/EVENT_BUS_INTEGRATION_GUIDE.md](architecture/EVENT_BUS_INTEGRATION_GUIDE.md), [architecture/EVENT_STREAMING_TOPICS.md](architecture/EVENT_STREAMING_TOPICS.md) |
| Author handlers | [guides/HANDLER_AUTHORING_GUIDE.md](guides/HANDLER_AUTHORING_GUIDE.md) |
| Work with contracts | [reference/contracts.md](reference/contracts.md) |
| Operate services | [operations/README.md](operations/README.md) |
| Run validation | [validation/README.md](validation/README.md) |

## Repository Ownership

| Area | This repo owns | Does not own |
|------|----------------|--------------|
| Core model and enum use | Imports and applies shared core primitives | Defining shared core abstractions |
| SPI protocols | Implements and depends on protocol contracts | Defining downstream implementation behavior in SPI |
| Runtime host | `RuntimeHostProcess`, service kernel, schedulers, domain plugin loading | Product-specific UI or workflow policy |
| Kafka transport | Kafka event bus, topic provisioning, DLQ, consumer-group operations | Domain event meaning outside infra contracts |
| Registration | Registration orchestrator, reducer, storage and service-discovery effects | Non-infra product onboarding policy |
| Secrets/config | Infisical bootstrap, config discovery, prefetch, env fallback rules | Storing secret values in docs |
| Validation | Infra validators and local validation commands | Change-control acceptance evidence |

## Architecture

| Document | Description |
|----------|-------------|
| [overview.md](architecture/overview.md) | High-level ONEX infrastructure runtime architecture |
| [CURRENT_NODE_ARCHITECTURE.md](architecture/CURRENT_NODE_ARCHITECTURE.md) | Node inventory, archetypes, and functional grouping |
| [REGISTRATION_WORKFLOW.md](architecture/REGISTRATION_WORKFLOW.md) | Registration workflow, FSM, topics, and error paths |
| [MESSAGE_DISPATCH_ENGINE.md](architecture/MESSAGE_DISPATCH_ENGINE.md) | Event routing and dispatch internals |
| [HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md](architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) | Current handler loading, handler categories, and classification rules |
| [EVENT_BUS_INTEGRATION_GUIDE.md](architecture/EVENT_BUS_INTEGRATION_GUIDE.md) | Kafka event bus integration |
| [EVENT_STREAMING_TOPICS.md](architecture/EVENT_STREAMING_TOPICS.md) | Topic catalog, schemas, and topic semantics |
| [TOPIC_CATALOG_ARCHITECTURE.md](architecture/TOPIC_CATALOG_ARCHITECTURE.md) | Topic catalog architecture |
| [CONFIG_DISCOVERY.md](architecture/CONFIG_DISCOVERY.md) | Contract-driven config discovery and Infisical prefetch |
| [DLQ_MESSAGE_FORMAT.md](architecture/DLQ_MESSAGE_FORMAT.md) | DLQ message schema |
| [LLM_INFRASTRUCTURE.md](architecture/LLM_INFRASTRUCTURE.md) | LLM infrastructure and cost-tracking boundary |
| [MCP_SERVICE_ARCHITECTURE.md](architecture/MCP_SERVICE_ARCHITECTURE.md) | MCP service layer |
| [SNAPSHOT_PUBLISHING.md](architecture/SNAPSHOT_PUBLISHING.md) | Snapshot publication patterns |
| [SHARED_ENUM_OWNERSHIP.md](architecture/SHARED_ENUM_OWNERSHIP.md) | Shared enum ownership boundary |
| [CIRCUIT_BREAKER_THREAD_SAFETY.md](architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md) | Async circuit breaker concurrency safety |

## Developer Workflows

| Section | Contents |
|---------|----------|
| [getting-started/](getting-started/README.md) | Quickstart, contributor setup, standalone setup, full platform setup |
| [guides/](guides/README.md) | Handler authoring, registration walkthrough, MCP integration, Infisical secrets |
| [reference/](reference/README.md) | Contract reference and node archetypes |
| [patterns/](patterns/README.md) | Implementation patterns for handlers, routing, DI, secrets, errors, retries, security, testing |
| [plugins/](plugins/README.md) | Compute plugin determinism and example plugin guidance |
| [testing/](testing/README.md) | Test strategy and integration testing |

## Operations

| Document | Description |
|----------|-------------|
| [operations/EVENT_BUS_OPERATIONS_RUNBOOK.md](operations/EVENT_BUS_OPERATIONS_RUNBOOK.md) | Kafka event bus deployment, config, monitoring, and troubleshooting |
| [operations/DLQ_REPLAY_RUNBOOK.md](operations/DLQ_REPLAY_RUNBOOK.md) | DLQ replay procedure and safety checks |
| [operations/DATABASE_INDEX_MONITORING_RUNBOOK.md](operations/DATABASE_INDEX_MONITORING_RUNBOOK.md) | Database index monitoring |
| [operations/THREAD_POOL_TUNING_RUNBOOK.md](operations/THREAD_POOL_TUNING_RUNBOOK.md) | Thread pool tuning for sync adapters |
| [runbooks/apply-migrations.md](runbooks/apply-migrations.md) | Database migration execution |
| [runbooks/emergency-runtime-refresh.md](runbooks/emergency-runtime-refresh.md) | Emergency runtime refresh |

## Governance And Standards

| Section | Contents |
|---------|----------|
| [decisions/README.md](decisions/README.md) | Accepted ADRs and architectural rationale |
| [standards/README.md](standards/README.md) | Documentation layout, terminology, and topic taxonomy |
| [validation/README.md](validation/README.md) | Local validation framework and commands |
| [migration/README.md](migration/README.md) | Current migration guides for active legacy removal |
| [performance/README.md](performance/README.md) | Current SLO and performance threshold docs |

## Documentation Maintenance

- Keep current runtime and operational guidance in stable docs.
- Keep one-time investigations, old designs, verification reports, and POC
  writeups out of the primary docs tree.
- Do not link public docs to private workspace repositories.
- If historical material contains current guidance, promote the current guidance
  into a stable doc before removing the historical file.
- Keep root `README.md` and this index aligned whenever sections are added,
  removed, or renamed.
