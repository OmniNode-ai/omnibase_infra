# CLAUDE.md - Omnibase Infrastructure

> **Python**: 3.12+ | **Framework**: ONEX Infrastructure
>
> **Shared standards**: See **`~/.claude/CLAUDE.md`** for Python/Git/Testing standards, PEP 604 type unions, architecture principles, environment configuration, infrastructure topology, PostgreSQL, Kafka/Redpanda, Docker networking, LLM endpoints, and environment variables. Those rules apply to this repo and are not repeated here.

---

## Table of Contents

1. [Repo Invariants](#repo-invariants)
2. [Non-Goals](#non-goals)
3. [Service Catalog Architecture](#service-catalog-architecture)
4. [Quick Reference](#quick-reference)
5. [Architecture: Four-Node Pattern](#architecture-four-node-pattern)
6. [Declarative Nodes](#declarative-nodes)
7. [Handler System](#handler-system)
8. [Intent Model Architecture](#intent-model-architecture)
9. [Error Handling](#error-handling)
10. [Infrastructure Patterns](#infrastructure-patterns)
11. [Pydantic Model Standards](#pydantic-model-standards)
12. [Testing and CI](#testing-and-ci)
13. [Contract-Driven Config Discovery](#contract-driven-config-discovery)
14. [Agent-Driven Development](#agent-driven-development)
15. [Common Pitfalls](#common-pitfalls)
16. [Release Process](#release-process)

---

## Repo Invariants

These are non-negotiable architectural truths:

- **Nodes are declarative** - `node.py` extends base class with NO custom logic
- **Handlers own logic** - Business logic lives in handlers, not nodes
- **Reducers are pure** - `delta(state, event) -> (new_state, intents[])` with no I/O
- **Orchestrators emit, never return** - ORCHESTRATOR nodes cannot return `result`
- **Contracts are source of truth** - YAML contracts define behavior, not code
- **Unidirectional flow** - EFFECT → COMPUTE → REDUCER → ORCHESTRATOR, never backwards
- **Container injection** - All services use `ModelONEXContainer` for DI

---

## Non-Goals

We explicitly do **NOT** optimize for:

- **Backwards compatibility** - This repo has no external consumers. Schemas, APIs, and interfaces may change without deprecation periods. If something needs to change, change it. No `_deprecated` suffixes, no shims, no compatibility layers.
- **Convenience over correctness** - Contract violations fail loudly
- **Business logic in nodes** - Nodes coordinate; handlers compute
- **Dynamic runtime behavior** - All behavior must be contract-declared
- **Implicit state** - All state transitions are explicit and auditable
- **Tight coupling** - Protocol-based DI enforces loose coupling
- **Versioned directories** - NEVER create `v1_0_0/`, `v2/` directories; version through `contract.yaml` fields only

**When you see deprecated or unused code: DELETE IT.** Do not:
- Leave it "for reference"
- Comment it out
- Add deprecation warnings
- Create compatibility shims
- Keep old function signatures with forwarding

---

## Install Model

`omnibase_infra` ships as both a **pip-installable package** and a **cloneable repository**.
The two serve different purposes.

### Pip Package (library + runtime CLIs)

Install via pip for:
- Using `omnibase_infra` as a library dependency in other ONEX services
- Running the bundled runtime CLIs

```bash
pip install omnibase-infra
# or
uv add omnibase-infra
```

**Bundled CLI entry points** (available after `pip install omnibase-infra`):

| Command | Entry Point | Purpose |
|---------|-------------|---------|
| `omni-infra` | `omnibase_infra.cli.commands:cli` | General CLI |
| `onex-runtime` | `omnibase_infra.runtime.kernel:main` | Start ONEX runtime |
| `onex-infra-test` | `omnibase_infra.cli.infra_test.cli:cli` | Infra test runner |
| `onex-git-hook-relay` | `omnibase_infra.cli.git_hook_relay:main` | Git hook relay |
| `onex-linear-relay` | `omnibase_infra.cli.linear_relay:main` | Linear relay |
| `onex-status` | `omnibase_infra.tui.__main__:run_status_tui` | Status TUI |

### Local Clone (operational scripts)

A **local clone is required** to run the operational scripts in `scripts/`. These scripts
are **not bundled** in the pip package — they live only in the repository source tree.

```bash
git clone https://github.com/OmniNode-ai/omnibase_infra.git
cd omnibase_infra
uv sync
```

**Scripts that require a local clone:**

| Script | Purpose | Requires Clone |
|--------|---------|---------------|
| `scripts/seed-infisical.py` | Populate Infisical from contract YAMLs | Yes |
| `scripts/bootstrap-infisical.sh` | Full first-time bootstrap sequence | Yes |
| `scripts/provision-infisical.py` | Create machine identities, write credentials back to `~/.omnibase/.env` | Yes |
| `scripts/setup-infisical-identity.sh` | Create runtime/admin machine identities | Yes |
| `scripts/create_kafka_topics.py` | Create Kafka/Redpanda topics | Yes |
| `scripts/validate.py` | Run ONEX validators | Yes |
| All other `scripts/*.py` | Operational, CI, or dev tooling | Yes |

**Why scripts require a clone:** These scripts scan the repository source tree directly
(e.g., `seed-infisical.py` iterates over `src/omnibase_infra/nodes/*/contract.yaml`),
write back to `~/.omnibase/.env`, or depend on shell tooling co-located with the repo.

### Decision Summary

| Use Case | Install Method |
|----------|---------------|
| Add `omnibase_infra` as a library dependency | `pip install omnibase-infra` |
| Run ONEX runtime services | `pip install omnibase-infra` → `onex-runtime` |
| Bootstrap Infisical (first-time setup) | Clone + `scripts/bootstrap-infisical.sh` |
| Seed Infisical from contracts | Clone + `uv run python scripts/seed-infisical.py` |
| Provision machine identities | Clone + `uv run python scripts/provision-infisical.py` |
| Run CI validators | Clone + `uv run python scripts/validate.py` |
| Develop nodes and handlers | Clone (full dev environment) |

> **Note on `sync-omnibase-env.py`**: This script is **not** part of `omnibase_infra`.
> It is provided by the `omniclaude` plugin and installed separately. See the
> [omniclaude README](https://github.com/OmniNode-ai/omniclaude) for details.

---

## Service Catalog Architecture

The service catalog is the authoritative source for all Docker infrastructure.
Every deployable unit is a typed YAML manifest; the compose file is generated, not hand-edited.

### Concepts

| Term | Description |
|------|-------------|
| **Manifest** | Typed YAML declaration of a single deployable service (`docker/catalog/services/<name>.yaml`) |
| **Bundle** | Named group of manifests deployed together (`docker/catalog/bundles.yaml`) |
| **Resolver** | Loads manifests + bundles, resolves transitive `includes`, returns `ResolvedStack` |
| **Generator** | Renders `ResolvedStack` → `docker-compose.generated.yml` |
| **Validator** | Checks that all `required_env` vars are present before start |

### Bundle Definitions

| Bundle | Contents | Purpose |
|--------|----------|---------|
| `core` | postgres, redpanda | Always-on infrastructure |
| `runtime` | valkey, migration-gate, forward-migration, omninode-runtime, runtime-effects, runtime-worker, agent-actions-consumer, skill-lifecycle-consumer, context-audit-consumer, intelligence-migration, intelligence-api, omninode-contract-resolver, autoheal + core | Full ONEX runtime stack |
| `memgraph` | omnibase-infra-memgraph | Graph memory — injects `OMNIMEMORY_*` env vars |
| `observability` | phoenix | LLM observability (Phoenix traces/evals) |
| `tracing` | (none) + observability | Injects OTEL env vars; phoenix pulled in transitively |
| `secrets` | infisical | Secrets management — injects `INFISICAL_ADDR` |
| `auth` | keycloak | Local OIDC/auth |

**Transitive resolution**: `runtime` includes `core`; `tracing` includes `observability`. The resolver expands all `includes` before collecting services.

**Env injection**: Each bundle may declare `inject_env` (hardcoded values injected into generated compose) and `inject_required_env` (vars that must be present in the operator environment at start time).

### onex CLI Commands

The `onex` CLI (`src/omnibase_infra/docker/catalog/cli.py`) is the primary operator interface.

```bash
# Generate compose file for one or more bundles
uv run python -m omnibase_infra.docker.catalog.cli generate core
uv run python -m omnibase_infra.docker.catalog.cli generate runtime memgraph

# Validate env completeness before starting
uv run python -m omnibase_infra.docker.catalog.cli validate runtime
uv run python -m omnibase_infra.docker.catalog.cli validate runtime memgraph

# Start a bundle (generate + validate + docker compose up)
uv run python -m omnibase_infra.docker.catalog.cli up core
uv run python -m omnibase_infra.docker.catalog.cli up runtime memgraph tracing

# Stop a running bundle
uv run python -m omnibase_infra.docker.catalog.cli down core
```

The shell functions `infra-up`, `infra-up-runtime`, `infra-up-memory`, and `infra-down` (defined in `~/.zshrc`) are backwards-compatible wrappers around `onex up/down`. They remain the preferred operator interface — do not bypass them with raw `docker compose -f <path>`.

### Shell Function → onex Mapping

| Shell Function | Equivalent onex Command |
|----------------|------------------------|
| `infra-up` | `onex up core` |
| `infra-up-runtime` | `onex up runtime` |
| `infra-up-memory` | `onex up runtime memgraph` |
| `infra-down` | `onex down <active-bundles>` |

### Adding a New Service

1. Create `docker/catalog/services/<name>.yaml` using an existing manifest as template.
2. Set `layer` to one of: `infrastructure`, `runtime`, `observability`, `auth`, `secrets`.
3. Declare all `required_env` vars that the container needs from the operator environment.
4. Add hardcoded container-internal addresses under `hardcoded_env` (never pass host-side env vars for internal addressing).
5. Add the service name to the appropriate bundle(s) in `docker/catalog/bundles.yaml`.
6. Run `uv run python -m omnibase_infra.docker.catalog.cli validate <bundle>` to confirm env contract.

### Env Var Contract

Three categories of env vars in the catalog:

| Category | Location | Behavior |
|----------|----------|----------|
| `required_env` | Per-manifest YAML | Must be set in operator env; validated before start |
| `hardcoded_env` | Per-manifest YAML | Container-internal addresses; never overrideable |
| `inject_env` | Per-bundle in `bundles.yaml` | Injected only when that bundle is selected |

**Rule**: Container-to-container addresses (e.g. `redpanda:9092`, `valkey:6379`) must live in `hardcoded_env`, never in `required_env`. Operator-supplied secrets (`POSTGRES_PASSWORD`, API keys) belong in `required_env`.

---

## Quick Reference

```bash
# Setup
uv sync && pre-commit install

# Testing
uv run pytest tests/                      # All tests
uv run pytest tests/ -n auto              # Parallel execution
uv run pytest tests/ -m unit              # Unit tests only
uv run pytest tests/ -m integration       # Integration tests only
uv run pytest tests/ --cov                # With coverage (60% minimum)

# Code Quality
uv run mypy src/omnibase_infra/           # Type checking
uv run ruff check src/ tests/             # Linting
pre-commit run --all-files                    # All hooks
```

## SPDX Headers

All source files in `src/`, `tests/`, `scripts/`, `examples/` require MIT SPDX headers.
Canonical spec: `omnibase_core/docs/conventions/FILE_HEADERS.md`

- Stamp missing headers: `onex spdx fix src tests scripts examples`
- Check without writing: `onex spdx fix --check src tests scripts examples`
- Bypass a file: add `# spdx-skip: <reason>` in the first 10 lines

---

### Git Commit Rules (repo-specific additions)

> `--no-verify` and hook rules: see `~/.claude/CLAUDE.md` Git Standards.

- **NEVER use `--no-gpg-sign`** unless explicitly requested
- **NEVER run git commits in background mode**

---

## Architecture: Four-Node Pattern

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   EFFECT    │───▶│   COMPUTE   │───▶│   REDUCER   │───▶│ORCHESTRATOR │
│ External I/O│    │  Transform  │    │  FSM State  │    │  Workflow   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Data Flow**: Unidirectional left-to-right. No backwards dependencies.

### Node Types

| Node | Contract Type | Purpose | Primary Output |
|------|--------------|---------|----------------|
| **EFFECT** | `EFFECT_GENERIC` | External I/O (APIs, DB, files) | `events[]` |
| **COMPUTE** | `COMPUTE_GENERIC` | Pure data transformation | `result` (required) |
| **REDUCER** | `REDUCER_GENERIC` | FSM state management | `projections[]` |
| **ORCHESTRATOR** | `ORCHESTRATOR_GENERIC` | Workflow coordination | `events[]`, `intents[]` |

### Import Path

```python
from omnibase_core.nodes import (
    NodeEffect,        # External I/O operations
    NodeCompute,       # Pure transformations
    NodeReducer,       # FSM-driven state
    NodeOrchestrator,  # Workflow coordination
)
```

### Layer Responsibilities

| Layer | Responsibility |
|-------|---------------|
| `omnibase_core` | Node archetypes, I/O models, enums |
| `omnibase_spi` | Protocol definitions |
| `omnibase_infra` | Infrastructure implementations |

---

## Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

```python
# CORRECT - Declarative node (extends base, no custom logic)
from omnibase_core.nodes import NodeOrchestrator
from omnibase_core.models.container.model_onex_container import ModelONEXContainer

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator - all behavior defined in contract.yaml."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
    # No custom code - driven entirely by contract
```

### Declarative Pattern Requirements

1. Extend base class from `omnibase_core.nodes`
2. Use `container: ModelONEXContainer` for dependency injection
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic

### Canonical Node Directory Structure

```text
nodes/<node_name>/
├── __init__.py           # Public exports
├── contract.yaml         # ONEX contract (REQUIRED)
├── node.py              # Declarative node class (REQUIRED)
├── models/              # Node-specific Pydantic models
│   ├── __init__.py
│   └── model_<name>.py
├── registry/            # Dependency injection registry
│   ├── __init__.py
│   └── registry_infra_<node_name>.py
├── handlers/            # Handler implementations (optional)
│   ├── __init__.py
│   └── handler_<name>.py
└── dispatchers/         # Dispatcher adapters (optional)
    ├── __init__.py
    └── dispatcher_<name>.py
```

### Contract Requirements

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Node identifier |
| `node_type` | string | Yes | `EFFECT_GENERIC`, `COMPUTE_GENERIC`, `REDUCER_GENERIC`, `ORCHESTRATOR_GENERIC` |
| `contract_version` | object | Yes | `{major, minor, patch}` |
| `node_version` | string/object | Yes | Semantic version |
| `description` | string | Yes | Node purpose |
| `input_model` | object | Yes | `{name, module, description}` |
| `output_model` | object | Yes | `{name, module, description}` |

---

## Handler System

### Handler Protocols

| Protocol | Purpose | Input/Output |
|----------|---------|--------------|
| `ProtocolHandler` | Envelope-based (runtime) | `ModelOnexEnvelope` → `ModelOnexEnvelope` |
| `ProtocolMessageHandler` | Category-based (dispatch) | `ModelEventEnvelope` → `ModelHandlerOutput` |

### Handler Routing Strategies

**`payload_type_match`** - Routes based on event payload model type (orchestrator handlers):
```yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelNodeIntrospectionEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected"
```

**`operation_match`** - Routes based on envelope operation (infrastructure handlers):
```yaml
handler_routing:
  routing_strategy: "operation_match"
  handlers:
    - operation: "register_node"
      handler:
        name: "HandlerConsulRegister"
        module: "omnibase_infra.nodes.node_registry_effect.handlers.handler_consul_register"
```

### Handler Classification

Handlers expose two classification properties:

```python
@property
def handler_type(self) -> EnumHandlerType:
    """Architectural role: INFRA_HANDLER, NODE_HANDLER, PROJECTION_HANDLER"""
    return EnumHandlerType.INFRA_HANDLER

@property
def handler_category(self) -> EnumHandlerTypeCategory:
    """Behavioral classification: EFFECT, COMPUTE, NONDETERMINISTIC_COMPUTE"""
    return EnumHandlerTypeCategory.EFFECT
```

### Handler No-Publish Constraint

**Handlers MUST NOT have direct event bus access** - only orchestrators may publish events.

| Constraint | Verification |
|------------|--------------|
| No bus parameters | `__init__`, `handle()` signatures |
| No bus attributes | No `_bus`, `_event_bus`, `_publisher` |
| No publish methods | No `publish()`, `emit()`, `send_event()` |

---

## Intent Model Architecture

**Overview**: Reducers emit intents that orchestrators route to Effect layer nodes. Payload models extend `BaseModel` directly (since omnibase_core 0.6.2).

### Two-Layer Intent Structure

| Layer | Model | Purpose |
|-------|-------|---------|
| 1. Typed Payload | `ModelPayloadConsulRegister` | Domain-specific Pydantic model with `intent_type` field |
| 2. Outer Container | `ModelIntent` | Standard intent envelope with `intent_type="extension"` |

### Defining Typed Payload Models

```python
# In nodes/reducers/models/model_payload_consul_register.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from uuid import UUID

class ModelPayloadConsulRegister(BaseModel):
    """Typed payload for Consul service registration.

    Note: Extends BaseModel directly (ModelIntentPayloadBase was removed in
    omnibase_core 0.6.2).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["consul.register"] = Field(default="consul.register")
    correlation_id: UUID
    service_id: str
    service_name: str
    tags: list[str]
    health_check: dict[str, str] | None = None
```

### Building Intents in Reducers

```python
from omnibase_core.models.reducer.model_intent import ModelIntent

# Build typed payload with domain data
consul_payload = ModelPayloadConsulRegister(
    correlation_id=correlation_id,
    service_id=f"onex-{node_type}-{node_id}",
    service_name=f"onex-{node_type}",
    tags=["node_type:effect"],
)

# Return as ModelIntent from reducer
return ModelIntent(
    intent_type="extension",
    target=f"consul://service/{service_name}",
    payload=consul_payload,
)
```

### Intent Type Routing

- `ModelIntent.intent_type` is always `"extension"` for infrastructure intents
- `payload.intent_type` contains the specific routing key (e.g., `"consul.register"`)
- Effect layer routes based on `payload.intent_type`

### Target URI Convention

Format: `{protocol}://{resource}/{identifier}`

Examples:
- `postgres://node_registrations/{node_id}`
- `consul://service/{service_name}`

---

## Error Handling

### Error Hierarchy

```text
ModelOnexError (omnibase_core)
└── RuntimeHostError (base infrastructure error)
    ├── ProtocolConfigurationError
    ├── SecretResolutionError
    ├── InfraConnectionError (transport-aware codes)
    │   ├── InfraConsulError
    │   └── InfraVaultError
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    ├── InfraRateLimitedError
    ├── InfraUnavailableError
    ├── EnvelopeValidationError
    ├── UnknownHandlerTypeError
    ├── ContainerWiringError
    │   ├── ServiceRegistrationError
    │   ├── ServiceResolutionError
    │   └── ContainerValidationError
    ├── ChainPropagationError
    ├── ArchitectureViolationError
    ├── BindingResolutionError
    ├── RepositoryError
    │   ├── RepositoryContractError
    │   ├── RepositoryValidationError
    │   ├── RepositoryExecutionError
    │   └── RepositoryTimeoutError
    └── ContractPublisherError
```

### Error Class Selection

| Scenario | Error Class |
|----------|-------------|
| Config invalid | `ProtocolConfigurationError` |
| Connection failed | `InfraConnectionError` |
| Timeout | `InfraTimeoutError` |
| Auth failed | `InfraAuthenticationError` |
| Rate limited | `InfraRateLimitedError` |
| Unavailable | `InfraUnavailableError` |
| Repository operation | `RepositoryError` (or subclass) |
| Container wiring | `ContainerWiringError` (or subclass) |

### Error Context Factory (MANDATORY)

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Auto-generate correlation_id (new error, no existing ID)
context = ModelInfraErrorContext.with_correlation(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
)

# Propagate existing correlation_id (preserve trace chain)
context = ModelInfraErrorContext.with_correlation(
    correlation_id=request.correlation_id,
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
)

raise InfraConnectionError("Failed to connect", context=context) from e
```

### Error Sanitization

**NEVER include**: passwords, API keys, PII, connection strings with credentials

**SAFE to include**: service names, operation names, correlation IDs, ports

Use utility functions from `omnibase_infra.utils.util_error_sanitization`:
- `sanitize_error_message()` - For DLQ/logs
- `sanitize_secret_path()` - For Vault paths
- `sanitize_consul_key()` - For Consul keys

---

## Infrastructure Patterns

### Transport Types

| Type | Value | Handler/Service |
|------|-------|-----------------|
| `HTTP` | `"http"` | `HandlerHTTP`, `ServiceHealth` |
| `DATABASE` | `"db"` | `HandlerDb`, `PostgresRepositoryRuntime` |
| `KAFKA` | `"kafka"` | `EventBusKafka`, `AdapterProtocolEventPublisherKafka` |
| `CONSUL` | `"consul"` | `HandlerConsul` |
| `VAULT` | `"vault"` | `HandlerVault` |
| `VALKEY` | `"valkey"` | (Planned) |
| `GRPC` | `"grpc"` | (Planned) |
| `RUNTIME` | `"runtime"` | `RuntimeHostProcess` |
| `MCP` | `"mcp"` | `HandlerMCP` |
| `FILESYSTEM` | `"filesystem"` | `HandlerFileSystem` |
| `INMEMORY` | `"inmemory"` | `EventBusInmemory` |
| `QDRANT` | `"qdrant"` | `HandlerQdrant` |
| `GRAPH` | `"graph"` | (Planned - Memgraph/Neo4j) |

### Circuit Breaker

Use `MixinAsyncCircuitBreaker` for external service integrations:

```python
class MyAdapter(MixinAsyncCircuitBreaker):
    def __init__(self, config):
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="my-service",
            transport_type=EnumInfraTransportType.HTTP,
            half_open_successes=1,
        )

    async def connect(self):
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("connect", correlation_id)
        # ... operation ...
```

**States**: CLOSED → OPEN (after threshold failures) → HALF_OPEN (after timeout) → CLOSED (on success)

### Dispatcher Resilience

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap dispatchers with circuit breakers.

Each dispatcher should:
- Implement `MixinAsyncCircuitBreaker` for external service calls
- Configure thresholds appropriate to their transport type
- Raise `InfraUnavailableError` when circuit opens

### Correlation ID Rules

1. Always propagate from incoming requests
2. Auto-generate with `uuid4()` if missing
3. Include in all error context

---

## Pydantic Model Standards

### File & Class Naming

| Type | File Pattern | Class Pattern |
|------|-------------|---------------|
| Model | `model_<name>.py` | `Model<Name>` |
| Adapter | `adapter_<name>.py` | `Adapter<Name>` |
| Dispatcher | `dispatcher_<name>.py` | `Dispatcher<Name>` |
| Enum | `enum_<name>.py` | `Enum<Name>` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` |
| Protocol | `protocol_<name>.py` | `Protocol<Name>` |
| Service | `service_<name>.py` | `Service<Name>` |
| Store | `store_<name>.py` | `Store<Purpose><Backend>` |
| Validator | `validator_<name>.py` | `Validator<Name>` |
| Registry (node) | `registry_infra_<name>.py` | `RegistryInfra<Name>` |
| Registry (standalone) | `registry_<purpose>.py` | `Registry<Purpose>` |

### ConfigDict Requirements

```python
# Standard pattern (most common)
model_config = ConfigDict(
    frozen=True,           # Immutability for thread safety
    extra="forbid",        # Strict validation
    from_attributes=True,  # ORM/pytest-xdist compatibility
)
```

### Field Patterns

```python
# Required field
field_name: FieldType = Field(..., description="Clear description")

# Optional field (prefer empty string over None for strings)
error_message: str = Field(default="", description="Empty if no error")

# Collections - use default_factory for mutable defaults
items: list[str] = Field(default_factory=list)

# Immutable collections - use tuple for frozen models
errors: tuple[ModelError, ...] = Field(default_factory=tuple)
```

### Custom `__bool__` for Result Models

Result models may override `__bool__` for idiomatic conditional checks:

```python
def __bool__(self) -> bool:
    """Allow using result in boolean context.

    Warning:
        **Non-standard __bool__ behavior**: Returns ``True`` only when
        ``is_valid`` is True. Differs from typical Pydantic behavior.
    """
    return self.is_valid
```

**Documentation requirement**: Always include a `Warning` section explaining non-standard behavior.

---

## Testing and CI

### Test Directory Structure

```text
tests/
├── conftest.py              # Root conftest with shared fixtures
├── helpers/                 # Test helper utilities
├── unit/                    # Auto-marked with `unit` marker
├── integration/             # Auto-marked with `integration` marker
├── chaos/                   # Auto-marked with `chaos` marker
├── replay/                  # Auto-marked with `replay` marker
├── performance/             # Auto-marked with `performance` marker
└── ci/                      # CI/CD specific tests
```

### Pytest Markers

| Marker | Description | Auto-applied |
|--------|-------------|--------------|
| `unit` | Unit tests in isolation | Yes |
| `integration` | Multi-component tests | Yes |
| `slow` | Tests >1s execution | No |
| `chaos` | Chaos engineering tests | Yes |
| `performance` | Performance/benchmark tests | Yes |
| `consul` | Tests requiring real Consul | No |
| `postgres` | Tests requiring PostgreSQL | No |
| `kafka` | Tests requiring Kafka | No |
| `serial` | Non-parallel tests | No |

### Running Tests

```bash
# All tests
uv run pytest tests/

# With coverage (60% minimum required)
uv run pytest tests/ --cov=omnibase_infra --cov-report=html

# By category
uv run pytest -m unit                    # Unit tests only
uv run pytest -m integration             # Integration tests only
uv run pytest -m "not slow"              # Exclude slow tests

# Parallel execution
uv run pytest tests/ -n auto

# Debug mode (no parallelism)
uv run pytest tests/ -n 0 -xvs
```

### Coverage Requirement

**Minimum 60% coverage required** (`fail_under = 60` in pyproject.toml)

### Common Fixtures

| Fixture | Purpose |
|---------|---------|
| `mock_container` | MagicMock ONEX container |
| `container_with_registries` | Real ModelONEXContainer with wired services |
| `event_bus` | In-memory event bus with cleanup |
| `cleanup_consul_test_services` | Cleans Consul test registrations |
| `cleanup_postgres_test_projections` | Cleans PostgreSQL test rows |

---

## Contract-Driven Config Discovery

Part of OMN-2287: Infisical-backed configuration management.

### Overview

The config discovery system extracts configuration requirements from ONEX
contract YAML files and resolves them from Infisical at runtime. It scans
three Pydantic-backed contract fields:

1. `metadata.transport_type` -- the transport type declared in metadata
2. `handler_routing.handlers[].handler_type` -- handler-level transport types
3. `dependencies[].type == "environment"` -- explicit env var dependencies

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TransportConfigMap` | `runtime/config_discovery/transport_config_map.py` | Maps transport types to Infisical paths |
| `ContractConfigExtractor` | `runtime/config_discovery/contract_config_extractor.py` | Scans contracts for config requirements |
| `ConfigPrefetcher` | `runtime/config_discovery/config_prefetcher.py` | Prefetches values through HandlerInfisical |
| `ModelTransportConfigSpec` | `runtime/config_discovery/models/model_transport_config_spec.py` | Spec for transport config in Infisical |
| `ModelConfigRequirements` | `runtime/config_discovery/models/model_config_requirements.py` | Aggregated requirements from contracts |

### Infisical Path Convention

```text
Shared:      /shared/<transport>/KEY
Per-service: /services/<service>/<transport>/KEY
```

### Bootstrap Sequence

```text
Step 1: PostgreSQL starts (POSTGRES_PASSWORD from .env)
Step 2: Valkey starts
Step 3: Infisical starts (depends_on: postgres + valkey healthy)
Step 4: Identity provisioning (first-time only)
Step 5: Seed runs (populates Infisical from contracts + .env values)
Step 6: Runtime services start (prefetch from Infisical)
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/bootstrap-infisical.sh` | Orchestrates the full bootstrap sequence |
| `scripts/seed-infisical.py` | Populates Infisical from contracts (safe by default, `--dry-run`) |
| `scripts/setup-infisical-identity.sh` | Creates machine identities (runtime=read-only, admin=read-write) |

### Opt-In Behavior

Config prefetch is **opt-in**: it only runs when `INFISICAL_ADDR` is set in the
environment. Without it, the runtime falls back to standard environment variable
resolution. This means local development works without Infisical.

### .env Reduction

The `.env.example` has been reduced from ~660 lines to ~30 lines (bootstrap-only).
The full pre-Infisical config is preserved in `docs/env-example-full.txt`.

---

## Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

**Prefer `subagent_type: "polymorphic-agent"`** for ONEX development workflows.

### Critical Policies

- **NEVER** use `run_in_background: true` for Task tool
- Parallel execution: call multiple Task tools in a **single message**

---

## Common Pitfalls

### Do NOT

1. **Skip base class initialization**
   ```python
   def __init__(self, container):
       pass  # WRONG - missing super().__init__(container)
   ```

2. **Add custom logic to declarative nodes**
   ```python
   class MyNode(NodeOrchestrator):
       def process(self, data):  # WRONG - nodes are declarative only
           return self._custom_logic(data)
   ```

3. **Return result from ORCHESTRATOR**
   ```python
   return ModelHandlerOutput.for_orchestrator(result={"status": "done"})  # ValueError!
   ```

4. **Use ModelIntentPayloadBase** (removed in omnibase_core 0.6.2)
   ```python
   from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase  # WRONG
   # Use: from pydantic import BaseModel
   ```

### DO

1. Always call `super().__init__(container)` in node constructors
2. Use `ModelONEXContainer` for dependency injection
3. Use protocol names for DI: `container.get_service("ProtocolEventBus")`
4. Keep nodes declarative - all logic in handlers
5. Use `ModelInfraErrorContext.with_correlation()` for error context

---

## Handler Plugin Loader

The runtime uses **plugin-based handler loading** from YAML contracts.

### Contract-Based Handler Declaration

```yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
      handler_module: "omnibase_infra.handlers.handler_node_introspected"
```

### Contract File Precedence

| Filename | Purpose |
|----------|---------|
| `handler_contract.yaml` | Dedicated handler contract (preferred) |
| `contract.yaml` | General ONEX contract with handler fields |

**FAIL-FAST**: When both files exist in the same directory, loader raises `AMBIGUOUS_CONTRACT_CONFIGURATION` error.

### Error Codes

| Code | Description |
|------|-------------|
| `HANDLER_LOADER_006` | `PROTOCOL_NOT_IMPLEMENTED` |
| `HANDLER_LOADER_010` | `MODULE_NOT_FOUND` |
| `HANDLER_LOADER_011` | `CLASS_NOT_FOUND` |
| `HANDLER_LOADER_012` | `IMPORT_ERROR` |
| `HANDLER_LOADER_013` | `NAMESPACE_NOT_ALLOWED` |
| `HANDLER_LOADER_040` | `AMBIGUOUS_CONTRACT_CONFIGURATION` |

### Security: Namespace Allowlisting

```python
# Restrict to trusted namespaces (recommended for production)
loader = HandlerPluginLoader(
    allowed_namespaces=["omnibase_infra.", "omnibase_core.", "myapp.handlers."]
)
```

---

## Release Process

### Version Compatibility Matrix (OMN-3203)

`src/omnibase_infra/runtime/version_compatibility.py` maintains a runtime
check that verified installed `omnibase_core` and `omnibase_spi` versions match
the constraints declared in `pyproject.toml`.

**How it works (after OMN-3203):**

`VERSION_MATRIX` is derived **automatically at import time** from `pyproject.toml`.
No manual update is required when bumping dependency versions — just update
`pyproject.toml` and the matrix follows.

A `_FALLBACK_MATRIX` with hardcoded values is used when `pyproject.toml` is
not present (e.g. installed package without source tree).  The fallback is kept
in sync with the `scripts/update_version_matrix.py` script.

**Release checklist for dependency bumps:**

1. Update `pyproject.toml` with new `>=X.Y.Z,<A.B.C` bounds.
2. Run `uv sync` to update `uv.lock`.
3. Run `uv run pytest tests/unit/runtime/test_version_compatibility.py` — the
   `test_matrix_matches_pyproject` test will catch any remaining drift.
4. The release workflow runs `scripts/update_version_matrix.py --check` as a
   pre-build gate; it also updates the fallback in-place if needed.

**Scripts:**

```bash
# Check that _FALLBACK_MATRIX matches pyproject.toml (CI mode — exits 1 on drift)
uv run python scripts/update_version_matrix.py --check

# Update _FALLBACK_MATRIX in-place
uv run python scripts/update_version_matrix.py
```

**What NOT to do:** Do not manually edit the `VERSION_MATRIX` or
`_FALLBACK_MATRIX` in `version_compatibility.py`.  Let `pyproject.toml` be the
single source of truth.

---

## Documentation

| Topic | Document |
|-------|----------|
| Any Type Enforcement | `docs/decisions/adr-any-type-pydantic-workaround.md` |
| Container DI | `docs/patterns/container_dependency_injection.md` |
| Error Handling | `docs/patterns/error_handling_patterns.md` |
| Error Recovery | `docs/patterns/error_recovery_patterns.md` |
| Circuit Breaker | `docs/patterns/circuit_breaker_implementation.md` |
| Dispatcher Resilience | `docs/patterns/dispatcher_resilience.md` |
| Protocol Patterns | `docs/patterns/protocol_patterns.md` |
| Security Patterns | `docs/patterns/security_patterns.md` |
| Handler Plugin Loader | `docs/patterns/handler_plugin_loader.md` |
| Mixin Dependencies | `docs/patterns/mixin_dependencies.md` |

---

**Python**: 3.12+ | **Ready?** → Check `docs/patterns/` for implementation guides

**Bottom Line**: Declarative nodes, container injection, agent-driven development. No backwards compatibility, no custom node logic.
