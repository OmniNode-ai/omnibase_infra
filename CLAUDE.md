# CLAUDE.md - Omnibase Infrastructure

> **Python**: 3.12+ | **Framework**: ONEX Infrastructure | **Shared Infrastructure**: See **`~/.claude/CLAUDE.md`** for PostgreSQL, Kafka/Redpanda, Docker networking, and environment variables.

---

## Table of Contents

1. [Repo Invariants](#repo-invariants)
2. [Non-Goals](#non-goals)
3. [Quick Reference](#quick-reference)
4. [Architecture: Four-Node Pattern](#architecture-four-node-pattern)
5. [Declarative Nodes](#declarative-nodes)
6. [Handler System](#handler-system)
7. [Intent Model Architecture](#intent-model-architecture)
8. [Error Handling](#error-handling)
9. [Infrastructure Patterns](#infrastructure-patterns)
10. [Pydantic Model Standards](#pydantic-model-standards)
11. [Testing and CI](#testing-and-ci)
12. [Agent-Driven Development](#agent-driven-development)
13. [Common Pitfalls](#common-pitfalls)

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

## Quick Reference

```bash
# Setup
poetry install && pre-commit install

# Testing
poetry run pytest tests/                      # All tests
poetry run pytest tests/ -n auto              # Parallel execution
poetry run pytest tests/ -m unit              # Unit tests only
poetry run pytest tests/ -m integration       # Integration tests only
poetry run pytest tests/ --cov                # With coverage (60% minimum)

# Code Quality
poetry run mypy src/omnibase_infra/           # Type checking
poetry run ruff check src/ tests/             # Linting
pre-commit run --all-files                    # All hooks
```

### Git Commit Rules

- **NEVER use `--no-verify`** when committing
- **NEVER use `--no-gpg-sign`** unless explicitly requested
- **NEVER skip hooks** - fix issues instead of bypassing
- **NEVER run git commits in background mode**

---

## Architecture: Four-Node Pattern

```
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

```
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
from pydantic import BaseModel, Field
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

```
ModelOnexError (omnibase_core)
└── RuntimeHostError (base infrastructure error)
    ├── ProtocolConfigurationError
    ├── SecretResolutionError
    ├── InfraConnectionError (transport-aware codes)
    │   ├── InfraConsulError
    │   └── InfraVaultError
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
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
| Registry | `registry_infra_<name>.py` | `RegistryInfra<Name>` |

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

### Type Annotation Style (PEP 604)

```python
# CORRECT
value: UUID | None = Field(default=None)
result: str | int = Field(...)

# WRONG - Do not use Optional
value: Optional[UUID] = Field(default=None)  # Forbidden
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

```
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
poetry run pytest tests/

# With coverage (60% minimum required)
poetry run pytest tests/ --cov=omnibase_infra --cov-report=html

# By category
poetry run pytest -m unit                    # Unit tests only
poetry run pytest -m integration             # Integration tests only
poetry run pytest -m "not slow"              # Exclude slow tests

# Parallel execution
poetry run pytest tests/ -n auto

# Debug mode (no parallelism)
poetry run pytest tests/ -n 0 -xvs
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
- Always use `poetry run` for Python commands
- Never allow direct pip or python execution

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

4. **Use pip instead of Poetry**
   ```bash
   pip install package  # WRONG - use poetry add
   ```

5. **Add backwards-compatibility hacks**
   ```python
   old_name = new_name  # WRONG - no re-exports for "compatibility"
   ```

6. **Use ModelIntentPayloadBase** (removed in omnibase_core 0.6.2)
   ```python
   from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase  # WRONG
   # Use: from pydantic import BaseModel
   ```

### DO

1. Always call `super().__init__(container)` in node constructors
2. Use `ModelONEXContainer` for dependency injection
3. Use protocol names for DI: `container.get_service("ProtocolEventBus")`
4. Use `poetry run` for all Python commands
5. Keep nodes declarative - all logic in handlers
6. Use `ModelInfraErrorContext.with_correlation()` for error context

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

## Documentation

| Topic | Document |
|-------|----------|
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
