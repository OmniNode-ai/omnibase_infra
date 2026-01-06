# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `circuit_breaker_implementation.md` - Circuit breaker details
- `dispatcher_resilience.md` - Dispatcher-owned resilience pattern
- `security_patterns.md` - Node introspection security, input validation, secrets

---

## MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

**Prefer `subagent_type: "polymorphic-agent"`** for ONEX development workflows.

## CRITICAL POLICIES

### No Background Agents
- **NEVER** use `run_in_background: true` for Task tool
- Parallel execution: call multiple Task tools in a **single message**

### No Backwards Compatibility
- Breaking changes are always acceptable
- Remove old patterns immediately

### No Versioned Directories
- **NEVER** create `v1_0_0/`, `v2/` directories
- Version through `contract.yaml` fields only

## MANDATORY: Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

```python
# CORRECT - Declarative node (extends base, no custom logic)
from omnibase_core.nodes import NodeOrchestrator

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator - all behavior defined in contract.yaml."""
    pass  # No custom code - driven entirely by contract
```

**Declarative Pattern Requirements:**
1. Extend base class from `omnibase_core.nodes`
2. Use `container: ModelONEXContainer` for dependency injection
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic

**Contract-Driven Handler Routing:**
```yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
```

## Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Use `object` for generic payloads
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class
- **PEP 604 unions** - Use `X | None` not `Optional[X]`

### File & Class Naming

| Type | File Pattern | Class Pattern |
|------|-------------|---------------|
| Model | `model_<name>.py` | `Model<Name>` |
| Enum | `enum_<name>.py` | `Enum<Name>` |
| Protocol | `protocol_<name>.py` or `protocols.py` | `Protocol<Name>` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` |
| Service | `service_<name>.py` | `Service<Name>` |
| Util | `util_<name>.py` | (functions) |
| Error | In `errors/` | `<Domain><Type>Error` |
| Node | `node.py` | `Node<Name><Type>` |

**Protocol File Naming**:
- Single protocol: `protocol_<name>.py`
- Domain-grouped: `protocols.py` when protocols are tightly coupled

### Enum Usage: Message Routing vs Node Validation

| Enum | Values | Purpose |
|------|--------|---------|
| `EnumMessageCategory` | `EVENT`, `COMMAND`, `INTENT` | Message routing |
| `EnumNodeOutputType` | `EVENT`, `COMMAND`, `INTENT`, `PROJECTION` | Node validation |

**Key Rule**: `PROJECTION` exists only in `EnumNodeOutputType` and is only valid for REDUCER nodes.

```python
# MESSAGE ROUTING
from omnibase_infra.enums import EnumMessageCategory
category = EnumMessageCategory.EVENT  # For dispatcher selection

# NODE VALIDATION
from omnibase_infra.enums import EnumNodeOutputType
output_type.is_routable()  # False for PROJECTION
```

**Related**: `docs/decisions/adr-enum-message-category-vs-node-output-type.md`

### Registry Naming Conventions

**Node-Specific**: `nodes/<name>/registry/registry_infra_<node_name>.py` → `RegistryInfra<NodeName>`

**Standalone**: `registry_<purpose>.py` → `Registry<Purpose>`

### Custom `__bool__` for Result Models

Result models may override `__bool__` to enable idiomatic conditional checks. This differs from standard Pydantic behavior where `bool(model)` always returns `True`.

**Current implementations**:
- `ModelReducerExecutionResult`: Returns `True` only if `has_intents` (intents tuple is non-empty)
- `ModelCategoryMatchResult`: Returns `True` only if `matched` is True

**Usage pattern**:
```python
result = reducer.reduce(state, event)
if result:  # True only if there are intents to process
    execute_intents(result.intents)
```

**Documentation requirement**: Always include a `Warning` section in the `__bool__` docstring explaining the non-standard behavior.

### Type Annotation Conventions

**Nullable Types**: Use `X | None` (PEP 604), not `Optional[X]`
```python
def get_user(id: str) -> User | None: ...
```

**Envelope Typing**: Use `ModelEventEnvelope[object]` for generic dispatchers
```python
async def process_event(envelope: ModelEventEnvelope[object]) -> str | None:
    """Process any event type - uses object for generic payloads."""
```

**Type Alias Pattern**: Use underscore-prefixed unions for Pydantic, protocols for type hints
```python
_IntentUnion = ModelCommandIntent | ModelEventIntent  # Pydantic validation
def process(intent: ProtocolRegistrationIntent): ...  # Function signature
```

### Intent Model Architecture

**Overview**: Reducers emit intents that orchestrators route to Effect layer nodes. The implementation uses typed payload models that extend `ModelIntentPayloadBase`.

**Two-Layer Intent Structure**:

| Layer | Model | Purpose |
|-------|-------|---------|
| 1. Typed Payload | `ModelPayloadConsulRegister` | Domain-specific Pydantic model with typed fields and `intent_type` |
| 2. Outer Container | `ModelIntent` | Standard intent envelope with `intent_type="extension"` |

**Defining Typed Payload Models** (in `nodes/reducers/models/`):
```python
from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase
from pydantic import Field
from typing import Literal
from uuid import UUID

class ModelPayloadConsulRegister(ModelIntentPayloadBase):
    """Typed payload for Consul service registration."""
    intent_type: Literal["consul.register"] = Field(default="consul.register")
    correlation_id: UUID
    service_id: str
    service_name: str
    tags: list[str]
    health_check: dict[str, str] | None = None
```

**Building Intents in Reducers**:
```python
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.nodes.reducers.models import ModelPayloadConsulRegister

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

**Intent Type Routing**:
- `ModelIntent.intent_type` is always `"extension"` for extension-based intents
- `payload.intent_type` contains the specific routing key (e.g., `"consul.register"`, `"postgres.upsert_registration"`)
- Effect layer routes based on `payload.intent_type`

**Target URI Convention**:
- Format: `{protocol}://{resource}/{identifier}`
- Examples: `postgres://node_registrations/{node_id}`, `consul://service/{service_name}`

**Serialization with Nested Models**:

When payload contains complex nested models (e.g., `ModelNodeRegistrationRecord`), use `SerializeAsAny`:
```python
from pydantic import BaseModel, Field, SerializeAsAny

class ModelPayloadPostgresUpsertRegistration(ModelIntentPayloadBase):
    """Typed payload for PostgreSQL upsert operations."""
    intent_type: Literal["postgres.upsert_registration"] = Field(
        default="postgres.upsert_registration"
    )
    correlation_id: UUID
    record: SerializeAsAny[BaseModel]  # Preserves subclass fields during serialization
```

**Why `SerializeAsAny`**: When a field is typed as `BaseModel` but contains a subclass, Pydantic only serializes base fields without this type wrapper.

**Accessing Payload Fields**:
```python
# Direct typed field access (no .data dict needed)
if isinstance(intent.payload, ModelPayloadConsulRegister):
    service_name = intent.payload.service_name
    tags = intent.payload.tags
```

**Effect Layer Routing**:

Effect nodes use `payload.intent_type` to route intents to appropriate handlers:
```python
from omnibase_infra.nodes.reducers.models import (
    ModelPayloadConsulRegister,
    ModelPayloadPostgresUpsertRegistration,
)

routing_table = {
    "consul.register": ConsulAdapter.handle_register,
    "postgres.upsert_registration": PostgresAdapter.handle_upsert,
}

async def route_intent(intent: ModelIntent) -> None:
    if intent.intent_type == "extension" and hasattr(intent.payload, "intent_type"):
        handler = routing_table.get(intent.payload.intent_type)
        if handler:
            await handler(intent.payload)
        else:
            raise ValueError(f"Unknown intent_type: {intent.payload.intent_type}")
```

**Reference Implementation**: `src/omnibase_infra/nodes/reducers/registration_reducer.py`
**Payload Models**: `src/omnibase_infra/nodes/reducers/models/model_payload_*.py`

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - `def __init__(self, container: ModelONEXContainer)`
- **Protocol Resolution** - Duck typing through protocols, never isinstance
- **OnexError Only** - `raise OnexError(...) from e`

### Node Archetypes (from `omnibase_core`)

| Layer | Responsibility |
|-------|---------------|
| `omnibase_core` | Node archetypes, I/O models, enums |
| `omnibase_spi` | Protocol definitions |
| `omnibase_infra` | Infrastructure implementations |

```python
from omnibase_core.nodes import (
    NodeEffect,        # External I/O operations
    NodeCompute,       # Pure transformations
    NodeReducer,       # State aggregation (FSM-driven)
    NodeOrchestrator,  # Workflow coordination
)
```

### Container-Based Dependency Injection

**All services MUST use ModelONEXContainer for dependency injection.**

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes import NodeOrchestrator

class MyOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
```

## Infrastructure Error Patterns

### Error Class Selection

| Scenario | Error Class |
|----------|-------------|
| Config invalid | `ProtocolConfigurationError` |
| Connection failed | `InfraConnectionError` |
| Timeout | `InfraTimeoutError` |
| Auth failed | `InfraAuthenticationError` |
| Unavailable | `InfraUnavailableError` |

### Error Context
```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect", context=context) from e
```

### Error Hierarchy
```
ModelOnexError (omnibase_core)
└── RuntimeHostError
    ├── ProtocolConfigurationError
    ├── InfraConnectionError (transport-aware codes)
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    └── InfraUnavailableError
```

### Error Sanitization
**NEVER include**: passwords, API keys, PII, connection strings with credentials
**SAFE to include**: service names, operation names, correlation IDs, ports

## Infrastructure Patterns

### Correlation ID Rules
1. Always propagate from incoming requests
2. Auto-generate with `uuid4()` if missing
3. Include in all error context

### Circuit Breaker

Use `MixinAsyncCircuitBreaker` for external service integrations:
```python
class MyAdapter(MixinAsyncCircuitBreaker):
    def __init__(self, config):
        self._init_circuit_breaker(
            threshold=5, reset_timeout=60.0,
            service_name="my-service",
            transport_type=EnumInfraTransportType.HTTP,
        )

    async def connect(self):
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("connect", correlation_id)
        # ... operation ...
```

**States**: CLOSED → OPEN (after failures) → HALF_OPEN → CLOSED (on success)

**See**: `docs/patterns/circuit_breaker_implementation.md` for full implementation details

### Transport Types
| Type | Value |
|------|-------|
| `DATABASE` | `"db"` |
| `KAFKA` | `"kafka"` |
| `HTTP` | `"http"` |
| `CONSUL` | `"consul"` |
| `VAULT` | `"vault"` |
| `VALKEY` | `"valkey"` |

### Dispatcher Resilience

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap dispatchers with circuit breakers.

Each dispatcher should:
- Implement `MixinAsyncCircuitBreaker` for external service calls
- Configure thresholds appropriate to their transport type
- Raise `InfraUnavailableError` when circuit opens

**See**: `docs/patterns/dispatcher_resilience.md` for full pattern details

### Handler No-Publish Constraint

**Handlers MUST NOT have direct event bus access** - only orchestrators may publish events.

| Constraint | Verification |
|------------|--------------|
| No bus parameters | `__init__`, `handle()` signatures |
| No bus attributes | No `_bus`, `_event_bus`, `_publisher` |
| No publish methods | No `publish()`, `emit()`, `send_event()` |
| Protocol compliance | `ProtocolHandler` has no bus methods |

**Integration Tests**: `tests/integration/handlers/test_handler_no_publish_constraint.py`

| Test Class | Coverage |
|------------|----------|
| `TestHttpRestHandlerBusIsolation` | HTTP handler bus isolation |
| `TestHandlerNodeIntrospectedBusIsolation` | Introspection handler isolation |
| `TestHandlerProtocolCompliance` | Protocol-level constraints |
| `TestOrchestratorBusAccessVerification` | Orchestrator-only bus access |
| `TestHandlerNoPublishConstraintCrossValidation` | Cross-handler constraint validation |

### Node Introspection Security

The `MixinNodeIntrospection` mixin uses Python reflection for service discovery. This has security implications:

**What Gets Exposed**: Public method names, signatures, protocol implementations, FSM state
**What is NOT Exposed**: Private methods (`_` prefix), source code, configuration values, secrets

**Built-in Protections**:
- Private method exclusion
- Utility method filtering (`get_*`, `set_*`, etc.)
- Operation keyword matching

**Best Practices**:
- Prefix internal/sensitive methods with `_`
- Use generic parameter names (e.g., `data` not `user_credentials`)
- Configure Kafka topic ACLs for introspection topics

**See**: `docs/patterns/security_patterns.md#introspection-security` for complete threat model and deployment checklist

### Service Integration Architecture
- **Adapter Pattern** - External services wrapped in ONEX adapters
- **Connection Pooling** - Database connections managed through pool managers
- **Event-Driven Communication** - Infrastructure events flow through Kafka adapters
- **Service Discovery** - Consul integration for dynamic service resolution
- **Secret Management** - Vault integration for secure credential handling

### Infrastructure 4-Node Pattern
- **EFFECT** - External service interactions (adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

## Node Structure

**Canonical Structure:**
```
nodes/<adapter>/
├── contract.yaml     # ONEX contract (handlers, routing, version)
├── node.py          # Declarative node extending base class
├── models/          # Node-specific models
└── registry/        # registry_infra_<name>.py
```

**Contract Requirements:**
- Semantic versioning (`contract_version`, `node_version`)
- Node type (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR)
- Strongly typed I/O (`input_model`, `output_model`)
- Handler routing (for orchestrators)
- Zero `Any` types

## Agent Architecture

| Category | Agents |
|----------|--------|
| Orchestration | `agent-onex-coordinator`, `agent-workflow-coordinator` |
| Development | `agent-contract-validator`, `agent-commit`, `agent-testing` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit` |
| Quality | `agent-pr-review`, `agent-pr-create` |

## Zero Tolerance

- `Any` types forbidden
- Direct coding without agent delegation
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations
- Imperative nodes with custom routing logic

## Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Consul | 8500 |
| Kafka | 9092 |
| Vault | 8200 |
| PostgreSQL | 5432 |

---

**Bottom Line**: Declarative nodes, container injection, agent-driven development. No backwards compatibility, no custom node logic.
