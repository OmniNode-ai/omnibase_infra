# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `circuit_breaker_implementation.md` - Circuit breaker details

---

## 🚨 MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

**Prefer `subagent_type: "polymorphic-agent"`** for ONEX development workflows.

## 🚫 CRITICAL POLICIES

### No Background Agents
- **NEVER** use `run_in_background: true` for Task tool
- Parallel execution: call multiple Task tools in a **single message**

### No Backwards Compatibility
- Breaking changes are always acceptable
- Remove old patterns immediately

### No Versioned Directories
- **NEVER** create `v1_0_0/`, `v2/` directories
- Version through `contract.yaml` fields only

## 🎯 MANDATORY: Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

```python
# CORRECT - Declarative node (extends base, no custom logic)
from omnibase_core.nodes import NodeOrchestrator

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator - all behavior defined in contract.yaml."""
    pass  # No custom code - driven entirely by contract
```

```python
# WRONG - Imperative node with custom logic
class NodeRegistrationOrchestrator:
    def __init__(self, projection_reader):  # Direct injection
        self._handlers = {...}  # Manual handler wiring

    async def handle(self, envelope):
        if isinstance(payload, EventA):  # Manual routing
            return self._handler_a.handle(...)
```

**Declarative Pattern Requirements:**
1. Extend base class from `omnibase_core.nodes` (`NodeEffect`, `NodeCompute`, `NodeReducer`, `NodeOrchestrator`)
2. Use `container: ModelONEXContainer` for dependency injection
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic
5. Base class handles routing, wiring, and execution from contract

**Contract-Driven Handler Routing:**
```yaml
# contract.yaml - handler_routing section
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
    - event_model: "ModelRuntimeTick"
      handler_class: "HandlerRuntimeTick"
```

## 🎯 Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Use `object` for generic payloads
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class
- **PEP 604 unions** - Use `X | None` not `Optional[X]`

### File & Class Naming

| Type | File Pattern | Class Pattern | Example |
|------|-------------|---------------|---------|
| Model | `model_<name>.py` | `Model<Name>` | `model_kafka_message.py` → `ModelKafkaMessage` |
| Enum | `enum_<name>.py` | `Enum<Name>` | `enum_handler_type.py` → `EnumHandlerType` |
| Protocol | `protocol_<name>.py` or `protocols.py` | `Protocol<Name>` | See note below |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` | `mixin_health_check.py` → `MixinHealthCheck` |
| Service | `service_<name>.py` | `Service<Name>` | `service_discovery.py` → `ServiceDiscovery` |
| Util | `util_<name>.py` | (functions) | `util_retry.py` → `retry_with_backoff()` |
| Error | In `errors/` | `<Domain><Type>Error` | `InfraConnectionError` |
| Node | `node.py` | `Node<Name><Type>` | `NodePostgresAdapterEffect` |

**Protocol File Naming**:
- **Single protocol**: Use `protocol_<name>.py` for standalone protocols (e.g., `protocol_event_bus.py` contains `ProtocolEventBus`)
- **Domain-grouped protocols**: Use `protocols.py` when multiple cohesive protocols belong to a specific domain or node module (e.g., `nodes/<name>/protocols.py` containing `ProtocolNodeInput`, `ProtocolNodeOutput`, `ProtocolNodeConfig`)

Domain grouping is preferred when:
- Protocols are tightly coupled and always used together
- Protocols define the complete interface for a single node or module
- Protocols share common type dependencies within the same bounded context

### Enum Usage: Message Routing vs Node Validation

ONEX uses two distinct enums for message categorization with different purposes:

| Enum | Values | Purpose | Location |
|------|--------|---------|----------|
| `EnumMessageCategory` | `EVENT`, `COMMAND`, `INTENT` | Message routing, topic parsing, dispatcher selection | `omnibase_infra.enums` |
| `EnumNodeOutputType` | `EVENT`, `COMMAND`, `INTENT`, `PROJECTION` | Execution shape validation, handler return type validation | `omnibase_infra.enums` |

**Quick Decision Guide**:
- **Routing a message?** Use `EnumMessageCategory`
- **Validating node output?** Use `EnumNodeOutputType`

**Key Difference - PROJECTION**:
- `PROJECTION` exists **only** in `EnumNodeOutputType`
- `PROJECTION` is **only valid for REDUCER nodes** (state aggregation outputs)
- Message routing never uses `PROJECTION` because projections are not routable messages

**Usage Examples**:

```python
from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType

# MESSAGE ROUTING - Use EnumMessageCategory
def parse_topic(topic: str) -> EnumMessageCategory:
    """Parse topic to determine message category for routing."""
    if ".event." in topic:
        return EnumMessageCategory.EVENT
    elif ".command." in topic:
        return EnumMessageCategory.COMMAND
    return EnumMessageCategory.INTENT

def select_dispatcher(category: EnumMessageCategory) -> ProtocolMessageDispatcher:
    """Select dispatcher based on message category."""
    return dispatcher_registry[category]

# NODE VALIDATION - Use EnumNodeOutputType
def validate_reducer_output(node_type: str, output_type: EnumNodeOutputType) -> bool:
    """Validate that output type is valid for node type."""
    if output_type == EnumNodeOutputType.PROJECTION:
        # PROJECTION only valid for REDUCER nodes
        return node_type == "REDUCER"
    return True

def get_handler_output_type(handler: ProtocolHandler) -> EnumNodeOutputType:
    """Get the declared output type for handler validation."""
    return handler.output_type  # May include PROJECTION for reducers
```

**Mapping Between Enums**:

`EnumNodeOutputType` provides helper methods for safe conversion:

```python
from omnibase_infra.enums import EnumNodeOutputType, EnumMessageCategory

# Convert node output type to message category (for routing after validation)
output_type = EnumNodeOutputType.EVENT
category = output_type.to_message_category()  # Returns EnumMessageCategory.EVENT

# PROJECTION cannot be converted - raises ValueError
projection = EnumNodeOutputType.PROJECTION
projection.to_message_category()  # Raises ValueError: PROJECTION has no message category

# Check if output type is routable
output_type.is_routable()  # True for EVENT, COMMAND, INTENT; False for PROJECTION
```

**Related**:
- ADR: `docs/decisions/adr-enum-message-category-vs-node-output-type.md`
- Ticket: OMN-974

### Registry Naming Conventions

**Node-Specific Registries** (`nodes/<name>/registry/`):
- File: `registry_infra_<node_name>.py`
- Class: `RegistryInfra<NodeName>`
- Examples: `registry_infra_postgres_adapter.py` → `RegistryInfraPostgresAdapter`
- Note: Versioned paths like `nodes/<name>/v1_0_0/registry/` are prohibited (see `docs/architecture/LEGACY_V1_MIGRATION.md`)

**Standalone Registries** (in domain directories):
- File: `registry_<purpose>.py`
- Class: `Registry<Purpose>`
- Examples: `registry_handler.py` → `RegistryHandler`, `registry_policy.py` → `RegistryPolicy`, `registry_compute.py` → `RegistryCompute`

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

**Nullable Types: Use `X | None` (PEP 604) over `Optional[X]`**

ONEX prefers the modern PEP 604 union syntax for nullable types. This is cleaner, more explicit, and aligns with Python 3.10+ best practices.

```python
# PREFERRED - PEP 604 union syntax
def get_user(id: str) -> User | None:
    """Return user or None if not found."""
    ...

def process_data(value: str | None = None) -> Result:
    """Process data with optional value."""
    ...

# Complex unions - also use pipe syntax
def parse_input(data: str | int | None) -> ParsedResult:
    """Parse string, int, or None input."""
    ...

# NOT PREFERRED - Optional syntax
from typing import Optional, Union

def get_user(id: str) -> Optional[User]:  # Avoid this
    ...

def parse_input(data: Optional[Union[str, int]]) -> ParsedResult:  # Avoid this
    ...
```

**Rationale**:
- `X | None` is visually clearer and more explicit about what the type represents
- Reduces import clutter (no need for `from typing import Optional`)
- Consistent with modern Python type annotation patterns
- `Optional[X]` can be misleading - it suggests the parameter is optional, not that it can be None

**Exception**: When maintaining compatibility with older codebases or when `typing.Optional` is already imported for other purposes, using `Optional` is acceptable but not preferred.

**Envelope Typing: Use `ModelEventEnvelope[object]` for Generic Dispatchers**

The dispatch engine and protocol dispatchers use `ModelEventEnvelope[object]` instead of `Any` for envelope parameters. This pattern satisfies the ONEX "no Any types" rule while maintaining the necessary flexibility for generic message handling.

```python
# CORRECT - Generic dispatcher (accepts any payload type)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

async def process_event(envelope: ModelEventEnvelope[object]) -> str | None:
    """Process any event type - uses object for generic payloads."""
    return "dev.processed.v1"

# CORRECT - Specific dispatcher (knows the exact payload type)
from my_models import UserCreatedEvent

async def process_user_created(envelope: ModelEventEnvelope[UserCreatedEvent]) -> str:
    """Process UserCreatedEvent - uses specific type when known."""
    user = envelope.payload  # Type-safe: UserCreatedEvent
    return f"dev.user.{user.user_id}.processed"

# WRONG - Never use Any
from typing import Any

async def process_event(envelope: ModelEventEnvelope[Any]) -> str:  # Avoid
    ...
```

**When to use each pattern**:

| Context | Type Parameter | Rationale |
|---------|----------------|-----------|
| Protocol definitions | `object` | Protocols define structural interfaces; payload type varies |
| Dispatch engine internals | `object` | Routes based on topic/category, not payload shape |
| Generic dispatcher functions | `object` | Must accept any payload type within a category |
| Concrete implementations | Specific type | When dispatcher knows exact payload type (e.g., `UserCreatedEvent`) |
| Test fixtures | Specific type | Tests create envelopes with known payload types |

**Rationale for `object` over `Any`**:
- `object` explicitly says "any Python object" - clearer intent than `Any`
- Type checkers treat `object` as the root of the type hierarchy
- Satisfies ONEX "no Any types" coding guideline
- Same runtime behavior as `Any` but with documented semantics
- Encourages narrowing to specific types where possible

**See also**: `src/omnibase_infra/runtime/message_dispatch_engine.py` for the complete design note.

**Type Alias vs Protocol Pattern**

ONEX uses a specific pattern for protocol-based models where Pydantic validation and type hints serve different purposes:

| Context | Pattern | Example |
|---------|---------|---------|
| Pydantic field validators | Underscore-prefixed union | `_IntentUnion = ModelCommandIntent \| ModelEventIntent` |
| Function type hints | Protocol interface | `def register(intent: ProtocolRegistrationIntent)` |

**Why this pattern exists**:
- **Pydantic validation**: Requires concrete union types to instantiate and validate model fields at runtime
- **Type hints**: Use protocols for duck typing and interface definitions (structural subtyping)
- **Internal vs External**: Underscore prefix signals internal implementation detail not part of public API

**Example**:
```python
from pydantic import BaseModel

from omnibase_core.protocols import ProtocolRegistrationIntent
from omnibase_infra.models import ModelCommandIntent, ModelEventIntent, ModelQueryIntent

# Internal: Concrete union for Pydantic validation
# Underscore prefix indicates this is an implementation detail
_IntentUnion = ModelCommandIntent | ModelEventIntent | ModelQueryIntent


class RegistrationRequest(BaseModel):
    """Request model with intent field requiring Pydantic validation."""

    # Pydantic needs concrete types to validate and instantiate
    intent: _IntentUnion


# External: Protocol for function signatures
def process_registration(intent: ProtocolRegistrationIntent) -> ModelRegistrationResult:
    """Process any registration intent.

    Protocol enables duck typing - any object implementing the protocol
    interface is accepted, not just the concrete union types.
    """
    ...
```

**When to use each**:

| Scenario | Use | Rationale |
|----------|-----|-----------|
| Pydantic model field | `_IntentUnion` (concrete union) | Pydantic must instantiate concrete types for validation |
| Function parameter type hint | `ProtocolRegistrationIntent` | Enables duck typing, accepts any conforming object |
| Return type annotation | Protocol or concrete type | Depends on whether caller needs specific type or interface |
| Internal helper functions | Either | Context-dependent; prefer protocol for flexibility |

**Naming convention**:
- Underscore-prefixed unions: `_<Name>Union` (e.g., `_IntentUnion`, `_HandlerUnion`)
- Protocols: `Protocol<Name>` (e.g., `ProtocolRegistrationIntent`, `ProtocolHandler`)

**Related**: OMN-1007 (Union Reduction)

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - `def __init__(self, container: ModelONEXContainer)`
- **Protocol Resolution** - Duck typing through protocols, never isinstance
- **OnexError Only** - `raise OnexError(...) from e`

### Node Archetypes & Core Models (from `omnibase_core`)

**Architecture Rule**: `omnibase_infra` extends base archetypes from `omnibase_core`. Never define new node archetypes in infra - they belong in core. This ensures consistent node contracts across the ONEX ecosystem.

| Layer | Responsibility | Example |
|-------|---------------|---------|
| `omnibase_core` | Node archetypes, I/O models, enums | `NodeReducer`, `ModelReducerInput` |
| `omnibase_spi` | Protocol definitions | `ProtocolReducerNode` |
| `omnibase_infra` | Infrastructure implementations | `NodeDualRegistrationReducer` |

**All node base classes and their I/O models come from `omnibase_core.nodes`:**

```python
from omnibase_core.nodes import (
    # Node base classes (archetypes)
    NodeEffect,           # External I/O operations
    NodeCompute,          # Pure transformations
    NodeReducer,          # State aggregation (FSM-driven)
    NodeOrchestrator,     # Workflow coordination

    # I/O models for each archetype
    ModelEffectInput, ModelEffectOutput, ModelEffectTransaction,
    ModelComputeInput, ModelComputeOutput,
    ModelReducerInput, ModelReducerOutput,
    ModelOrchestratorInput, ModelOrchestratorOutput,

    # Enums for node behavior
    EnumReductionType,      # sum, count, avg, min, max, custom
    EnumConflictResolution, # last_write_wins, merge, error
    EnumStreamingMode,      # batch, streaming, hybrid
    EnumExecutionMode,      # sequential, parallel, conditional
    EnumWorkflowState,      # pending, running, completed, failed
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
        # Dependencies resolved from container, not passed directly
```

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

### Enum Usage

| Enum | Purpose |
|------|---------|
| `EnumMessageCategory` | Message routing (`EVENT`, `COMMAND`, `INTENT`) |
| `EnumNodeOutputType` | Node validation (adds `PROJECTION` for reducers) |

## 🚨 Infrastructure Error Patterns

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

## 🏗️ Infrastructure Patterns

### Correlation ID Rules
1. Always propagate from incoming requests
2. Auto-generate with `uuid4()` if missing
3. Include in all error context

### Circuit Breaker (MixinAsyncCircuitBreaker)

Use for external service integrations:
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

**See**: `docs/patterns/circuit_breaker_implementation.md` for full details

### Transport Types
| Type | Value |
|------|-------|
| `DATABASE` | `"db"` |
| `KAFKA` | `"kafka"` |
| `HTTP` | `"http"` |
| `CONSUL` | `"consul"` |
| `VAULT` | `"vault"` |
| `VALKEY` | `"valkey"` |

**Error Context**:
- Blocked requests raise `InfraUnavailableError` with proper `ModelInfraErrorContext`
- Includes correlation_id, operation, service_name, circuit state
- Provides retry_after_seconds for clients
- All errors follow infrastructure error sanitization guidelines

**Configuration Guidelines**:
```python
# High-reliability service (strict failure tolerance)
self._init_circuit_breaker(
    threshold=3,              # Open after 3 failures
    reset_timeout=120.0,      # 2 minutes recovery
    service_name="critical-service",
    transport_type=EnumInfraTransportType.DATABASE,
)

# Best-effort service (lenient failure tolerance)
self._init_circuit_breaker(
    threshold=10,             # Open after 10 failures
    reset_timeout=30.0,       # 30 seconds recovery
    service_name="cache-service",
    transport_type=EnumInfraTransportType.VALKEY,
)
```

**Integration with Error Recovery**:
```python
from pydantic import BaseModel

async def publish_with_retry(self, message: BaseModel) -> None:
    """Publish message with circuit breaker and retry logic."""
    correlation_id = uuid4()
    max_retries = 3

    for attempt in range(max_retries):
        # Check circuit breaker before each attempt
        async with self._circuit_breaker_lock:
            try:
                await self._check_circuit_breaker("publish", correlation_id)
            except InfraUnavailableError:
                # Circuit open - fail fast without retrying
                raise

        try:
            # Attempt publish (outside lock for I/O)
            await self._kafka_producer.send(message)

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
            return  # Success

        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("publish", correlation_id)

            if attempt == max_retries - 1:
                raise  # Last attempt failed

            # Exponential backoff before retry
            await asyncio.sleep(2 ** attempt)
```

**Monitoring and Observability**:
```python
# Circuit breaker exposes state for monitoring (access under lock)
async with self._circuit_breaker_lock:
    is_open = self._circuit_breaker_open           # bool: True if circuit open
    failure_count = self._circuit_breaker_failures  # int: consecutive failures
    open_until = self._circuit_breaker_open_until   # float: timestamp for auto-reset

# Log state transitions for operational insights
if is_open:
    logger.warning(
        "Circuit breaker opened",
        extra={
            "service_name": self.service_name,
            "failure_count": failure_count,
            "open_until": open_until,
            "correlation_id": str(correlation_id),
        },
    )
```

**Related Work**:
- Protocol definition: OMN-861 (Phase 2 - omnibase_spi)
- Implementation: `src/omnibase_infra/mixins/mixin_async_circuit_breaker.py`
- Concurrency safety docs: `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md`
- Example usage: VaultHandler, KafkaEventBus integration
- Error handling: See "Error Recovery Patterns" section above

**Best Practices**:
- Always propagate correlation_id through circuit breaker calls
- Use descriptive operation names ("connect", "publish", "query")
- Configure threshold and timeout based on service characteristics
- Monitor circuit breaker state transitions for operational insights
- Combine with retry logic for transient failures
- Use circuit breaker for fail-fast behavior when service is down
- Never call circuit breaker methods without holding `_circuit_breaker_lock`
- Never suppress InfraUnavailableError from circuit breaker
- Never use circuit breaker for non-transient errors

**See Also**: `docs/patterns/async_thread_safety_pattern.md` - Lock scope, counter accuracy, and callback execution patterns for asyncio concurrency.

### Dispatcher Resilience Pattern

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap dispatchers with circuit breakers.

**Design Rationale**:
- **Separation of concerns**: Each dispatcher knows its specific failure modes and recovery strategies
- **Transport-specific tuning**: Kafka dispatchers need different thresholds than HTTP dispatchers
- **No hidden behavior**: Engine users see exactly what resilience each dispatcher provides
- **Composability**: Dispatchers can combine circuit breakers with retry, backoff, or degradation

**Dispatcher Implementation Pattern**:
```python
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.runtime import ProtocolMessageDispatcher


class MyDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher with built-in circuit breaker resilience."""

    def __init__(self, config: DispatcherConfig):
        self._init_circuit_breaker(
            threshold=config.failure_threshold,
            reset_timeout=config.reset_timeout_seconds,
            service_name=f"dispatcher.{config.target_service}",
            transport_type=config.transport_type,
        )

    # NOTE: ModelEventEnvelope[object] is used instead of Any to satisfy ONEX "no Any types"
    # rule. Dispatchers must accept envelopes with any payload type since the dispatch
    # engine routes based on topic/category, not payload shape. Using `object` provides
    # the same flexibility as Any but with explicit semantics.
    async def handle(self, envelope: ModelEventEnvelope[object]) -> ModelDispatchResult:
        """Handle message with circuit breaker protection."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("handle", envelope.correlation_id)

        try:
            result = await self._do_handle(envelope)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
            return result
        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("handle", envelope.correlation_id)
            raise
```

**What the Engine Does NOT Do**:
- Does not wrap dispatcher calls with circuit breakers
- Does not implement retry logic around dispatchers
- Does not catch and suppress dispatcher errors (except for error aggregation)

**What Dispatchers Should Do**:
- Implement `MixinAsyncCircuitBreaker` for external service calls
- Configure thresholds appropriate to their transport type
- Raise `InfraUnavailableError` when circuit opens (engine will capture this)
- Optionally combine with retry logic for transient failures

### Node Introspection Security Considerations

The `MixinNodeIntrospection` mixin uses Python reflection (via the `inspect` module) to automatically discover node capabilities. This provides powerful service discovery but has security implications that developers should understand.

**Threat Model**:

Introspection data could be valuable to an attacker for:
- **Reconnaissance**: Learning what operations a node supports to identify attack vectors (e.g., discovering `decrypt_*`, `admin_*` methods)
- **Architecture mapping**: Understanding system topology through protocol and mixin discovery
- **Version fingerprinting**: Identifying outdated versions with known vulnerabilities
- **State inference**: Deducing system state or health from FSM state values

**What Gets Exposed via Introspection**:
- Public method names (potential operations a node can perform)
- Method signatures (parameter names and type annotations - names like `api_key`, `decrypt_key` reveal sensitive purposes)
- Protocol and mixin implementations (discovered capabilities)
- FSM state information (if `MixinFSM` is present)
- Endpoint URLs (health, API, metrics paths)
- Node metadata (name, version, type from contract)

**What is NOT Exposed**:
- Private methods (prefixed with `_`) - completely excluded from discovery
- Method implementations or source code - only signatures, not logic
- Configuration values - secrets, connection strings, etc. are not exposed
- Environment variables or runtime parameters
- Request/response payloads or historical data

**Built-in Protections**:
The mixin includes several filtering mechanisms to limit exposure:
- **Private method exclusion**: Methods prefixed with `_` are excluded from capability discovery
- **Utility method filtering**: Common utility prefixes (`get_*`, `set_*`, `initialize*`, `start_*`, `stop_*`) are filtered out
- **Operation keyword matching**: Only methods matching operation keywords (`execute`, `handle`, `process`, `run`, `invoke`, `call`) are reported as capabilities. Node-type-specific keywords (e.g., `fetch`, `compute`, `aggregate`, `orchestrate`) are also available.
- **Configurable exclusions**: The `exclude_prefixes` parameter allows additional filtering
- **Caching with TTL**: Introspection data is cached to reduce reflection frequency

**Best Practices for Node Developers**:
- Prefix internal/sensitive methods with `_` to exclude them from introspection
- Avoid exposing sensitive business logic in public method names
- Use generic operation names that don't reveal implementation details (e.g., `process_request` instead of `decrypt_and_forward_to_payment_gateway`)
- Use generic parameter names (e.g., `data` instead of `user_credentials`, `payload` instead of `encrypted_secret`)
- Review exposed capabilities before deploying to production environments
- Consider network segmentation for introspection event topics in multi-tenant environments
- Use the `exclude_prefixes` parameter to filter additional method patterns if needed

**Example - Reviewing Exposed Capabilities**:

> **Note on Example Models**: The following example uses simplified `NodeInput` and `NodeOutput`
> placeholder models for demonstration purposes only. These are **not** production model names.
> In production ONEX nodes, input/output models follow the `Model<NodeName>Input` and
> `Model<NodeName>Output` naming convention (e.g., `ModelVaultHandlerInput`, `ModelKafkaAdapterOutput`).
> See the "File & Class Naming Conventions" section above for complete naming rules.

```python
from pydantic import BaseModel

from omnibase_infra.mixins import MixinNodeIntrospection


# Simplified placeholder models for demonstration purposes.
# In production ONEX nodes, these would be named following the convention:
# - Model<NodeName>Input (e.g., ModelVaultHandlerInput, ModelConsulAdapterInput)
# - Model<NodeName>Output (e.g., ModelVaultHandlerOutput, ModelConsulAdapterOutput)
# See docs/architecture/CURRENT_NODE_ARCHITECTURE.md for full examples.


class NodeInput(BaseModel):
    """Placeholder input model for demonstration.

    Represents the typed input payload for a node operation.
    ONEX nodes use strongly-typed Pydantic models to ensure
    type safety and enable contract-driven validation.
    """

    value: str


class NodeOutput(BaseModel):
    """Placeholder output model for demonstration.

    Represents the typed output result from a node operation.
    All ONEX nodes return strongly-typed Pydantic models,
    never raw dictionaries or untyped data.
    """

    processed: bool


class MyNode(MixinNodeIntrospection):
    def execute_operation(self, data: NodeInput) -> NodeOutput:
        """Public operation - WILL be exposed."""
        return self._internal_process(data)

    def _internal_process(self, data: NodeInput) -> NodeOutput:
        """Private method - will NOT be exposed."""
        return NodeOutput(processed=True)

    def get_status(self) -> str:
        """Utility method - will NOT be exposed (get_* prefix filtered)."""
        return "healthy"


# Review what gets exposed
node = MyNode()
capabilities = node.get_capabilities()
# capabilities will only include "execute_operation"
```

**Network Security Considerations**:
- Introspection data is published to Kafka topics (`node.introspection`, `node.heartbeat`, `node.request_introspection`)
- In multi-tenant environments, ensure proper topic ACLs are configured
- Consider whether introspection topics should be accessible outside the cluster boundary
- Monitor introspection topic consumers for unauthorized access
- The registry listener responds to ANY request on the request topic without authentication - secure the topic with Kafka ACLs

**Production Deployment Checklist**:
1. Review `get_capabilities()` output for each node before deployment
2. Verify no sensitive method names or parameter names are exposed
3. Configure Kafka topic ACLs to restrict introspection topic access
4. Consider disabling `enable_registry_listener` if not needed
5. Monitor introspection topic consumer groups for unexpected consumers
6. Use network segmentation to isolate introspection traffic if required

**Cache Operations**:
The mixin manages introspection cache with TTL-based invalidation:

- **Cache Variables**: `_introspection_cache`, `_introspection_cached_at`, `_introspection_cache_ttl`
- **Cache Methods**:
  - `get_introspection_data()` - Returns cached data if TTL not expired, otherwise refreshes
  - `invalidate_introspection_cache()` - Clears cache to force refresh on next call (synchronous)

**Concurrency Safety Considerations**:

The `MixinNodeIntrospection` is designed for **single-threaded asyncio usage** and does NOT provide internal thread synchronization. It provides **coroutine safety** (protection against concurrent asyncio coroutines) but NOT **thread safety** (protection against multiple OS threads). Cache operations require understanding the concurrency model for safe usage.

**Instance-Level Cache** (`_introspection_cache`, `_introspection_cached_at`):
- Cache operations are **synchronous** (no async locking)
- Safe for cooperative asyncio concurrency (single event loop)
- **NOT safe** for multi-threaded access without external synchronization
- `invalidate_introspection_cache()` does not acquire any locks
- If called concurrently with `get_introspection_data()`, cache state may be inconsistent

**Class-Level Cache** (`_class_method_cache`):
- Shared `ClassVar` across all instances of the same class
- Population uses check-then-set pattern (not atomic)
- **Benign race condition**: Multiple threads may populate the cache simultaneously, but all produce identical results since method signatures are immutable after class definition
- Use `_invalidate_class_method_cache()` to clear if dynamic method registration occurs

**Background Tasks** (heartbeat loop, registry listener):
- Run as asyncio tasks within the event loop
- Designed for cooperative async concurrency
- Share access to instance state without locking
- **NOT safe** to call task methods from multiple threads

**Multi-Threaded Usage Pattern**:

If using this mixin in a multi-threaded application (e.g., with `concurrent.futures.ThreadPoolExecutor`), external synchronization is required:

```python
import threading

class ThreadSafeNode(MixinNodeIntrospection):
    def __init__(self, config):
        self._introspection_lock = threading.Lock()
        self.initialize_introspection(...)

    def invalidate_introspection_cache(self) -> None:
        with self._introspection_lock:
            super().invalidate_introspection_cache()

    async def get_introspection_data(self):
        # Note: async lock needed for async methods
        # Consider asyncio.Lock() for async coordination
        return await super().get_introspection_data()
```

**Recommendation**: For high-concurrency async environments, prefer using a single asyncio event loop with cooperative multitasking rather than multi-threading.

**Related**:
- Implementation: `src/omnibase_infra/mixins/mixin_node_introspection.py`
- Concurrency Safety Pattern: `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md` (similar pattern)
- Ticket: OMN-893
- See `MixinNodeIntrospection.get_capabilities()` for filtering logic details

### Service Integration Architecture
- **Adapter Pattern** - External services wrapped in ONEX adapters (Consul, Kafka, Vault)
- **Connection Pooling** - Database connections managed through dedicated pool managers
- **Event-Driven Communication** - Infrastructure events flow through Kafka adapters
- **Service Discovery** - Consul integration for dynamic service resolution
- **Secret Management** - Vault integration for secure credential handling

### Infrastructure 4-Node Pattern
- **EFFECT** - External service interactions (adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

## 🚀 Node Structure

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

## 🤖 Agent Architecture

| Category | Agents |
|----------|--------|
| Orchestration | `agent-onex-coordinator`, `agent-workflow-coordinator` |
| Development | `agent-contract-validator`, `agent-commit`, `agent-testing` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit` |
| Quality | `agent-pr-review`, `agent-pr-create` |

## 🔒 Zero Tolerance

- `Any` types forbidden
- Direct coding without agent delegation
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations
- Imperative nodes with custom routing logic

## 📦 Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Consul | 8500 |
| Kafka | 9092 |
| Vault | 8200 |
| PostgreSQL | 5432 |

---

**Bottom Line**: Declarative nodes, container injection, agent-driven development. No backwards compatibility, no custom node logic.
