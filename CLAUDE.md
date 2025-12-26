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

| Type | File Pattern | Class Pattern |
|------|-------------|---------------|
| Model | `model_<name>.py` | `Model<Name>` |
| Enum | `enum_<name>.py` | `Enum<Name>` |
| Protocol | `protocol_<name>.py` | `Protocol<Name>` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` |
| Node | `node.py` | `Node<Name><Type>` |
| Error | In `errors/` | `<Domain><Type>Error` |

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
