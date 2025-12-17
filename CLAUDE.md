# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules and references for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `correlation_id_tracking.md` - Request tracing
- `circuit_breaker_implementation.md` - Circuit breaker details

---

## 🚨 MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

## 🚫 CRITICAL POLICY: NO BACKWARDS COMPATIBILITY

- Breaking changes are always acceptable
- No deprecated code maintenance
- Remove old patterns immediately

## 🎯 Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Always use specific types
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class

### File & Class Naming Conventions

| Type | File Pattern | Class Pattern | Example |
|------|-------------|---------------|---------|
| Model | `model_<name>.py` | `Model<Name>` | `model_kafka_message.py` → `ModelKafkaMessage` |
| Enum | `enum_<name>.py` | `Enum<Name>` | `enum_handler_type.py` → `EnumHandlerType` |
| Protocol | `protocol_<name>.py` | `Protocol<Name>` | `protocol_event_bus.py` → `ProtocolEventBus` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` | `mixin_health_check.py` → `MixinHealthCheck` |
| Service | `service_<name>.py` | `Service<Name>` | `service_discovery.py` → `ServiceDiscovery` |
| Util | `util_<name>.py` | (functions) | `util_retry.py` → `retry_with_backoff()` |
| Error | In `errors/` | `<Domain><Type>Error` | `InfraConnectionError` |
| Node | `node.py` | `Node<Name><Type>` | `NodePostgresAdapterEffect` |

### Registry Naming Conventions

**Node-Specific Registries** (`nodes/<name>/v<version>/registry/`):
- File: `registry_infra_<node_name>.py`
- Class: `RegistryInfra<NodeName>`
- Examples: `registry_infra_postgres_adapter.py` → `RegistryInfraPostgresAdapter`

**Standalone Registries** (in domain directories):
- File: `registry_<purpose>.py`
- Class: `Registry<Purpose>`
- Examples: `registry_handler.py` → `RegistryHandler`, `registry_policy.py` → `RegistryPolicy`, `registry_compute.py` → `RegistryCompute`

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

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - `def __init__(self, container: ModelONEXContainer)`
- **Protocol Resolution** - Duck typing through protocols, never isinstance
- **OnexError Only** - `raise OnexError(...) from e`

### Node Archetypes & Core Models (from `omnibase_core`)

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

**Architecture Rule**: `omnibase_infra` implements infrastructure-specific nodes by extending these base classes. Never define new node archetypes in infra - they belong in core.

| Layer | Responsibility | Example |
|-------|---------------|---------|
| `omnibase_core` | Node archetypes, I/O models, enums | `NodeReducer`, `ModelReducerInput` |
| `omnibase_spi` | Protocol definitions | `ProtocolReducerNode` |
| `omnibase_infra` | Infrastructure implementations | `NodeDualRegistrationReducer` |

### Container-Based Dependency Injection

**All services MUST use ModelONEXContainer for dependency injection.**

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

# Bootstrap and resolve
container = ModelONEXContainer()
wire_infrastructure_services(container)
service = container.service_registry.resolve_service(ServiceType)
```

**See**: `docs/patterns/container_dependency_injection.md` for complete patterns and examples

## 🚨 Infrastructure Error Patterns

### Error Class Selection (Quick Reference)

| Scenario | Error Class | Transport Code |
|----------|-------------|----------------|
| Config invalid | `ProtocolConfigurationError` | N/A |
| Secret not found | `SecretResolutionError` | N/A |
| Connection failed | `InfraConnectionError` | `DATABASE_CONNECTION_ERROR` / `NETWORK_ERROR` / `SERVICE_UNAVAILABLE` |
| Timeout | `InfraTimeoutError` | Same as connection |
| Auth failed | `InfraAuthenticationError` | Same as connection |
| Unavailable | `InfraUnavailableError` | `SERVICE_UNAVAILABLE` |

### Error Context Example

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="postgresql-primary",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect", context=context) from original_error
```

**See Also**:
- `docs/patterns/error_handling_patterns.md` - Complete error hierarchy and usage
- `docs/patterns/error_recovery_patterns.md` - Recovery strategies (backoff, circuit breaker, degradation)
- `docs/patterns/correlation_id_tracking.md` - Request tracing patterns
- `docs/patterns/circuit_breaker_implementation.md` - Circuit breaker details

## 🏗️ Infrastructure Architecture

### Correlation ID Assignment Rules

Correlation IDs enable distributed tracing across infrastructure components:

1. **Always propagate**: Pass `correlation_id` from incoming requests to error context
2. **Auto-generation**: If no `correlation_id` exists, generate one using `uuid4()`
3. **UUID format**: Use UUID4 format for all new correlation IDs
4. **Include everywhere**: Add `correlation_id` in all error context for tracing

```python
from uuid import UUID, uuid4

# Pattern 1: Propagate from request
correlation_id = request.correlation_id or uuid4()

# Pattern 2: Generate if not available
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.KAFKA,
    operation="produce_message",
    correlation_id=correlation_id,
)

# Pattern 3: Extract from incoming event
correlation_id = event.metadata.get("correlation_id")
if isinstance(correlation_id, str):
    correlation_id = UUID(correlation_id)
```

### Error Sanitization Guidelines

**NEVER include in error messages or context**:
- Passwords, API keys, tokens, secrets
- Full connection strings with credentials
- PII (names, emails, SSNs, phone numbers)
- Internal IP addresses (in production logs)
- Private keys or certificates
- Session tokens or cookies

**SAFE to include**:
- Service names (e.g., "postgresql", "kafka")
- Operation names (e.g., "connect", "query", "authenticate")
- Correlation IDs (always include for tracing)
- Error codes (e.g., `EnumCoreErrorCode.DATABASE_CONNECTION_ERROR`)
- Sanitized hostnames (e.g., "db.example.com")
- Port numbers
- Retry counts and timeout values
- Resource identifiers (non-sensitive)

```python
# BAD - Exposes credentials
raise InfraConnectionError(
    f"Failed to connect with password={password}",  # NEVER DO THIS
    context=context,
)

# GOOD - Sanitized error message
raise InfraConnectionError(
    "Failed to connect to database",
    context=context,
    host="db.example.com",
    port=5432,
    retry_count=3,
)

# BAD - Full connection string
raise InfraConnectionError(
    f"Connection failed: {connection_string}",  # May contain credentials
    context=context,
)

# GOOD - Sanitized connection info
raise InfraConnectionError(
    "Connection failed",
    context=context,
    host=parsed_host,
    port=parsed_port,
    database=database_name,
)
```

### Error Hierarchy Reference

```
ModelOnexError (from omnibase_core)
└── RuntimeHostError (base infrastructure error)
    ├── ProtocolConfigurationError  # Config validation failures
    ├── SecretResolutionError       # Secret/credential resolution
    ├── InfraConnectionError        # Connection failures
    ├── InfraTimeoutError           # Operation timeouts
    ├── InfraAuthenticationError    # Auth/authz failures
    └── InfraUnavailableError           # Resource unavailable
```

### Error Code Mapping Reference

| Error Class | EnumCoreErrorCode | HTTP Equivalent |
|-------------|-------------------|-----------------|
| `ProtocolConfigurationError` | `INVALID_CONFIGURATION` | 400 Bad Request |
| `SecretResolutionError` | `RESOURCE_NOT_FOUND` | 404 Not Found |
| `InfraConnectionError` | **Transport-aware** (see below) | 503 Service Unavailable |
| `InfraTimeoutError` | `TIMEOUT_ERROR` | 504 Gateway Timeout |
| `InfraAuthenticationError` | `AUTHENTICATION_ERROR` | 401 Unauthorized |
| `InfraUnavailableError` | `SERVICE_UNAVAILABLE` | 503 Service Unavailable |

#### InfraConnectionError Transport-Aware Error Codes

`InfraConnectionError` automatically selects the appropriate error code based on `context.transport_type`:

| Transport Type | EnumCoreErrorCode | Rationale |
|----------------|-------------------|-----------|
| `DATABASE` | `DATABASE_CONNECTION_ERROR` | Specific database connection error |
| `HTTP` | `NETWORK_ERROR` | Network-level transport failure |
| `GRPC` | `NETWORK_ERROR` | Network-level transport failure |
| `KAFKA` | `SERVICE_UNAVAILABLE` | Message broker service unavailable |
| `CONSUL` | `SERVICE_UNAVAILABLE` | Service discovery unavailable |
| `VAULT` | `SERVICE_UNAVAILABLE` | Secret management service unavailable |
| `VALKEY` | `SERVICE_UNAVAILABLE` | Cache service unavailable |
| `None` (no context) | `SERVICE_UNAVAILABLE` | Generic fallback |

```python
# Example: Transport-aware error code selection
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Database connection -> DATABASE_CONNECTION_ERROR
db_context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.DATABASE)
db_error = InfraConnectionError("DB failed", context=db_context)
assert db_error.model.error_code.name == "DATABASE_CONNECTION_ERROR"

# HTTP connection -> NETWORK_ERROR
http_context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.HTTP)
http_error = InfraConnectionError("API failed", context=http_context)
assert http_error.model.error_code.name == "NETWORK_ERROR"

# Kafka connection -> SERVICE_UNAVAILABLE
kafka_context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.KAFKA)
kafka_error = InfraConnectionError("Kafka failed", context=kafka_context)
assert kafka_error.model.error_code.name == "SERVICE_UNAVAILABLE"
```

### Error Recovery Patterns

Infrastructure errors often require recovery strategies. Here are common patterns for handling infrastructure failures:

#### Retry with Exponential Backoff (Connection Errors)

Use exponential backoff for transient connection failures. This pattern is ideal for `InfraConnectionError` when services are temporarily unavailable:

```python
import time
from uuid import uuid4
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

def connect_with_retry(host: str, port: int, max_retries: int = 3) -> Connection:
    """Connect to database with exponential backoff retry strategy."""
    correlation_id = uuid4()
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="connect",
        target_name="postgresql-primary",
        correlation_id=correlation_id,
    )

    for attempt in range(max_retries):
        try:
            return create_connection(host, port)
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise InfraConnectionError(
                    f"Failed to connect after {max_retries} attempts",
                    context=context,
                    host=host,
                    port=port,
                    retry_count=attempt + 1,
                ) from e

            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            time.sleep(wait_time)
```

#### Circuit Breaker Pattern (Unavailable Services)

Use the circuit breaker pattern for `InfraUnavailableError` to prevent cascading failures and give services time to recover:

```python
import time
from enum import Enum
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

class CircuitState(str, Enum):
    """Circuit breaker state machine."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Prevents cascading failures with configurable circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        context: ModelInfraErrorContext = None,
    ):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED
        self.context = context or ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="circuit_breaker",
            target_name="service",
        )

    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
            else:
                raise InfraUnavailableError(
                    "Circuit breaker is open - service temporarily unavailable",
                    context=self.context,
                    circuit_state=self.state.value,
                    retry_after_seconds=int(
                        self.reset_timeout - (time.time() - self.last_failure_time)
                    ),
                )

        try:
            result = func(*args, **kwargs)

            # Success - reset circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Open circuit if threshold exceeded
            if self.failure_count >= self.threshold:
                self.state = CircuitState.OPEN

            raise
```

#### Graceful Degradation (Timeout Errors)

Use graceful degradation for `InfraTimeoutError` to maintain service availability with reduced functionality:

```python
from omnibase_infra.errors import InfraTimeoutError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

def fetch_with_timeout_fallback(
    primary_func,
    fallback_func,
    timeout_seconds: float = 5.0,
    correlation_id = None,
) -> dict:
    """Fetch from primary source with graceful degradation to fallback."""
    import signal

    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="fetch",
        target_name="primary-source",
        correlation_id=correlation_id,
    )

    def timeout_handler(signum, frame):
        raise TimeoutError("Operation exceeded timeout")

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))

    try:
        # Try primary data source
        return {"data": primary_func(), "source": "primary", "degraded": False}

    except TimeoutError as e:
        # Log timeout but continue with fallback
        context_with_fallback = ModelInfraErrorContext(
            transport_type=context.transport_type,
            operation=context.operation,
            target_name=context.target_name,
            correlation_id=context.correlation_id,
        )

        try:
            # Use fallback source (cache, secondary database, etc.)
            return {
                "data": fallback_func(),
                "source": "fallback",
                "degraded": True,
                "warning": f"Primary source timed out, using fallback data",
            }

        except Exception as fallback_error:
            raise InfraTimeoutError(
                "Primary timeout and fallback failed",
                context=context_with_fallback,
                timeout_seconds=timeout_seconds,
            ) from fallback_error

    finally:
        signal.alarm(0)  # Cancel alarm
```

#### Credential Refresh (Authentication Errors)

Use credential refresh for `InfraAuthenticationError` to handle token expiration gracefully:

```python
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
import time

class CredentialRefreshManager:
    """Manages credential refresh with automatic token renewal."""

    def __init__(
        self,
        credential_provider,
        refresh_threshold_seconds: float = 300.0,
    ):
        self.provider = credential_provider
        self.refresh_threshold = refresh_threshold_seconds
        self.current_credential = None
        self.credential_expires_at = 0.0
        self.context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="credential_refresh",
            target_name="vault-server",
        )

    def get_valid_credential(self):
        """Get credential, refreshing if near expiration."""
        current_time = time.time()

        # Check if credential exists and is still valid
        if (
            self.current_credential is not None
            and current_time < self.credential_expires_at - self.refresh_threshold
        ):
            return self.current_credential

        # Credential missing, expired, or approaching expiration - refresh
        try:
            credential = self.provider.refresh_credential()
            self.current_credential = credential
            self.credential_expires_at = (
                current_time + credential.get("ttl_seconds", 3600)
            )
            return credential

        except Exception as e:
            raise InfraAuthenticationError(
                "Failed to refresh authentication credentials",
                context=self.context,
                provider="vault",
            ) from e

    def call_with_auth(self, func, *args, **kwargs):
        """Execute function with automatic credential refresh on auth failure."""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                credential = self.get_valid_credential()
                return func(*args, credential=credential, **kwargs)

            except InfraAuthenticationError as e:
                if attempt == max_retries - 1:
                    # Last attempt failed - propagate error
                    raise

                # Force refresh and retry
                self.current_credential = None
                self.credential_expires_at = 0.0
```

### Transport Type Reference

Use `EnumInfraTransportType` for transport identification in error context:

| Transport Type | Value | Usage |
|---------------|-------|-------|
| `HTTP` | `"http"` | REST API transport |
| `DATABASE` | `"db"` | PostgreSQL, etc. |
| `KAFKA` | `"kafka"` | Kafka message broker |
| `CONSUL` | `"consul"` | Service discovery |
| `VAULT` | `"vault"` | Secret management |
| `VALKEY` | `"valkey"` | Cache/message transport |
| `GRPC` | `"grpc"` | gRPC protocol |

## 🏗️ Infrastructure-Specific Patterns

### Accepted Pattern Exceptions

**KafkaEventBus Complexity** (Documented Exception):
The KafkaEventBus intentionally violates pattern validator thresholds:
- **14 methods** (threshold: 10) - Required for event bus pattern implementation
- **10 __init__ parameters** (threshold: 5) - Backwards compatibility during config migration

This complexity is acceptable and documented because:
1. **Event Bus Pattern Requirements**: Lifecycle, pub/sub, circuit breaker, protocol compatibility
2. **Backwards Compatibility**: Gradual migration from direct parameters to config objects
3. **Infrastructure Cohesion**: Keeping related event bus operations together improves maintainability
4. **Well-Documented**: Design rationale documented in class and method docstrings

The violations are **intentional infrastructure patterns**, not code smells. See:
- `src/omnibase_infra/event_bus/kafka_event_bus.py` - Full design documentation
- `src/omnibase_infra/validation/infra_validators.py` - Validation notes

### Circuit Breaker Pattern (MixinAsyncCircuitBreaker)

All infrastructure adapters and services should use `MixinAsyncCircuitBreaker` for fault tolerance and automatic recovery.

**When to Use**:
- External service integrations (Kafka, Consul, Vault, Redis, PostgreSQL)
- Network operations that can fail transiently
- Any infrastructure component requiring automatic fault recovery
- Services with configurable failure thresholds and reset timeouts

**Integration Pattern**:
```python
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType
from uuid import uuid4

class MyInfrastructureAdapter(MixinAsyncCircuitBreaker):
    def __init__(self, config: MyConfig):
        # Initialize circuit breaker with service-specific settings
        # This creates self._circuit_breaker_lock automatically
        self._init_circuit_breaker(
            threshold=5,                    # Max failures before opening
            reset_timeout=60.0,             # Seconds until auto-reset
            service_name=f"my-service.{environment}",
            transport_type=EnumInfraTransportType.HTTP,  # Or KAFKA, CONSUL, etc.
        )

    async def connect(self) -> None:
        """Connect to external service with circuit breaker protection."""
        correlation_id = uuid4()

        # Check circuit breaker before operation (caller-held lock pattern)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("connect", correlation_id)

        try:
            # Attempt connection (outside lock for I/O operations)
            await self._do_connect()

            # Record success (resets circuit breaker)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        except Exception as e:
            # Record failure (may open circuit)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("connect", correlation_id)
            raise
```

**Thread Safety**:
- Circuit breaker methods REQUIRE caller to hold `self._circuit_breaker_lock`
- The lock is created automatically by `_init_circuit_breaker()`
- Always use `async with self._circuit_breaker_lock:` before calling circuit breaker methods
- Never call circuit breaker methods without lock protection

**State Transitions**:
- **CLOSED**: Normal operation, requests allowed
- **OPEN**: Too many failures, requests blocked (raises `InfraUnavailableError`)
- **HALF_OPEN**: After timeout, testing if service recovered
- **CLOSED**: Service recovered, normal operation resumed

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
async def publish_with_retry(self, message: dict) -> None:
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
- Thread safety docs: `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md`
- Example usage: VaultAdapter, KafkaEventBus integration
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

### Service Integration Architecture
- **Adapter Pattern** - External services wrapped in ONEX adapters (Consul, Kafka, Vault)
- **Connection Pooling** - Database connections managed through dedicated pool managers
- **Event-Driven Communication** - Infrastructure events flow through Kafka adapters
- **Service Discovery** - Consul integration for dynamic service resolution
- **Secret Management** - Vault integration for secure credential handling

### Infrastructure 4-Node Pattern
Infrastructure tools follow ONEX 4-node architecture:
- **EFFECT** - External service interactions (Consul, Kafka, Vault adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

### Service Adapters
| Adapter | Purpose |
|---------|---------|
| `consul_adapter` | Service discovery |
| `kafka_adapter` | Event streaming |
| `vault_adapter` | Secret management |
| `postgres_adapter` | Database operations |

## 🤖 Agent Architecture

### Orchestration Agents

| Agent | Purpose |
|-------|---------|
| `agent-onex-coordinator` | Primary routing and workflow orchestration |
| `agent-workflow-coordinator` | Multi-step execution, sub-agent fleet coordination |
| `agent-ticket-manager` | Ticket lifecycle, dependency analysis |

### Specialist Agents

| Category | Agents |
|----------|--------|
| Development | `agent-contract-validator`, `agent-contract-driven-generator`, `agent-ast-generator`, `agent-commit` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit`, `agent-performance`, `agent-production-monitor` |
| Quality | `agent-pr-review`, `agent-pr-create`, `agent-address-pr-comments`, `agent-testing` |
| Intelligence | `agent-research`, `agent-debug-intelligence`, `agent-rag-query`, `agent-rag-update` |

## 🔒 Zero Tolerance Policies

- `Any` types forbidden
- Direct coding without agent delegation prohibited
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations

## 🔧 DevOps Quick Reference

**Container Troubleshooting**: `docker logs <container>` first, then `docker inspect` for exit codes
- **Exit 0 OK**: Init containers (topic creation, migrations, SSL cert gen)
- **Should run**: Services (web, brokers, databases, load balancers)


## 📦 Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Infrastructure Hub | 8085 |
| Consul | 8500 (HTTP), 8600 (DNS) |
| Kafka | 9092 (plaintext), 9093 (SSL) |
| Vault | 8200 |
| PostgreSQL | 5432 |
| Debug Dashboard | 8096 |

## 🚀 Node Structure Pattern

```
nodes/<adapter>/v1_0_0/
├── contract.yaml          # ONEX contract definition
├── node.py               # Node<Name><Type> implementation
├── models/               # Node-specific models
└── registry/             # registry_infra_<name>.py
```

**Contract Requirements**:
- Semantic versioning (`contract_version`, `node_version`)
- Node type (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR)
- Strongly typed I/O (`input_model`, `output_model`)
- Protocol-based dependencies
- Zero `Any` types, use `omnibase_core.*` imports

---

**Bottom Line**: Agent-driven development. Route through orchestrators, delegate to specialists. Strong typing, contract-driven configuration, no backwards compatibility.
