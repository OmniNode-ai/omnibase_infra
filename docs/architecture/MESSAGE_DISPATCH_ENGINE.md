# Message Dispatch Engine Architecture

## Overview

The Message Dispatch Engine (`MessageDispatchEngine`) is a runtime routing engine for
message dispatching based on topic category and message type. It routes incoming messages
to registered dispatchers and collects dispatcher outputs for publishing.

**Implementation**: `src/omnibase_infra/runtime/message_dispatch_engine.py`
**Ticket**: OMN-934
**Version**: 0.4.0

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Pure Routing** | Routes messages to dispatchers, no workflow inference or semantic understanding |
| **Deterministic** | Same input always produces same dispatcher selection |
| **Fan-out Support** | Multiple dispatchers can process the same message type concurrently |
| **Freeze-After-Init** | Thread-safe after registration phase completes |
| **Observable** | Structured logging and comprehensive metrics collection |

### What the Engine Does

- Route registration for topic pattern matching
- Dispatcher registration by category and message type
- Message dispatch with category validation
- Metrics collection for observability
- Structured logging for debugging and monitoring

### What the Engine Does NOT Do

- Infer workflow semantics from message content
- Manage dispatcher lifecycle (dispatchers are external)
- Perform message transformation or enrichment
- Make decisions about message ordering or priority
- Wrap dispatchers with circuit breakers (dispatchers own their resilience)

## Architecture Diagram

```
+------------------------------------------------------------------+
|                   Message Dispatch Engine                         |
+------------------------------------------------------------------+
|                                                                  |
|   1. Parse Topic       2. Validate          3. Match Dispatchers |
|        |                   |                       |             |
|        |  topic string     |  category match       |             |
|        |-------------------|----------------------|             |
|        |                   |                       |             |
|        | EnumMessageCategory                       | dispatchers[]|
|        |<------------------|                       |------------>|
|        |                   |                       |             |
|   4. Execute Dispatchers 5. Collect Outputs  6. Return Result    |
|        |                   |                       |             |
|        | dispatcher outputs|  aggregate           |             |
|        |-------------------|----------------------|             |
|        |                   |                       |             |
|        |                   |  ModelDispatchResult  |             |
|        |<------------------|<---------------------|             |
|                                                                  |
+------------------------------------------------------------------+
```

## Dispatch Sequence Diagram

```
User/Caller          MessageDispatchEngine            Dispatchers (1..N)
     |                        |                              |
     | dispatch(topic, env)   |                              |
     |----------------------->|                              |
     |                        |                              |
     |                        | 1. Parse topic               |
     |                        |    EnumMessageCategory       |
     |                        |    .from_topic(topic)        |
     |                        |----------------------------->|
     |                        |                              |
     |                        | 2. Validate envelope         |
     |                        |    (topic != null,           |
     |                        |     envelope != null)        |
     |                        |----------------------------->|
     |                        |                              |
     |                        | 3. Find matching dispatchers |
     |                        |    _find_matching_dispatchers|
     |                        |    (topic, category, type)   |
     |                        |----------------------------->|
     |                        |                              |
     |                        |<-- dispatchers[] ------------|
     |                        |                              |
     |                        | 4. Execute each dispatcher   |
     |                        |    (fan-out pattern)         |
     |                        |----------------------------->|
     |                        |                              |
     |                        |    For each dispatcher:      |
     |                        |    +------------------------+|
     |                        |    | _execute_dispatcher    ||
     |                        |    |   - async: await       ||
     |                        |    |   - sync: executor     ||
     |                        |    +------------------------+|
     |                        |                              |
     |                        |<-- DispatcherOutput ---------|
     |                        |    (str | list[str] | None)  |
     |                        |                              |
     |                        | 5. Aggregate outputs         |
     |                        |    - Collect output topics   |
     |                        |    - Collect errors          |
     |                        |    - Update metrics          |
     |                        |----------------------------->|
     |                        |                              |
     | ModelDispatchResult    | 6. Return result             |
     |<-----------------------|                              |
     |  - dispatch_id         |                              |
     |  - status              |                              |
     |  - outputs[]           |                              |
     |  - duration_ms         |                              |
     |  - error_message       |                              |
     |                        |                              |
```

## Registration Phase

### Route Registration

Routes define how messages are matched to dispatchers based on topic pattern,
message category, and optionally message type.

```python
from omnibase_infra.runtime import MessageDispatchEngine
from omnibase_infra.models.dispatch import ModelDispatchRoute
from omnibase_infra.enums import EnumMessageCategory

engine = MessageDispatchEngine()

# Register a route with glob pattern matching
engine.register_route(ModelDispatchRoute(
    route_id="order-events",
    topic_pattern="*.order.events.*",
    message_category=EnumMessageCategory.EVENT,
    dispatcher_id="order-dispatcher",
    priority=100,
    description="Routes order-related events",
))
```

**Route Pattern Matching**:
- `*` matches any single segment (e.g., `dev` in `dev.order.events.v1`)
- `**` matches multiple segments (e.g., `order.events.v1` in `dev.order.events.v1`)
- Case-insensitive matching

### Dispatcher Registration

Dispatchers process messages that match their category and (optionally) message type.

```python
async def process_user_event(
    envelope: ModelEventEnvelope[object]
) -> str | list[str] | None:
    """
    Process user event and optionally return output topics.

    Returns:
        - str: A single output topic to publish to
        - list[str]: Multiple output topics
        - None: No output topics
    """
    user_data = envelope.payload
    # Process the event
    return "dev.user.processed.v1"  # Output topic

engine.register_dispatcher(
    dispatcher_id="user-event-dispatcher",
    dispatcher=process_user_event,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreated", "UserUpdated"},
)
```

### Freeze Pattern

The engine follows a **freeze-after-init** pattern for thread safety:

```
┌─────────────────────┐    ┌────────────┐    ┌──────────────────────────┐
│ Registration Phase  │───>│  freeze()  │───>│    Dispatch Phase        │
│ (single-threaded)   │    │            │    │ (multi-threaded safe)    │
└─────────────────────┘    └────────────┘    └──────────────────────────┘
```

1. **Registration Phase** (single-threaded): Register routes and dispatchers
2. **Freeze**: Call `freeze()` to validate and prevent further modifications
3. **Dispatch Phase** (multi-threaded safe): Route messages to dispatchers

```python
engine = MessageDispatchEngine()
engine.register_dispatcher("d1", dispatcher, EnumMessageCategory.EVENT)
engine.register_route(route)
engine.freeze()  # Validates and freezes

# Now thread-safe for concurrent dispatch
result = await engine.dispatch("dev.user.events.v1", envelope)
```

**Freeze Validation**:
- All routes must reference existing dispatchers
- Raises `ModelOnexError(ITEM_NOT_REGISTERED)` if validation fails

## Dispatch Phase

### Dispatch Flow

1. **Parse Topic**: Extract message category from topic string using `EnumMessageCategory.from_topic()`
2. **Validate**: Ensure topic and envelope are valid
3. **Match**: Find all dispatchers matching category + message type
4. **Execute**: Run dispatchers (fan-out for multiple matches)
5. **Collect**: Aggregate outputs and errors
6. **Return**: Build `ModelDispatchResult`

### Dispatcher Output Types

Dispatchers can return:
- `str`: A single output topic (e.g., `"dev.notification.v1"`)
- `list[str]`: Multiple output topics (e.g., `["dev.email.v1", "dev.sms.v1"]`)
- `None`: No output topics to publish

### Fan-Out Pattern

Multiple dispatchers can process the same message type concurrently:

```python
# Register multiple dispatchers for the same category
engine.register_dispatcher(
    dispatcher_id="audit-logger",
    dispatcher=log_to_audit,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreated"},  # Handles UserCreated
)

engine.register_dispatcher(
    dispatcher_id="notification-sender",
    dispatcher=send_notification,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreated"},  # Also handles UserCreated
)

# Both dispatchers execute when a UserCreated event arrives
result = await engine.dispatch("dev.user.events.v1", user_created_envelope)
# result.outputs contains outputs from both dispatchers
```

### Error Handling

- Dispatcher exceptions are caught and sanitized (no credential leakage)
- Dispatch continues to other dispatchers even if one fails
- Errors are aggregated in `ModelDispatchResult.error_message`
- Each dispatcher error is logged with correlation ID for tracing

```python
# If one dispatcher fails, others still execute
# result.status will be HANDLER_ERROR
# result.error_message contains sanitized error details
# result.outputs contains outputs from successful dispatchers
```

## Thread Safety Model

### Freeze-After-Init Pattern

The engine uses a two-phase lifecycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                    REGISTRATION PHASE                           │
│                    (Single-Threaded)                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ register_route()    │    │ register_dispatcher()│            │
│  │ Protected by        │    │ Protected by         │            │
│  │ _registration_lock  │    │ _registration_lock   │            │
│  └─────────────────────┘    └─────────────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                         freeze()                                 │
│                         ↓                                        │
├─────────────────────────────────────────────────────────────────┤
│                      DISPATCH PHASE                              │
│                    (Multi-Threaded Safe)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ dispatch()          │    │ get_metrics()       │            │
│  │ Read-only access    │    │ Metrics lock        │            │
│  │ to routes/          │    │ protected           │            │
│  │ dispatchers         │    │                     │            │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Metrics Lock

Structured metrics updates are protected by `_metrics_lock`:

```python
# All read-modify-write operations within single lock acquisition
with self._metrics_lock:
    self._structured_metrics = self._structured_metrics.record_dispatch(
        duration_ms=duration_ms,
        success=True,
        category=topic_category,
        topic=topic,
    )
```

**Key Properties**:
- Lock protects read-modify-write cycles on `_structured_metrics`
- Lock is NOT held during dispatcher execution (I/O operations)
- Computations within lock are pure and fast (no I/O)
- Returns consistent snapshots via `get_structured_metrics()`

### Sync Dispatcher Execution

Sync dispatchers execute via `run_in_executor()` using the default `ThreadPoolExecutor`:

```
+------------------------------------------------------------------+
|                     WARNING: SYNC DISPATCHERS                     |
+------------------------------------------------------------------+
|                                                                  |
|  Sync dispatchers MUST be non-blocking (< 100ms execution).      |
|                                                                  |
|  Blocking dispatchers can exhaust the thread pool, causing:      |
|    - Starvation of other sync dispatchers                        |
|    - Delayed async dispatcher scheduling                         |
|    - Potential deadlocks under high load                         |
|    - Increased latency for all executor-based operations         |
|                                                                  |
|  BEST PRACTICES:                                                 |
|    - Sync dispatchers: < 100ms execution time                    |
|    - For blocking I/O: Use ASYNC dispatchers                     |
|    - For CPU-bound work: Consider ProcessPoolExecutor            |
|    - Monitor dispatcher_execution_count metrics                  |
|                                                                  |
+------------------------------------------------------------------+
```

## Observability

### Structured Logging

| Level | Events |
|-------|--------|
| **INFO** | Dispatch start/complete with topic, category, dispatcher count |
| **DEBUG** | Dispatcher execution details, routing decisions |
| **WARNING** | No dispatchers found, category mismatches |
| **ERROR** | Dispatcher exceptions, validation failures |

**Log Context Fields**:
- `topic`: The topic being dispatched
- `category`: Message category (event, command, intent)
- `message_type`: Specific message type
- `dispatcher_id`: Dispatcher identifier(s)
- `dispatcher_count`: Number of dispatchers matched
- `duration_ms`: Dispatch duration in milliseconds
- `correlation_id`: UUID for distributed tracing
- `trace_id`: Trace ID from envelope
- `error_code`: Error code if dispatch failed

### Structured Metrics

Access via `get_structured_metrics()`:

```python
metrics = engine.get_structured_metrics()

# Overall statistics
print(f"Total dispatches: {metrics.total_dispatches}")
print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
print(f"Min latency: {metrics.min_latency_ms:.2f} ms")
print(f"Max latency: {metrics.max_latency_ms:.2f} ms")

# Latency histogram for distribution analysis
for bucket, count in metrics.latency_histogram.items():
    print(f"  {bucket}: {count}")

# Per-category breakdown
for category, count in metrics.category_metrics.items():
    print(f"  {category}: {count} dispatches")
```

### Per-Dispatcher Metrics

```python
metrics = engine.get_dispatcher_metrics("user-event-dispatcher")
if metrics:
    print(f"Executions: {metrics.execution_count}")
    print(f"Error rate: {metrics.error_rate:.1%}")
    print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
    print(f"Last error: {metrics.last_error_message}")
```

### Latency Histogram Buckets

```python
LATENCY_HISTOGRAM_BUCKETS = (
    1.0,      # le_1ms
    5.0,      # le_5ms
    10.0,     # le_10ms
    25.0,     # le_25ms
    50.0,     # le_50ms
    100.0,    # le_100ms
    250.0,    # le_250ms
    500.0,    # le_500ms
    1000.0,   # le_1000ms (1s)
    2500.0,   # le_2500ms (2.5s)
    5000.0,   # le_5000ms (5s)
    10000.0,  # le_10000ms (10s)
    # gt_10000ms for anything above
)
```

## Resilience Patterns

### Dispatcher-Owned Resilience

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap
dispatchers with circuit breakers.

**Rationale**:
- **Separation of concerns**: Each dispatcher knows its specific failure modes and recovery strategies
- **Transport-specific tuning**: Kafka dispatchers need different thresholds than HTTP dispatchers
- **No hidden behavior**: Engine users see exactly what resilience each dispatcher provides
- **Composability**: Dispatchers can combine circuit breakers with retry, backoff, degradation

**Dispatcher Implementation Pattern**:

```python
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType

class MyDispatcher(MixinAsyncCircuitBreaker, ProtocolMessageDispatcher):
    """Dispatcher with built-in circuit breaker resilience."""

    def __init__(self, config: DispatcherConfig):
        self._init_circuit_breaker(
            threshold=config.failure_threshold,
            reset_timeout=config.reset_timeout_seconds,
            service_name=f"dispatcher.{config.target_service}",
            transport_type=config.transport_type,
        )

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelDispatchResult:
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

See `CLAUDE.md` section "Dispatcher Resilience Pattern" for complete implementation guidance.
See `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md` for circuit breaker details.

### Error Sanitization

Exception messages are sanitized before storage to prevent credential leakage:

```python
# Sensitive patterns that trigger redaction
_SENSITIVE_PATTERNS = (
    "password", "secret", "token", "api_key",
    "credential", "bearer", "private_key",
    "connection_string", "postgres://", "kafka://",
    "mongodb://", "mysql://", "redis://", "amqp://",
)

# Example: Sensitive data detected
# Input: "Failed with password=secret123"
# Output: "ValueError: [REDACTED - potentially sensitive data]"
```

## Integration Examples

### Basic Usage

```python
from omnibase_infra.runtime import MessageDispatchEngine
from omnibase_infra.models.dispatch import ModelDispatchRoute
from omnibase_infra.enums import EnumMessageCategory
from omnibase_core.models.events import ModelEventEnvelope
from uuid import uuid4

# 1. Create engine with optional custom logger
import logging
logger = logging.getLogger("dispatch-engine")
engine = MessageDispatchEngine(logger=logger)

# 2. Define and register dispatchers
async def handle_user_created(envelope: ModelEventEnvelope[object]) -> str:
    user = envelope.payload
    print(f"User created: {user}")
    return "dev.notification.events.v1"  # Output topic

engine.register_dispatcher(
    dispatcher_id="user-created-handler",
    dispatcher=handle_user_created,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreatedEvent"},
)

# 3. Register routes
engine.register_route(ModelDispatchRoute(
    route_id="user-events",
    topic_pattern="*.user.events.*",
    message_category=EnumMessageCategory.EVENT,
    dispatcher_id="user-created-handler",
))

# 4. Freeze to enable dispatch
engine.freeze()

# 5. Dispatch messages (thread-safe after freeze)
envelope = ModelEventEnvelope(
    payload=UserCreatedEvent(user_id="123", email="test@example.com"),
    correlation_id=uuid4(),
)
result = await engine.dispatch("dev.user.events.v1", envelope)

# 6. Check result
if result.is_successful():
    print(f"Dispatched successfully, outputs: {result.outputs}")
else:
    print(f"Dispatch failed: {result.error_message}")
```

### Fan-Out with Multiple Dispatchers

```python
# Register multiple dispatchers for the same event type
async def log_audit(envelope):
    """Log to audit system."""
    audit_logger.log(envelope.payload)
    return None  # No output topic

async def send_email(envelope):
    """Send welcome email."""
    await email_service.send_welcome(envelope.payload.email)
    return "dev.email.sent.v1"

async def update_analytics(envelope):
    """Update analytics dashboard."""
    await analytics.track("user_created", envelope.payload)
    return ["dev.analytics.v1", "dev.metrics.v1"]  # Multiple outputs

engine.register_dispatcher(
    dispatcher_id="audit-logger",
    dispatcher=log_audit,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreatedEvent"},
)

engine.register_dispatcher(
    dispatcher_id="email-sender",
    dispatcher=send_email,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreatedEvent"},
)

engine.register_dispatcher(
    dispatcher_id="analytics-updater",
    dispatcher=update_analytics,
    category=EnumMessageCategory.EVENT,
    message_types={"UserCreatedEvent"},
)

# Register routes for each dispatcher
for dispatcher_id in ["audit-logger", "email-sender", "analytics-updater"]:
    engine.register_route(ModelDispatchRoute(
        route_id=f"user-events-{dispatcher_id}",
        topic_pattern="*.user.events.*",
        message_category=EnumMessageCategory.EVENT,
        dispatcher_id=dispatcher_id,
    ))

engine.freeze()

# All three dispatchers execute for UserCreatedEvent
result = await engine.dispatch("dev.user.events.v1", user_created_envelope)
# result.outputs = ["dev.email.sent.v1", "dev.analytics.v1", "dev.metrics.v1"]
# result.output_count = 3
```

### Using Protocol-Based Dispatchers

```python
from omnibase_infra.runtime import ProtocolMessageDispatcher
from omnibase_core.enums import EnumNodeKind

class OrderEventDispatcher:
    """Protocol-compliant dispatcher for order events."""

    @property
    def dispatcher_id(self) -> str:
        return "order-event-dispatcher"

    @property
    def category(self) -> EnumMessageCategory:
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        return {"OrderCreated", "OrderUpdated", "OrderCancelled"}

    @property
    def node_kind(self) -> EnumNodeKind:
        return EnumNodeKind.REDUCER

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelDispatchResult:
        try:
            # Process order event
            order = envelope.payload
            await self._process_order(order)

            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.SUCCESS,
                topic=envelope.topic,
                dispatcher_id=self.dispatcher_id,
                outputs=["dev.order.processed.v1"],
            )
        except Exception as e:
            return ModelDispatchResult(
                dispatch_id=uuid4(),
                status=EnumDispatchStatus.HANDLER_ERROR,
                topic=envelope.topic,
                dispatcher_id=self.dispatcher_id,
                error_message=str(e),
            )

# Register using DispatcherRegistry (alternative to MessageDispatchEngine)
from omnibase_infra.runtime import DispatcherRegistry

registry = DispatcherRegistry()
registry.register_dispatcher(OrderEventDispatcher())
registry.freeze()
```

## Validation Thresholds

The Message Dispatch Engine models contribute to infrastructure validation thresholds:

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Union types (INFRA_MAX_UNIONS) | ~350 | <200 | ~148 from dispatch models |
| Pattern violations | Exempted | Strict | See exempted_patterns list |

### Documented Exemptions

The following pattern exemptions are documented in `infra_validators.py`:

1. **KafkaEventBus** (14 methods, 10 __init__ params)
   - Event bus pattern: lifecycle, pub/sub, circuit breaker, protocol compatibility

2. **RuntimeHostProcess** (11+ methods, 6+ params)
   - Central coordinator: lifecycle, message handling, graceful shutdown

3. **PolicyRegistry** (many methods)
   - Domain registry pattern: CRUD, query, lifecycle operations

4. **MixinNodeIntrospection** (many methods)
   - Introspection mixin: capability discovery, caching, publishing

5. **ExecutionShapeValidator** (many methods)
   - AST analysis: handler detection, return analysis, violation detection

### Strict Mode Timeline

```
Status: DISABLED (INFRA_PATTERNS_STRICT = False)
Target Re-enable Date: 2026-03-01 (Q1 2026)

Prerequisites:
1. Complete OMN-934 (this ticket)
2. Complete H1 Legacy Migration
3. Reduce INFRA_MAX_UNIONS to <200
4. Validate all components pass or have exemptions

Review Cadence: Monthly until re-enabled
```

## Models Reference

### ModelDispatchRoute

```python
from omnibase_infra.models.dispatch import ModelDispatchRoute
from omnibase_infra.enums import EnumMessageCategory

route = ModelDispatchRoute(
    route_id="user-events-route",      # Unique identifier
    topic_pattern="*.user.events.*",   # Glob pattern
    message_category=EnumMessageCategory.EVENT,
    dispatcher_id="user-dispatcher",   # Reference to registered dispatcher
    message_type="UserCreatedEvent",   # Optional type filter (None = all)
    priority=100,                      # Higher = matched first
    enabled=True,                      # Can be disabled
    description="Routes user events", # Human-readable description
    correlation_id=uuid4(),            # Optional for tracing
    metadata={"team": "users"},        # Optional metadata
)

# Pattern matching
route.matches_topic("dev.user.events.v1")  # True
route.matches(
    "dev.user.events.v1",
    EnumMessageCategory.EVENT,
    "UserCreatedEvent"
)  # True
```

### ModelDispatchResult

```python
from omnibase_infra.models.dispatch import ModelDispatchResult
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_core.enums import EnumCoreErrorCode

result = ModelDispatchResult(
    dispatch_id=uuid4(),                          # Unique dispatch ID
    status=EnumDispatchStatus.SUCCESS,            # Dispatch status
    route_id="user-events-route",                 # Matched route
    dispatcher_id="user-dispatcher",              # Executed dispatcher(s)
    topic="dev.user.events.v1",                   # Target topic
    message_category=EnumMessageCategory.EVENT,   # Message category
    message_type="UserCreatedEvent",              # Message type
    duration_ms=45.2,                             # Duration in ms
    started_at=datetime.now(UTC),                 # Start timestamp
    completed_at=datetime.now(UTC),               # End timestamp
    outputs=["dev.notification.v1"],              # Output topics
    output_count=1,                               # Number of outputs
    error_message=None,                           # Error message (if failed)
    error_code=None,                              # Error code (if failed)
    correlation_id=uuid4(),                       # Correlation ID
    trace_id=uuid4(),                             # Trace ID
    span_id=uuid4(),                              # Span ID
)

# Status helpers
result.is_successful()    # True if SUCCESS
result.is_error()         # True if any error status
result.requires_retry()   # True if retriable failure
result.is_terminal()      # True if completed (success or failure)
```

### EnumDispatchStatus

```python
from omnibase_infra.enums import EnumDispatchStatus

class EnumDispatchStatus(str, Enum):
    SUCCESS = "success"           # Dispatch completed successfully
    NO_HANDLER = "no_handler"     # No dispatcher found for message
    HANDLER_ERROR = "handler_error"  # Dispatcher raised exception
    INVALID_MESSAGE = "invalid_message"  # Message validation failed
    TIMEOUT = "timeout"           # Dispatch timed out
    ROUTED = "routed"             # Message routed (intermediate state)
```

### ModelDispatchMetrics

```python
from omnibase_infra.models.dispatch import ModelDispatchMetrics

metrics = ModelDispatchMetrics()

# Record a dispatch (returns new instance - copy-on-write pattern)
metrics = metrics.record_dispatch(
    duration_ms=45.2,
    success=True,
    category=EnumMessageCategory.EVENT,
    dispatcher_id="user-dispatcher",
    topic="dev.user.events.v1",
)

# Access computed properties
metrics.avg_latency_ms   # Average latency
metrics.success_rate     # Success rate (0.0 to 1.0)
metrics.error_rate       # Error rate (0.0 to 1.0)

# Per-dispatcher metrics
dispatcher_metrics = metrics.get_dispatcher_metrics("user-dispatcher")
```

## Related Documentation

- **Implementation**: `src/omnibase_infra/runtime/message_dispatch_engine.py`
- **Dispatcher Registry**: `src/omnibase_infra/runtime/dispatcher_registry.py`
- **Runtime Host**: `src/omnibase_infra/runtime/runtime_host_process.py`
- **Validation**: `src/omnibase_infra/validation/infra_validators.py`
- **Circuit Breaker**: `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md`
- **Error Patterns**: `CLAUDE.md` - "Error Recovery Patterns" section
- **Dispatcher Resilience**: `CLAUDE.md` - "Dispatcher Resilience Pattern" section
