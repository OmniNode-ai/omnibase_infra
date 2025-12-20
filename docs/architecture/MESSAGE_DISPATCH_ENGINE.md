# Message Dispatch Engine Architecture

## Overview

The Message Dispatch Engine (`MessageDispatchEngine`) is a runtime routing engine for
message dispatching based on topic category and message type. It routes incoming messages
to registered dispatchers and collects dispatcher outputs for publishing.

**Implementation**: `src/omnibase_infra/runtime/message_dispatch_engine.py`
**Ticket**: OMN-934

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Pure Routing** | Routes messages to dispatchers, no workflow inference |
| **Deterministic** | Same input always produces same dispatcher selection |
| **Fan-out Support** | Multiple dispatchers can process the same message type |
| **Freeze-After-Init** | Thread-safe after registration phase completes |
| **Observable** | Structured logging and comprehensive metrics |

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
User/Caller          MessageDispatchEngine         Dispatcher(s)
     |                        |                         |
     | dispatch(topic, env)   |                         |
     |----------------------->|                         |
     |                        |                         |
     |                        | 1. Parse topic category |
     |                        |------------------------>|
     |                        |                         |
     |                        | 2. Validate envelope    |
     |                        |------------------------>|
     |                        |                         |
     |                        | 3. Find matching        |
     |                        |    dispatchers          |
     |                        |------------------------>|
     |                        |                         |
     |                        | 4. Execute dispatcher   |
     |                        |------------------------>|
     |                        |                         |
     |                        |    DispatcherOutput     |
     |                        |<------------------------|
     |                        |                         |
     |                        | (repeat for fan-out)    |
     |                        |                         |
     |                        | 5. Aggregate outputs    |
     |                        |------------------------>|
     |                        |                         |
     | ModelDispatchResult    |                         |
     |<-----------------------|                         |
     |                        |                         |
```

## Registration Phase

### Route Registration

Routes define how messages are matched to dispatchers based on topic pattern,
message category, and optionally message type.

```python
engine.register_route(ModelDispatchRoute(
    route_id="order-events",
    topic_pattern="*.order.events.*",
    message_category=EnumMessageCategory.EVENT,
    dispatcher_id="order-dispatcher",
))
```

### Dispatcher Registration

Dispatchers process messages that match their category and (optionally) message type.

```python
async def process_user_event(envelope: ModelEventEnvelope[object]) -> str | None:
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

#### Envelope Typing Pattern

Dispatchers use `ModelEventEnvelope[object]` instead of `Any` for envelope parameters:

- **`ModelEventEnvelope[object]`**: For generic dispatchers that accept any payload type.
  The dispatch engine routes based on topic/category/message_type, not payload shape.
- **`ModelEventEnvelope[SpecificType]`**: For concrete implementations that know the
  exact payload type (e.g., `ModelEventEnvelope[UserCreatedEvent]`).

Using `object` instead of `Any`:
- Satisfies ONEX "no Any types" coding guideline
- Explicitly states "any Python object" with clearer intent
- Same runtime behavior as `Any` but with documented semantics

See `CLAUDE.md` section "Envelope Typing" for the complete typing guidelines.

### Freeze Pattern

The engine follows a **freeze-after-init** pattern for thread safety:

1. **Registration Phase** (single-threaded): Register routes and dispatchers
2. **Freeze**: Call `freeze()` to prevent further modifications
3. **Dispatch Phase** (multi-threaded safe): Route messages to dispatchers

```python
engine = MessageDispatchEngine()
engine.register_dispatcher("d1", dispatcher, EnumMessageCategory.EVENT)
engine.register_route(route)
engine.freeze()  # Validates and freezes
# Now thread-safe for concurrent dispatch
```

## Dispatch Phase

### Dispatch Flow

1. **Parse Topic**: Extract message category from topic string
2. **Validate**: Ensure envelope is valid
3. **Match**: Find all dispatchers matching category + message type
4. **Execute**: Run dispatchers (fan-out for multiple matches)
5. **Collect**: Aggregate outputs and errors
6. **Return**: Build `ModelDispatchResult`

### Dispatcher Output Types

Dispatchers can return:
- `str`: A single output topic
- `list[str]`: Multiple output topics
- `None`: No output topics to publish

### Error Handling

- Dispatcher exceptions are caught and sanitized (no credential leakage)
- Dispatch continues to other dispatchers even if one fails
- Errors are aggregated in `ModelDispatchResult.error_message`

## Thread Safety

### Metrics Lock

Structured metrics updates are protected by `_metrics_lock`:

```python
with self._metrics_lock:
    self._structured_metrics = self._structured_metrics.record_dispatch(
        duration_ms=duration_ms,
        success=True,
        category=topic_category,
        topic=topic,
    )
```

### Lock Hold Time Minimization

The engine minimizes lock hold time by:
1. Reading existing metrics under lock
2. Computing updates outside lock
3. Updating atomically with `model_copy()` under lock

### Sync Dispatcher Execution

Sync dispatchers execute via `run_in_executor()`:

```
WARNING: Sync dispatchers MUST be non-blocking (< 100ms execution).
Blocking dispatchers can exhaust the thread pool, causing:
- Starvation of other sync dispatchers
- Delayed async dispatcher scheduling
- Potential deadlocks under high load

For blocking I/O operations, use async dispatchers instead.
```

## Observability

### Structured Logging

| Level | Events |
|-------|--------|
| **INFO** | Dispatch start/complete with topic, category, dispatcher count |
| **DEBUG** | Dispatcher execution details, routing decisions |
| **WARNING** | No dispatchers found, category mismatches |
| **ERROR** | Dispatcher exceptions, validation failures |

### Structured Metrics

Access via `get_structured_metrics()`:

```python
metrics = engine.get_structured_metrics()
print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
```

### Per-Dispatcher Metrics

```python
metrics = engine.get_dispatcher_metrics("user-event-dispatcher")
if metrics:
    print(f"Executions: {metrics.execution_count}")
    print(f"Error rate: {metrics.error_rate:.1%}")
```

## Resilience Patterns

### Dispatcher-Owned Resilience

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap
dispatchers with circuit breakers.

**Rationale**:
- Separation of concerns: Each dispatcher knows its specific failure modes
- Transport-specific tuning: Kafka dispatchers need different thresholds than HTTP
- No hidden behavior: Engine users see exactly what resilience each dispatcher provides
- Composability: Dispatchers can combine circuit breakers with retry, backoff, degradation

See `CLAUDE.md` section "Dispatcher Resilience Pattern" for implementation guidance.

### Error Sanitization

Exception messages are sanitized before storage to prevent credential leakage:

```python
# Sensitive patterns that trigger redaction
_SENSITIVE_PATTERNS = (
    "password", "secret", "token", "api_key",
    "credential", "bearer", "private_key",
    "connection_string", "postgres://", "kafka://", ...
)
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
   - AST analysis: dispatcher detection, return analysis, violation detection

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

## Related Documentation

- **Implementation**: `src/omnibase_infra/runtime/message_dispatch_engine.py`
- **Dispatcher Registry**: `src/omnibase_infra/runtime/dispatcher_registry.py`
- **Runtime Host**: `src/omnibase_infra/runtime/runtime_host_process.py`
- **Validation**: `src/omnibase_infra/validation/infra_validators.py`
- **Circuit Breaker**: `docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md`
- **Error Patterns**: `CLAUDE.md` - "Error Recovery Patterns" section

## Models Reference

### ModelDispatchRoute

```python
ModelDispatchRoute(
    route_id: str,           # Unique identifier
    topic_pattern: str,      # Glob pattern (e.g., "*.user.events.*")
    message_category: EnumMessageCategory,
    dispatcher_id: str,      # Reference to registered dispatcher
    message_type: str | None = None,  # Optional type filter
    enabled: bool = True,
)
```

### ModelDispatchResult

```python
ModelDispatchResult(
    dispatch_id: UUID,
    status: EnumDispatchStatus,
    topic: str,
    message_category: EnumMessageCategory | None,
    message_type: str | None,
    duration_ms: float,
    outputs: list[str] | None,  # Output topics from dispatchers
    output_count: int,
    error_message: str | None,
    error_code: EnumCoreErrorCode | None,
    correlation_id: UUID | None,
    trace_id: UUID | None,
    span_id: UUID | None,
)
```

### EnumDispatchStatus

```python
class EnumDispatchStatus(str, Enum):
    SUCCESS = "success"
    """Message was successfully routed, dispatched, and outputs published."""

    ROUTED = "routed"
    """Message was successfully routed to a dispatcher (pending execution)."""

    NO_DISPATCHER = "no_dispatcher"
    """No dispatcher was registered for the message type/topic."""

    DISPATCHER_ERROR = "dispatcher_error"
    """Dispatcher execution failed with an exception."""

    TIMEOUT = "timeout"
    """Dispatcher execution exceeded the configured timeout."""

    INVALID_MESSAGE = "invalid_message"
    """Message failed validation before dispatch."""

    PUBLISH_FAILED = "publish_failed"
    """Dispatcher succeeded but output publishing failed."""

    SKIPPED = "skipped"
    """Message was intentionally skipped (e.g., filtered, deduplicated)."""
```

The enum also provides helper methods:
- `is_terminal()` - Returns `True` for all statuses except `ROUTED`
- `is_successful()` - Returns `True` only for `SUCCESS`
- `is_error()` - Returns `True` for `NO_DISPATCHER`, `DISPATCHER_ERROR`, `TIMEOUT`, `INVALID_MESSAGE`, `PUBLISH_FAILED`
- `requires_retry()` - Returns `True` for transient failures: `TIMEOUT`, `PUBLISH_FAILED`
