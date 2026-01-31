> **Navigation**: [Home](../index.md) > [Reference](../reference/README.md) > Event Bus API

# Event Bus API Reference

This document provides API reference for the ONEX event bus subsystem, including adapters, protocols, and implementation details.

## Overview

The ONEX event bus provides asynchronous event publishing and subscription capabilities. The architecture follows a layered approach:

| Layer          | Component                            | Purpose                              |
| -------------- | ------------------------------------ | ------------------------------------ |
| Protocol       | `ProtocolEventPublisher`             | SPI-defined interface for publishing |
| Adapter        | `AdapterProtocolEventPublisherKafka` | Production Kafka implementation      |
| Implementation | `EventBusKafka`                      | Low-level Kafka producer/consumer    |

---

## AdapterProtocolEventPublisherKafka

Production-grade adapter implementing `ProtocolEventPublisher` from `omnibase_spi`. Bridges the SPI protocol to `EventBusKafka` for production event publishing.

**Module**: `omnibase_infra.event_bus.adapters.adapter_protocol_event_publisher_kafka`

**Parent Ticket**: OMN-1764

### Purpose

This adapter provides a standard interface for event publishing while delegating resilience (circuit breaker, retry, backoff) to the underlying `EventBusKafka`. It implements the `ProtocolEventPublisher` protocol from `omnibase_spi`, enabling consistent event publishing across the ONEX infrastructure.

### Relationship to ProtocolEventPublisher

The adapter implements all methods required by `ProtocolEventPublisher`:

| Protocol Method  | Adapter Implementation                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------- |
| `publish()`      | Builds `ModelEventEnvelope`, serializes to JSON, delegates to `EventBusKafka.publish()` |
| `get_metrics()`  | Returns `ModelPublisherMetrics` with circuit breaker state from underlying bus          |
| `close()`        | Marks adapter closed, stops underlying `EventBusKafka`                                  |

### Constructor

```python
def __init__(
    self,
    bus: EventBusKafka,
    service_name: str = "kafka-publisher",
    instance_id: str | None = None,
) -> None
```

| Parameter      | Type            | Default             | Description                                                                  |
| -------------- | --------------- | ------------------- | ---------------------------------------------------------------------------- |
| `bus`          | `EventBusKafka` | (required)          | The EventBusKafka instance to bridge to. Must be started before publishing.  |
| `service_name` | `str`           | `"kafka-publisher"` | Service name included in envelope metadata for tracing.                      |
| `instance_id`  | `str \| None`   | `None`              | Instance identifier. Defaults to a generated UUID if not provided.           |

### Methods

#### publish()

```python
async def publish(
    self,
    event_type: str,
    payload: JsonType,
    correlation_id: str | None = None,
    causation_id: str | None = None,
    metadata: dict[str, ContextValue] | None = None,
    topic: str | None = None,
    partition_key: str | None = None,
) -> bool
```

Publish an event with canonical `ModelEventEnvelope` serialization.

| Parameter        | Type                              | Default      | Description                                                            |
| ---------------- | --------------------------------- | ------------ | ---------------------------------------------------------------------- |
| `event_type`     | `str`                             | (required)   | Fully-qualified event type (e.g., `"omninode.user.event.created.v1"`). |
| `payload`        | `JsonType`                        | (required)   | Event payload data (dict, list, or primitive JSON types).              |
| `correlation_id` | `str \| None`                     | `None`       | Correlation ID for request tracing. Converted to UUID.                 |
| `causation_id`   | `str \| None`                     | `None`       | Causation ID for event sourcing chains. Stored in metadata tags.       |
| `metadata`       | `dict[str, ContextValue] \| None` | `None`       | Additional metadata as context values.                                 |
| `topic`          | `str \| None`                     | `None`       | Explicit topic override. When `None`, uses `event_type` as topic.      |
| `partition_key`  | `str \| None`                     | `None`       | Partition key for message ordering. Encoded to UTF-8 bytes.            |

**Returns**: `bool` - `True` if published successfully, `False` otherwise.

**Raises**: `InfraUnavailableError` if adapter has been closed.

#### get_metrics()

```python
async def get_metrics(self) -> JsonType
```

Get publisher metrics including circuit breaker status from underlying bus.

**Returns**: Dictionary with all metrics (see Metrics section below).

#### reset_metrics()

```python
async def reset_metrics(self) -> None
```

Reset all publisher metrics to initial values. Useful for test isolation.

**Note**: This is an async method. Does NOT affect the closed state of the adapter.

#### close()

```python
async def close(self, timeout_seconds: float = 30.0) -> None
```

Close the publisher and release resources.

| Parameter         | Type    | Default | Description                     |
| ----------------- | ------- | ------- | ------------------------------- |
| `timeout_seconds` | `float` | `30.0`  | Timeout for cleanup operations. |

After closing, any calls to `publish()` will raise `InfraUnavailableError`.

---

### Usage Example

```python
from omnibase_infra.event_bus import EventBusKafka
from omnibase_infra.event_bus.adapters import AdapterProtocolEventPublisherKafka

# Create and start the bus
bus = EventBusKafka.from_env()
await bus.start()

# Create adapter
adapter = AdapterProtocolEventPublisherKafka(
    bus=bus,
    service_name="my-service",
)

# Publish events
success = await adapter.publish(
    event_type="user.created.v1",
    payload={"user_id": "123"},
    correlation_id="corr-abc",
)

# Check metrics
metrics = await adapter.get_metrics()
print(f"Published: {metrics['events_published']}")

# Cleanup
await adapter.close()
```

### Advanced Usage: Explicit Topic and Partition Key

```python
# Publish to explicit topic with partition key for ordering
success = await adapter.publish(
    event_type="order.placed.v1",
    payload={"order_id": "ord-456", "customer_id": "cust-789"},
    topic="orders.high-priority",  # Override default topic routing
    partition_key="cust-789",       # Ensure customer's events go to same partition
    correlation_id="corr-xyz",
    causation_id="cmd-123",         # Link to originating command
)
```

---

### Metrics

The adapter tracks publishing statistics via `ModelPublisherMetrics`:

| Metric                   | Type    | Description                                                                              |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `events_published`       | `int`   | Total count of successfully published events.                                            |
| `events_failed`          | `int`   | Total count of failed publish attempts.                                                  |
| `events_sent_to_dlq`     | `int`   | Always 0 - publish path does not use DLQ.                                                |
| `total_publish_time_ms`  | `float` | Cumulative publish time in milliseconds.                                                 |
| `avg_publish_time_ms`    | `float` | Average publish latency (computed from total/count).                                     |
| `circuit_breaker_opens`  | `int`   | Count of circuit breaker open events from underlying bus.                                |
| `retries_attempted`      | `int`   | Total retry attempts from underlying bus (if available).                                 |
| `circuit_breaker_status` | `str`   | Current circuit breaker state from underlying bus (`"closed"`, `"open"`, `"half_open"`). |
| `current_failures`       | `int`   | Current consecutive failure count.                                                       |

---

### Design Decisions

#### 1. Delegates Resilience to EventBusKafka (No Double Circuit Breaker)

The adapter does NOT implement its own circuit breaker. Resilience (circuit breaker, retry with exponential backoff, connection pooling) is delegated to the underlying `EventBusKafka`.

**Rationale**: Prevents "double circuit breaker" anti-pattern where two independent circuit breakers could interact unpredictably. The `EventBusKafka` already has comprehensive resilience built in.

#### 2. Publish Returns bool - Exceptions Are Caught, Not Propagated

All exceptions during publish are caught, logged, and result in `False` being returned. No exceptions propagate to the caller (except `InfraUnavailableError` for closed adapter).

**Rationale**: Allows callers to implement their own retry/fallback logic without needing to handle infrastructure-specific exception types. The metrics track failure counts for observability.

#### 3. Topic Routing: Explicit topic > event_type

Topic selection follows this precedence:
1. If `topic` parameter is provided, use it directly (explicit override)
2. Otherwise, derive topic from `event_type` (default routing)

**Rationale**: Provides flexibility for advanced routing scenarios while maintaining simple defaults.

#### 4. DLQ Metric Always 0 (Publish Path)

The `events_sent_to_dlq` metric is always 0 for this adapter because the publish path does not use a dead letter queue. DLQ is a consumer-side concept for handling messages that cannot be processed.

#### 5. Causation ID Stored in Metadata Tags

Since `ModelEventEnvelope` does not have a dedicated `causation_id` field, the adapter stores it in `metadata.tags["causation_id"]`. This preserves full correlation tracking for event sourcing chains.

#### 6. Partition Key Encoding

The `partition_key` is encoded to UTF-8 bytes as per the SPI specification. This ensures consistent partitioning behavior across different Kafka client implementations.

---

### Error Handling

| Scenario                       | Behavior                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| Publish succeeds               | Returns `True`, increments `events_published`                |
| Publish fails (any exception)  | Returns `False`, increments `events_failed`, logs exception  |
| Adapter closed                 | Raises `InfraUnavailableError("Publisher has been closed")`  |
| Invalid correlation_id format  | Generates new UUID, logs warning with original value         |
| Close fails                    | Logs warning, continues (best-effort cleanup)                |

---

### Related Documentation

- [Event Bus Shapes (As-Is)](../as_is/04_EVENT_BUS_SHAPES.md) - Event bus interface overview
- [Event Bus Coverage Report](../validation/EVENT_BUS_COVERAGE_REPORT.md) - Feature validation status
- [Circuit Breaker Implementation](../patterns/circuit_breaker_implementation.md) - Resilience patterns
- [Correlation ID Tracking](../patterns/correlation_id_tracking.md) - Tracing patterns
- [DLQ Replay Runbook](../operations/DLQ_REPLAY_RUNBOOK.md) - Dead letter queue operations
