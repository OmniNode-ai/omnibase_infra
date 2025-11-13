# Event Publishing Patterns - Quick Reference

**Version:** 1.0.0
**Date:** 2025-11-05

---

## Quick Start

### 1. Generate All Event Methods

```python
from omninode_bridge.codegen.patterns import generate_event_publishing_methods

code = generate_event_publishing_methods(
    node_type="orchestrator",
    service_name="orchestrator",
    operations=["orchestration", "routing"],
    include_state_events=True,
    include_metric_events=True,
)

print(code)  # Complete event publishing methods
```

### 2. Generate Single Event

```python
from omninode_bridge.codegen.patterns import generate_operation_started_event

code = generate_operation_started_event(
    service_name="orchestrator",
    operation="orchestration"
)

print(code)  # Single _publish_orchestration_started_event method
```

### 3. Get Event Catalog

```python
from omninode_bridge.codegen.patterns import get_event_type_catalog

catalog = get_event_type_catalog(
    node_type="orchestrator",
    operations=["orchestration", "routing"],
)

print(catalog)  # Dictionary of all event types
```

---

## Event Patterns

### Operation Lifecycle Events

**Pattern:** `{node_type}.{operation}.{lifecycle_stage}`

**Lifecycle Stages:**
- `started` - Operation initiated
- `completed` - Operation succeeded
- `failed` - Operation failed

**Example:**
```
orchestrator.orchestration.started
orchestrator.orchestration.completed
orchestrator.orchestration.failed
```

### State Events

**Pattern:** `{node_type}.state.changed`

**Example:**
```
orchestrator.state.changed
reducer.state.changed
```

### Metric Events

**Pattern:** `{node_type}.metric.recorded`

**Example:**
```
orchestrator.metric.recorded
reducer.metric.recorded
```

---

## Generated Method Signatures

### Operation Started

```python
async def _publish_{operation}_started_event(
    self,
    correlation_id: UUID,
    input_data: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
```

### Operation Completed

```python
async def _publish_{operation}_completed_event(
    self,
    correlation_id: UUID,
    result_data: dict[str, Any],
    execution_time_ms: float,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
```

### Operation Failed

```python
async def _publish_{operation}_failed_event(
    self,
    correlation_id: UUID,
    error: Exception,
    execution_time_ms: float,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
```

### State Changed

```python
async def _publish_state_changed_event(
    self,
    correlation_id: UUID,
    old_state: str,
    new_state: str,
    transition_reason: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
```

### Metric Recorded

```python
async def _publish_metric_recorded_event(
    self,
    metric_name: str,
    metric_value: float,
    metric_unit: str,
    correlation_id: Optional[UUID] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
```

---

## Usage in Node Classes

### Example: Orchestrator Node

```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    async def orchestrate(
        self,
        input_data: dict[str, Any]
    ) -> dict[str, Any]:
        correlation_id = uuid4()
        start_time = time.time()

        try:
            # Publish started event
            await self._publish_orchestration_started_event(
                correlation_id=correlation_id,
                input_data=input_data,
            )

            # Perform orchestration
            result = await self._perform_orchestration(input_data)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Publish completed event
            await self._publish_orchestration_completed_event(
                correlation_id=correlation_id,
                result_data=result,
                execution_time_ms=execution_time_ms,
            )

            return result

        except Exception as e:
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Publish failed event
            await self._publish_orchestration_failed_event(
                correlation_id=correlation_id,
                error=e,
                execution_time_ms=execution_time_ms,
            )

            raise
```

### Example: State Transition

```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    async def _transition_state(
        self,
        correlation_id: UUID,
        new_state: str,
        reason: str
    ) -> None:
        old_state = self.current_state

        # Update state
        self.current_state = new_state

        # Publish state changed event
        await self._publish_state_changed_event(
            correlation_id=correlation_id,
            old_state=old_state,
            new_state=new_state,
            transition_reason=reason,
        )
```

### Example: Metric Recording

```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    async def _record_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        correlation_id: Optional[UUID] = None
    ) -> None:
        # Publish metric recorded event
        await self._publish_metric_recorded_event(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit="ms",
            correlation_id=correlation_id,
            metadata={
                "node_id": self.node_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
```

---

## Event Catalog by Node Type

### Orchestrator (3 operations)

```json
{
  "operation_lifecycle": [
    "orchestrator.orchestration.started",
    "orchestrator.orchestration.completed",
    "orchestrator.orchestration.failed",
    "orchestrator.routing.started",
    "orchestrator.routing.completed",
    "orchestrator.routing.failed",
    "orchestrator.intelligence_query.started",
    "orchestrator.intelligence_query.completed",
    "orchestrator.intelligence_query.failed"
  ],
  "state_events": [
    "orchestrator.state.changed"
  ],
  "metric_events": [
    "orchestrator.metric.recorded"
  ]
}
```

**Total:** 11 events

### Reducer (2 operations)

```json
{
  "operation_lifecycle": [
    "reducer.aggregation.started",
    "reducer.aggregation.completed",
    "reducer.aggregation.failed",
    "reducer.state_snapshot.started",
    "reducer.state_snapshot.completed",
    "reducer.state_snapshot.failed"
  ],
  "state_events": [
    "reducer.state.changed"
  ],
  "metric_events": [
    "reducer.metric.recorded"
  ]
}
```

**Total:** 8 events

---

## OnexEnvelopeV1 Compliance

All events are wrapped in ModelOnexEnvelopeV1:

```python
{
    "envelope_version": "1.0",
    "event_id": UUID,
    "event_type": str,
    "event_timestamp": datetime (ISO 8601),
    "source_node_id": str,
    "correlation_id": UUID | None,
    "payload": {
        # Event-specific data
    },
    "metadata": {
        # Event metadata
    }
}
```

---

## Required Dependencies

### Node Class Requirements

Your node class must have:

```python
class YourNode:
    node_id: str                    # Node identifier
    kafka_client: KafkaClient       # Kafka client instance
    default_namespace: str          # Namespace for topic routing
```

### Import Requirements

Generated code requires:

```python
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
```

**Note:** Fallback imports are included for testing without omnibase_core.

---

## Best Practices

### 1. Always Use Correlation IDs

```python
# ✅ Good
correlation_id = uuid4()
await self._publish_orchestration_started_event(
    correlation_id=correlation_id,
    input_data=input_data,
)

# ❌ Bad
await self._publish_orchestration_started_event(
    correlation_id=None,  # Missing correlation tracking
    input_data=input_data,
)
```

### 2. Capture Execution Time

```python
# ✅ Good
start_time = time.time()
result = await self._perform_operation()
execution_time_ms = (time.time() - start_time) * 1000

await self._publish_operation_completed_event(
    correlation_id=correlation_id,
    result_data=result,
    execution_time_ms=execution_time_ms,
)

# ❌ Bad
await self._publish_operation_completed_event(
    correlation_id=correlation_id,
    result_data=result,
    execution_time_ms=0,  # No actual measurement
)
```

### 3. Use Try/Except for Operations

```python
# ✅ Good
try:
    await self._publish_operation_started_event(...)
    result = await self._perform_operation()
    await self._publish_operation_completed_event(...)
    return result
except Exception as e:
    await self._publish_operation_failed_event(
        correlation_id=correlation_id,
        error=e,
        execution_time_ms=execution_time_ms,
    )
    raise

# ❌ Bad
result = await self._perform_operation()
await self._publish_operation_completed_event(...)
# No error handling or failed event
```

### 4. Include Metadata for Context

```python
# ✅ Good
await self._publish_orchestration_started_event(
    correlation_id=correlation_id,
    input_data=input_data,
    metadata={
        "source_service": "api_gateway",
        "request_id": "req-123",
        "user_id": "user-456",
    },
)

# ❌ Acceptable but less useful
await self._publish_orchestration_started_event(
    correlation_id=correlation_id,
    input_data=input_data,
    # No additional metadata
)
```

---

## Troubleshooting

### Issue: Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'omnibase_core'
```

**Solution:**
The generated code includes fallback imports. If omnibase_core is not available, a basic implementation will be used for testing.

### Issue: Kafka Client Not Connected

**Problem:**
```
Kafka unavailable, logging orchestration started event
```

**Solution:**
This is expected behavior when Kafka is unavailable. Events are logged instead of published. Ensure `self.kafka_client` is initialized and connected.

### Issue: Missing Node Attributes

**Problem:**
```python
AttributeError: 'YourNode' object has no attribute 'node_id'
```

**Solution:**
Ensure your node class has required attributes:
```python
self.node_id: str
self.kafka_client: KafkaClient
self.default_namespace: str
```

---

## API Reference

### generate_event_publishing_methods()

```python
def generate_event_publishing_methods(
    node_type: str,
    service_name: str,
    operations: list[str],
    include_state_events: bool = True,
    include_metric_events: bool = True,
) -> str:
```

**Parameters:**
- `node_type` - Node type identifier (e.g., "orchestrator", "reducer")
- `service_name` - Service name for Kafka topic routing
- `operations` - List of operations to generate events for
- `include_state_events` - Generate state change events (default: True)
- `include_metric_events` - Generate metric recording events (default: True)

**Returns:** Complete Python code with all event publishing methods

### get_event_type_catalog()

```python
def get_event_type_catalog(
    node_type: str,
    operations: list[str],
    include_state_events: bool = True,
    include_metric_events: bool = True,
) -> dict[str, list[str]]:
```

**Parameters:**
- `node_type` - Node type identifier
- `operations` - List of operations
- `include_state_events` - Include state events (default: True)
- `include_metric_events` - Include metric events (default: True)

**Returns:** Dictionary mapping event categories to event type lists

---

## Additional Resources

- **Full Implementation Report:** [EVENT_PUBLISHING_PATTERNS_IMPLEMENTATION_REPORT.md](./EVENT_PUBLISHING_PATTERNS_IMPLEMENTATION_REPORT.md)
- **Source Code:** `src/omninode_bridge/codegen/patterns/event_publishing.py`
- **OnexEnvelopeV1 Model:** `src/omninode_bridge/nodes/registry/v1_0_0/models/model_onex_envelope_v1.py`

---

**Last Updated:** 2025-11-05
**Version:** 1.0.0
