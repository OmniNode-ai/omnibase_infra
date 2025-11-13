# ONEX EventBus Quick Reference

**3-Second Validation**: Is your node using EventBus correctly?

**Correlation ID**: parallel-solve-patterns-001

---

## Quick Validation Checklist

**For Orchestrator Nodes**:
```bash
✓ Has self.event_bus = container.get_service("event_bus")
✓ Has await self.event_bus.publish_action_event(...)
✓ Has await self.event_bus.wait_for_completion(...)
✓ Has fallback to legacy mode when EventBus unavailable
✓ Uses kafka_client.publish_with_envelope() for events
✗ NO raw Kafka imports (from kafka import ...)
✗ NO TODO comments for event integration
✗ NO HTTP endpoints for event publishing
```

**For Reducer/Effect Nodes**:
```bash
✓ Has self._pending_event_intents: list[Any] = []
✓ Generates ModelIntent(intent_type=EnumIntentType.PUBLISH_EVENT.value, ...)
✓ Returns intents in output model
✓ Has get_pending_event_intents() method
✓ Uses kafka_client.publish_with_envelope() for optional publishing
✗ NO direct I/O operations in reduction logic
✗ NO raw Kafka imports (from kafka import ...)
✗ NO TODO comments for event integration
```

---

## DO vs DON'T Examples

### Publishing Events

#### ✅ DO (Orchestrator)

```python
async def _publish_event(self, event_type: EnumWorkflowEvent, data: dict[str, Any]):
    """Publish event using KafkaClient with OnexEnvelopeV1 wrapping."""

    topic_name = event_type.get_topic_name(namespace=self.default_namespace)

    if self.kafka_client and self.kafka_client.is_connected:
        await self.kafka_client.publish_with_envelope(
            event_type=event_type.value,
            source_node_id=self.node_id,
            payload=data,
            topic=topic_name,
            correlation_id=data.get("workflow_id"),
            metadata={"event_category": "workflow_orchestration"},
        )
```

#### ✅ DO (Reducer/Effect)

```python
async def _publish_event(self, event_type: EnumReducerEvent, data: dict[str, Any]):
    """Generate intent + optional immediate publishing."""

    # Generate intent (ALWAYS)
    intent = ModelIntent(
        intent_type=EnumIntentType.PUBLISH_EVENT.value,
        target="event_bus",
        payload={"event_type": event_type.value, "data": data},
        priority=0,
    )
    self._pending_event_intents.append(intent)

    # Optional immediate publishing
    if self.kafka_client and self.kafka_client.is_connected:
        await self.kafka_client.publish_with_envelope(...)
```

#### ❌ DON'T

```python
# WRONG: Raw Kafka producer
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
producer.send('my-topic', event_data)  # NO!

# WRONG: TODO comments
# TODO: Implement Kafka integration  # NO!

# WRONG: HTTP endpoint
@app.post("/publish-event")
async def publish_event(data):
    await kafka_producer.send(data)  # NO!
```

---

### Topic Routing

#### ✅ DO

```python
# Use EnumEvent.get_topic_name()
topic_name = event_type.get_topic_name(namespace=self.default_namespace)

await self.kafka_client.publish_with_envelope(
    topic=topic_name,  # Dynamic topic from enum
    ...
)
```

#### ❌ DON'T

```python
# WRONG: Hardcoded topic
topic = "omninode.bridge.workflow.events"  # NO!
await self.kafka_client.publish(topic, data)
```

---

### EventBus Initialization (Orchestrator)

#### ✅ DO

```python
def __init__(self, container: ModelContainer):
    super().__init__(container)

    # Get EventBus from container
    self.event_bus = container.get_service("event_bus")
    if self.event_bus is None:
        from ....services.event_bus import EventBusService
        self.event_bus = EventBusService(
            kafka_client=self.kafka_client,
            node_id=self.node_id,
            namespace=self.default_namespace,
        )
        container.register_service("event_bus", self.event_bus)
```

#### ❌ DON'T

```python
def __init__(self):
    # WRONG: No container integration
    self.event_bus = EventBusService(...)  # NO!

    # WRONG: No graceful degradation
    # Just crash if EventBus is unavailable
```

---

### Workflow Coordination (Orchestrator)

#### ✅ DO

```python
async def execute_orchestration(self, contract):
    # Check if event-driven coordination is available
    if self.event_bus and self.event_bus.is_initialized:
        return await self._execute_event_driven_workflow(contract)
    else:
        # Fallback to legacy mode
        emit_log_event(LogLevel.WARNING, "EventBus unavailable - using legacy mode")
        return await self._execute_legacy_workflow(contract)
```

#### ❌ DON'T

```python
async def execute_orchestration(self, contract):
    # WRONG: No fallback, crashes if EventBus unavailable
    return await self._execute_event_driven_workflow(contract)  # NO!
```

---

### Intent Collection (Reducer/Effect)

#### ✅ DO

```python
async def execute_reduction(self, contract) -> ModelReducerOutputState:
    # ... aggregation logic ...

    # Collect all intents
    intents = []

    # Persistence intents
    intents.append(ModelIntent(intent_type=EnumIntentType.PERSIST_STATE.value, ...))

    # FSM intents
    fsm_intents = self._fsm_manager.get_pending_intents()
    intents.extend(fsm_intents)

    # Event publishing intents
    event_intents = self.get_pending_event_intents()
    intents.extend(event_intents)

    # Return with intents
    return ModelReducerOutputState(
        aggregations=dict(aggregated_data),
        intents=intents,  # CRITICAL!
    )
```

#### ❌ DON'T

```python
async def execute_reduction(self, contract) -> ModelReducerOutputState:
    # ... aggregation logic ...

    # WRONG: Forgot to collect intents
    return ModelReducerOutputState(
        aggregations=dict(aggregated_data),
        # Missing intents!  # NO!
    )
```

---

### Graceful Degradation

#### ✅ DO

```python
try:
    if self.kafka_client and self.kafka_client.is_connected:
        success = await self.kafka_client.publish_with_envelope(...)
        if success:
            emit_log_event(LogLevel.DEBUG, "Event published")
        else:
            emit_log_event(LogLevel.WARNING, "Event publish failed")
    else:
        emit_log_event(LogLevel.DEBUG, "Kafka unavailable, event logged")
except Exception as e:
    emit_log_event(LogLevel.WARNING, f"Error publishing event: {e}")
    # Continue execution - don't crash
```

#### ❌ DON'T

```python
# WRONG: No error handling, crashes if Kafka is down
await self.kafka_client.publish_with_envelope(...)  # NO!
```

---

### OnexEnvelopeV1 Wrapping

#### ✅ DO

```python
# Use publish_with_envelope() for automatic OnexEnvelopeV1 wrapping
await self.kafka_client.publish_with_envelope(
    event_type=event_type.value,
    source_node_id=self.node_id,
    payload=data,
    topic=topic_name,
    correlation_id=workflow_id,  # UUID for tracing
    metadata={
        "event_category": "workflow_orchestration",
        "node_type": "orchestrator",
    },
)
```

#### ❌ DON'T

```python
# WRONG: Raw data without envelope
await self.kafka_client.publish(topic, data)  # NO!

# WRONG: Manual envelope construction
envelope = {
    "envelope": {"version": "1.0", ...},
    "payload": data,
}
await self.kafka_client.publish(topic, envelope)  # NO!
```

---

## Pattern Selection (1-Second Guide)

| Node Type | Pattern | Key Methods |
|-----------|---------|-------------|
| **Orchestrator** | Direct EventBus | `event_bus.publish_action_event()`, `event_bus.wait_for_completion()` |
| **Reducer** | Intent-Based | `ModelIntent(...)`, `get_pending_event_intents()` |
| **Effect** | Intent-Based | `ModelIntent(...)`, `get_pending_event_intents()` |
| **Compute** | None | No event publishing (pure computation) |

---

## Common Mistakes

### Mistake 1: Forgetting to Initialize EventBus in Orchestrator

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'publish_action_event'`

**Fix**:
```python
# In __init__:
self.event_bus = container.get_service("event_bus")
if self.event_bus is None:
    self.event_bus = EventBusService(...)
    container.register_service("event_bus", self.event_bus)
```

---

### Mistake 2: Forgetting to Return Intents in Reducer

**Symptom**: Intents are generated but never executed

**Fix**:
```python
# In execute_reduction:
event_intents = self.get_pending_event_intents()
intents.extend(event_intents)

return ModelReducerOutputState(
    ...,
    intents=intents,  # Don't forget!
)
```

---

### Mistake 3: Hardcoding Topic Names

**Symptom**: Events go to wrong topics when namespace changes

**Fix**:
```python
# Use EnumEvent.get_topic_name() instead of hardcoded strings
topic_name = event_type.get_topic_name(namespace=self.default_namespace)
```

---

### Mistake 4: Missing Correlation ID

**Symptom**: Cannot trace events across workflow

**Fix**:
```python
# Always pass correlation_id
await self.kafka_client.publish_with_envelope(
    ...,
    correlation_id=workflow_id,  # UUID for tracing
)
```

---

### Mistake 5: No Graceful Degradation

**Symptom**: Node crashes when Kafka is unavailable

**Fix**:
```python
# Check availability before use
if self.kafka_client and self.kafka_client.is_connected:
    await self.kafka_client.publish_with_envelope(...)
else:
    emit_log_event(LogLevel.DEBUG, "Kafka unavailable, event logged")
```

---

## Reference Implementations

**Compliant Orchestrator**:
```
/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py
```

**Compliant Reducer**:
```
/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/reducer/v1_0_0/node.py
```

---

## Need More Details?

See comprehensive documentation:
- `docs/patterns/onex/EVENTBUS_INTEGRATION_PATTERNS.md` - Full patterns, anti-patterns, migration guide, testing

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Correlation ID**: parallel-solve-patterns-001
