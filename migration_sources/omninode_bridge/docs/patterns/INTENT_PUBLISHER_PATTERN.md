# Intent Publisher Pattern - ONEX v2.0

## Overview

The Intent Publisher Pattern enables ONEX nodes to coordinate actions (like publishing events) without performing direct domain I/O, maintaining architectural boundaries while enabling necessary coordination.

## The Problem

In ONEX architecture, REDUCER nodes must stay pure - they perform aggregation logic but shouldn't perform I/O operations. However, reducers need to publish their aggregated results. This creates a tension:

- ❌ **Direct Publishing**: Violates REDUCER purity, couples to Kafka
- ❌ **No Publishing**: Aggregated results aren't available to other nodes
- ✅ **Intent Publishing**: Coordination I/O that maintains architectural boundaries

## The Solution

### Architecture

```
┌──────────────┐
│   REDUCER    │
│   (Pure)     │
│              │
│ 1. Aggregate │
│ 2. Build     │
│    Event     │
│ 3. Publish   │
│    INTENT    │───┐
└──────────────┘   │
                   ▼
            ┌─────────────┐
            │    Kafka    │
            │ (Intent     │
            │  Topic)     │
            └──────┬──────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Intent Executor │
          │                 │
          │ 4. Consume      │
          │    Intent       │
          │ 5. Execute via  │
          │    EFFECT       │
          └────────┬────────┘
                   │
                   ▼
            ┌─────────────┐
            │    Kafka    │
            │  (Domain    │
            │   Topic)    │
            └─────────────┘
```

### Key Principles

1. **Domain Logic Stays Pure**: Aggregation/computation has no I/O
2. **Coordination I/O is Explicit**: Intent publishing is clearly coordination
3. **Async Decoupling**: Intent execution happens independently
4. **Traceable**: All intents logged with correlation IDs

## Implementation

### 1. Add Subcontract Reference

```yaml
# contract.yaml
subcontracts:
  refs:
    - "./contracts/intent_publisher.yaml"

mixins:
  - "MixinIntentPublisher"
```

### 2. Use the Mixin

```python
from omninode_bridge.mixins import MixinIntentPublisher

class NodeCodegenMetricsReducer(NodeReducer, MixinIntentPublisher):
    def __init__(self, container):
        super().__init__(container)

        # Initialize mixin
        self._init_intent_publisher(container)

    async def execute_reduction(self, events):
        # Pure aggregation
        metrics_state = self._aggregate(events)

        # Build event (pure)
        event = self._build_metrics_event(metrics_state)

        # Publish intent (coordination I/O)
        await self.publish_event_intent(
            target_topic=TOPIC_METRICS_RECORDED,
            target_key=str(metrics_state.id),
            event=event
        )

        return metrics_state
```

### 3. Intent Executor (Future)

```python
class NodeIntentExecutor:
    """Executes intents from coordination topic."""

    async def run(self):
        async for envelope in self.consume(TOPIC_EVENT_PUBLISH_INTENT):
            intent = ModelEventPublishIntent(**envelope.payload)

            # Execute via EFFECT
            await self.kafka_effect.publish(
                topic=intent.target_topic,
                key=intent.target_key,
                value=intent.target_event_payload
            )
```

## Benefits

### ✅ Architectural Benefits

1. **ONEX Compliance**: Maintains REDUCER purity
2. **Separation of Concerns**: Domain vs coordination I/O
3. **Unidirectional Dependencies**: No circular dependencies
4. **Testability**: Pure logic testable without Kafka

### ✅ Operational Benefits

1. **Observability**: All intents visible on Kafka
2. **Retry Logic**: Intent execution can retry independently
3. **Monitoring**: Track intent success/failure rates
4. **Flexibility**: Swap Kafka for other brokers

### ✅ Development Benefits

1. **Clear Boundaries**: What's domain vs coordination
2. **Reusability**: Same intent, different executors
3. **Mockability**: Easy to mock intent publishing in tests
4. **Debuggability**: Intent flow traceable via correlation IDs

## Usage Guidelines

### When to Use

✅ **Use intent publishing when:**
- Node needs to publish events but wants to stay pure
- Event publishing is coordination, not domain logic
- Want decoupled event production from publishing
- Need traceable, auditable coordination

❌ **Don't use intent publishing when:**
- Node is already an EFFECT (direct I/O is its job)
- Event publishing IS the domain logic
- Synchronous execution required
- Overhead not justified

### Nodes That Should Use This

| Node Type | Use Intent Publisher? | Reason |
|-----------|----------------------|---------|
| **REDUCER** | ✅ Yes | Must stay pure, publish aggregated results |
| **COMPUTE** | ✅ Yes | Pure computation, publish computed results |
| **ORCHESTRATOR** | ✅ Maybe | For coordination events only |
| **EFFECT** | ❌ No | Direct I/O is its purpose |

## Event Schemas

### ModelEventPublishIntent

```python
{
    "intent_id": "uuid",
    "correlation_id": "uuid",
    "created_at": "2025-10-23T12:00:00Z",
    "created_by": "metrics_reducer_v1_0_0",

    "target_topic": "dev.omninode.metrics.v1",
    "target_key": "metrics-123",
    "target_event_type": "GENERATION_METRICS_RECORDED",
    "target_event_payload": {...},

    "priority": 5,
    "retry_policy": {...}
}
```

### ModelIntentExecutionResult

```python
{
    "intent_id": "uuid",
    "correlation_id": "uuid",
    "executed_at": "2025-10-23T12:00:01Z",
    "success": true,
    "error_message": null,
    "execution_duration_ms": 15.3
}
```

## Testing

### Unit Tests

```python
@pytest.mark.asyncio
async def test_reducer_publishes_intent():
    # Mock Kafka client
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)

    # Create reducer with mixin
    reducer = NodeCodegenMetricsReducer(container)

    # Execute reduction
    result = await reducer.execute_reduction(events)

    # Verify intent published
    assert len(kafka_client.published_messages) == 1
    msg = kafka_client.published_messages[0]
    assert msg["topic"] == TOPIC_EVENT_PUBLISH_INTENT

    # Verify intent payload
    intent = json.loads(msg["value"])
    assert intent["target_topic"] == TOPIC_METRICS_RECORDED
```

### Integration Tests

```python
@pytest.mark.integration
async def test_intent_executor_publishes_event():
    # Start intent executor
    executor = NodeIntentExecutor(container)
    executor_task = asyncio.create_task(executor.run())

    # Reducer publishes intent
    reducer = NodeCodegenMetricsReducer(container)
    await reducer.execute_reduction(events)

    # Wait for intent execution
    await asyncio.sleep(0.1)

    # Verify event published to target topic
    messages = await consume_topic(TOPIC_METRICS_RECORDED)
    assert len(messages) == 1
```

## Performance

### Targets

- Intent publishing: <10ms
- Intent execution: <50ms
- End-to-end latency: <60ms

### Monitoring

```python
# Intent publishing metrics
intent_publish_duration = histogram("intent_publish_ms")
intent_publish_errors = counter("intent_publish_errors")

# Intent execution metrics
intent_execution_duration = histogram("intent_execution_ms")
intent_execution_success_rate = gauge("intent_execution_success_rate")
```

## Migration Guide

### From Direct Publishing

**Before:**
```python
class MyReducer(NodeReducer):
    async def execute_reduction(self, events):
        result = self._aggregate(events)

        # ❌ Direct publishing
        await kafka_client.publish(
            topic=TOPIC_METRICS,
            key=str(result.id),
            value=event.model_dump_json()
        )

        return result
```

**After:**
```python
class MyReducer(NodeReducer, MixinIntentPublisher):
    def __init__(self, container):
        super().__init__(container)
        self._init_intent_publisher(container)

    async def execute_reduction(self, events):
        result = self._aggregate(events)

        # ✅ Intent publishing
        event = self._build_event(result)
        await self.publish_event_intent(
            target_topic=TOPIC_METRICS,
            target_key=str(result.id),
            event=event
        )

        return result
```

## Future Enhancements

1. **Intent Prioritization**: High-priority intents executed first
2. **Intent Batching**: Batch multiple intents for efficiency
3. **Intent TTL**: Expire unexecuted intents after timeout
4. **Intent Status API**: Query intent execution status
5. **Intent Replay**: Replay failed intents for debugging

## References

- ONEX v2.0 Architecture Specification
- [Contract-First Development](./CONTRACT_FIRST_DEVELOPMENT.md)
- [Event-Driven Architecture](./EVENT_DRIVEN_ARCHITECTURE.md)
- [Mixin Patterns](./MIXIN_PATTERNS.md)

## Status

- ✅ Pattern defined
- ✅ Implemented in omninode_bridge
- ✅ Tests passing (7/7)
- ✅ Documented
- ⏳ Pending: Copy to omnibase_core
- ⏳ Pending: Intent executor implementation
