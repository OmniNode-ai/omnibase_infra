# Intent Publisher Pattern for omnibase_core

This directory contains files to be copied to `omnibase_core` for the next PR.

## Purpose

The Intent Publisher pattern allows ONEX nodes to coordinate actions (like publishing events) without performing direct domain I/O. This maintains architectural boundaries while enabling necessary coordination.

## Files to Copy

### 1. Intent Event Models
**Source:** `events/models/intent_events.py`
**Destination:** `omnibase_core/src/omnibase_core/events/models/intent_events.py`

Contains:
- `ModelEventPublishIntent` - Intent to publish an event
- `ModelIntentExecutionResult` - Result of intent execution
- `TOPIC_EVENT_PUBLISH_INTENT` - Kafka topic for intents

### 2. IntentPublisherMixin
**Source:** `mixins/intent_publisher_mixin.py`
**Destination:** `omnibase_core/src/omnibase_core/mixins/intent_publisher_mixin.py`

Contains:
- `IntentPublisherMixin` - Mixin providing intent publishing capability
- `IntentPublishResult` - Result of publishing an intent

### 3. Intent Publisher Subcontract Template
**Source:** `contracts/intent_publisher.yaml`
**Destination:** `omnibase_core/docs/contracts/templates/intent_publisher.yaml`

Contains:
- Subcontract specification for intent publishing capability
- Kafka configuration
- Event schemas
- Usage examples

## Integration Steps for omnibase_core

### 1. Copy Files

```bash
# From omninode_bridge root
cd FOR_OMNIBASE_CORE

# Copy to omnibase_core (adjust paths as needed)
cp events/models/intent_events.py /path/to/omnibase_core/src/omnibase_core/events/models/
cp mixins/intent_publisher_mixin.py /path/to/omnibase_core/src/omnibase_core/mixins/
cp contracts/intent_publisher.yaml /path/to/omnibase_core/docs/contracts/templates/
```

### 2. Update Imports

In `omnibase_core/src/omnibase_core/mixins/__init__.py`:

```python
from omnibase_core.mixins.intent_publisher_mixin import (
    IntentPublisherMixin,
    IntentPublishResult,
)

__all__ = [
    "IntentPublisherMixin",
    "IntentPublishResult",
]
```

In `omnibase_core/src/omnibase_core/events/models/__init__.py`:

```python
from omnibase_core.events.models.intent_events import (
    ModelEventPublishIntent,
    ModelIntentExecutionResult,
    TOPIC_EVENT_PUBLISH_INTENT,
)

__all__ = [
    "ModelEventPublishIntent",
    "ModelIntentExecutionResult",
    "TOPIC_EVENT_PUBLISH_INTENT",
]
```

### 3. Update IntentPublisherMixin Imports

In `mixins/intent_publisher_mixin.py`, change:

```python
# OLD (omninode_bridge)
from omninode_bridge.events.models.intent_events import (
    TOPIC_EVENT_PUBLISH_INTENT,
    ModelEventPublishIntent,
)

# NEW (omnibase_core)
from omnibase_core.events.models.intent_events import (
    TOPIC_EVENT_PUBLISH_INTENT,
    ModelEventPublishIntent,
)
```

### 4. Add Tests

Copy tests from `omninode_bridge/tests/unit/test_intent_publisher_mixin.py` to `omnibase_core/tests/mixins/test_intent_publisher_mixin.py`

Update test imports:

```python
# OLD
from omninode_bridge.mixins import IntentPublisherMixin, IntentPublishResult
from omninode_bridge.events.models.intent_events import TOPIC_EVENT_PUBLISH_INTENT

# NEW
from omnibase_core.mixins import IntentPublisherMixin, IntentPublishResult
from omnibase_core.events.models.intent_events import TOPIC_EVENT_PUBLISH_INTENT
```

### 5. Update Documentation

Add to `omnibase_core/docs/architecture/MIXINS.md`:

```markdown
## IntentPublisherMixin

Provides intent publishing capability for ONEX nodes that need coordination I/O.

### Purpose
Allows nodes to coordinate actions without performing direct domain I/O, maintaining ONEX architectural boundaries.

### Usage
```python
from omnibase_core.mixins import IntentPublisherMixin

class MyReducer(NodeReducer, IntentPublisherMixin):
    def __init__(self, container):
        super().__init__(container)
        self._init_intent_publisher(container)

    async def execute_reduction(self, events):
        result = self._aggregate(events)

        event = MetricsEvent(...)
        await self.publish_event_intent(
            target_topic=TOPIC_METRICS,
            target_key=str(result.id),
            event=event
        )

        return result
```

### Contract Requirements
- Subcontract: `intent_publisher.yaml`
- Service: `kafka_client`
```

## Architecture Benefits

### 1. **ONEX Compliance**
- Maintains separation between domain logic and coordination I/O
- REDUCER nodes stay pure (no direct event publishing)
- Coordination I/O is explicit and traceable

### 2. **Testability**
- Pure node logic testable without Kafka
- Intent building is pure (deterministic)
- Intent execution can be mocked

### 3. **Flexibility**
- Same intent can be executed by different effects
- Can swap Kafka for other message brokers
- Intent execution can be monitored/retried independently

### 4. **Observability**
- All intents logged on Kafka coordination topic
- Intent execution results traceable
- Correlation IDs for end-to-end tracing

## Pattern Usage

### Nodes That Should Use This Pattern

- ✅ **REDUCER nodes** - Need to publish aggregated results
- ✅ **COMPUTE nodes** - Need to publish computed results
- ✅ **ORCHESTRATOR nodes** - Need to coordinate multiple services
- ❌ **EFFECT nodes** - These perform I/O directly (no intent needed)

### When to Use Intent Publishing

Use intent publishing when:
1. Node needs to publish events but wants to stay architecturally pure
2. Event publishing is coordination, not domain logic
3. Want to decouple event production from publishing
4. Need traceable, auditable coordination

Don't use intent publishing when:
1. Node is already an EFFECT (direct I/O is its job)
2. Event publishing is core domain logic (rethink your architecture)
3. Synchronous execution is required (intents are async)

## Implementation Status

- ✅ Event models defined
- ✅ Mixin implemented
- ✅ Subcontract specification complete
- ✅ Tests written (7/7 passing)
- ✅ Documented
- ✅ Integrated in omninode_bridge (metrics reducer)
- ⏳ Pending: Copy to omnibase_core

## Future Enhancements

1. **Intent Executor Node** - Dedicated node for executing intents
2. **Intent Status Tracking** - Query intent execution status
3. **Intent Prioritization** - Execute high-priority intents first
4. **Intent Batching** - Batch multiple intents for efficiency
5. **Intent TTL** - Expire unexecuted intents after timeout

## Contact

For questions or issues with this pattern, contact the omninode team.
