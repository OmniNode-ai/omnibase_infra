# ONEX EventBus Integration Patterns

**Purpose**: Document correct EventBus patterns for ONEX v2.0 nodes to ensure Archon RAG can guide future agents.

**Correlation ID**: parallel-solve-patterns-001

## Table of Contents

1. [Why EventBus is Mandatory](#why-eventbus-is-mandatory)
2. [Pattern 1: Direct EventBus (Orchestrator Nodes)](#pattern-1-direct-eventbus-orchestrator-nodes)
3. [Pattern 2: Intent-Based (Effect/Reducer Nodes)](#pattern-2-intent-based-effectreducer-nodes)
4. [Anti-Patterns (What NOT to Do)](#anti-patterns-what-not-to-do)
5. [Validation Checklist](#validation-checklist)
6. [Migration Guide](#migration-guide)
7. [Testing Templates](#testing-templates)

---

## Why EventBus is Mandatory

**ONEX v2.0 Architecture Principle**: All inter-node communication MUST use EventBus for event-driven coordination.

**Benefits**:
- ✅ **Standardized Event Format**: OnexEnvelopeV1 wrapping ensures consistent event structure
- ✅ **Decoupling**: Nodes communicate via events, not direct HTTP calls
- ✅ **Observability**: All events are Kafka-backed with full audit trail
- ✅ **Graceful Degradation**: Nodes can fall back to logging when Kafka is unavailable
- ✅ **Correlation Tracking**: UUID-based correlation enables end-to-end workflow tracing
- ✅ **Replay Capability**: Intent-based patterns enable event replay and debugging

**Critical Rule**: If you're writing code that says `TODO: Implement Kafka integration`, you're doing it wrong. EventBus integration is NOT optional.

---

## Pattern 1: Direct EventBus (Orchestrator Nodes)

**Use When**: Your node is an **Orchestrator** that coordinates workflows across multiple nodes.

**Example**: NodeBridgeOrchestrator

### Implementation Steps

#### 1. Initialize EventBus from Container

```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelContainer) -> None:
        super().__init__(container)

        # Get or create EventBus from container for event-driven coordination
        self.event_bus = container.get_service("event_bus")
        if self.event_bus is None:
            # Initialize EventBus service
            if self.kafka_client:
                from ....services.event_bus import EventBusService

                self.event_bus = EventBusService(
                    kafka_client=self.kafka_client,
                    node_id=self.node_id,
                    namespace=self.default_namespace,
                )
                container.register_service("event_bus", self.event_bus)
                emit_log_event(
                    LogLevel.INFO,
                    "EventBus service initialized successfully",
                    {"node_id": self.node_id, "namespace": self.default_namespace},
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    "EventBus not available - Kafka client required for event-driven coordination",
                    {"node_id": self.node_id},
                )
                self.event_bus = None
```

**Key Points**:
- Get EventBus from container first (dependency injection)
- Create EventBusService only if not already available
- Register with container for reuse across nodes
- Graceful degradation when Kafka is unavailable

#### 2. Use EventBus for Workflow Coordination

```python
async def _execute_event_driven_workflow(
    self, contract: ModelContractOrchestrator
) -> ModelStampResponseOutput:
    """Execute workflow using event-driven coordination with reducer."""

    # Publish Action event to trigger reducer processing
    success = await self.event_bus.publish_action_event(
        correlation_id=workflow_id,
        action_type="ORCHESTRATE_WORKFLOW",
        payload={
            "operation": "metadata_stamping",
            "input_data": contract.input_data,
            "namespace": self.default_namespace,
            "workflow_type": "metadata_stamping",
        },
    )

    # Wait for completion event from reducer (configurable timeout)
    event = await self.event_bus.wait_for_completion(
        correlation_id=workflow_id,
        timeout_seconds=performance_config.WORKFLOW_COMPLETION_TIMEOUT_SECONDS,
    )

    # Handle event based on type
    if event_type == "STATE_COMMITTED":
        return await self._handle_success(workflow_id, event, start_time)
    elif event_type == "REDUCER_GAVE_UP":
        return await self._handle_failure(workflow_id, event, start_time, retry_count=0)
```

**Key Points**:
- Use `publish_action_event()` to trigger reducer processing
- Use `wait_for_completion()` to await reducer response
- Handle StateCommitted and ReducerGaveUp events
- Pass correlation_id for end-to-end tracing

#### 3. Fallback to Legacy Mode When EventBus Unavailable

```python
async def execute_orchestration(
    self, contract: ModelContractOrchestrator
) -> ModelStampResponseOutput:
    """Execute stamping workflow orchestration with event-driven coordination."""

    # Check if event-driven coordination is available
    if self.event_bus and self.event_bus.is_initialized:
        return await self._execute_event_driven_workflow(contract)
    else:
        # Fallback to legacy synchronous execution
        emit_log_event(
            LogLevel.WARNING,
            "EventBus not available - falling back to legacy synchronous workflow",
            {"node_id": self.node_id, "workflow_id": str(contract.correlation_id)},
        )
        return await self._execute_legacy_workflow(contract)
```

**Key Points**:
- Always check `event_bus.is_initialized` before use
- Provide fallback for backward compatibility
- Log degraded mode for monitoring

#### 4. Publish Events via KafkaClient (Direct Path)

```python
async def _publish_event(
    self, event_type: EnumWorkflowEvent, data: dict[str, Any]
) -> None:
    """Publish event to Kafka using EventType subcontract with OnexEnvelopeV1 wrapping."""

    try:
        # Get Kafka topic name
        topic_name = event_type.get_topic_name(namespace=self.default_namespace)

        # Publish to Kafka if client is available
        if self.kafka_client and self.kafka_client.is_connected:
            # Extract correlation ID from data (workflow_id)
            correlation_id = data.get("workflow_id")

            # Add node metadata to payload
            payload = {
                **data,
                "node_id": self.node_id,
                "published_at": datetime.now(UTC).isoformat(),
            }

            # Publish with OnexEnvelopeV1 wrapping for standardized event format
            success = await self.kafka_client.publish_with_envelope(
                event_type=event_type.value,
                source_node_id=self.node_id,
                payload=payload,
                topic=topic_name,
                correlation_id=correlation_id,
                metadata={
                    "event_category": "workflow_orchestration",
                    "node_type": "orchestrator",
                    "namespace": self.default_namespace,
                },
            )

            if success:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Published Kafka event (OnexEnvelopeV1): {event_type.value}",
                    {
                        "node_id": self.node_id,
                        "event_type": event_type.value,
                        "topic_name": topic_name,
                        "correlation_id": correlation_id,
                        "envelope_wrapped": True,
                    },
                )
    except Exception as e:
        # Log error but don't fail workflow
        emit_log_event(
            LogLevel.WARNING,
            f"Failed to publish Kafka event: {event_type.value}",
            {"node_id": self.node_id, "event_type": event_type.value, "error": str(e)},
        )
```

**Key Points**:
- Use `event_type.get_topic_name()` for topic routing (DO NOT hardcode topics)
- Always use `publish_with_envelope()` for OnexEnvelopeV1 wrapping
- Include correlation_id for tracing
- Add metadata for event categorization
- Graceful error handling (don't fail workflow on event publish errors)

#### 5. Lifecycle Management

```python
async def startup(self) -> None:
    """Node startup lifecycle hook."""

    # Initialize EventBus service
    if self.event_bus and not self.event_bus.is_initialized:
        try:
            await self.event_bus.initialize()
            emit_log_event(
                LogLevel.INFO,
                "EventBus initialized and ready for event-driven coordination",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"EventBus initialization failed, continuing without event-driven coordination: {e}",
                {"node_id": self.node_id, "error": str(e)},
            )

async def shutdown(self) -> None:
    """Node shutdown lifecycle hook."""

    # Shutdown EventBus service
    if self.event_bus and self.event_bus.is_initialized:
        try:
            await self.event_bus.shutdown()
            emit_log_event(
                LogLevel.INFO,
                "EventBus shutdown successfully",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"EventBus shutdown failed: {e}",
                {"node_id": self.node_id, "error": str(e)},
            )
```

**Key Points**:
- Initialize EventBus in `startup()` before use
- Shutdown EventBus in `shutdown()` to clean up resources
- Graceful error handling for lifecycle operations

---

## Pattern 2: Intent-Based (Effect/Reducer Nodes)

**Use When**: Your node is a **Reducer** or **Effect** node following ONEX v2.0 Pure Function Pattern.

**Example**: NodeBridgeReducer

**Core Principle**: NO direct I/O operations. Generate intents for all side effects.

### Implementation Steps

#### 1. Initialize KafkaClient (Optional Direct Publishing)

```python
class NodeBridgeReducer(NodeReducer):
    def __init__(self, container: ModelContainer) -> None:
        super().__init__(container)

        # Pending event publishing intents (ONEX v2.0 pure function pattern)
        self._pending_event_intents: list[Any] = []

        # Configuration for intent generation (no I/O dependencies)
        self.default_namespace = container.config.get("default_namespace", "omninode.bridge")
        self.kafka_broker_url = container.config.get("kafka_broker_url", "localhost:9092")

        # Get or create KafkaClient from container (skip if in health check mode)
        health_check_mode = container.config.get("health_check_mode", False)
        self.kafka_client = container.get_service("kafka_client")

        if self.kafka_client is None and not health_check_mode:
            # Import KafkaClient
            try:
                from ....services.kafka_client import KafkaClient

                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=3,
                    timeout_seconds=performance_config.KAFKA_CLIENT_TIMEOUT_SECONDS,
                )
                container.register_service("kafka_client", self.kafka_client)
            except ImportError:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {"node_id": getattr(self, "node_id", "reducer")},
                )
                self.kafka_client = None
```

**Key Points**:
- Initialize `_pending_event_intents` list for intent collection
- Get KafkaClient from container for optional immediate publishing
- Graceful degradation when KafkaClient is unavailable

#### 2. Generate PublishEvent Intents (Pure Function Pattern)

```python
async def _publish_event(
    self, event_type: EnumReducerEvent, data: dict[str, Any]
) -> None:
    """
    Generate PublishEvent intent and optionally publish to Kafka.

    ONEX v2.0 Pure Function Pattern:
    - Generates PublishEvent intent for orchestrator
    - Optionally publishes directly to Kafka if client is available
    - Intents allow replay and orchestrator tracking
    """
    try:
        # Import ModelIntent and EnumIntentType here to avoid circular imports
        from .models.enum_intent_type import EnumIntentType
        from .models.model_intent import ModelIntent

        # Get Kafka topic name
        topic_name = event_type.get_topic_name(namespace=self.default_namespace)

        # Extract correlation ID from data
        correlation_id = data.get("correlation_id") or data.get("aggregation_id")

        # Generate PublishEvent intent (ONEX v2.0 pure function pattern)
        intent = ModelIntent(
            intent_type=EnumIntentType.PUBLISH_EVENT.value,
            target="event_bus",
            payload={
                "event_type": event_type.value,
                "topic": topic_name,
                "data": data,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            priority=0,  # Normal priority for event publishing
        )
        self._pending_event_intents.append(intent)

        # Publish to Kafka if client is available (immediate feedback)
        if self.kafka_client and self.kafka_client.is_connected:
            # ... (same as Pattern 1, publish_with_envelope)
```

**Key Points**:
- ALWAYS generate intent first (pure function)
- Intent contains all data needed for orchestrator to execute
- Optional immediate Kafka publishing for real-time feedback
- Intents stored in `_pending_event_intents` list

#### 3. Return Intents in Output

```python
async def execute_reduction(
    self, contract: ModelContractReducer,
) -> ModelReducerOutputState:
    """Execute pure metadata aggregation and state reduction."""

    # ... aggregation logic ...

    # Initialize intents list for side effects
    intents: list[ModelIntent] = []

    # Intent: Persist aggregated state (if StateManagement configured)
    if self._state_config or hasattr(contract, "state_management"):
        intents.append(
            ModelIntent(
                intent_type=EnumIntentType.PERSIST_STATE.value,
                target="store_effect",
                payload={
                    "aggregated_data": dict(aggregated_data),
                    "fsm_states": fsm_states,
                    "aggregation_id": aggregation_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                priority=1,  # High priority for persistence
            )
        )

    # Collect intents from FSMStateManager (FSM transition persistence intents)
    fsm_intents = self._fsm_manager.get_pending_intents()
    intents.extend(fsm_intents)

    # Collect event publishing intents (ONEX v2.0 pure function pattern)
    event_intents = self.get_pending_event_intents()
    intents.extend(event_intents)

    # Return aggregation results with intents
    return ModelReducerOutputState(
        aggregation_type=aggregation_type,
        total_items=total_items,
        total_size_bytes=total_size_bytes,
        namespaces=list(aggregated_data.keys()),
        aggregations=dict(aggregated_data),
        fsm_states=fsm_states,
        intents=intents,  # <-- CRITICAL: Return all intents for orchestrator
        aggregation_duration_ms=duration_ms,
        items_per_second=items_per_second,
    )
```

**Key Points**:
- Collect all intents before returning
- Include intents in output model
- Orchestrator will execute intents via EFFECT nodes

#### 4. Provide Intent Retrieval Method

```python
def get_pending_event_intents(self) -> list[Any]:
    """
    Retrieve and clear pending event publishing intents.

    Returns:
        List of pending PublishEvent intents for orchestrator
    """
    intents = self._pending_event_intents.copy()
    self._pending_event_intents.clear()
    return intents
```

**Key Points**:
- Clear intents after retrieval (prevent duplicate execution)
- Return copy to avoid external mutation

---

## Anti-Patterns (What NOT to Do)

### ❌ Anti-Pattern 1: Direct Kafka Producer Usage

**WRONG**:
```python
from kafka import KafkaProducer

class MyNode:
    def __init__(self):
        # DON'T DO THIS!
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    async def publish_event(self, event_data):
        # DON'T DO THIS!
        self.producer.send('my-topic', event_data)
```

**Why Wrong**:
- No OnexEnvelopeV1 wrapping
- No correlation tracking
- No graceful degradation
- No container integration
- Hardcoded topic names

**RIGHT** (Pattern 1 - Orchestrator):
```python
async def _publish_event(
    self, event_type: EnumWorkflowEvent, data: dict[str, Any]
) -> None:
    """Publish event to Kafka using EventType subcontract with OnexEnvelopeV1 wrapping."""

    topic_name = event_type.get_topic_name(namespace=self.default_namespace)

    if self.kafka_client and self.kafka_client.is_connected:
        success = await self.kafka_client.publish_with_envelope(
            event_type=event_type.value,
            source_node_id=self.node_id,
            payload=payload,
            topic=topic_name,
            correlation_id=correlation_id,
            metadata={...},
        )
```

**RIGHT** (Pattern 2 - Reducer):
```python
async def _publish_event(
    self, event_type: EnumReducerEvent, data: dict[str, Any]
) -> None:
    """Generate PublishEvent intent and optionally publish to Kafka."""

    # Generate intent (pure function)
    intent = ModelIntent(
        intent_type=EnumIntentType.PUBLISH_EVENT.value,
        target="event_bus",
        payload={...},
        priority=0,
    )
    self._pending_event_intents.append(intent)

    # Optional immediate publishing
    if self.kafka_client and self.kafka_client.is_connected:
        await self.kafka_client.publish_with_envelope(...)
```

---

### ❌ Anti-Pattern 2: TODO Comments for Event Integration

**WRONG**:
```python
async def process_workflow(self):
    # Process workflow
    result = await self._do_work()

    # TODO: Publish event to Kafka when workflow completes
    # TODO: Implement EventBus integration

    return result
```

**Why Wrong**:
- EventBus integration is NOT optional
- TODO indicates incomplete implementation
- Missing observability and tracing

**RIGHT**:
```python
async def process_workflow(self):
    # Publish workflow started event
    await self._publish_event(
        EnumWorkflowEvent.WORKFLOW_STARTED,
        {"workflow_id": str(workflow_id), "timestamp": datetime.now(UTC).isoformat()},
    )

    # Process workflow
    result = await self._do_work()

    # Publish workflow completed event
    await self._publish_event(
        EnumWorkflowEvent.WORKFLOW_COMPLETED,
        {"workflow_id": str(workflow_id), "result": result},
    )

    return result
```

---

### ❌ Anti-Pattern 3: HTTP Endpoints for Event Publishing

**WRONG**:
```python
@app.post("/publish-event")
async def publish_event(event_data: dict):
    # DON'T DO THIS!
    # Events should not be published via HTTP endpoints
    await kafka_producer.send(event_data)
    return {"status": "published"}
```

**Why Wrong**:
- Events are internal coordination mechanism, not external API
- HTTP adds latency and coupling
- No standardized event format
- Breaks event-driven architecture

**RIGHT**:
```python
# Events are published internally via EventBus or intents
# NO HTTP endpoints needed for event publishing
```

---

### ❌ Anti-Pattern 4: Hardcoded Topic Names

**WRONG**:
```python
async def publish_event(self, event_data):
    # DON'T DO THIS!
    topic = "omninode.bridge.workflow.events"  # Hardcoded!
    await self.kafka_producer.send(topic, event_data)
```

**Why Wrong**:
- Topic naming should be centralized in EnumEvent
- No namespace support
- Hard to refactor

**RIGHT**:
```python
async def _publish_event(
    self, event_type: EnumWorkflowEvent, data: dict[str, Any]
) -> None:
    # Use EnumEvent.get_topic_name() for standardized topic routing
    topic_name = event_type.get_topic_name(namespace=self.default_namespace)

    await self.kafka_client.publish_with_envelope(
        event_type=event_type.value,
        topic=topic_name,  # Dynamic topic from enum
        ...
    )
```

---

### ❌ Anti-Pattern 5: Missing OnexEnvelopeV1 Wrapping

**WRONG**:
```python
async def publish_event(self, event_data):
    # DON'T DO THIS!
    await self.kafka_producer.send(topic, event_data)  # Raw data, no envelope
```

**Why Wrong**:
- No standardized event format
- Missing metadata (correlation_id, timestamp, source_node_id)
- Breaks event tracing and replay

**RIGHT**:
```python
success = await self.kafka_client.publish_with_envelope(
    event_type=event_type.value,
    source_node_id=self.node_id,
    payload=payload,
    topic=topic_name,
    correlation_id=correlation_id,  # UUID for tracing
    metadata={
        "event_category": "workflow_orchestration",
        "node_type": "orchestrator",
        "namespace": self.default_namespace,
    },
)
```

---

### ❌ Anti-Pattern 6: No Graceful Degradation

**WRONG**:
```python
async def publish_event(self, event_data):
    # DON'T DO THIS!
    await self.kafka_client.publish(topic, event_data)  # Fails if Kafka is down
```

**Why Wrong**:
- Node fails completely if Kafka is unavailable
- No fallback mechanism
- Poor resilience

**RIGHT** (Orchestrator):
```python
async def execute_orchestration(self, contract: ModelContractOrchestrator):
    # Check if event-driven coordination is available
    if self.event_bus and self.event_bus.is_initialized:
        return await self._execute_event_driven_workflow(contract)
    else:
        # Fallback to legacy synchronous execution
        emit_log_event(
            LogLevel.WARNING,
            "EventBus not available - falling back to legacy synchronous workflow",
            {"node_id": self.node_id},
        )
        return await self._execute_legacy_workflow(contract)
```

**RIGHT** (Reducer/Effect):
```python
async def _publish_event(self, event_type: EnumReducerEvent, data: dict[str, Any]):
    # Generate intent (ALWAYS works, no I/O)
    intent = ModelIntent(...)
    self._pending_event_intents.append(intent)

    # Optional immediate publishing (gracefully degrades)
    if self.kafka_client and self.kafka_client.is_connected:
        try:
            await self.kafka_client.publish_with_envelope(...)
        except Exception as e:
            emit_log_event(LogLevel.WARNING, f"Failed to publish event: {e}")
    else:
        emit_log_event(LogLevel.DEBUG, f"Kafka unavailable, event logged: {event_type}")
```

---

## Validation Checklist

Use this checklist to validate EventBus integration in your ONEX v2.0 node:

### Orchestrator Nodes

- [ ] **Container Integration**: EventBus retrieved from container via `container.get_service("event_bus")`
- [ ] **EventBus Initialization**: EventBusService created with KafkaClient, node_id, namespace
- [ ] **Event-Driven Workflow**: Uses `event_bus.publish_action_event()` and `event_bus.wait_for_completion()`
- [ ] **Fallback Mode**: Implements legacy synchronous workflow when EventBus unavailable
- [ ] **Event Publishing**: Uses `kafka_client.publish_with_envelope()` with OnexEnvelopeV1
- [ ] **Topic Routing**: Uses `event_type.get_topic_name(namespace=...)` (NO hardcoded topics)
- [ ] **Correlation Tracking**: Passes correlation_id to all event methods
- [ ] **Lifecycle Management**: Calls `event_bus.initialize()` in startup(), `event_bus.shutdown()` in shutdown()
- [ ] **Health Checks**: Implements EventBus health check in `_check_event_bus_health()`
- [ ] **Graceful Degradation**: Logs warnings and continues when EventBus fails
- [ ] **NO TODO Comments**: EventBus integration is complete, not "TODO"
- [ ] **NO Direct Kafka**: No `from kafka import KafkaProducer` or raw Kafka usage
- [ ] **NO HTTP Events**: No HTTP endpoints for event publishing

### Reducer/Effect Nodes

- [ ] **Intent List**: Initializes `_pending_event_intents: list[Any] = []`
- [ ] **KafkaClient Optional**: Gets KafkaClient from container for optional publishing
- [ ] **Intent Generation**: Uses `ModelIntent(intent_type=EnumIntentType.PUBLISH_EVENT.value, ...)`
- [ ] **Intent Collection**: Calls `get_pending_event_intents()` before returning output
- [ ] **Intent Return**: Includes `intents` in output model
- [ ] **Pure Functions**: NO direct I/O, all side effects via intents
- [ ] **Optional Publishing**: Publishes to Kafka if client available, otherwise logs
- [ ] **Topic Routing**: Uses `event_type.get_topic_name(namespace=...)` (NO hardcoded topics)
- [ ] **Correlation Tracking**: Extracts correlation_id from data for tracing
- [ ] **Graceful Degradation**: Works without Kafka (pure intent generation)
- [ ] **NO TODO Comments**: EventBus integration is complete, not "TODO"
- [ ] **NO Direct Kafka**: No `from kafka import KafkaProducer` or raw Kafka usage
- [ ] **NO HTTP Events**: No HTTP endpoints for event publishing

---

## Migration Guide

### From Raw Kafka to EventBus

If you have existing code using raw Kafka producers, follow these steps:

#### Step 1: Remove Raw Kafka Imports

**Before**:
```python
from kafka import KafkaProducer
import json

class MyNode:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
```

**After**:
```python
# NO raw Kafka imports needed

class MyNode:
    def __init__(self, container: ModelContainer):
        super().__init__(container)

        # Get KafkaClient from container
        self.kafka_client = container.get_service("kafka_client")
        if self.kafka_client is None:
            from ....services.kafka_client import KafkaClient
            self.kafka_client = KafkaClient(...)
            container.register_service("kafka_client", self.kafka_client)
```

#### Step 2: Replace Hardcoded Topics with EnumEvent

**Before**:
```python
topic = "omninode.bridge.workflow.events"
self.producer.send(topic, event_data)
```

**After**:
```python
# Define enum for your events
class EnumWorkflowEvent(str, Enum):
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"

    def get_topic_name(self, namespace: str = "omninode.bridge") -> str:
        return f"{namespace}.workflow.{self.value}"

# Use enum for topic routing
topic_name = EnumWorkflowEvent.WORKFLOW_STARTED.get_topic_name(
    namespace=self.default_namespace
)
```

#### Step 3: Add OnexEnvelopeV1 Wrapping

**Before**:
```python
self.producer.send(topic, event_data)
```

**After**:
```python
await self.kafka_client.publish_with_envelope(
    event_type=EnumWorkflowEvent.WORKFLOW_STARTED.value,
    source_node_id=self.node_id,
    payload=event_data,
    topic=topic_name,
    correlation_id=workflow_id,
    metadata={
        "event_category": "workflow_orchestration",
        "node_type": "orchestrator",
        "namespace": self.default_namespace,
    },
)
```

#### Step 4: Add Graceful Degradation

**Before**:
```python
# No error handling - fails if Kafka is down
self.producer.send(topic, event_data)
```

**After**:
```python
try:
    if self.kafka_client and self.kafka_client.is_connected:
        success = await self.kafka_client.publish_with_envelope(...)
        if success:
            emit_log_event(LogLevel.DEBUG, f"Published event: {event_type}")
        else:
            emit_log_event(LogLevel.WARNING, f"Failed to publish event: {event_type}")
    else:
        emit_log_event(LogLevel.DEBUG, f"Kafka unavailable, logging event: {event_type}")
except Exception as e:
    emit_log_event(
        LogLevel.WARNING,
        f"Error publishing event: {event_type}",
        {"error": str(e)},
    )
```

#### Step 5: For Reducer/Effect Nodes, Add Intent Generation

**Before** (direct Kafka publishing):
```python
async def process_data(self, data):
    result = await self._do_work(data)
    self.producer.send(topic, result)  # Direct I/O!
    return result
```

**After** (intent-based):
```python
async def process_data(self, data):
    result = await self._do_work(data)

    # Generate intent (pure function)
    intent = ModelIntent(
        intent_type=EnumIntentType.PUBLISH_EVENT.value,
        target="event_bus",
        payload={
            "event_type": EnumReducerEvent.DATA_PROCESSED.value,
            "data": result,
        },
        priority=0,
    )
    self._pending_event_intents.append(intent)

    # Optional immediate publishing
    if self.kafka_client and self.kafka_client.is_connected:
        await self.kafka_client.publish_with_envelope(...)

    return result
```

---

## Testing Templates

### Unit Test Template for Orchestrator EventBus Integration

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

@pytest.mark.asyncio
async def test_event_driven_workflow_success():
    """Test event-driven workflow with EventBus coordination."""

    # Arrange
    container = MagicMock()
    mock_event_bus = MagicMock()
    mock_event_bus.is_initialized = True
    mock_event_bus.publish_action_event = AsyncMock(return_value=True)
    mock_event_bus.wait_for_completion = AsyncMock(
        return_value={
            "event_type": "STATE_COMMITTED",
            "payload": {"state": {"stamp_id": "test-123"}},
        }
    )
    container.get_service.return_value = mock_event_bus

    node = NodeBridgeOrchestrator(container)
    workflow_id = uuid4()
    contract = ModelContractOrchestrator(
        correlation_id=workflow_id,
        input_data={"content": "test content"},
    )

    # Act
    result = await node.execute_orchestration(contract)

    # Assert
    assert result.workflow_state == EnumWorkflowState.COMPLETED
    mock_event_bus.publish_action_event.assert_called_once()
    mock_event_bus.wait_for_completion.assert_called_once()
    assert mock_event_bus.publish_action_event.call_args[1]["correlation_id"] == workflow_id

@pytest.mark.asyncio
async def test_fallback_to_legacy_when_eventbus_unavailable():
    """Test graceful degradation to legacy workflow when EventBus unavailable."""

    # Arrange
    container = MagicMock()
    container.get_service.return_value = None  # No EventBus available

    node = NodeBridgeOrchestrator(container)
    workflow_id = uuid4()
    contract = ModelContractOrchestrator(
        correlation_id=workflow_id,
        input_data={"content": "test content"},
    )

    # Act
    result = await node.execute_orchestration(contract)

    # Assert - should still complete successfully via legacy mode
    assert result.workflow_state == EnumWorkflowState.COMPLETED
```

### Unit Test Template for Reducer Intent Generation

```python
import pytest
from uuid import uuid4

@pytest.mark.asyncio
async def test_publish_event_generates_intent():
    """Test that _publish_event generates PublishEvent intent."""

    # Arrange
    container = MagicMock()
    node = NodeBridgeReducer(container)

    # Act
    await node._publish_event(
        EnumReducerEvent.AGGREGATION_STARTED,
        {"aggregation_id": "test-123", "timestamp": "2025-01-01T00:00:00Z"},
    )

    # Assert
    intents = node.get_pending_event_intents()
    assert len(intents) == 1
    assert intents[0].intent_type == EnumIntentType.PUBLISH_EVENT.value
    assert intents[0].target == "event_bus"
    assert intents[0].payload["event_type"] == EnumReducerEvent.AGGREGATION_STARTED.value

@pytest.mark.asyncio
async def test_execute_reduction_returns_intents():
    """Test that execute_reduction returns all collected intents."""

    # Arrange
    container = MagicMock()
    node = NodeBridgeReducer(container)

    contract = ModelContractReducer(
        input_state={
            "items": [
                {"namespace": "test", "workflow_id": str(uuid4()), "file_size": 100}
            ]
        }
    )

    # Act
    result = await node.execute_reduction(contract)

    # Assert - should include PublishEvent intents for lifecycle events
    assert len(result.intents) > 0
    event_intents = [i for i in result.intents if i.intent_type == EnumIntentType.PUBLISH_EVENT.value]
    assert len(event_intents) > 0  # Should have at least AGGREGATION_STARTED and AGGREGATION_COMPLETED
```

---

## Summary

**Golden Rules**:

1. **Orchestrator Nodes**: Use EventBusService for event-driven coordination + direct KafkaClient for event publishing
2. **Reducer/Effect Nodes**: Generate PublishEvent intents + optional immediate KafkaClient publishing
3. **ALWAYS**: Use OnexEnvelopeV1 wrapping via `kafka_client.publish_with_envelope()`
4. **NEVER**: Use raw Kafka producers, hardcoded topics, or TODO comments for event integration
5. **ALWAYS**: Implement graceful degradation when Kafka/EventBus unavailable
6. **ALWAYS**: Use correlation_id for end-to-end workflow tracing

**Pattern Selection Guide**:

- **Orchestrator Node** → Pattern 1 (Direct EventBus)
- **Reducer Node** → Pattern 2 (Intent-Based)
- **Effect Node** → Pattern 2 (Intent-Based)
- **Compute Node** → Usually no event publishing needed (pure computation)

**When in Doubt**: Check the compliant implementations:
- `/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py` (Pattern 1)
- `/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/reducer/v1_0_0/node.py` (Pattern 2)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Correlation ID**: parallel-solve-patterns-001
