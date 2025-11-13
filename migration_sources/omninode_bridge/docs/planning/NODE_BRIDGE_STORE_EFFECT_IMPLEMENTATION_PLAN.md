# NodeBridgeStoreEffect EFFECT Node - Implementation Plan

**Status**: Draft
**Created**: 2025-10-23
**Author**: Polymorphic Agent
**Context**: Pure Reducer Refactor - Intent Publisher Pattern (ONEX v2.0)

## Executive Summary

This document provides a comprehensive implementation plan for **NodeBridgeStoreEffect**, an EFFECT node that consumes intents emitted by the refactored **NodeBridgeReducer** and performs all I/O operations (PostgreSQL persistence, Kafka event publishing, FSM state recovery).

**Background**: Poly 6 refactored NodeBridgeReducer to follow the Intent Publisher pattern (ONEX v2.0 pure function architecture). The reducer now emits intents instead of performing direct I/O. This EFFECT node will consume those intents and execute the actual side effects.

**Key Differentiator**: This is **separate from the existing NodeStoreEffect** which handles canonical workflow state persistence (CQRS pattern). NodeBridgeStoreEffect handles reducer-specific operations (aggregated state, FSM transitions, event publishing).

## 1. Architecture Overview

### 1.1 EFFECT Node Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                  NodeBridgeReducer (COMPUTE)                │
│                                                             │
│  Pure Function: Aggregates metadata, updates FSM states    │
│  Emits Intents: PERSIST_STATE, PERSIST_FSM_TRANSITION,     │
│                 PUBLISH_EVENT, RECOVER_FSM_STATES           │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ intents[]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│             NodeBridgeStoreEffect (EFFECT)                  │
│                                                             │
│  Consume Intents: Route by intent_type                     │
│  Perform I/O:                                               │
│    - PostgreSQL: PERSIST_STATE, PERSIST_FSM_TRANSITION,    │
│                  RECOVER_FSM_STATES                         │
│    - Kafka: PUBLISH_EVENT                                   │
│  Return Results: Success/failure for each intent           │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ results[]
                           ▼
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
   ┌─────────────────┐           ┌──────────────────┐
   │  PostgreSQL     │           │  Kafka           │
   │  - bridge_      │           │  - Event Topics  │
   │    states       │           │  - OnexEnvelopeV1│
   │  - fsm_         │           │  - DLQ support   │
   │    transitions  │           │                  │
   └─────────────────┘           └──────────────────┘
```

### 1.2 Integration with Orchestrator

The orchestrator (or reducer's caller) is responsible for:
1. Calling `reducer.execute_reduction(contract)`
2. Extracting `output.intents` from result
3. Routing intents to NodeBridgeStoreEffect
4. Processing results and updating state

### 1.3 Error Handling Strategy

**Circuit Breaker Integration**:
- PostgreSQL operations use circuit breaker (already in PostgresConnectionManager)
- Kafka operations use circuit breaker (already in KafkaClient)
- EFFECT node tracks failures per operation type

**Retry Logic**:
- PostgreSQL: Transient errors retry 3x with exponential backoff
- Kafka: Built-in retry via KafkaClient (max 3 attempts)
- Intent-level: Failed intents return error results (orchestrator decides retry)

**Graceful Degradation**:
- If PostgreSQL fails: Return error result, log, update metrics
- If Kafka fails: Log, send to DLQ, return partial success
- If all fail: Return comprehensive error result with context

## 2. Node Structure

```
src/omninode_bridge/nodes/bridge_store_effect/v1_0_0/
├── contract.yaml                    # ONEX v2.0 contract
├── node.py                          # Main EFFECT node implementation
├── models/
│   ├── __init__.py
│   ├── model_intent_result.py      # Success/failure results
│   ├── model_persist_state_result.py
│   ├── model_fsm_result.py
│   ├── model_publish_result.py
│   └── events/
│       ├── __init__.py
│       ├── model_state_persisted.py      # STATE_PERSISTED event
│       ├── model_fsm_transitioned.py     # FSM_STATE_TRANSITIONED event
│       └── model_operation_failed.py     # OPERATION_FAILED event
├── handlers/
│   ├── __init__.py
│   ├── persist_state_handler.py    # PERSIST_STATE intent handler
│   ├── fsm_handler.py              # FSM transition/recovery handlers
│   └── publish_event_handler.py    # PUBLISH_EVENT intent handler
└── tests/
    ├── __init__.py
    ├── test_node.py
    ├── test_persist_state_handler.py
    ├── test_fsm_handler.py
    └── test_publish_event_handler.py
```

### 2.1 Why Separate from Existing NodeStoreEffect?

**Existing NodeStoreEffect** (`nodes/store_effect/v1_0_0/`):
- Purpose: CQRS write path for **canonical workflow state**
- Operations: Optimistic concurrency control, version management
- Tables: `canonical_workflow_state`, `state_projections`
- Events: `StateCommitted`, `StateConflict`
- Service: CanonicalStoreService

**New NodeBridgeStoreEffect** (`nodes/bridge_store_effect/v1_0_0/`):
- Purpose: Intent consumer for **reducer I/O operations**
- Operations: Aggregated state persistence, FSM transitions, event publishing
- Tables: `bridge_states`, `fsm_transitions`
- Events: `STATE_PERSISTED`, `FSM_STATE_TRANSITIONED`, `OPERATION_FAILED`
- Service: Direct PostgreSQL + Kafka integration

**Recommendation**: Keep them separate due to distinct responsibilities and data models.

## 3. Intent Handlers

### 3.1 PUBLISH_EVENT Handler

**Intent Type**: `EnumIntentType.PUBLISH_EVENT`
**Target**: `event_bus`
**Priority**: 0 (normal)

**Input Structure**:
```python
{
    "intent_type": "PublishEvent",
    "target": "event_bus",
    "payload": {
        "event_type": "AGGREGATION_STARTED",  # EnumReducerEvent value
        "aggregation_id": "uuid-string",
        "aggregation_type": "namespace_grouping",
        "batch_size": 100,
        "window_size_ms": 5000,
        "timestamp": "2025-10-23T10:00:00Z"
    },
    "priority": 0
}
```

**Processing Steps**:
1. Extract `event_type` and `payload` from intent
2. Call `kafka_client.publish_with_envelope()` with OnexEnvelopeV1 format
3. Use reducer's aggregation_id as correlation_id for tracing
4. Handle Kafka errors (DLQ, retry, circuit breaker)
5. Return result with success/failure status

**Output Structure**:
```python
ModelPublishResult(
    success=True,
    event_type="AGGREGATION_STARTED",
    topic="omninode_bridge.reducer.events.aggregation_started",
    event_id="uuid",
    error=None
)
```

**Error Handling**:
- Kafka timeout: Retry 3x, send to DLQ, return failure
- Connection error: Circuit breaker opens, return failure
- Validation error: Log, return failure (no retry)

**Performance Target**: <30ms per event (p95)

### 3.2 PERSIST_STATE Handler

**Intent Type**: `EnumIntentType.PERSIST_STATE`
**Target**: `store_effect`
**Priority**: 1 (high)

**Input Structure**:
```python
{
    "intent_type": "PersistState",
    "target": "store_effect",
    "payload": {
        "aggregated_data": {
            "omninode.services.metadata": {
                "total_stamps": 100,
                "total_size_bytes": 1024000,
                "unique_file_types_count": 5,
                "file_types": ["application/json", "text/plain", ...],
                "unique_workflows_count": 10,
                "workflow_ids": ["uuid1", "uuid2", ...]
            }
        },
        "fsm_states": {
            "workflow-uuid-1": "COMPLETED",
            "workflow-uuid-2": "PROCESSING"
        },
        "aggregation_id": "uuid-string",
        "timestamp": "2025-10-23T10:00:00Z"
    },
    "priority": 1
}
```

**Processing Steps**:
1. Extract `aggregated_data`, `fsm_states`, `aggregation_id` from payload
2. For each namespace in `aggregated_data`:
   - Upsert to `bridge_states` table
   - Use namespace as key for conflict resolution
   - Update `total_workflows_processed`, `total_items_aggregated`
   - Store aggregation metadata as JSONB
3. Use PostgreSQL transaction for atomicity
4. Handle conflicts with ON CONFLICT DO UPDATE
5. Return result with persisted namespaces

**Database Query**:
```sql
-- Upsert aggregated state per namespace
INSERT INTO bridge_states (
    bridge_id,
    namespace,
    total_workflows_processed,
    total_items_aggregated,
    aggregation_metadata,
    current_fsm_state,
    last_aggregation_timestamp,
    created_at,
    updated_at
) VALUES (
    $1, -- bridge_id (aggregation_id)
    $2, -- namespace
    $3, -- total_workflows (derived from unique_workflows_count)
    $4, -- total_items (total_stamps)
    $5, -- aggregation_metadata (JSONB: file_types, workflow_ids, etc.)
    $6, -- current_fsm_state (derived from most common state)
    $7, -- last_aggregation_timestamp
    NOW(),
    NOW()
)
ON CONFLICT (bridge_id, namespace)
DO UPDATE SET
    total_workflows_processed = bridge_states.total_workflows_processed + EXCLUDED.total_workflows_processed,
    total_items_aggregated = bridge_states.total_items_aggregated + EXCLUDED.total_items_aggregated,
    aggregation_metadata = EXCLUDED.aggregation_metadata,
    current_fsm_state = EXCLUDED.current_fsm_state,
    last_aggregation_timestamp = EXCLUDED.last_aggregation_timestamp,
    updated_at = NOW()
RETURNING bridge_id, namespace, total_workflows_processed, total_items_aggregated;
```

**Output Structure**:
```python
ModelPersistStateResult(
    success=True,
    namespaces_persisted=["omninode.services.metadata", "omninode.services.workflow"],
    total_workflows=10,
    total_items=100,
    aggregation_id="uuid",
    error=None
)
```

**Error Handling**:
- Transaction failure: Rollback, log, return error
- Connection pool exhausted: Wait with timeout, retry, return error
- Constraint violation: Log, return error (data issue)

**Performance Target**: <50ms per intent (p95)

### 3.3 PERSIST_FSM_TRANSITION Handler

**Intent Type**: `EnumIntentType.PERSIST_FSM_TRANSITION`
**Target**: `store_effect`
**Priority**: 1 (high)

**Input Structure**:
```python
{
    "intent_type": "PersistFSMTransition",
    "target": "store_effect",
    "payload": {
        "workflow_id": "uuid-string",
        "current_state": "COMPLETED",
        "previous_state": "PROCESSING",
        "transition_count": 3,
        "transition_record": {
            "from_state": "PROCESSING",
            "to_state": "COMPLETED",
            "trigger": "aggregation_update",
            "timestamp": "2025-10-23T10:00:00Z",
            "metadata": {"namespace": "omninode.services.metadata"}
        },
        "transition_history": [
            {
                "from_state": "PENDING",
                "to_state": "PROCESSING",
                "trigger": "workflow_started",
                "timestamp": "2025-10-23T09:00:00Z"
            },
            ...
        ],
        "metadata": {"namespace": "omninode.services.metadata"},
        "timestamp": "2025-10-23T10:00:00Z"
    },
    "priority": 1
}
```

**Processing Steps**:
1. Extract `workflow_id`, state data, transition record
2. Insert transition record into `fsm_transitions` table
3. Use transaction to ensure consistency
4. Optionally publish `FSM_STATE_TRANSITIONED` event to Kafka
5. Return result with transition ID

**Database Query**:
```sql
-- Insert FSM transition record
INSERT INTO fsm_transitions (
    id,
    entity_id,
    entity_type,
    from_state,
    to_state,
    transition_event,
    transition_data,
    created_at
) VALUES (
    uuid_generate_v4(),
    $1, -- workflow_id
    'workflow_execution',
    $2, -- from_state (previous_state)
    $3, -- to_state (current_state)
    $4, -- transition_event (trigger)
    $5, -- transition_data (JSONB: metadata, timestamp, etc.)
    NOW()
)
RETURNING id, entity_id, from_state, to_state, created_at;
```

**Output Structure**:
```python
ModelFSMResult(
    success=True,
    workflow_id="uuid",
    transition_id="uuid",
    from_state="PROCESSING",
    to_state="COMPLETED",
    event_published=True,
    error=None
)
```

**Error Handling**:
- Transaction failure: Rollback, log, return error
- Duplicate transition: Log warning, return success (idempotent)
- Kafka publish failure: Log, continue (transition persisted)

**Performance Target**: <30ms per intent (p95)

### 3.4 RECOVER_FSM_STATES Handler

**Intent Type**: `EnumIntentType.RECOVER_FSM_STATES`
**Target**: `store_effect`
**Priority**: 2 (highest)

**Input Structure**:
```python
{
    "intent_type": "RecoverFSMStates",
    "target": "store_effect",
    "payload": {
        "recovery_id": "uuid-string",
        "timestamp": "2025-10-23T10:00:00Z",
        "request_all_workflows": True
    },
    "priority": 2
}
```

**Processing Steps**:
1. Query `fsm_transitions` table for latest states per workflow
2. Build state cache data structure for reducer
3. Return recovered states for in-memory restoration
4. Handle large result sets with pagination (if needed)

**Database Query**:
```sql
-- Recover latest FSM state for each workflow
WITH latest_transitions AS (
    SELECT DISTINCT ON (entity_id)
        entity_id AS workflow_id,
        to_state AS current_state,
        from_state AS previous_state,
        transition_data,
        created_at AS last_transition_at
    FROM fsm_transitions
    WHERE entity_type = 'workflow_execution'
    ORDER BY entity_id, created_at DESC
)
SELECT
    workflow_id,
    current_state,
    previous_state,
    transition_data,
    last_transition_at
FROM latest_transitions
WHERE last_transition_at > NOW() - INTERVAL '7 days'  -- Only recent workflows
ORDER BY last_transition_at DESC
LIMIT 10000;  -- Safety limit
```

**Output Structure**:
```python
ModelFSMResult(
    success=True,
    recovered_states={
        "workflow-uuid-1": {
            "current_state": "COMPLETED",
            "previous_state": "PROCESSING",
            "transition_count": 3,
            "metadata": {...},
            "last_transition_at": "2025-10-23T09:55:00Z"
        },
        ...
    },
    total_recovered=150,
    failed=0,
    error=None
)
```

**Error Handling**:
- Query timeout: Return partial results, log warning
- Connection failure: Return empty dict, log error
- Memory limit: Paginate, return first N results

**Performance Target**: <200ms for 10,000 workflows (p95)

## 4. Database Operations

### 4.1 Connection Management

**Use Existing PostgresConnectionManager**:
```python
from omninode_bridge.infrastructure.postgres_connection_manager import (
    PostgresConnectionManager,
    ModelPostgresConfig
)

# Initialize from environment
config = ModelPostgresConfig.from_environment()
manager = PostgresConnectionManager(config)
await manager.initialize()

# Use context manager for transactions
async with manager.transaction() as conn:
    await conn.execute(query, *params)
```

**Connection Pool Configuration**:
- Min connections: 5
- Max connections: 50
- Command timeout: 60s
- Max queries per connection: 50,000
- Pool exhaustion threshold: 90%

### 4.2 Table Schemas

**bridge_states** (already exists - migration 004):
```sql
CREATE TABLE IF NOT EXISTS bridge_states (
    bridge_id UUID PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    total_workflows_processed INTEGER NOT NULL DEFAULT 0,
    total_items_aggregated INTEGER NOT NULL DEFAULT 0,
    aggregation_metadata JSONB DEFAULT '{}',
    current_fsm_state VARCHAR(50) NOT NULL,
    last_aggregation_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_bridge_states_namespace ON bridge_states(namespace);
CREATE INDEX idx_bridge_states_fsm_state ON bridge_states(current_fsm_state);
```

**fsm_transitions** (already exists - migration 003):
```sql
CREATE TABLE IF NOT EXISTS fsm_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    from_state VARCHAR(50),
    to_state VARCHAR(50) NOT NULL,
    transition_event VARCHAR(100) NOT NULL,
    transition_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_fsm_transitions_entity ON fsm_transitions(entity_id, entity_type);
CREATE INDEX idx_fsm_transitions_created_at ON fsm_transitions(created_at DESC);
```

### 4.3 Transaction Handling

**ACID Compliance**:
- Use `manager.transaction()` context manager
- Default isolation: `read_committed`
- Automatic rollback on exception
- Connection released after commit/rollback

**Example**:
```python
async def persist_aggregated_state(
    self,
    aggregated_data: dict,
    fsm_states: dict,
    aggregation_id: str
) -> ModelPersistStateResult:
    try:
        async with self.db_manager.transaction() as conn:
            # Upsert bridge states for each namespace
            for namespace, data in aggregated_data.items():
                await conn.execute(upsert_query, ...)

            # Transaction commits automatically if no exception
            return ModelPersistStateResult(success=True, ...)

    except asyncpg.PostgresError as e:
        # Transaction automatically rolled back
        logger.error(f"PostgreSQL error: {e}")
        return ModelPersistStateResult(success=False, error=str(e))
```

### 4.4 Connection Pooling Strategy

**Pool Size Calculation**:
- Concurrent workflows: ~100
- Intents per workflow: ~5
- Intent parallelism: ~10% (5 concurrent)
- Pool size: 20-30 connections optimal

**Monitoring**:
- Use `manager.get_pool_stats()` for metrics
- Alert on pool utilization >90%
- Alert on checkout time >100ms
- Track slow queries >100ms

## 5. Kafka Operations

### 5.1 Event Publishing

**Use Existing KafkaClient**:
```python
from omninode_bridge.services.kafka_client import KafkaClient

# Initialize from environment
kafka = KafkaClient()
await kafka.connect()

# Publish with OnexEnvelopeV1
await kafka.publish_with_envelope(
    event_type="AGGREGATION_STARTED",
    source_node_id=str(self.node_id),
    payload={
        "aggregation_id": aggregation_id,
        "timestamp": datetime.now(UTC).isoformat()
    },
    correlation_id=correlation_id
)
```

**Event Types to Publish**:

1. **STATE_PERSISTED** (after PERSIST_STATE success):
```python
{
    "event_type": "STATE_PERSISTED",
    "payload": {
        "aggregation_id": "uuid",
        "namespaces_count": 2,
        "workflows_count": 10,
        "timestamp": "2025-10-23T10:00:00Z"
    }
}
```

2. **FSM_STATE_TRANSITIONED** (after PERSIST_FSM_TRANSITION success):
```python
{
    "event_type": "FSM_STATE_TRANSITIONED",
    "payload": {
        "workflow_id": "uuid",
        "from_state": "PROCESSING",
        "to_state": "COMPLETED",
        "transition_id": "uuid",
        "timestamp": "2025-10-23T10:00:00Z"
    }
}
```

3. **OPERATION_FAILED** (on any failure):
```python
{
    "event_type": "OPERATION_FAILED",
    "payload": {
        "intent_type": "PERSIST_STATE",
        "error_code": "DATABASE_ERROR",
        "error_message": "Connection pool exhausted",
        "timestamp": "2025-10-23T10:00:00Z"
    }
}
```

### 5.2 Topics

**Service Prefix**: `omninode_bridge`

**Topic Naming Convention**:
```
omninode_bridge.reducer.events.{event_type}
```

**Example Topics**:
- `omninode_bridge.reducer.events.state_persisted`
- `omninode_bridge.reducer.events.fsm_state_transitioned`
- `omninode_bridge.reducer.events.operation_failed`

### 5.3 Error Handling and DLQ

**KafkaClient Features** (already implemented):
- Circuit breaker protection
- Retry with exponential backoff (3 attempts)
- Dead Letter Queue (DLQ) support
- Automatic DLQ topic creation: `{topic}.dlq`

**Handling Kafka Failures**:
```python
success = await kafka.publish_with_envelope(...)
if not success:
    # Event sent to DLQ automatically
    logger.warning(f"Event sent to DLQ: {event_type}")
    # Continue processing (persistence succeeded)
    return ModelPublishResult(
        success=True,  # Partial success
        event_published=False,
        dlq_fallback=True
    )
```

## 6. Integration Points

### 6.1 Orchestrator Integration

**Option A: Direct Intent Execution** (Recommended for MVP):
```python
# In orchestrator or caller
reducer_output = await reducer.execute_reduction(contract)

# Extract intents
intents = reducer_output.intents

# Route intents to EFFECT node
store_effect = container.get_node("bridge_store_effect")
results = await store_effect.execute_intents(intents)

# Process results
for intent, result in zip(intents, results):
    if not result.success:
        logger.error(f"Intent {intent.intent_type} failed: {result.error}")
```

**Option B: Event-Driven (Future Enhancement)**:
```python
# Reducer publishes intents to Kafka topic
await kafka.publish_with_envelope(
    event_type="INTENT_EMITTED",
    payload={"intents": [intent.model_dump() for intent in intents]}
)

# NodeBridgeStoreEffect subscribes to intent topic
# Processes intents asynchronously
# Publishes result events back to Kafka
```

### 6.2 Result Communication

**Synchronous Results** (MVP):
```python
class ModelIntentResult(BaseModel):
    """Base result model for all intent handlers."""
    success: bool
    intent_type: str
    error: Optional[str] = None
    execution_time_ms: float
    timestamp: datetime
```

**Async Results** (Future):
- Publish result events to Kafka
- Orchestrator subscribes to result topics
- Use correlation_id for matching

### 6.3 Workflow Coordination Patterns

**Pattern 1: Sequential Intent Execution**:
```python
results = []
for intent in intents:
    result = await store_effect.execute_intent(intent)
    results.append(result)
    if not result.success and intent.priority > 1:
        # Fail fast for high-priority intents
        break
```

**Pattern 2: Parallel Intent Execution** (where safe):
```python
# Group intents by type
publish_intents = [i for i in intents if i.intent_type == "PublishEvent"]
persist_intents = [i for i in intents if i.intent_type == "PersistState"]

# Execute in parallel
publish_results, persist_results = await asyncio.gather(
    store_effect.execute_intents(publish_intents),
    store_effect.execute_intents(persist_intents)
)
```

**Pattern 3: Priority-Based Execution**:
```python
# Sort by priority (highest first)
sorted_intents = sorted(intents, key=lambda x: x.priority, reverse=True)

# Execute high-priority first
results = await store_effect.execute_intents(sorted_intents)
```

## 7. Testing Strategy

### 7.1 Unit Tests

**Test Coverage Targets**:
- Intent handlers: 100%
- Database operations: 100%
- Kafka operations: 100%
- Error handling: 100%
- Overall: >95%

**Key Test Cases**:

1. **PUBLISH_EVENT Handler**:
   - Success path: Event published to Kafka
   - Kafka timeout: DLQ fallback
   - Circuit breaker: Return failure
   - Invalid event type: Validation error

2. **PERSIST_STATE Handler**:
   - Success: State persisted to bridge_states
   - Conflict: ON CONFLICT DO UPDATE
   - Transaction failure: Rollback
   - Connection pool exhausted: Retry

3. **PERSIST_FSM_TRANSITION Handler**:
   - Success: Transition persisted
   - Duplicate: Idempotent handling
   - Invalid state: Validation error

4. **RECOVER_FSM_STATES Handler**:
   - Success: States recovered
   - Empty database: Return empty dict
   - Query timeout: Partial results

### 7.2 Integration Tests

**Test PostgreSQL Integration**:
```python
@pytest.mark.asyncio
async def test_persist_state_integration(postgres_fixture):
    """Test PERSIST_STATE with real PostgreSQL."""
    manager = PostgresConnectionManager(config)
    await manager.initialize()

    handler = PersistStateHandler(manager, kafka_client)

    intent = ModelIntent(
        intent_type="PersistState",
        target="store_effect",
        payload={
            "aggregated_data": {...},
            "aggregation_id": str(uuid4())
        }
    )

    result = await handler.execute(intent)

    assert result.success
    assert result.namespaces_persisted == ["test.namespace"]

    # Verify in database
    async with manager.acquire_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM bridge_states WHERE namespace = $1",
            "test.namespace"
        )
        assert row is not None
```

**Test Kafka Integration**:
```python
@pytest.mark.asyncio
async def test_publish_event_integration(kafka_fixture):
    """Test PUBLISH_EVENT with real Kafka."""
    kafka = KafkaClient()
    await kafka.connect()

    handler = PublishEventHandler(kafka)

    intent = ModelIntent(
        intent_type="PublishEvent",
        target="event_bus",
        payload={
            "event_type": "STATE_PERSISTED",
            "aggregation_id": str(uuid4())
        }
    )

    result = await handler.execute(intent)

    assert result.success
    assert result.event_published

    # Verify event in Kafka
    messages = await kafka.consume_messages(
        topic=result.topic,
        group_id="test-consumer",
        max_messages=1
    )
    assert len(messages) == 1
```

### 7.3 Error Scenario Tests

**Test Database Failures**:
```python
@pytest.mark.asyncio
async def test_persist_state_transaction_failure(mock_postgres):
    """Test PERSIST_STATE with transaction failure."""
    # Mock transaction failure
    mock_postgres.transaction.side_effect = asyncpg.PostgresError("Connection lost")

    handler = PersistStateHandler(mock_postgres, kafka_client)
    result = await handler.execute(intent)

    assert not result.success
    assert "Connection lost" in result.error
    assert result.execution_time_ms > 0
```

**Test Kafka Failures**:
```python
@pytest.mark.asyncio
async def test_publish_event_kafka_timeout(mock_kafka):
    """Test PUBLISH_EVENT with Kafka timeout."""
    # Mock Kafka timeout
    mock_kafka.publish_with_envelope.return_value = False

    handler = PublishEventHandler(mock_kafka)
    result = await handler.execute(intent)

    # Partial success (DLQ fallback)
    assert result.success
    assert not result.event_published
    assert result.dlq_fallback
```

### 7.4 Performance Tests

**Load Testing**:
```python
@pytest.mark.performance
async def test_concurrent_intent_execution():
    """Test 1000 concurrent intents."""
    intents = [
        create_persist_state_intent() for _ in range(500)
    ] + [
        create_publish_event_intent() for _ in range(500)
    ]

    start = time.perf_counter()
    results = await store_effect.execute_intents(intents)
    duration = (time.perf_counter() - start) * 1000

    # Performance targets
    assert duration < 10000  # <10s for 1000 intents
    assert all(r.execution_time_ms < 100 for r in results)  # <100ms per intent
    assert sum(1 for r in results if r.success) / len(results) > 0.95  # >95% success
```

## 8. Performance Requirements

### 8.1 Latency Targets

| Operation | Target | P95 | P99 | Notes |
|-----------|--------|-----|-----|-------|
| PUBLISH_EVENT | <20ms | <30ms | <50ms | Kafka publish with envelope |
| PERSIST_STATE | <40ms | <50ms | <80ms | PostgreSQL upsert per namespace |
| PERSIST_FSM_TRANSITION | <20ms | <30ms | <50ms | PostgreSQL insert |
| RECOVER_FSM_STATES | <100ms | <200ms | <300ms | PostgreSQL query (10K workflows) |
| Overall Intent | <50ms | <80ms | <150ms | Average across all types |

### 8.2 Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Intents/second | 1000+ | Sustained throughput |
| PERSIST_STATE/sec | 500+ | Database-bound |
| PUBLISH_EVENT/sec | 2000+ | Kafka-bound |
| Concurrent workflows | 100+ | Parallel execution |

### 8.3 Circuit Breaker Thresholds

**PostgreSQL Circuit Breaker**:
- Failure threshold: 5 failures in 30s
- Timeout: 60s
- Half-open retry: 10s

**Kafka Circuit Breaker**:
- Failure threshold: 5 failures in 30s
- Timeout: 30s
- Half-open retry: 5s

### 8.4 Connection Pool Sizing

**PostgreSQL Pool**:
- Min connections: 10
- Max connections: 50
- Optimal: 20-30 (based on load testing)
- Exhaustion threshold: 90%

**Kafka Producer**:
- Batch size: 100 messages (environment-dependent)
- Linger: 10ms (production), 0ms (development)
- Compression: lz4 (production), none (development)

## 9. Implementation Phases

### Phase 1: Basic EFFECT Node Structure + PUBLISH_EVENT Handler (2-3 days)

**Deliverables**:
- [ ] Create `nodes/bridge_store_effect/v1_0_0/` structure
- [ ] Implement `contract.yaml` with ONEX v2.0 compliance
- [ ] Implement `node.py` with NodeEffect base class
- [ ] Implement PUBLISH_EVENT handler
- [ ] Unit tests for PUBLISH_EVENT handler
- [ ] Integration test with Kafka

**Files Created**:
```
nodes/bridge_store_effect/v1_0_0/
├── contract.yaml
├── node.py
├── models/
│   ├── model_intent_result.py
│   └── model_publish_result.py
└── handlers/
    └── publish_event_handler.py
```

**Acceptance Criteria**:
- PUBLISH_EVENT handler publishes events to Kafka
- OnexEnvelopeV1 format used
- DLQ fallback on failure
- Metrics tracked (success/failure counts)
- Tests pass with >90% coverage

### Phase 2: PERSIST_STATE Handler with PostgreSQL (3-4 days)

**Deliverables**:
- [ ] Implement PERSIST_STATE handler
- [ ] PostgreSQL integration with connection manager
- [ ] Transaction handling with rollback
- [ ] Unit tests for PERSIST_STATE handler
- [ ] Integration test with PostgreSQL

**Files Created**:
```
handlers/persist_state_handler.py
models/model_persist_state_result.py
tests/test_persist_state_handler.py
```

**Acceptance Criteria**:
- State persisted to `bridge_states` table
- ON CONFLICT DO UPDATE handles duplicates
- Transactions rollback on error
- Performance <50ms (p95)
- Tests pass with >95% coverage

### Phase 3: FSM Handlers (PERSIST_FSM_TRANSITION, RECOVER_FSM_STATES) (3-4 days)

**Deliverables**:
- [ ] Implement PERSIST_FSM_TRANSITION handler
- [ ] Implement RECOVER_FSM_STATES handler
- [ ] FSM state recovery query optimization
- [ ] Unit tests for both handlers
- [ ] Integration tests with PostgreSQL

**Files Created**:
```
handlers/fsm_handler.py
models/model_fsm_result.py
events/model_fsm_transitioned.py
tests/test_fsm_handler.py
```

**Acceptance Criteria**:
- FSM transitions persisted to `fsm_transitions` table
- Recovery query returns 10K workflows <200ms
- Idempotent handling for duplicate transitions
- Tests pass with >95% coverage

### Phase 4: Error Handling, Circuit Breaker, Resilience (2-3 days)

**Deliverables**:
- [ ] Comprehensive error handling for all handlers
- [ ] Circuit breaker integration (use existing)
- [ ] Retry logic with exponential backoff
- [ ] DLQ fallback for Kafka failures
- [ ] Error scenario tests

**Acceptance Criteria**:
- Circuit breaker opens on repeated failures
- Retry logic tested with transient errors
- DLQ receives failed events
- Graceful degradation on partial failures
- Error tests pass with 100% coverage

### Phase 5: Performance Optimization and Load Testing (3-4 days)

**Deliverables**:
- [ ] Connection pool tuning
- [ ] Batch intent execution optimization
- [ ] Parallel execution where safe
- [ ] Load tests (1000+ intents/sec)
- [ ] Performance benchmarks

**Acceptance Criteria**:
- Throughput >1000 intents/sec
- Latency <50ms (p95) for PERSIST_STATE
- Latency <30ms (p95) for PUBLISH_EVENT
- Pool utilization <80% under load
- Performance tests pass

### Total Estimated Time: 13-18 days

**Milestones**:
- Day 3: Phase 1 complete (PUBLISH_EVENT working)
- Day 7: Phase 2 complete (PERSIST_STATE working)
- Day 11: Phase 3 complete (FSM handlers working)
- Day 14: Phase 4 complete (Error handling robust)
- Day 18: Phase 5 complete (Performance targets met)

## 10. Migration Strategy

### 10.1 Deployment Alongside Existing Reducer

**Step 1: Deploy NodeBridgeStoreEffect** (without connecting to reducer):
```bash
# Deploy new EFFECT node
kubectl apply -f deployments/bridge-store-effect.yaml

# Verify health
curl http://bridge-store-effect:8080/health
```

**Step 2: Connect Reducer to EFFECT Node** (staged rollout):
```python
# Feature flag for intent-based I/O
ENABLE_INTENT_PUBLISHER = os.getenv("ENABLE_INTENT_PUBLISHER", "false")

if ENABLE_INTENT_PUBLISHER == "true":
    # Use NodeBridgeStoreEffect
    results = await store_effect.execute_intents(output.intents)
else:
    # Use existing direct I/O (deprecated)
    await legacy_persist_state(output)
```

**Step 3: Gradual Traffic Shift**:
- Week 1: 10% traffic to new EFFECT node
- Week 2: 50% traffic
- Week 3: 100% traffic
- Week 4: Remove legacy code

### 10.2 Orchestrator Changes Needed

**Minimal Changes** (if orchestrator exists):
```python
# Before (legacy)
output = await reducer.execute_reduction(contract)
# I/O happens inside reducer

# After (intent-based)
output = await reducer.execute_reduction(contract)
if output.intents:
    # Route intents to EFFECT node
    results = await bridge_store_effect.execute_intents(output.intents)
```

**No Changes** (if orchestrator doesn't exist):
- EFFECT node can be called directly from reducer's caller
- No orchestrator required for MVP

### 10.3 Backward Compatibility Considerations

**Reducer Backward Compatibility**:
- Pure reducer emits intents (new behavior)
- Intents are optional in output model
- Legacy reducers return empty intent list
- EFFECT node handles empty intent list gracefully

**Database Schema Compatibility**:
- `bridge_states` table already exists (migration 004)
- `fsm_transitions` table already exists (migration 003)
- No schema changes required
- Existing data compatible with new operations

**Kafka Topic Compatibility**:
- New topics: `omninode_bridge.reducer.events.*`
- Existing topics: `omninode_bridge.workflow.events.*`
- No conflicts, can coexist
- Consumers need to subscribe to new topics

### 10.4 Rollback Plan

**Rollback Triggers**:
- Error rate >5%
- Latency >2x baseline
- Database connection failures
- Kafka publish failures >10%

**Rollback Steps**:
1. Set feature flag: `ENABLE_INTENT_PUBLISHER=false`
2. Restart reducer pods (reverts to legacy I/O)
3. Monitor error rate and latency
4. Investigate issues
5. Re-deploy with fixes

**Data Consistency**:
- EFFECT node uses same tables as legacy code
- No data migration required
- Rollback is safe (no data loss)

## 11. Open Questions

### 11.1 Design Decisions Needing Clarification

**Q1: Intent Routing - Who's Responsible?**
- **Option A**: Orchestrator routes intents to EFFECT node (requires orchestrator changes)
- **Option B**: Reducer directly calls EFFECT node (tight coupling)
- **Option C**: Kafka-based routing (intents published as events, EFFECT subscribes)
- **Recommendation**: Option B for MVP, Option C for production

**Q2: Intent Execution Order - Sequential or Parallel?**
- **Option A**: Sequential execution (simplest, preserves order)
- **Option B**: Parallel execution (faster, but requires dependency analysis)
- **Option C**: Priority-based with partial parallelism
- **Recommendation**: Option A for MVP, Option C for performance

**Q3: Error Recovery - Retry at What Level?**
- **Option A**: EFFECT node retries (3x per intent)
- **Option B**: Orchestrator retries (pass failed intents back)
- **Option C**: No retry (orchestrator decides)
- **Recommendation**: Option A for transient errors, Option B for business logic errors

**Q4: FSM State Recovery - When to Trigger?**
- **Option A**: On reducer startup (automatic recovery)
- **Option B**: On-demand (orchestrator calls recovery)
- **Option C**: Scheduled (cron job)
- **Recommendation**: Option A for MVP (startup recovery)

### 11.2 Performance Trade-offs

**Q5: Connection Pool Size - How Many Connections?**
- Current: 5-50 connections
- Recommendation: 20-30 for reducer workload
- Trade-off: More connections = more overhead, fewer connections = pool exhaustion

**Q6: Batch Intent Execution - What's Optimal Batch Size?**
- Options: 1, 10, 50, 100 intents per batch
- Trade-off: Larger batches = better throughput, higher latency
- Recommendation: 10-20 intents per batch

**Q7: Kafka Batching - Linger Time?**
- Production: 10ms (batching enabled)
- Development: 0ms (batching disabled)
- Trade-off: Batching = higher throughput, higher latency

### 11.3 Alternative Approaches Considered

**Alternative 1: Unified StoreEffect**
- Extend existing NodeStoreEffect to handle reducer intents
- Pros: Single EFFECT node, code reuse
- Cons: Mixed responsibilities, harder to maintain
- **Decision**: Rejected - separate concerns

**Alternative 2: Event-Driven Intent Processing**
- Reducer publishes intents to Kafka
- EFFECT node subscribes and processes async
- Pros: Decoupled, scalable
- Cons: More complex, eventual consistency
- **Decision**: Future enhancement (not MVP)

**Alternative 3: Inline I/O (No EFFECT Node)**
- Keep I/O inside reducer (current state)
- Pros: Simplest, no new components
- Cons: Violates pure function pattern, not ONEX v2.0 compliant
- **Decision**: Rejected - architectural goal is pure functions

## 12. Appendix

### 12.1 Related Documents

- [Pure Reducer Refactor Plan](./PURE_REDUCER_REFACTOR_PLAN.md) - Overall refactor strategy
- [NodeBridgeReducer Implementation](../../src/omninode_bridge/nodes/reducer/v1_0_0/node.py) - Refactored reducer
- [EnumIntentType](../../src/omninode_bridge/nodes/reducer/v1_0_0/models/enum_intent_type.py) - Intent definitions
- [ModelIntent](../../src/omninode_bridge/nodes/reducer/v1_0_0/models/model_intent.py) - Intent model
- [PostgresConnectionManager](../../src/omninode_bridge/infrastructure/postgres_connection_manager.py) - DB client
- [KafkaClient](../../src/omninode_bridge/services/kafka_client.py) - Kafka client
- [NodeStoreEffect](../../src/omninode_bridge/nodes/store_effect/v1_0_0/node.py) - Existing EFFECT node (canonical state)

### 12.2 ONEX v2.0 Compliance Checklist

- [x] Suffix-based naming: `NodeBridgeStoreEffect`
- [x] Import from omnibase_core
- [x] ModelContainer dependency injection
- [x] Strong typing (Pydantic v2)
- [x] ModelOnexError error handling
- [x] Contract-driven (contract.yaml)
- [x] Event-driven (Kafka integration)
- [x] Metrics tracking
- [x] Health checks
- [x] Transaction management
- [ ] Lifecycle methods (startup, shutdown)
- [ ] Circuit breaker integration
- [ ] Comprehensive tests

### 12.3 Example Contract YAML

```yaml
# NodeBridgeStoreEffect Contract - ONEX v2.0
node_type: effect
node_name: bridge_store_effect
version: 1.0.0
description: |
  Bridge Store Effect Node - Intent Consumer for Reducer I/O Operations

  Consumes intents from NodeBridgeReducer and performs:
  - PostgreSQL persistence (bridge_states, fsm_transitions)
  - Kafka event publishing (STATE_PERSISTED, FSM_STATE_TRANSITIONED)
  - FSM state recovery

capabilities:
  - execute_intents
  - persist_aggregated_state
  - persist_fsm_transitions
  - recover_fsm_states
  - publish_events
  - track_metrics

dependencies:
  postgres_connection_manager:
    type: PostgresConnectionManager
    required: true
  kafka_client:
    type: KafkaClient
    required: true

performance:
  persistence_latency_ms:
    p95: 50
    p99: 80
  publish_latency_ms:
    p95: 30
    p99: 50
  throughput:
    target: 1000
    unit: "intents/second"

metrics:
  - name: intents_executed_total
    type: counter
  - name: intents_failed_total
    type: counter
  - name: avg_intent_latency_ms
    type: gauge
  - name: database_operations_total
    type: counter
  - name: kafka_events_published_total
    type: counter
```

### 12.4 Glossary

- **Intent**: A request to perform a side effect (I/O operation)
- **EFFECT Node**: Node that executes side effects (I/O)
- **COMPUTE Node**: Pure function node (no I/O)
- **DLQ**: Dead Letter Queue for failed messages
- **Circuit Breaker**: Pattern to prevent cascading failures
- **OnexEnvelopeV1**: Standard Kafka event envelope format
- **FSM**: Finite State Machine for workflow state tracking
- **Aggregated State**: Reduced state across multiple workflows

---

**End of Implementation Plan**

**Next Steps**:
1. Review plan with team
2. Address open questions
3. Begin Phase 1 implementation
4. Set up monitoring and alerting
5. Plan load testing strategy
