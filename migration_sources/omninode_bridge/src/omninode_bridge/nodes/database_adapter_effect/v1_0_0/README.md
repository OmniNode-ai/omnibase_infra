# Database Adapter Effect Node v1.0.0

**ONEX v2.0 Compliant Effect Node for PostgreSQL Database Operations**

## Overview

The **Database Adapter Effect Node** is an ONEX v2.0 compliant effect node that consumes Kafka events from bridge nodes (Orchestrator, Reducer, Registry) and persists data to PostgreSQL. Following the event-driven architecture principle, this node handles all external database interactions, enabling bridge nodes to focus on their core orchestration and aggregation logic.

### Architectural Principle

> **ONEX Architecture**: Effect nodes handle all external system interactions (databases, APIs, file systems).

This separation provides:
- **Decoupling**: Bridge nodes focus on business logic, not database concerns
- **Scalability**: Database adapter can scale independently
- **Testability**: Bridge nodes can be tested without database infrastructure
- **Resilience**: Database failures don't crash orchestration workflows
- **Event Replay**: Failed operations can be retried from Kafka event log

## Node Type: EFFECT

Effect nodes in ONEX v2.0 architecture are responsible for:
- External service interactions (databases, APIs, message queues)
- Side effects and state persistence
- Event-driven data synchronization
- Idempotent operation handling

## Contract Structure

```yaml
node_name: "node_bridge_database_adapter_effect"
contract_name: "BridgeDatabaseAdapterEffectContract"
node_type: "EFFECT"
input_model: "ModelDatabaseOperationInput"
output_model: "ModelDatabaseOperationOutput"
```

### Dependencies (Protocol Injection)

The node declares three protocol dependencies resolved through ONEX registry:

1. **ProtocolConnectionPoolManager**: PostgreSQL connection pooling
2. **ProtocolQueryExecutor**: SQL query execution
3. **ProtocolTransactionManager**: Transaction management

## Supported Operations (6 Total)

### 1. persist_workflow_execution
**Purpose**: Insert/update workflow execution records
**Source Events**: `WORKFLOW_STARTED`, `WORKFLOW_COMPLETED`, `WORKFLOW_FAILED`
**Database Table**: `workflow_executions`
**Operation Type**: INSERT (on start) / UPDATE (on completion/failure)

Tracks complete workflow lifecycle from start to completion.

### 2. persist_workflow_step
**Purpose**: Insert workflow step history
**Source Events**: `STEP_COMPLETED`
**Database Table**: `workflow_steps`
**Operation Type**: INSERT

Records individual step execution within workflows for detailed tracking.

### 3. persist_bridge_state
**Purpose**: Upsert bridge aggregation state
**Source Events**: `STATE_AGGREGATION_COMPLETED`
**Database Table**: `bridge_states`
**Operation Type**: UPSERT (INSERT with ON CONFLICT DO UPDATE)

Maintains aggregated state for NodeBridgeReducer with atomic updates.

### 4. persist_fsm_transition
**Purpose**: Insert FSM state transition records
**Source Events**: `STATE_TRANSITION`
**Database Table**: `fsm_transitions`
**Operation Type**: INSERT

Audit trail for all finite state machine transitions across the system.

### 5. persist_metadata_stamp
**Purpose**: Insert metadata stamp audit records
**Source Events**: `STAMP_CREATED`
**Database Table**: `metadata_stamps`
**Operation Type**: INSERT

Audit trail linking metadata stamps to workflow executions.

### 6. update_node_heartbeat
**Purpose**: Update node heartbeat timestamps
**Source Events**: `NODE_HEARTBEAT`
**Database Table**: `node_registrations`
**Operation Type**: UPDATE

Keeps node health status current for monitoring and service discovery.

## Event-to-Database Operation Mapping

| Event Type | Source Node | Operation | Target Table |
|-----------|-------------|-----------|--------------|
| `WORKFLOW_STARTED` | Orchestrator | INSERT | `workflow_executions` |
| `WORKFLOW_COMPLETED` | Orchestrator | UPDATE | `workflow_executions` |
| `WORKFLOW_FAILED` | Orchestrator | UPDATE | `workflow_executions` |
| `STEP_COMPLETED` | Orchestrator | INSERT | `workflow_steps` |
| `STAMP_CREATED` | Orchestrator | INSERT | `metadata_stamps` |
| `STATE_TRANSITION` | Orchestrator/Reducer | INSERT | `fsm_transitions` |
| `STATE_AGGREGATION_COMPLETED` | Reducer | UPSERT | `bridge_states` |
| `NODE_HEARTBEAT` | All Nodes | UPDATE | `node_registrations` |

## Database Schema

### workflow_executions
```sql
CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL UNIQUE,
    workflow_type VARCHAR(100) NOT NULL,
    current_state VARCHAR(50) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### workflow_steps
```sql
CREATE TABLE workflow_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_name VARCHAR(100) NOT NULL,
    step_order INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER,
    step_data JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### bridge_states
```sql
CREATE TABLE bridge_states (
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
```

### fsm_transitions
```sql
CREATE TABLE fsm_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    from_state VARCHAR(50),
    to_state VARCHAR(50) NOT NULL,
    transition_event VARCHAR(100) NOT NULL,
    transition_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### metadata_stamps
```sql
CREATE TABLE metadata_stamps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_correlation_id UUID REFERENCES workflow_executions(id) ON DELETE SET NULL,
    file_hash VARCHAR(128) NOT NULL,
    stamp_data JSONB NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### node_registrations
```sql
CREATE TABLE node_registrations (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100) NOT NULL,
    node_version VARCHAR(50) NOT NULL,
    capabilities JSONB DEFAULT '{}',
    endpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    health_status VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Architecture Components

### NodeBridgeDatabaseAdapterEffect
Main node class inheriting from `NodeEffectService`. Handles:
- Protocol dependency resolution through registry
- Operation routing based on operation type
- Error handling with circuit breaker pattern
- Performance metrics tracking
- Correlation ID preservation across operations

### DatabaseCircuitBreaker
Resilience pattern for database operations:
- **States**: CLOSED, OPEN, HALF_OPEN
- **Failure Threshold**: 5 consecutive failures
- **Timeout**: 30 seconds before retry
- **Recovery**: Automatic half-open state for retry attempts

### DatabaseStructuredLogger
Correlation-aware logging:
- UUID correlation tracking across operations
- Performance categorization (FAST, SLOW, VERY_SLOW)
- Operation type classification
- JSON-formatted structured output

### SecurityValidator
Input validation for database operations:
- SQL injection protection
- Parameter sanitization
- Input type validation
- JSONB payload validation

## Performance Targets

| Metric | Target | Criticality |
|--------|--------|-------------|
| Database Operations (p95) | < 10ms | High |
| Event Processing Latency (p95) | < 50ms | High |
| Connection Pool Efficiency | > 90% | Medium |
| Throughput | 1000+ events/sec | High |
| Memory Usage | < 256MB | Medium |

## Usage Example

### Event-Driven Pattern

```python
from omninode_bridge.nodes.database_adapter_effect.v1_0_0 import NodeBridgeDatabaseAdapterEffect
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.registry import RegistryBridgeDatabaseAdapter

# Initialize registry with PostgreSQL configuration
registry = RegistryBridgeDatabaseAdapter({
    "host": "localhost",
    "port": 5432,
    "database": "bridge_db",
    "user": "bridge_user",
    "password": "bridge_password",  # pragma: allowlist secret
    "pool_size": 20
})

# Initialize database adapter node
adapter = NodeBridgeDatabaseAdapterEffect(registry)
await adapter.initialize()

# Process database operation from Kafka event
from .models.inputs.model_database_operation_input import ModelDatabaseOperationInput

operation = ModelDatabaseOperationInput(
    operation_type="persist_workflow_execution",
    correlation_id=uuid4(),
    workflow_execution_data={
        "workflow_type": "metadata_stamping",
        "current_state": "PROCESSING",
        "namespace": "default"
    }
)

result = await adapter.process(operation)
print(f"Success: {result.success}, Rows Affected: {result.rows_affected}")
```

### Direct Operation Call

```python
# Direct call to specific operation handler
result = await adapter._persist_bridge_state(
    ModelDatabaseOperationInput(
        operation_type="persist_bridge_state",
        correlation_id=uuid4(),
        bridge_state_data={
            "bridge_id": str(uuid4()),
            "namespace": "production",
            "total_workflows_processed": 1500,
            "total_items_aggregated": 45000,
            "current_fsm_state": "ACTIVE"
        }
    )
)
```

## Integration with Bridge Nodes

### NodeBridgeOrchestrator Integration

```python
# In NodeBridgeOrchestrator, emit events instead of direct DB calls
await self._publish_event(
    event_type=EnumWorkflowEvent.WORKFLOW_STARTED,
    correlation_id=correlation_id,
    event_data={
        "workflow_type": "metadata_stamping",
        "namespace": self.namespace,
        "started_at": datetime.utcnow().isoformat()
    }
)

# Database adapter consumes event and persists to workflow_executions table
```

### NodeBridgeReducer Integration

```python
# In NodeBridgeReducer, emit state aggregation events
await self._publish_event(
    event_type="STATE_AGGREGATION_COMPLETED",
    correlation_id=correlation_id,
    event_data=bridge_state.model_dump()
)

# Database adapter consumes event and UPSERTs to bridge_states table
```

## Error Handling

### OnexError Integration
All errors are wrapped in `OnexError` with context:

```python
try:
    await self._query_executor.execute(sql, parameters)
except Exception as e:
    raise OnexError(
        message=f"Failed to persist workflow execution",
        context={
            "correlation_id": str(correlation_id),
            "operation_type": "persist_workflow_execution",
            "error": str(e)
        }
    )
```

### Circuit Breaker Protection

```python
@self._circuit_breaker.protected
async def _execute_database_operation(self, sql: str, params: list) -> Any:
    """Database operation with circuit breaker protection."""
    return await self._query_executor.execute(sql, params)
```

## Monitoring & Observability

### Health Check Endpoint
```python
health = await adapter.get_health_status()
# Returns:
# {
#     "healthy": True,
#     "database_version": "PostgreSQL 15.3",
#     "connection_stats": {
#         "pool_size": 20,
#         "active_connections": 8,
#         "idle_connections": 12,
#         "total_queries": 15420
#     },
#     "circuit_breaker_status": "CLOSED"
# }
```

### Performance Metrics
```python
metrics = await adapter.get_metrics()
# Returns:
# {
#     "operations_by_type": {
#         "persist_workflow_execution": 2540,
#         "persist_workflow_step": 8120,
#         "persist_bridge_state": 1200
#     },
#     "avg_execution_times": {
#         "persist_workflow_execution": 8.5,
#         "persist_bridge_state": 12.3
#     },
#     "error_rate": 0.002,
#     "pool_efficiency": 0.94
# }
```

## Implementation Phases

### Phase 1: Foundation ✅ (Current)
- [x] Directory structure creation
- [x] Contract YAML definition
- [x] Node skeleton implementation
- [x] Model definitions
- [x] README documentation

### Phase 2: Operation Handlers (Week 1)
- [ ] Implement `_persist_workflow_execution()` (Agent 1)
- [ ] Implement `_persist_workflow_step()` (Agent 2)
- [ ] Implement `_persist_bridge_state()` (Agent 3)
- [ ] Implement `_persist_fsm_transition()` (Agent 4)
- [ ] Implement `_persist_metadata_stamp()` (Agent 5)
- [ ] Implement `_update_node_heartbeat()` (Agent 6)

### Phase 3: Core Integration (Week 1-2)
- [ ] Initialize method implementation (Agent 7)
- [ ] Process method routing logic (Agent 7)
- [ ] Health check implementation (Agent 8)
- [ ] Metrics collection (Agent 8)

### Phase 4: Event Consumer (Week 2)
- [ ] Kafka consumer implementation
- [ ] Event-to-operation mapping
- [ ] Error handling and retry logic
- [ ] Dead letter queue integration

### Phase 5: Testing & Validation (Week 2-3)
- [ ] Unit tests for all operations
- [ ] Integration tests with PostgreSQL
- [ ] Event consumer tests
- [ ] Performance benchmarking
- [ ] Load testing (1000+ events/sec)

## Success Criteria

### Functionality ✅
- All 6 database operations implemented
- Event consumer processing 1000+ events/second
- Zero data loss (at-least-once delivery)
- UUID correlation tracking maintained

### Performance ✅
- Database operations < 10ms (p95)
- Event processing latency < 50ms (p95)
- Connection pool efficiency > 90%
- No memory leaks under sustained load

### Quality ✅
- 100% type coverage with Pydantic models
- Comprehensive error handling with OnexError
- ONEX v2.0 compliance validation
- Contract validation with ModelContractEffect

### Integration ✅
- Seamless integration with bridge nodes
- No breaking changes to event schema
- Backward compatible with PostgreSQL schema
- Works with Docker Compose environment

## References

- **Planning Document**: `docs/planning/DATABASE_ADAPTER_EFFECT_NODE_PLAN.md`
- **Reference Implementation**: `src/omnibase/nodes/node_postgres_adapter_effect/` (omnibase_3 repo)
- **Bridge Nodes**: `src/omninode_bridge/nodes/`
- **ONEX v2.0 Specification**: `docs/onex/` (omnibase_3 repo)

## Contributing

When implementing operation handlers:
1. Follow the TODO markers in `node.py` for phase assignments
2. Use consistent error handling with `OnexError`
3. Maintain UUID correlation tracking
4. Add comprehensive docstrings
5. Include performance metrics tracking
6. Write unit tests before implementation

## License

Part of the omninode_bridge project. See root LICENSE file for details.

---

**Implementation Status**: Phase 1 Complete ✅
**Next Phase**: Operation Handlers (Phase 2)
**Agent Assignment**: 8-agent parallel execution plan
