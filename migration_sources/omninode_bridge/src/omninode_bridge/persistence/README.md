# OmniNode Bridge Persistence Layer

**High-level CRUD operations with transaction support for bridge nodes**

## Overview

The persistence layer provides convenience wrapper functions around the generic CRUD handlers in the `database_adapter_effect` node, offering type-safe, transactional database operations with comprehensive error handling.

### Key Features

- ✅ **Type-Safe Operations**: Uses EntityRegistry and DatabaseAdapterProtocol for strong typing
- ✅ **Protocol-Based Typing**: No `Any` types - all node parameters use DatabaseAdapterProtocol
- ✅ **Transaction Management**: PostgreSQL autocommit for single operations
- ✅ **Error Handling**: OnexError with detailed context
- ✅ **UUID Correlation Tracking**: End-to-end observability
- ✅ **Performance Monitoring**: Execution time tracking and logging
- ✅ **ONEX v2.0 Compliant**: Follows ONEX patterns and conventions

## Architecture

```
┌────────────────────────────────────────────┐
│   Bridge Nodes (Orchestrator, Reducer)    │
│         └─ High-Level CRUD Functions       │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│        Persistence Layer (This Module)     │
│  ┌──────────────┬──────────────────────┐  │
│  │ Bridge State │ Workflow Execution   │  │
│  │ CRUD         │ CRUD                 │  │
│  └──────────────┴──────────────────────┘  │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│     Database Adapter Effect Node           │
│  ┌────────────────────────────────────┐   │
│  │   Generic CRUD Handlers            │   │
│  │   (8 operations: INSERT, UPDATE,   │   │
│  │    DELETE, QUERY, UPSERT, etc.)    │   │
│  └────────────────────────────────────┘   │
└────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│        PostgreSQL Database                 │
│  ┌──────────────┬──────────────────────┐  │
│  │ bridge_states│ workflow_executions  │  │
│  └──────────────┴──────────────────────┘  │
└────────────────────────────────────────────┘
```

## Type Safety with DatabaseAdapterProtocol

All CRUD functions use **DatabaseAdapterProtocol** instead of `Any` for strong typing:

```python
from omninode_bridge.persistence.protocols import DatabaseAdapterProtocol

async def create_bridge_state(
    bridge_id: UUID,
    namespace: str,
    current_fsm_state: str,
    node: DatabaseAdapterProtocol,  # ✅ Strong typing, no Any!
    correlation_id: UUID,
    ...
) -> ModelBridgeState:
    ...
```

### Benefits

1. **Type Safety**: mypy validates node has `process()` method at compile time
2. **IDE Support**: Autocomplete for `process()` method signature
3. **No Circular Imports**: Protocol uses structural subtyping (duck typing)
4. **Runtime Validation**: Optional `isinstance()` checks if needed
5. **Dependency Inversion**: Depends on Protocol interface, not concrete implementations

### Protocol Definition

```python
class DatabaseAdapterProtocol(Protocol):
    """Protocol for database adapter nodes."""

    async def process(
        self,
        operation_input: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """Process database operation."""
        ...

    @property
    def _logger(self) -> Any:  # Optional logger
        """Optional logger instance."""
        ...
```

**See**: `src/omninode_bridge/persistence/protocols.py` for full Protocol definition.

## Modules

### 1. Bridge State CRUD (`bridge_state_crud.py`)

Operations for `ModelBridgeState` entity (maps to `bridge_states` table).

**Functions:**
- `create_bridge_state()` - INSERT new bridge state
- `update_bridge_state()` - UPDATE existing bridge state
- `get_bridge_state()` - QUERY single bridge state by ID
- `list_bridge_states()` - QUERY multiple bridge states with filters
- `delete_bridge_state()` - DELETE bridge state
- `upsert_bridge_state()` - INSERT or UPDATE (ON CONFLICT)

### 2. Workflow Execution CRUD (`workflow_execution_crud.py`)

Operations for `ModelWorkflowExecution` entity (maps to `workflow_executions` table).

**Functions:**
- `create_workflow_execution()` - INSERT new workflow
- `update_workflow_execution()` - UPDATE existing workflow
- `get_workflow_execution()` - QUERY single workflow by correlation_id
- `list_workflow_executions()` - QUERY multiple workflows with filters
- `delete_workflow_execution()` - DELETE workflow

## Transaction Strategy

### Single-Statement Operations

All CRUD operations use **PostgreSQL autocommit** for single-statement atomicity:

- **INSERT**: Automatically wrapped in transaction by PostgreSQL
- **UPDATE**: Atomic single-statement update with WHERE clause
- **DELETE**: Atomic single-statement delete with WHERE clause
- **UPSERT**: Atomic ON CONFLICT DO UPDATE operation

**Rationale**: Single operations don't require explicit transaction management, reducing overhead while maintaining ACID compliance.

### Batch Operations

For batch operations requiring all-or-nothing semantics, use the generic `BATCH_INSERT` handler which provides explicit transaction wrapping:

```python
# Example: Batch insert with transaction
operation_input = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.BATCH_INSERT,
    entity_type=EnumEntityType.BRIDGE_STATE,
    correlation_id=correlation_id,
    batch_entities=[state1, state2, state3, ...]
)
result = await node.process(operation_input)
```

**See**: `docs/TRANSACTION_CONSISTENCY_MODEL.md` for comprehensive transaction documentation.

## Error Handling

All functions follow the ONEX error handling pattern:

### Error Flow

```python
try:
    # 1. Validate inputs
    if not bridge_id:
        raise OnexError(
            error_code=EnumCoreErrorCode.VALIDATION_ERROR,
            message="bridge_id cannot be None",
            context={"operation": "create_bridge_state", ...}
        )

    # 2. Execute operation through generic CRUD handler
    result = await node.process(operation_input)

    # 3. Check operation result
    if not result.success:
        raise OnexError(
            error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
            message=f"Failed: {result.error_message}",
            context={...}
        )

    # 4. Return result
    return entity

except OnexError:
    raise  # Re-raise OnexError as-is
except Exception as e:
    # Wrap unexpected errors
    raise OnexError(
        error_code=EnumCoreErrorCode.INTERNAL_ERROR,
        message=f"Unexpected error: {e!s}",
        context={...},
        original_error=e
    )
```

### Error Types

| Error Code | Raised When |
|------------|-------------|
| `VALIDATION_ERROR` | Invalid inputs (None values, empty strings) |
| `DATABASE_OPERATION_ERROR` | Database operation fails (CREATE, UPDATE, DELETE, UPSERT) |
| `DATABASE_QUERY_ERROR` | Database query fails (GET, LIST) |
| `NOT_FOUND` | Entity not found (GET, UPDATE, DELETE) |
| `INTERNAL_ERROR` | Unexpected exceptions |

## Usage Examples

### Bridge State Operations

#### Create Bridge State

```python
from omninode_bridge.persistence import create_bridge_state
from uuid import uuid4

bridge = await create_bridge_state(
    bridge_id=uuid4(),
    namespace="production",
    current_fsm_state="IDLE",
    total_workflows_processed=0,
    total_items_aggregated=0,
    aggregation_metadata={"version": "1.0.0"},
    node=database_adapter_node,
    correlation_id=uuid4()
)
```

#### Update Bridge State

```python
from omninode_bridge.persistence import update_bridge_state
from datetime import datetime, UTC

updated = await update_bridge_state(
    bridge_id=bridge.bridge_id,
    updates={
        "total_workflows_processed": 100,
        "total_items_aggregated": 1000,
        "current_fsm_state": "PROCESSING",
        "last_aggregation_timestamp": datetime.now(UTC)
    },
    node=database_adapter_node,
    correlation_id=uuid4()
)
```

#### Query Bridge States

```python
from omninode_bridge.persistence import list_bridge_states

# Query by namespace
production_bridges = await list_bridge_states(
    filters={"namespace": "production"},
    node=database_adapter_node,
    correlation_id=uuid4()
)

# Query by FSM state with pagination
active_bridges = await list_bridge_states(
    filters={"current_fsm_state": "PROCESSING"},
    limit=50,
    offset=0,
    sort_by="last_aggregation_timestamp",
    sort_order="desc",
    node=database_adapter_node,
    correlation_id=uuid4()
)
```

#### Upsert Bridge State

```python
from omninode_bridge.persistence import upsert_bridge_state

# Insert if doesn't exist, update if exists
bridge = await upsert_bridge_state(
    bridge_id=bridge_id,
    namespace="production",
    current_fsm_state="PROCESSING",
    total_workflows_processed=200,
    node=database_adapter_node,
    correlation_id=uuid4()
)
```

### Workflow Execution Operations

#### Create Workflow Execution

```python
from omninode_bridge.persistence import create_workflow_execution
from datetime import datetime, UTC

workflow = await create_workflow_execution(
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="PENDING",
    namespace="production",
    metadata={
        "source": "api",
        "user_id": "user_123",
        "api_version": "v1"
    },
    node=database_adapter_node,
    request_correlation_id=uuid4()
)
```

#### Update Workflow Execution

```python
from omninode_bridge.persistence import update_workflow_execution

# Update to PROCESSING
await update_workflow_execution(
    correlation_id=workflow.correlation_id,
    updates={
        "current_state": "PROCESSING",
        "started_at": datetime.now(UTC)
    },
    node=database_adapter_node,
    request_correlation_id=uuid4()
)

# Update to COMPLETED
await update_workflow_execution(
    correlation_id=workflow.correlation_id,
    updates={
        "current_state": "COMPLETED",
        "completed_at": datetime.now(UTC),
        "execution_time_ms": 1234
    },
    node=database_adapter_node,
    request_correlation_id=uuid4()
)
```

#### Query Workflow Executions

```python
from omninode_bridge.persistence import list_workflow_executions

# Query by namespace and state
processing_workflows = await list_workflow_executions(
    filters={
        "namespace": "production",
        "current_state": "PROCESSING"
    },
    node=database_adapter_node,
    request_correlation_id=uuid4()
)

# Query failed workflows in last hour
from datetime import timedelta
one_hour_ago = datetime.now(UTC) - timedelta(hours=1)

failed_workflows = await list_workflow_executions(
    filters={
        "current_state": "FAILED",
        "completed_at__gte": one_hour_ago
    },
    limit=100,
    sort_by="completed_at",
    sort_order="desc",
    node=database_adapter_node,
    request_correlation_id=uuid4()
)
```

## Performance Considerations

### Targets

- **CREATE**: < 10ms (p95)
- **UPDATE**: < 10ms (p95)
- **GET**: < 5ms (p95)
- **QUERY**: < 20ms for 100 results (p95)
- **DELETE**: < 5ms (p95)
- **UPSERT**: < 15ms (p95)

### Optimization Tips

1. **Use Filters**: Always filter queries by indexed columns:
   - `bridge_states`: `bridge_id` (PK), `namespace`, `current_fsm_state`
   - `workflow_executions`: `correlation_id` (unique), `namespace`, `current_state`

2. **Pagination**: Use `limit` and `offset` for large result sets:
   ```python
   page = await list_workflows(limit=50, offset=100, ...)
   ```

3. **Batch Operations**: For multiple inserts, use `BATCH_INSERT`:
   ```python
   # More efficient than individual inserts
   operation_input = ModelDatabaseOperationInput(
       operation_type=EnumDatabaseOperationType.BATCH_INSERT,
       entity_type=EnumEntityType.BRIDGE_STATE,
       batch_entities=[state1, state2, ...]
   )
   ```

4. **Connection Pooling**: Database adapter uses connection pool (20-50 connections)

## Integration with Bridge Nodes

### NodeBridgeOrchestrator Integration

```python
from omninode_bridge.persistence import (
    create_workflow_execution,
    update_workflow_execution
)

class NodeBridgeOrchestrator:
    async def start_workflow(self, request):
        # Create workflow execution record
        workflow = await create_workflow_execution(
            correlation_id=request.correlation_id,
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace=request.namespace,
            started_at=datetime.now(UTC),
            node=self.database_adapter,
            request_correlation_id=uuid4()
        )

        # Execute workflow steps...

        # Update to COMPLETED
        await update_workflow_execution(
            correlation_id=workflow.correlation_id,
            updates={
                "current_state": "COMPLETED",
                "completed_at": datetime.now(UTC),
                "execution_time_ms": execution_time
            },
            node=self.database_adapter,
            request_correlation_id=uuid4()
        )
```

### NodeBridgeReducer Integration

```python
from omninode_bridge.persistence import upsert_bridge_state

class NodeBridgeReducer:
    async def aggregate(self, items):
        # Upsert aggregation state
        bridge_state = await upsert_bridge_state(
            bridge_id=self.bridge_id,
            namespace=self.namespace,
            current_fsm_state="PROCESSING",
            total_workflows_processed=self.workflow_count,
            total_items_aggregated=self.item_count,
            last_aggregation_timestamp=datetime.now(UTC),
            node=self.database_adapter,
            correlation_id=uuid4()
        )
```

## Testing

### Unit Tests

```python
import pytest
from uuid import uuid4
from omninode_bridge.persistence import create_bridge_state

@pytest.mark.asyncio
async def test_create_bridge_state(mock_db_node):
    bridge = await create_bridge_state(
        bridge_id=uuid4(),
        namespace="test",
        current_fsm_state="IDLE",
        node=mock_db_node,
        correlation_id=uuid4()
    )

    assert bridge.namespace == "test"
    assert bridge.current_fsm_state == "IDLE"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_bridge_state_crud_flow(real_db_node):
    bridge_id = uuid4()

    # CREATE
    bridge = await create_bridge_state(
        bridge_id=bridge_id,
        namespace="test",
        current_fsm_state="IDLE",
        node=real_db_node,
        correlation_id=uuid4()
    )

    # READ
    retrieved = await get_bridge_state(
        bridge_id=bridge_id,
        node=real_db_node,
        correlation_id=uuid4()
    )
    assert retrieved.bridge_id == bridge_id

    # UPDATE
    updated = await update_bridge_state(
        bridge_id=bridge_id,
        updates={"current_fsm_state": "PROCESSING"},
        node=real_db_node,
        correlation_id=uuid4()
    )
    assert updated.current_fsm_state == "PROCESSING"

    # DELETE
    success = await delete_bridge_state(
        bridge_id=bridge_id,
        node=real_db_node,
        correlation_id=uuid4()
    )
    assert success is True
```

## Troubleshooting

### Common Issues

#### 1. ValidationError: "entity_type not registered"

**Cause**: Entity type not registered in EntityRegistry

**Solution**: Ensure entity type is added to `EnumEntityType` and registered in `EntityRegistry._ENTITY_MODELS`

#### 2. NOT_FOUND Error on Update/Delete

**Cause**: Entity doesn't exist in database

**Solution**: Use `upsert_bridge_state()` for insert-or-update operations

#### 3. Unique Constraint Violation

**Cause**: Attempting to insert duplicate correlation_id or bridge_id

**Solution**: Use `upsert_bridge_state()` or check if entity exists first

#### 4. Transaction Deadlock

**Cause**: Concurrent updates to same record

**Solution**: Implement retry logic with exponential backoff

## References

- **Protocol Definition**: `src/omninode_bridge/persistence/protocols.py`
- **Generic CRUD Handlers**: `src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/_generic_crud_handlers.py`
- **Transaction Documentation**: `docs/TRANSACTION_CONSISTENCY_MODEL.md`
- **Entity Registry**: `src/omninode_bridge/infrastructure/entity_registry.py`
- **Database Schema**: `migrations/`

## License

Copyright © 2025 OmniNode Bridge Project
