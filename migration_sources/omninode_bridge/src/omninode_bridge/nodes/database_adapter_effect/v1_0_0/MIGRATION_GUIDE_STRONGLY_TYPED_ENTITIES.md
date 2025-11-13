# Migration Guide: Strongly-Typed Entity Pattern

**Date**: October 8, 2025
**Agent**: Agent 4 - Input Model Refactoring
**Status**: Phase 2 Complete - Input Model Updated with EntityUnion
**Location**: src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/

## Overview

This migration guide explains how to transition from the dict[str, Any] pattern to the strongly-typed EntityUnion pattern for database operations in the Database Adapter Effect Node.

## What Changed

### Before (dict[str, Any] Pattern)

```python
# Old pattern - Using dict[str, Any] for entity data
from uuid import uuid4
from models.inputs.model_database_operation_input import ModelDatabaseOperationInput

input_data = ModelDatabaseOperationInput(
    operation_type="persist_workflow_execution",
    correlation_id=uuid4(),
    workflow_execution_data={  # ❌ dict[str, Any] - no type safety
        "workflow_type": "metadata_stamping",
        "current_state": "PROCESSING",
        "namespace": "production",
        "started_at": "2025-10-08T12:00:00Z"
    }
)
```

**Problems with this approach:**
- ❌ No type checking at development time
- ❌ No IDE autocomplete for entity fields
- ❌ No validation until runtime (in handler)
- ❌ Easy to introduce typos in field names
- ❌ Unclear what fields are required vs optional
- ❌ Difficult to refactor when schema changes

### After (Strongly-Typed EntityUnion Pattern)

```python
# New pattern - Using strongly-typed entity models
from uuid import uuid4
from datetime import datetime
from models.inputs.model_database_operation_input import ModelDatabaseOperationInput
from models.entities import ModelWorkflowExecution
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType

# Create strongly-typed entity
workflow = ModelWorkflowExecution(  # ✅ Strong typing with Pydantic
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="PROCESSING",
    namespace="production",
    started_at=datetime.utcnow()
)

# Use in database operation input
input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.INSERT,
    entity_type=EnumEntityType.WORKFLOW_EXECUTION,
    correlation_id=uuid4(),
    entity=workflow  # ✅ Strongly typed!
)
```

**Benefits:**
- ✅ Type checking at development time (mypy, IDE)
- ✅ IDE autocomplete for all entity fields
- ✅ Automatic Pydantic validation on creation
- ✅ Typos caught immediately by type checker
- ✅ Clear contracts - required vs optional fields
- ✅ Easy refactoring with IDE support

## Migration Steps

### Step 1: Import Entity Models

Replace old input model imports with entity models:

```python
# Before
from models.inputs.model_workflow_execution_input import ModelWorkflowExecutionInput

# After
from models.entities import ModelWorkflowExecution, EntityUnion
from enums.enum_entity_type import EnumEntityType
```

### Step 2: Create Entity Instances

Replace dict creation with entity model instantiation:

```python
# Before (dict-based)
workflow_data = {
    "workflow_type": "metadata_stamping",
    "current_state": "PROCESSING",
    "namespace": "production"
}

# After (strongly-typed)
from models.entities import ModelWorkflowExecution
workflow = ModelWorkflowExecution(
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="PROCESSING",
    namespace="production",
    started_at=datetime.utcnow()
)
```

### Step 3: Update Database Operation Input

Use the new entity and entity_type fields:

```python
# Before
input_data = ModelDatabaseOperationInput(
    operation_type="persist_workflow_execution",
    correlation_id=correlation_id,
    workflow_execution_data=workflow_data  # Old dict field
)

# After
input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.INSERT,
    entity_type=EnumEntityType.WORKFLOW_EXECUTION,
    correlation_id=correlation_id,
    entity=workflow  # New strongly-typed field
)
```

## Entity Type Mapping

| Old Field Name | New Entity Type | Entity Model |
|----------------|-----------------|--------------|
| workflow_execution_data | WORKFLOW_EXECUTION | ModelWorkflowExecution |
| workflow_step_data | WORKFLOW_STEP | ModelWorkflowStep |
| bridge_state_data | BRIDGE_STATE | ModelBridgeState |
| fsm_transition_data | FSM_TRANSITION | ModelFSMTransition |
| metadata_stamp_data | METADATA_STAMP | ModelMetadataStamp |
| node_heartbeat_data | NODE_HEARTBEAT | ModelNodeHeartbeat |

## Migration Examples

### Example 1: Workflow Execution INSERT

```python
# Before (dict pattern)
from uuid import uuid4

input_data = ModelDatabaseOperationInput(
    operation_type="persist_workflow_execution",
    correlation_id=uuid4(),
    workflow_execution_data={
        "workflow_type": "metadata_stamping",
        "current_state": "PROCESSING",
        "namespace": "production",
        "started_at": "2025-10-08T12:00:00Z",
        "metadata": {"source": "api"}
    }
)

# After (strongly-typed pattern)
from uuid import uuid4
from datetime import datetime
from models.entities import ModelWorkflowExecution
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType

workflow = ModelWorkflowExecution(
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="PROCESSING",
    namespace="production",
    started_at=datetime.utcnow(),
    metadata={"source": "api"}
)

input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.INSERT,
    entity_type=EnumEntityType.WORKFLOW_EXECUTION,
    correlation_id=uuid4(),
    entity=workflow
)
```

### Example 2: Workflow Step BATCH_INSERT

```python
# Before (dict pattern)
steps_data = [
    {
        "workflow_id": str(workflow_id),
        "step_name": "hash_generation",
        "step_order": 1,
        "status": "COMPLETED",
        "execution_time_ms": 2
    },
    {
        "workflow_id": str(workflow_id),
        "step_name": "stamp_creation",
        "step_order": 2,
        "status": "COMPLETED",
        "execution_time_ms": 8
    }
]

input_data = ModelDatabaseOperationInput(
    operation_type="batch_insert_workflow_steps",  # Old specific operation
    correlation_id=correlation_id,
    workflow_step_data=steps_data  # Old dict list
)

# After (strongly-typed pattern)
from models.entities import ModelWorkflowStep
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType

steps = [
    ModelWorkflowStep(
        workflow_id=workflow_id,
        step_name="hash_generation",
        step_order=1,
        status="COMPLETED",
        execution_time_ms=2
    ),
    ModelWorkflowStep(
        workflow_id=workflow_id,
        step_name="stamp_creation",
        step_order=2,
        status="COMPLETED",
        execution_time_ms=8
    )
]

input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.BATCH_INSERT,
    entity_type=EnumEntityType.WORKFLOW_STEP,
    correlation_id=correlation_id,
    batch_entities=steps  # Strongly-typed list
)
```

### Example 3: Bridge State UPSERT

```python
# Before (dict pattern)
input_data = ModelDatabaseOperationInput(
    operation_type="persist_bridge_state",
    correlation_id=correlation_id,
    bridge_state_data={
        "bridge_id": str(bridge_id),
        "namespace": "production",
        "total_workflows_processed": 200,
        "total_items_aggregated": 1000,
        "current_fsm_state": "idle",
        "aggregation_metadata": {"file_types": ["jpeg", "pdf"]}
    }
)

# After (strongly-typed pattern)
from models.entities import ModelBridgeState
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType

bridge_state = ModelBridgeState(
    bridge_id=bridge_id,
    namespace="production",
    total_workflows_processed=200,
    total_items_aggregated=1000,
    current_fsm_state="idle",
    aggregation_metadata={"file_types": ["jpeg", "pdf"]},
    last_aggregation_timestamp=datetime.utcnow()
)

input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.UPSERT,
    entity_type=EnumEntityType.BRIDGE_STATE,
    correlation_id=correlation_id,
    query_filters={"bridge_id": str(bridge_id)},  # For conflict detection
    entity=bridge_state
)
```

### Example 4: Metadata Stamp QUERY

```python
# Before (dict pattern)
input_data = ModelDatabaseOperationInput(
    operation_type="query_metadata_stamps",
    correlation_id=correlation_id,
    # No entity data needed for queries
    query_filters={"namespace": "production"},
    limit=100
)

# After (strongly-typed pattern)
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType

input_data = ModelDatabaseOperationInput(
    operation_type=EnumDatabaseOperationType.QUERY,
    entity_type=EnumEntityType.METADATA_STAMP,
    correlation_id=correlation_id,
    query_filters={"namespace": "production"},
    sort_by="created_at",
    sort_order="desc",
    limit=100,
    offset=0
)

# Query results will be returned as list[ModelMetadataStamp]
# for type-safe result handling
```

## Breaking Changes

### Removed Fields

The following fields have been removed from `ModelDatabaseOperationInput`:

- ❌ `workflow_execution_data: dict[str, Any]`
- ❌ `workflow_step_data: dict[str, Any]`
- ❌ `bridge_state_data: dict[str, Any]`
- ❌ `fsm_transition_data: dict[str, Any]`
- ❌ `metadata_stamp_data: dict[str, Any]`
- ❌ `node_heartbeat_data: dict[str, Any]`

### Added Fields

New strongly-typed fields:

- ✅ `entity_type: str` (required) - From EnumEntityType
- ✅ `entity: Optional[EntityUnion]` - Strongly-typed entity instance
- ✅ `batch_entities: Optional[list[EntityUnion]]` - For batch operations
- ✅ `query_filters: Optional[dict[str, Any]]` - For UPDATE/DELETE/QUERY
- ✅ `sort_by: Optional[str]` - For QUERY sorting
- ✅ `sort_order: Optional[Literal["asc", "desc"]]` - Sort direction
- ✅ `limit: Optional[int]` - Query result limit
- ✅ `offset: Optional[int]` - Query pagination offset

### Operation Type Changes

Old specific operation types have been replaced with generic CRUD operations:

| Old Operation Type | New Operation Type | Entity Type |
|--------------------|-------------------|-------------|
| persist_workflow_execution | INSERT | WORKFLOW_EXECUTION |
| persist_workflow_step | INSERT | WORKFLOW_STEP |
| persist_bridge_state | UPSERT | BRIDGE_STATE |
| persist_fsm_transition | INSERT | FSM_TRANSITION |
| persist_metadata_stamp | INSERT | METADATA_STAMP |
| update_node_heartbeat | UPSERT | NODE_HEARTBEAT |
| query_metadata_stamps | QUERY | METADATA_STAMP |
| (new) batch_insert_steps | BATCH_INSERT | WORKFLOW_STEP |

## Entity Model Reference

### ModelWorkflowExecution

```python
from models.entities import ModelWorkflowExecution
from uuid import uuid4
from datetime import datetime

workflow = ModelWorkflowExecution(
    # Database-generated (optional for INSERT)
    id=None,  # Auto-generated by database
    created_at=None,  # Auto-managed by database
    updated_at=None,  # Auto-managed by database

    # Required fields
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="PROCESSING",  # PENDING, PROCESSING, COMPLETED, FAILED
    namespace="production",

    # Optional fields
    started_at=datetime.utcnow(),
    completed_at=None,
    execution_time_ms=None,
    error_message=None,
    metadata={"source": "api"}
)
```

### ModelWorkflowStep

```python
from models.entities import ModelWorkflowStep

step = ModelWorkflowStep(
    # Database-generated
    id=None,
    created_at=None,
    updated_at=None,

    # Required fields
    workflow_id=uuid4(),
    step_name="generate_blake3_hash",
    step_order=1,
    status="COMPLETED",  # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED

    # Optional fields
    execution_time_ms=2,
    step_data={"file_hash": "abc123...", "performance_grade": "A"},
    error_message=None
)
```

### ModelBridgeState

```python
from models.entities import ModelBridgeState

bridge_state = ModelBridgeState(
    # Database-generated
    id=None,
    created_at=None,
    updated_at=None,

    # Required fields
    bridge_id=uuid4(),
    namespace="production",
    total_workflows_processed=150,
    total_items_aggregated=750,
    current_fsm_state="aggregating",  # idle, active, aggregating, persisting

    # Optional fields
    aggregation_metadata={"file_types": ["jpeg", "pdf"]},
    last_aggregation_timestamp=datetime.utcnow()
)
```

### ModelFSMTransition

```python
from models.entities import ModelFSMTransition

transition = ModelFSMTransition(
    # Database-generated
    id=None,
    created_at=None,

    # Required fields
    entity_id=uuid4(),
    entity_type="workflow",  # workflow, bridge_reducer, node_registry
    to_state="COMPLETED",
    transition_event="workflow_completed",

    # Optional fields
    from_state="PROCESSING",  # None for initial state
    transition_data={"execution_time_ms": 1234}
)
```

### ModelMetadataStamp

```python
from models.entities import ModelMetadataStamp

stamp = ModelMetadataStamp(
    # Database-generated
    id=None,
    created_at=None,
    updated_at=None,

    # Required fields
    file_hash="abc123def456...",  # 64-128 char hex
    stamp_data={"stamp_type": "inline", "file_size_bytes": 1024},
    namespace="production",

    # Optional fields
    workflow_id=uuid4()  # Can be None for direct stamping
)
```

### ModelNodeHeartbeat

```python
from models.entities import ModelNodeHeartbeat

heartbeat = ModelNodeHeartbeat(
    # Database-generated
    id=None,
    created_at=None,
    updated_at=None,

    # Required fields
    node_id="orchestrator-01",
    health_status="HEALTHY",  # HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN, OFFLINE

    # Optional fields
    metadata={
        "version": "1.0.0",
        "uptime_seconds": 3600,
        "active_workflows": 42
    },
    last_heartbeat=datetime.utcnow()
)
```

## Type Safety Benefits

### 1. Development-Time Type Checking

```python
# ✅ IDE catches typos immediately
workflow = ModelWorkflowExecution(
    correlation_id=uuid4(),
    workflow_typ="metadata_stamping"  # ❌ Typo! IDE shows error
)

# ✅ IDE suggests valid fields
workflow = ModelWorkflowExecution(
    correlation_id=uuid4(),
    workflow_  # IDE autocomplete shows: workflow_type
)
```

### 2. Automatic Validation

```python
# ✅ Pydantic validates on creation
workflow = ModelWorkflowExecution(
    correlation_id=uuid4(),
    workflow_type="metadata_stamping",
    current_state="INVALID_STATE"  # ❌ Validation error!
)
# ValidationError: current_state must be PENDING, PROCESSING, COMPLETED, or FAILED
```

### 3. Clear Contracts

```python
# ✅ Clear which fields are required vs optional
class ModelWorkflowExecution(BaseModel):
    # Required (no default)
    correlation_id: UUID
    workflow_type: str
    current_state: str
    namespace: str

    # Optional (has default)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None
```

### 4. IDE Autocomplete

```python
# ✅ IDE knows all available fields
workflow = ModelWorkflowExecution(...)

# Type workflow. and IDE shows:
# - correlation_id
# - workflow_type
# - current_state
# - namespace
# - started_at
# - completed_at
# - etc.
```

## Testing with Strongly-Typed Entities

```python
import pytest
from uuid import uuid4
from datetime import datetime
from models.entities import ModelWorkflowExecution
from models.inputs.model_database_operation_input import ModelDatabaseOperationInput
from enums.enum_database_operation_type import EnumDatabaseOperationType
from enums.enum_entity_type import EnumEntityType


def test_create_workflow_execution_with_entity():
    """Test INSERT operation with strongly-typed entity."""

    # Create entity
    workflow = ModelWorkflowExecution(
        correlation_id=uuid4(),
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="test_app",
        started_at=datetime.utcnow()
    )

    # Create operation input
    input_data = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.INSERT,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        entity=workflow
    )

    # Validate types
    assert isinstance(input_data.entity, ModelWorkflowExecution)
    assert input_data.operation_type == EnumDatabaseOperationType.INSERT
    assert input_data.entity_type == EnumEntityType.WORKFLOW_EXECUTION


def test_entity_validation_catches_invalid_data():
    """Test that Pydantic validation catches invalid entity data."""

    with pytest.raises(ValidationError) as exc_info:
        workflow = ModelWorkflowExecution(
            correlation_id="not-a-uuid",  # ❌ Should be UUID
            workflow_type="",  # ❌ min_length=1
            current_state="X" * 100,  # ❌ max_length=50
            namespace="production"
        )

    # Validation error contains all issues
    errors = exc_info.value.errors()
    assert len(errors) == 3  # Three validation failures
```

## Rollout Strategy

### Phase 1: ✅ Complete (Agent 1-3)
- Create entity models (ModelWorkflowExecution, etc.)
- Create EntityUnion type
- Create entity __init__.py with exports

### Phase 2: ✅ Complete (Agent 4)
- Update ModelDatabaseOperationInput with EntityUnion
- Remove old dict[str, Any] fields
- Add entity, batch_entities, query_filters fields
- Create migration guide (this document)

### Phase 3: Pending (Agent 5-8)
- Update node.py handlers to use strongly-typed entities
- Update Kafka event consumers to create entity instances
- Update tests to use entity models
- Validate end-to-end with integration tests

## FAQ

**Q: Can I still use dict[str, Any] for entity data?**
A: No, the dict[str, Any] fields have been removed. You must use strongly-typed entity models.

**Q: What if I need to add a new field to an entity?**
A: Add the field to the entity model (e.g., ModelWorkflowExecution). This will automatically propagate to all usages.

**Q: How do I handle database-generated fields like `id` and `created_at`?**
A: These fields are Optional in entity models. Set them to None when creating new records. They will be populated when reading from the database.

**Q: Can I use partial entities for UPDATE operations?**
A: Yes, create an entity with only the fields you want to update. The handler will use `model_dump(exclude_none=True)` to get only the provided fields.

**Q: What about backward compatibility?**
A: This is a breaking change. All code using the old dict[str, Any] pattern must be updated. See migration examples above.

## Summary

The strongly-typed EntityUnion pattern provides:

- ✅ **Type Safety**: Catch errors at development time, not runtime
- ✅ **IDE Support**: Autocomplete, go-to-definition, refactoring
- ✅ **Validation**: Automatic Pydantic validation on entity creation
- ✅ **Clarity**: Clear contracts for required vs optional fields
- ✅ **Maintainability**: Easy to refactor when schema changes
- ✅ **Documentation**: Self-documenting with type hints

This migration improves code quality, reduces bugs, and enhances developer experience throughout the database adapter effect node.
