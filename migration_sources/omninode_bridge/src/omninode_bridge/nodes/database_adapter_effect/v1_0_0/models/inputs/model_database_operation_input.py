#!/usr/bin/env python3
"""
ModelDatabaseOperationInput - Strongly-Typed Database Operation Input.

This is the primary input model for the database adapter effect node,
providing operation routing and correlation tracking with strongly-typed
entity support for type-safe database operations.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelDatabaseOperationInput
- UUID correlation tracking across operations
- Strongly-typed entity validation via EntityUnion
- Generic CRUD pattern with EnumDatabaseOperationType
- Comprehensive field validation with Pydantic v2

Migration from dict[str, Any]:
This model now uses EntityUnion for strongly-typed entity data instead of
the previous dict[str, Any] fields (workflow_execution_data, etc.).

Before (dict-based):
    input_data = ModelDatabaseOperationInput(
        operation_type="persist_workflow_execution",
        correlation_id=uuid4(),
        workflow_execution_data={"workflow_type": "...", ...}  # dict[str, Any]
    )

After (strongly-typed):
    from models.entities import ModelWorkflowExecution
    workflow = ModelWorkflowExecution(
        correlation_id=uuid4(),
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="production"
    )
    input_data = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.INSERT,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        entity=workflow  # Strongly typed!
    )
"""

from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Import EntityUnion from entities module
from ..entities import EntityUnion


class ModelDatabaseOperationInput(BaseModel):
    """
    Strongly-typed input model for database adapter effect node operations.

    This model serves as the primary routing mechanism for all database
    operations. It uses strongly-typed EntityUnion for type-safe entity
    validation and manipulation.

    Generic CRUD Pattern:
    - INSERT: Create new record with entity field
    - UPDATE: Update records matching query_filters with entity data
    - DELETE: Delete records matching query_filters
    - QUERY: Retrieve records matching query_filters with pagination
    - UPSERT: Insert or update record with entity field
    - BATCH_INSERT: Create multiple records with batch_entities
    - BATCH_UPDATE: Update multiple records with batch_entities
    - BATCH_DELETE: Delete multiple records with batch_entities
    - COUNT: Count records matching query_filters
    - EXISTS: Check if records exist matching query_filters
    - HEALTH_CHECK: Verify database connectivity

    Entity Types (via EntityUnion):
    - ModelWorkflowExecution: Workflow execution records
    - ModelWorkflowStep: Workflow step history
    - ModelBridgeState: Bridge aggregation state
    - ModelFSMTransition: FSM state transition records
    - ModelMetadataStamp: Metadata stamp audit records
    - ModelNodeHeartbeat: Node heartbeat and registration

    Example (INSERT with strongly-typed entity):
        >>> from uuid import uuid4
        >>> from entities import ModelWorkflowExecution
        >>> from enums.enum_database_operation_type import EnumDatabaseOperationType
        >>> from enums.enum_entity_type import EnumEntityType
        >>>
        >>> workflow = ModelWorkflowExecution(
        ...     correlation_id=uuid4(),
        ...     workflow_type="metadata_stamping",
        ...     current_state="PROCESSING",
        ...     namespace="production",
        ...     started_at=datetime.now(timezone.utc)
        ... )
        >>>
        >>> input_data = ModelDatabaseOperationInput(
        ...     operation_type=EnumDatabaseOperationType.INSERT,
        ...     entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        ...     correlation_id=uuid4(),
        ...     entity=workflow  # ✅ Strongly typed!
        ... )

    Example (QUERY with filters and pagination):
        >>> query_input = ModelDatabaseOperationInput(
        ...     operation_type=EnumDatabaseOperationType.QUERY,
        ...     entity_type=EnumEntityType.METADATA_STAMP,
        ...     correlation_id=uuid4(),
        ...     query_filters={"namespace": "production"},
        ...     sort_by="created_at",
        ...     sort_order="desc",
        ...     limit=100,
        ...     offset=0
        ... )

    Example (BATCH_INSERT with multiple entities):
        >>> from entities import ModelWorkflowStep
        >>> steps = [
        ...     ModelWorkflowStep(
        ...         workflow_id=uuid4(),
        ...         step_name="hash_generation",
        ...         step_order=1,
        ...         status="COMPLETED"
        ...     ),
        ...     ModelWorkflowStep(
        ...         workflow_id=uuid4(),
        ...         step_name="stamp_creation",
        ...         step_order=2,
        ...         status="COMPLETED"
        ...     )
        ... ]
        >>>
        >>> batch_input = ModelDatabaseOperationInput(
        ...     operation_type=EnumDatabaseOperationType.BATCH_INSERT,
        ...     entity_type=EnumEntityType.WORKFLOW_STEP,
        ...     correlation_id=uuid4(),
        ...     batch_entities=steps  # ✅ List of strongly-typed entities
        ... )

    Example (UPDATE with entity and filters):
        >>> from entities import ModelBridgeState
        >>> updated_state = ModelBridgeState(
        ...     bridge_id=uuid4(),
        ...     namespace="production",
        ...     total_workflows_processed=200,
        ...     total_items_aggregated=1000,
        ...     current_fsm_state="idle"
        ... )
        >>>
        >>> update_input = ModelDatabaseOperationInput(
        ...     operation_type=EnumDatabaseOperationType.UPDATE,
        ...     entity_type=EnumEntityType.BRIDGE_STATE,
        ...     correlation_id=uuid4(),
        ...     query_filters={"bridge_id": str(uuid4())},
        ...     entity=updated_state  # ✅ Strongly typed update data
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Operation Routing ===
    operation_type: str = Field(
        ...,
        description="""
        Type of database operation to perform (from EnumDatabaseOperationType).

        Generic CRUD operations:
        - insert: Create new record
        - update: Update existing records
        - delete: Delete records
        - query: Retrieve records
        - upsert: Insert or update record
        - batch_insert: Create multiple records
        - batch_update: Update multiple records
        - batch_delete: Delete multiple records
        - count: Count matching records
        - exists: Check if records exist
        - health_check: Database health check

        This field routes the input to the appropriate handler method
        in the database adapter effect node.
        """,
    )

    # === Entity Type (Required for Generic Pattern) ===
    entity_type: str = Field(
        ...,
        description="""
        Type of database entity to operate on (from EnumEntityType).

        Valid entity types:
        - workflow_execution: Workflow execution records
        - workflow_step: Workflow step history
        - bridge_state: Bridge aggregation state
        - fsm_transition: FSM state transition records
        - metadata_stamp: Metadata stamp audit records
        - node_heartbeat: Node heartbeat and registration

        Combined with operation_type, this determines the exact database
        operation to perform (e.g., INSERT + workflow_execution).
        """,
    )

    # === Correlation Tracking ===
    correlation_id: UUID = Field(
        ...,
        description="""
        UUID correlation identifier for request tracking.

        This ID is preserved across the entire workflow, from the original
        API request through Kafka events to database persistence. It enables
        distributed tracing and end-to-end observability.
        """,
    )

    # === Strongly-Typed Entity Data ===
    entity: Optional[EntityUnion] = Field(
        default=None,
        description="""
        Strongly-typed entity instance for INSERT/UPDATE/UPSERT operations.

        This field replaces the previous dict[str, Any] fields with proper
        Pydantic models for type safety and validation.

        Type must match entity_type field:
        - entity_type=WORKFLOW_EXECUTION → entity: ModelWorkflowExecution
        - entity_type=WORKFLOW_STEP → entity: ModelWorkflowStep
        - entity_type=BRIDGE_STATE → entity: ModelBridgeState
        - entity_type=FSM_TRANSITION → entity: ModelFSMTransition
        - entity_type=METADATA_STAMP → entity: ModelMetadataStamp
        - entity_type=NODE_HEARTBEAT → entity: ModelNodeHeartbeat

        Required for:
        - INSERT: New entity to create
        - UPDATE: Fields to update (partial entity)
        - UPSERT: Entity to insert or update

        Optional/Unused for:
        - DELETE: Use query_filters only
        - QUERY: Use query_filters for filtering
        - COUNT/EXISTS: Use query_filters for filtering
        - BATCH operations: Use batch_entities instead

        Migration from dict[str, Any]:
            Before: workflow_execution_data={"workflow_type": "...", ...}
            After: entity=ModelWorkflowExecution(workflow_type="...", ...)
        """,
    )

    # === Batch Entity Data ===
    batch_entities: Optional[list[EntityUnion]] = Field(
        default=None,
        description="""
        List of strongly-typed entities for batch operations.

        Required for:
        - BATCH_INSERT: List of entities to insert
        - BATCH_UPDATE: List of entities with update data
        - BATCH_DELETE: Not used (use query_filters in batch)

        Each entity in the list must match the entity_type field.

        Example:
            batch_entities=[
                ModelWorkflowStep(step_name="step1", ...),
                ModelWorkflowStep(step_name="step2", ...),
            ]
        """,
    )

    # === Query Filters ===
    query_filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="""
        Dictionary of filter conditions for UPDATE/DELETE/QUERY operations.

        Used to identify which records to operate on. Supports various
        comparison operators and field matching.

        Required for:
        - UPDATE: Identify records to update
        - DELETE: Identify records to delete
        - UPSERT: Identify conflict field (unique constraint)
        - EXISTS: Check if matching records exist

        Optional for:
        - QUERY: Filter results (if None, returns all records)
        - COUNT: Filter counts (if None, counts all records)

        Example (simple equality):
            query_filters={"namespace": "production", "status": "PROCESSING"}

        Example (complex filters):
            query_filters={
                "namespace": "production",
                "created_at__gte": "2025-10-01",  // >= operator
                "execution_time_ms__lt": 1000      // < operator
            }

        Example (IN operator):
            query_filters={
                "status__in": ["PROCESSING", "PENDING"]
            }
        """,
    )

    # === Query Pagination and Sorting ===
    sort_by: Optional[str] = Field(
        default=None,
        description="""
        Field name to sort query results by.

        Only used for QUERY operations. Defaults to primary key if not specified.

        Example: sort_by="created_at"
        """,
    )

    sort_order: Optional[Literal["asc", "desc"]] = Field(
        default="desc",
        description="""
        Sort order for query results.

        - "asc": Ascending order (oldest first)
        - "desc": Descending order (newest first, default)

        Only used for QUERY operations.
        """,
    )

    limit: Optional[int] = Field(
        default=None,
        description="""
        Maximum number of records to return for QUERY operations.

        If not specified, returns all matching records (use with caution
        on large datasets).

        Example: limit=100  # Return first 100 records
        """,
        ge=1,
        le=10000,
    )

    offset: Optional[int] = Field(
        default=None,
        description="""
        Number of records to skip for QUERY operations (pagination).

        Used with limit for paginated queries.

        Example: offset=100, limit=100  # Return records 101-200
        """,
        ge=0,
    )

    # === JSONB Validation Control ===
    strict_jsonb_validation: bool = Field(
        default=True,
        description="""
        Control JSONB field deserialization behavior for QUERY operations.

        When True (default - fail-fast):
        - Corrupted JSONB data raises OnexError immediately
        - Operation fails with detailed context
        - Ensures data integrity at cost of availability

        When False (graceful degradation):
        - Corrupted JSONB data logs warning and continues
        - Field remains as raw string value
        - Maintains availability at cost of data consistency

        Use Cases:
        - True: Production queries requiring data integrity
        - False: Recovery operations, debugging, or non-critical queries

        Example:
            # Fail-fast (default)
            query_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.QUERY,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                strict_jsonb_validation=True  # Raise error on corrupted JSONB
            )

            # Graceful degradation
            recovery_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.QUERY,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                strict_jsonb_validation=False  # Continue with raw string on corruption
            )
        """,
    )
