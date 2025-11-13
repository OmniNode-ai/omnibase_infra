"""
Bridge State CRUD Operations.

High-level persistence functions for ModelBridgeState with transaction support.
Uses generic CRUD handlers from database adapter for type-safe operations.

ONEX v2.0 Compliance:
- Transaction-wrapped database operations
- OnexError exception handling
- UUID correlation tracking
- Type-safe entity validation

Usage Example:
    >>> from uuid import uuid4
    >>> from datetime import datetime, UTC
    >>>
    >>> # Create bridge state
    >>> bridge_state = await create_bridge_state(
    ...     bridge_id=uuid4(),
    ...     namespace="production",
    ...     current_fsm_state="IDLE",
    ...     node=database_adapter_node,
    ...     correlation_id=uuid4()
    ... )
    >>>
    >>> # Update aggregation counters
    >>> updated = await update_bridge_state(
    ...     bridge_id=bridge_state.bridge_id,
    ...     updates={
    ...         "total_workflows_processed": 100,
    ...         "total_items_aggregated": 1000,
    ...         "last_aggregation_timestamp": datetime.now(UTC)
    ...     },
    ...     node=database_adapter_node,
    ...     correlation_id=uuid4()
    ... )
    >>>
    >>> # Query by namespace
    >>> states = await list_bridge_states(
    ...     filters={"namespace": "production"},
    ...     node=database_adapter_node,
    ...     correlation_id=uuid4()
    ... )
"""

import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.infrastructure.entities.model_bridge_state import ModelBridgeState
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)
from omninode_bridge.persistence.protocols import DatabaseAdapterProtocol

# Aliases for compatibility
OnexError = ModelOnexError


async def create_bridge_state(
    bridge_id: UUID,
    namespace: str,
    current_fsm_state: str,
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
    total_workflows_processed: int = 0,
    total_items_aggregated: int = 0,
    aggregation_metadata: Optional[dict[str, Any]] = None,
    last_aggregation_timestamp: Optional[datetime] = None,
) -> ModelBridgeState:
    """
    Create new bridge state record.

    Uses INSERT operation with transaction support. Raises OnexError if
    bridge_id already exists.

    Args:
        bridge_id: Unique identifier for bridge instance
        namespace: Multi-tenant namespace
        current_fsm_state: Initial FSM state
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking
        total_workflows_processed: Initial workflow count (default: 0)
        total_items_aggregated: Initial item count (default: 0)
        aggregation_metadata: Additional metadata (default: {})
        last_aggregation_timestamp: Initial timestamp (default: None)

    Returns:
        Created ModelBridgeState entity

    Raises:
        OnexError: If validation fails, insert fails, or bridge_id exists

    Example:
        >>> bridge = await create_bridge_state(
        ...     bridge_id=uuid4(),
        ...     namespace="prod",
        ...     current_fsm_state="IDLE",
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not bridge_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="bridge_id cannot be None",
                context={
                    "operation": "create_bridge_state",
                    "correlation_id": str(correlation_id),
                },
            )

        if not namespace or not namespace.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="namespace cannot be empty",
                context={
                    "operation": "create_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        if not current_fsm_state or not current_fsm_state.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="current_fsm_state cannot be empty",
                context={
                    "operation": "create_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 2: Build entity
        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace.strip(),
            current_fsm_state=current_fsm_state.strip(),
            total_workflows_processed=total_workflows_processed,
            total_items_aggregated=total_items_aggregated,
            aggregation_metadata=aggregation_metadata or {},
            last_aggregation_timestamp=last_aggregation_timestamp,
        )

        # Step 3: Execute INSERT through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            entity=bridge_state,
        )

        result = await node.process(operation_input)

        # Step 4: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to create bridge state: {result.error_message}",
                context={
                    "operation": "create_bridge_state",
                    "bridge_id": str(bridge_id),
                    "namespace": namespace,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                operation_type="create_bridge_state",
                additional_context={
                    "bridge_id": str(bridge_id),
                    "namespace": namespace,
                    "fsm_state": current_fsm_state,
                },
            )

        return bridge_state

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error creating bridge state: {e!s}",
            context={
                "operation": "create_bridge_state",
                "bridge_id": str(bridge_id) if bridge_id else None,
                "namespace": namespace,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )


async def update_bridge_state(
    bridge_id: UUID,
    updates: dict[str, Any],
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
) -> ModelBridgeState:
    """
    Update existing bridge state record.

    Uses UPDATE operation with WHERE clause on bridge_id. Only updates
    provided fields, leaving others unchanged.

    Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.

    Args:
        bridge_id: Bridge identifier to update
        updates: Dictionary of fields to update (e.g., {"total_workflows_processed": 100})
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking

    Returns:
        Updated ModelBridgeState entity

    Raises:
        OnexError: If validation fails, update fails, or bridge not found

    Example:
        >>> updated = await update_bridge_state(
        ...     bridge_id=bridge_id,
        ...     updates={
        ...         "total_workflows_processed": 150,
        ...         "current_fsm_state": "PROCESSING",
        ...     },
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not bridge_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="bridge_id cannot be None",
                context={
                    "operation": "update_bridge_state",
                    "correlation_id": str(correlation_id),
                },
            )

        if not updates:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="updates dictionary cannot be empty",
                context={
                    "operation": "update_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 2: Fetch existing state
        existing_state = await get_bridge_state(
            bridge_id=bridge_id, node=node, correlation_id=correlation_id
        )

        # Step 3: Apply updates to existing state
        updated_data = existing_state.model_dump(exclude={"created_at", "updated_at"})
        updated_data.update(updates)

        # Step 4: Validate updated entity
        updated_state = ModelBridgeState(**updated_data)

        # Step 5: Execute UPDATE through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            entity=updated_state,
            query_filters={"bridge_id": bridge_id},
        )

        result = await node.process(operation_input)

        # Step 6: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to update bridge state: {result.error_message}",
                context={
                    "operation": "update_bridge_state",
                    "bridge_id": str(bridge_id),
                    "updates": updates,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        if result.rows_affected == 0:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Bridge state not found: {bridge_id}",
                context={
                    "operation": "update_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=result.rows_affected,
                operation_type="update_bridge_state",
                additional_context={
                    "bridge_id": str(bridge_id),
                    "fields_updated": list(updates.keys()),
                },
            )

        return updated_state

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error updating bridge state: {e!s}",
            context={
                "operation": "update_bridge_state",
                "bridge_id": str(bridge_id) if bridge_id else None,
                "updates": updates,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )


async def get_bridge_state(
    bridge_id: UUID,
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
) -> ModelBridgeState:
    """
    Retrieve bridge state by bridge_id.

    Uses QUERY operation with WHERE clause on bridge_id.

    Args:
        bridge_id: Bridge identifier to retrieve
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking

    Returns:
        ModelBridgeState entity

    Raises:
        OnexError: If validation fails, query fails, or bridge not found

    Example:
        >>> state = await get_bridge_state(
        ...     bridge_id=bridge_id,
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
        >>> print(state.namespace, state.current_fsm_state)
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not bridge_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="bridge_id cannot be None",
                context={
                    "operation": "get_bridge_state",
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 2: Execute QUERY through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            query_filters={"bridge_id": bridge_id},
            limit=1,
        )

        result = await node.process(operation_input)

        # Step 3: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to query bridge state: {result.error_message}",
                context={
                    "operation": "get_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        # Step 4: Parse result
        items = result.result_data.get("items", [])
        if not items:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Bridge state not found: {bridge_id}",
                context={
                    "operation": "get_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 5: Deserialize to entity
        bridge_state = ModelBridgeState(**items[0])

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                operation_type="get_bridge_state",
                additional_context={
                    "bridge_id": str(bridge_id),
                    "namespace": bridge_state.namespace,
                },
            )

        return bridge_state

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error retrieving bridge state: {e!s}",
            context={
                "operation": "get_bridge_state",
                "bridge_id": str(bridge_id) if bridge_id else None,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )


async def list_bridge_states(
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> list[ModelBridgeState]:
    """
    Query bridge states with filters, pagination, and sorting.

    Uses QUERY operation with optional WHERE clause.

    Args:
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking
        filters: Optional query filters (e.g., {"namespace": "prod", "current_fsm_state": "IDLE"})
        limit: Maximum results to return (default: 100)
        offset: Number of results to skip (default: 0)
        sort_by: Field to sort by (default: "created_at")
        sort_order: Sort order "asc" or "desc" (default: "desc")

    Returns:
        List of ModelBridgeState entities

    Raises:
        OnexError: If validation fails or query fails

    Example:
        >>> # Query by namespace
        >>> states = await list_bridge_states(
        ...     filters={"namespace": "production"},
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
        >>>
        >>> # Query by FSM state with pagination
        >>> active_states = await list_bridge_states(
        ...     filters={"current_fsm_state": "PROCESSING"},
        ...     limit=50,
        ...     offset=0,
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Execute QUERY through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            query_filters=filters,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        result = await node.process(operation_input)

        # Step 2: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to query bridge states: {result.error_message}",
                context={
                    "operation": "list_bridge_states",
                    "filters": filters,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        # Step 3: Parse results
        items = result.result_data.get("items", [])
        bridge_states = [ModelBridgeState(**item) for item in items]

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=len(bridge_states),
                operation_type="list_bridge_states",
                additional_context={
                    "filters": filters,
                    "result_count": len(bridge_states),
                    "limit": limit,
                    "offset": offset,
                },
            )

        return bridge_states

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error listing bridge states: {e!s}",
            context={
                "operation": "list_bridge_states",
                "filters": filters,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )


async def delete_bridge_state(
    bridge_id: UUID,
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
) -> bool:
    """
    Delete bridge state record.

    Uses DELETE operation with WHERE clause on bridge_id.

    Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.

    Args:
        bridge_id: Bridge identifier to delete
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking

    Returns:
        True if deleted successfully

    Raises:
        OnexError: If validation fails, delete fails, or bridge not found

    Example:
        >>> success = await delete_bridge_state(
        ...     bridge_id=bridge_id,
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not bridge_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="bridge_id cannot be None",
                context={
                    "operation": "delete_bridge_state",
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 2: Execute DELETE through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            query_filters={"bridge_id": bridge_id},
        )

        result = await node.process(operation_input)

        # Step 3: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to delete bridge state: {result.error_message}",
                context={
                    "operation": "delete_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        if result.rows_affected == 0:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Bridge state not found: {bridge_id}",
                context={
                    "operation": "delete_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=result.rows_affected,
                operation_type="delete_bridge_state",
                additional_context={
                    "bridge_id": str(bridge_id),
                },
            )

        return True

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error deleting bridge state: {e!s}",
            context={
                "operation": "delete_bridge_state",
                "bridge_id": str(bridge_id) if bridge_id else None,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )


async def upsert_bridge_state(
    bridge_id: UUID,
    namespace: str,
    current_fsm_state: str,
    node: DatabaseAdapterProtocol,
    correlation_id: UUID,
    total_workflows_processed: int = 0,
    total_items_aggregated: int = 0,
    aggregation_metadata: Optional[dict[str, Any]] = None,
    last_aggregation_timestamp: Optional[datetime] = None,
) -> ModelBridgeState:
    """
    Insert or update bridge state (UPSERT).

    Uses PostgreSQL ON CONFLICT for atomic upsert operation.
    If bridge_id exists, updates all fields except bridge_id.

    Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.

    Args:
        bridge_id: Unique identifier for bridge instance
        namespace: Multi-tenant namespace
        current_fsm_state: FSM state
        node: DatabaseAdapterEffect node instance
        correlation_id: Correlation ID for operation tracking
        total_workflows_processed: Workflow count (default: 0)
        total_items_aggregated: Item count (default: 0)
        aggregation_metadata: Additional metadata (default: {})
        last_aggregation_timestamp: Timestamp (default: None)

    Returns:
        Upserted ModelBridgeState entity

    Raises:
        OnexError: If validation fails or upsert fails

    Example:
        >>> bridge = await upsert_bridge_state(
        ...     bridge_id=bridge_id,
        ...     namespace="prod",
        ...     current_fsm_state="PROCESSING",
        ...     total_workflows_processed=200,
        ...     node=db_node,
        ...     correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not bridge_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="bridge_id cannot be None",
                context={
                    "operation": "upsert_bridge_state",
                    "correlation_id": str(correlation_id),
                },
            )

        if not namespace or not namespace.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="namespace cannot be empty",
                context={
                    "operation": "upsert_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        if not current_fsm_state or not current_fsm_state.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="current_fsm_state cannot be empty",
                context={
                    "operation": "upsert_bridge_state",
                    "bridge_id": str(bridge_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Step 2: Build entity
        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace.strip(),
            current_fsm_state=current_fsm_state.strip(),
            total_workflows_processed=total_workflows_processed,
            total_items_aggregated=total_items_aggregated,
            aggregation_metadata=aggregation_metadata or {},
            last_aggregation_timestamp=last_aggregation_timestamp,
        )

        # Step 3: Execute UPSERT through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.BRIDGE_STATE,
            correlation_id=correlation_id,
            entity=bridge_state,
            query_filters={"bridge_id": bridge_id},  # Conflict key
        )

        result = await node.process(operation_input)

        # Step 4: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to upsert bridge state: {result.error_message}",
                context={
                    "operation": "upsert_bridge_state",
                    "bridge_id": str(bridge_id),
                    "namespace": namespace,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                operation_type="upsert_bridge_state",
                additional_context={
                    "bridge_id": str(bridge_id),
                    "namespace": namespace,
                    "fsm_state": current_fsm_state,
                },
            )

        return bridge_state

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error upserting bridge state: {e!s}",
            context={
                "operation": "upsert_bridge_state",
                "bridge_id": str(bridge_id) if bridge_id else None,
                "namespace": namespace,
                "correlation_id": str(correlation_id),
                "execution_time_ms": execution_time_ms,
            },
            original_error=e,
        )
