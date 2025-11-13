"""
Workflow Execution CRUD Operations.

High-level persistence functions for ModelWorkflowExecution with transaction support.
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
    >>> # Create workflow execution
    >>> workflow = await create_workflow_execution(
    ...     correlation_id=uuid4(),
    ...     workflow_type="metadata_stamping",
    ...     current_state="PENDING",
    ...     namespace="production",
    ...     node=database_adapter_node,
    ...     request_correlation_id=uuid4()
    ... )
    >>>
    >>> # Update workflow to PROCESSING
    >>> updated = await update_workflow_execution(
    ...     correlation_id=workflow.correlation_id,
    ...     updates={
    ...         "current_state": "PROCESSING",
    ...         "started_at": datetime.now(UTC)
    ...     },
    ...     node=database_adapter_node,
    ...     request_correlation_id=uuid4()
    ... )
    >>>
    >>> # Query workflows by namespace
    >>> workflows = await list_workflow_executions(
    ...     filters={"namespace": "production", "current_state": "PROCESSING"},
    ...     node=database_adapter_node,
    ...     request_correlation_id=uuid4()
    ... )
"""

import time
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)
from omninode_bridge.persistence.protocols import DatabaseAdapterProtocol

# Aliases for compatibility
OnexError = ModelOnexError


async def create_workflow_execution(
    correlation_id: UUID,
    workflow_type: str,
    current_state: str,
    namespace: str,
    node: DatabaseAdapterProtocol,
    request_correlation_id: UUID,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    execution_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ModelWorkflowExecution:
    """
    Create new workflow execution record.

    Uses INSERT operation with transaction support. Raises OnexError if
    correlation_id already exists (unique constraint).

    Args:
        correlation_id: Unique workflow correlation identifier
        workflow_type: Type of workflow (e.g., "metadata_stamping")
        current_state: Initial FSM state (e.g., "PENDING", "PROCESSING")
        namespace: Multi-tenant namespace
        node: DatabaseAdapterEffect node instance
        request_correlation_id: Correlation ID for this request
        started_at: Workflow start timestamp (default: None)
        completed_at: Workflow completion timestamp (default: None)
        execution_time_ms: Execution time in milliseconds (default: None)
        error_message: Error message if failed (default: None)
        metadata: Extended metadata (default: {})

    Returns:
        Created ModelWorkflowExecution entity

    Raises:
        OnexError: If validation fails, insert fails, or correlation_id exists

    Example:
        >>> workflow = await create_workflow_execution(
        ...     correlation_id=uuid4(),
        ...     workflow_type="metadata_stamping",
        ...     current_state="PENDING",
        ...     namespace="prod",
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not correlation_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="correlation_id cannot be None",
                context={
                    "operation": "create_workflow_execution",
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        if not workflow_type or not workflow_type.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="workflow_type cannot be empty",
                context={
                    "operation": "create_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        if not current_state or not current_state.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="current_state cannot be empty",
                context={
                    "operation": "create_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        if not namespace or not namespace.strip():
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="namespace cannot be empty",
                context={
                    "operation": "create_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        # Step 2: Build entity (default started_at to current UTC time if None)
        workflow_execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type.strip(),
            current_state=current_state.strip(),
            namespace=namespace.strip(),
            started_at=started_at or datetime.now(UTC),
            completed_at=completed_at,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Step 3: Execute INSERT through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=request_correlation_id,
            entity=workflow_execution,
        )

        result = await node.process(operation_input)

        # Step 4: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to create workflow execution: {result.error_message}",
                context={
                    "operation": "create_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "workflow_type": workflow_type,
                    "namespace": namespace,
                    "request_correlation_id": str(request_correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=request_correlation_id,
                execution_time_ms=execution_time_ms_total,
                rows_affected=1,
                operation_type="create_workflow_execution",
                additional_context={
                    "correlation_id": str(correlation_id),
                    "workflow_type": workflow_type,
                    "namespace": namespace,
                    "current_state": current_state,
                },
            )

        return workflow_execution

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error creating workflow execution: {e!s}",
            context={
                "operation": "create_workflow_execution",
                "correlation_id": str(correlation_id) if correlation_id else None,
                "workflow_type": workflow_type,
                "namespace": namespace,
                "request_correlation_id": str(request_correlation_id),
                "execution_time_ms": execution_time_ms_total,
            },
            original_error=e,
        )


async def update_workflow_execution(
    correlation_id: UUID,
    updates: dict[str, Any],
    node: DatabaseAdapterProtocol,
    request_correlation_id: UUID,
) -> ModelWorkflowExecution:
    """
    Update existing workflow execution record.

    Uses UPDATE operation with WHERE clause on correlation_id. Only updates
    provided fields, leaving others unchanged.

    Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.

    Args:
        correlation_id: Workflow correlation identifier to update
        updates: Dictionary of fields to update (e.g., {"current_state": "COMPLETED"})
        node: DatabaseAdapterEffect node instance
        request_correlation_id: Correlation ID for this request

    Returns:
        Updated ModelWorkflowExecution entity

    Raises:
        OnexError: If validation fails, update fails, or workflow not found

    Example:
        >>> updated = await update_workflow_execution(
        ...     correlation_id=workflow_id,
        ...     updates={
        ...         "current_state": "COMPLETED",
        ...         "completed_at": datetime.now(UTC),
        ...         "execution_time_ms": 1234
        ...     },
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not correlation_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="correlation_id cannot be None",
                context={
                    "operation": "update_workflow_execution",
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        if not updates:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="updates dictionary cannot be empty",
                context={
                    "operation": "update_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        # Step 2: Fetch existing workflow
        existing_workflow = await get_workflow_execution(
            correlation_id=correlation_id,
            node=node,
            request_correlation_id=request_correlation_id,
        )

        # Step 3: Apply updates to existing workflow
        updated_data = existing_workflow.model_dump(
            exclude={"id", "created_at", "updated_at"}
        )
        updated_data.update(updates)

        # Step 4: Validate updated entity
        updated_workflow = ModelWorkflowExecution(**updated_data)

        # Step 5: Execute UPDATE through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=request_correlation_id,
            entity=updated_workflow,
            query_filters={"correlation_id": correlation_id},
        )

        result = await node.process(operation_input)

        # Step 6: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to update workflow execution: {result.error_message}",
                context={
                    "operation": "update_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "updates": updates,
                    "request_correlation_id": str(request_correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        if result.rows_affected == 0:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Workflow execution not found: {correlation_id}",
                context={
                    "operation": "update_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=request_correlation_id,
                execution_time_ms=execution_time_ms_total,
                rows_affected=result.rows_affected,
                operation_type="update_workflow_execution",
                additional_context={
                    "correlation_id": str(correlation_id),
                    "fields_updated": list(updates.keys()),
                },
            )

        return updated_workflow

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error updating workflow execution: {e!s}",
            context={
                "operation": "update_workflow_execution",
                "correlation_id": str(correlation_id) if correlation_id else None,
                "updates": updates,
                "request_correlation_id": str(request_correlation_id),
                "execution_time_ms": execution_time_ms_total,
            },
            original_error=e,
        )


async def get_workflow_execution(
    correlation_id: UUID,
    node: DatabaseAdapterProtocol,
    request_correlation_id: UUID,
) -> ModelWorkflowExecution:
    """
    Retrieve workflow execution by correlation_id.

    Uses QUERY operation with WHERE clause on correlation_id.

    Args:
        correlation_id: Workflow correlation identifier to retrieve
        node: DatabaseAdapterEffect node instance
        request_correlation_id: Correlation ID for this request

    Returns:
        ModelWorkflowExecution entity

    Raises:
        OnexError: If validation fails, query fails, or workflow not found

    Example:
        >>> workflow = await get_workflow_execution(
        ...     correlation_id=workflow_id,
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
        >>> print(workflow.current_state, workflow.workflow_type)
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not correlation_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="correlation_id cannot be None",
                context={
                    "operation": "get_workflow_execution",
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        # Step 2: Execute QUERY through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=request_correlation_id,
            query_filters={"correlation_id": correlation_id},
            limit=1,
        )

        result = await node.process(operation_input)

        # Step 3: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to query workflow execution: {result.error_message}",
                context={
                    "operation": "get_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        # Step 4: Parse result
        items = result.result_data.get("items", [])
        if not items:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Workflow execution not found: {correlation_id}",
                context={
                    "operation": "get_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        # Step 5: Deserialize to entity
        workflow_execution = ModelWorkflowExecution(**items[0])

        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=request_correlation_id,
                execution_time_ms=execution_time_ms_total,
                rows_affected=1,
                operation_type="get_workflow_execution",
                additional_context={
                    "correlation_id": str(correlation_id),
                    "workflow_type": workflow_execution.workflow_type,
                    "namespace": workflow_execution.namespace,
                },
            )

        return workflow_execution

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error retrieving workflow execution: {e!s}",
            context={
                "operation": "get_workflow_execution",
                "correlation_id": str(correlation_id) if correlation_id else None,
                "request_correlation_id": str(request_correlation_id),
                "execution_time_ms": execution_time_ms_total,
            },
            original_error=e,
        )


async def list_workflow_executions(
    node: DatabaseAdapterProtocol,
    request_correlation_id: UUID,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> list[ModelWorkflowExecution]:
    """
    Query workflow executions with filters, pagination, and sorting.

    Uses QUERY operation with optional WHERE clause.

    Args:
        node: DatabaseAdapterEffect node instance
        request_correlation_id: Correlation ID for this request
        filters: Optional query filters (e.g., {"namespace": "prod", "current_state": "PROCESSING"})
        limit: Maximum results to return (default: 100)
        offset: Number of results to skip (default: 0)
        sort_by: Field to sort by (default: "created_at")
        sort_order: Sort order "asc" or "desc" (default: "desc")

    Returns:
        List of ModelWorkflowExecution entities

    Raises:
        OnexError: If validation fails or query fails

    Example:
        >>> # Query by namespace and state
        >>> workflows = await list_workflow_executions(
        ...     filters={"namespace": "production", "current_state": "PROCESSING"},
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
        >>>
        >>> # Query with pagination
        >>> page_1 = await list_workflow_executions(
        ...     limit=50,
        ...     offset=0,
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Execute QUERY through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=request_correlation_id,
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
                message=f"Failed to query workflow executions: {result.error_message}",
                context={
                    "operation": "list_workflow_executions",
                    "filters": filters,
                    "request_correlation_id": str(request_correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        # Step 3: Parse results
        items = result.result_data.get("items", [])
        workflow_executions = [ModelWorkflowExecution(**item) for item in items]

        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=request_correlation_id,
                execution_time_ms=execution_time_ms_total,
                rows_affected=len(workflow_executions),
                operation_type="list_workflow_executions",
                additional_context={
                    "filters": filters,
                    "result_count": len(workflow_executions),
                    "limit": limit,
                    "offset": offset,
                },
            )

        return workflow_executions

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error listing workflow executions: {e!s}",
            context={
                "operation": "list_workflow_executions",
                "filters": filters,
                "request_correlation_id": str(request_correlation_id),
                "execution_time_ms": execution_time_ms_total,
            },
            original_error=e,
        )


async def delete_workflow_execution(
    correlation_id: UUID,
    node: DatabaseAdapterProtocol,
    request_correlation_id: UUID,
) -> bool:
    """
    Delete workflow execution record.

    Uses DELETE operation with WHERE clause on correlation_id.

    Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.

    Args:
        correlation_id: Workflow correlation identifier to delete
        node: DatabaseAdapterEffect node instance
        request_correlation_id: Correlation ID for this request

    Returns:
        True if deleted successfully

    Raises:
        OnexError: If validation fails, delete fails, or workflow not found

    Example:
        >>> success = await delete_workflow_execution(
        ...     correlation_id=workflow_id,
        ...     node=db_node,
        ...     request_correlation_id=uuid4()
        ... )
    """
    start_time = time.perf_counter()

    try:
        # Step 1: Validate inputs
        if not correlation_id:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="correlation_id cannot be None",
                context={
                    "operation": "delete_workflow_execution",
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        # Step 2: Execute DELETE through generic CRUD handler
        operation_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=request_correlation_id,
            query_filters={"correlation_id": correlation_id},
        )

        result = await node.process(operation_input)

        # Step 3: Check operation result
        if not result.success:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_ERROR,
                message=f"Failed to delete workflow execution: {result.error_message}",
                context={
                    "operation": "delete_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                    "execution_time_ms": result.execution_time_ms,
                },
            )

        if result.rows_affected == 0:
            raise OnexError(
                error_code=EnumCoreErrorCode.NOT_FOUND,
                message=f"Workflow execution not found: {correlation_id}",
                context={
                    "operation": "delete_workflow_execution",
                    "correlation_id": str(correlation_id),
                    "request_correlation_id": str(request_correlation_id),
                },
            )

        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)

        # Log success
        if hasattr(node, "_logger") and node._logger:
            node._logger.log_operation_complete(
                correlation_id=request_correlation_id,
                execution_time_ms=execution_time_ms_total,
                rows_affected=result.rows_affected,
                operation_type="delete_workflow_execution",
                additional_context={
                    "correlation_id": str(correlation_id),
                },
            )

        return True

    except OnexError:
        raise
    except Exception as e:
        execution_time_ms_total = int((time.perf_counter() - start_time) * 1000)
        raise OnexError(
            error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            message=f"Unexpected error deleting workflow execution: {e!s}",
            context={
                "operation": "delete_workflow_execution",
                "correlation_id": str(correlation_id) if correlation_id else None,
                "request_correlation_id": str(request_correlation_id),
                "execution_time_ms": execution_time_ms_total,
            },
            original_error=e,
        )
