#!/usr/bin/env python3
"""
Database Operation Type Enumeration for Generic CRUD Pattern.

Defines generic database operation types that support full CRUD operations
with batch processing and utility functions. Part of the O.N.E. v0.1
compliant generic database adapter pattern.

ONEX v2.0 Compliance:
- Enum-based naming: EnumDatabaseOperationType
- Generic CRUD pattern for maximum flexibility
- String-based enum for JSON serialization
- Integration with ModelDatabaseOperationInput/Output

Migration Note:
This replaces the previous specific operation types (persist_*, update_*, query_*)
with a generic CRUD + entity type pattern for improved flexibility and extensibility.

See Also:
- /docs/GENERIC_DATABASE_ADAPTER_PATTERN.md for architecture overview
- EnumEntityType for database entity types
- ModelDatabaseOperationInput for request structure
- ModelDatabaseOperationOutput for response structure
"""

from enum import Enum


class EnumDatabaseOperationType(str, Enum):
    """
    Generic database operation types for CRUD + batch operations.

    This enum provides standard database operation types that can be combined
    with EnumEntityType to create flexible, extensible database operations
    without requiring new operation types for each entity.

    Operation Categories:
        Core CRUD: INSERT, UPDATE, DELETE, QUERY, UPSERT
        Batch Operations: BATCH_INSERT, BATCH_UPDATE, BATCH_DELETE
        Utility Operations: COUNT, EXISTS, HEALTH_CHECK

    Usage Examples:
        # Insert a new workflow execution
        operation_type = EnumDatabaseOperationType.INSERT
        entity_type = EnumEntityType.WORKFLOW_EXECUTION

        # Query metadata stamps with filtering
        operation_type = EnumDatabaseOperationType.QUERY
        entity_type = EnumEntityType.METADATA_STAMP

        # Batch insert workflow steps
        operation_type = EnumDatabaseOperationType.BATCH_INSERT
        entity_type = EnumEntityType.WORKFLOW_STEP

        # Update node heartbeat (insert or update)
        operation_type = EnumDatabaseOperationType.UPSERT
        entity_type = EnumEntityType.NODE_HEARTBEAT

    See Also:
        /docs/GENERIC_DATABASE_ADAPTER_PATTERN.md for complete usage examples
        and migration guidance from the previous specific operation pattern.
    """

    # Core CRUD Operations
    INSERT = "insert"
    """
    Insert a new record into the database.

    Requires:
        - entity_type: Target entity to insert
        - entity_data: Dictionary of field values to insert

    Returns:
        - result_data: Inserted record with generated ID
        - rows_affected: 1 on success

    Example:
        {
            "operation_type": "insert",
            "entity_type": "workflow_execution",
            "entity_data": {
                "workflow_id": "wf-123",
                "status": "processing",
                "namespace": "test_app"
            }
        }

    Error Cases:
        - Duplicate key violation (if record already exists)
        - Constraint violations (invalid foreign keys, etc.)
        - Required field missing in entity_data
    """

    UPDATE = "update"
    """
    Update existing record(s) matching query filters.

    Requires:
        - entity_type: Target entity to update
        - query_filters: Dictionary of filter conditions
        - entity_data: Dictionary of fields to update

    Returns:
        - result_data: Updated record(s)
        - rows_affected: Number of rows updated

    Example:
        {
            "operation_type": "update",
            "entity_type": "workflow_execution",
            "query_filters": {"workflow_id": "wf-123"},
            "entity_data": {
                "status": "completed",
                "completed_at": "2025-10-08T12:05:00Z"
            }
        }

    Error Cases:
        - No records match query_filters (rows_affected = 0)
        - Invalid field names in entity_data
        - Constraint violations on updated values
    """

    DELETE = "delete"
    """
    Delete record(s) matching query filters.

    Requires:
        - entity_type: Target entity to delete from
        - query_filters: Dictionary of filter conditions

    Returns:
        - rows_affected: Number of rows deleted

    Example:
        {
            "operation_type": "delete",
            "entity_type": "fsm_transition",
            "query_filters": {
                "created_before": "2024-01-01"
            }
        }

    Error Cases:
        - No records match query_filters (rows_affected = 0)
        - Foreign key constraint violations (if other records reference deleted data)

    Warning:
        This is a hard delete. Consider soft deletes (UPDATE with is_deleted flag)
        for audit trail preservation.
    """

    QUERY = "query"
    """
    Query record(s) with filtering, sorting, and pagination.

    Requires:
        - entity_type: Target entity to query

    Optional:
        - query_filters: Dictionary of filter conditions
        - sort_by: Field name to sort by
        - sort_order: "asc" or "desc" (default: "desc")
        - limit: Maximum number of records to return
        - offset: Number of records to skip (for pagination)

    Returns:
        - result_data: List of matching records
        - rows_affected: Number of records returned

    Example:
        {
            "operation_type": "query",
            "entity_type": "metadata_stamp",
            "query_filters": {
                "namespace": "test_app",
                "created_after": "2025-10-01"
            },
            "sort_by": "created_at",
            "sort_order": "desc",
            "limit": 100
        }

    Error Cases:
        - Invalid field names in query_filters
        - Invalid sort_by field name
        - Negative limit or offset values
    """

    UPSERT = "upsert"
    """
    Insert record if it doesn't exist, update if it does (INSERT ON CONFLICT UPDATE).

    Requires:
        - entity_type: Target entity
        - query_filters: Dictionary identifying unique record (for conflict detection)
        - entity_data: Dictionary of field values to insert/update

    Returns:
        - result_data: Inserted or updated record
        - rows_affected: 1 on success

    Example:
        {
            "operation_type": "upsert",
            "entity_type": "node_heartbeat",
            "query_filters": {"node_id": "orchestrator-node-1"},
            "entity_data": {
                "node_id": "orchestrator-node-1",
                "last_heartbeat": "2025-10-08T12:00:00Z",
                "status": "healthy"
            }
        }

    Implementation:
        Uses PostgreSQL INSERT ... ON CONFLICT ... DO UPDATE for atomicity.

    Error Cases:
        - Invalid unique constraint fields in query_filters
        - Constraint violations on insert/update
    """

    # Batch Operations
    BATCH_INSERT = "batch_insert"
    """
    Insert multiple records in a single transaction.

    Requires:
        - entity_type: Target entity to insert into
        - batch_data: List of dictionaries, each containing field values

    Returns:
        - result_data: List of inserted records with generated IDs
        - rows_affected: Number of rows inserted

    Example:
        {
            "operation_type": "batch_insert",
            "entity_type": "workflow_step",
            "batch_data": [
                {"step_name": "hash_generation", "status": "completed"},
                {"step_name": "stamp_creation", "status": "completed"},
                {"step_name": "event_publishing", "status": "in_progress"}
            ]
        }

    Performance:
        Uses bulk insert with COPY or INSERT VALUES for efficiency.
        Significantly faster than individual INSERT operations.

    Error Cases:
        - Empty batch_data list
        - Duplicate key violations (entire batch may fail)
        - Constraint violations in any record (entire batch may fail)

    Transaction Behavior:
        All inserts succeed or all fail (atomic operation).
    """

    BATCH_UPDATE = "batch_update"
    """
    Update multiple records in a single transaction.

    Requires:
        - entity_type: Target entity to update
        - batch_data: List of dictionaries, each containing:
            - filters: Query filters to identify record
            - updates: Fields to update

    Returns:
        - result_data: List of updated records
        - rows_affected: Total number of rows updated across all batch items

    Example:
        {
            "operation_type": "batch_update",
            "entity_type": "workflow_step",
            "batch_data": [
                {
                    "filters": {"step_id": "step-1"},
                    "updates": {"status": "completed"}
                },
                {
                    "filters": {"step_id": "step-2"},
                    "updates": {"status": "failed"}
                }
            ]
        }

    Transaction Behavior:
        All updates succeed or all fail (atomic operation).

    Error Cases:
        - Empty batch_data list
        - No records match any filter (partial success possible)
        - Constraint violations in any update
    """

    BATCH_DELETE = "batch_delete"
    """
    Delete multiple records in a single transaction.

    Requires:
        - entity_type: Target entity to delete from
        - batch_data: List of dictionaries, each containing query filters

    Returns:
        - rows_affected: Total number of rows deleted

    Example:
        {
            "operation_type": "batch_delete",
            "entity_type": "fsm_transition",
            "batch_data": [
                {"workflow_id": "wf-1"},
                {"workflow_id": "wf-2"},
                {"workflow_id": "wf-3"}
            ]
        }

    Transaction Behavior:
        All deletes succeed or all fail (atomic operation).

    Error Cases:
        - Empty batch_data list
        - Foreign key constraint violations
        - No records match any filter (may not be error depending on use case)

    Warning:
        This is a hard delete. Consider batch UPDATE with is_deleted flag
        for audit trail preservation.
    """

    # Utility Operations
    COUNT = "count"
    """
    Count records matching query filters.

    Requires:
        - entity_type: Target entity to count

    Optional:
        - query_filters: Dictionary of filter conditions

    Returns:
        - result_data: {"count": <number>}
        - rows_affected: 0 (no data modification)

    Example:
        {
            "operation_type": "count",
            "entity_type": "workflow_execution",
            "query_filters": {
                "status": "processing",
                "namespace": "test_app"
            }
        }

    Performance:
        Uses SELECT COUNT(*) for efficiency. For large tables, consider
        approximate counts or caching if exact counts aren't required.

    Error Cases:
        - Invalid field names in query_filters
    """

    EXISTS = "exists"
    """
    Check if any records match query filters.

    Requires:
        - entity_type: Target entity to check
        - query_filters: Dictionary of filter conditions

    Returns:
        - result_data: {"exists": true/false}
        - rows_affected: 0 (no data modification)

    Example:
        {
            "operation_type": "exists",
            "entity_type": "metadata_stamp",
            "query_filters": {
                "file_hash": "blake3:abc123...",
                "namespace": "test_app"
            }
        }

    Performance:
        Uses SELECT EXISTS for efficiency. Stops at first matching record.

    Error Cases:
        - Invalid field names in query_filters
    """

    HEALTH_CHECK = "health_check"
    """
    Verify database connectivity and responsiveness.

    Requires:
        - entity_type: Not used (can be any value)

    Returns:
        - result_data: {
            "status": "healthy",
            "response_time_ms": <latency>,
            "connection_pool_size": <active_connections>
          }
        - rows_affected: 0 (no data modification)

    Example:
        {
            "operation_type": "health_check",
            "entity_type": "workflow_execution"
        }

    Implementation:
        Executes simple query (SELECT 1) and measures response time.
        Checks connection pool health and available connections.

    Error Cases:
        - Database unreachable (connection timeout)
        - All connections exhausted (connection pool full)
        - Slow query response (> threshold)

    Usage:
        Used by monitoring systems and readiness probes.
    """

    def is_read_operation(self) -> bool:
        """
        Check if this operation is read-only (no data modification).

        Returns:
            True for QUERY, COUNT, EXISTS, HEALTH_CHECK operations.
            False for INSERT, UPDATE, DELETE, UPSERT, and batch operations.

        Usage:
            Useful for determining transaction isolation levels, caching
            strategies, and replica routing (read operations can use replicas).
        """
        return self in (
            EnumDatabaseOperationType.QUERY,
            EnumDatabaseOperationType.COUNT,
            EnumDatabaseOperationType.EXISTS,
            EnumDatabaseOperationType.HEALTH_CHECK,
        )

    def is_batch_operation(self) -> bool:
        """
        Check if this operation processes multiple records.

        Returns:
            True for BATCH_INSERT, BATCH_UPDATE, BATCH_DELETE operations.
            False for single-record operations.

        Usage:
            Useful for performance optimization (batching, connection pooling),
            transaction size estimation, and monitoring batch operation metrics.
        """
        return self in (
            EnumDatabaseOperationType.BATCH_INSERT,
            EnumDatabaseOperationType.BATCH_UPDATE,
            EnumDatabaseOperationType.BATCH_DELETE,
        )

    def requires_entity_data(self) -> bool:
        """
        Check if this operation requires entity_data field.

        Returns:
            True for INSERT, UPDATE, UPSERT operations.
            False for DELETE, QUERY, COUNT, EXISTS, HEALTH_CHECK.
            Batch operations use batch_data instead.

        Usage:
            Used for input validation to ensure required fields are present
            before processing the operation.
        """
        return self in (
            EnumDatabaseOperationType.INSERT,
            EnumDatabaseOperationType.UPDATE,
            EnumDatabaseOperationType.UPSERT,
        )

    def requires_query_filters(self) -> bool:
        """
        Check if this operation requires query_filters field.

        Returns:
            True for UPDATE, DELETE, UPSERT, EXISTS operations.
            False for INSERT, COUNT, HEALTH_CHECK.
            QUERY operation has optional filters.

        Usage:
            Used for input validation to ensure required fields are present
            before processing the operation.
        """
        return self in (
            EnumDatabaseOperationType.UPDATE,
            EnumDatabaseOperationType.DELETE,
            EnumDatabaseOperationType.UPSERT,
            EnumDatabaseOperationType.EXISTS,
        )
