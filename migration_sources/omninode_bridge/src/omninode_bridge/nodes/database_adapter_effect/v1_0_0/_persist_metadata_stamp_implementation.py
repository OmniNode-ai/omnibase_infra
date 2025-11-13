"""
Reference Implementation: _persist_metadata_stamp method (Agent 5).

This file contains the complete implementation that can be integrated
into node.py when the Phase 2 metadata stamp persistence feature is activated.

Author: Agent 5
Date: October 8, 2025
"""

# Complete implementation for _persist_metadata_stamp method


async def _persist_metadata_stamp(
    self, input_data: ModelDatabaseOperationInput
) -> ModelDatabaseOperationOutput:
    """
    Persist metadata stamp audit record (INSERT).

    Consumes events from NodeBridgeOrchestrator:
    - STAMP_CREATED: INSERT stamp audit trail

    Operation: INSERT metadata_stamps table (append-only audit log)
    Performance Target: < 10ms per operation
    Handles: stamp_id, file_hash, stamped_content, timestamps

    Args:
        input_data: Full ModelDatabaseOperationInput with:
            - correlation_id: UUID for tracing
            - metadata_stamp_data: dict with:
                - file_hash: str (64-128 hex characters)
                - namespace: str
                - stamp_data: dict (containing stamped_content and operation)
                - workflow_id: UUID | None

    Returns:
        ModelDatabaseOperationOutput with operation results

    Implementation: Phase 2, Agent 5
    """
    import json
    import re

    from .circuit_breaker import CircuitBreakerOpenError
    from .structured_logger import DatabaseOperationType

    start_time = time.perf_counter()

    # Extract correlation_id from parent ModelDatabaseOperationInput
    correlation_id = input_data.correlation_id
    operation_type = EnumDatabaseOperationType.PERSIST_METADATA_STAMP

    # Extract metadata_stamp_data from input
    metadata_stamp_data = input_data.metadata_stamp_data

    # Validate metadata_stamp_data exists
    if not metadata_stamp_data:
        error_msg = (
            "metadata_stamp_data is required for persist_metadata_stamp operation"
        )
        self._logger.log_operation_error(
            correlation_id=correlation_id,
            error=error_msg,
            operation_type=DatabaseOperationType.QUERY,
        )
        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=operation_type.value,
            correlation_id=correlation_id,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            rows_affected=0,
            error_message=error_msg,
        )

    try:
        # Step 1: Extract and validate input fields from metadata_stamp_data
        file_hash = metadata_stamp_data.get("file_hash")
        namespace = metadata_stamp_data.get("namespace")
        stamp_data = metadata_stamp_data.get("stamp_data", {})
        workflow_id = metadata_stamp_data.get("workflow_id")

        # Extract stamped_content and operation from stamp_data
        stamped_content = stamp_data.get("stamped_content", "")
        operation = stamp_data.get("operation", "CREATED")

        # Step 2: Input validation
        if not file_hash:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message="file_hash is required for metadata stamp persistence",
            )

        # Validate file_hash format (BLAKE3: exactly 64 hex characters)
        if not re.match(r"^[a-f0-9]{64}$", file_hash):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message=f"Invalid file_hash format: {file_hash[:20]}... (expected exactly 64 hex characters)",
            )

        # Validate namespace
        if not namespace or not isinstance(namespace, str):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message="namespace must be a non-empty string",
            )

        # Validate stamped_content
        if not stamped_content or not isinstance(stamped_content, str):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message="stamped_content must be a non-empty string",
            )

        # Validate operation
        valid_operations = ["CREATED", "VALIDATED", "UPDATED"]
        if operation not in valid_operations:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_FAILED,
                message=f"operation must be one of {valid_operations}, got: {operation}",
            )

        # Step 3: Log operation start
        self._logger.log_operation_start(
            correlation_id=correlation_id,
            operation_type=DatabaseOperationType.QUERY,
            metadata={
                "table": "metadata_stamps",
                "file_hash": file_hash,
                "namespace": namespace,
                "operation": operation,
                "has_workflow": workflow_id is not None,
            },
        )

        # Step 4: Build SQL INSERT query
        # Note: stamp_metadata is stored as JSONB
        stamp_metadata_json = json.dumps(stamp_data)

        sql = """
            INSERT INTO metadata_stamps (
                file_hash, correlation_id, namespace,
                stamped_content, stamp_metadata, operation
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            RETURNING stamp_id, created_at;
        """

        params = [
            file_hash,
            correlation_id,
            namespace,
            stamped_content,
            stamp_metadata_json,
            operation,
        ]

        # Step 5: Execute with circuit breaker protection
        try:
            result = await self._circuit_breaker.execute(
                self._query_executor.execute_query,
                sql,
                *params,
            )

            # Extract returned values
            stamp_id = result[0]["stamp_id"] if result else None
            created_at = result[0]["created_at"] if result else None

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Step 6: Log operation success
            self._logger.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                operation_type=DatabaseOperationType.QUERY,
                additional_context={
                    "stamp_id": str(stamp_id),
                    "file_hash": file_hash,
                    "namespace": namespace,
                    "operation": operation,
                },
            )

            # Step 7: Return success output
            return ModelDatabaseOperationOutput(
                success=True,
                operation_type=operation_type.value,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                error_message=None,
            )

        except CircuitBreakerOpenError as e:
            # Circuit breaker is open - database temporarily unavailable
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Circuit breaker OPEN: {e!s}"

            self._logger.log_operation_error(
                correlation_id=correlation_id,
                error=error_msg,
                operation_type=DatabaseOperationType.QUERY,
                additional_context={
                    "file_hash": file_hash,
                    "namespace": namespace,
                },
            )

            return ModelDatabaseOperationOutput(
                success=False,
                operation_type=operation_type.value,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=0,
                error_message=error_msg,
            )

    except OnexError as e:
        # OnexError already has context - propagate with logging
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        self._logger.log_operation_error(
            correlation_id=correlation_id,
            error=e.message,
            operation_type=DatabaseOperationType.QUERY,
        )

        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=operation_type.value,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=0,
            error_message=e.message,
        )

    except Exception as e:
        # Wrap unexpected exceptions in OnexError
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        error_msg = f"Database operation failed: {e!s}"

        self._logger.log_operation_error(
            correlation_id=correlation_id,
            error=error_msg,
            operation_type=DatabaseOperationType.QUERY,
        )

        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=operation_type.value,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=0,
            error_message=error_msg,
        )
