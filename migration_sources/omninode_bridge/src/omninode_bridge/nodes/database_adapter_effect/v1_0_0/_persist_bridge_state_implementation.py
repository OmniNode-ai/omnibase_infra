"""
Implementation of _persist_bridge_state method for NodeBridgeDatabaseAdapterEffect.

This is a standalone implementation that should be integrated into node.py.
Agent 3 implementation for Phase 2.
"""

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from .enums.enum_database_operation_type import EnumDatabaseOperationType
from .models.inputs.model_bridge_state_input import ModelBridgeStateInput
from .models.outputs.model_database_operation_output import ModelDatabaseOperationOutput


async def _persist_bridge_state(
    self, input_data: ModelBridgeStateInput
) -> ModelDatabaseOperationOutput:
    """
    Persist bridge aggregation state to database (UPSERT).

    Consumes events from NodeBridgeReducer:
    - STATE_AGGREGATION_COMPLETED: UPSERT aggregation state

    Uses PostgreSQL ON CONFLICT for UPSERT:
    - INSERT if bridge_id doesn't exist
    - UPDATE counters and metadata if exists

    Args:
        input_data: ModelBridgeStateInput with bridge state data

    Returns:
        ModelDatabaseOperationOutput with operation results

    Performance Target: < 10ms per operation

    Implementation: Phase 2, Agent 3 ✅
    """
    operation_start_time = time.perf_counter()
    correlation_id = uuid4()  # Generate correlation ID for this operation

    try:
        # Step 1: Validate input data and log operation start
        self._logger.log_operation_start(
            correlation_id=correlation_id,
            operation_type="persist_bridge_state",
            metadata={
                "bridge_id": str(input_data.bridge_id),
                "namespace": input_data.namespace,
                "total_workflows_processed": input_data.total_workflows_processed,
                "total_items_aggregated": input_data.total_items_aggregated,
                "current_fsm_state": input_data.current_fsm_state,
            },
        )

        # Validate namespace
        if not input_data.namespace or len(input_data.namespace.strip()) == 0:
            raise ValueError("Namespace must be a non-empty string")

        # Validate FSM state
        valid_fsm_states = ["idle", "active", "aggregating", "persisting"]
        if input_data.current_fsm_state not in valid_fsm_states:
            raise ValueError(
                f"Invalid FSM state '{input_data.current_fsm_state}'. "
                f"Must be one of: {', '.join(valid_fsm_states)}"
            )

        # Validate counters
        if input_data.total_workflows_processed < 0:
            raise ValueError(
                f"total_workflows_processed must be >= 0, got {input_data.total_workflows_processed}"
            )

        if input_data.total_items_aggregated < 0:
            raise ValueError(
                f"total_items_aggregated must be >= 0, got {input_data.total_items_aggregated}"
            )

        # Step 2: Construct UPSERT SQL query
        # UPSERT increments counters on conflict (cumulative aggregation)
        sql_query = """
            INSERT INTO bridge_states (
                bridge_id, namespace, total_workflows_processed,
                total_items_aggregated, aggregation_metadata,
                current_fsm_state, last_aggregation_timestamp,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, NOW(), NOW())
            ON CONFLICT (bridge_id) DO UPDATE SET
                namespace = EXCLUDED.namespace,
                total_workflows_processed = bridge_states.total_workflows_processed + EXCLUDED.total_workflows_processed,
                total_items_aggregated = bridge_states.total_items_aggregated + EXCLUDED.total_items_aggregated,
                aggregation_metadata = EXCLUDED.aggregation_metadata,
                current_fsm_state = EXCLUDED.current_fsm_state,
                last_aggregation_timestamp = EXCLUDED.last_aggregation_timestamp,
                updated_at = NOW()
            RETURNING bridge_id, namespace, total_workflows_processed, total_items_aggregated;
        """

        # Prepare parameters
        params = [
            input_data.bridge_id,
            input_data.namespace,
            input_data.total_workflows_processed,
            input_data.total_items_aggregated,
            json.dumps(input_data.aggregation_metadata),
            input_data.current_fsm_state,
            input_data.last_aggregation_timestamp or datetime.now(UTC),
        ]

        # Step 3: Execute query with circuit breaker protection
        async def _execute_upsert():
            """Execute UPSERT operation with transaction management."""
            # Use TransactionManager for atomic operation
            async with self._transaction_manager.begin():
                # Execute query using QueryExecutor
                result = await self._query_executor.execute(
                    sql_query, *params, timeout=5.0
                )
                return result

        # Wrap in circuit breaker
        result = await self._circuit_breaker.execute(_execute_upsert)

        # Step 4: Process result
        execution_time_ms = (time.perf_counter() - operation_start_time) * 1000
        rows_affected = len(result) if result else 1

        # Log UPSERT operation (INSERT or UPDATE)
        operation_type = "INSERT" if rows_affected == 1 else "UPDATE"
        self._logger.log_operation_complete(
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            operation_type="persist_bridge_state",
            additional_context={
                "operation_type": operation_type,
                "bridge_id": str(input_data.bridge_id),
                "namespace": input_data.namespace,
                "fsm_state_transition": input_data.current_fsm_state,
            },
        )

        # Step 5: Return success result
        return ModelDatabaseOperationOutput(
            success=True,
            operation_type=EnumDatabaseOperationType.PERSIST_BRIDGE_STATE.value,
            correlation_id=correlation_id,
            execution_time_ms=int(execution_time_ms),
            rows_affected=rows_affected,
            error_message=None,
        )

    except ValueError as e:
        # Validation error
        execution_time_ms = (time.perf_counter() - operation_start_time) * 1000

        self._logger.log_operation_error(
            correlation_id=correlation_id,
            error=f"Validation error: {e!s}",
            operation_type="persist_bridge_state",
            sanitized=True,
        )

        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=EnumDatabaseOperationType.PERSIST_BRIDGE_STATE.value,
            correlation_id=correlation_id,
            execution_time_ms=int(execution_time_ms),
            rows_affected=0,
            error_message=f"Validation error: {e!s}",
        )

    except Exception as e:
        # Database or other error
        execution_time_ms = (time.perf_counter() - operation_start_time) * 1000

        self._logger.log_operation_error(
            correlation_id=correlation_id,
            error=e,
            operation_type="persist_bridge_state",
            sanitized=True,
            additional_context={
                "bridge_id": str(input_data.bridge_id),
                "namespace": input_data.namespace,
            },
        )

        # In production, this would raise OnexError
        # For now, return error result
        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=EnumDatabaseOperationType.PERSIST_BRIDGE_STATE.value,
            correlation_id=correlation_id,
            execution_time_ms=int(execution_time_ms),
            rows_affected=0,
            error_message=f"Database error: {e!s}",
        )
