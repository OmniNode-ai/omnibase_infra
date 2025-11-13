"""
Workflow Step Persistence Implementation for Database Adapter Effect Node.

This file contains the complete implementation of the _persist_workflow_step() method
along with helper methods for creating success/error outputs.

Implementation: Phase 2, Agent 2
Created: October 2025
"""

import json
import time
from uuid import UUID, uuid4

from .models.inputs.model_workflow_step_input import ModelWorkflowStepInput
from .models.outputs.model_database_operation_output import ModelDatabaseOperationOutput


class WorkflowStepPersistenceMixin:
    """
    Mixin providing workflow step persistence functionality.

    This mixin should be integrated into NodeBridgeDatabaseAdapterEffect class.
    It provides the _persist_workflow_step() method and helper methods for
    creating ModelDatabaseOperationOutput instances.
    """

    async def _persist_workflow_step(
        self, input_data: ModelWorkflowStepInput
    ) -> ModelDatabaseOperationOutput:
        """
        Persist workflow step history (INSERT).

        Consumes events from NodeBridgeOrchestrator:
        - STEP_COMPLETED: INSERT step history with execution time

        Operation: INSERT workflow_steps table (append-only history)
        Performance Target: < 10ms per operation
        Handles: step_id, workflow_id, step_name, step_status, timestamps

        Args:
            input_data: ModelWorkflowStepInput with workflow step data

        Returns:
            ModelDatabaseOperationOutput with operation results

        Implementation: Phase 2, Agent 2
        """
        start_time = time.perf_counter()
        correlation_id = uuid4()  # Generate for this operation

        try:
            # Step 1: Input Validation
            self._logger.log_operation_start(
                correlation_id=correlation_id,
                operation_type="persist_workflow_step",
                metadata={
                    "workflow_id": str(input_data.workflow_id),
                    "step_name": input_data.step_name,
                    "step_order": input_data.step_order,
                    "status": input_data.status,
                },
            )

            # Validate step_status is valid
            valid_statuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]
            if input_data.status not in valid_statuses:
                error_msg = f"Invalid step status: {input_data.status}. Must be one of {valid_statuses}"
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error_msg,
                    operation_type="persist_workflow_step",
                )
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                return self._create_error_output(
                    correlation_id=correlation_id,
                    operation_type="persist_workflow_step",
                    error_msg=error_msg,
                    execution_time_ms=execution_time_ms,
                )

            # Validate step_order >= 1
            if input_data.step_order < 1:
                error_msg = f"Invalid step_order: {input_data.step_order}. Must be >= 1"
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error_msg,
                    operation_type="persist_workflow_step",
                )
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                return self._create_error_output(
                    correlation_id=correlation_id,
                    operation_type="persist_workflow_step",
                    error_msg=error_msg,
                    execution_time_ms=execution_time_ms,
                )

            # Generate step_id
            step_id = uuid4()

            # Calculate execution_time_ms if completed_at is available
            execution_time_ms_value = input_data.execution_time_ms

            # Prepare input_data and output_data as JSONB
            input_data_jsonb = json.dumps(
                input_data.step_data if input_data.step_data else {}
            )
            output_data_jsonb = json.dumps({})  # Empty for now, can be populated later

            # Step 2: Construct SQL Query
            sql_query = """
            INSERT INTO workflow_steps (
                step_id, workflow_id, correlation_id, step_name, step_order,
                step_status, input_data, output_data, error_message,
                started_at, completed_at, execution_time_ms, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING step_id;
            """

            # Prepare query parameters
            query_params = [
                step_id,  # $1 - step_id
                input_data.workflow_id,  # $2 - workflow_id
                correlation_id,  # $3 - correlation_id (generated for this operation)
                input_data.step_name,  # $4 - step_name
                input_data.step_order,  # $5 - step_order
                input_data.status,  # $6 - step_status
                input_data_jsonb,  # $7 - input_data (JSONB)
                output_data_jsonb,  # $8 - output_data (JSONB)
                input_data.error_message,  # $9 - error_message
                input_data.created_at,  # $10 - started_at (using created_at)
                (
                    input_data.created_at if input_data.status == "COMPLETED" else None
                ),  # $11 - completed_at
                execution_time_ms_value,  # $12 - execution_time_ms
                input_data.created_at,  # $13 - created_at
            ]

            # Validate query with SecurityValidator
            query_validation = self._security_validator.validate_query(sql_query)
            if not query_validation.valid:
                error_msg = (
                    f"Query validation failed: {'; '.join(query_validation.errors)}"
                )
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error_msg,
                    operation_type="persist_workflow_step",
                )
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                return self._create_error_output(
                    correlation_id=correlation_id,
                    operation_type="persist_workflow_step",
                    error_msg=error_msg,
                    execution_time_ms=execution_time_ms,
                )

            # Validate parameters
            params_validation = self._security_validator.validate_parameters(
                query_params
            )
            if not params_validation.valid:
                error_msg = f"Parameter validation failed: {'; '.join(params_validation.errors)}"
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error_msg,
                    operation_type="persist_workflow_step",
                )
                execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                return self._create_error_output(
                    correlation_id=correlation_id,
                    operation_type="persist_workflow_step",
                    error_msg=error_msg,
                    execution_time_ms=execution_time_ms,
                )

            # Log any warnings from validation
            for warning in query_validation.warnings + params_validation.warnings:
                self._logger.logger.warning(
                    "Validation warning",
                    correlation_id=str(correlation_id),
                    warning=warning,
                )

            # Step 3: Execute with Circuit Breaker and Transaction
            async def _execute_insert():
                """Inner function for circuit breaker execution."""
                async with self._transaction_manager.begin():
                    result = await self._query_executor.execute_query(
                        sql_query, query_params
                    )
                    return result

            self._logger.log_query_start(
                correlation_id=correlation_id,
                query=self._logger.sanitize_query(sql_query),
                params_count=len(query_params),
                table="workflow_steps",
            )

            # Execute with circuit breaker protection
            try:
                result = await self._circuit_breaker.execute(_execute_insert)
            except Exception as circuit_error:
                # Check if it's a CircuitBreakerOpenError
                if circuit_error.__class__.__name__ == "CircuitBreakerOpenError":
                    error_msg = f"Circuit breaker is OPEN: {circuit_error!s}"
                    self._logger.log_operation_error(
                        correlation_id=correlation_id,
                        error=error_msg,
                        operation_type="persist_workflow_step",
                    )
                    execution_time_ms = int((time.perf_counter() - start_time) * 1000)
                    return self._create_error_output(
                        correlation_id=correlation_id,
                        operation_type="persist_workflow_step",
                        error_msg=error_msg,
                        execution_time_ms=execution_time_ms,
                    )
                raise  # Re-raise other exceptions

            # Step 4: Success Logging and Metrics
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            self._logger.log_query_success(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
            )

            # Performance check
            if execution_time_ms > 10:
                self._logger.logger.warning(
                    "Workflow step persistence exceeded performance target",
                    correlation_id=str(correlation_id),
                    execution_time_ms=execution_time_ms,
                    target_ms=10,
                )

            # Return success output
            return self._create_success_output(
                correlation_id=correlation_id,
                operation_type="persist_workflow_step",
                execution_time_ms=execution_time_ms,
                rows_affected=1,
            )

        except Exception as e:
            # Step 5: Error Handling
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Check for foreign key violations (workflow_id not found)
            error_str = str(e).lower()
            if "foreign key" in error_str and "workflow_id" in error_str:
                error_msg = f"Foreign key violation: workflow_id '{input_data.workflow_id}' does not exist in workflow_executions table"
                self._logger.log_operation_error(
                    correlation_id=correlation_id,
                    error=error_msg,
                    operation_type="persist_workflow_step",
                )
                return self._create_error_output(
                    correlation_id=correlation_id,
                    operation_type="persist_workflow_step",
                    error_msg=error_msg,
                    execution_time_ms=execution_time_ms,
                )

            # Generic error handling
            error_msg = f"Failed to persist workflow step: {e!s}"
            self._logger.log_operation_error(
                correlation_id=correlation_id,
                error=e,
                operation_type="persist_workflow_step",
            )

            return self._create_error_output(
                correlation_id=correlation_id,
                operation_type="persist_workflow_step",
                error_msg=error_msg,
                execution_time_ms=execution_time_ms,
            )

    def _create_success_output(
        self,
        correlation_id: UUID,
        operation_type: str,
        execution_time_ms: int,
        rows_affected: int,
    ) -> ModelDatabaseOperationOutput:
        """
        Create success output for database operations.

        Args:
            correlation_id: Request correlation ID
            operation_type: Type of database operation
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected

        Returns:
            ModelDatabaseOperationOutput with success status
        """
        return ModelDatabaseOperationOutput(
            success=True,
            operation_type=operation_type,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            error_message=None,
        )

    def _create_error_output(
        self,
        correlation_id: UUID,
        operation_type: str,
        error_msg: str,
        execution_time_ms: int,
    ) -> ModelDatabaseOperationOutput:
        """
        Create error output for database operations.

        Args:
            correlation_id: Request correlation ID
            operation_type: Type of database operation
            error_msg: Error message
            execution_time_ms: Execution time in milliseconds

        Returns:
            ModelDatabaseOperationOutput with error status
        """
        return ModelDatabaseOperationOutput(
            success=False,
            operation_type=operation_type,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=0,
            error_message=error_msg,
        )
