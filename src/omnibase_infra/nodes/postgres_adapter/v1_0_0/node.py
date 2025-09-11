"""PostgreSQL Adapter Node - Message Bus Bridge for Database Operations.

This adapter serves as a bridge between the ONEX message bus and PostgreSQL database operations.
It converts event envelopes containing database requests into direct PostgreSQL connection manager calls.
"""

import time
import uuid
from typing import Any, Dict, Optional

from omnibase_core.core.base_onex_registry import BaseOnexRegistry
from omnibase_core.core.errors.core_errors import CoreErrorCode
from omnibase_core.exceptions.base_onex_error import OnexError
from omnibase_core.nodes.base.node_effect_service import NodeEffectService

from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_query_response import ModelPostgresQueryResponse
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest
from omnibase_infra.models.postgres.model_postgres_health_response import ModelPostgresHealthResponse
from omnibase_infra.nodes.postgres_adapter.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.nodes.postgres_adapter.v1_0_0.models.model_postgres_adapter_output import ModelPostgresAdapterOutput


class PostgresAdapterNode(NodeEffectService):
    """
    PostgreSQL Adapter Node - Message Bus Bridge.
    
    Converts message bus envelopes containing database requests into direct
    PostgreSQL connection manager operations. This follows the ONEX adapter
    pattern where adapters serve as bridges between the event-driven message
    bus and external service APIs.
    
    Message Flow:
    Event Envelope → PostgreSQL Adapter → PostgreSQL Connection Manager → Database
    """

    def __init__(self, registry: BaseOnexRegistry):
        """Initialize PostgreSQL adapter with registry injection."""
        super().__init__(registry)
        self._connection_manager: Optional[PostgresConnectionManager] = None

    @property
    def connection_manager(self) -> PostgresConnectionManager:
        """Get PostgreSQL connection manager instance (lazy initialization)."""
        if self._connection_manager is None:
            # In production, this would be injected via registry
            # For now, use direct instantiation
            self._connection_manager = PostgresConnectionManager()
        return self._connection_manager

    async def process(self, input_data: ModelPostgresAdapterInput) -> ModelPostgresAdapterOutput:
        """
        Process PostgreSQL adapter request.
        
        Routes message envelope to appropriate database operation based on operation_type.
        Handles both query execution and health check operations.
        
        Args:
            input_data: Input envelope containing operation type and request data
            
        Returns:
            Output envelope with operation results
        """
        start_time = time.perf_counter()
        
        try:
            # Route based on operation type
            if input_data.operation_type == "query":
                return await self._handle_query_operation(input_data, start_time)
            elif input_data.operation_type == "health_check":
                return await self._handle_health_check_operation(input_data, start_time)
            else:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            if isinstance(e, OnexError):
                error_message = str(e)
            else:
                error_message = f"PostgreSQL adapter error: {str(e)}"
                
            return ModelPostgresAdapterOutput(
                operation_type=input_data.operation_type,
                success=False,
                error_message=error_message,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context={"error_type": type(e).__name__}
            )

    async def _handle_query_operation(
        self, 
        input_data: ModelPostgresAdapterInput, 
        start_time: float
    ) -> ModelPostgresAdapterOutput:
        """Handle database query operation."""
        if not input_data.query_request:
            raise OnexError(
                error_code=CoreErrorCode.VALIDATION_ERROR,
                message="Query request is required for query operation",
            )

        query_request = input_data.query_request
        
        try:
            # Execute query through connection manager
            result = await self.connection_manager.execute_query(
                query_request.query,
                *query_request.parameters,
                timeout=query_request.timeout,
                record_metrics=query_request.record_metrics,
            )
            
            # Convert result to response format
            if isinstance(result, list):  # SELECT query result
                data = [dict(record) for record in result] if result else []
                rows_affected = len(data)
                status_message = f"SELECT returned {rows_affected} rows"
            else:  # Non-SELECT query result (status string)
                data = None
                status_message = str(result) if result else "Query executed successfully"
                # Parse rows affected from status string
                if result and result.split():
                    try:
                        rows_affected = int(result.split()[-1])
                    except (ValueError, IndexError):
                        rows_affected = 0
                else:
                    rows_affected = 0

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Create query response
            query_response = ModelPostgresQueryResponse(
                success=True,
                data=data,
                status_message=status_message,
                rows_affected=rows_affected,
                execution_time_ms=execution_time_ms,
                correlation_id=query_request.correlation_id or input_data.correlation_id,
                context=query_request.context,
            )

            return ModelPostgresAdapterOutput(
                operation_type="query",
                query_response=query_response,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_message = str(e)

            # Create error query response
            query_response = ModelPostgresQueryResponse(
                success=False,
                data=None,
                rows_affected=0,
                execution_time_ms=execution_time_ms,
                correlation_id=query_request.correlation_id or input_data.correlation_id,
                error_message=error_message,
                context=query_request.context,
            )

            return ModelPostgresAdapterOutput(
                operation_type="query",
                query_response=query_response,
                success=False,
                error_message=error_message,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def _handle_health_check_operation(
        self, 
        input_data: ModelPostgresAdapterInput, 
        start_time: float
    ) -> ModelPostgresAdapterOutput:
        """Handle database health check operation."""
        if not input_data.health_request:
            raise OnexError(
                error_code=CoreErrorCode.VALIDATION_ERROR,
                message="Health request is required for health_check operation",
            )

        health_request = input_data.health_request

        try:
            # Perform health check through connection manager
            health_data = await self.connection_manager.health_check()
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Filter response based on request parameters
            filtered_health_data: Dict[str, Any] = {
                "status": health_data["status"],
                "timestamp": health_data["timestamp"],
                "errors": health_data["errors"],
            }

            if health_request.include_connection_stats and "connection_pool" in health_data:
                filtered_health_data["connection_pool"] = health_data["connection_pool"]

            if health_request.include_performance_metrics and "performance" in health_data:
                filtered_health_data["performance"] = health_data["performance"]

            if health_request.include_schema_info and "schema_info" in health_data:
                filtered_health_data["schema_info"] = health_data["schema_info"]

            # Include database info if available
            if "database_info" in health_data:
                filtered_health_data["database_info"] = health_data["database_info"]

            # Create health response
            health_response = ModelPostgresHealthResponse(
                status=filtered_health_data["status"],
                timestamp=filtered_health_data["timestamp"],
                connection_pool=filtered_health_data.get("connection_pool"),
                database_info=filtered_health_data.get("database_info"),
                schema_info=filtered_health_data.get("schema_info"),
                performance=filtered_health_data.get("performance"),
                errors=filtered_health_data["errors"],
                correlation_id=health_request.correlation_id or input_data.correlation_id,
                context=health_request.context,
            )

            return ModelPostgresAdapterOutput(
                operation_type="health_check",
                health_response=health_response,
                success=health_response.status in ["healthy", "degraded"],
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_message = str(e)

            # Create error health response
            health_response = ModelPostgresHealthResponse(
                status="unhealthy",
                timestamp=time.time(),
                errors=[error_message],
                correlation_id=health_request.correlation_id or input_data.correlation_id,
                context=health_request.context,
            )

            return ModelPostgresAdapterOutput(
                operation_type="health_check",
                health_response=health_response,
                success=False,
                error_message=error_message,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def initialize(self) -> None:
        """Initialize the PostgreSQL adapter and connection manager."""
        try:
            await self.connection_manager.initialize()
        except Exception as e:
            raise OnexError(
                error_code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize PostgreSQL adapter: {str(e)}",
            ) from e

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        if self._connection_manager:
            try:
                await self._connection_manager.close()
            except Exception as e:
                # Log error but don't raise during cleanup
                pass
            finally:
                self._connection_manager = None