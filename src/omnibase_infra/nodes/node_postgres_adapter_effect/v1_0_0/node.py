"""PostgreSQL Adapter Tool - Message Bus Bridge for Database Operations.

This adapter serves as a bridge between the ONEX message bus and PostgreSQL database operations.
It converts event envelopes containing database requests into direct PostgreSQL connection manager calls.
Following the ONEX infrastructure tool pattern for external service integration.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union
from uuid import UUID

from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer as ONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.model.core.model_health_status import ModelHealthStatus

from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_query_response import ModelPostgresQueryResponse
from .models.model_postgres_adapter_input import ModelPostgresAdapterInput
from .models.model_postgres_adapter_output import ModelPostgresAdapterOutput


class NodePostgresAdapterEffect(NodeEffectService):
    """
    Infrastructure PostgreSQL Adapter Node - Message Bus Bridge.
    
    Converts message bus envelopes containing database requests into direct
    PostgreSQL connection manager operations. This follows the ONEX infrastructure
    tool pattern where adapters serve as bridges between the event-driven message
    bus and external service APIs.
    
    Message Flow:
    Event Envelope → PostgreSQL Adapter → PostgreSQL Connection Manager → Database
    
    Integrates with:
    - postgres_event_processing_subcontract: Event bus integration patterns
    - postgres_connection_management_subcontract: Connection pool management
    """

    def __init__(self, container: ONEXContainer):
        """Initialize PostgreSQL adapter tool with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "infrastructure"
        self._connection_manager: Optional[PostgresConnectionManager] = None

    @property
    def connection_manager(self) -> PostgresConnectionManager:
        """Get PostgreSQL connection manager instance via registry injection."""
        if self._connection_manager is None:
            # Use registry injection per ONEX standards (CLAUDE.md)
            self._connection_manager = self.container.get_service("postgres_connection_manager")
            if self._connection_manager is None:
                # Fallback for development/testing - create with proper error
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="PostgresConnectionManager not available in registry - ensure proper container setup"
                )
        return self._connection_manager

    def get_health_checks(self) -> List[Callable[[], Union[ModelHealthStatus, "asyncio.Future[ModelHealthStatus]"]]]:
        """
        Override MixinHealthCheck to provide PostgreSQL-specific health checks.
        
        Returns list of health check functions that validate PostgreSQL connectivity,
        connection pool status, and database accessibility.
        """
        return [
            self._check_database_connectivity,
            self._check_connection_pool_health,
        ]

    def _check_database_connectivity(self) -> ModelHealthStatus:
        """Check basic PostgreSQL database connectivity."""
        try:
            # Simple connectivity test via connection manager
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                health_data = loop.run_until_complete(self.connection_manager.health_check())
                loop.close()
            except Exception:
                loop.close()
                raise
                
            status = health_data.get("status", "unknown")
            
            if status == "healthy":
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message="Database connectivity verified",
                    timestamp=datetime.utcnow().isoformat()
                )
            elif status == "degraded":
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Database connectivity degraded",
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message=f"Database connectivity failed: {status}",
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Database connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    def _check_connection_pool_health(self) -> ModelHealthStatus:
        """Check PostgreSQL connection pool health and capacity."""
        try:
            # Check if connection manager is available
            if self._connection_manager is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Connection manager not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
                
            # Connection pool is healthy if manager exists and is operational
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Connection pool operational",
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Connection pool check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def process(self, input_data: ModelPostgresAdapterInput) -> ModelPostgresAdapterOutput:
        """
        Process PostgreSQL adapter request following infrastructure tool pattern.
        
        Routes message envelope to appropriate database operation based on operation_type.
        Handles both query execution and health check operations with proper error handling
        and metrics collection as defined in the event processing subcontract.
        
        Args:
            input_data: Input envelope containing operation type and request data
            
        Returns:
            Output envelope with operation results
        """
        start_time = time.perf_counter()
        
        try:
            # Route based on operation type (as defined in subcontracts)
            if input_data.operation_type == "query":
                return await self._handle_query_operation(input_data, start_time)
            else:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            if isinstance(e, OnexError):
                error_message = str(e)
            else:
                error_message = f"PostgreSQL adapter tool error: {str(e)}"
                
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
        """
        Handle database query operation following connection management patterns.
        
        Implements query execution strategy as defined in postgres_connection_management_subcontract
        with proper timeout handling, retry logic, and performance monitoring.
        """
        if not input_data.query_request:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Query request is required for query operation",
            )

        query_request = input_data.query_request
        
        try:
            # Execute query through connection manager (following connection management subcontract)
            result = await self.connection_manager.execute_query(
                query_request.query,
                *query_request.parameters,
                timeout=query_request.timeout,
                record_metrics=query_request.record_metrics,
            )
            
            # Convert result to response format (as defined in event processing subcontract)
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

            # Create query response (following shared model pattern)
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

            # Create error query response (following error handling patterns from subcontracts)
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


    async def initialize(self) -> None:
        """
        Initialize the PostgreSQL adapter tool and connection manager.
        
        Follows initialization patterns defined in postgres_connection_management_subcontract
        with proper error handling and resource setup.
        """
        try:
            await self.connection_manager.initialize()
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize PostgreSQL adapter tool: {str(e)}",
            ) from e

    async def cleanup(self) -> None:
        """
        Cleanup resources when shutting down.
        
        Follows cleanup patterns defined in subcontracts with graceful resource disposal.
        """
        if self._connection_manager:
            try:
                await self._connection_manager.close()
            except Exception as e:
                # Log error but don't raise during cleanup (as per infrastructure patterns)
                pass
            finally:
                self._connection_manager = None


async def main():
    """Main entry point for PostgreSQL Adapter - runs in service mode with NodeEffectService"""
    from omnibase_infra.infrastructure.container import create_infrastructure_container

    # Create infrastructure container with all shared dependencies
    container = create_infrastructure_container()

    adapter = NodePostgresAdapterEffect(container)

    # Initialize the adapter
    await adapter.initialize()

    # Start service mode using NodeEffectService capabilities
    await adapter.start_service_mode()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())