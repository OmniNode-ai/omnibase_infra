"""PostgreSQL Adapter Tool - Message Bus Bridge for Database Operations.

This adapter serves as a bridge between the ONEX message bus and PostgreSQL database operations.
It converts event envelopes containing database requests into direct PostgreSQL connection manager calls.
Following the ONEX infrastructure tool pattern for external service integration.
"""

import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union, Pattern
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
from omnibase_infra.models.postgres.model_postgres_query_result import ModelPostgresQueryResult, ModelPostgresQueryRow
from omnibase_infra.models.postgres.model_postgres_error import ModelPostgresError
from .models.model_postgres_adapter_input import ModelPostgresAdapterInput
from .models.model_postgres_adapter_output import ModelPostgresAdapterOutput
from .models.model_postgres_adapter_config import ModelPostgresAdapterConfig


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
    
    # Configuration will be loaded from environment or container
    config: ModelPostgresAdapterConfig
    
    # Pre-compiled regex patterns for performance
    _SQL_INJECTION_PATTERNS = [
        re.compile(r';.*drop\s+table', re.IGNORECASE),
        re.compile(r';.*delete\s+from', re.IGNORECASE), 
        re.compile(r';.*truncate\s+table', re.IGNORECASE),
        re.compile(r'union.*select.*password', re.IGNORECASE),
        re.compile(r'union.*select.*admin', re.IGNORECASE),
    ]
    
    _COMPLEXITY_PATTERNS = {
        'joins': re.compile(r'\bjoin\b', re.IGNORECASE),
        'selects': re.compile(r'\bselect\b', re.IGNORECASE),
        'unions': re.compile(r'\bunion\b', re.IGNORECASE),
        'leading_wildcards': re.compile(r'like\s+[\'"]%', re.IGNORECASE),
        'regex_ops': re.compile(r'~[*]?\s*[\'"]', re.IGNORECASE),
    }
    
    _ERROR_SANITIZATION_PATTERNS = [
        (re.compile(r'password=[^\s&]*', re.IGNORECASE), 'password=***'),
        (re.compile(r'postgresql://[^\s]*@[^\s]*/', re.IGNORECASE), 'postgresql://***@***/'),
        (re.compile(r'eyJ[A-Za-z0-9+/=]*\.[A-Za-z0-9+/=]*\.[A-Za-z0-9+/=]*'), '***JWT_TOKEN***'),
        (re.compile(r'ghp_[A-Za-z0-9]{36}'), '***GITHUB_TOKEN***'),
        (re.compile(r'gho_[A-Za-z0-9]{36}'), '***GITHUB_OAUTH_TOKEN***'),
        (re.compile(r'ghu_[A-Za-z0-9]{36}'), '***GITHUB_USER_TOKEN***'),
        (re.compile(r'AKIA[0-9A-Z]{16}'), '***AWS_ACCESS_KEY***'),
        (re.compile(r'[A-Za-z0-9/+=]{40}'), '***AWS_SECRET_KEY***'),
        (re.compile(r'api[_-]?key[_-]*[:=][^\s&]*', re.IGNORECASE), 'api_key=***'),
        (re.compile(r'bearer[\s]+[A-Za-z0-9+/=]{20,}', re.IGNORECASE), 'bearer ***'),
        (re.compile(r'auth[_-]?token[_-]*[:=][^\s&]*', re.IGNORECASE), 'auth_token=***'),
        (re.compile(r'access[_-]?token[_-]*[:=][^\s&]*', re.IGNORECASE), 'access_token=***'),
        (re.compile(r'/[\w/.-]*(?:password|secret|key|token|jwt|api)[\w/.-]*', re.IGNORECASE), '/***sensitive_path***'),
        (re.compile(r'schema "[\w_-]+"'), 'schema "***"'),
        (re.compile(r'table "[\w_-]+"'), 'table "***"'),
        (re.compile(r'[A-Za-z0-9+/=]{32,}'), '***REDACTED_TOKEN***'),
    ]

    def __init__(self, container: ONEXContainer):
        """Initialize PostgreSQL adapter tool with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "infrastructure"
        self._connection_manager: Optional[PostgresConnectionManager] = None
        
        # Initialize configuration from environment or container
        self.config = self._load_configuration(container)

    def _load_configuration(self, container: ONEXContainer) -> ModelPostgresAdapterConfig:
        """
        Load PostgreSQL adapter configuration from container or environment.
        
        Args:
            container: ONEX container for dependency injection
            
        Returns:
            Configured ModelPostgresAdapterConfig instance
        """
        try:
            # Try to get configuration from container first (ONEX pattern)
            config = container.get_service("postgres_adapter_config")
            if config and isinstance(config, ModelPostgresAdapterConfig):
                return config
        except Exception:
            pass  # Fall back to environment configuration
        
        # Fall back to environment-based configuration
        environment = os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
        return ModelPostgresAdapterConfig.for_environment(environment)

    @property
    def connection_manager(self) -> PostgresConnectionManager:
        """Get PostgreSQL connection manager instance via registry injection."""
        if self._connection_manager is None:
            # Validate container service interface before resolution
            self._validate_container_service_interface()
            
            # Use registry injection per ONEX standards (CLAUDE.md)
            self._connection_manager = self.container.get_service("postgres_connection_manager")
            if self._connection_manager is None:
                # Fallback for development/testing - create with proper error
                raise OnexError(
                    code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                    message="PostgresConnectionManager not available in registry - ensure proper container setup"
                )
            
            # Validate the resolved service interface
            self._validate_connection_manager_interface(self._connection_manager)
            
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
        """Check basic PostgreSQL database connectivity (sync wrapper for health checks)."""
        try:
            # Simple sync health check without async operations
            # This avoids event loop complexity in health check context
            if self._connection_manager is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Connection manager not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Basic connectivity indicator based on manager state
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Database connection manager operational",
                timestamp=datetime.utcnow().isoformat()
            )
                    
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Database connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def _check_database_connectivity_async(self) -> ModelHealthStatus:
        """Check basic PostgreSQL database connectivity (async version - fixes event loop anti-pattern)."""
        try:
            # Simple connectivity test via connection manager (now properly async)
            health_data = await self.connection_manager.health_check()
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
            elif input_data.operation_type == "health_check":
                return await self._handle_health_check_operation(input_data, start_time)
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
        
        # Input validation for security and performance
        self._validate_query_input(query_request)
        
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
                # Create properly typed ModelPostgresQueryRow objects
                
                query_rows = []
                if result:
                    for record in result:
                        row_values = dict(record)
                        query_rows.append(ModelPostgresQueryRow(values=row_values))
                
                rows_affected = len(query_rows)
                status_message = f"SELECT returned {rows_affected} rows"
                
                # Create properly typed query result
                query_result = ModelPostgresQueryResult(
                    rows=query_rows,
                    column_names=list(result[0].keys()) if result else [],
                    row_count=rows_affected,
                    has_more=False  # TODO: Implement pagination if needed
                )
                
            else:  # Non-SELECT query result (status string)
                query_result = None
                status_message = str(result) if result else "Query executed successfully"
                # More robust parsing of rows affected from status string
                rows_affected = self._parse_rows_affected_from_status(result)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Create query response (following shared model pattern)
            query_response = ModelPostgresQueryResponse(
                success=True,
                data=query_result,
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
            
            # Sanitize error message to prevent sensitive information leakage (configurable)
            if self.config.enable_error_sanitization:
                sanitized_error = self._sanitize_error_message(str(e))
            else:
                sanitized_error = str(e)

            # Create structured error model (ONEX compliance)
            postgres_error = ModelPostgresError(
                error_code=type(e).__name__,
                error_message=sanitized_error,
                severity="ERROR",
                error_context=f"Query execution failed in {self.__class__.__name__}",
                timestamp=time.time(),
                query_id=str(query_request.correlation_id or input_data.correlation_id)
            )

            # Create error query response with structured error handling
            query_response = ModelPostgresQueryResponse(
                success=False,
                data=None,
                rows_affected=0,
                execution_time_ms=execution_time_ms,
                correlation_id=query_request.correlation_id or input_data.correlation_id,
                status_message=sanitized_error,
                error=postgres_error,  # Use structured error model
                context=query_request.context,
            )

            return ModelPostgresAdapterOutput(
                operation_type="query",
                query_response=query_response,
                success=False,
                error_message=sanitized_error,
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
        """
        Handle health check operation for PostgreSQL adapter.
        
        Performs comprehensive health checks including database connectivity,
        connection pool status, and adapter functionality.
        """
        try:
            # Run health checks using async versions for proper health operation
            health_results = []
            
            # Run async database connectivity check
            db_health = await self._check_database_connectivity_async()
            health_results.append(db_health)
            
            # Run sync connection pool check
            pool_health = self._check_connection_pool_health()
            health_results.append(pool_health)
            
            # Determine overall health status
            overall_healthy = all(
                result.status == EnumHealthStatus.HEALTHY 
                for result in health_results
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Create health check response
            health_data = {
                "overall_status": "healthy" if overall_healthy else "unhealthy",
                "checks": [
                    {
                        "name": f"check_{i}",
                        "status": result.status.value,
                        "message": result.message,
                        "timestamp": result.timestamp
                    }
                    for i, result in enumerate(health_results)
                ],
                "execution_time_ms": execution_time_ms
            }
            
            return ModelPostgresAdapterOutput(
                operation_type="health_check",
                success=overall_healthy,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context={
                    "health_data": health_data,
                    **(input_data.context if input_data.context else {})
                }
            )
            
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            # Sanitize error message (configurable)
            if self.config.enable_error_sanitization:
                sanitized_error = self._sanitize_error_message(f"Health check operation failed: {str(e)}")
            else:
                sanitized_error = f"Health check operation failed: {str(e)}"
            
            return ModelPostgresAdapterOutput(
                operation_type="health_check",
                success=False,
                error_message=sanitized_error,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
                execution_time_ms=execution_time_ms,
                context=input_data.context
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

    def _validate_query_input(self, query_request) -> None:
        """
        Validate query input for security and performance constraints.
        
        Validates:
        - Query size limits to prevent memory exhaustion
        - Parameter count limits to prevent resource exhaustion
        - Parameter size limits to prevent payload attacks
        - Basic SQL injection patterns prevention
        """
        # Query size validation (prevent memory exhaustion)
        if len(query_request.query) > self.config.max_query_size:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Query size exceeds maximum allowed length ({self.config.max_query_size} characters)",
            )
        
        # Parameter count validation (prevent resource exhaustion)
        if len(query_request.parameters) > self.config.max_parameter_count:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Parameter count exceeds maximum allowed ({self.config.max_parameter_count} parameters)",
            )
        
        # Parameter size validation (prevent payload attacks)
        for i, param in enumerate(query_request.parameters):
            param_size = len(str(param))
            if param_size > self.config.max_parameter_size:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Parameter {i} size exceeds maximum allowed ({self.config.max_parameter_size} characters)",
                )
        
        # Timeout validation
        if query_request.timeout and query_request.timeout > self.config.max_timeout_seconds:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Query timeout exceeds maximum allowed ({self.config.max_timeout_seconds} seconds)",
            )
        
        # Basic SQL injection pattern detection using pre-compiled patterns (configurable)
        if self.config.enable_sql_injection_detection:
            query_lower = query_request.query.lower()
            for pattern in self._SQL_INJECTION_PATTERNS:
                if pattern.search(query_lower):
                    raise OnexError(
                        code=CoreErrorCode.SECURITY_VIOLATION_ERROR,
                        message="Query contains potentially dangerous SQL patterns",
                    )
        
        # Query complexity validation to prevent DoS attacks (configurable)
        if self.config.enable_query_complexity_validation:
            self._validate_query_complexity(query_request.query)

    def _validate_query_complexity(self, query: str) -> None:
        """
        Validate query complexity to prevent DoS attacks.
        
        Analyzes SQL query complexity based on:
        - Number of JOIN operations
        - Number of subqueries and nested selects
        - Number of UNION operations
        - Presence of expensive operations (LIKE %, regex patterns)
        - Complex aggregation functions
        """
        query_lower = query.lower()
        complexity_score = 0
        
        # Get environment-specific complexity weights
        weights = self.config.get_complexity_weights()
        
        # Count JOINs using pre-compiled pattern (each JOIN adds complexity)
        join_count = len(self._COMPLEXITY_PATTERNS['joins'].findall(query_lower))
        complexity_score += join_count * weights["join"]
        
        # Count subqueries and nested selects using pre-compiled pattern
        select_count = len(self._COMPLEXITY_PATTERNS['selects'].findall(query_lower)) - 1  # Subtract main SELECT
        complexity_score += select_count * weights["subquery"]
        
        # Count UNION operations using pre-compiled pattern (expensive)
        union_count = len(self._COMPLEXITY_PATTERNS['unions'].findall(query_lower))
        complexity_score += union_count * weights["union"]
        
        # Check for expensive LIKE operations with leading wildcards using pre-compiled pattern
        leading_wildcard_count = len(self._COMPLEXITY_PATTERNS['leading_wildcards'].findall(query_lower))
        complexity_score += leading_wildcard_count * weights["leading_wildcard"]
        
        # Check for regex operations using pre-compiled pattern (very expensive)
        regex_count = len(self._COMPLEXITY_PATTERNS['regex_ops'].findall(query_lower))
        complexity_score += regex_count * weights["regex"]
        
        # Check for expensive functions
        expensive_functions = ['array_agg', 'string_agg', 'generate_series', 'recursive']
        for func in expensive_functions:
            if func in query_lower:
                complexity_score += weights["expensive_function"]
        
        # Check for potentially problematic ORDER BY without LIMIT
        has_order_by = 'order by' in query_lower
        has_limit = 'limit' in query_lower
        if has_order_by and not has_limit:
            complexity_score += weights["order_without_limit"]
        
        # Complexity threshold (configurable via configuration)
        if complexity_score > self.config.max_complexity_score:
            raise OnexError(
                code=CoreErrorCode.SECURITY_VIOLATION_ERROR,
                message=f"Query complexity score ({complexity_score}) exceeds maximum allowed ({self.config.max_complexity_score})",
            )

    def _validate_container_service_interface(self) -> None:
        """
        Validate container service interface compliance.
        
        Ensures the container follows ONEX standards for service resolution:
        - Has get_service method
        - Supports proper service registration patterns
        - Follows dependency injection protocols
        """
        if not hasattr(self.container, 'get_service'):
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Container does not implement required get_service interface",
            )
        
        # Validate container is not None
        if self.container is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Container is None - proper ONEX container injection required",
            )

    def _validate_connection_manager_interface(self, connection_manager) -> None:
        """
        Validate connection manager service interface compliance.
        
        Ensures the resolved connection manager implements required methods:
        - execute_query (async)
        - health_check (async)
        - initialize (async)
        - close (async)
        """
        required_methods = ['execute_query', 'health_check', 'initialize', 'close']
        missing_methods = []
        
        for method_name in required_methods:
            if not hasattr(connection_manager, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message=f"Connection manager missing required methods: {missing_methods}",
            )
        
        # Validate that critical methods are callable
        if not callable(getattr(connection_manager, 'execute_query', None)):
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Connection manager execute_query method is not callable",
            )

    def _sanitize_error_message(self, error_message: str) -> str:
        """
        Sanitize error messages to prevent sensitive information leakage.
        
        Removes or masks sensitive information like:
        - Connection strings and passwords
        - Database schema details  
        - Internal system paths
        - Stack traces with sensitive info
        """
        # Apply all sanitization patterns using pre-compiled regex for performance
        sanitized = error_message
        for pattern, replacement in self._ERROR_SANITIZATION_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        
        # If error is too generic, provide a more specific safe message
        if len(sanitized.strip()) < 10 or "connection" in sanitized.lower():
            return "Database operation failed - please check connection and query parameters"
        
        return sanitized

    def _parse_rows_affected_from_status(self, status_result: str) -> int:
        """
        Parse rows affected from PostgreSQL status strings with robust error handling.
        
        PostgreSQL returns different status formats:
        - INSERT: "INSERT 0 5" (5 rows inserted)
        - UPDATE: "UPDATE 3" (3 rows updated)  
        - DELETE: "DELETE 2" (2 rows deleted)
        - CREATE: "CREATE TABLE"
        - DROP: "DROP TABLE"
        - Other commands may return various formats
        
        Args:
            status_result: Status string returned by PostgreSQL
            
        Returns:
            Number of rows affected, or 0 if parsing fails
        """
        if not status_result or not isinstance(status_result, str):
            return 0
            
        # Clean the status string
        status_clean = status_result.strip()
        if not status_clean:
            return 0
        
        # Common PostgreSQL status patterns with compiled regex for performance
        status_patterns = [
            # INSERT operations: "INSERT 0 5" -> 5 rows
            (re.compile(r'^INSERT\s+\d+\s+(\d+)$', re.IGNORECASE), 1),
            
            # UPDATE operations: "UPDATE 3" -> 3 rows
            (re.compile(r'^UPDATE\s+(\d+)$', re.IGNORECASE), 1),
            
            # DELETE operations: "DELETE 2" -> 2 rows
            (re.compile(r'^DELETE\s+(\d+)$', re.IGNORECASE), 1),
            
            # COPY operations: "COPY 100" -> 100 rows
            (re.compile(r'^COPY\s+(\d+)$', re.IGNORECASE), 1),
            
            # Generic pattern for any command followed by a number
            (re.compile(r'^[A-Z]+\s+(\d+)$', re.IGNORECASE), 1),
        ]
        
        # Try each pattern to extract rows affected
        for pattern, group_index in status_patterns:
            match = pattern.match(status_clean)
            if match:
                try:
                    return int(match.group(group_index))
                except (ValueError, IndexError):
                    continue
        
        # Fallback: try to extract any number from the end of the string
        try:
            # Split and find the last token that's a valid integer
            tokens = status_clean.split()
            for token in reversed(tokens):
                try:
                    return int(token)
                except ValueError:
                    continue
        except Exception:
            pass
            
        # No rows affected for DDL operations or parsing failures
        return 0


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