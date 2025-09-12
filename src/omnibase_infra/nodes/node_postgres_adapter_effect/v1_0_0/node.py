"""PostgreSQL Adapter Tool - Message Bus Bridge for Database Operations.

This adapter serves as a bridge between the ONEX message bus and PostgreSQL database operations.
It converts event envelopes containing database requests into direct PostgreSQL connection manager calls.
Following the ONEX infrastructure tool pattern for external service integration.
"""

import asyncio
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Union, Pattern
from uuid import UUID, uuid4

from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
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


class PostgresStructuredLogger:
    """
    Structured logger for PostgreSQL adapter operations with correlation ID tracking.
    
    Provides consistent, structured logging across all database operations with:
    - Correlation ID tracking for request tracing
    - Performance metrics logging
    - Error context preservation
    - Security-aware message sanitization
    """
    
    def __init__(self, logger_name: str = "postgres_adapter"):
        """Initialize structured logger with correlation ID support."""
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            # Configure structured logging format if not already configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(operation)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _build_extra(self, correlation_id: Optional[UUID], operation: str, **kwargs) -> dict:
        """Build extra fields for structured logging."""
        extra = {
            'correlation_id': str(correlation_id) if correlation_id else 'no-correlation',
            'operation': operation,
            'component': 'postgres_adapter',
            'node_type': 'effect',
        }
        extra.update(kwargs)
        return extra
    
    def info(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log info level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log warning level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", 
              exception: Optional[Exception] = None, **kwargs):
        """Log error level message with structured fields and exception context."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        if exception:
            extra['exception_type'] = type(exception).__name__
            extra['exception_message'] = str(exception)
        self.logger.error(message, extra=extra, exc_info=exception is not None)
    
    def debug(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log debug level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.debug(message, extra=extra)
    
    def log_query_start(self, correlation_id: UUID, query: str, params_count: int):
        """Log start of database query execution."""
        self.info(
            f"Starting database query execution (params: {params_count})",
            correlation_id=correlation_id,
            operation="query_start",
            query_length=len(query),
            parameters_count=params_count,
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )
    
    def log_query_success(self, correlation_id: UUID, execution_time_ms: float, rows_affected: int):
        """Log successful database query completion."""
        self.info(
            f"Database query completed successfully in {execution_time_ms:.2f}ms (rows: {rows_affected})",
            correlation_id=correlation_id,
            operation="query_success",
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            performance_category="fast" if execution_time_ms < 100 else "slow" if execution_time_ms < 1000 else "very_slow"
        )
    
    def log_query_error(self, correlation_id: UUID, execution_time_ms: float, exception: Exception):
        """Log database query error with context."""
        self.error(
            f"Database query failed after {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation="query_error",
            exception=exception,
            execution_time_ms=execution_time_ms,
            error_category=self._categorize_db_error(exception)
        )
    
    def log_circuit_breaker_event(self, correlation_id: Optional[UUID], event: str, state: str, **kwargs):
        """Log circuit breaker state changes and events."""
        self.warning(
            f"Circuit breaker {event} - state: {state}",
            correlation_id=correlation_id,
            operation="circuit_breaker",
            circuit_state=state,
            event_type=event,
            **kwargs
        )
    
    def log_health_check(self, check_name: str, status: str, execution_time_ms: float, **kwargs):
        """Log health check results."""
        level_method = self.info if status == "healthy" else self.warning if status == "degraded" else self.error
        level_method(
            f"Health check '{check_name}' returned {status} in {execution_time_ms:.2f}ms",
            operation="health_check",
            check_name=check_name,
            health_status=status,
            execution_time_ms=execution_time_ms,
            **kwargs
        )
    
    def _categorize_db_error(self, exception: Exception) -> str:
        """Categorize database errors for better observability."""
        error_str = str(exception).lower()
        if "connection" in error_str or "timeout" in error_str:
            return "connectivity"
        elif "syntax" in error_str or "invalid" in error_str:
            return "query_syntax"
        elif "permission" in error_str or "access" in error_str:
            return "authorization"
        elif "constraint" in error_str or "duplicate" in error_str:
            return "data_integrity"
        else:
            return "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states for database connectivity failures."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class DatabaseCircuitBreaker:
    """
    Circuit breaker implementation for database connectivity failures.
    
    Prevents cascading failures by monitoring database operation failures
    and temporarily blocking requests when failure thresholds are exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60, half_open_max_calls: int = 3):
        """
        Initialize circuit breaker with configurable thresholds.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting recovery
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            OnexError: If circuit is open or function fails
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Database circuit breaker is OPEN - service temporarily unavailable",
                    )
            
            # In half-open state, limit calls
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Database circuit breaker is HALF_OPEN - maximum test calls exceeded",
                    )
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise
    
    async def _record_success(self):
        """Record successful operation and potentially close circuit."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Reset to closed state after successful test
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
                self.half_open_calls = 0
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self, exception: Exception):
        """Record failed operation and potentially open circuit."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure >= timedelta(seconds=self.timeout_seconds)
    
    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "half_open_calls": self.half_open_calls if self.state == CircuitBreakerState.HALF_OPEN else 0,
        }


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
    
    # Pre-compiled regex patterns for PostgreSQL status parsing (performance optimization)
    _ROWS_AFFECTED_PATTERNS = [
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

    def __init__(self, container: ModelONEXContainer):
        """Initialize PostgreSQL adapter tool with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "infrastructure"
        self._connection_manager: Optional[PostgresConnectionManager] = None
        self._connection_manager_lock = asyncio.Lock()
        self._connection_manager_sync_lock = threading.Lock()
        
        # Initialize circuit breaker for database connectivity failures
        self._circuit_breaker = DatabaseCircuitBreaker(
            failure_threshold=5,  # Open circuit after 5 failures
            timeout_seconds=60,   # Wait 60 seconds before retry
            half_open_max_calls=3  # Allow 3 test calls in half-open state
        )
        
        # Initialize structured logger with correlation ID support
        self._logger = PostgresStructuredLogger("postgres_adapter_node")
        
        # Initialize configuration from environment or container
        self.config = self._load_configuration(container)
        
        # Log adapter initialization
        self._logger.info(
            "PostgreSQL adapter initialized successfully",
            operation="initialization",
            node_type=self.node_type,
            domain=self.domain
        )

    def _load_configuration(self, container: ModelONEXContainer) -> ModelPostgresAdapterConfig:
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
            if config and hasattr(config, 'postgres_host') and hasattr(config, 'postgres_port'):
                return config
        except Exception:
            pass  # Fall back to environment configuration
        
        # Fall back to environment-based configuration
        environment = os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
        return ModelPostgresAdapterConfig.for_environment(environment)
    
    def _validate_correlation_id(self, correlation_id: Optional[UUID]) -> UUID:
        """
        Validate and normalize correlation ID to prevent injection attacks.
        
        Args:
            correlation_id: Optional correlation ID to validate
            
        Returns:
            Valid UUID correlation ID
            
        Raises:
            OnexError: If correlation ID format is invalid
        """
        if correlation_id is None:
            # Generate a new correlation ID if none provided
            return uuid4()
            
        if hasattr(correlation_id, 'replace') and hasattr(correlation_id, 'split'):  # String-like
            try:
                # Try to parse string as UUID to validate format
                correlation_id = UUID(correlation_id)
            except ValueError as e:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message="Invalid correlation ID format - must be valid UUID"
                ) from e
                
        if not hasattr(correlation_id, 'hex'):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID must be UUID type"
            )
            
        # Additional validation: ensure it's not an empty UUID
        if correlation_id == UUID('00000000-0000-0000-0000-000000000000'):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID cannot be empty UUID"
            )
            
        return correlation_id

    @property
    def connection_manager(self) -> PostgresConnectionManager:
        """
        Get PostgreSQL connection manager instance via registry injection with thread safety.
        
        Note: For async operations, prefer get_connection_manager_async() to avoid mixing sync/async patterns.
        """
        with self._connection_manager_sync_lock:
            if self._connection_manager is None:
                # Validate container service interface before resolution
                self._validate_container_service_interface()
                
                # Use container injection per ONEX standards
                self._connection_manager = self.container.get_service("postgres_connection_manager")
                
                # Validate the resolved service interface
                self._validate_connection_manager_interface(self._connection_manager)
                
            return self._connection_manager
    
    async def get_connection_manager_async(self) -> PostgresConnectionManager:
        """
        Get PostgreSQL connection manager instance via registry injection with thread safety.
        
        Returns:
            PostgresConnectionManager instance
            
        Raises:
            OnexError: If connection manager cannot be resolved
        """
        async with self._connection_manager_lock:
            if self._connection_manager is None:
                # Validate container service interface before resolution
                self._validate_container_service_interface()
                
                # Use container injection per ONEX standards
                self._connection_manager = self.container.get_service("postgres_connection_manager")
                
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
            self._check_circuit_breaker_health,
        ]

    def _check_database_connectivity(self) -> ModelHealthStatus:
        """Check basic PostgreSQL database connectivity (sync wrapper for health checks)."""
        start_time = time.perf_counter()
        try:
            # Simple sync health check without async operations
            # This avoids event loop complexity in health check context
            if self._connection_manager is None:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._logger.log_health_check(
                    check_name="database_connectivity",
                    status="degraded",
                    execution_time_ms=execution_time_ms,
                    reason="connection_manager_not_initialized"
                )
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Connection manager not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Basic connectivity indicator based on manager state
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.log_health_check(
                check_name="database_connectivity",
                status="healthy",
                execution_time_ms=execution_time_ms,
                reason="connection_manager_operational"
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Database connection manager operational",
                timestamp=datetime.utcnow().isoformat()
            )
                    
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.log_health_check(
                check_name="database_connectivity",
                status="unhealthy",
                execution_time_ms=execution_time_ms,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Database connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def _check_database_connectivity_async(self) -> ModelHealthStatus:
        """Check basic PostgreSQL database connectivity (async version for operation handlers)."""
        try:
            # Async connectivity test via connection manager
            connection_manager = await self.get_connection_manager_async()
            health_data = await connection_manager.health_check()
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
        """Check PostgreSQL connection pool health and capacity (sync version for mixin)."""
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

    async def _check_connection_pool_health_async(self) -> ModelHealthStatus:
        """Check PostgreSQL connection pool health and capacity (async version for operation handlers)."""
        try:
            # Check if connection manager is available with proper async access
            connection_manager = await self.get_connection_manager_async()
            stats = connection_manager.get_connection_stats()
            
            # Check pool health based on connection stats
            if stats.failed_connections > stats.total_connections * 0.1:  # More than 10% failures
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message=f"High connection failure rate: {stats.failed_connections}/{stats.total_connections}",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Connection pool healthy: {stats.size}/{stats.total_connections} connections",
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Connection pool check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    def _check_circuit_breaker_health(self) -> ModelHealthStatus:
        """Check circuit breaker health and state (sync version for health checks)."""
        try:
            circuit_state = self._circuit_breaker.get_state()
            state_value = circuit_state["state"]
            
            if state_value == CircuitBreakerState.CLOSED.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message=f"Circuit breaker CLOSED - failures: {circuit_state['failure_count']}",
                    timestamp=datetime.utcnow().isoformat()
                )
            elif state_value == CircuitBreakerState.HALF_OPEN.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message=f"Circuit breaker HALF_OPEN - testing recovery ({circuit_state['half_open_calls']} calls)",
                    timestamp=datetime.utcnow().isoformat()
                )
            else:  # OPEN state
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message=f"Circuit breaker OPEN - service temporarily unavailable (failures: {circuit_state['failure_count']})",
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Circuit breaker health check failed: {str(e)}",
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
            # Validate and normalize correlation ID to prevent injection attacks
            validated_correlation_id = self._validate_correlation_id(input_data.correlation_id)
            
            # Update the input data with validated correlation ID if it was modified
            if validated_correlation_id != input_data.correlation_id:
                input_data.correlation_id = validated_correlation_id
            
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
            
            if hasattr(e, 'code') and hasattr(e, 'message'):  # OnexError-like
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
        correlation_id = input_data.correlation_id
        
        # Log query start with structured logging
        self._logger.log_query_start(
            correlation_id=correlation_id,
            query=query_request.query,
            params_count=len(query_request.parameters)
        )
        
        # Input validation for security and performance
        self._validate_query_input(query_request)
        
        try:
            # Execute query through connection manager with circuit breaker protection
            connection_manager = await self.get_connection_manager_async()
            
            # Wrap database call in circuit breaker for failure protection
            result = await self._circuit_breaker.call(
                connection_manager.execute_query,
                query_request.query,
                *query_request.parameters,
                timeout=query_request.timeout,
                record_metrics=query_request.record_metrics,
            )
            
            # Convert result to response format (as defined in event processing subcontract)
            if hasattr(result, '__iter__') and hasattr(result, '__len__'):  # List-like (SELECT query result)
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

            # Log successful query completion
            self._logger.log_query_success(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=rows_affected
            )

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
            
            # Log query error with structured logging
            self._logger.log_query_error(
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                exception=e
            )
            
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
            # Run health checks using async versions for consistent async operation
            health_results = []
            
            # Run async database connectivity check
            db_health = await self._check_database_connectivity_async()
            health_results.append(db_health)
            
            # Run async connection pool check (fixed async/sync mixing)
            pool_health = await self._check_connection_pool_health_async()
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
            connection_manager = await self.get_connection_manager_async()
            await connection_manager.initialize()
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
        if not status_result or not (hasattr(status_result, 'strip') and hasattr(status_result, 'split')):  # String-like check
            return 0
            
        # Clean the status string
        status_clean = status_result.strip()
        if not status_clean:
            return 0
        
        # Try each pre-compiled pattern to extract rows affected (performance optimized)
        for pattern, group_index in self._ROWS_AFFECTED_PATTERNS:
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