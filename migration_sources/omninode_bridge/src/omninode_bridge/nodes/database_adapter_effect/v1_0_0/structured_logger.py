"""
Structured logger for database adapter with correlation ID tracking and query sanitization.

This module provides comprehensive logging for database operations with:
- Correlation ID tracking across all log entries
- Query sanitization to remove sensitive data (passwords, connection strings)
- Performance categorization (fast < 100ms, slow >= 100ms)
- Operation-based filtering (query, transaction, health_check)
- Error message sanitization to prevent information leakage
"""

import re
from enum import Enum
from typing import Any
from uuid import UUID

import structlog


class DatabaseOperationType(Enum):
    """Database operation types for categorization and filtering."""

    QUERY = "query"
    TRANSACTION = "transaction"
    HEALTH_CHECK = "health_check"
    CONNECTION_ACQUIRE = "connection_acquire"
    CONNECTION_RELEASE = "connection_release"
    BATCH_INSERT = "batch_insert"
    PREPARED_STATEMENT = "prepared_statement"


class PerformanceCategory(Enum):
    """Performance categorization based on execution time."""

    FAST = "fast"  # < 100ms
    SLOW = "slow"  # >= 100ms
    VERY_SLOW = "very_slow"  # >= 1000ms


class DatabaseStructuredLogger:
    """
    Structured logger with correlation ID tracking for database operations.

    Provides security-aware logging with automatic sanitization of sensitive
    information in queries and error messages.

    Features:
    - Correlation ID tracking for distributed tracing
    - Query sanitization (passwords, connection strings, sensitive data)
    - Performance categorization (fast/slow/very_slow)
    - Operation-based filtering
    - Error message sanitization
    - JSON output format via structlog

    Example:
        logger = DatabaseStructuredLogger(component="metadata_stamping_db")
        logger.log_operation_start(
            correlation_id=uuid.uuid4(),
            operation_type=DatabaseOperationType.QUERY,
            metadata={"table": "metadata_stamps", "params": 3}
        )
    """

    def __init__(
        self,
        component: str = "database_adapter",
        node_type: str = "effect",
        enable_query_sanitization: bool = True,
        enable_error_sanitization: bool = True,
    ):
        """
        Initialize structured logger.

        Args:
            component: Component name for log context
            node_type: Node type (effect, service, etc.)
            enable_query_sanitization: Enable query sanitization
            enable_error_sanitization: Enable error message sanitization
        """
        self.component = component
        self.node_type = node_type
        self.enable_query_sanitization = enable_query_sanitization
        self.enable_error_sanitization = enable_error_sanitization

        # Get structlog logger
        self.logger = structlog.get_logger(f"{component}.database")

        # Pre-compile sanitization patterns for performance
        self._query_sanitization_patterns = self._compile_query_patterns()
        self._error_sanitization_patterns = self._compile_error_patterns()

    def _compile_query_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Compile regex patterns for query sanitization."""
        return [
            # Password in connection strings
            (re.compile(r"password=[^\s;&]*", re.IGNORECASE), "password=***"),
            (re.compile(r"pwd=[^\s;&]*", re.IGNORECASE), "pwd=***"),
            # PostgreSQL connection strings
            (
                re.compile(r"postgresql://[^:]+:[^@]+@", re.IGNORECASE),
                "postgresql://***:***@",
            ),
            (
                re.compile(r"postgres://[^:]+:[^@]+@", re.IGNORECASE),
                "postgres://***:***@",
            ),
            # Generic connection strings
            (
                re.compile(r"(user|username)=[^\s;&]+", re.IGNORECASE),
                r"\1=***",
            ),
            # API keys and tokens
            (re.compile(r"api[_-]?key=[^\s;&]*", re.IGNORECASE), "api_key=***"),
            (re.compile(r"token=[^\s;&]*", re.IGNORECASE), "token=***"),
            (re.compile(r"secret=[^\s;&]*", re.IGNORECASE), "secret=***"),
            # Credit card patterns (basic)
            (
                re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
                "****-****-****-****",
            ),
            # Email addresses (partial masking)
            (
                re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
                r"***@\2",
            ),
        ]

    def _compile_error_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Compile regex patterns for error message sanitization."""
        return [
            # Database connection strings
            (re.compile(r"postgresql://[^\s]*"), "postgresql://***"),
            (re.compile(r"postgres://[^\s]*"), "postgres://***"),
            # Database schema names
            (re.compile(r'schema "[\w_-]+"', re.IGNORECASE), 'schema "***"'),
            (re.compile(r'database "[\w_-]+"', re.IGNORECASE), 'database "***"'),
            (re.compile(r'table "[\w_-]+"', re.IGNORECASE), 'table "***"'),
            # Passwords and secrets
            (re.compile(r"password=[^\s]*", re.IGNORECASE), "password=***"),
            (re.compile(r"pwd=[^\s]*", re.IGNORECASE), "pwd=***"),
            (re.compile(r"secret=[^\s]*", re.IGNORECASE), "secret=***"),
            # File paths that might contain sensitive info
            (re.compile(r"/home/[\w_-]+", re.IGNORECASE), "/home/***"),
            (re.compile(r"/users/[\w_-]+", re.IGNORECASE), "/users/***"),
            # IP addresses (partial masking)
            (re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b"), r"\1.***"),
            # Port numbers in connection strings
            (re.compile(r":[\d]{4,5}@"), ":***@"),
        ]

    def sanitize_query(self, sql: str) -> str:
        """
        Sanitize SQL query by removing sensitive data.

        Removes or masks:
        - Passwords and credentials
        - Connection strings
        - API keys and tokens
        - Credit card numbers
        - Email addresses (partial)

        Args:
            sql: SQL query string to sanitize

        Returns:
            Sanitized SQL query string
        """
        if not self.enable_query_sanitization:
            return sql

        sanitized = sql
        for pattern, replacement in self._query_sanitization_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def sanitize_error(self, error_msg: str) -> str:
        """
        Sanitize error message to prevent sensitive information leakage.

        Removes or masks:
        - Connection strings and passwords
        - Database schema details
        - Internal system paths
        - IP addresses (partial)
        - Port numbers

        Args:
            error_msg: Error message to sanitize

        Returns:
            Sanitized error message
        """
        if not self.enable_error_sanitization:
            return error_msg

        sanitized = error_msg
        for pattern, replacement in self._error_sanitization_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def _categorize_performance(self, execution_time_ms: float) -> PerformanceCategory:
        """
        Categorize operation performance based on execution time.

        Args:
            execution_time_ms: Execution time in milliseconds

        Returns:
            Performance category
        """
        if execution_time_ms < 100:
            return PerformanceCategory.FAST
        elif execution_time_ms < 1000:
            return PerformanceCategory.SLOW
        else:
            return PerformanceCategory.VERY_SLOW

    def _build_base_context(
        self,
        correlation_id: UUID | str | None,
        operation_type: DatabaseOperationType | str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Build base context for log entries.

        Args:
            correlation_id: Correlation ID for distributed tracing
            operation_type: Type of database operation
            **kwargs: Additional context fields

        Returns:
            Dictionary with base context fields
        """
        context = {
            "correlation_id": (
                str(correlation_id) if correlation_id else "no-correlation"
            ),
            "operation": (
                operation_type.value
                if isinstance(operation_type, DatabaseOperationType)
                else operation_type
            ),
            "component": self.component,
            "node_type": self.node_type,
        }
        context.update(kwargs)
        return context

    def log_operation_start(
        self,
        correlation_id: UUID | str | None,
        operation_type: DatabaseOperationType | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log start of database operation execution.

        Args:
            correlation_id: Correlation ID for distributed tracing
            operation_type: Type of operation (query, transaction, etc.)
            metadata: Additional operation metadata (table, params, etc.)
        """
        context = self._build_base_context(
            correlation_id=correlation_id,
            operation_type=operation_type,
            operation_status="started",
        )

        # Add metadata if provided
        if metadata:
            # Sanitize any query strings in metadata
            if "query" in metadata and self.enable_query_sanitization:
                metadata = metadata.copy()
                metadata["query"] = self.sanitize_query(metadata["query"])
                metadata["query_length"] = len(metadata.get("query", ""))

            context["metadata"] = metadata

        self.logger.info(
            f"Starting database operation: {operation_type}",
            **context,
        )

    def log_operation_complete(
        self,
        correlation_id: UUID | str | None,
        execution_time_ms: float,
        rows_affected: int | None = None,
        operation_type: DatabaseOperationType | str = DatabaseOperationType.QUERY,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Log successful completion of database operation.

        Args:
            correlation_id: Correlation ID for distributed tracing
            execution_time_ms: Operation execution time in milliseconds
            rows_affected: Number of rows affected/returned (if applicable)
            operation_type: Type of operation
            additional_context: Additional context information
        """
        performance_category = self._categorize_performance(execution_time_ms)

        context = self._build_base_context(
            correlation_id=correlation_id,
            operation_type=operation_type,
            operation_status="completed",
            execution_time_ms=round(execution_time_ms, 3),
            performance_category=performance_category.value,
        )

        if rows_affected is not None:
            context["rows_affected"] = rows_affected

        if additional_context:
            context.update(additional_context)

        # Use different log levels based on performance
        if performance_category == PerformanceCategory.FAST:
            self.logger.info(
                f"Database operation completed in {execution_time_ms:.2f}ms",
                **context,
            )
        elif performance_category == PerformanceCategory.SLOW:
            self.logger.warning(
                f"Slow database operation completed in {execution_time_ms:.2f}ms",
                **context,
            )
        else:  # VERY_SLOW
            self.logger.error(
                f"Very slow database operation completed in {execution_time_ms:.2f}ms",
                **context,
            )

    def log_operation_error(
        self,
        correlation_id: UUID | str | None,
        error: Exception | str,
        operation_type: DatabaseOperationType | str = DatabaseOperationType.QUERY,
        sanitized: bool = True,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Log database operation error.

        Args:
            correlation_id: Correlation ID for distributed tracing
            error: Exception or error message
            operation_type: Type of operation that failed
            sanitized: Whether to sanitize error message (default: True)
            additional_context: Additional context information
        """
        error_message = str(error)

        # Sanitize error message if enabled
        if sanitized and self.enable_error_sanitization:
            error_message = self.sanitize_error(error_message)

        context = self._build_base_context(
            correlation_id=correlation_id,
            operation_type=operation_type,
            operation_status="failed",
            error_message=error_message,
            error_type=(
                type(error).__name__ if isinstance(error, Exception) else "unknown"
            ),
            error_sanitized=sanitized,
        )

        if additional_context:
            context.update(additional_context)

        self.logger.error(
            f"Database operation failed: {operation_type}",
            **context,
        )

    def log_query_start(
        self,
        correlation_id: UUID | str | None,
        query: str,
        params_count: int = 0,
        table: str | None = None,
    ) -> None:
        """
        Log start of database query execution (convenience method).

        Args:
            correlation_id: Correlation ID for distributed tracing
            query: SQL query string
            params_count: Number of query parameters
            table: Target table name (if applicable)
        """
        metadata = {
            "query": query,
            "params_count": params_count,
        }
        if table:
            metadata["table"] = table

        self.log_operation_start(
            correlation_id=correlation_id,
            operation_type=DatabaseOperationType.QUERY,
            metadata=metadata,
        )

    def log_query_success(
        self,
        correlation_id: UUID | str | None,
        execution_time_ms: float,
        rows_affected: int,
    ) -> None:
        """
        Log successful query completion (convenience method).

        Args:
            correlation_id: Correlation ID for distributed tracing
            execution_time_ms: Query execution time in milliseconds
            rows_affected: Number of rows affected/returned
        """
        self.log_operation_complete(
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            operation_type=DatabaseOperationType.QUERY,
        )

    def log_transaction_start(
        self,
        correlation_id: UUID | str | None,
        isolation_level: str = "read_committed",
    ) -> None:
        """
        Log start of database transaction.

        Args:
            correlation_id: Correlation ID for distributed tracing
            isolation_level: Transaction isolation level
        """
        self.log_operation_start(
            correlation_id=correlation_id,
            operation_type=DatabaseOperationType.TRANSACTION,
            metadata={"isolation_level": isolation_level},
        )

    def log_transaction_complete(
        self,
        correlation_id: UUID | str | None,
        execution_time_ms: float,
        operations_count: int = 0,
    ) -> None:
        """
        Log successful transaction completion.

        Args:
            correlation_id: Correlation ID for distributed tracing
            execution_time_ms: Transaction execution time in milliseconds
            operations_count: Number of operations in transaction
        """
        self.log_operation_complete(
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
            operation_type=DatabaseOperationType.TRANSACTION,
            additional_context={"operations_count": operations_count},
        )

    def log_health_check(
        self,
        correlation_id: UUID | str | None,
        status: str,
        response_time_ms: float,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Log health check result.

        Args:
            correlation_id: Correlation ID for distributed tracing
            status: Health check status (healthy, unhealthy, degraded)
            response_time_ms: Health check response time
            additional_info: Additional health check information
        """
        context = {
            "health_status": status,
        }
        if additional_info:
            context.update(additional_info)

        if status == "healthy":
            self.log_operation_complete(
                correlation_id=correlation_id,
                execution_time_ms=response_time_ms,
                operation_type=DatabaseOperationType.HEALTH_CHECK,
                additional_context=context,
            )
        else:
            self.log_operation_error(
                correlation_id=correlation_id,
                error=f"Health check status: {status}",
                operation_type=DatabaseOperationType.HEALTH_CHECK,
                sanitized=False,
                additional_context=context,
            )

    def log_connection_acquired(
        self,
        correlation_id: UUID | str | None,
        pool_size: int,
        available_connections: int,
    ) -> None:
        """
        Log database connection acquisition.

        Args:
            correlation_id: Correlation ID for distributed tracing
            pool_size: Total pool size
            available_connections: Available connections in pool
        """
        self.log_operation_start(
            correlation_id=correlation_id,
            operation_type=DatabaseOperationType.CONNECTION_ACQUIRE,
            metadata={
                "pool_size": pool_size,
                "available_connections": available_connections,
                "pool_utilization": round(
                    (pool_size - available_connections) / pool_size * 100, 2
                ),
            },
        )

    def log_connection_released(
        self,
        correlation_id: UUID | str | None,
        connection_lifetime_ms: float,
    ) -> None:
        """
        Log database connection release.

        Args:
            correlation_id: Correlation ID for distributed tracing
            connection_lifetime_ms: Connection lifetime in milliseconds
        """
        self.log_operation_complete(
            correlation_id=correlation_id,
            execution_time_ms=connection_lifetime_ms,
            operation_type=DatabaseOperationType.CONNECTION_RELEASE,
        )


# Global logger instance cache
_logger_instances: dict[str, DatabaseStructuredLogger] = {}


def get_database_logger(
    component: str = "database_adapter",
    node_type: str = "effect",
    enable_query_sanitization: bool = True,
    enable_error_sanitization: bool = True,
) -> DatabaseStructuredLogger:
    """
    Get or create a database structured logger instance.

    Args:
        component: Component name for log context
        node_type: Node type (effect, service, etc.)
        enable_query_sanitization: Enable query sanitization
        enable_error_sanitization: Enable error message sanitization

    Returns:
        DatabaseStructuredLogger instance
    """
    cache_key = f"{component}:{node_type}"
    if cache_key not in _logger_instances:
        _logger_instances[cache_key] = DatabaseStructuredLogger(
            component=component,
            node_type=node_type,
            enable_query_sanitization=enable_query_sanitization,
            enable_error_sanitization=enable_error_sanitization,
        )
    return _logger_instances[cache_key]


__all__ = [
    "DatabaseStructuredLogger",
    "DatabaseOperationType",
    "PerformanceCategory",
    "get_database_logger",
]
