# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Error Response Mixin.

Provides standardized PostgreSQL exception handling for persistence operations
in NodeContractPersistenceEffect. Extracts the common ~60-line exception
handling pattern into a reusable mixin.

Architecture:
    MixinPostgresErrorResponse is designed to be mixed into PostgreSQL
    persistence classes to provide consistent error handling, sanitization,
    logging, and ModelBackendResult construction.

    The mixin handles:
    - TimeoutError/InfraTimeoutError -> TIMEOUT_ERROR code
    - InfraAuthenticationError -> AUTH_ERROR code
    - InfraConnectionError -> CONNECTION_ERROR code
    - RepositoryExecutionError -> operation-specific error code
    - Generic Exception -> UNKNOWN_ERROR code

Error Sanitization:
    All error messages are sanitized using utility functions to prevent
    exposure of sensitive information (credentials, connection strings)
    in logs and responses.

Logging:
    - Timeout/Connection errors: logger.warning (retriable)
    - Auth errors: logger.exception (non-retriable, needs attention)
    - Repository errors: logger.warning (may be retriable)
    - Unknown errors: logger.exception (needs investigation)

Usage:
    >>> class MyPersistence(MixinPostgresErrorResponse):
    ...     async def handle(self, payload, correlation_id):
    ...         start_time = time.perf_counter()
    ...         try:
    ...             # ... database operation ...
    ...         except Exception as e:
    ...             ctx = PostgresErrorContext(
    ...                 exception=e,
    ...                 operation="my_operation",
    ...                 correlation_id=correlation_id,
    ...                 start_time=start_time,
    ...                 log_context={"my_field": "value"},
    ...                 operation_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
    ...             )
    ...             return self._build_error_response(ctx)

Related:
    - NodeContractPersistenceEffect: Parent effect node
    - EnumPostgresErrorCode: Error code enumeration
    - ModelBackendResult: Structured result model
    - OMN-1845: Implementation ticket
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    RepositoryExecutionError,
)
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_infra.nodes.effects.models.model_backend_result import (
        ModelBackendResult,
    )

logger = logging.getLogger(__name__)


@dataclass
class PostgresErrorContext:
    """Context for PostgreSQL error handling.

    Encapsulates all parameters needed for error handling to reduce
    function parameter count and improve readability.

    Attributes:
        exception: The exception that was raised during the operation.
        operation: Name of the operation for logging.
        correlation_id: Request correlation ID for distributed tracing.
        start_time: Result of time.perf_counter() captured before operation.
        log_context: Additional context fields for log messages.
        operation_error_code: Error code for RepositoryExecutionError.
    """

    exception: Exception
    operation: str
    correlation_id: UUID
    start_time: float
    log_context: dict[str, object] = field(default_factory=dict)
    operation_error_code: EnumPostgresErrorCode | None = None


class MixinPostgresErrorResponse:
    """Mixin providing standardized PostgreSQL exception handling.

    Consolidates the common exception handling pattern used across all
    PostgreSQL handlers in NodeContractPersistenceEffect. This ensures
    consistent error classification, sanitization, logging, and result
    construction.

    The mixin is designed to be used with any class that needs to handle
    PostgreSQL operation errors and return ModelBackendResult.

    Error Handling Matrix:
        | Exception Type           | Error Code       | Log Level  | Retriable |
        |--------------------------|------------------|------------|-----------|
        | TimeoutError             | TIMEOUT_ERROR    | warning    | Yes       |
        | InfraTimeoutError        | TIMEOUT_ERROR    | warning    | Yes       |
        | InfraAuthenticationError | AUTH_ERROR       | exception  | No        |
        | InfraConnectionError     | CONNECTION_ERROR | warning    | Yes       |
        | RepositoryExecutionError | (configurable)   | warning    | Maybe     |
        | Exception (catch-all)    | UNKNOWN_ERROR    | exception  | No        |

    Example:
        >>> class HandlerPostgresExample(MixinPostgresErrorResponse):
        ...     async def handle(self, payload, correlation_id):
        ...         start_time = time.perf_counter()
        ...         try:
        ...             async with self._pool.acquire() as conn:
        ...                 await conn.execute("SELECT 1")
        ...             duration_ms = (time.perf_counter() - start_time) * 1000
        ...             return ModelBackendResult(
        ...                 success=True,
        ...                 duration_ms=duration_ms,
        ...                 backend_id="postgres",
        ...                 correlation_id=correlation_id,
        ...             )
        ...         except Exception as e:
        ...             ctx = PostgresErrorContext(
        ...                 exception=e,
        ...                 operation="example_operation",
        ...                 correlation_id=correlation_id,
        ...                 start_time=start_time,
        ...             )
        ...             return self._build_error_response(ctx)

    See Also:
        - HandlerPostgresContractUpsert: Example handler using this mixin
        - HandlerPostgresTopicUpdate: Example handler using this mixin
        - EnumPostgresErrorCode: Error code classification
    """

    def _build_error_response(
        self,
        ctx: PostgresErrorContext,
    ) -> ModelBackendResult:
        """Build ModelBackendResult for PostgreSQL operation exceptions.

        Processes an exception raised during a PostgreSQL operation and
        returns a properly constructed ModelBackendResult with:
        - Appropriate error code based on exception type
        - Sanitized error message (no credentials/PII)
        - Operation duration in milliseconds
        - Correlation ID for distributed tracing

        Args:
            ctx: PostgresErrorContext containing all error handling parameters.

        Returns:
            ModelBackendResult with:
                - success: Always False (this is error handling)
                - error: Sanitized error message safe for logs/responses
                - error_code: EnumPostgresErrorCode based on exception type
                - duration_ms: Operation duration in milliseconds
                - backend_id: Always "postgres"
                - correlation_id: Passed through for tracing

        Note:
            This method never raises exceptions. All error paths return
            a properly constructed ModelBackendResult.
        """
        # Extract context fields for readability
        exception = ctx.exception
        operation = ctx.operation
        correlation_id = ctx.correlation_id
        start_time = ctx.start_time
        log_context = ctx.log_context
        operation_error_code = ctx.operation_error_code
        # Local import to avoid circular import at module load time
        # (mixins/__init__.py loads before nodes/__init__.py in some paths)
        from omnibase_infra.nodes.effects.models.model_backend_result import (
            ModelBackendResult as BackendResult,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Build base log context
        base_context: dict[str, object] = {
            "correlation_id": str(correlation_id),
            "duration_ms": duration_ms,
        }

        # Merge caller-provided context
        if log_context:
            base_context.update(log_context)

        # Handle timeout errors - retriable infrastructure failures
        if isinstance(exception, (TimeoutError, InfraTimeoutError)):
            sanitized_error = sanitize_error_message(exception)
            base_context["error"] = sanitized_error

            logger.warning(
                f"{operation} timed out",
                extra=base_context,
            )

            return BackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.TIMEOUT_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        # Handle authentication errors - non-retriable configuration failures
        if isinstance(exception, InfraAuthenticationError):
            sanitized_error = sanitize_error_message(exception)
            base_context["error"] = sanitized_error

            logger.exception(
                f"{operation} authentication failed",
                extra=base_context,
            )

            return BackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.AUTH_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        # Handle connection errors - retriable infrastructure failures
        if isinstance(exception, InfraConnectionError):
            sanitized_error = sanitize_error_message(exception)
            base_context["error"] = sanitized_error

            logger.warning(
                f"{operation} connection failed",
                extra=base_context,
            )

            return BackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.CONNECTION_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        # Handle repository execution errors - operation-specific failures
        if isinstance(exception, RepositoryExecutionError):
            sanitized_error = sanitize_error_message(exception)
            base_context["error"] = sanitized_error

            logger.warning(
                f"{operation} execution failed",
                extra=base_context,
            )

            # Use operation-specific error code if provided, else fall back to UNKNOWN
            error_code = operation_error_code or EnumPostgresErrorCode.UNKNOWN_ERROR

            return BackendResult(
                success=False,
                error=sanitized_error,
                error_code=error_code,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        # Generic catch-all for unexpected exceptions
        # This catch-all is required because database adapters may raise
        # unexpected exceptions beyond typed infrastructure errors (e.g.,
        # driver errors, encoding errors, connection pool errors, asyncpg-
        # specific exceptions). All errors must be sanitized to prevent
        # credential exposure.
        sanitized_error = sanitize_backend_error("postgres", exception)
        base_context["error"] = sanitized_error
        base_context["error_type"] = type(exception).__name__

        logger.exception(
            f"{operation} failed with unexpected error",
            extra=base_context,
        )

        return BackendResult(
            success=False,
            error=sanitized_error,
            error_code=EnumPostgresErrorCode.UNKNOWN_ERROR,
            duration_ms=duration_ms,
            backend_id="postgres",
            correlation_id=correlation_id,
        )


__all__: list[str] = ["MixinPostgresErrorResponse", "PostgresErrorContext"]
