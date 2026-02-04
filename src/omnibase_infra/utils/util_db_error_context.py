# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database operation error handling context manager.

This module provides an async context manager for consistent error handling
in database operations using asyncpg. It transforms low-level asyncpg exceptions
into ONEX infrastructure errors with proper context propagation.

Key Features:
    - Transforms asyncpg exceptions to ONEX infrastructure errors
    - Propagates correlation IDs for distributed tracing
    - Integrates with circuit breaker failure recording
    - Handles timeout errors with proper timeout context

Exception Mapping:
    | asyncpg Exception         | ONEX Error           | When                    |
    |---------------------------|----------------------|-------------------------|
    | QueryCanceledError        | InfraTimeoutError    | statement_timeout hit   |
    | PostgresConnectionError   | InfraConnectionError | Connection lost/failed  |
    | PostgresError (other)     | RuntimeHostError     | Other database errors   |

Usage Pattern:
    This context manager is designed to wrap the body of database write methods,
    handling the exception transformation consistently. It does NOT manage
    transactions or connections - use ``transaction_context()`` for that.

Example:
    >>> from omnibase_infra.utils import db_operation_error_context
    >>> from omnibase_infra.mixins import MixinAsyncCircuitBreaker
    >>>
    >>> class MyWriter(MixinAsyncCircuitBreaker):
    ...     async def write_data(self, data: list[Model], correlation_id: UUID | None = None):
    ...         op_correlation_id = correlation_id or uuid4()
    ...
    ...         async with db_operation_error_context(
    ...             operation="write_data",
    ...             target_name="my_table",
    ...             correlation_id=op_correlation_id,
    ...             timeout_seconds=30.0,
    ...             circuit_breaker=self,  # Pass self for circuit breaker integration
    ...         ):
    ...             async with self._pool.acquire() as conn:
    ...                 await conn.executemany(sql, data)

Related Modules:
    - ``util_db_transaction.py``: Transaction context manager for asyncpg
    - ``omnibase_infra.mixins.MixinAsyncCircuitBreaker``: Circuit breaker mixin

Related Tickets:
    - OMN-1890: Store injection metrics with corrected schema
    - PR #237: Extract shared error handling into context manager

.. versionadded:: 0.11.0
    Created to extract shared error handling from WriterInjectionEffectivenessPostgres.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)

logger = logging.getLogger(__name__)


class ProtocolCircuitBreakerFailureRecorder(Protocol):
    """Protocol for circuit breaker failure recording.

    This protocol defines the minimal interface required for recording
    circuit breaker failures. It allows the error context manager to
    work with any object that implements these methods.

    The protocol uses asyncio.Lock for thread-safe circuit breaker state
    access. This matches the MixinAsyncCircuitBreaker implementation.
    """

    _circuit_breaker_lock: asyncio.Lock
    """Lock for thread-safe circuit breaker state access."""

    async def _record_circuit_failure(
        self,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Record a circuit breaker failure.

        Args:
            operation: Name of the operation that failed.
            correlation_id: Correlation ID for tracing.
        """
        ...


@asynccontextmanager
async def db_operation_error_context(
    operation: str,
    target_name: str,
    correlation_id: UUID | None = None,
    timeout_seconds: float | None = None,
    circuit_breaker: ProtocolCircuitBreakerFailureRecorder | None = None,
) -> AsyncIterator[tuple[UUID, ModelInfraErrorContext]]:
    """Async context manager for database operation error handling.

    Wraps database operations with consistent exception handling, converting
    asyncpg exceptions to ONEX infrastructure errors. Optionally integrates
    with circuit breaker failure recording.

    This context manager yields a tuple of (correlation_id, error_context)
    that can be used within the wrapped code for logging or additional
    error context.

    Args:
        operation: Name of the operation being performed (e.g., "write_data").
            Used in error messages and circuit breaker failure recording.
        target_name: Name of the target table or resource (e.g., "users").
            Used in error context for debugging.
        correlation_id: Optional correlation ID for distributed tracing.
            If not provided, a new UUID is generated.
        timeout_seconds: Optional query timeout in seconds. If provided,
            timeout errors will include this value in the error context.
        circuit_breaker: Optional circuit breaker instance for failure recording.
            If provided, failures will be recorded via ``_record_circuit_failure()``.
            Must implement ``ProtocolCircuitBreakerFailureRecorder`` protocol.

    Yields:
        Tuple of (correlation_id, ModelInfraErrorContext):
            - correlation_id: The correlation ID (either provided or generated)
            - error_context: Pre-built error context for use in the operation

    Raises:
        InfraTimeoutError: When asyncpg.QueryCanceledError is caught
            (statement_timeout exceeded).
        InfraConnectionError: When asyncpg.PostgresConnectionError is caught
            (connection lost or failed).
        RuntimeHostError: When other asyncpg.PostgresError exceptions are caught.

    Example:
        Basic usage without circuit breaker:

        >>> async with db_operation_error_context(
        ...     operation="insert_users",
        ...     target_name="users",
        ...     timeout_seconds=30.0,
        ... ) as (corr_id, context):
        ...     async with pool.acquire() as conn:
        ...         await conn.executemany(sql, users)

        With circuit breaker integration:

        >>> class MyWriter(MixinAsyncCircuitBreaker):
        ...     async def write(self, data: list[Model]) -> int:
        ...         async with db_operation_error_context(
        ...             operation="write",
        ...             target_name="my_table",
        ...             timeout_seconds=self._query_timeout,
        ...             circuit_breaker=self,
        ...         ) as (corr_id, context):
        ...             # Database operations here
        ...             pass

    Note:
        This context manager does NOT:
        - Manage transactions (use ``transaction_context()`` for that)
        - Acquire connections from the pool
        - Set statement_timeout (caller must do this)
        - Check circuit breaker state (use ``_check_circuit_breaker()`` first)

        The caller is responsible for:
        1. Checking circuit breaker state before entering
        2. Acquiring connections and starting transactions
        3. Setting statement_timeout on the connection
        4. Resetting circuit breaker on success

    Warning:
        Circuit breaker failure recording requires the circuit breaker lock.
        This context manager acquires the lock internally when recording
        failures to ensure thread-safety.
    """
    op_correlation_id = correlation_id or uuid4()

    # Build error context upfront for use in exception handlers
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation=operation,
        target_name=target_name,
        correlation_id=op_correlation_id,
    )

    try:
        yield (op_correlation_id, context)

    except asyncpg.QueryCanceledError as e:
        # Record circuit breaker failure if provided
        if circuit_breaker is not None:
            async with circuit_breaker._circuit_breaker_lock:
                await circuit_breaker._record_circuit_failure(
                    operation=operation,
                    correlation_id=op_correlation_id,
                )

        # Build timeout-specific context
        timeout_context = ModelTimeoutErrorContext(
            transport_type=context.transport_type,
            operation=context.operation,
            target_name=context.target_name,
            correlation_id=context.correlation_id,
            timeout_seconds=timeout_seconds or 0.0,
        )

        logger.warning(
            "Database operation timed out",
            extra={
                "operation": operation,
                "target_name": target_name,
                "correlation_id": str(op_correlation_id),
                "timeout_seconds": timeout_seconds,
            },
        )

        raise InfraTimeoutError(
            f"{operation} timed out",
            context=timeout_context,
        ) from e

    except asyncpg.PostgresConnectionError as e:
        # Record circuit breaker failure if provided
        if circuit_breaker is not None:
            async with circuit_breaker._circuit_breaker_lock:
                await circuit_breaker._record_circuit_failure(
                    operation=operation,
                    correlation_id=op_correlation_id,
                )

        logger.warning(
            "Database connection failed",
            extra={
                "operation": operation,
                "target_name": target_name,
                "correlation_id": str(op_correlation_id),
                "error": str(e),
            },
        )

        raise InfraConnectionError(
            f"Database connection failed during {operation}",
            context=context,
        ) from e

    except asyncpg.PostgresError as e:
        # Record circuit breaker failure if provided
        if circuit_breaker is not None:
            async with circuit_breaker._circuit_breaker_lock:
                await circuit_breaker._record_circuit_failure(
                    operation=operation,
                    correlation_id=op_correlation_id,
                )

        logger.warning(
            "Database error occurred",
            extra={
                "operation": operation,
                "target_name": target_name,
                "correlation_id": str(op_correlation_id),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )

        raise RuntimeHostError(
            f"Database error during {operation}: {type(e).__name__}",
            context=context,
        ) from e


__all__: list[str] = [
    "ProtocolCircuitBreakerFailureRecorder",
    "db_operation_error_context",
]
