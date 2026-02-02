# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared execution core for PostgreSQL operation handlers.

This mixin centralizes the mechanical aspects of PostgreSQL handler execution:
- Timing via time.perf_counter()
- Error classification and sanitization
- ModelBackendResult construction
- Structured logging with correlation IDs

By extracting this boilerplate into a reusable mixin, handlers are reduced from
~200 lines to ~30 lines, eliminating drift risk where error handling patterns
could diverge across handlers.

Architecture:
    Handlers inherit from MixinPostgresOpExecutor and call _execute_postgres_op()
    with their operation-specific logic wrapped in a callable. The mixin handles
    all timing, error classification, sanitization, and result construction.

Error Classification:
    - TimeoutError, InfraTimeoutError → POSTGRES_TIMEOUT_ERROR (retriable)
    - InfraAuthenticationError → POSTGRES_AUTH_ERROR (non-retriable)
    - InfraConnectionError → POSTGRES_CONNECTION_ERROR (retriable)
    - RepositoryExecutionError → op_error_code (handler-specified)
    - Exception → POSTGRES_UNKNOWN_ERROR (non-retriable)

Usage:
    ```python
    class HandlerPostgresHeartbeat(MixinPostgresOpExecutor):
        async def handle(self, payload, correlation_id) -> ModelBackendResult:
            return await self._execute_postgres_op(
                op_error_code=EnumPostgresErrorCode.HEARTBEAT_ERROR,
                correlation_id=correlation_id,
                log_context={"contract_id": payload.contract_id},
                fn=lambda: self._do_heartbeat(payload),
            )
    ```

Related:
    - EnumPostgresErrorCode: Error code enumeration with retriability metadata
    - MixinAsyncCircuitBreaker: Circuit breaker mixin (not integrated here)
    - OMN-1857: Extraction ticket for this mixin

Note on Circuit Breaker:
    Per OMN-1857 design decision 1A, the executor should manage circuit breaker
    internally. However, this initial implementation focuses on the core execution
    mechanics. Circuit breaker integration will be added as a follow-up once the
    basic pattern is validated.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    RepositoryExecutionError,
)
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MixinPostgresOpExecutor:
    """Shared execution core for PostgreSQL operation handlers.

    Centralizes timing, error handling, sanitization, and result construction
    for PostgreSQL operations. Handlers inherit this mixin and delegate to
    _execute_postgres_op() for consistent mechanical behavior.

    This mixin does NOT manage circuit breaker state - that responsibility
    remains with the handler or a separate MixinAsyncCircuitBreaker composition.

    Example:
        ```python
        class HandlerPostgresUpsert(MixinPostgresOpExecutor):
            def __init__(self, pool: asyncpg.Pool) -> None:
                self._pool = pool

            async def handle(
                self, payload: ModelPayloadUpsertContract, correlation_id: UUID
            ) -> ModelBackendResult:
                return await self._execute_postgres_op(
                    op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
                    correlation_id=correlation_id,
                    log_context={
                        "contract_id": payload.contract_id,
                        "node_name": payload.node_name,
                    },
                    fn=lambda: self._execute_upsert(payload),
                )
        ```

    See Also:
        - EnumPostgresErrorCode: Error codes with is_retriable property
        - sanitize_error_message: Error sanitization utility
        - ModelBackendResult: Structured result model
    """

    async def _execute_postgres_op(
        self,
        *,
        op_error_code: EnumPostgresErrorCode,
        correlation_id: UUID,
        log_context: dict[str, object],
        fn: Callable[[], Awaitable[T]],
    ) -> ModelBackendResult:
        """Execute a PostgreSQL operation with timing, error handling, and sanitization.

        This method wraps the actual database operation (fn) with:
        1. Timing measurement via time.perf_counter()
        2. Exception classification into appropriate error codes
        3. Error message sanitization to prevent credential exposure
        4. Structured logging with correlation ID and context
        5. ModelBackendResult construction

        Args:
            op_error_code: Operation-specific error code for non-infrastructure
                failures (e.g., UPSERT_ERROR, HEARTBEAT_ERROR). Used when the
                operation fails due to business logic or query issues rather
                than connection/auth problems.
            correlation_id: Request correlation ID for distributed tracing.
            log_context: Additional fields for structured logging (e.g.,
                contract_id, node_name). Included in all log messages.
            fn: Async callable that performs the actual database operation.
                Should return any value on success. The return value is not
                used - only success/failure matters.

        Returns:
            ModelBackendResult with:
                - success: True if fn() completed without exception
                - error: Sanitized error message (empty string on success)
                - error_code: Appropriate EnumPostgresErrorCode
                - duration_ms: Operation duration in milliseconds
                - backend_id: "postgres"
                - correlation_id: Passed through for tracing

        Error Classification:
            | Exception Type              | Error Code              | Retriable |
            |-----------------------------|-------------------------|-----------|
            | TimeoutError                | TIMEOUT_ERROR           | Yes       |
            | InfraTimeoutError           | TIMEOUT_ERROR           | Yes       |
            | InfraAuthenticationError    | AUTH_ERROR              | No        |
            | InfraConnectionError        | CONNECTION_ERROR        | Yes       |
            | RepositoryExecutionError    | op_error_code           | No        |
            | Exception                   | UNKNOWN_ERROR           | No        |

        Note:
            This method never raises exceptions. All errors are captured,
            sanitized, logged, and returned in the result model.
        """
        start_time = time.perf_counter()
        log_extra = {
            "correlation_id": str(correlation_id),
            **log_context,
        }

        try:
            # Execute the operation
            await fn()

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "PostgreSQL operation completed successfully",
                extra={**log_extra, "duration_ms": duration_ms},
            )

            return ModelBackendResult(
                success=True,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "PostgreSQL operation timed out",
                extra={
                    **log_extra,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.TIMEOUT_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.exception(
                "PostgreSQL authentication failed",
                extra={
                    **log_extra,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.AUTH_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "PostgreSQL connection failed",
                extra={
                    **log_extra,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.CONNECTION_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except RepositoryExecutionError as e:
            # Query/operation failure - use handler-provided error code
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            logger.warning(
                "PostgreSQL operation failed",
                extra={
                    **log_extra,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                    "error_code": op_error_code.value,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=op_error_code,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all for driver errors, encoding errors, pool errors
            # Unknown error - non-retriable, requires investigation
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_backend_error("postgres", e)
            logger.exception(
                "PostgreSQL operation failed with unexpected error",
                extra={
                    **log_extra,
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.UNKNOWN_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__ = ["MixinPostgresOpExecutor"]
