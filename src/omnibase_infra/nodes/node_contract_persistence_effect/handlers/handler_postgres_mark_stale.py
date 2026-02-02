# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for batch marking stale contracts as inactive.

This handler encapsulates PostgreSQL-specific staleness marking logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresMarkStale is responsible for:
    - Executing batch update operations against PostgreSQL
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult with affected row count

    This extraction supports the declarative node pattern where
    NodeContractPersistenceEffect delegates backend-specific operations
    to dedicated handlers.

Operation:
    Marks all active contracts with last_seen_at before the stale_cutoff
    timestamp as inactive. This is a batch operation that may affect
    multiple rows in a single execution.

SQL:
    UPDATE contracts
    SET is_active = FALSE, deregistered_at = $1
    WHERE is_active = TRUE AND last_seen_at < $2

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different payload instances. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadMarkStale: Payload model defining staleness parameters
    - ModelBackendResult: Structured result model for backend operations
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_mark_stale import (
        ModelPayloadMarkStale,
    )

_logger = logging.getLogger(__name__)

# SQL for batch marking stale contracts
_MARK_STALE_SQL = """
UPDATE contracts
SET is_active = FALSE, deregistered_at = $1
WHERE is_active = TRUE AND last_seen_at < $2
"""


class HandlerPostgresMarkStale:
    """Handler for batch marking stale contracts as inactive.

    Encapsulates PostgreSQL-specific batch staleness marking logic extracted
    from NodeContractPersistenceEffect for declarative node compliance. The
    handler provides a clean interface for executing batch updates with proper
    timing and error sanitization.

    The staleness operation marks contracts as inactive if their last_seen_at
    timestamp is older than the specified stale_cutoff, supporting contract
    lifecycle management through automatic deregistration of stale nodes.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> pool = MagicMock()
        >>> pool.execute = AsyncMock(return_value="UPDATE 5")
        >>> handler = HandlerPostgresMarkStale(pool)
        >>> payload = MagicMock(stale_cutoff=datetime.now(), checked_at=datetime.now())
        >>> result = await handler.handle(payload, uuid4())
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadMarkStale: Payload model for staleness parameters
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for executing batch update
                operations against the contracts table.
        """
        self._pool = pool

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role of this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification of this handler."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        payload: ModelPayloadMarkStale,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute batch staleness update on contracts.

        Marks all active contracts with last_seen_at before stale_cutoff
        as inactive. This is a batch operation that may affect multiple
        rows in a single execution.

        Args:
            payload: Staleness parameters containing:
                - stale_cutoff: Contracts older than this are marked stale
                - checked_at: Timestamp used for deregistered_at value
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if update completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            The number of affected rows is logged for observability but not
            returned in the result model (ModelBackendResult does not support
            metadata). Callers requiring the count should query the database
            separately or rely on log aggregation.

            Error messages are sanitized using sanitize_error_message to
            prevent exposure of connection strings, credentials, or other
            sensitive information in logs and responses.
        """
        start_time = time.perf_counter()

        try:
            # Execute batch update - returns status string like "UPDATE 5"
            status = await self._pool.execute(
                _MARK_STALE_SQL,
                payload.checked_at,
                payload.stale_cutoff,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Parse affected row count from status (format: "UPDATE N")
            affected_rows = 0
            if status and status.startswith("UPDATE "):
                try:
                    affected_rows = int(status.split()[1])
                except (IndexError, ValueError):
                    pass  # Fallback to 0 if parsing fails

            # Log for observability (since ModelBackendResult doesn't support metadata)
            _logger.info(
                "Mark stale operation completed",
                extra={
                    "correlation_id": str(correlation_id),
                    "affected_rows": affected_rows,
                    "stale_cutoff": payload.stale_cutoff.isoformat(),
                    "duration_ms": duration_ms,
                },
            )

            return ModelBackendResult(
                success=True,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during batch update - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.warning(
                "Mark stale operation timed out",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="MARK_STALE_TIMEOUT_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.exception(
                "Mark stale operation authentication failed",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="MARK_STALE_AUTH_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.warning(
                "Mark stale operation connection failed",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="MARK_STALE_CONNECTION_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all - database adapter may raise unexpected exceptions
            # beyond typed infrastructure errors (e.g., driver errors, encoding errors,
            # connection pool errors). Required to sanitize errors and prevent credential exposure.
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.exception(
                "Mark stale operation failed with unexpected error",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="MARK_STALE_UNKNOWN_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__: list[str] = ["HandlerPostgresMarkStale"]
