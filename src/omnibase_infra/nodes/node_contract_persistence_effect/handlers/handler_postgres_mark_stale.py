# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for batch marking stale contracts as inactive.

This handler encapsulates PostgreSQL-specific staleness marking logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresMarkStale is responsible for:
    - Executing batch update operations against PostgreSQL
    - Returning structured ModelBackendResult with affected row count

    Timing, error classification, and sanitization are delegated to
    MixinPostgresOpExecutor to eliminate boilerplate drift across handlers.

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
    - MixinPostgresOpExecutor: Shared execution core for timing/error handling
    - OMN-1845: Implementation ticket
    - OMN-1857: Executor extraction ticket
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult

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


class HandlerPostgresMarkStale(MixinPostgresOpExecutor):
    """Handler for batch marking stale contracts as inactive.

    Encapsulates PostgreSQL-specific batch staleness marking logic extracted
    from NodeContractPersistenceEffect for declarative node compliance.

    Timing, error classification, and sanitization are handled by the
    MixinPostgresOpExecutor base class, reducing boilerplate and ensuring
    consistent error handling across all PostgreSQL handlers.

    The staleness operation marks contracts as inactive if their last_seen_at
    timestamp is older than the specified stale_cutoff, supporting contract
    lifecycle management through automatic deregistration of stale nodes.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> conn = MagicMock()
        >>> conn.execute = AsyncMock(return_value="UPDATE 5")
        >>> pool = MagicMock()
        >>> pool.acquire = MagicMock(return_value=AsyncContextManager(conn))
        >>> handler = HandlerPostgresMarkStale(pool)
        >>> payload = MagicMock(stale_cutoff=datetime.now(), checked_at=datetime.now())
        >>> result = await handler.handle(payload, uuid4())
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadMarkStale: Payload model for staleness parameters
        - MixinPostgresOpExecutor: Shared execution mechanics
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
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.MARK_STALE_ERROR,
            correlation_id=correlation_id,
            log_context={
                "stale_cutoff": payload.stale_cutoff.isoformat(),
            },
            fn=lambda: self._execute_mark_stale(payload, correlation_id),
        )

    async def _execute_mark_stale(
        self,
        payload: ModelPayloadMarkStale,
        correlation_id: UUID,
    ) -> None:
        """Execute the batch staleness UPDATE query.

        Args:
            payload: Mark stale payload with timestamps.
            correlation_id: Correlation ID for logging.

        Raises:
            Any exception from asyncpg (handled by MixinPostgresOpExecutor).
        """
        async with self._pool.acquire() as conn:
            # Execute batch update - returns status string like "UPDATE 5"
            status = await conn.execute(
                _MARK_STALE_SQL,
                payload.checked_at,
                payload.stale_cutoff,
            )

        # Parse affected row count from status (format: "UPDATE N")
        affected_rows = 0
        if status and status.startswith("UPDATE "):
            try:
                affected_rows = int(status.split()[1])
            except (IndexError, ValueError):
                pass  # Fallback to 0 if parsing fails

        # Log for observability
        _logger.info(
            "Mark stale operation completed",
            extra={
                "correlation_id": str(correlation_id),
                "affected_rows": affected_rows,
                "stale_cutoff": payload.stale_cutoff.isoformat(),
            },
        )


__all__: list[str] = ["HandlerPostgresMarkStale"]
