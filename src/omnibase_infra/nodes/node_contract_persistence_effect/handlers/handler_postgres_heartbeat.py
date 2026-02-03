# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for updating contract heartbeat timestamp.

This handler encapsulates PostgreSQL-specific heartbeat update logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresHeartbeat is responsible for:
    - Executing heartbeat timestamp updates against PostgreSQL
    - Tracking whether the target row was found
    - Returning structured ModelBackendResult

    Timing, error classification, and sanitization are delegated to
    MixinPostgresOpExecutor to eliminate boilerplate drift across handlers.

Operation:
    Updates the last_seen_at timestamp for an active contract identified
    by contract_id. Only active contracts (is_active = TRUE) are updated.
    If the contract is not found or is inactive, the operation succeeds
    but row_found is logged as false.

SQL:
    UPDATE contracts
    SET last_seen_at = $1
    WHERE contract_id = $2 AND is_active = TRUE

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different payload instances. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadUpdateHeartbeat: Payload model defining heartbeat parameters
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

    from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_update_heartbeat import (
        ModelPayloadUpdateHeartbeat,
    )

_logger = logging.getLogger(__name__)

# SQL for updating heartbeat timestamp
_UPDATE_HEARTBEAT_SQL = """
UPDATE contracts
SET last_seen_at = $1
WHERE contract_id = $2 AND is_active = TRUE
"""


class HandlerPostgresHeartbeat(MixinPostgresOpExecutor):
    """Handler for updating contract heartbeat timestamp.

    Encapsulates PostgreSQL-specific heartbeat update logic extracted from
    NodeContractPersistenceEffect for declarative node compliance. The handler
    provides a clean interface for executing timestamp updates.

    Timing, error classification, and sanitization are handled by the
    MixinPostgresOpExecutor base class, reducing boilerplate and ensuring
    consistent error handling across all PostgreSQL handlers.

    The heartbeat operation updates the last_seen_at timestamp for an active
    contract, supporting contract lifecycle management by tracking node
    liveness.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> conn = MagicMock()
        >>> conn.execute = AsyncMock(return_value="UPDATE 1")
        >>> pool = MagicMock()
        >>> pool.acquire = MagicMock(return_value=AsyncContextManager(conn))
        >>> handler = HandlerPostgresHeartbeat(pool)
        >>> payload = MagicMock(
        ...     contract_id="my-node:1.0.0",
        ...     last_seen_at=datetime.now(),
        ... )
        >>> result = await handler.handle(payload, uuid4())
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadUpdateHeartbeat: Payload model for heartbeat parameters
        - MixinPostgresOpExecutor: Shared execution mechanics
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for executing heartbeat
                update operations against the contracts table.
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
        payload: ModelPayloadUpdateHeartbeat,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute heartbeat timestamp update for a contract.

        Updates the last_seen_at timestamp for the contract identified
        by contract_id, if the contract exists and is active.

        Args:
            payload: Heartbeat parameters containing:
                - contract_id: Derived natural key (node_name:major.minor.patch)
                - last_seen_at: New heartbeat timestamp
                - node_name: Contract node name (for logging)
                - source_node_id: Optional source node ID (for logging)
                - uptime_seconds: Optional node uptime (for logging)
                - sequence_number: Optional heartbeat sequence (for logging)
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if update completed (even if row not found)
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            The row_found status is logged for observability but not
            returned in the result model (ModelBackendResult does not support
            metadata). The operation is considered successful even if no row
            was found (the contract may have been deregistered), which allows
            heartbeat processing to continue without errors.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.HEARTBEAT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "contract_id": payload.contract_id,
                "node_name": payload.node_name,
                "source_node_id": payload.source_node_id,
                "uptime_seconds": payload.uptime_seconds,
                "sequence_number": payload.sequence_number,
            },
            fn=lambda: self._execute_heartbeat(payload, correlation_id),
        )

    async def _execute_heartbeat(
        self,
        payload: ModelPayloadUpdateHeartbeat,
        correlation_id: UUID,
    ) -> None:
        """Execute the heartbeat UPDATE query.

        This method contains the handler-specific business logic:
        - Execute the UPDATE query
        - Parse the affected row count
        - Log success/warning based on row_found

        Args:
            payload: Heartbeat payload with contract_id and timestamp.
            correlation_id: Correlation ID for logging.

        Raises:
            Any exception from asyncpg (handled by MixinPostgresOpExecutor).
        """
        async with self._pool.acquire() as conn:
            # Execute update - returns status string like "UPDATE 1" or "UPDATE 0"
            status = await conn.execute(
                _UPDATE_HEARTBEAT_SQL,
                payload.last_seen_at,
                payload.contract_id,
            )

        # Parse affected row count from status (format: "UPDATE N")
        row_found = False
        if status and status.startswith("UPDATE "):
            try:
                affected_rows = int(status.split()[1])
                row_found = affected_rows > 0
            except (IndexError, ValueError):
                pass  # Fallback to False if parsing fails

        # Log for observability
        _logger.info(
            "Heartbeat update completed",
            extra={
                "correlation_id": str(correlation_id),
                "contract_id": payload.contract_id,
                "node_name": payload.node_name,
                "row_found": row_found,
                "source_node_id": payload.source_node_id,
                "uptime_seconds": payload.uptime_seconds,
                "sequence_number": payload.sequence_number,
            },
        )

        # Log warning if row not found (contract may be deregistered)
        if not row_found:
            _logger.warning(
                "Heartbeat update found no matching active contract",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "node_name": payload.node_name,
                },
            )


__all__: list[str] = ["HandlerPostgresHeartbeat"]
