# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for updating contract heartbeat timestamp.

This handler encapsulates PostgreSQL-specific heartbeat update logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresHeartbeat is responsible for:
    - Executing heartbeat timestamp updates against PostgreSQL
    - Timing operation duration for observability
    - Tracking whether the target row was found
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

    This extraction supports the declarative node pattern where
    NodeContractPersistenceEffect delegates backend-specific operations
    to dedicated handlers.

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
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult
from omnibase_infra.utils import sanitize_error_message

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


class HandlerPostgresHeartbeat:
    """Handler for updating contract heartbeat timestamp.

    Encapsulates PostgreSQL-specific heartbeat update logic extracted from
    NodeContractPersistenceEffect for declarative node compliance. The handler
    provides a clean interface for executing timestamp updates with proper
    timing and error sanitization.

    The heartbeat operation updates the last_seen_at timestamp for an active
    contract, supporting contract lifecycle management by tracking node
    liveness.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> from unittest.mock import AsyncMock, MagicMock
        >>> pool = MagicMock()
        >>> pool.execute = AsyncMock(return_value="UPDATE 1")
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
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for executing heartbeat
                update operations against the contracts table.
        """
        self._pool = pool

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

            Error messages are sanitized using sanitize_error_message to
            prevent exposure of connection strings, credentials, or other
            sensitive information in logs and responses.
        """
        start_time = time.perf_counter()

        try:
            # Execute update - returns status string like "UPDATE 1" or "UPDATE 0"
            status = await self._pool.execute(
                _UPDATE_HEARTBEAT_SQL,
                payload.last_seen_at,
                payload.contract_id,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Parse affected row count from status (format: "UPDATE N")
            row_found = False
            if status and status.startswith("UPDATE "):
                try:
                    affected_rows = int(status.split()[1])
                    row_found = affected_rows > 0
                except (IndexError, ValueError):
                    pass  # Fallback to False if parsing fails

            # Log for observability (since ModelBackendResult doesn't support metadata)
            _logger.info(
                "Heartbeat update completed",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "node_name": payload.node_name,
                    "row_found": row_found,
                    "duration_ms": duration_ms,
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

            return ModelBackendResult(
                success=True,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during update - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.warning(
                "Heartbeat update timed out",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="HEARTBEAT_TIMEOUT_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.exception(
                "Heartbeat update authentication failed",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="HEARTBEAT_AUTH_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.warning(
                "Heartbeat update connection failed",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="HEARTBEAT_CONNECTION_ERROR",
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
                "Heartbeat update failed with unexpected error",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "duration_ms": duration_ms,
                    "error": sanitized_error,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="HEARTBEAT_UNKNOWN_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__: list[str] = ["HandlerPostgresHeartbeat"]
