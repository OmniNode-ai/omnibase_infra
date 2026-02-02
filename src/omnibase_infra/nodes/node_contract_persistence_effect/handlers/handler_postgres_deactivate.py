# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL contract deactivation (soft-delete).

This handler encapsulates PostgreSQL-specific deactivation logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresDeactivate is responsible for:
    - Executing soft-delete operations against PostgreSQL
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

    The deactivation operation performs a soft delete by marking the contract
    record as inactive (is_active=FALSE) and setting deregistered_at timestamp,
    preserving historical data for auditing.

    This extraction supports the declarative node pattern where
    NodeContractPersistenceEffect delegates backend-specific operations to
    dedicated handlers.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadDeactivateContract: Input payload model
    - ModelBackendResult: Structured result model for backend operations
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)

logger = logging.getLogger(__name__)

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.contract_registry_reducer.models import (
        ModelPayloadDeactivateContract,
    )

# SQL for soft-deleting a contract by marking it inactive
# Uses parameterized query: $1 = deregistered_at, $2 = contract_id
# RETURNING contract_id allows us to check if the row existed
SQL_DEACTIVATE_CONTRACT = """
UPDATE contracts
SET is_active = FALSE, deregistered_at = $1
WHERE contract_id = $2
RETURNING contract_id
"""


class HandlerPostgresDeactivate:
    """Handler for PostgreSQL contract deactivation (soft-delete).

    Encapsulates all PostgreSQL-specific deactivation logic extracted from
    NodeContractPersistenceEffect for declarative node compliance. The handler
    provides a clean interface for executing soft-delete operations with proper
    timing and error sanitization.

    The deactivation operation marks a contract as inactive (soft delete)
    rather than performing a hard delete, preserving audit trails and enabling
    potential reactivation.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> handler = HandlerPostgresDeactivate(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadDeactivateContract: Input payload model
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for executing database operations.
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
        payload: ModelPayloadDeactivateContract,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute PostgreSQL contract deactivation (soft-delete).

        Performs the deactivation operation against PostgreSQL with:
        - Operation timing via time.perf_counter()
        - Parameterized query execution
        - Error sanitization for security
        - Structured result construction

        The deactivation marks the contract record as inactive without
        deleting the underlying data, supporting audit requirements and
        potential reactivation scenarios.

        Args:
            payload: Deactivation payload containing contract_id and
                deactivated_at timestamp.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if deactivation completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            This handler never raises exceptions. All errors are caught,
            sanitized, and returned in ModelBackendResult.

            If the contract_id doesn't exist, success=True is still returned
            but with an appropriate message indicating no row was found.
            This follows the idempotency principle - deactivating a
            non-existent or already-deactivated contract is not an error.
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    SQL_DEACTIVATE_CONTRACT,
                    payload.deactivated_at,
                    payload.contract_id,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # result will be the contract_id if row was updated, None otherwise
            row_found = result is not None

            # Log the not-found case for observability
            if not row_found:
                logger.info(
                    "Contract not found during deactivation (idempotent no-op)",
                    extra={
                        "contract_id": payload.contract_id,
                        "correlation_id": str(correlation_id),
                    },
                )

            return ModelBackendResult(
                success=True,  # Operation succeeded (idempotent - no-op if not found)
                error=None,  # No error - operation was successful
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during deactivation - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.TIMEOUT_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.AUTH_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.CONNECTION_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except (
            Exception
        ) as e:  # ONEX: catch-all - database adapter may raise unexpected exceptions
            # beyond typed infrastructure errors (e.g., asyncpg errors, encoding errors,
            # connection pool errors). Required to sanitize errors and prevent credential exposure.
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code=EnumPostgresErrorCode.UNKNOWN_ERROR,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )


__all__: list[str] = ["HandlerPostgresDeactivate"]
