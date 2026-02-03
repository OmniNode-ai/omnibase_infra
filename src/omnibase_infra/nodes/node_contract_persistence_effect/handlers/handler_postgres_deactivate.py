# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL contract deactivation (soft-delete).

This handler encapsulates PostgreSQL-specific deactivation logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresDeactivate is responsible for:
    - Executing soft-delete operations against PostgreSQL
    - Returning structured ModelBackendResult

    Timing, error classification, and sanitization are delegated to
    MixinPostgresOpExecutor to eliminate boilerplate drift across handlers.

    The deactivation operation performs a soft delete by marking the contract
    record as inactive (is_active=FALSE) and setting deregistered_at timestamp,
    preserving historical data for auditing.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadDeactivateContract: Input payload model
    - ModelBackendResult: Structured result model for backend operations
    - MixinPostgresOpExecutor: Shared execution core for timing/error handling
    - OMN-1845: Implementation ticket
    - OMN-1857: Executor extraction ticket
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult

logger = logging.getLogger(__name__)

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


class HandlerPostgresDeactivate(MixinPostgresOpExecutor):
    """Handler for PostgreSQL contract deactivation (soft-delete).

    Encapsulates all PostgreSQL-specific deactivation logic extracted from
    NodeContractPersistenceEffect for declarative node compliance.

    Timing, error classification, and sanitization are handled by the
    MixinPostgresOpExecutor base class, reducing boilerplate and ensuring
    consistent error handling across all PostgreSQL handlers.

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
        - MixinPostgresOpExecutor: Shared execution mechanics
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

        Performs the deactivation operation against PostgreSQL.
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
            If the contract_id doesn't exist, success=True is still returned
            but with an appropriate message indicating no row was found.
            This follows the idempotency principle - deactivating a
            non-existent or already-deactivated contract is not an error.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.DEACTIVATE_ERROR,
            correlation_id=correlation_id,
            log_context={
                "contract_id": payload.contract_id,
            },
            fn=lambda: self._execute_deactivate(payload, correlation_id),
        )

    async def _execute_deactivate(
        self,
        payload: ModelPayloadDeactivateContract,
        correlation_id: UUID,
    ) -> None:
        """Execute the deactivation UPDATE query.

        Args:
            payload: Deactivation payload with contract_id and timestamp.
            correlation_id: Correlation ID for logging.

        Raises:
            Any exception from asyncpg (handled by MixinPostgresOpExecutor).
        """
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                SQL_DEACTIVATE_CONTRACT,
                payload.deactivated_at,
                payload.contract_id,
            )

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


__all__: list[str] = ["HandlerPostgresDeactivate"]
