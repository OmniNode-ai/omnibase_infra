# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL topic reference cleanup.

This handler encapsulates PostgreSQL-specific topic cleanup logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresCleanupTopics is responsible for:
    - Removing contract_id from topics.contract_ids JSONB arrays
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

    This handler removes a contract_id from the contract_ids JSONB array in
    all topics that reference it. Topic rows are NOT deleted even when the
    contract_ids array becomes empty.

Topic Orphan Handling (OMN-1709):
    When all contracts are removed from a topic, the topic row remains with
    empty contract_ids array. This is intentional:
    - Preserves topic routing history for auditing and debugging
    - Allows topic reactivation if a new contract references the same topic
    - Avoids complex cascading deletes during high-volume deregistration

    To clean up orphaned topics, run:
        DELETE FROM topics WHERE contract_ids = '[]' AND is_active = FALSE;

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadCleanupTopicReferences: Input payload model
    - ModelBackendResult: Structured result model for backend operations
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
    - OMN-1709: Topic orphan handling documentation
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins import MixinPostgresErrorResponse, PostgresErrorContext
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult

if TYPE_CHECKING:
    from omnibase_infra.nodes.contract_registry_reducer.models import (
        ModelPayloadCleanupTopicReferences,
    )

# SQL for removing contract_id from all topic contract_ids arrays
# Uses parameterized query: $1 = contract_id (as text)
#
# JSONB operators:
#   - `contract_ids ? $1` checks if string exists as element in JSONB array
#   - `contract_ids - $1` removes the string element from JSONB array
#
# Note: updated_at is handled by the trigger (trigger_topics_updated_at)
# but we update it explicitly here for clarity and for cases where
# the trigger might not be installed.
SQL_CLEANUP_TOPIC_REFERENCES = """
UPDATE topics
SET contract_ids = contract_ids - $1,
    updated_at = NOW()
WHERE contract_ids ? $1
"""


class HandlerPostgresCleanupTopics(MixinPostgresErrorResponse):
    """Handler for removing contract references from topics.

    Encapsulates all PostgreSQL-specific topic cleanup logic extracted from
    NodeContractPersistenceEffect for declarative node compliance. The handler
    provides a clean interface for removing contract_id from all topic
    contract_ids arrays with proper timing and error sanitization.

    Important: Topic rows are NOT deleted even when contract_ids becomes empty.
    This is intentional per OMN-1709 - orphaned topics are preserved for
    auditing and can be cleaned up separately if needed.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> handler = HandlerPostgresCleanupTopics(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadCleanupTopicReferences: Input payload model
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
        payload: ModelPayloadCleanupTopicReferences,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Remove contract_id from all topic contract_ids arrays.

        Performs the cleanup operation against PostgreSQL with:
        - Operation timing via time.perf_counter()
        - Parameterized query execution
        - Error sanitization for security
        - Structured result construction

        The cleanup removes the contract_id from all topics.contract_ids
        JSONB arrays. Topic rows are preserved even if contract_ids becomes
        empty (per OMN-1709 topic orphan handling).

        Args:
            payload: Cleanup payload containing contract_id to remove and
                cleaned_at timestamp.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if cleanup completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            This handler never raises exceptions. All errors are caught,
            sanitized, and returned in ModelBackendResult.

            If no topics contain the contract_id, success=True is still
            returned. This follows the idempotency principle - cleaning up
            references for a contract that has no topic associations is
            not an error.
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                # Execute the update and get the number of affected rows
                result = await conn.execute(
                    SQL_CLEANUP_TOPIC_REFERENCES,
                    payload.contract_id,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Parse the result to get affected row count
            # asyncpg execute returns a string like "UPDATE N"
            topics_updated = 0
            if result and result.startswith("UPDATE "):
                try:
                    topics_updated = int(result.split(" ")[1])
                except (ValueError, IndexError):
                    pass  # Keep default of 0 if parsing fails

            # Log for observability
            logger.info(
                "Topic cleanup operation completed",
                extra={
                    "correlation_id": str(correlation_id),
                    "contract_id": payload.contract_id,
                    "topics_updated": topics_updated,
                    "duration_ms": duration_ms,
                },
            )

            return ModelBackendResult(
                success=True,
                error=None,
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=correlation_id,
            )

        except Exception as e:
            ctx = PostgresErrorContext(
                exception=e,
                operation="cleanup_topics",
                correlation_id=correlation_id,
                start_time=start_time,
                log_context={"contract_id": payload.contract_id},
                operation_error_code=EnumPostgresErrorCode.CLEANUP_ERROR,
            )
            return self._build_error_response(ctx)


__all__: list[str] = ["HandlerPostgresCleanupTopics"]
