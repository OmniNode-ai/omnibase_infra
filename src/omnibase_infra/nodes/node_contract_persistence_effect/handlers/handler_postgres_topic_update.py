# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for PostgreSQL topic routing table updates.

This handler encapsulates PostgreSQL-specific persistence logic for the
NodeContractPersistenceEffect node, following the declarative node pattern where
handlers are extracted for testability and separation of concerns.

Architecture:
    HandlerPostgresTopicUpdate is responsible for:
    - Normalizing topic suffixes (stripping environment prefixes)
    - Executing upsert operations against the PostgreSQL topics table
    - Managing JSONB array of contract_ids for topic-to-contract mapping
    - Timing operation duration for observability
    - Sanitizing error messages before inclusion in results
    - Returning structured ModelBackendResult

Topic Normalization:
    Topics in contracts may include environment placeholders (e.g., "{env}.topic.name")
    or actual environment prefixes (e.g., "dev.topic.name"). Before storage, these
    prefixes are stripped to store only the topic suffix. This enables:
    - Environment-agnostic topic routing queries
    - Consistent topic identity across environments
    - Simplified topic discovery and management

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different payload instances. Thread-safety depends on the
    underlying asyncpg connection pool implementation.

SQL Security:
    All SQL queries use parameterized queries with positional placeholders
    ($1, $2, etc.) to prevent SQL injection attacks. The asyncpg library
    handles proper escaping and type conversion for all parameters.

Related:
    - NodeContractPersistenceEffect: Parent effect node that coordinates handlers
    - ModelPayloadUpdateTopic: Input payload model
    - ModelBackendResult: Structured result model for backend operations
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket (source of intents)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins import MixinPostgresErrorResponse, PostgresErrorContext
from omnibase_infra.nodes.effects.models import ModelBackendResult

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.contract_registry_reducer.models import (
        ModelPayloadUpdateTopic,
    )

logger = logging.getLogger(__name__)

# Known environment prefixes to strip from topic suffixes before storage.
# The placeholder prefix is checked first, then actual environment prefixes.
_ENVIRONMENT_PREFIXES: tuple[str, ...] = (
    "{env}.",  # Placeholder prefix (most common)
    "dev.",
    "prod.",
    "staging.",
    "local.",
    "test.",
)

# SQL statement for topic upsert with JSONB array merge.
# Uses ON CONFLICT to handle existing topic+direction combinations.
# The JSONB containment operator (?) checks if contract_id already exists in array.
# If not present, appends to array; otherwise keeps existing array unchanged.
SQL_UPSERT_TOPIC = """
INSERT INTO topics (topic_suffix, direction, contract_ids, first_seen_at, last_seen_at, is_active)
VALUES ($1, $2, jsonb_build_array($3), $4, $5, TRUE)
ON CONFLICT (topic_suffix, direction) DO UPDATE SET
    contract_ids = CASE
        WHEN NOT topics.contract_ids ? $3
        THEN topics.contract_ids || to_jsonb($3::text)
        ELSE topics.contract_ids
    END,
    last_seen_at = EXCLUDED.last_seen_at,
    is_active = TRUE,
    updated_at = NOW()
RETURNING topic_suffix, direction, contract_ids;
"""


def normalize_topic_for_storage(topic: str) -> str:
    """Strip environment placeholder/prefix from topic before storage.

    Topics in contracts may include environment placeholders like "{env}." or
    actual environment prefixes like "dev.", "prod.", etc. This function
    normalizes topics by stripping these prefixes to store only the topic suffix.

    The normalization enables environment-agnostic topic routing queries and
    consistent topic identity across different deployment environments.

    Args:
        topic: The topic string to normalize, potentially with an environment
            prefix (e.g., "{env}.onex.evt.platform.contract-registered.v1" or
            "dev.onex.evt.platform.contract-registered.v1").

    Returns:
        The normalized topic suffix without environment prefix
        (e.g., "onex.evt.platform.contract-registered.v1").

    Examples:
        >>> normalize_topic_for_storage("{env}.onex.evt.platform.contract-registered.v1")
        'onex.evt.platform.contract-registered.v1'

        >>> normalize_topic_for_storage("dev.onex.evt.platform.contract-registered.v1")
        'onex.evt.platform.contract-registered.v1'

        >>> normalize_topic_for_storage("onex.evt.platform.contract-registered.v1")
        'onex.evt.platform.contract-registered.v1'
    """
    for prefix in _ENVIRONMENT_PREFIXES:
        if topic.startswith(prefix):
            return topic[len(prefix) :]
    return topic


class HandlerPostgresTopicUpdate(MixinPostgresErrorResponse):
    """Handler for PostgreSQL topic routing table updates.

    Encapsulates all PostgreSQL-specific persistence logic for topic
    routing updates. The handler provides a clean interface for executing
    upsert operations with proper topic normalization, timing, and error
    sanitization.

    This handler never raises exceptions - all errors are captured and
    returned in the ModelBackendResult with appropriate error codes.

    Topic Contract Mapping:
        The topics table stores a JSONB array of contract_ids that reference
        each topic. This handler uses JSONB operations to safely add contracts
        to the array without duplicates:
        - If contract_id not in array: append to array
        - If contract_id already in array: keep array unchanged

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerPostgresTopicUpdate(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True

    See Also:
        - NodeContractPersistenceEffect: Parent node that uses this handler
        - ModelPayloadUpdateTopic: Input payload model
        - contract.yaml: Handler routing configuration
        - normalize_topic_for_storage: Topic normalization function
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool for database operations.
                The pool should be pre-configured and ready for use.
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
        payload: ModelPayloadUpdateTopic,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute PostgreSQL topic routing table update.

        Performs the upsert operation against the topics table with:
        - Topic suffix normalization (strip environment prefix)
        - JSONB array merge for contract_ids
        - Operation timing via time.perf_counter()
        - Parameterized SQL for injection prevention
        - Error sanitization for security
        - Structured result construction

        Args:
            payload: Update topic payload containing topic_suffix, direction,
                contract_id, and last_seen_at timestamp.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with:
                - success: True if upsert completed successfully
                - error: Sanitized error message if failed
                - error_code: Error code for programmatic handling
                - duration_ms: Operation duration in milliseconds
                - backend_id: Set to "postgres"
                - correlation_id: Passed through for tracing

        Note:
            This method never raises exceptions. All errors are captured
            and returned in the result model with appropriate error codes.

            Error messages are sanitized using sanitize_error_message to
            prevent exposure of connection strings, credentials, or other
            sensitive information in logs and responses.

            Topic suffixes are normalized before storage - environment
            prefixes like "{env}." or "dev." are stripped.
        """
        start_time = time.perf_counter()

        # Normalize topic suffix by stripping environment prefix
        normalized_topic = normalize_topic_for_storage(payload.topic_suffix)

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(
                    SQL_UPSERT_TOPIC,
                    normalized_topic,
                    payload.direction,
                    payload.contract_id,
                    payload.last_seen_at,  # first_seen_at (on insert)
                    payload.last_seen_at,  # last_seen_at
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result is not None:
                logger.info(
                    "Topic update completed",
                    extra={
                        "topic_suffix": normalized_topic,
                        "original_topic": payload.topic_suffix,
                        "direction": payload.direction,
                        "contract_id": payload.contract_id,
                        "duration_ms": duration_ms,
                        "correlation_id": str(correlation_id),
                    },
                )
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )
            else:
                # RETURNING clause should always return a row on success
                # If None, something unexpected happened
                logger.warning(
                    "Topic update returned no result",
                    extra={
                        "topic_suffix": normalized_topic,
                        "direction": payload.direction,
                        "duration_ms": duration_ms,
                        "correlation_id": str(correlation_id),
                    },
                )
                return ModelBackendResult(
                    success=False,
                    error="postgres operation failed: no result returned",
                    error_code=EnumPostgresErrorCode.TOPIC_UPDATE_ERROR,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )

        except Exception as e:
            ctx = PostgresErrorContext(
                exception=e,
                operation="topic_update",
                correlation_id=correlation_id,
                start_time=start_time,
                log_context={
                    "topic_suffix": normalized_topic,
                    "direction": payload.direction,
                },
                operation_error_code=EnumPostgresErrorCode.TOPIC_UPDATE_ERROR,
            )
            return self._build_error_response(ctx)


__all__: list[str] = ["HandlerPostgresTopicUpdate", "normalize_topic_for_storage"]
