# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Outcome update handler for NodeDeltaBundleEffect.

Updates a delta_bundles row with the final PR outcome (merged, reverted,
closed) and sets merged_at and bundle_completed_at timestamps.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
    - Migration 039: delta_bundles table
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

    from omnibase_infra.nodes.node_delta_bundle_effect.models.model_payload_update_outcome import (
        ModelPayloadUpdateOutcome,
    )

logger = logging.getLogger(__name__)

# Update outcome -- sets outcome, merged_at, bundle_completed_at, updated_at.
# Only updates rows where outcome is NULL to prevent double-updates.
SQL_UPDATE_OUTCOME = """
UPDATE delta_bundles
SET outcome = $1,
    merged_at = $2,
    bundle_completed_at = NOW(),
    updated_at = NOW()
WHERE pr_ref = $3
  AND head_sha = $4
  AND outcome IS NULL;
"""


class HandlerUpdateOutcome(MixinPostgresOpExecutor):
    """PR outcome update handler.

    Updates a delta_bundles row with the final outcome. Only modifies rows
    where outcome IS NULL (idempotent -- second call for same bundle is a
    no-op).

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerUpdateOutcome(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialise handler with asyncpg connection pool.

        Args:
            pool: asyncpg connection pool. Should be pre-configured and ready.
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
        payload: ModelPayloadUpdateOutcome,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Update a delta bundle with the PR outcome.

        Args:
            payload: Outcome update payload from pr-outcome event.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with success=True on successful update or
            idempotent skip (outcome already set). Returns success=False
            only on infrastructure errors.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "outcome": payload.outcome,
            },
            fn=lambda: self._execute_update(payload, correlation_id),
        )

    async def _execute_update(
        self,
        payload: ModelPayloadUpdateOutcome,
        correlation_id: UUID,
    ) -> None:
        """Execute the outcome update.

        Args:
            payload: Outcome update payload.
            correlation_id: Correlation ID for logging.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPDATE_OUTCOME,
                payload.outcome,
                payload.merged_at,
                payload.pr_ref,
                payload.head_sha,
            )

        logger.debug(
            "Delta bundle outcome updated (WHERE outcome IS NULL -- already-set bundles skipped)",
            extra={
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "outcome": payload.outcome,
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["HandlerUpdateOutcome"]
