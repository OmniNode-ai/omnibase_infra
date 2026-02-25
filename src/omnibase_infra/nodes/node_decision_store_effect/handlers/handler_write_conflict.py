# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotent conflict-pair insert handler for NodeDecisionStoreEffect.

Inserts a single conflict pair into decision_conflicts using
INSERT ON CONFLICT DO NOTHING semantics. Safe to call multiple times
for the same pair — duplicate calls are silently ignored.

The handler normalises the (decision_min_id, decision_max_id) pair ordering
to satisfy the DB constraint chk_conflict_pair_order (min < max).

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations (decision_conflicts table)
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

    from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_conflict import (
        ModelPayloadWriteConflict,
    )

logger = logging.getLogger(__name__)

# Idempotent insert — ON CONFLICT (decision_min_id, decision_max_id) DO NOTHING.
# The UNIQUE constraint uk_conflict_pair guarantees no duplicates.
SQL_INSERT_CONFLICT = """
INSERT INTO decision_conflicts (
    conflict_id, decision_min_id, decision_max_id,
    structural_confidence, final_severity, status, detected_at
) VALUES (
    gen_random_uuid(), $1, $2, $3, $4, 'OPEN', NOW()
)
ON CONFLICT (decision_min_id, decision_max_id) DO NOTHING;
"""


def _ordered_pair(a: UUID, b: UUID) -> tuple[UUID, UUID]:
    """Return (min_id, max_id) pair satisfying chk_conflict_pair_order."""
    return (a, b) if a < b else (b, a)


class HandlerWriteConflict(MixinPostgresOpExecutor):
    """Idempotent conflict-pair insert handler.

    Inserts a (decision_min_id, decision_max_id) pair into decision_conflicts.
    Uses ON CONFLICT DO NOTHING so repeated calls for the same pair are safe.

    Normalises the UUID pair order before insert to satisfy the DB constraint
    chk_conflict_pair_order (min < max).

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerWriteConflict(pool)
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
        payload: ModelPayloadWriteConflict,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Insert a conflict pair idempotently into decision_conflicts.

        Args:
            payload: Conflict write payload containing the pair and scores.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with success=True on successful insert or
            idempotent skip (ON CONFLICT DO NOTHING). Returns success=False
            only on infrastructure errors.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "decision_min_id": str(payload.decision_min_id),
                "decision_max_id": str(payload.decision_max_id),
            },
            fn=lambda: self._execute_insert(payload, correlation_id),
        )

    async def _execute_insert(
        self,
        payload: ModelPayloadWriteConflict,
        correlation_id: UUID,
    ) -> None:
        """Execute the idempotent conflict-pair insert.

        Normalises pair order before executing SQL to guarantee min < max.

        Args:
            payload: Conflict write payload.
            correlation_id: Correlation ID for logging.
        """
        min_id, max_id = _ordered_pair(payload.decision_min_id, payload.decision_max_id)

        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_INSERT_CONFLICT,
                min_id,
                max_id,
                payload.structural_confidence,
                payload.final_severity,
            )

        logger.debug(
            "Conflict pair insert executed (ON CONFLICT DO NOTHING — duplicates silently skipped)",
            extra={
                "decision_min_id": str(min_id),
                "decision_max_id": str(max_id),
                "structural_confidence": payload.structural_confidence,
                "final_severity": payload.final_severity,
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["HandlerWriteConflict"]
