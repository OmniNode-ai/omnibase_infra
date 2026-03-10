# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Metrics rollup upsert handler for NodeDeltaMetricsEffect.

Upserts a single rollup contribution into delta_metrics_by_model using
INSERT ON CONFLICT DO UPDATE semantics. Atomically increments the appropriate
counters (total_prs, merged_prs, reverted_prs, quarantine_prs, fix_prs)
and recalculates avg_gate_violations using a running average formula.

Related Tickets:
    - OMN-3142: NodeDeltaMetricsEffect implementation
    - Migration 040: delta_metrics_by_model table
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

    from omnibase_infra.nodes.node_delta_metrics_effect.models.model_payload_upsert_metrics import (
        ModelPayloadUpsertMetrics,
    )

logger = logging.getLogger(__name__)

# Upsert rollup -- ON CONFLICT (coding_model, subsystem, period_start, period_end)
# DO UPDATE with incremented counters.
# Uses a running average formula for avg_gate_violations:
#   new_avg = (old_avg * old_total + new_count) / (old_total + 1)
SQL_UPSERT_METRICS = """
INSERT INTO delta_metrics_by_model (
    coding_model, subsystem,
    total_prs, merged_prs, reverted_prs, quarantine_prs, fix_prs,
    avg_gate_violations,
    period_start, period_end
) VALUES (
    $1, $2,
    1,
    CASE WHEN $3 = 'merged' THEN 1 ELSE 0 END,
    CASE WHEN $3 = 'reverted' THEN 1 ELSE 0 END,
    CASE WHEN $4 = 'QUARANTINE' THEN 1 ELSE 0 END,
    CASE WHEN $5 THEN 1 ELSE 0 END,
    $6,
    $7, $8
)
ON CONFLICT (coding_model, subsystem, period_start, period_end) DO UPDATE SET
    total_prs = delta_metrics_by_model.total_prs + 1,
    merged_prs = delta_metrics_by_model.merged_prs
        + CASE WHEN $3 = 'merged' THEN 1 ELSE 0 END,
    reverted_prs = delta_metrics_by_model.reverted_prs
        + CASE WHEN $3 = 'reverted' THEN 1 ELSE 0 END,
    quarantine_prs = delta_metrics_by_model.quarantine_prs
        + CASE WHEN $4 = 'QUARANTINE' THEN 1 ELSE 0 END,
    fix_prs = delta_metrics_by_model.fix_prs
        + CASE WHEN $5 THEN 1 ELSE 0 END,
    avg_gate_violations = (
        COALESCE(delta_metrics_by_model.avg_gate_violations, 0)
            * delta_metrics_by_model.total_prs + $6
    ) / (delta_metrics_by_model.total_prs + 1),
    computed_at = NOW();
"""


class HandlerUpsertMetrics(MixinPostgresOpExecutor):
    """Metrics rollup upsert handler.

    Upserts a single bundle's contribution into delta_metrics_by_model.
    Atomically increments counters and recalculates the running average
    for gate violations. Safe to call multiple times -- each call adds
    one more contribution to the rollup.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerUpsertMetrics(pool)
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
        payload: ModelPayloadUpsertMetrics,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Upsert a metrics rollup contribution.

        Args:
            payload: Metrics upsert payload from bundle completion signal.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with success=True on successful upsert.
            Returns success=False only on infrastructure errors.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "coding_model": payload.coding_model,
                "subsystem": payload.subsystem,
                "outcome": payload.outcome,
                "period": f"{payload.period_start}/{payload.period_end}",
            },
            fn=lambda: self._execute_upsert(payload, correlation_id),
        )

    async def _execute_upsert(
        self,
        payload: ModelPayloadUpsertMetrics,
        correlation_id: UUID,
    ) -> None:
        """Execute the metrics rollup upsert.

        Args:
            payload: Metrics upsert payload.
            correlation_id: Correlation ID for logging.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_UPSERT_METRICS,
                payload.coding_model,
                payload.subsystem,
                payload.outcome,
                payload.gate_decision,
                payload.is_fix_pr,
                payload.gate_violation_count,
                payload.period_start,
                payload.period_end,
            )

        logger.debug(
            "Delta metrics upsert executed (ON CONFLICT DO UPDATE -- counters incremented)",
            extra={
                "coding_model": payload.coding_model,
                "subsystem": payload.subsystem,
                "outcome": payload.outcome,
                "gate_decision": payload.gate_decision,
                "is_fix_pr": payload.is_fix_pr,
                "period": f"{payload.period_start}/{payload.period_end}",
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["HandlerUpsertMetrics"]
