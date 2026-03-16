# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Idempotent bundle insert handler for NodeDeltaBundleEffect.

Inserts a delta bundle record into delta_bundles using
INSERT ON CONFLICT (pr_ref, head_sha) DO NOTHING semantics. Safe to call
multiple times for the same (pr_ref, head_sha) pair -- duplicate calls are
silently ignored.

Parses PR labels for the ``stabilizes:<original_pr_ref>`` convention to
detect fix-PRs and populate is_fix_pr + stabilizes_pr_ref fields.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
    - Migration 039: delta_bundles table
"""

from __future__ import annotations

import json
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

    from omnibase_infra.nodes.node_delta_bundle_effect.models.model_payload_write_bundle import (
        ModelPayloadWriteBundle,
    )

logger = logging.getLogger(__name__)

# Prefix for the stabilization label convention.
_STABILIZES_PREFIX = "stabilizes:"

# Idempotent insert -- ON CONFLICT (pr_ref, head_sha) DO NOTHING.
SQL_INSERT_BUNDLE = """
INSERT INTO delta_bundles (
    bundle_id, pr_ref, head_sha, base_sha,
    coding_model, subsystem,
    gate_decision, gate_violations,
    is_fix_pr, stabilizes_pr_ref
) VALUES (
    $1, $2, $3, $4,
    $5, $6,
    $7, $8::jsonb,
    $9, $10
)
ON CONFLICT (pr_ref, head_sha) DO NOTHING;
"""


def parse_stabilizes_label(labels: list[str]) -> str | None:
    """Extract the stabilized PR ref from labels.

    Scans labels for the first matching ``stabilizes:<pr_ref>`` entry and
    returns the ``<pr_ref>`` portion. Returns None if no matching label is
    found.

    Args:
        labels: List of PR label strings.

    Returns:
        The stabilized PR ref string, or None if not found.

    Examples:
        >>> parse_stabilizes_label(["bug", "stabilizes:owner/repo#42"])
        'owner/repo#42'
        >>> parse_stabilizes_label(["enhancement", "docs"])
        >>> # Returns None
    """
    for label in labels:
        if label.startswith(_STABILIZES_PREFIX):
            value = label[len(_STABILIZES_PREFIX) :].strip()
            if value:
                return value
    return None


class HandlerWriteBundle(MixinPostgresOpExecutor):
    """Idempotent delta bundle insert handler.

    Inserts a delta bundle record from a merge-gate-decision event into the
    delta_bundles table. Uses ON CONFLICT DO NOTHING so repeated calls for
    the same (pr_ref, head_sha) pair are safe.

    Detects fix-PRs by scanning labels for ``stabilizes:<original_pr_ref>``.

    Attributes:
        _pool: asyncpg connection pool for database operations.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> handler = HandlerWriteBundle(pool)
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
        payload: ModelPayloadWriteBundle,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Insert a delta bundle idempotently into delta_bundles.

        Args:
            payload: Bundle write payload from merge-gate-decision event.
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
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "bundle_id": str(payload.bundle_id),
            },
            fn=lambda: self._execute_insert(payload, correlation_id),
        )

    async def _execute_insert(
        self,
        payload: ModelPayloadWriteBundle,
        correlation_id: UUID,
    ) -> None:
        """Execute the idempotent bundle insert.

        Parses labels for fix-PR detection before executing SQL.

        Args:
            payload: Bundle write payload.
            correlation_id: Correlation ID for logging.
        """
        stabilizes_ref = parse_stabilizes_label(payload.labels)
        is_fix_pr = stabilizes_ref is not None

        async with self._pool.acquire() as conn:
            await conn.execute(
                SQL_INSERT_BUNDLE,
                payload.bundle_id,
                payload.pr_ref,
                payload.head_sha,
                payload.base_sha,
                payload.coding_model,
                payload.subsystem,
                payload.gate_decision,
                json.dumps(payload.gate_violations),
                is_fix_pr,
                stabilizes_ref,
            )

        logger.debug(
            "Delta bundle insert executed (ON CONFLICT DO NOTHING -- duplicates silently skipped)",
            extra={
                "pr_ref": payload.pr_ref,
                "head_sha": payload.head_sha,
                "bundle_id": str(payload.bundle_id),
                "gate_decision": payload.gate_decision,
                "is_fix_pr": is_fix_pr,
                "stabilizes_pr_ref": stabilizes_ref,
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["HandlerWriteBundle", "parse_stabilizes_label"]
