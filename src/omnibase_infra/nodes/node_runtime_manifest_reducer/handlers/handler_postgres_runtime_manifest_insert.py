# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for PostgreSQL runtime manifest INSERT (OMN-11197).

Append-only: this handler only INSERTs. It never UPDATEs existing rows.
The unique index idx_runtime_manifests_dedup on
(runtime_profile, topology_hash, started_at) silently deduplicates repeated
events from the same process boot via ON CONFLICT DO NOTHING.

SQL Security:
    All queries use positional placeholders ($1, $2, …) — no string interpolation.
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

    from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
        ModelPayloadInsertRuntimeManifest,
    )

logger = logging.getLogger(__name__)

SQL_INSERT_RUNTIME_MANIFEST = """
INSERT INTO runtime_manifests (
    runtime_profile,
    contract_hash,
    topology_hash,
    manifest_hash,
    contracts,
    owned_command_topics,
    subscribed_event_topics,
    handlers,
    skipped_contracts,
    failed_contracts,
    ownership_violations,
    image_digest,
    started_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
ON CONFLICT (runtime_profile, topology_hash, started_at) DO NOTHING
RETURNING id;
"""


class HandlerPostgresRuntimeManifestInsert(MixinPostgresOpExecutor):
    """Append-only INSERT handler for the runtime_manifests projection table.

    Receives a ModelPayloadInsertRuntimeManifest and performs a single INSERT.
    Duplicate startup events (same runtime_profile + topology_hash + started_at)
    are silently ignored via ON CONFLICT DO NOTHING.

    Attributes:
        _pool: asyncpg connection pool.

    Example:
        >>> handler = HandlerPostgresRuntimeManifestInsert(pool)
        >>> result = await handler.handle(payload, correlation_id)
        >>> result.success
        True
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        payload: ModelPayloadInsertRuntimeManifest,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """INSERT a runtime manifest row, ignoring exact duplicate startups.

        Args:
            payload: Manifest payload with all projection fields.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelBackendResult indicating success or failure.
        """
        return await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context={
                "runtime_profile": payload.runtime_profile,
                "manifest_hash": payload.manifest_hash,
            },
            fn=lambda: self._execute_insert(payload, correlation_id),
        )

    async def _execute_insert(
        self,
        payload: ModelPayloadInsertRuntimeManifest,
        correlation_id: UUID,
    ) -> None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                SQL_INSERT_RUNTIME_MANIFEST,
                payload.runtime_profile,
                payload.contract_hash,
                payload.topology_hash,
                payload.manifest_hash,
                json.dumps(list(payload.contracts)),
                json.dumps(sorted(payload.owned_command_topics)),
                json.dumps(sorted(payload.subscribed_event_topics)),
                json.dumps(list(payload.handlers)),
                json.dumps(list(payload.skipped_contracts)),
                json.dumps(list(payload.failed_contracts)),
                json.dumps(list(payload.ownership_violations)),
                payload.image_digest,
                payload.started_at,
            )

        if row is None:
            logger.debug(
                "Runtime manifest duplicate skipped (ON CONFLICT DO NOTHING)",
                extra={
                    "runtime_profile": payload.runtime_profile,
                    "topology_hash": payload.topology_hash,
                    "started_at": payload.started_at.isoformat(),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.info(
                "Runtime manifest inserted",
                extra={
                    "id": row["id"],
                    "runtime_profile": payload.runtime_profile,
                    "manifest_hash": payload.manifest_hash,
                    "correlation_id": str(correlation_id),
                },
            )


__all__ = ["HandlerPostgresRuntimeManifestInsert"]
