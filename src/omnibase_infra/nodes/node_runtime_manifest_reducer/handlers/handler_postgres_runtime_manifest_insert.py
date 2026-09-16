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
from uuid import UUID, uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumPostgresErrorCode,
)
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
    ModelPayloadInsertRuntimeManifest,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.runtime.models.model_runtime_manifest_published import (
        ModelRuntimeManifestPublished,
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
    started_at,
    attach_state,
    attach_required_contracts,
    attach_attached_contracts,
    attach_not_ready_contracts
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
ON CONFLICT (runtime_profile, topology_hash, started_at) DO NOTHING
RETURNING id;
"""


class HandlerPostgresRuntimeManifestInsert(MixinPostgresOpExecutor):
    """Append-only INSERT handler for the runtime_manifests projection table.

    Receives the published ModelRuntimeManifestPublished event, folds it into a
    ModelPayloadInsertRuntimeManifest and performs a single INSERT. Duplicate
    startup events (same runtime_profile + topology_hash + started_at) are
    silently ignored via ON CONFLICT DO NOTHING.

    Attributes:
        _pool: asyncpg connection pool, injected by name from
            ``service_kernel._build_runtime_handler_dependencies``.

    Example:
        >>> handler = HandlerPostgresRuntimeManifestInsert(pool)
        >>> result = await handler.handle(envelope)
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
        envelope: ModelEventEnvelope[ModelRuntimeManifestPublished],
    ) -> ModelBackendResult:
        """INSERT a runtime manifest row, ignoring exact duplicate startups.

        This is the auto-wiring dispatch entrypoint for
        ``onex.evt.omnibase-infra.runtime-manifest-published.v1``. Auto-wiring
        calls ``handle`` with exactly ONE positional argument on every path, so
        the correlation id is read off the envelope rather than taken as a second
        parameter — the second parameter is what made every manifest event
        dead-letter with ``TypeError: handle() missing 1 required positional
        argument: 'correlation_id'`` once OMN-17296's publisher fix let the event
        resolve to this dispatcher at all.

        The contract declares the matching ``event_model``, so the adapter has
        already validated the wire payload into ``ModelRuntimeManifestPublished``
        before this runs; the fold into the INSERT payload happens on the model
        (``from_manifest_event``), not here.

        A missing correlation id is generated rather than refused, per this
        repo's standing correlation-id rule ("always propagate from incoming
        requests; auto-generate with uuid4() if missing"), and the substitution
        is logged at WARNING so it stays observable instead of silent. Refusing
        would trade the row — the durable OMN-15512 attach-readiness surface,
        which is the point of this projection — for a tracing field the row is
        not keyed on: its identity is
        ``(runtime_profile, topology_hash, started_at)``.

        Args:
            envelope: Event envelope carrying the published boot manifest.

        Returns:
            ModelBackendResult indicating success or failure.
        """
        payload = ModelPayloadInsertRuntimeManifest.from_manifest_event(
            envelope.payload
        )
        correlation_id = envelope.correlation_id
        if correlation_id is None:
            correlation_id = uuid4()
            logger.warning(
                "runtime-manifest-published envelope carried no correlation_id; "
                "generated one for this INSERT. publish_runtime_manifest declares "
                "it required, so an envelope without one did not come from the "
                "sanctioned publisher and the producer is worth checking.",
                extra={
                    "runtime_profile": payload.runtime_profile,
                    "generated_correlation_id": str(correlation_id),
                },
            )
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
        # OMN-15512: flatten the attach-readiness aggregate across four
        # columns. `attach_state` stays 'unknown' (the column default) when the
        # per-contract interleave never ran — deliberately distinct from
        # 'ready', which asserts it ran and every contract attached.
        readiness = payload.attach_readiness
        if readiness is None:
            attach_state = "unknown"
            attach_required = 0
            attach_attached = 0
            attach_not_ready_json = "[]"
        else:
            attach_state = readiness.state.value
            attach_required = readiness.required_contracts
            attach_attached = readiness.attached_contracts
            attach_not_ready_json = json.dumps(
                [r.model_dump(mode="json") for r in readiness.not_ready_results]
            )

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
                attach_state,
                attach_required,
                attach_attached,
                attach_not_ready_json,
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
