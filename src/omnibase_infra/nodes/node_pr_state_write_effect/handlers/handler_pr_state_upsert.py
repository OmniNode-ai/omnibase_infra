# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerPrStateUpsert - UPSERT one ModelPayloadPrStateUpsert into pr_state.

pr_state is a latest-known-state projection: ON CONFLICT (repo, pr_number) DO
UPDATE keeps exactly one row per PR, refreshed on every producer cycle. This
is the intentional divergence from build_loop_runs (append-only, no ON
CONFLICT) and matches event_ledger's use of `RETURNING (xmax = 0)` to detect
insert-vs-update (same idiom used by HandlerWriteDecision's Stage 1 upsert).

Composes with HandlerDb (PostgreSQL transport) for circuit-breaker
protection, error classification, and connection-pool management, mirroring
HandlerLedgerAppend/HandlerBuildLoopAppend (OMN-14140's internally-composed
HandlerDb pattern, so the auto-wiring resolver's known-injectable
constructor-argument set can build this handler from `container` alone).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.nodes.node_pr_state_projection_compute.models import (
    ModelPayloadPrStateUpsert,
)
from omnibase_infra.nodes.node_pr_state_write_effect.models import (
    ModelPrStateUpsertResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_PR_STATE_UPSERT: str = "pr-state-upsert-handler"

# UPSERT keyed on (repo, pr_number). RETURNING (xmax = 0) distinguishes an
# INSERT (was_insert=True) from an UPDATE via the ON CONFLICT branch —
# same idiom as HandlerWriteDecision's SQL_UPSERT_DECISION.
_SQL_UPSERT = """
INSERT INTO public.pr_state (
    repo,
    pr_number,
    triage_state,
    title,
    ci_status,
    review_decision,
    mergeable,
    merge_state_status,
    merge_queue_state,
    base_ref,
    head_ref,
    source,
    correlation_id,
    as_of,
    is_draft,
    projected_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW()
)
ON CONFLICT (repo, pr_number) DO UPDATE SET
    triage_state        = EXCLUDED.triage_state,
    title                = EXCLUDED.title,
    ci_status            = EXCLUDED.ci_status,
    review_decision      = EXCLUDED.review_decision,
    mergeable            = EXCLUDED.mergeable,
    merge_state_status   = EXCLUDED.merge_state_status,
    merge_queue_state    = EXCLUDED.merge_queue_state,
    base_ref             = EXCLUDED.base_ref,
    head_ref             = EXCLUDED.head_ref,
    source               = EXCLUDED.source,
    correlation_id       = EXCLUDED.correlation_id,
    as_of                = EXCLUDED.as_of,
    is_draft             = EXCLUDED.is_draft,
    projected_at         = NOW()
RETURNING (xmax = 0) AS was_insert
"""


class HandlerPrStateUpsert:
    """EFFECT handler that UPSERTs one row into public.pr_state."""

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        db_dsn: str | None = None,
    ) -> None:
        """Initialize the pr_state upsert handler.

        Args:
            container: ONEX dependency injection container. HandlerDb is
                composed internally from this container (matching OMN-14140's
                pattern). ``container`` defaults to ``None`` (mirroring
                HandlerBuildLoopAppend) so the generic auto-wiring resolver's
                zero-required-param fast path can construct this handler
                directly, without threading a container through
                `_materialize_known_handler_dependencies` -- the kernel's
                explicit `service_kernel.py` registration always supplies a
                real container for the live intent-bridge path; this
                constructor only needs to tolerate a bare construction for
                the contract's defensive Kafka subscribe_topics path.
            db_dsn: Optional PostgreSQL DSN supplied by the runtime
                auto-wiring boundary. Handlers do not read environment
                directly; runtime composition owns that IO boundary.
        """
        self._container = container
        self._db_handler = HandlerDb(container) if container is not None else None
        self._db_dsn = db_dsn.strip() if db_dsn else ""
        self._initialized: bool = False
        self._db_init_lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        config_dsn = config.get("dsn")
        if isinstance(config_dsn, str) and config_dsn.strip():
            self._db_dsn = config_dsn.strip()
        await self._ensure_db_ready()
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        if self._initialized and self._db_handler is not None:
            await self._db_handler.shutdown()
        self._initialized = False
        logger.info("HandlerPrStateUpsert shutdown complete")

    async def _ensure_db_ready(self) -> None:
        """Lazily connect the composed HandlerDb on first real use.

        The auto-wiring resolver constructs contract-routed handlers from
        `container` alone and never calls their `initialize()` method
        (OMN-14140), so this handler owns its HandlerDb connection lifecycle
        instead of relying on an external initialize() call.
        """
        if self._initialized:
            return
        async with self._db_init_lock:
            if self._initialized:
                return
            dsn = self._db_dsn
            if self._container is None or self._db_handler is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="pr_state.upsert.connect",
                )
                raise RuntimeHostError(
                    "Missing ONEX container for pr_state persistence -- provide "
                    "container at construction",
                    context=ctx,
                )
            if not dsn:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="pr_state.upsert.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for pr_state persistence -- provide "
                    "db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

    async def upsert(
        self,
        payload: ModelPayloadPrStateUpsert,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelPrStateUpsertResult:
        """UPSERT one row into pr_state.

        ``correlation_id`` resolution order: caller-supplied → payload-embedded
        → fresh uuid4.
        """
        correlation_id = correlation_id or payload.correlation_id or uuid4()

        await self._ensure_db_ready()

        parameters: list[object] = [
            payload.repo,  # $1
            payload.pr_number,  # $2
            payload.triage_state,  # $3
            payload.title,  # $4
            payload.ci_status,  # $5
            payload.review_decision,  # $6
            payload.mergeable,  # $7
            payload.merge_state_status,  # $8
            payload.merge_queue_state,  # $9
            payload.base_ref,  # $10
            payload.head_ref,  # $11
            payload.source,  # $12
            str(payload.correlation_id) if payload.correlation_id else None,  # $13
            payload.as_of,  # $14
            payload.is_draft,  # $15
        ]

        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {
                "sql": _SQL_UPSERT,
                "parameters": parameters,
            },
            "correlation_id": str(correlation_id),
        }

        logger.debug(
            "Upserting pr_state row",
            extra={
                "repo": payload.repo,
                "pr_number": payload.pr_number,
                "triage_state": payload.triage_state,
                "correlation_id": str(correlation_id),
            },
        )

        if self._db_handler is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="pr_state.upsert",
            )
            raise RuntimeHostError("Database handler is not available", context=ctx)

        db_result = await self._db_handler.execute(envelope)

        if db_result.result is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="pr_state.upsert",
            )
            raise RuntimeHostError("Database operation returned no result", context=ctx)

        rows = db_result.result.payload.rows
        if not rows:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="pr_state.upsert",
            )
            raise RuntimeHostError("UPSERT into pr_state returned no row", context=ctx)

        was_insert = bool(rows[0]["was_insert"])
        return ModelPrStateUpsertResult(
            success=True,
            repo=payload.repo,
            pr_number=payload.pr_number,
            was_insert=was_insert,
        )

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelPrStateUpsertResult]:
        """Contract-routed operation_match entry point."""
        payload_raw = self._extract_envelope_field(envelope, "payload")
        if payload_raw is None:
            payload_raw = envelope
        payload = (
            payload_raw
            if isinstance(payload_raw, ModelPayloadPrStateUpsert)
            else ModelPayloadPrStateUpsert.model_validate(payload_raw)
        )

        envelope_correlation_id = self._extract_envelope_field(
            envelope, "correlation_id"
        )
        correlation_id = self._safe_correlation_id(
            envelope_correlation_id or payload.correlation_id
        )
        result = await self.upsert(payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_PR_STATE_UPSERT,
            result=result,
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelPrStateUpsertResult]:
        """ProtocolHandler entry point."""
        correlation_id = self._safe_correlation_id(envelope.get("correlation_id"))
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="pr_state.upsert",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        payload = ModelPayloadPrStateUpsert.model_validate(payload_raw)
        result = await self.upsert(payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_PR_STATE_UPSERT,
            result=result,
        )

    @staticmethod
    def _extract_envelope_field(envelope: object, key: str) -> object:
        if isinstance(envelope, dict):
            return envelope.get(key)
        return getattr(envelope, key, None)

    @staticmethod
    def _safe_correlation_id(raw: object) -> UUID:
        """Parse a correlation ID from envelope-supplied raw input.

        Returns a fresh UUID if `raw` is missing, empty, or unparseable —
        pr_state projection is best-effort read-model refresh and must never
        drop a refresh over a malformed correlation_id.
        """
        if not raw:
            return uuid4()
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()


__all__ = [
    "HandlerPrStateUpsert",
    "HANDLER_ID_PR_STATE_UPSERT",
]
