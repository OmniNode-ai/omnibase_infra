# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerBuildLoopAppend - INSERT one ModelPayloadBuildLoopAppend into build_loop_runs.

Mirrors the canonical HandlerLedgerAppend, except that build_loop_runs is
**append-only**: there is no ON CONFLICT clause and no idempotency key. Retried
terminal events surface as duplicate rows so that duplication is observable.

Composes with HandlerDb (PostgreSQL transport) for circuit-breaker protection,
error classification, and connection-pool management.
"""

from __future__ import annotations

import asyncio
import json
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
from omnibase_infra.nodes.node_build_loop_projection_compute.models import (
    ModelPayloadBuildLoopAppend,
)
from omnibase_infra.nodes.node_build_loop_write_effect.models import (
    ModelBuildLoopAppendResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_BUILD_LOOP_APPEND: str = "build-loop-append-handler"

# Append-only INSERT. No ON CONFLICT — retried terminal events become duplicate rows.
# RETURNING is used to confirm the row was written (for the success bool + id echo).
_SQL_APPEND = """
INSERT INTO public.build_loop_runs (
    id,
    run_id,
    workflow_name,
    event_type,
    terminal_event_at,
    payload
) VALUES ($1, $2, $3, $4, $5, $6)
RETURNING id
"""


class HandlerBuildLoopAppend:
    """EFFECT handler that INSERTs one row into public.build_loop_runs."""

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        db_dsn: str | None = None,
    ) -> None:
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
        logger.info("HandlerBuildLoopAppend shutdown complete")

    async def _ensure_db_ready(self) -> None:
        if self._initialized:
            return
        async with self._db_init_lock:
            if self._initialized:
                return
            dsn = self._db_dsn
            if self._container is None or self._db_handler is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="build_loop.append.connect",
                )
                raise RuntimeHostError(
                    "Missing ONEX container for build_loop persistence -- provide "
                    "container at construction",
                    context=ctx,
                )
            if not dsn:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="build_loop.append.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for build_loop persistence -- provide "
                    "db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

    async def append(
        self,
        payload: ModelPayloadBuildLoopAppend,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelBuildLoopAppendResult:
        """INSERT one row into build_loop_runs.

        ``correlation_id`` resolution order: caller-supplied → payload-embedded
        → fresh uuid4. Pass the envelope-derived correlation_id from
        ``execute()`` to keep request traceability intact.
        """
        correlation_id = correlation_id or payload.correlation_id or uuid4()

        await self._ensure_db_ready()

        payload_json = json.dumps(payload.payload)
        parameters: list[object] = [
            str(payload.id),  # $1 UUID
            payload.run_id,  # $2 TEXT
            payload.workflow_name,  # $3 TEXT
            payload.event_type,  # $4 TEXT
            payload.terminal_event_at,  # $5 TIMESTAMPTZ
            payload_json,  # $6 JSONB
        ]

        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {
                "sql": _SQL_APPEND,
                "parameters": parameters,
            },
            "correlation_id": str(correlation_id),
        }

        logger.debug(
            "Appending build_loop terminal event",
            extra={
                "id": str(payload.id),
                "run_id": payload.run_id,
                "workflow_name": payload.workflow_name,
                "correlation_id": str(correlation_id),
            },
        )

        if self._db_handler is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_loop.append",
            )
            raise RuntimeHostError("Database handler is not available", context=ctx)

        db_result = await self._db_handler.execute(envelope)

        if db_result.result is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_loop.append",
            )
            raise RuntimeHostError("Database operation returned no result", context=ctx)

        rows = db_result.result.payload.rows
        if not rows:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_loop.append",
            )
            raise RuntimeHostError(
                "INSERT into build_loop_runs returned no row", context=ctx
            )

        inserted_id = UUID(str(rows[0]["id"]))
        return ModelBuildLoopAppendResult(
            success=True,
            id=inserted_id,
            run_id=payload.run_id,
            workflow_name=payload.workflow_name,
        )

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelBuildLoopAppendResult]:
        """Contract-routed operation_match entry point."""
        payload_raw = self._extract_envelope_field(envelope, "payload")
        if payload_raw is None:
            payload_raw = envelope
        payload = (
            payload_raw
            if isinstance(payload_raw, ModelPayloadBuildLoopAppend)
            else ModelPayloadBuildLoopAppend.model_validate(payload_raw)
        )

        envelope_correlation_id = self._extract_envelope_field(
            envelope, "correlation_id"
        )
        correlation_id = self._safe_correlation_id(
            envelope_correlation_id or payload.correlation_id
        )
        result = await self.append(payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_BUILD_LOOP_APPEND,
            result=result,
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelBuildLoopAppendResult]:
        """ProtocolHandler entry point."""
        correlation_id = self._safe_correlation_id(envelope.get("correlation_id"))
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_loop.append",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        payload = ModelPayloadBuildLoopAppend.model_validate(payload_raw)
        result = await self.append(payload, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_BUILD_LOOP_APPEND,
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
        we never want a malformed envelope to surface as ValueError to the
        runtime, since terminal-event persistence is best-effort audit.
        """
        if not raw:
            return uuid4()
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()


__all__ = [
    "HandlerBuildLoopAppend",
    "HANDLER_ID_BUILD_LOOP_APPEND",
]
