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
from omnibase_infra.nodes.node_build_loop_projection_compute.models import (
    ModelPayloadBuildLoopAppend,
)
from omnibase_infra.nodes.node_build_loop_write_effect.models import (
    ModelBuildLoopAppendResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_db import HandlerDb

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
        container: ModelONEXContainer,
        db_handler: HandlerDb,
    ) -> None:
        self._container = container
        self._db_handler = db_handler
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        if not getattr(self._db_handler, "_initialized", False):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
            )
            raise RuntimeHostError(
                "HandlerDb must be initialized before HandlerBuildLoopAppend",
                context=ctx,
            )
        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("HandlerBuildLoopAppend shutdown complete")

    async def append(
        self,
        payload: ModelPayloadBuildLoopAppend,
    ) -> ModelBuildLoopAppendResult:
        """INSERT one row into build_loop_runs."""
        correlation_id = payload.correlation_id or uuid4()

        if not self._initialized:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_loop.append",
            )
            raise RuntimeHostError(
                "HandlerBuildLoopAppend not initialized. Call initialize() first.",
                context=ctx,
            )

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

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelBuildLoopAppendResult]:
        """ProtocolHandler entry point."""
        correlation_id_raw = envelope.get("correlation_id")
        correlation_id = (
            UUID(str(correlation_id_raw)) if correlation_id_raw else uuid4()
        )
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
        result = await self.append(payload)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_BUILD_LOOP_APPEND,
            result=result,
        )


__all__ = [
    "HandlerBuildLoopAppend",
    "HANDLER_ID_BUILD_LOOP_APPEND",
]
