# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerBuildLoopProjection - terminal-event → ModelIntent transformer.

Mirrors the canonical HandlerLedgerProjection. Receives a Kafka
ModelEventMessage carrying onex.evt.omnimarket.build-loop-orchestrator-completed.v1,
extracts identifying fields (run_id, workflow_name, event_type,
terminal_event_at), and emits a ModelIntent with a ModelPayloadBuildLoopAppend
payload for NodeBuildLoopWriteEffect to persist.

Ticket: OMN-9774
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.types import JsonType
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_build_loop_projection_compute.models.model_payload_build_loop_append import (
    ModelPayloadBuildLoopAppend,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_BUILD_LOOP_PROJECTION: str = "build-loop-projection-handler"


class HandlerBuildLoopProjection:
    """COMPUTE handler that projects terminal events into write intents."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def initialize(self, config: dict[str, object]) -> None:
        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("HandlerBuildLoopProjection shutdown complete")

    def project(self, message: ModelEventMessage) -> ModelIntent:
        """Transform a terminal-event message into a build_loop append intent."""
        payload = self._extract_payload(message)
        return ModelIntent(
            intent_type=payload.intent_type,
            target=f"postgres://build_loop_runs/{payload.id}",
            payload=payload,
        )

    async def handle(
        self,
        message: object,
    ) -> ModelHandlerOutput[ModelIntent]:
        """Contract-typed auto-wiring entry point."""
        raw_message = self._coerce_event_message(message)
        intent = self.project(raw_message)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=raw_message.headers.correlation_id,
            handler_id=HANDLER_ID_BUILD_LOOP_PROJECTION,
            result=intent,
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelIntent]:
        """ProtocolHandler entry point — extract message, delegate to project()."""
        correlation_id = self._safe_correlation_id(envelope.get("correlation_id"))
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="build_loop_projection.execute",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope: expected dict, "
                f"got {type(payload_raw).__name__}",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        message = ModelEventMessage.model_validate(payload_raw)
        intent = self.project(message)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_BUILD_LOOP_PROJECTION,
            result=intent,
        )

    def _extract_payload(
        self, message: ModelEventMessage
    ) -> ModelPayloadBuildLoopAppend:
        """Extract terminal-event fields into a ModelPayloadBuildLoopAppend.

        Required: a non-empty event body that decodes to a JSON object.
        Best-effort: every other field; if a value is missing the row is still
        persisted with sensible fallbacks so duplicate / partial events are
        surfaced rather than silently dropped (consistent with the
        append-only semantics of build_loop_runs).
        """
        headers = message.headers
        header_correlation_id = headers.correlation_id if headers else None

        # ModelEventMessage.value is `bytes` (non-nullable per schema) so we do
        # not check for None — empty bytes will raise during JSON decoding,
        # which is handled below.
        try:
            decoded = json.loads(message.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=header_correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="build_loop_projection.extract_payload",
            )
            raise RuntimeHostError(
                f"Cannot decode terminal-event body as JSON: {type(e).__name__}",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            ) from e

        if not isinstance(decoded, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=header_correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="build_loop_projection.extract_payload",
            )
            raise RuntimeHostError(
                "Terminal event body must decode to a JSON object, "
                f"got {type(decoded).__name__}.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        # Events may arrive raw or wrapped in a ModelEventEnvelope; unwrap.
        body: dict[str, JsonType] = (
            decoded["payload"] if isinstance(decoded.get("payload"), dict) else decoded
        )

        run_id = self._first_str(body, ("run_id", "workflow_run_id", "id")) or "unknown"
        workflow_name = (
            self._first_str(body, ("workflow_name", "workflow", "name")) or "build_loop"
        )
        event_type = (
            self._first_str(body, ("event_type", "type"))
            or "build-loop-orchestrator-completed"
        )
        terminal_event_at = self._extract_timestamp(body)
        correlation_id = self._extract_correlation_id(body, header_correlation_id)

        return ModelPayloadBuildLoopAppend(
            run_id=run_id,
            workflow_name=workflow_name,
            event_type=event_type,
            terminal_event_at=terminal_event_at,
            payload=body,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _coerce_event_message(raw: object) -> ModelEventMessage:
        """Accept direct ModelEventMessage or an auto-wired envelope wrapper."""
        if isinstance(raw, ModelEventMessage):
            return raw
        payload = getattr(raw, "payload", raw)
        if isinstance(raw, dict):
            payload = raw.get("payload", raw)
        return ModelEventMessage.model_validate(payload)

    @staticmethod
    def _safe_correlation_id(raw: object) -> UUID:
        """Parse a correlation ID from envelope-supplied raw input.

        Returns a fresh UUID if `raw` is missing, empty, or unparseable —
        we never want a malformed envelope to surface as ValueError to the
        runtime, since terminal-event projection is best-effort audit.
        """
        if not raw:
            return uuid4()
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()

    @staticmethod
    def _first_str(body: dict[str, JsonType], keys: tuple[str, ...]) -> str | None:
        for k in keys:
            v = body.get(k)
            if isinstance(v, str) and v:
                return v
        return None

    @staticmethod
    def _extract_timestamp(body: dict[str, JsonType]) -> datetime:
        for k in ("terminal_event_at", "timestamp", "completed_at", "occurred_at"):
            raw = body.get(k)
            if isinstance(raw, str) and raw:
                try:
                    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                # Persist into TIMESTAMPTZ; assume UTC if input lacked a tzinfo.
                return (
                    parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
                )
        return datetime.now(UTC)

    @staticmethod
    def _extract_correlation_id(
        body: dict[str, JsonType], fallback: UUID | None
    ) -> UUID | None:
        raw = body.get("correlation_id")
        if isinstance(raw, str) and raw:
            try:
                return UUID(raw)
            except ValueError:
                pass
        return fallback


__all__ = [
    "HandlerBuildLoopProjection",
    "HANDLER_ID_BUILD_LOOP_PROJECTION",
]
