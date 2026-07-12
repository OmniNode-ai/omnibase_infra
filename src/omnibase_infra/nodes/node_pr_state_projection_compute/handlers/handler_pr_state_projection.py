# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerPrStateProjection - GitHub PR status event -> ModelIntent transformer.

Mirrors the canonical HandlerBuildLoopProjection. Receives a Kafka
ModelEventMessage carrying onex.evt.github.pr-status.v1 (published by
node_github_pr_poller_effect), extracts identifying fields (repo, pr_number,
triage_state, title), and emits a ModelIntent with a ModelPayloadPrStateUpsert
payload for NodePrStateWriteEffect to persist.

Ticket: OMN-14375
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
from omnibase_infra.nodes.node_pr_state_projection_compute.models.model_payload_pr_state_upsert import (
    ModelPayloadPrStateUpsert,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_PR_STATE_PROJECTION: str = "pr-state-projection-handler"


def _require_str(
    body: dict[str, JsonType],
    key: str,
    correlation_id: UUID | None,
) -> str:
    """Return a required non-empty string field, or raise RuntimeHostError.

    Module-level (not a handler method) since it is a pure function of its
    arguments -- mirrors the compute_triage_state / structural_confidence
    module-level-helper precedent used elsewhere in this repo's handlers.
    """
    value = body.get(key)
    if isinstance(value, str) and value:
        return value
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.KAFKA,
        operation="pr_state_projection.extract_payload",
    )
    raise RuntimeHostError(
        f"PR status event missing required field '{key}'",
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        context=context,
    )


def _require_int(
    body: dict[str, JsonType],
    key: str,
    correlation_id: UUID | None,
) -> int:
    """Return a required integer field, or raise RuntimeHostError."""
    value = body.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=correlation_id,
        transport_type=EnumInfraTransportType.KAFKA,
        operation="pr_state_projection.extract_payload",
    )
    raise RuntimeHostError(
        f"PR status event missing required integer field '{key}'",
        error_code=EnumCoreErrorCode.INVALID_INPUT,
        context=context,
    )


class HandlerPrStateProjection:
    """COMPUTE handler that projects GitHub PR status events into write intents."""

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
        logger.info("HandlerPrStateProjection shutdown complete")

    def project(self, message: ModelEventMessage) -> ModelIntent:
        """Transform a GitHub PR status event message into a pr_state upsert intent."""
        payload = self._extract_payload(message)
        return ModelIntent(
            intent_type=payload.intent_type,
            target=f"postgres://pr_state/{payload.repo}:{payload.pr_number}",
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
            handler_id=HANDLER_ID_PR_STATE_PROJECTION,
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
                operation="pr_state_projection.execute",
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
            handler_id=HANDLER_ID_PR_STATE_PROJECTION,
            result=intent,
        )

    def _extract_payload(self, message: ModelEventMessage) -> ModelPayloadPrStateUpsert:
        """Extract GitHub PR status fields into a ModelPayloadPrStateUpsert.

        Required: repo and pr_number (a status event without an identifiable
        PR is a producer bug, not a partial-data case) and triage_state.
        Best-effort: every other field; the CI/review/merge-queue columns are
        reserved for a richer producer (see migration 091_pr_state.sql) and
        default to None here since the current poller payload does not carry
        them.
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
                operation="pr_state_projection.extract_payload",
            )
            raise RuntimeHostError(
                f"Cannot decode PR status event body as JSON: {type(e).__name__}",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            ) from e

        if not isinstance(decoded, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=header_correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="pr_state_projection.extract_payload",
            )
            raise RuntimeHostError(
                "PR status event body must decode to a JSON object, "
                f"got {type(decoded).__name__}.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        # Events may arrive raw or wrapped in a ModelEventEnvelope; unwrap.
        body: dict[str, JsonType] = (
            decoded["payload"] if isinstance(decoded.get("payload"), dict) else decoded
        )

        repo = _require_str(body, "repo", header_correlation_id)
        pr_number = _require_int(body, "pr_number", header_correlation_id)
        triage_state = self._first_str(body, ("triage_state",)) or "needs_review"
        title = self._first_str(body, ("title",)) or ""
        as_of = self._extract_timestamp(body)
        correlation_id = self._extract_correlation_id(body, header_correlation_id)

        return ModelPayloadPrStateUpsert(
            repo=repo,
            pr_number=pr_number,
            triage_state=triage_state,
            title=title,
            correlation_id=correlation_id,
            as_of=as_of,
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
        runtime, since pr_state projection is best-effort read-model refresh.
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
        for k in ("as_of", "timestamp", "updated_at", "occurred_at"):
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
    "HandlerPrStateProjection",
    "HANDLER_ID_PR_STATE_PROJECTION",
]
