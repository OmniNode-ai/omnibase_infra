# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for ledger projection - transforms ModelEventMessage to ModelIntent.

This handler encapsulates the compute logic for projecting platform events
to ledger append intents. It follows the ONEX declarative pattern where
all business logic resides in handlers, not nodes.

Design Rationale - Best-Effort Metadata Extraction:
    The audit ledger serves as the system's source of truth. Events must NEVER
    be dropped due to metadata extraction failures. All metadata fields are
    extracted best-effort - parsing errors result in None/empty values, not
    exceptions. Only a missing event_value (the raw bytes) causes an error.

Bytes Encoding:
    Kafka event keys and values are bytes. Since bytes cannot safely cross
    intent boundaries (serialization issues), they are base64-encoded at this
    transform layer. The Effect layer decodes before storage.

Ticket: OMN-1648, OMN-1726
"""

from __future__ import annotations

import base64
import logging
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
    EnumResponseStatus,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_LEDGER_PROJECTION: str = "ledger-projection-handler"


class HandlerLedgerProjection:
    """Handler that transforms platform events to ledger append intents.

    This handler implements the compute logic for the ledger projection node,
    extracting metadata from ModelEventMessage and producing ModelIntent with
    ModelPayloadLedgerAppend payloads.

    CRITICAL INVARIANTS:
    - NEVER drop events due to metadata extraction failure
    - event_value is REQUIRED (raises RuntimeHostError if None)
    - correlation_id and other metadata are OPTIONAL
    - Best-effort extraction - parsing errors yield None, not exceptions

    Attributes:
        handler_type: EnumHandlerType.COMPUTE_HANDLER
        handler_category: EnumHandlerTypeCategory.COMPUTE

    Example:
        >>> handler = HandlerLedgerProjection(container)
        >>> message = ModelEventMessage(
        ...     topic="agent.routing.completed.v1",
        ...     value=b'{"agent": "code-quality"}',
        ...     headers=headers,
        ...     partition=0,
        ...     offset="42",
        ... )
        >>> result = await handler.execute({"payload": message.model_dump()})
        >>> # result.result contains the ModelIntent with ledger.append payload
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the ledger projection handler.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.COMPUTE_HANDLER - This handler performs pure
            computation (event transformation) without side effects.
        """
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.COMPUTE - This handler performs pure,
            deterministic transformations without side effects.
        """
        return EnumHandlerTypeCategory.COMPUTE

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler.

        Args:
            config: Configuration dict (currently unused).
        """
        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerLedgerProjection shutdown complete")

    def project(self, message: ModelEventMessage) -> ModelIntent:
        """Transform event message to ledger append intent.

        Extracts metadata from the incoming Kafka event and produces a
        ModelIntent with a ModelPayloadLedgerAppend payload for the Effect
        layer to persist to PostgreSQL.

        Args:
            message: The incoming Kafka event message to transform.

        Returns:
            ModelIntent with intent_type="extension" containing the ledger
            append payload for the Effect layer.

        Raises:
            RuntimeHostError: If message.value is None (event body is
                required), with KAFKA transport context and INVALID_INPUT
                error code.

        INVARIANTS:
        - Never drop events due to metadata extraction failure
        - event_value is REQUIRED (raises RuntimeHostError if None)
        - correlation_id is optional
        """
        payload = self._extract_ledger_metadata(message)
        return ModelIntent(
            intent_type=payload.intent_type,
            target=f"postgres://event_ledger/{payload.topic}/{payload.partition}/{payload.kafka_offset}",
            payload=payload,
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelIntent]:
        """Execute ledger projection from envelope (ProtocolHandler interface).

        This method provides the standard handler interface for contract-driven
        invocation. It extracts the ModelEventMessage from the envelope and
        delegates to the project() method.

        Args:
            envelope: Request envelope containing:
                - operation: "ledger.project"
                - payload: ModelEventMessage as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelIntent.

        Raises:
            RuntimeHostError: If message.value is None or payload is
                missing/invalid.
        """
        correlation_id_raw = envelope.get("correlation_id")
        correlation_id = (
            UUID(str(correlation_id_raw)) if correlation_id_raw else uuid4()
        )
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="ledger_projection.execute",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope: expected dict, "
                f"got {type(payload_raw).__name__}",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        # Parse payload into typed model
        message = ModelEventMessage.model_validate(payload_raw)

        # Execute projection
        intent = self.project(message)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_LEDGER_PROJECTION,
            result=intent,
        )

    def _b64(self, b: bytes | None) -> str | None:
        """Base64 encode bytes, returning None for None input.

        Args:
            b: Bytes to encode, or None.

        Returns:
            Base64-encoded string, or None if input was None.
        """
        if b is None:
            return None
        return base64.b64encode(b).decode("ascii")

    def _normalize_headers(
        self, headers: ModelEventHeaders | None
    ) -> dict[str, JsonType]:
        """Convert event headers to JSON-safe dictionary.

        Uses Pydantic's model_dump with mode="json" to ensure all values
        are JSON-serializable. Returns empty dict for None input.

        Args:
            headers: Event headers model, or None.

        Returns:
            JSON-safe dictionary of header values, or empty dict.
        """
        if headers is None:
            return {}
        try:
            return headers.model_dump(mode="json")
        except Exception:
            # Best-effort: try to get correlation_id for logging context
            correlation_id = getattr(headers, "correlation_id", None)
            logger.warning(
                "Failed to serialize event headers, returning empty dict. "
                "correlation_id=%s",
                correlation_id,
                exc_info=True,
            )
            return {}

    def _parse_offset(
        self, offset: str | None, correlation_id: UUID | None = None
    ) -> int:
        """Parse Kafka offset string to integer.

        Args:
            offset: Offset string from Kafka, or None.
            correlation_id: Optional correlation ID for logging context.

        Returns:
            Parsed offset as integer, or 0 if None or unparseable.
        """
        if offset is None:
            return 0
        try:
            return int(offset)
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse offset '%s' as integer, defaulting to 0. "
                "correlation_id=%s",
                offset,
                correlation_id,
            )
            return 0

    def _extract_ledger_metadata(
        self, message: ModelEventMessage
    ) -> ModelPayloadLedgerAppend:
        """Extract ledger metadata from event message.

        Main extraction logic that transforms a ModelEventMessage into a
        ModelPayloadLedgerAppend. Uses best-effort extraction for all
        metadata fields - only event_value being None causes an error.

        Args:
            message: The event message to extract metadata from.

        Returns:
            Populated ledger append payload ready for the Effect layer.

        Raises:
            RuntimeHostError: If message.value is None (with INVALID_INPUT
                error code and KAFKA transport context).

        Field Mapping:
            | Payload Field    | Source                          | Required |
            |------------------|---------------------------------|----------|
            | topic            | message.topic                   | YES      |
            | partition        | message.partition               | YES*     |
            | kafka_offset     | message.offset                  | YES*     |
            | event_key        | base64(message.key)             | NO       |
            | event_value      | base64(message.value)           | YES      |
            | correlation_id   | message.headers.correlation_id  | NO       |
            | event_type       | message.headers.event_type      | NO       |
            | source           | message.headers.source          | NO       |
            | envelope_id      | message.headers.message_id      | NO       |
            | event_timestamp  | message.headers.timestamp       | NO       |
            | onex_headers     | headers.model_dump(mode="json") | NO       |

            * Defaults to 0 if not available (for consumed messages, these
              should always be present, but we handle None defensively).
        """
        # Extract headers best-effort for correlation context in error paths
        headers = message.headers
        correlation_id = headers.correlation_id if headers else None

        # CRITICAL: event_value is required - this is the only case where we raise
        if message.value is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="ledger_projection.extract_metadata",
            )
            raise RuntimeHostError(
                "Cannot create ledger entry: message.value is None. "
                "Event body is required for audit ledger persistence.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        # Base64 encode the raw bytes
        event_value_b64 = self._b64(message.value)
        # Defensive check - _b64 should never return None for non-None input
        if event_value_b64 is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="ledger_projection.extract_metadata",
            )
            raise RuntimeHostError(
                "Unexpected: base64 encoding of message.value returned None. "
                "This should never happen for non-None bytes input.",
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                context=context,
            )

        event_key_b64 = self._b64(message.key)

        # headers and correlation_id already extracted above for error context
        onex_headers = self._normalize_headers(headers)

        # Build payload with best-effort metadata extraction
        return ModelPayloadLedgerAppend(
            # Required Kafka position fields (defensive defaults for None)
            topic=message.topic,
            partition=message.partition if message.partition is not None else 0,
            kafka_offset=self._parse_offset(
                message.offset, correlation_id=correlation_id
            ),
            # Raw event data as base64
            event_key=event_key_b64,
            event_value=event_value_b64,
            # Extracted metadata (all optional, best-effort)
            onex_headers=onex_headers,
            correlation_id=correlation_id,
            envelope_id=headers.message_id if headers else None,
            event_type=headers.event_type if headers else None,
            source=headers.source if headers else None,
            event_timestamp=headers.timestamp if headers else None,
        )


__all__ = ["HandlerLedgerProjection"]
