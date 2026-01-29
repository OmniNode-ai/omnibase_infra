# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeLedgerProjectionCompute - Extracts metadata from platform events for ledger persistence.

This COMPUTE node transforms ModelEventMessage into ModelPayloadLedgerAppend wrapped
in a ModelIntent for the Effect layer. It follows the ONEX declarative pattern:
    - DECLARATIVE node driven by contract.yaml
    - Subscribes to 7 platform topics via contract configuration
    - Transforms events to ledger append intents

Design Rationale - Best-Effort Metadata Extraction:
    The audit ledger serves as the system's source of truth. Events must NEVER
    be dropped due to metadata extraction failures. All metadata fields are
    extracted best-effort - parsing errors result in None/empty values, not
    exceptions. Only a missing event_value (the raw bytes) causes an error.

Bytes Encoding:
    Kafka event keys and values are bytes. Since bytes cannot safely cross
    intent boundaries (serialization issues), they are base64-encoded at this
    transform layer. The Effect layer decodes before storage.

Subscribed Topics:
    - onex.evt.platform.node-registration.v1
    - onex.evt.platform.node-introspection.v1
    - onex.evt.platform.node-heartbeat.v1
    - onex.cmd.platform.request-introspection.v1
    - onex.evt.platform.fsm-state-transitions.v1
    - onex.intent.platform.runtime-tick.v1
    - onex.snapshot.platform.registration-snapshots.v1

Ticket: OMN-1648
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from omnibase_core.errors import OnexError
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.types import JsonType
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


class NodeLedgerProjectionCompute(NodeCompute):
    """COMPUTE node that extracts metadata from platform events for ledger persistence.

    Declarative node - subscribes to 7 platform topics via contract.yaml.
    Transforms ModelEventMessage -> ModelPayloadLedgerAppend -> ModelIntent.

    This node implements the ONEX ledger projection pattern:
    1. Receives raw Kafka events as ModelEventMessage
    2. Extracts metadata best-effort (never fails on parse errors)
    3. Base64-encodes bytes for safe intent serialization
    4. Emits ModelIntent with "ledger.append" payload for Effect layer

    CRITICAL INVARIANTS:
    - NEVER drop events due to metadata extraction failure
    - event_value is REQUIRED (raises OnexError if None)
    - correlation_id and other metadata are OPTIONAL
    - Best-effort extraction - parsing errors yield None, not exceptions

    Attributes:
        container: ONEX dependency injection container.

    Example:
        ```python
        from omnibase_core.container import ModelONEXContainer

        container = ModelONEXContainer()
        node = NodeLedgerProjectionCompute(container)

        # Transform event to ledger intent
        message = ModelEventMessage(
            topic="agent.routing.completed.v1",
            value=b'{"agent": "code-quality"}',
            headers=headers,
            partition=0,
            offset="42",
        )
        intent = node.compute(message)
        # intent.payload.intent_type == "ledger.append"
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the ledger projection compute node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)

    def compute(self, message: ModelEventMessage) -> ModelIntent:
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
            OnexError: If message.value is None (event body is required).

        INVARIANTS:
        - Never drop events due to metadata extraction failure
        - event_value is REQUIRED (raises OnexError if None)
        - correlation_id is optional
        """
        payload = self._extract_ledger_metadata(message)
        return ModelIntent(
            intent_type="extension",
            target=f"postgres://event_ledger/{payload.topic}/{payload.partition}/{payload.kafka_offset}",
            payload=payload,
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
            OnexError: If message.value is None.

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
        # CRITICAL: event_value is required - this is the only case where we raise
        if message.value is None:
            raise OnexError(
                "Cannot create ledger entry: message.value is None. "
                "Event body is required for audit ledger persistence."
            )

        # Base64 encode the raw bytes
        event_value_b64 = self._b64(message.value)
        # Defensive check - _b64 should never return None for non-None input
        if event_value_b64 is None:
            raise OnexError(
                "Unexpected: base64 encoding of message.value returned None. "
                "This should never happen for non-None bytes input."
            )

        event_key_b64 = self._b64(message.key)

        # Extract headers best-effort
        headers = message.headers
        # Extract correlation_id early for logging context in helper methods
        correlation_id = headers.correlation_id if headers else None
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


__all__ = ["NodeLedgerProjectionCompute"]
