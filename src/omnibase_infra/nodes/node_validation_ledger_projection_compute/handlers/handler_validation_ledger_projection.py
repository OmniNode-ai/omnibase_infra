# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for validation ledger projection - transforms Kafka events to ledger entries.

This handler encapsulates the compute logic for projecting cross-repo validation
events into ModelValidationLedgerEntry for persistence in the
validation_event_ledger table.

Design Rationale - Best-Effort Metadata Extraction:
    The handler extracts domain fields (run_id, repo_id, event_type) from the
    Kafka message value (JSON). If JSON parsing fails, the handler logs a warning
    and raises, since domain fields are required for meaningful ledger entries.
    However, auxiliary metadata (event_version) uses defensive defaults.

Bytes Encoding:
    The raw Kafka message value is base64-encoded for transport in the model.
    SHA-256 hash is computed for integrity verification during replay.

Ticket: OMN-1908
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.errors import OnexError
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.validation_ledger import ModelValidationLedgerEntry

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_VALIDATION_LEDGER_PROJECTION: str = "validation-ledger-projection-handler"


class HandlerValidationLedgerProjection:
    """Handler that transforms cross-repo validation events to ledger entries.

    Implements the compute logic for the validation ledger projection node.
    Extracts domain fields from the Kafka message JSON payload, base64-encodes
    the raw bytes, and computes an envelope hash for integrity verification.

    INVARIANTS:
    - event_value is REQUIRED (raises OnexError if None)
    - run_id and repo_id are REQUIRED from the JSON payload
    - event_version defaults to "v1" if not extractable
    - SHA-256 hash computed from raw bytes for replay verification

    Attributes:
        handler_type: EnumHandlerType.COMPUTE_HANDLER
        handler_category: EnumHandlerTypeCategory.COMPUTE
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the validation ledger projection handler.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler."""
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
        logger.info("HandlerValidationLedgerProjection shutdown complete")

    def project(self, message: ModelEventMessage) -> ModelValidationLedgerEntry:
        """Transform Kafka event message to validation ledger entry.

        Extracts domain metadata (run_id, repo_id, event_type) from the
        message JSON payload, base64-encodes the raw bytes, and computes
        a SHA-256 hash for integrity verification.

        Args:
            message: The incoming Kafka event message.

        Returns:
            ModelValidationLedgerEntry ready for persistence.

        Raises:
            OnexError: If message.value is None or required domain fields
                cannot be extracted from the payload.
        """
        if message.value is None:
            raise OnexError(
                "Cannot create validation ledger entry: message.value is None. "
                "Event body is required for validation ledger persistence."
            )

        # Base64 encode raw bytes for transport
        envelope_bytes_b64 = base64.b64encode(message.value).decode("ascii")

        # Compute SHA-256 hash for integrity verification
        envelope_hash = hashlib.sha256(message.value).hexdigest()

        # Extract domain fields from JSON payload
        run_id, repo_id, event_type, event_version, occurred_at = (
            self._extract_domain_fields(message)
        )

        # Use topic as event_type if not extractable from payload
        if event_type is None:
            event_type = message.topic

        # Kafka position
        partition = message.partition if message.partition is not None else 0
        kafka_offset = self._parse_offset(message.offset)

        return ModelValidationLedgerEntry(
            id=uuid4(),
            run_id=run_id,
            repo_id=repo_id,
            event_type=event_type,
            event_version=event_version,
            occurred_at=occurred_at,
            kafka_topic=message.topic,
            kafka_partition=partition,
            kafka_offset=kafka_offset,
            envelope_bytes=envelope_bytes_b64,
            envelope_hash=envelope_hash,
            created_at=datetime.now(UTC),
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelValidationLedgerEntry]:
        """Execute validation ledger projection from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: "validation_ledger.project"
                - payload: ModelEventMessage as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelValidationLedgerEntry.
        """
        correlation_id_raw = envelope.get("correlation_id")
        correlation_id = (
            UUID(str(correlation_id_raw)) if correlation_id_raw else uuid4()
        )
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            raise RuntimeError("Missing or invalid 'payload' in envelope")

        message = ModelEventMessage.model_validate(payload_raw)
        entry = self.project(message)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VALIDATION_LEDGER_PROJECTION,
            result=entry,
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _extract_domain_fields(
        self, message: ModelEventMessage
    ) -> tuple[UUID, str, str | None, str, datetime]:
        """Extract domain fields from JSON event payload.

        Best-effort extraction: required fields (run_id, repo_id) raise
        on failure; optional fields use defensive defaults.

        Args:
            message: Event message with JSON value bytes.

        Returns:
            Tuple of (run_id, repo_id, event_type, event_version, occurred_at).

        Raises:
            OnexError: If run_id or repo_id cannot be extracted.
        """
        try:
            payload = json.loads(message.value)
        except (json.JSONDecodeError, TypeError) as e:
            raise OnexError(
                "Cannot parse validation event JSON payload. "
                f"topic={message.topic}, error={type(e).__name__}: {e}"
            ) from e

        if not isinstance(payload, dict):
            raise OnexError(
                f"Validation event payload is not a dict: {type(payload).__name__}. "
                f"topic={message.topic}"
            )

        # Required: run_id
        run_id_raw = payload.get("run_id")
        if run_id_raw is None:
            raise OnexError(
                f"Validation event missing required field 'run_id'. "
                f"topic={message.topic}"
            )
        try:
            run_id = UUID(str(run_id_raw))
        except (ValueError, TypeError) as e:
            raise OnexError(
                f"Invalid run_id '{run_id_raw}': {e}. topic={message.topic}"
            ) from e

        # Required: repo_id
        repo_id = payload.get("repo_id")
        if not repo_id or not isinstance(repo_id, str):
            raise OnexError(
                f"Validation event missing or invalid 'repo_id'. topic={message.topic}"
            )

        # Best-effort: event_type from payload, fallback to topic
        event_type = payload.get("event_type")
        if isinstance(event_type, str) and event_type:
            pass  # use extracted value
        else:
            event_type = None  # caller falls back to topic

        # Best-effort: event_version
        event_version = "v1"
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, str) and schema_version:
            event_version = schema_version

        # Best-effort: occurred_at from timestamp field
        occurred_at = datetime.now(UTC)
        timestamp_raw = payload.get("timestamp")
        if timestamp_raw is not None:
            try:
                if isinstance(timestamp_raw, str):
                    occurred_at = datetime.fromisoformat(timestamp_raw)
                elif isinstance(timestamp_raw, (int, float)):
                    occurred_at = datetime.fromtimestamp(timestamp_raw, tz=UTC)
            except (ValueError, TypeError, OSError):
                logger.warning(
                    "Failed to parse timestamp '%s', using current time. "
                    "topic=%s, run_id=%s",
                    timestamp_raw,
                    message.topic,
                    run_id,
                )

        return run_id, repo_id, event_type, event_version, occurred_at

    def _parse_offset(self, offset: str | None) -> int:
        """Parse Kafka offset string to integer.

        Args:
            offset: Offset string from Kafka, or None.

        Returns:
            Parsed offset as integer, or 0 if None or unparseable.
        """
        if offset is None:
            return 0
        try:
            return int(offset)
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse offset '%s' as integer, defaulting to 0",
                offset,
            )
            return 0


__all__ = ["HandlerValidationLedgerProjection"]
