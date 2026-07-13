# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for projecting validation events into ledger entries.

Pure COMPUTE handler that extracts metadata from Kafka messages and prepares
them for validation ledger persistence. Follows best-effort metadata extraction:
events are NEVER dropped due to parsing failures.

Design Rationale - Best-Effort Metadata Extraction:
    The validation ledger serves as the system's source of truth for cross-repo
    validation events. Events must NEVER be dropped due to metadata extraction
    failures. All metadata fields are extracted best-effort - parsing errors
    result in fallback values, not exceptions.

Bytes Encoding:
    Kafka event values are bytes. Raw bytes are passed through for BYTEA storage
    in PostgreSQL. A SHA-256 hash of the raw bytes is also computed for integrity
    verification and deterministic replay. Base64 encoding for the read path is
    handled at the SQL layer via encode(envelope_bytes, 'base64').

    ``handle()`` (OMN-14524) additionally base64-encodes the raw bytes into the
    ``ModelIntent`` payload it emits -- bytes cannot safely cross intent
    boundaries, the same rule ``HandlerLedgerProjection`` follows for
    ``event_value``. The Effect layer (``HandlerValidationLedgerAppend``)
    decodes it before storage.

Ticket: OMN-1908, OMN-14524
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.validation_ledger import (
    ModelPayloadValidationLedgerAppend,
)
from omnibase_infra.utils import sanitize_error_string

logger = logging.getLogger(__name__)

# Handler ID for identification in logs and outputs
HANDLER_ID_VALIDATION_LEDGER_PROJECTION: str = "validation-ledger-projection-handler"


class HandlerValidationLedgerProjection:
    """Handler that projects validation events into ledger entry fields.

    The compute logic for the validation ledger
    projection node, extracting metadata from raw Kafka messages and
    producing dictionaries matching ModelValidationLedgerEntry fields.

    CRITICAL INVARIANTS:
    - NEVER drop events due to metadata extraction failure
    - Raw bytes (value) are REQUIRED (raises RuntimeHostError if None/empty)
    - All other metadata uses best-effort extraction with fallbacks
    - No I/O operations - pure COMPUTE handler

    Attributes:
        handler_type: EnumHandlerType.INFRA_HANDLER
        handler_category: EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    Example:
        >>> handler = HandlerValidationLedgerProjection()
        >>> result = handler.project(
        ...     topic="onex.evt.validation.cross-repo-run-started.v1",  # onex-topic-allow: pending contract auto-wiring
        ...     partition=0,
        ...     offset=42,
        ...     value=b'{"run_id": "abc-123", "repo_id": "omnibase_core"}',
        ... )
        >>> result["event_type"]
        'onex.evt.validation.cross-repo-run-started.v1'  # onex-topic-allow: pending contract auto-wiring
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler for validation ledger projection.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE - This handler
            performs transformations that use nondeterministic operations
            (uuid4, datetime.now) for fallback metadata values.
        """
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    def project(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        value: bytes,
        _headers: dict[str, bytes] | None = None,
    ) -> dict[str, object]:
        """Project a Kafka message into validation ledger entry fields.

        Extracts metadata from the raw Kafka message and computes a SHA-256
        hash of the raw bytes. Raw bytes are passed through as envelope_bytes
        for BYTEA storage in PostgreSQL. Uses best-effort metadata extraction
        from the JSON payload.

        Args:
            topic: Kafka topic the message was consumed from.
            partition: Kafka partition number.
            offset: Kafka offset within the partition.
            value: Raw Kafka message value (bytes). REQUIRED.
            _headers: Optional Kafka message headers. If present, the handler
                extracts ``correlation_id`` for distributed tracing.

        Returns:
            Dict with keys matching the write-path parameters of
            ProtocolValidationLedgerRepository.append():
                - run_id: UUID (extracted or generated)
                - repo_id: str (extracted or "unknown")
                - event_type: str (extracted or from topic)
                - event_version: str (extracted or "unknown")
                - occurred_at: datetime (extracted or now UTC)
                - kafka_topic: str
                - kafka_partition: int
                - kafka_offset: int
                - envelope_bytes: bytes (raw Kafka value for BYTEA storage)
                - envelope_hash: str (SHA-256 hex digest)

        Raises:
            RuntimeHostError: If ``value`` is None or empty bytes
                (with error_code=INVALID_INPUT).

        Field Extraction Strategy:
            | Field          | Primary Source        | Fallback              |
            |----------------|-----------------------|-----------------------|
            | run_id         | payload["run_id"]     | uuid4()               |
            | repo_id        | payload["repo_id"]    | "unknown"             |
            | event_type     | payload["event_type"] | topic name            |
            | event_version  | topic suffix          | "unknown"             |
            | occurred_at    | payload["timestamp"]  | datetime.now(utc)     |
        """
        # Prefer correlation_id from Kafka headers for distributed tracing;
        # fall back to a fresh uuid4 when absent or unparseable.
        correlation_id = self._extract_header_correlation_id(_headers)

        if not value:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="project_validation_event",
            )
            raise RuntimeHostError(
                "Cannot create validation ledger entry: value is None or empty. "
                "Raw event bytes are required for ledger persistence.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

        try:
            # Compute SHA-256 hash for integrity verification
            envelope_hash = hashlib.sha256(value).hexdigest()

            # Best-effort metadata extraction from JSON payload
            run_id, repo_id, event_type, event_version, occurred_at = (
                self._extract_metadata(value, topic)
            )

            return {
                "run_id": run_id,
                "repo_id": repo_id,
                "event_type": event_type,
                "event_version": event_version,
                "occurred_at": occurred_at,
                "kafka_topic": topic,
                "kafka_partition": partition,
                "kafka_offset": offset,
                "envelope_bytes": value,
                "envelope_hash": envelope_hash,
            }
        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="project_validation_event",
            )
            raise RuntimeHostError(
                f"Unexpected error during validation ledger projection: "
                f"{sanitize_error_string(str(e))}",
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                context=context,
            ) from e

    async def handle(
        self,
        message: object,
    ) -> ModelHandlerOutput[ModelIntent]:
        """Contract-typed auto-wiring entry point (OMN-14524).

        ``node_validation_ledger_projection_compute``'s contract declares
        ``operation_match`` routing with no ``event_model``, so auto-wiring
        (``handler_wiring._make_dispatch_callback``) invokes
        ``handle(envelope)`` directly rather than ``execute()``. Without this
        method the callback binds ``_missing_handle``, which raises on every
        dispatched event -- and the auto-wired consume boundary
        log-and-discards that exception, so events are acked and the
        validation ledger stays empty with nothing surfaced. This mirrors the
        exact OMN-14516 defect class for the sibling event_ledger.

        Unlike ``project()`` (which returns a plain dict of extracted
        metadata for existing callers), ``handle()`` wraps that dict into a
        ``ModelPayloadValidationLedgerAppend`` and emits a ``ModelIntent`` --
        the shape the kernel's generic intent-routing derivation (OMN-14516)
        requires to route to ``node_validation_ledger_write_effect`` via
        ``intent_consumption.intent_routing_table``.

        The value delivered here is whatever
        ``MessageDispatchEngine._materialize_envelope_with_bindings`` produces
        on the live dispatch path -- a **dict** (``{"payload": ..., ...}``),
        not an attribute-bearing envelope. ``_coerce_event_message`` accepts
        both shapes so this works for the real runtime path and for
        object-shaped envelopes.
        """
        raw_message = self._coerce_event_message(message)
        headers = raw_message.headers
        correlation_id = headers.correlation_id if headers else None
        header_bytes: dict[str, bytes] | None = (
            {"correlation_id": str(correlation_id).encode("utf-8")}
            if correlation_id is not None
            else None
        )

        projected = self.project(
            topic=raw_message.topic,
            partition=raw_message.partition if raw_message.partition is not None else 0,
            offset=self._parse_offset(raw_message.offset),
            value=raw_message.value or b"",
            _headers=header_bytes,
        )

        envelope_bytes_raw = projected["envelope_bytes"]
        assert isinstance(envelope_bytes_raw, bytes)
        # project() returns dict[str, object] since its callers only need key
        # lookup; handle() knows the concrete per-key types project() actually
        # populates (see its Returns: docstring), so narrow via cast() rather
        # than a blanket type: ignore per field.
        payload = ModelPayloadValidationLedgerAppend(
            run_id=cast("UUID", projected["run_id"]),
            repo_id=cast("str", projected["repo_id"]),
            event_type=cast("str", projected["event_type"]),
            event_version=cast("str", projected["event_version"]),
            occurred_at=cast("datetime", projected["occurred_at"]),
            kafka_topic=cast("str", projected["kafka_topic"]),
            kafka_partition=cast("int", projected["kafka_partition"]),
            kafka_offset=cast("int", projected["kafka_offset"]),
            envelope_bytes=base64.b64encode(envelope_bytes_raw).decode("ascii"),
            envelope_hash=cast("str", projected["envelope_hash"]),
            correlation_id=correlation_id,
        )
        intent = ModelIntent(
            intent_type=payload.intent_type,
            target=(
                f"postgres://validation_event_ledger/{payload.kafka_topic}/"
                f"{payload.kafka_partition}/{payload.kafka_offset}"
            ),
            payload=payload,
        )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id if correlation_id is not None else uuid4(),
            handler_id=HANDLER_ID_VALIDATION_LEDGER_PROJECTION,
            result=intent,
        )

    @staticmethod
    def _coerce_event_message(raw: object) -> ModelEventMessage:
        """Accept a direct ModelEventMessage or an auto-wired envelope wrapper.

        The dispatch engine delivers a materialized dict on the live runtime
        path and a ``ModelEventEnvelope``-like object from the auto_wiring
        event-bus callback and test doubles; both must work here. Mirrors
        ``HandlerLedgerProjection._coerce_event_message``.
        """
        if isinstance(raw, ModelEventMessage):
            return raw
        payload = (
            raw.get("payload", raw)
            if isinstance(raw, dict)
            else getattr(raw, "payload", raw)
        )
        if isinstance(payload, ModelEventMessage):
            return payload
        return ModelEventMessage.model_validate(payload)

    @staticmethod
    def _parse_offset(offset: str | None) -> int:
        """Parse a Kafka offset string (as carried by ModelEventMessage) to int.

        Returns 0 if None or unparseable -- best-effort, mirroring
        ``HandlerLedgerProjection._parse_offset``.
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

    @staticmethod
    def _extract_header_correlation_id(
        headers: dict[str, bytes] | None,
    ) -> UUID:
        """Extract correlation_id from Kafka headers, falling back to uuid4.

        Kafka messages produced by the ONEX event bus include a
        ``correlation_id`` header (UTF-8 encoded UUID string). When
        present and valid, the handler reuses this ID so that the entire
        processing chain shares a single trace. If the header is absent,
        empty, or contains an invalid UUID, a fresh uuid4 is generated.

        Args:
            headers: Kafka message headers as ``{key: value_bytes}``,
                or None when no headers were delivered.

        Returns:
            Parsed UUID from the header, or a newly generated UUID.
        """
        if headers is None:
            return uuid4()
        raw = headers.get("correlation_id")
        if raw is None:
            return uuid4()
        try:
            return UUID(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, AttributeError):
            logger.warning(
                "Failed to parse correlation_id from Kafka header "
                "(raw=%r), using generated fallback",
                raw,
            )
            return uuid4()

    def _extract_metadata(
        self,
        value: bytes,
        topic: str,
    ) -> tuple[UUID, str, str, str, datetime]:
        """Extract metadata fields from raw Kafka message bytes.

        Best-effort extraction: if JSON parsing fails or fields are missing,
        fallback values are used. Events are NEVER dropped.

        Args:
            value: Raw Kafka message bytes.
            topic: Kafka topic name (used as fallback for event_type
                and for extracting event_version).

        Returns:
            Tuple of (run_id, repo_id, event_type, event_version, occurred_at).
        """
        run_id: UUID = uuid4()
        repo_id: str = "unknown"
        event_type: str = topic
        event_version: str = self._extract_version_from_topic(topic)
        occurred_at: datetime = datetime.now(UTC)

        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(
                "Failed to JSON-decode validation event from topic=%s, "
                "using fallback metadata values",
                topic,
                exc_info=True,
            )
            return run_id, repo_id, event_type, event_version, occurred_at

        if not isinstance(payload, dict):
            logger.warning(
                "Validation event payload is not a dict (type=%s) from "
                "topic=%s, using fallback metadata values",
                type(payload).__name__,
                topic,
            )
            return run_id, repo_id, event_type, event_version, occurred_at

        # Extract run_id
        run_id = self._extract_uuid(payload, "run_id", default=run_id)

        # Extract repo_id
        raw_repo_id = payload.get("repo_id")
        if isinstance(raw_repo_id, str) and raw_repo_id:
            repo_id = raw_repo_id

        # Extract event_type
        raw_event_type = payload.get("event_type")
        if isinstance(raw_event_type, str) and raw_event_type:
            event_type = raw_event_type

        # Extract event_version from event_type or topic
        event_version = self._extract_version_from_topic(event_type)

        # Extract occurred_at from timestamp field
        occurred_at = self._extract_timestamp(payload, "timestamp", default=occurred_at)

        return run_id, repo_id, event_type, event_version, occurred_at

    def _extract_uuid(
        self,
        payload: dict[str, object],
        key: str,
        default: UUID,
    ) -> UUID:
        """Extract a UUID field from payload with fallback.

        Args:
            payload: Parsed JSON payload dict.
            key: Key to extract from payload.
            default: Fallback UUID if extraction fails.

        Returns:
            Extracted UUID or default.
        """
        raw = payload.get(key)
        if raw is None:
            return default
        try:
            return UUID(str(raw))
        except (ValueError, AttributeError):
            logger.warning(
                "Failed to parse '%s' as UUID from validation event "
                "(value=%r), using generated fallback",
                key,
                raw,
            )
            return default

    def _extract_timestamp(
        self,
        payload: dict[str, object],
        key: str,
        default: datetime,
    ) -> datetime:
        """Extract an ISO-8601 timestamp from payload with fallback.

        Args:
            payload: Parsed JSON payload dict.
            key: Key to extract from payload.
            default: Fallback datetime if extraction fails.

        Returns:
            Extracted datetime or default.
        """
        raw = payload.get(key)
        if raw is None:
            return default
        try:
            parsed = datetime.fromisoformat(str(raw))
            # Ensure timezone-aware: naive ISO strings (no tz suffix) get UTC
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse '%s' as ISO timestamp from validation "
                "event (value=%r), using current UTC time as fallback",
                key,
                raw,
            )
            return default

    @staticmethod
    def _extract_version_from_topic(topic: str) -> str:
        """Extract version suffix from a topic or event_type string.

        Looks for a trailing version segment like ".v1", ".v2", etc.

        Args:
            topic: Topic or event_type string
                (e.g., "onex.evt.validation.cross-repo-run-started.v1").  # onex-topic-allow: pending contract auto-wiring

        Returns:
            Version string (e.g., "v1") or "unknown" if not found.
        """
        parts = topic.rsplit(".", maxsplit=1)
        if len(parts) == 2 and parts[1].startswith("v") and parts[1][1:].isdigit():
            return parts[1]
        return "unknown"


__all__ = ["HandlerValidationLedgerProjection"]
