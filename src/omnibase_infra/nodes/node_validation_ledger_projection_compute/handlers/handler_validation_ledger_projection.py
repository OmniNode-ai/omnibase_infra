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

Ticket: OMN-1908
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

logger = logging.getLogger(__name__)

# Handler ID for identification in logs and outputs
HANDLER_ID_VALIDATION_LEDGER_PROJECTION: str = "validation-ledger-projection-handler"


class HandlerValidationLedgerProjection:
    """Handler that projects validation events into ledger entry fields.

    This handler implements the compute logic for the validation ledger
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
        ...     topic="onex.evt.validation.cross-repo-run-started.v1",
        ...     partition=0,
        ...     offset=42,
        ...     value=b'{"run_id": "abc-123", "repo_id": "omnibase_core"}',
        ... )
        >>> result["event_type"]
        'onex.evt.validation.cross-repo-run-started.v1'
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
            _headers: Optional Kafka message headers (currently unused, reserved
                for future header-based routing).

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
        if not value:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="project_validation_event",
            )
            raise RuntimeHostError(
                "Cannot create validation ledger entry: value is None or empty. "
                "Raw event bytes are required for ledger persistence.",
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                context=context,
            )

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
            return datetime.fromisoformat(str(raw))
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
                (e.g., "onex.evt.validation.cross-repo-run-started.v1").

        Returns:
            Version string (e.g., "v1") or "unknown" if not found.
        """
        parts = topic.rsplit(".", maxsplit=1)
        if len(parts) == 2 and parts[1].startswith("v") and parts[1][1:].isdigit():
            return parts[1]
        return "unknown"


__all__ = ["HandlerValidationLedgerProjection"]
