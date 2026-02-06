# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerValidationLedgerProjection.

Tests validate:
- Valid JSON event processing (metadata extraction, base64, SHA-256)
- Malformed JSON best-effort fallback behaviour
- Empty/None value rejection (EnvelopeValidationError)
- Event version extraction from topic/event_type
- Handler property classification (handler_type, handler_category)
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection import (
    HandlerValidationLedgerProjection,
)


@pytest.fixture
def handler() -> HandlerValidationLedgerProjection:
    """Create a fresh handler instance."""
    return HandlerValidationLedgerProjection()


# ===========================================================================
# Handler Properties
# ===========================================================================


class TestHandlerProperties:
    """Tests for handler classification properties."""

    def test_handler_type_is_infra_handler(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that handler_type returns INFRA_HANDLER."""
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_nondeterministic_compute(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that handler_category returns NONDETERMINISTIC_COMPUTE."""
        assert (
            handler.handler_category == EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE
        )


# ===========================================================================
# Valid JSON Processing
# ===========================================================================


class TestValidJsonProcessing:
    """Tests for processing valid JSON events."""

    def test_extracts_run_id(self, handler: HandlerValidationLedgerProjection) -> None:
        """Test that run_id is extracted from the JSON payload."""
        run_id = str(uuid4())
        payload = json.dumps(
            {
                "run_id": run_id,
                "repo_id": "omnibase_core",
                "event_type": "onex.validation.cross_repo.run.started.v1",
                "timestamp": "2026-01-15T12:00:00+00:00",
            }
        ).encode()

        result = handler.project(
            topic="onex.validation.cross_repo.run.started.v1",
            partition=0,
            offset=42,
            value=payload,
        )

        assert result["run_id"] == UUID(run_id)

    def test_extracts_repo_id(self, handler: HandlerValidationLedgerProjection) -> None:
        """Test that repo_id is extracted from the JSON payload."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "omnibase_infra",
                "event_type": "onex.validation.cross_repo.run.started.v1",
            }
        ).encode()

        result = handler.project(
            topic="onex.validation.cross_repo.run.started.v1",
            partition=1,
            offset=10,
            value=payload,
        )

        assert result["repo_id"] == "omnibase_infra"

    def test_extracts_event_type(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that event_type is extracted from the JSON payload."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "core",
                "event_type": "custom.event.type.v2",
            }
        ).encode()

        result = handler.project(
            topic="onex.validation.cross_repo.run.started.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        assert result["event_type"] == "custom.event.type.v2"

    def test_envelope_bytes_is_base64_encoded(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that envelope_bytes is the base64-encoded form of raw value."""
        raw = b'{"run_id": "abc"}'
        expected_b64 = base64.b64encode(raw).decode("ascii")

        result = handler.project(
            topic="t",
            partition=0,
            offset=0,
            value=raw,
        )

        assert result["envelope_bytes"] == expected_b64

    def test_envelope_hash_is_sha256_hex(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that envelope_hash is the SHA-256 hex digest of raw value."""
        raw = b'{"run_id": "abc"}'
        expected_hash = hashlib.sha256(raw).hexdigest()

        result = handler.project(
            topic="t",
            partition=0,
            offset=0,
            value=raw,
        )

        assert result["envelope_hash"] == expected_hash

    def test_kafka_fields_passed_through(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that kafka_topic, kafka_partition, kafka_offset are passed through."""
        payload = json.dumps({"run_id": str(uuid4())}).encode()

        result = handler.project(
            topic="my.topic.v1",
            partition=3,
            offset=999,
            value=payload,
        )

        assert result["kafka_topic"] == "my.topic.v1"
        assert result["kafka_partition"] == 3
        assert result["kafka_offset"] == 999

    def test_extracts_timestamp_as_occurred_at(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that timestamp field is extracted as occurred_at."""
        ts = "2026-01-15T12:00:00+00:00"
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "timestamp": ts,
            }
        ).encode()

        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        assert isinstance(result["occurred_at"], datetime)
        assert result["occurred_at"].year == 2026


# ===========================================================================
# Event Version Extraction
# ===========================================================================


class TestEventVersionExtraction:
    """Tests for _extract_version_from_topic."""

    def test_version_from_event_type_v1(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test version extracted as 'v1' from event_type ending in .v1."""
        payload = json.dumps(
            {
                "event_type": "onex.validation.cross_repo.run.started.v1",
            }
        ).encode()

        result = handler.project(
            topic="onex.validation.cross_repo.run.started.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        assert result["event_version"] == "v1"

    def test_version_from_event_type_v2(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test version extracted as 'v2' from event_type ending in .v2."""
        payload = json.dumps(
            {
                "event_type": "some.event.v2",
            }
        ).encode()

        result = handler.project(
            topic="t",
            partition=0,
            offset=0,
            value=payload,
        )

        assert result["event_version"] == "v2"

    def test_version_unknown_when_no_version_suffix(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test version is 'unknown' when event_type has no version suffix."""
        payload = json.dumps(
            {
                "event_type": "some.event.without.version",
            }
        ).encode()

        result = handler.project(
            topic="t",
            partition=0,
            offset=0,
            value=payload,
        )

        assert result["event_version"] == "unknown"

    def test_version_from_topic_fallback(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test version extracted from topic when event_type not in payload."""
        payload = json.dumps({"run_id": str(uuid4())}).encode()

        result = handler.project(
            topic="onex.validation.cross_repo.run.started.v3",
            partition=0,
            offset=0,
            value=payload,
        )

        # event_type falls back to topic, version is extracted from that
        assert result["event_version"] == "v3"


# ===========================================================================
# Malformed JSON / Best-Effort Fallback
# ===========================================================================


class TestMalformedJsonFallback:
    """Tests for best-effort metadata extraction from malformed payloads."""

    def test_invalid_json_does_not_raise(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that invalid JSON bytes do NOT raise an exception."""
        raw = b"this is not json"
        result = handler.project(
            topic="my.topic.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        # Should return a valid dict with fallback values
        assert isinstance(result, dict)

    def test_invalid_json_uses_generated_uuid_for_run_id(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that malformed JSON gets a generated UUID for run_id."""
        raw = b"not-json"
        result = handler.project(
            topic="my.topic.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        assert isinstance(result["run_id"], UUID)

    def test_invalid_json_uses_unknown_for_repo_id(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that malformed JSON gets 'unknown' for repo_id."""
        raw = b"not-json"
        result = handler.project(
            topic="my.topic.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        assert result["repo_id"] == "unknown"

    def test_invalid_json_uses_topic_for_event_type(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that malformed JSON gets topic name as event_type."""
        raw = b"not-json"
        result = handler.project(
            topic="my.topic.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        assert result["event_type"] == "my.topic.v1"

    def test_invalid_json_still_produces_base64_and_hash(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that malformed JSON still produces valid base64 and hash."""
        raw = b"not-json"
        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        assert result["envelope_bytes"] == base64.b64encode(raw).decode("ascii")
        assert result["envelope_hash"] == hashlib.sha256(raw).hexdigest()

    def test_non_dict_json_uses_fallback(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that a JSON array (not dict) triggers fallback values."""
        raw = json.dumps([1, 2, 3]).encode()
        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        # Should use fallback repo_id since payload is not a dict
        assert result["repo_id"] == "unknown"

    def test_missing_fields_uses_fallback(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that missing JSON fields use fallback values."""
        raw = json.dumps({"unrelated": "field"}).encode()
        result = handler.project(
            topic="my.topic.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        # run_id should be a generated UUID
        assert isinstance(result["run_id"], UUID)
        # repo_id should be "unknown"
        assert result["repo_id"] == "unknown"
        # event_type should fall back to topic
        assert result["event_type"] == "my.topic.v1"


# ===========================================================================
# Empty / None Value Rejection
# ===========================================================================


class TestEmptyValueRejection:
    """Tests for EnvelopeValidationError on None or empty value."""

    def test_none_value_raises_envelope_validation_error(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that None value raises EnvelopeValidationError."""
        from omnibase_infra.errors import EnvelopeValidationError

        with pytest.raises(EnvelopeValidationError, match="value is None or empty"):
            handler.project(
                topic="t",
                partition=0,
                offset=0,
                value=None,  # type: ignore[arg-type]
            )

    def test_empty_bytes_raises_envelope_validation_error(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that empty bytes raises EnvelopeValidationError."""
        from omnibase_infra.errors import EnvelopeValidationError

        with pytest.raises(EnvelopeValidationError, match="value is None or empty"):
            handler.project(
                topic="t",
                partition=0,
                offset=0,
                value=b"",
            )


# ===========================================================================
# Timestamp Extraction Edge Cases
# ===========================================================================


class TestTimestampEdgeCases:
    """Tests for edge cases in timestamp extraction."""

    def test_invalid_timestamp_uses_fallback(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that an invalid timestamp falls back to current UTC time."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "timestamp": "not-a-timestamp",
            }
        ).encode()

        before = datetime.now(UTC)
        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=payload,
        )
        after = datetime.now(UTC)

        occurred_at = result["occurred_at"]
        assert isinstance(occurred_at, datetime)
        # Fallback should be approximately "now"
        assert before <= occurred_at <= after

    def test_invalid_run_id_uses_fallback_uuid(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that an invalid run_id value falls back to generated UUID."""
        payload = json.dumps(
            {
                "run_id": "not-a-uuid",
            }
        ).encode()

        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        # Should still be a valid UUID (generated fallback)
        assert isinstance(result["run_id"], UUID)
