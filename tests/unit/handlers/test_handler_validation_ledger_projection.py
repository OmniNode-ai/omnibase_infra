# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerValidationLedgerProjection.

Tests validate:
- Valid JSON event processing (metadata extraction, raw bytes, SHA-256)
- Malformed JSON best-effort fallback behaviour
- Empty/None value rejection (RuntimeHostError with INVALID_INPUT)
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

pytestmark = [pytest.mark.unit]

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.validation_ledger import (
    ModelPayloadValidationLedgerAppend,
)
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
                "event_type": "onex.evt.validation.cross-repo-run-started.v1",
                "timestamp": "2026-01-15T12:00:00+00:00",
            }
        ).encode()

        result = handler.project(
            topic="onex.evt.validation.cross-repo-run-started.v1",
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
                "event_type": "onex.evt.validation.cross-repo-run-started.v1",
            }
        ).encode()

        result = handler.project(
            topic="onex.evt.validation.cross-repo-run-started.v1",
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
            topic="onex.evt.validation.cross-repo-run-started.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        assert result["event_type"] == "custom.event.type.v2"

    def test_envelope_bytes_is_raw_bytes(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that envelope_bytes is the raw value bytes for BYTEA storage."""
        raw = b'{"run_id": "abc"}'

        result = handler.project(
            topic="t",
            partition=0,
            offset=0,
            value=raw,
        )

        assert result["envelope_bytes"] == raw
        assert isinstance(result["envelope_bytes"], bytes)

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
                "event_type": "onex.evt.validation.cross-repo-run-started.v1",
            }
        ).encode()

        result = handler.project(
            topic="onex.evt.validation.cross-repo-run-started.v1",
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
            topic="onex.evt.validation.cross-repo-run-started.v3",
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

    def test_invalid_json_still_produces_raw_bytes_and_hash(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that malformed JSON still produces raw bytes and valid hash."""
        raw = b"not-json"
        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=raw,
        )
        assert result["envelope_bytes"] == raw
        assert isinstance(result["envelope_bytes"], bytes)
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
    """Tests for RuntimeHostError on None or empty value."""

    def test_none_value_raises_runtime_host_error(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that None value raises RuntimeHostError with INVALID_INPUT."""
        from omnibase_infra.errors import RuntimeHostError

        with pytest.raises(RuntimeHostError, match="value is None or empty"):
            handler.project(
                topic="t",
                partition=0,
                offset=0,
                value=None,  # type: ignore[arg-type]
            )

    def test_empty_bytes_raises_runtime_host_error(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that empty bytes raises RuntimeHostError with INVALID_INPUT."""
        from omnibase_infra.errors import RuntimeHostError

        with pytest.raises(RuntimeHostError, match="value is None or empty"):
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

    def test_naive_timestamp_gets_utc_timezone(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Test that a naive ISO timestamp (no tz suffix) is assigned UTC.

        The _extract_timestamp method adds UTC when parsed.tzinfo is None,
        ensuring occurred_at is always timezone-aware for TIMESTAMPTZ storage.
        """
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "timestamp": "2026-01-15T12:00:00",
            }
        ).encode()

        result = handler.project(
            topic="t.v1",
            partition=0,
            offset=0,
            value=payload,
        )

        occurred_at = result["occurred_at"]
        assert isinstance(occurred_at, datetime)
        assert occurred_at.tzinfo is not None
        assert occurred_at.tzinfo == UTC
        assert occurred_at.year == 2026
        assert occurred_at.month == 1
        assert occurred_at.day == 15
        assert occurred_at.hour == 12
        assert occurred_at.minute == 0

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


# ===========================================================================
# handle() -- OMN-14524 dispatch entrypoint
# ===========================================================================


class TestHandleEmitsValidationLedgerAppendIntent:
    """Tests for handle(), the contract-typed auto-wiring entry point.

    Mirrors HandlerLedgerProjection's equivalent coverage
    (test_handle_projects_dict_envelope_to_ledger_append_intent) -- these
    tests exist because a green ``project()`` suite gave zero signal that
    validation_event_ledger was structurally unable to receive rows (OMN-14524):
    project() was never reachable from the live dispatch path at all.
    """

    @pytest.mark.asyncio
    async def test_handle_projects_dict_envelope_to_intent(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Drive handle() with the DICT envelope the live dispatch path delivers.

        ``MessageDispatchEngine._materialize_envelope_with_bindings`` hands the
        handler a dict, not an attribute-bearing envelope -- the OMN-14140 trap.
        """
        correlation_id = uuid4()
        raw_value = json.dumps(
            {
                "run_id": "8e6a1f0e-2222-4444-8888-000000000001",
                "repo_id": "omnibase_core",
                "event_type": "onex.evt.validation.cross-repo-run-started.v1",
            }
        ).encode()
        message = ModelEventMessage(
            topic="onex.evt.validation.cross-repo-run-started.v1",
            value=raw_value,
            headers=ModelEventHeaders(
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC),
                source="test-producer",
                event_type="ValidationRunStarted",
            ),
            partition=2,
            offset="17",
        )

        output = await handler.handle({"payload": message.model_dump()})

        intent = output.result
        assert isinstance(intent, ModelIntent)
        assert isinstance(intent.payload, ModelPayloadValidationLedgerAppend)
        # intent_type is the routing key the kernel's IntentExecutor dispatches
        # on -- it must match node_validation_ledger_write_effect's
        # "validation_ledger.append" operation exactly, or the intent is
        # produced and then dropped on the floor.
        assert intent.payload.intent_type == "validation_ledger.append"
        assert intent.payload.kafka_topic == (
            "onex.evt.validation.cross-repo-run-started.v1"
        )
        assert intent.payload.kafka_partition == 2
        assert intent.payload.kafka_offset == 17
        assert intent.payload.correlation_id == correlation_id
        assert str(intent.payload.run_id) == "8e6a1f0e-2222-4444-8888-000000000001"
        assert intent.payload.repo_id == "omnibase_core"
        # envelope_bytes is base64-encoded at this boundary -- bytes cannot
        # safely cross intent boundaries; the Effect layer decodes it.
        assert intent.payload.envelope_bytes == base64.b64encode(raw_value).decode(
            "ascii"
        )
        assert intent.payload.envelope_hash == hashlib.sha256(raw_value).hexdigest()

    @pytest.mark.asyncio
    async def test_handle_uses_header_correlation_id_for_output_and_payload(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """output.correlation_id and payload.correlation_id must be the SAME id.

        ModelEventHeaders.correlation_id always defaults to a fresh UUID
        (default_factory=uuid4) -- it is never None on a real envelope -- so
        this asserts the one degradation path that IS reachable: the header's
        id is threaded through consistently to both the handler output and
        the emitted intent payload, never silently regenerated in one place
        and not the other.
        """
        message = ModelEventMessage(
            topic="onex.evt.validation.cross-repo-run-completed.v1",
            value=b"{}",
            headers=ModelEventHeaders(
                timestamp=datetime.now(UTC),
                source="test-producer",
                event_type="ValidationRunCompleted",
            ),
            partition=0,
            offset="0",
        )

        output = await handler.handle({"payload": message.model_dump()})

        assert isinstance(output.result, ModelIntent)
        assert output.correlation_id == output.result.payload.correlation_id

    def test_handler_exposes_dispatch_entrypoint(self) -> None:
        """Without handle(), auto-wiring binds _missing_handle and every event raises.

        Mirrors HandlerLedgerProjection's equivalent pin -- absence here is an
        unambiguous defect (OMN-14524).
        """
        assert callable(getattr(HandlerValidationLedgerProjection, "handle", None)), (
            "HandlerValidationLedgerProjection must expose a handle() dispatch "
            "entrypoint; without it handler_wiring binds the _missing_handle "
            "sentinel, which raises on every dispatched event and is then "
            "swallowed by the consume boundary (events acked, "
            "validation_event_ledger empty, nothing surfaced)."
        )
