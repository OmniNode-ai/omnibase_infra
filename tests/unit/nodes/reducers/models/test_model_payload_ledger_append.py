# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelPayloadLedgerAppend intent payload model.

This module validates the ModelPayloadLedgerAppend model used for audit ledger
append intents emitted by reducers and processed by the ledger write effect node.

Test Coverage:
    - Required fields: topic, partition, kafka_offset, event_value
    - Optional fields: event_key, correlation_id, envelope_id, event_type, source, event_timestamp
    - intent_type is always "ledger.append"
    - Field constraints (min_length, ge, etc.)
    - Immutability (frozen=True)
    - Serialization and round-trip

Related:
    - OMN-1646: Event Ledger Schema and Models
    - PR #208: Add event ledger schema and models
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

# =============================================================================
# ModelPayloadLedgerAppend Construction Tests
# =============================================================================


class TestModelPayloadLedgerAppendConstruction:
    """Tests for ModelPayloadLedgerAppend construction and validation."""

    def test_create_minimal_payload(self) -> None:
        """Create payload with only required fields."""
        payload = ModelPayloadLedgerAppend(
            topic="test.events",
            partition=0,
            kafka_offset=100,
            event_value="SGVsbG8gV29ybGQ=",  # base64 for "Hello World"
        )

        assert payload.topic == "test.events"
        assert payload.partition == 0
        assert payload.kafka_offset == 100
        assert payload.event_value == "SGVsbG8gV29ybGQ="
        assert payload.intent_type == "ledger.append"

    def test_create_full_payload(self) -> None:
        """Create payload with all fields populated."""
        correlation_id = uuid4()
        envelope_id = uuid4()
        event_timestamp = datetime.now(UTC)

        payload = ModelPayloadLedgerAppend(
            topic="domain.service.event.v1",
            partition=5,
            kafka_offset=99999,
            event_key="dXNlci0xMjM=",  # base64 for "user-123"
            event_value="eyJldmVudCI6ICJkYXRhIn0=",  # base64 for JSON
            onex_headers={"x-onex-trace-id": "abc123"},
            correlation_id=correlation_id,
            envelope_id=envelope_id,
            event_type="user.created",
            source="user-service",
            event_timestamp=event_timestamp,
        )

        assert payload.topic == "domain.service.event.v1"
        assert payload.partition == 5
        assert payload.kafka_offset == 99999
        assert payload.event_key == "dXNlci0xMjM="
        assert payload.event_value == "eyJldmVudCI6ICJkYXRhIn0="
        assert payload.onex_headers == {"x-onex-trace-id": "abc123"}
        assert payload.correlation_id == correlation_id
        assert payload.envelope_id == envelope_id
        assert payload.event_type == "user.created"
        assert payload.source == "user-service"
        assert payload.event_timestamp == event_timestamp
        assert payload.intent_type == "ledger.append"

    def test_optional_fields_default_correctly(self) -> None:
        """Optional fields have correct default values."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        # All nullable fields should default to None
        assert payload.event_key is None
        assert payload.correlation_id is None
        assert payload.envelope_id is None
        assert payload.event_type is None
        assert payload.source is None
        assert payload.event_timestamp is None
        # Dict field defaults to empty dict
        assert payload.onex_headers == {}


class TestModelPayloadLedgerAppendIntentType:
    """Tests for intent_type discriminator field."""

    def test_intent_type_default_is_ledger_append(self) -> None:
        """intent_type defaults to 'ledger.append'."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        assert payload.intent_type == "ledger.append"

    def test_intent_type_cannot_be_changed(self) -> None:
        """intent_type is a Literal and cannot be set to other values."""
        # The Literal type enforces the value at construction time
        # Attempting to pass a different value should fail validation
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadLedgerAppend(
                intent_type="wrong.type",  # type: ignore[arg-type]
                topic="test",
                partition=0,
                kafka_offset=0,
                event_value="dGVzdA==",
            )

        errors = exc_info.value.errors()
        assert any("intent_type" in str(e) for e in errors)


# =============================================================================
# ModelPayloadLedgerAppend Validation Tests
# =============================================================================


class TestModelPayloadLedgerAppendValidation:
    """Tests for ModelPayloadLedgerAppend validation constraints."""

    def test_topic_cannot_be_empty(self) -> None:
        """topic must have at least 1 character (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadLedgerAppend(
                topic="",  # Empty - invalid
                partition=0,
                kafka_offset=0,
                event_value="dGVzdA==",
            )

        errors = exc_info.value.errors()
        assert any("topic" in str(e) for e in errors)

    def test_event_value_cannot_be_empty(self) -> None:
        """event_value must have at least 1 character (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadLedgerAppend(
                topic="test",
                partition=0,
                kafka_offset=0,
                event_value="",  # Empty - invalid
            )

        errors = exc_info.value.errors()
        assert any("event_value" in str(e) for e in errors)

    def test_partition_cannot_be_negative(self) -> None:
        """partition must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadLedgerAppend(
                topic="test",
                partition=-1,  # Invalid
                kafka_offset=0,
                event_value="dGVzdA==",
            )

        errors = exc_info.value.errors()
        assert any("partition" in str(e) for e in errors)

    def test_kafka_offset_cannot_be_negative(self) -> None:
        """kafka_offset must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPayloadLedgerAppend(
                topic="test",
                partition=0,
                kafka_offset=-1,  # Invalid
                event_value="dGVzdA==",
            )

        errors = exc_info.value.errors()
        assert any("kafka_offset" in str(e) for e in errors)

    def test_partition_at_zero_boundary(self) -> None:
        """partition accepts zero (boundary value)."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        assert payload.partition == 0

    def test_kafka_offset_at_zero_boundary(self) -> None:
        """kafka_offset accepts zero (boundary value)."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        assert payload.kafka_offset == 0

    def test_large_partition_and_offset_values(self) -> None:
        """Large partition and offset values are accepted."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=999999,
            kafka_offset=9999999999999,
            event_value="dGVzdA==",
        )

        assert payload.partition == 999999
        assert payload.kafka_offset == 9999999999999


# =============================================================================
# ModelPayloadLedgerAppend Immutability Tests
# =============================================================================


class TestModelPayloadLedgerAppendImmutability:
    """Tests for ModelPayloadLedgerAppend immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """ModelPayloadLedgerAppend is immutable - assignment raises error."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        with pytest.raises((TypeError, ValidationError)):
            payload.topic = "new-topic"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelPayloadLedgerAppend(
                topic="test",
                partition=0,
                kafka_offset=0,
                event_value="dGVzdA==",
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# Serialization Tests
# =============================================================================


class TestModelPayloadLedgerAppendSerialization:
    """Tests for ModelPayloadLedgerAppend serialization."""

    def test_payload_to_dict(self) -> None:
        """ModelPayloadLedgerAppend serializes to dict correctly."""
        correlation_id = uuid4()

        payload = ModelPayloadLedgerAppend(
            topic="test.events",
            partition=3,
            kafka_offset=42,
            event_value="dGVzdA==",
            correlation_id=correlation_id,
        )

        data = payload.model_dump()

        assert data["topic"] == "test.events"
        assert data["partition"] == 3
        assert data["kafka_offset"] == 42
        assert data["event_value"] == "dGVzdA=="
        assert data["correlation_id"] == correlation_id
        assert data["intent_type"] == "ledger.append"

    def test_payload_json_round_trip(self) -> None:
        """ModelPayloadLedgerAppend survives JSON round-trip."""
        correlation_id = uuid4()
        envelope_id = uuid4()
        event_timestamp = datetime.now(UTC)

        payload = ModelPayloadLedgerAppend(
            topic="domain.service.event.v1",
            partition=5,
            kafka_offset=99999,
            event_key="a2V5",
            event_value="dmFsdWU=",
            onex_headers={"trace": "123"},
            correlation_id=correlation_id,
            envelope_id=envelope_id,
            event_type="test.event",
            source="test-service",
            event_timestamp=event_timestamp,
        )

        json_str = payload.model_dump_json()
        restored = ModelPayloadLedgerAppend.model_validate_json(json_str)

        assert restored.topic == payload.topic
        assert restored.partition == payload.partition
        assert restored.kafka_offset == payload.kafka_offset
        assert restored.event_key == payload.event_key
        assert restored.event_value == payload.event_value
        assert restored.correlation_id == payload.correlation_id
        assert restored.envelope_id == payload.envelope_id
        assert restored.event_type == payload.event_type
        assert restored.source == payload.source
        assert restored.intent_type == "ledger.append"

    def test_intent_type_serializes_correctly(self) -> None:
        """intent_type serializes as 'ledger.append' string."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        json_str = payload.model_dump_json()
        assert '"ledger.append"' in json_str


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestModelPayloadLedgerAppendEdgeCases:
    """Edge case tests for ModelPayloadLedgerAppend."""

    def test_payload_with_empty_onex_headers(self) -> None:
        """Empty onex_headers dict is valid."""
        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            onex_headers={},
        )

        assert payload.onex_headers == {}

    def test_payload_with_complex_onex_headers(self) -> None:
        """Complex nested onex_headers are accepted."""
        headers = {
            "x-trace-id": "abc123",
            "x-correlation-ids": ["id1", "id2"],
            "x-metadata": {"region": "us-east", "version": 2},
        }

        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            onex_headers=headers,
        )

        assert payload.onex_headers == headers

    def test_payload_with_very_long_topic(self) -> None:
        """Long topic names are accepted."""
        long_topic = "domain.subdomain.service.event.namespace.version.v1"

        payload = ModelPayloadLedgerAppend(
            topic=long_topic,
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        assert payload.topic == long_topic

    def test_payload_with_large_event_value(self) -> None:
        """Large event_value strings are accepted."""
        # Simulate a large base64 encoded payload
        large_value = "A" * 10000

        payload = ModelPayloadLedgerAppend(
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value=large_value,
        )

        assert len(payload.event_value) == 10000

    def test_payload_with_single_character_values(self) -> None:
        """Single character topic and event_value are valid (min_length=1)."""
        payload = ModelPayloadLedgerAppend(
            topic="t",
            partition=0,
            kafka_offset=0,
            event_value="x",
        )

        assert payload.topic == "t"
        assert payload.event_value == "x"

    def test_from_attributes_config(self) -> None:
        """Model supports from_attributes=True for ORM compatibility."""

        # Create a mock object with attributes matching the model fields
        class MockRecord:
            topic = "test"
            partition = 0
            kafka_offset = 0
            event_value = "dGVzdA=="
            event_key = None
            onex_headers = {}
            correlation_id = None
            envelope_id = None
            event_type = None
            source = None
            event_timestamp = None

        mock = MockRecord()
        payload = ModelPayloadLedgerAppend.model_validate(mock, from_attributes=True)

        assert payload.topic == "test"
        assert payload.partition == 0
        assert payload.kafka_offset == 0
        assert payload.event_value == "dGVzdA=="


__all__: list[str] = [
    "TestModelPayloadLedgerAppendConstruction",
    "TestModelPayloadLedgerAppendIntentType",
    "TestModelPayloadLedgerAppendValidation",
    "TestModelPayloadLedgerAppendImmutability",
    "TestModelPayloadLedgerAppendSerialization",
    "TestModelPayloadLedgerAppendEdgeCases",
]
