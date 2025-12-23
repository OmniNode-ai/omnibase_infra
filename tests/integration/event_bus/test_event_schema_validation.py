# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for event schema validation.

These tests validate that event schemas (ModelEventMessage, ModelEventHeaders)
correctly enforce validation rules and maintain data integrity across the
event bus publish/subscribe cycle.

Test categories:
- ModelEventMessage Validation: Field validation, immutability, required fields
- ModelEventHeaders Validation: Header field validation, defaults, constraints
- Invalid Schema Rejection: Pydantic validation error handling
- Header Completeness: Ensure all required headers are present and valid
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

# =============================================================================
# Test Configuration
# =============================================================================

pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_headers() -> dict[str, object]:
    """Create sample valid headers for testing."""
    from omnibase_infra.event_bus.models import ModelEventHeaders

    return {
        "source": "test-service",
        "event_type": "test.event.created",
        "correlation_id": uuid4(),
        "trace_id": "trace-123",
        "span_id": "span-456",
        "priority": "normal",
        "schema_version": "1.0.0",
    }


@pytest.fixture
async def started_event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Provide a started InMemoryEventBus instance."""
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

    bus = InMemoryEventBus(environment="test", group="schema-validation")
    await bus.start()
    yield bus
    await bus.close()


# =============================================================================
# ModelEventHeaders Validation Tests
# =============================================================================


class TestModelEventHeadersValidation:
    """Tests for ModelEventHeaders schema validation."""

    def test_headers_with_required_fields_only(self) -> None:
        """Verify headers can be created with only required fields."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
        )

        assert headers.source == "test-service"
        assert headers.event_type == "test.event"
        # Default values should be set
        assert headers.content_type == "application/json"
        assert headers.priority == "normal"
        assert headers.retry_count == 0
        assert headers.max_retries == 3
        assert headers.schema_version == "1.0.0"

    def test_headers_with_all_fields(self) -> None:
        """Verify headers accept all valid fields."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        correlation_id = uuid4()
        message_id = uuid4()
        timestamp = datetime.now(UTC)

        headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event.v1",
            content_type="application/json",
            correlation_id=correlation_id,
            message_id=message_id,
            timestamp=timestamp,
            schema_version="2.0.0",
            destination="target-service",
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="parent-span-789",
            operation_name="process_order",
            priority="high",
            routing_key="orders.us-east",
            partition_key="customer-123",
            retry_count=2,
            max_retries=5,
            ttl_seconds=3600,
        )

        assert headers.correlation_id == correlation_id
        assert headers.message_id == message_id
        assert headers.timestamp == timestamp
        assert headers.schema_version == "2.0.0"
        assert headers.destination == "target-service"
        assert headers.trace_id == "trace-123"
        assert headers.span_id == "span-456"
        assert headers.parent_span_id == "parent-span-789"
        assert headers.operation_name == "process_order"
        assert headers.priority == "high"
        assert headers.routing_key == "orders.us-east"
        assert headers.partition_key == "customer-123"
        assert headers.retry_count == 2
        assert headers.max_retries == 5
        assert headers.ttl_seconds == 3600

    def test_headers_default_correlation_id_generation(self) -> None:
        """Verify correlation_id is auto-generated when not provided."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers1 = ModelEventHeaders(source="test", event_type="event")
        headers2 = ModelEventHeaders(source="test", event_type="event")

        # Each should have unique auto-generated correlation_id
        assert headers1.correlation_id is not None
        assert headers2.correlation_id is not None
        assert headers1.correlation_id != headers2.correlation_id
        assert isinstance(headers1.correlation_id, UUID)

    def test_headers_default_message_id_generation(self) -> None:
        """Verify message_id is auto-generated when not provided."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers1 = ModelEventHeaders(source="test", event_type="event")
        headers2 = ModelEventHeaders(source="test", event_type="event")

        # Each should have unique auto-generated message_id
        assert headers1.message_id is not None
        assert headers2.message_id is not None
        assert headers1.message_id != headers2.message_id
        assert isinstance(headers1.message_id, UUID)

    def test_headers_default_timestamp_generation(self) -> None:
        """Verify timestamp is auto-generated when not provided."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        before = datetime.now(UTC)
        headers = ModelEventHeaders(source="test", event_type="event")
        after = datetime.now(UTC)

        assert headers.timestamp is not None
        assert before <= headers.timestamp <= after

    def test_headers_priority_validation(self) -> None:
        """Verify priority field only accepts valid values."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        # Valid priorities
        for priority in ["low", "normal", "high", "critical"]:
            headers = ModelEventHeaders(
                source="test",
                event_type="event",
                priority=priority,
            )
            assert headers.priority == priority

        # Invalid priority should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ModelEventHeaders(
                source="test",
                event_type="event",
                priority="invalid",
            )
        assert "priority" in str(exc_info.value)

    def test_headers_immutability(self) -> None:
        """Verify headers are immutable after creation."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
        )

        with pytest.raises(ValidationError):
            headers.source = "modified-service"

    def test_headers_no_extra_fields_allowed(self) -> None:
        """Verify extra fields are rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        with pytest.raises(ValidationError) as exc_info:
            ModelEventHeaders(
                source="test",
                event_type="event",
                unknown_field="value",
            )
        assert "extra" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_headers_validate_method(self) -> None:
        """Verify validate_headers() method works correctly."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        # Valid headers
        valid_headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
        )
        assert await valid_headers.validate_headers() is True

        # Headers with correlation_id should also be valid
        headers_with_corr = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            correlation_id=uuid4(),
        )
        assert await headers_with_corr.validate_headers() is True


# =============================================================================
# ModelEventMessage Validation Tests
# =============================================================================


class TestModelEventMessageValidation:
    """Tests for ModelEventMessage schema validation."""

    def test_message_with_required_fields(self) -> None:
        """Verify message can be created with required fields."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")
        message = ModelEventMessage(
            topic="test.topic",
            value=b"test-value",
            headers=headers,
        )

        assert message.topic == "test.topic"
        assert message.value == b"test-value"
        assert message.headers == headers
        assert message.key is None
        assert message.offset is None
        assert message.partition is None

    def test_message_with_all_fields(self) -> None:
        """Verify message accepts all valid fields."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")
        message = ModelEventMessage(
            topic="test.topic.v1",
            key=b"message-key",
            value=b'{"data": "value"}',
            headers=headers,
            offset="12345",
            partition=2,
        )

        assert message.topic == "test.topic.v1"
        assert message.key == b"message-key"
        assert message.value == b'{"data": "value"}'
        assert message.headers == headers
        assert message.offset == "12345"
        assert message.partition == 2

    def test_message_immutability(self) -> None:
        """Verify message is immutable after creation."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")
        message = ModelEventMessage(
            topic="test.topic",
            value=b"test-value",
            headers=headers,
        )

        with pytest.raises(ValidationError):
            message.topic = "modified.topic"

    def test_message_no_extra_fields_allowed(self) -> None:
        """Verify extra fields are rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")

        with pytest.raises(ValidationError) as exc_info:
            ModelEventMessage(
                topic="test.topic",
                value=b"test-value",
                headers=headers,
                unknown_field="value",
            )
        assert "extra" in str(exc_info.value).lower()

    def test_message_requires_headers(self) -> None:
        """Verify message requires headers field."""
        from omnibase_infra.event_bus.models import ModelEventMessage

        with pytest.raises(ValidationError) as exc_info:
            ModelEventMessage(
                topic="test.topic",
                value=b"test-value",
            )
        assert "headers" in str(exc_info.value).lower()

    def test_message_requires_topic(self) -> None:
        """Verify message requires topic field."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")

        with pytest.raises(ValidationError) as exc_info:
            ModelEventMessage(
                value=b"test-value",
                headers=headers,
            )
        assert "topic" in str(exc_info.value).lower()

    def test_message_requires_value(self) -> None:
        """Verify message requires value field."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")

        with pytest.raises(ValidationError) as exc_info:
            ModelEventMessage(
                topic="test.topic",
                headers=headers,
            )
        assert "value" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_message_ack_method(self) -> None:
        """Verify ack() method exists and is callable."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")
        message = ModelEventMessage(
            topic="test.topic",
            value=b"test-value",
            headers=headers,
        )

        # ack() should not raise (no-op for in-memory)
        await message.ack()


# =============================================================================
# Invalid Schema Rejection Tests
# =============================================================================


class TestInvalidSchemaRejection:
    """Tests for proper rejection of invalid schemas."""

    def test_headers_missing_source_rejected(self) -> None:
        """Verify headers without source are rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        with pytest.raises(ValidationError) as exc_info:
            ModelEventHeaders(event_type="test.event")
        assert "source" in str(exc_info.value).lower()

    def test_headers_missing_event_type_rejected(self) -> None:
        """Verify headers without event_type are rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        with pytest.raises(ValidationError) as exc_info:
            ModelEventHeaders(source="test-service")
        assert "event_type" in str(exc_info.value).lower()

    def test_headers_invalid_priority_rejected(self) -> None:
        """Verify headers with invalid priority are rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        with pytest.raises(ValidationError):
            ModelEventHeaders(
                source="test",
                event_type="event",
                priority="urgent",  # Invalid - should be 'critical'
            )

    def test_message_invalid_topic_type_rejected(self) -> None:
        """Verify message with non-string topic is rejected."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")

        with pytest.raises(ValidationError):
            ModelEventMessage(
                topic=12345,  # Should be string
                value=b"test-value",
                headers=headers,
            )

    def test_message_invalid_value_type_rejected(self) -> None:
        """Verify message with invalid value type is rejected.

        Note: Pydantic may coerce strings to bytes, so we test with a type
        that cannot be coerced to bytes (e.g., integer).
        """
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")

        with pytest.raises(ValidationError):
            ModelEventMessage(
                topic="test.topic",
                value=12345,  # Integer cannot be coerced to bytes
                headers=headers,
            )

    def test_message_invalid_headers_type_rejected(self) -> None:
        """Verify message with invalid headers type is rejected."""
        from omnibase_infra.event_bus.models import ModelEventMessage

        with pytest.raises(ValidationError):
            ModelEventMessage(
                topic="test.topic",
                value=b"test-value",
                headers={"source": "test"},  # Should be ModelEventHeaders
            )


# =============================================================================
# Header Completeness Validation Tests
# =============================================================================


class TestHeaderCompleteness:
    """Tests for header completeness in event bus operations."""

    @pytest.mark.asyncio
    async def test_published_message_has_complete_headers(
        self,
        started_event_bus: InMemoryEventBus,
    ) -> None:
        """Verify published messages have complete headers."""
        from omnibase_infra.event_bus.models import ModelEventMessage

        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await started_event_bus.subscribe("test.completeness", "test-group", handler)
        await started_event_bus.publish("test.completeness", None, b"test-value")

        assert len(received_messages) == 1
        msg = received_messages[0]

        # Verify header completeness
        assert msg.headers.source is not None
        assert msg.headers.event_type is not None
        assert msg.headers.correlation_id is not None
        assert msg.headers.message_id is not None
        assert msg.headers.timestamp is not None
        assert msg.headers.content_type is not None
        assert msg.headers.priority is not None
        assert msg.headers.retry_count is not None
        assert msg.headers.max_retries is not None

    @pytest.mark.asyncio
    async def test_custom_headers_preserved_through_publish(
        self,
        started_event_bus: InMemoryEventBus,
    ) -> None:
        """Verify custom headers are preserved through publish/subscribe cycle."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        received_messages: list[ModelEventMessage] = []
        custom_correlation_id = uuid4()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await started_event_bus.subscribe("test.custom", "test-group", handler)

        custom_headers = ModelEventHeaders(
            source="custom-source",
            event_type="custom.event",
            correlation_id=custom_correlation_id,
            trace_id="trace-xyz",
            span_id="span-abc",
            priority="high",
            routing_key="custom.routing.key",
        )

        await started_event_bus.publish(
            "test.custom",
            b"custom-key",
            b"custom-value",
            custom_headers,
        )

        assert len(received_messages) == 1
        msg = received_messages[0]

        # Verify custom headers are preserved
        assert msg.headers.source == "custom-source"
        assert msg.headers.event_type == "custom.event"
        assert msg.headers.correlation_id == custom_correlation_id
        assert msg.headers.trace_id == "trace-xyz"
        assert msg.headers.span_id == "span-abc"
        assert msg.headers.priority == "high"
        assert msg.headers.routing_key == "custom.routing.key"

    @pytest.mark.asyncio
    async def test_message_metadata_preserved(
        self,
        started_event_bus: InMemoryEventBus,
    ) -> None:
        """Verify message metadata (topic, key, offset, partition) is preserved."""
        from omnibase_infra.event_bus.models import ModelEventMessage

        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await started_event_bus.subscribe("test.metadata", "test-group", handler)
        await started_event_bus.publish("test.metadata", b"msg-key", b"msg-value")

        assert len(received_messages) == 1
        msg = received_messages[0]

        # Verify message metadata
        assert msg.topic == "test.metadata"
        assert msg.key == b"msg-key"
        assert msg.value == b"msg-value"
        assert msg.offset is not None  # Should have offset from InMemoryEventBus
        assert (
            msg.partition is not None
        )  # Should have partition (0 for InMemoryEventBus)

    @pytest.mark.asyncio
    async def test_sequential_messages_have_unique_ids(
        self,
        started_event_bus: InMemoryEventBus,
    ) -> None:
        """Verify sequential messages have unique correlation and message IDs."""
        from omnibase_infra.event_bus.models import ModelEventMessage

        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await started_event_bus.subscribe("test.unique", "test-group", handler)

        # Publish multiple messages
        for i in range(5):
            await started_event_bus.publish("test.unique", None, f"value-{i}".encode())

        assert len(received_messages) == 5

        # Collect all IDs
        correlation_ids = {msg.headers.correlation_id for msg in received_messages}
        message_ids = {msg.headers.message_id for msg in received_messages}
        offsets = {msg.offset for msg in received_messages}

        # All IDs should be unique
        assert len(correlation_ids) == 5
        assert len(message_ids) == 5
        assert len(offsets) == 5


# =============================================================================
# Schema Serialization Tests
# =============================================================================


class TestSchemaSerialzation:
    """Tests for schema serialization and deserialization."""

    def test_headers_json_serialization(self) -> None:
        """Verify headers can be serialized to JSON."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            priority="high",
        )

        # Serialize to dict (JSON-compatible)
        headers_dict = headers.model_dump(mode="json")

        assert headers_dict["source"] == "test-service"
        assert headers_dict["event_type"] == "test.event"
        assert headers_dict["priority"] == "high"
        assert "correlation_id" in headers_dict
        assert "message_id" in headers_dict
        assert "timestamp" in headers_dict

    def test_headers_round_trip_serialization(self) -> None:
        """Verify headers survive JSON round-trip serialization."""
        import json

        from omnibase_infra.event_bus.models import ModelEventHeaders

        original = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            trace_id="trace-123",
            span_id="span-456",
            priority="critical",
            routing_key="orders.priority",
        )

        # Serialize to JSON string and back
        json_str = json.dumps(original.model_dump(mode="json"))
        recreated = ModelEventHeaders.model_validate_json(json_str)

        assert recreated.source == original.source
        assert recreated.event_type == original.event_type
        assert recreated.trace_id == original.trace_id
        assert recreated.span_id == original.span_id
        assert recreated.priority == original.priority
        assert recreated.routing_key == original.routing_key
        assert recreated.correlation_id == original.correlation_id

    def test_message_json_serialization(self) -> None:
        """Verify message can be serialized to JSON-compatible dict."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        headers = ModelEventHeaders(source="test", event_type="event")
        message = ModelEventMessage(
            topic="test.topic",
            key=b"key",
            value=b'{"data": "value"}',
            headers=headers,
            offset="123",
            partition=0,
        )

        # Serialize to dict
        msg_dict = message.model_dump(mode="json")

        assert msg_dict["topic"] == "test.topic"
        # Bytes are base64 encoded in JSON mode
        assert "headers" in msg_dict
        assert msg_dict["offset"] == "123"
        assert msg_dict["partition"] == 0
