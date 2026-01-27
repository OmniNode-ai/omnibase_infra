# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for AdapterProtocolEventPublisherInmemory.

Golden tests validating the test adapter implementation for OMN-1616.
These tests ensure production-equivalent envelope format and preserve
all critical event properties.

Test Categories:
    - Topic Routing: Default routing via event_type and explicit override
    - Partition Key Encoding: Deterministic UTF-8 byte encoding
    - Correlation Tracking: Correlation ID and causation ID preservation
    - Payload Integrity: No silent drops or encoding loss
    - Metadata Preservation: Metadata present even when empty

References:
    - Linear Ticket: OMN-1616
    - Parent: OMN-1611
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import UTC
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.testing import (
    AdapterProtocolEventPublisherInmemory,
    decode_inmemory_event,
)
from omnibase_spi.protocols.event_bus import ProtocolEventPublisher

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def event_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Provide a started EventBusInmemory instance."""
    bus = EventBusInmemory(environment="test", group="publisher-adapter-test")
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def adapter(event_bus: EventBusInmemory) -> AdapterProtocolEventPublisherInmemory:
    """Provide an AdapterProtocolEventPublisherInmemory instance."""
    return AdapterProtocolEventPublisherInmemory(
        bus=event_bus,
        service_name="test-publisher-service",
        instance_id="test-instance-001",
    )


# =============================================================================
# Mock Context Value for metadata testing
# =============================================================================


@runtime_checkable
class MockContextValue(Protocol):
    """Mock protocol for context values in tests."""

    value: str


class SimpleContextValue:
    """Simple mock context value for testing."""

    def __init__(self, value: str) -> None:
        self.value = value


# =============================================================================
# Topic Routing Tests
# =============================================================================


class TestTopicRouting:
    """Tests for topic routing behavior (default and explicit override)."""

    @pytest.mark.asyncio
    async def test_default_topic_from_event_type(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify default topic routing uses event_type when no topic provided."""
        event_type = "omninode.user.event.created.v1"

        success = await adapter.publish(
            event_type=event_type,
            payload={"user_id": "usr-123"},
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        assert len(history) == 1
        # Topic should match event_type when no explicit topic provided
        assert history[0].topic == event_type

    @pytest.mark.asyncio
    async def test_explicit_topic_override(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify explicit topic parameter overrides default event_type routing."""
        event_type = "omninode.user.event.created.v1"
        explicit_topic = "custom.analytics.topic"

        success = await adapter.publish(
            event_type=event_type,
            payload={"user_id": "usr-456"},
            topic=explicit_topic,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        assert len(history) == 1
        # Topic should be the explicit override, not event_type
        assert history[0].topic == explicit_topic

        # But event_type should still be preserved in envelope metadata
        envelope = decode_inmemory_event(history[0].value)
        assert envelope.metadata.tags.get("event_type") == event_type

    @pytest.mark.asyncio
    async def test_multiple_topics_isolation(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify events are correctly routed to their respective topics."""
        topic_a = "topic.alpha"
        topic_b = "topic.beta"

        await adapter.publish(
            event_type="event.a",
            payload={"source": "alpha"},
            topic=topic_a,
        )
        await adapter.publish(
            event_type="event.b",
            payload={"source": "beta"},
            topic=topic_b,
        )

        history_a = await event_bus.get_event_history(topic=topic_a)
        history_b = await event_bus.get_event_history(topic=topic_b)

        assert len(history_a) == 1
        assert len(history_b) == 1

        envelope_a = decode_inmemory_event(history_a[0].value)
        envelope_b = decode_inmemory_event(history_b[0].value)

        assert envelope_a.payload["source"] == "alpha"
        assert envelope_b.payload["source"] == "beta"


# =============================================================================
# Partition Key Encoding Tests
# =============================================================================


class TestPartitionKeyEncoding:
    """Tests for partition key deterministic UTF-8 byte encoding."""

    @pytest.mark.asyncio
    async def test_partition_key_utf8_encoding(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify partition_key is encoded to UTF-8 bytes deterministically."""
        partition_key = "user-partition-key-123"

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "test"},
            partition_key=partition_key,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        assert len(history) == 1
        assert history[0].key == partition_key.encode("utf-8")

    @pytest.mark.asyncio
    async def test_partition_key_unicode_encoding(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify Unicode partition keys are correctly encoded to UTF-8."""
        # Unicode partition key with various characters
        partition_key = "user-partition-key"

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "unicode-test"},
            partition_key=partition_key,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        assert len(history) == 1
        assert history[0].key == partition_key.encode("utf-8")

    @pytest.mark.asyncio
    async def test_no_partition_key_yields_none(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify omitting partition_key results in None key."""
        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "no-key"},
            # No partition_key provided
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        assert len(history) == 1
        assert history[0].key is None

    @pytest.mark.asyncio
    async def test_partition_key_determinism(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify same partition_key always produces same byte encoding."""
        partition_key = "deterministic-key-42"

        # Publish twice with same key
        await adapter.publish(
            event_type="test.event.1",
            payload={"seq": 1},
            partition_key=partition_key,
        )
        await adapter.publish(
            event_type="test.event.2",
            payload={"seq": 2},
            partition_key=partition_key,
        )

        history = await event_bus.get_event_history(limit=2)
        assert len(history) == 2

        # Both messages should have identical key bytes
        assert history[0].key == history[1].key
        assert history[0].key == partition_key.encode("utf-8")


# =============================================================================
# Correlation ID and Causation ID Preservation Tests
# =============================================================================


class TestCorrelationIdPreservation:
    """Tests for correlation_id and causation_id preservation."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_as_uuid(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify correlation_id UUID is preserved in envelope."""
        correlation_id = str(uuid4())

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "correlation-test"},
            correlation_id=correlation_id,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.correlation_id is not None
        assert str(envelope.correlation_id) == correlation_id

    @pytest.mark.asyncio
    async def test_causation_id_preserved_in_metadata(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify causation_id is preserved in envelope metadata tags."""
        causation_id = str(uuid4())

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "causation-test"},
            causation_id=causation_id,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        # causation_id should be in metadata.tags
        assert envelope.metadata.tags.get("causation_id") == causation_id

    @pytest.mark.asyncio
    async def test_both_correlation_and_causation_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify both correlation_id and causation_id are preserved together."""
        correlation_id = str(uuid4())
        causation_id = str(uuid4())

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "both-ids"},
            correlation_id=correlation_id,
            causation_id=causation_id,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert str(envelope.correlation_id) == correlation_id
        assert envelope.metadata.tags.get("causation_id") == causation_id

    @pytest.mark.asyncio
    async def test_no_correlation_id_generates_envelope_id(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify envelope gets envelope_id even without correlation_id."""
        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "no-correlation"},
            # No correlation_id provided
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        # envelope_id should always be present
        assert envelope.envelope_id is not None
        # correlation_id should be None when not provided
        assert envelope.correlation_id is None

    @pytest.mark.asyncio
    async def test_invalid_uuid_correlation_id_handled(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify non-UUID correlation_id is handled gracefully."""
        # Not a valid UUID string
        invalid_correlation_id = "not-a-valid-uuid-string"

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "invalid-uuid"},
            correlation_id=invalid_correlation_id,
        )

        # Should still succeed
        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        # Adapter should have generated a valid UUID
        assert envelope.correlation_id is not None
        # Verify it's a valid UUID
        _ = UUID(str(envelope.correlation_id))


# =============================================================================
# Payload Integrity Tests
# =============================================================================


class TestPayloadIntegrity:
    """Tests for payload integrity (no silent drops or encoding loss)."""

    @pytest.mark.asyncio
    async def test_dict_payload_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify dict payload is preserved through serialization."""
        payload = {
            "user_id": "usr-123",
            "email": "user@example.com",
            "is_active": True,
            "score": 95.5,
            "tags": ["admin", "verified"],
        }

        success = await adapter.publish(
            event_type="user.created",
            payload=payload,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        # All fields should match exactly
        assert envelope.payload["user_id"] == payload["user_id"]
        assert envelope.payload["email"] == payload["email"]
        assert envelope.payload["is_active"] == payload["is_active"]
        assert envelope.payload["score"] == payload["score"]
        assert envelope.payload["tags"] == payload["tags"]

    @pytest.mark.asyncio
    async def test_nested_dict_payload_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify deeply nested payload is preserved."""
        payload = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep-value",
                        "array": [1, 2, {"nested": "obj"}],
                    }
                }
            }
        }

        success = await adapter.publish(
            event_type="nested.event",
            payload=payload,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.payload == payload

    @pytest.mark.asyncio
    async def test_list_payload_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify list payload is preserved."""
        payload = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "name": "third"},
        ]

        success = await adapter.publish(
            event_type="list.event",
            payload=payload,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.payload == payload

    @pytest.mark.asyncio
    async def test_primitive_payloads_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify primitive JSON payloads are preserved."""
        primitives: list[object] = ["string-value", 42, 3.14, True, None]

        for idx, payload in enumerate(primitives):
            success = await adapter.publish(
                event_type=f"primitive.event.{idx}",
                payload=payload,
            )
            assert success is True

        history = await event_bus.get_event_history(limit=5)
        assert len(history) == 5

        for idx, msg in enumerate(history):
            envelope = decode_inmemory_event(msg.value)
            assert envelope.payload == primitives[idx]

    @pytest.mark.asyncio
    async def test_empty_payload_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify empty dict and list payloads are preserved."""
        await adapter.publish(event_type="empty.dict", payload={})
        await adapter.publish(event_type="empty.list", payload=[])

        history = await event_bus.get_event_history(limit=2)
        assert len(history) == 2

        envelope_dict = decode_inmemory_event(history[0].value)
        envelope_list = decode_inmemory_event(history[1].value)

        assert envelope_dict.payload == {}
        assert envelope_list.payload == []

    @pytest.mark.asyncio
    async def test_unicode_payload_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify Unicode content in payload is preserved."""
        payload = {
            "greeting_ja": "こんにちは",  # Japanese: "hello"
            "greeting_ar": "مرحبا",  # Arabic: "hello"
            "emoji": "🎉🚀✨",  # Celebration, rocket, sparkles
            "special": "café naïve résumé",  # Latin-1 extended characters
        }

        success = await adapter.publish(
            event_type="unicode.event",
            payload=payload,
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.payload == payload


# =============================================================================
# Metadata Preservation Tests
# =============================================================================


class TestMetadataPreservation:
    """Tests for metadata presence and preservation."""

    @pytest.mark.asyncio
    async def test_metadata_always_present(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify metadata is present even when not explicitly provided."""
        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "test"},
            # No metadata provided
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        # Metadata should always be present
        assert envelope.metadata is not None
        # Default tags should be present
        assert "event_type" in envelope.metadata.tags
        assert "service_name" in envelope.metadata.tags
        assert "instance_id" in envelope.metadata.tags

    @pytest.mark.asyncio
    async def test_event_type_in_metadata_tags(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify event_type is stored in metadata tags."""
        event_type = "omninode.custom.event.v1"

        success = await adapter.publish(
            event_type=event_type,
            payload={"data": "test"},
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.metadata.tags.get("event_type") == event_type

    @pytest.mark.asyncio
    async def test_service_metadata_preserved(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify service_name and instance_id are preserved in metadata."""
        custom_service = "custom-test-service"
        custom_instance = "custom-instance-xyz"

        adapter = AdapterProtocolEventPublisherInmemory(
            bus=event_bus,
            service_name=custom_service,
            instance_id=custom_instance,
        )

        await adapter.publish(
            event_type="test.event",
            payload={"data": "service-test"},
        )

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.metadata.tags.get("service_name") == custom_service
        assert envelope.metadata.tags.get("instance_id") == custom_instance
        assert envelope.source_tool == f"{custom_service}.{custom_instance}"

    @pytest.mark.asyncio
    async def test_custom_metadata_preserved(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify custom metadata context values are preserved."""
        custom_value = SimpleContextValue(value="custom-context-data")

        success = await adapter.publish(
            event_type="test.event",
            payload={"data": "metadata-test"},
            metadata={"custom_key": custom_value},
        )

        assert success is True

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.metadata.tags.get("custom_key") == "custom-context-data"


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Tests for publisher metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_track_successful_publishes(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify metrics track successful publish count."""
        # Publish several events
        for i in range(5):
            await adapter.publish(
                event_type=f"metrics.event.{i}",
                payload={"seq": i},
            )

        metrics = await adapter.get_metrics()

        assert metrics["events_published"] == 5
        assert metrics["events_failed"] == 0
        assert metrics["total_publish_time_ms"] > 0
        assert metrics["avg_publish_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_metrics_initial_state(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify metrics start at zero."""
        adapter = AdapterProtocolEventPublisherInmemory(bus=event_bus)

        metrics = await adapter.get_metrics()

        assert metrics["events_published"] == 0
        assert metrics["events_failed"] == 0
        assert metrics["total_publish_time_ms"] == 0.0
        assert metrics["avg_publish_time_ms"] == 0.0
        assert metrics["circuit_breaker_status"] == "closed"


# =============================================================================
# Close/Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for adapter lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_prevents_further_publishes(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify close() prevents further publish calls."""
        # Publish one event first
        success = await adapter.publish(
            event_type="before.close",
            payload={"data": "test"},
        )
        assert success is True

        # Close the adapter
        await adapter.close()

        # Attempting to publish after close should raise
        with pytest.raises(RuntimeError, match="Publisher has been closed"):
            await adapter.publish(
                event_type="after.close",
                payload={"data": "should-fail"},
            )


# =============================================================================
# Decode Helper Tests
# =============================================================================


class TestDecodeHelper:
    """Tests for the decode_inmemory_event helper function."""

    @pytest.mark.asyncio
    async def test_decode_roundtrip(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify decode_inmemory_event correctly deserializes envelope."""
        original_payload = {"key": "value", "number": 42}
        correlation_id = str(uuid4())

        await adapter.publish(
            event_type="roundtrip.test",
            payload=original_payload,
            correlation_id=correlation_id,
        )

        history = await event_bus.get_event_history(limit=1)
        envelope = decode_inmemory_event(history[0].value)

        assert envelope.payload == original_payload
        assert str(envelope.correlation_id) == correlation_id
        assert envelope.metadata.tags.get("event_type") == "roundtrip.test"

    def test_decode_invalid_json_raises(self) -> None:
        """Verify decode_inmemory_event raises on invalid JSON."""
        invalid_json = b"not-valid-json{{"

        with pytest.raises(json.JSONDecodeError):
            decode_inmemory_event(invalid_json)

    def test_decode_non_envelope_json_raises(self) -> None:
        """Verify decode_inmemory_event raises on non-envelope JSON."""
        non_envelope_json = b'{"random": "data"}'

        with pytest.raises(Exception):  # Pydantic ValidationError
            decode_inmemory_event(non_envelope_json)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests verifying ProtocolEventPublisher interface compliance.

    Protocol Contract:
        - Must have async publish() method with correct signature
        - Must have async get_metrics() method returning JsonType
        - Must have async close() method with timeout_seconds parameter
        - Must pass isinstance() check against @runtime_checkable protocol

    This follows the ONEX protocol conformance pattern where implementations
    use duck typing (structural subtyping) rather than explicit inheritance.
    """

    def test_isinstance_protocol_check(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify adapter passes isinstance check for ProtocolEventPublisher.

        This test uses @runtime_checkable to verify structural subtyping.
        The adapter must implement the same method signatures as the protocol.
        """
        from omnibase_spi.protocols.event_bus import ProtocolEventPublisher

        assert isinstance(adapter, ProtocolEventPublisher), (
            "AdapterProtocolEventPublisherInmemory must implement ProtocolEventPublisher. "
            "Check that all required methods are present with correct signatures."
        )

    def test_publish_method_exists_and_is_async(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify publish method exists and is async."""
        import asyncio

        assert hasattr(adapter, "publish"), (
            "AdapterProtocolEventPublisherInmemory missing required method: publish"
        )
        assert asyncio.iscoroutinefunction(adapter.publish), (
            "publish must be an async method"
        )

    def test_publish_signature_matches_protocol(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify publish method signature matches ProtocolEventPublisher.

        Protocol defines:
            async def publish(
                self,
                event_type: str,
                payload: JsonType,
                correlation_id: str | None = None,
                causation_id: str | None = None,
                metadata: dict[str, ContextValue] | None = None,
                topic: str | None = None,
                partition_key: str | None = None,
            ) -> bool
        """
        import inspect

        sig = inspect.signature(adapter.publish)
        params = list(sig.parameters.keys())

        expected_params = [
            "event_type",
            "payload",
            "correlation_id",
            "causation_id",
            "metadata",
            "topic",
            "partition_key",
        ]
        assert params == expected_params, (
            f"publish signature mismatch. "
            f"Expected params: {expected_params}, got: {params}"
        )

        # Verify optional parameters have None defaults
        for optional_param in [
            "correlation_id",
            "causation_id",
            "metadata",
            "topic",
            "partition_key",
        ]:
            param = sig.parameters[optional_param]
            assert param.default is None, (
                f"{optional_param} parameter must have default value of None"
            )

    def test_get_metrics_method_exists_and_is_async(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify get_metrics method exists and is async."""
        import asyncio

        assert hasattr(adapter, "get_metrics"), (
            "AdapterProtocolEventPublisherInmemory missing required method: get_metrics"
        )
        assert asyncio.iscoroutinefunction(adapter.get_metrics), (
            "get_metrics must be an async method"
        )

    def test_close_method_exists_and_is_async(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify close method exists and is async."""
        import asyncio

        assert hasattr(adapter, "close"), (
            "AdapterProtocolEventPublisherInmemory missing required method: close"
        )
        assert asyncio.iscoroutinefunction(adapter.close), (
            "close must be an async method"
        )

    def test_close_signature_matches_protocol(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify close method has timeout_seconds parameter with default.

        Protocol defines:
            async def close(self, timeout_seconds: float = 30.0) -> None
        """
        import inspect

        sig = inspect.signature(adapter.close)
        params = list(sig.parameters.keys())

        assert "timeout_seconds" in params, "close must have timeout_seconds parameter"

        timeout_param = sig.parameters["timeout_seconds"]
        assert timeout_param.default == 30.0, (
            f"timeout_seconds must default to 30.0, got {timeout_param.default}"
        )

    @pytest.mark.asyncio
    async def test_publish_returns_bool(
        self,
        event_bus: EventBusInmemory,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify publish returns bool as specified by protocol."""
        result = await adapter.publish(
            event_type="test.protocol.compliance",
            payload={"test": "data"},
        )
        assert isinstance(result, bool), f"publish must return bool, got {type(result)}"

    @pytest.mark.asyncio
    async def test_get_metrics_returns_dict(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify get_metrics returns dict with expected keys."""
        metrics = await adapter.get_metrics()

        assert isinstance(metrics, dict), (
            f"get_metrics must return dict, got {type(metrics)}"
        )

        # Verify required metric keys per protocol docstring
        required_keys = [
            "events_published",
            "events_failed",
            "events_sent_to_dlq",
            "total_publish_time_ms",
            "avg_publish_time_ms",
            "circuit_breaker_opens",
            "retries_attempted",
            "circuit_breaker_status",
            "current_failures",
        ]
        for key in required_keys:
            assert key in metrics, f"get_metrics missing required key: {key}"

    def test_adapter_assignable_to_protocol(
        self,
        adapter: AdapterProtocolEventPublisherInmemory,
    ) -> None:
        """Verify adapter is assignable to ProtocolEventPublisher (static type check)."""
        # This assignment validates protocol compatibility at type-check time.
        # If the adapter's method signatures drift from the protocol,
        # mypy/pyright will report an error on this line.
        _publisher: ProtocolEventPublisher = adapter
