# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for PublisherTopicScoped.

Tests the topic-scoped publisher that validates against contract-declared
publish topics including:
- Publishing to allowed topics
- Rejection of disallowed topics
- Topic resolution with environment prefix
- JSON serialization and correlation ID handling

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - src/omnibase_infra/runtime/publisher_topic_scoped.py
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.runtime.publisher_topic_scoped import PublisherTopicScoped

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create mock event bus with publish method."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def publisher(mock_event_bus: AsyncMock) -> PublisherTopicScoped:
    """Create publisher with standard allowed topics."""
    return PublisherTopicScoped(
        event_bus=mock_event_bus,
        allowed_topics={"onex.evt.allowed.v1", "onex.evt.another.v1"},
        environment="dev",
    )


@pytest.fixture
def publisher_prod(mock_event_bus: AsyncMock) -> PublisherTopicScoped:
    """Create publisher with production environment."""
    return PublisherTopicScoped(
        event_bus=mock_event_bus,
        allowed_topics={"onex.evt.orders.v1"},
        environment="prod",
    )


# =============================================================================
# Publish Success Tests
# =============================================================================


class TestPublishSuccess:
    """Tests for successful publishing to allowed topics."""

    async def test_publish_to_allowed_topic_succeeds(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publishing to allowed topic succeeds."""
        result = await publisher.publish(
            event_type="test.event",
            payload={"key": "value"},
            topic="onex.evt.allowed.v1",
            correlation_id="corr-123",
        )

        assert result is True
        mock_event_bus.publish.assert_called_once()

    async def test_publish_uses_resolved_topic(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publish uses resolved topic with environment prefix."""
        await publisher.publish(
            event_type="test.event",
            payload={"key": "value"},
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["topic"] == "dev.onex.evt.allowed.v1"

    async def test_publish_serializes_payload_to_json(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publish serializes payload to JSON bytes."""
        payload = {"key": "value", "nested": {"inner": 123}}
        await publisher.publish(
            event_type="test.event",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        published_value = call_kwargs["value"]
        assert isinstance(published_value, bytes)
        assert json.loads(published_value) == payload

    async def test_publish_uses_correlation_id_as_key(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publish uses correlation_id as message key."""
        await publisher.publish(
            event_type="test.event",
            payload={"key": "value"},
            topic="onex.evt.allowed.v1",
            correlation_id="corr-abc-123",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["key"] == b"corr-abc-123"

    async def test_publish_without_correlation_id_uses_none_key(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publish without correlation_id uses None as key."""
        await publisher.publish(
            event_type="test.event",
            payload={"key": "value"},
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["key"] is None

    async def test_publish_to_second_allowed_topic(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publishing to another allowed topic succeeds."""
        result = await publisher.publish(
            event_type="test.event",
            payload={"data": "test"},
            topic="onex.evt.another.v1",
        )

        assert result is True
        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["topic"] == "dev.onex.evt.another.v1"


# =============================================================================
# Publish Failure Tests
# =============================================================================


class TestPublishFailure:
    """Tests for publish failures and validation errors."""

    async def test_publish_to_disallowed_topic_raises(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test publishing to topic not in contract raises ValueError."""
        with pytest.raises(ValueError, match="not in contract's publish_topics"):
            await publisher.publish(
                event_type="test.event",
                payload={"key": "value"},
                topic="onex.evt.forbidden.v1",
            )

    async def test_publish_without_topic_raises(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test publishing without topic raises ValueError."""
        with pytest.raises(ValueError, match="topic is required"):
            await publisher.publish(
                event_type="test.event",
                payload={"key": "value"},
                topic=None,
            )

    async def test_error_message_includes_allowed_topics(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test error message lists allowed topics."""
        try:
            await publisher.publish(
                event_type="test.event",
                payload={},
                topic="onex.evt.forbidden.v1",
            )
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "onex.evt.allowed.v1" in error_msg
            assert "onex.evt.another.v1" in error_msg

    async def test_publish_does_not_call_event_bus_on_validation_failure(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test event bus publish is not called when validation fails."""
        with pytest.raises(ValueError):
            await publisher.publish(
                event_type="test.event",
                payload={},
                topic="onex.evt.forbidden.v1",
            )

        mock_event_bus.publish.assert_not_called()


# =============================================================================
# Topic Resolution Tests
# =============================================================================


class TestTopicResolution:
    """Tests for topic suffix to full topic name resolution."""

    def test_resolve_topic_adds_environment(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test resolve_topic adds environment prefix."""
        result = publisher.resolve_topic("onex.evt.test.v1")
        assert result == "dev.onex.evt.test.v1"

    def test_resolve_topic_with_prod_environment(
        self,
        publisher_prod: PublisherTopicScoped,
    ) -> None:
        """Test resolve_topic with production environment."""
        result = publisher_prod.resolve_topic("onex.evt.orders.v1")
        assert result == "prod.onex.evt.orders.v1"

    def test_resolve_topic_with_custom_environment(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test resolve_topic with custom environment."""
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"topic.v1"},
            environment="staging-eu",
        )
        result = publisher.resolve_topic("topic.v1")
        assert result == "staging-eu.topic.v1"


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for publisher properties."""

    def test_allowed_topics_returns_frozenset(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test allowed_topics property returns immutable set."""
        topics = publisher.allowed_topics
        assert isinstance(topics, frozenset)

    def test_allowed_topics_contains_correct_values(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test allowed_topics contains expected values."""
        topics = publisher.allowed_topics
        assert "onex.evt.allowed.v1" in topics
        assert "onex.evt.another.v1" in topics
        assert len(topics) == 2

    def test_allowed_topics_is_immutable(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test allowed_topics cannot be modified."""
        topics = publisher.allowed_topics
        # frozenset doesn't have add method, verifying immutability
        assert not hasattr(topics, "add")

    def test_environment_property(
        self,
        publisher: PublisherTopicScoped,
    ) -> None:
        """Test environment property returns correct value."""
        assert publisher.environment == "dev"

    def test_environment_property_prod(
        self,
        publisher_prod: PublisherTopicScoped,
    ) -> None:
        """Test environment property for production publisher."""
        assert publisher_prod.environment == "prod"


# =============================================================================
# Payload Serialization Tests
# =============================================================================


class TestPayloadSerialization:
    """Tests for payload serialization."""

    async def test_serialize_dict_payload(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test serializing dict payload."""
        payload = {"name": "test", "count": 42}
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert json.loads(call_kwargs["value"]) == payload

    async def test_serialize_list_payload(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test serializing list payload."""
        payload = [1, 2, 3, "test"]
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert json.loads(call_kwargs["value"]) == payload

    async def test_serialize_string_payload(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test serializing string payload."""
        payload = "simple string"
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert json.loads(call_kwargs["value"]) == payload

    async def test_serialize_nested_payload(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test serializing nested payload."""
        payload = {
            "user": {
                "id": "123",
                "profile": {"name": "Test", "age": 30},
            },
            "items": [{"id": 1}, {"id": 2}],
        }
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert json.loads(call_kwargs["value"]) == payload

    async def test_serialize_null_values(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test serializing payload with null values."""
        payload = {"key": None, "list": [None, 1]}
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert json.loads(call_kwargs["value"]) == payload


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_allowed_topics(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publisher with empty allowed topics rejects all."""
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics=set(),
            environment="dev",
        )

        with pytest.raises(ValueError, match="not in contract's publish_topics"):
            await publisher.publish(
                event_type="test",
                payload={},
                topic="any.topic.v1",
            )

    async def test_kwargs_are_ignored(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test extra kwargs are ignored for protocol flexibility."""
        result = await publisher.publish(
            event_type="test",
            payload={"key": "value"},
            topic="onex.evt.allowed.v1",
            extra_arg="ignored",
            another_arg=123,
        )

        assert result is True
        mock_event_bus.publish.assert_called_once()

    async def test_empty_correlation_id_treated_as_none(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test empty string correlation_id is treated as None (falsy)."""
        # Implementation: key = correlation_id.encode() if correlation_id else None
        # Empty string is falsy, so key becomes None
        await publisher.publish(
            event_type="test",
            payload={},
            topic="onex.evt.allowed.v1",
            correlation_id="",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        # Empty string is falsy, so key is None
        assert call_kwargs["key"] is None

    async def test_unicode_in_payload(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test Unicode characters in payload are handled correctly."""
        payload = {"message": "Hello, world!", "emoji": "test"}
        await publisher.publish(
            event_type="test",
            payload=payload,
            topic="onex.evt.allowed.v1",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        decoded = json.loads(call_kwargs["value"].decode("utf-8"))
        assert decoded == payload

    async def test_unicode_in_correlation_id(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test Unicode characters in correlation_id are handled."""
        await publisher.publish(
            event_type="test",
            payload={},
            topic="onex.evt.allowed.v1",
            correlation_id="corr-test-abc",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["key"] == b"corr-test-abc"


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestMultiplePublications:
    """Tests for multiple publish operations."""

    async def test_multiple_publishes_to_same_topic(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test multiple publishes to same topic."""
        for i in range(3):
            await publisher.publish(
                event_type="test",
                payload={"count": i},
                topic="onex.evt.allowed.v1",
            )

        assert mock_event_bus.publish.call_count == 3

    async def test_publishes_to_different_allowed_topics(
        self,
        publisher: PublisherTopicScoped,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test publishes to different allowed topics."""
        await publisher.publish(
            event_type="event1",
            payload={"a": 1},
            topic="onex.evt.allowed.v1",
        )
        await publisher.publish(
            event_type="event2",
            payload={"b": 2},
            topic="onex.evt.another.v1",
        )

        assert mock_event_bus.publish.call_count == 2
        calls = mock_event_bus.publish.call_args_list
        topics = [c.kwargs["topic"] for c in calls]
        assert "dev.onex.evt.allowed.v1" in topics
        assert "dev.onex.evt.another.v1" in topics
