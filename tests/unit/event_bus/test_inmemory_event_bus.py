# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for InMemoryEventBus.

Comprehensive test suite covering all public methods, edge cases,
error handling, and concurrent operation scenarios.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage


class TestInMemoryEventBusLifecycle:
    """Test suite for event bus lifecycle management."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture with test configuration."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_start_and_close(self, event_bus: InMemoryEventBus) -> None:
        """Test bus lifecycle - start and close operations."""
        # Initially not started
        health = await event_bus.health_check()
        assert health["healthy"] is False
        assert health["started"] is False

        # Start the bus
        await event_bus.start()
        health = await event_bus.health_check()
        assert health["healthy"] is True
        assert health["started"] is True

        # Close the bus
        await event_bus.close()
        health = await event_bus.health_check()
        assert health["healthy"] is False
        assert health["started"] is False

    @pytest.mark.asyncio
    async def test_multiple_start_calls(self, event_bus: InMemoryEventBus) -> None:
        """Test that multiple start calls are safe."""
        await event_bus.start()
        await event_bus.start()  # Second start should be idempotent

        health = await event_bus.health_check()
        assert health["started"] is True

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_close_calls(self, event_bus: InMemoryEventBus) -> None:
        """Test that multiple close calls are safe."""
        await event_bus.start()
        await event_bus.close()
        await event_bus.close()  # Second close should be idempotent

        health = await event_bus.health_check()
        assert health["started"] is False

    @pytest.mark.asyncio
    async def test_shutdown_alias(self, event_bus: InMemoryEventBus) -> None:
        """Test shutdown() is an alias for close()."""
        await event_bus.start()
        await event_bus.shutdown()

        health = await event_bus.health_check()
        assert health["started"] is False

    @pytest.mark.asyncio
    async def test_initialize_with_config(self) -> None:
        """Test initialize() method with configuration override."""
        event_bus = InMemoryEventBus()
        await event_bus.initialize(
            {
                "environment": "production",
                "group": "prod-group",
                "max_history": 500,
            }
        )

        assert event_bus.environment == "production"
        assert event_bus.group == "prod-group"
        health = await event_bus.health_check()
        assert health["started"] is True

        await event_bus.close()


class TestInMemoryEventBusProperties:
    """Test suite for event bus properties."""

    def test_default_properties(self) -> None:
        """Test default property values."""
        event_bus = InMemoryEventBus()
        assert event_bus.environment == "local"
        assert event_bus.group == "default"
        assert event_bus.adapter is event_bus

    def test_custom_properties(self) -> None:
        """Test custom property values."""
        event_bus = InMemoryEventBus(
            environment="staging",
            group="worker-group",
            max_history=2000,
        )
        assert event_bus.environment == "staging"
        assert event_bus.group == "worker-group"
        assert event_bus.adapter is event_bus


class TestInMemoryEventBusPublish:
    """Test suite for publish operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_publish_requires_start(self, event_bus: InMemoryEventBus) -> None:
        """Test that publish fails if bus not started."""
        with pytest.raises(InfraUnavailableError, match="not started"):
            await event_bus.publish("test-topic", None, b"test")

    @pytest.mark.asyncio
    async def test_publish_basic(self, event_bus: InMemoryEventBus) -> None:
        """Test basic publish operation."""
        await event_bus.start()

        await event_bus.publish("test-topic", b"key1", b"value1")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].topic == "test-topic"
        assert history[0].key == b"key1"
        assert history[0].value == b"value1"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_with_none_key(self, event_bus: InMemoryEventBus) -> None:
        """Test publish with None key."""
        await event_bus.start()

        await event_bus.publish("test-topic", None, b"value")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].key is None

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_with_custom_headers(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test publish with custom headers."""
        await event_bus.start()

        headers = ModelEventHeaders(
            source="custom-source",
            event_type="custom-event",
            priority="high",
        )
        await event_bus.publish("test-topic", None, b"value", headers)

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].headers.source == "custom-source"
        assert history[0].headers.event_type == "custom-event"
        assert history[0].headers.priority == "high"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_auto_generates_headers(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that publish auto-generates headers when not provided."""
        await event_bus.start()

        await event_bus.publish("my-topic", None, b"value")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].headers.source == "test.test-group"
        assert history[0].headers.event_type == "my-topic"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_offset_increments(self, event_bus: InMemoryEventBus) -> None:
        """Test that publish increments topic offset."""
        await event_bus.start()

        await event_bus.publish("topic1", None, b"msg1")
        await event_bus.publish("topic1", None, b"msg2")
        await event_bus.publish("topic2", None, b"msg3")

        offset1 = await event_bus.get_topic_offset("topic1")
        offset2 = await event_bus.get_topic_offset("topic2")

        assert offset1 == 2
        assert offset2 == 1

        history = await event_bus.get_event_history()
        assert history[0].offset == "0"
        assert history[1].offset == "1"
        assert history[2].offset == "0"  # Different topic, starts at 0

        await event_bus.close()


class TestInMemoryEventBusSubscribe:
    """Test suite for subscribe operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_subscribe_receives_published_message(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test basic publish/subscribe flow - subscriber receives published message."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        unsubscribe = await event_bus.subscribe("test-topic", "group1", handler)

        await event_bus.publish("test-topic", b"key1", b"value1")

        assert len(received) == 1
        assert received[0].value == b"value1"
        assert received[0].key == b"key1"
        assert received[0].topic == "test-topic"

        await unsubscribe()
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_topic(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test multiple subscribers receive messages."""
        await event_bus.start()

        received1: list[ModelEventMessage] = []
        received2: list[ModelEventMessage] = []

        async def handler1(msg: ModelEventMessage) -> None:
            received1.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            received2.append(msg)

        await event_bus.subscribe("test-topic", "group1", handler1)
        await event_bus.subscribe("test-topic", "group2", handler2)

        await event_bus.publish("test-topic", None, b"test")

        assert len(received1) == 1
        assert len(received2) == 1

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_different_topics(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test subscribers only receive messages for their topics."""
        await event_bus.start()

        received1: list[ModelEventMessage] = []
        received2: list[ModelEventMessage] = []

        async def handler1(msg: ModelEventMessage) -> None:
            received1.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            received2.append(msg)

        await event_bus.subscribe("topic1", "group1", handler1)
        await event_bus.subscribe("topic2", "group2", handler2)

        await event_bus.publish("topic1", None, b"for-topic1")
        await event_bus.publish("topic2", None, b"for-topic2")

        assert len(received1) == 1
        assert received1[0].value == b"for-topic1"
        assert len(received2) == 1
        assert received2[0].value == b"for-topic2"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus: InMemoryEventBus) -> None:
        """Test unsubscribe removes handler."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        unsubscribe = await event_bus.subscribe("test-topic", "group1", handler)
        await event_bus.publish("test-topic", None, b"first")
        assert len(received) == 1

        await unsubscribe()
        await event_bus.publish("test-topic", None, b"second")
        assert len(received) == 1  # Should not receive second message

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_double_unsubscribe_safe(self, event_bus: InMemoryEventBus) -> None:
        """Test that double unsubscribe is safe."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsubscribe = await event_bus.subscribe("test-topic", "group1", handler)
        await unsubscribe()
        await unsubscribe()  # Should not raise

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_error_handling(self, event_bus: InMemoryEventBus) -> None:
        """Test that subscriber errors don't affect other subscribers."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional test error")

        async def good_handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        await event_bus.subscribe("test-topic", "fail-group", failing_handler)
        await event_bus.subscribe("test-topic", "good-group", good_handler)

        # Should not raise, and good_handler should still receive
        await event_bus.publish("test-topic", None, b"test")
        assert len(received) == 1

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_same_handler_multiple_groups(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test same handler subscribed under different groups."""
        await event_bus.start()

        call_count = 0

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal call_count
            call_count += 1

        await event_bus.subscribe("test-topic", "group1", handler)
        await event_bus.subscribe("test-topic", "group2", handler)

        await event_bus.publish("test-topic", None, b"test")

        # Handler should be called twice (once per subscription)
        assert call_count == 2

        await event_bus.close()


class TestInMemoryEventBusHistory:
    """Test suite for event history operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_event_history_basic(self, event_bus: InMemoryEventBus) -> None:
        """Test basic event history tracking."""
        await event_bus.start()

        await event_bus.publish("topic1", None, b"msg1")
        await event_bus.publish("topic2", None, b"msg2")
        await event_bus.publish("topic1", None, b"msg3")

        # Get all history
        history = await event_bus.get_event_history(limit=10)
        assert len(history) == 3

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_event_history_filter_by_topic(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test event history filtering by topic."""
        await event_bus.start()

        await event_bus.publish("topic1", None, b"msg1")
        await event_bus.publish("topic2", None, b"msg2")
        await event_bus.publish("topic1", None, b"msg3")

        # Filter by topic
        history = await event_bus.get_event_history(limit=10, topic="topic1")
        assert len(history) == 2
        assert all(msg.topic == "topic1" for msg in history)

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_event_history_filter_applied_before_limit(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that topic filter is applied BEFORE limit.

        This verifies the fix for the bug where limit was applied before
        the topic filter, causing users to get fewer results than expected.

        Scenario: If we have 10 messages total (5 on topic1, 5 on topic2)
        interleaved, and request limit=3 with topic="topic1", we should
        get exactly 3 messages from topic1 (the most recent 3).
        """
        await event_bus.start()

        # Publish interleaved messages: topic1 and topic2 alternating
        # Order: t1_0, t2_0, t1_1, t2_1, t1_2, t2_2, t1_3, t2_3, t1_4, t2_4
        for i in range(5):
            await event_bus.publish("topic1", None, f"t1_msg{i}".encode())
            await event_bus.publish("topic2", None, f"t2_msg{i}".encode())

        # Request limit=3 with topic filter
        # If filter is applied BEFORE limit: we get 3 topic1 messages (t1_2, t1_3, t1_4)
        # If filter is applied AFTER limit: we might get 0-2 topic1 messages
        history = await event_bus.get_event_history(limit=3, topic="topic1")

        # Should get exactly 3 messages, all from topic1
        assert len(history) == 3
        assert all(msg.topic == "topic1" for msg in history)

        # Should be the most recent 3 topic1 messages
        assert history[0].value == b"t1_msg2"
        assert history[1].value == b"t1_msg3"
        assert history[2].value == b"t1_msg4"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_event_history_limit(self, event_bus: InMemoryEventBus) -> None:
        """Test event history limit parameter."""
        await event_bus.start()

        for i in range(10):
            await event_bus.publish("test", None, f"msg{i}".encode())

        history = await event_bus.get_event_history(limit=5)
        assert len(history) == 5
        # Should return the last 5 messages
        assert history[0].value == b"msg5"
        assert history[4].value == b"msg9"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_clear_event_history(self, event_bus: InMemoryEventBus) -> None:
        """Test clearing event history."""
        await event_bus.start()

        await event_bus.publish("test-topic", None, b"msg1")
        history = await event_bus.get_event_history()
        assert len(history) == 1

        await event_bus.clear_event_history()
        history = await event_bus.get_event_history()
        assert len(history) == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_max_history_limit(self) -> None:
        """Test history is limited to max_history."""
        event_bus = InMemoryEventBus(max_history=5)
        await event_bus.start()

        for i in range(10):
            await event_bus.publish("test", None, f"msg{i}".encode())

        history = await event_bus.get_event_history(limit=100)
        assert len(history) == 5
        # Should have the last 5 messages
        assert history[0].value == b"msg5"
        assert history[4].value == b"msg9"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_history_empty_on_new_bus(self, event_bus: InMemoryEventBus) -> None:
        """Test that new bus has empty history."""
        await event_bus.start()

        history = await event_bus.get_event_history()
        assert len(history) == 0

        await event_bus.close()


class TestInMemoryEventBusSubscriberCount:
    """Test suite for subscriber count operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_subscriber_count_initial(self, event_bus: InMemoryEventBus) -> None:
        """Test subscriber count is zero initially."""
        await event_bus.start()

        count = await event_bus.get_subscriber_count()
        assert count == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_count_increments(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test subscriber count increments on subscribe."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        assert await event_bus.get_subscriber_count() == 0

        await event_bus.subscribe("topic1", "g1", handler)
        assert await event_bus.get_subscriber_count() == 1

        await event_bus.subscribe("topic2", "g2", handler)
        assert await event_bus.get_subscriber_count() == 2

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_count_filter_by_topic(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test subscriber count filtering by topic."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        await event_bus.subscribe("topic1", "g1", handler)
        await event_bus.subscribe("topic1", "g2", handler)
        await event_bus.subscribe("topic2", "g3", handler)

        assert await event_bus.get_subscriber_count() == 3
        assert await event_bus.get_subscriber_count(topic="topic1") == 2
        assert await event_bus.get_subscriber_count(topic="topic2") == 1
        assert await event_bus.get_subscriber_count(topic="nonexistent") == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_count_decrements_on_unsubscribe(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test subscriber count decrements on unsubscribe."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsub1 = await event_bus.subscribe("topic1", "g1", handler)
        unsub2 = await event_bus.subscribe("topic1", "g2", handler)

        assert await event_bus.get_subscriber_count() == 2

        await unsub1()
        assert await event_bus.get_subscriber_count() == 1

        await unsub2()
        assert await event_bus.get_subscriber_count() == 0

        await event_bus.close()


class TestInMemoryEventBusTopics:
    """Test suite for topic listing operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_get_topics_empty(self, event_bus: InMemoryEventBus) -> None:
        """Test get_topics returns empty list initially."""
        await event_bus.start()

        topics = await event_bus.get_topics()
        assert topics == []

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_get_topics_with_subscribers(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test get_topics returns topics with active subscribers."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        await event_bus.subscribe("topic1", "g1", handler)
        await event_bus.subscribe("topic2", "g2", handler)

        topics = await event_bus.get_topics()
        assert set(topics) == {"topic1", "topic2"}

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_get_topics_excludes_empty_topics(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test get_topics excludes topics with no subscribers."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsub = await event_bus.subscribe("topic1", "g1", handler)
        await event_bus.subscribe("topic2", "g2", handler)

        topics = await event_bus.get_topics()
        assert set(topics) == {"topic1", "topic2"}

        await unsub()
        topics = await event_bus.get_topics()
        assert set(topics) == {"topic2"}

        await event_bus.close()


class TestInMemoryEventBusBroadcast:
    """Test suite for broadcast and group send operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_broadcast_to_environment(self, event_bus: InMemoryEventBus) -> None:
        """Test broadcast_to_environment."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        # Subscribe to broadcast topic
        await event_bus.subscribe("test.broadcast", "group1", handler)
        await event_bus.broadcast_to_environment("test_cmd", {"key": "value"})

        assert len(received) == 1
        payload = json.loads(received[0].value)
        assert payload["command"] == "test_cmd"
        assert payload["payload"] == {"key": "value"}

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_broadcast_to_specific_environment(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test broadcast to a specific target environment."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        # Subscribe to production broadcast topic
        await event_bus.subscribe("production.broadcast", "group1", handler)
        await event_bus.broadcast_to_environment(
            "deploy_cmd", {"version": "1.0"}, target_environment="production"
        )

        assert len(received) == 1

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_send_to_group(self, event_bus: InMemoryEventBus) -> None:
        """Test send_to_group."""
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        # Subscribe to group topic
        await event_bus.subscribe("test.target-group", "group1", handler)
        await event_bus.send_to_group("test_cmd", {"key": "value"}, "target-group")

        assert len(received) == 1
        payload = json.loads(received[0].value)
        assert payload["command"] == "test_cmd"
        assert payload["payload"] == {"key": "value"}

        await event_bus.close()


class TestInMemoryEventBusPublishEnvelope:
    """Test suite for publish_envelope operation."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_publish_envelope_with_pydantic_model(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test publish_envelope with a Pydantic model."""
        await event_bus.start()

        class TestEnvelope(BaseModel):
            message: str
            count: int

        envelope = TestEnvelope(message="hello", count=42)
        await event_bus.publish_envelope(envelope, "test-topic")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        payload = json.loads(history[0].value)
        assert payload["message"] == "hello"
        assert payload["count"] == 42

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_envelope_with_dict(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test publish_envelope with a plain dict."""
        await event_bus.start()

        envelope = {"message": "hello", "count": 42}
        await event_bus.publish_envelope(envelope, "test-topic")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        payload = json.loads(history[0].value)
        assert payload["message"] == "hello"
        assert payload["count"] == 42

        await event_bus.close()


class TestInMemoryEventBusHealthCheck:
    """Test suite for health check operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_health_check_not_started(self, event_bus: InMemoryEventBus) -> None:
        """Test health check when not started."""
        health = await event_bus.health_check()

        assert health["healthy"] is False
        assert health["started"] is False
        assert health["environment"] == "test"
        assert health["group"] == "test-group"
        assert health["subscriber_count"] == 0
        assert health["topic_count"] == 0
        assert health["history_size"] == 0

    @pytest.mark.asyncio
    async def test_health_check_started(self, event_bus: InMemoryEventBus) -> None:
        """Test health check when started."""
        await event_bus.start()
        health = await event_bus.health_check()

        assert health["healthy"] is True
        assert health["started"] is True

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_health_check_with_activity(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test health check reflects activity."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        await event_bus.subscribe("topic1", "g1", handler)
        await event_bus.subscribe("topic2", "g2", handler)
        await event_bus.publish("topic1", None, b"msg1")
        await event_bus.publish("topic1", None, b"msg2")

        health = await event_bus.health_check()

        assert health["subscriber_count"] == 2
        assert health["topic_count"] == 2
        assert health["history_size"] == 2

        await event_bus.close()


class TestInMemoryEventBusConsumingLoop:
    """Test suite for start_consuming operation."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_start_consuming_auto_starts(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that start_consuming auto-starts the bus."""

        # Create a task that starts consuming
        async def consume_briefly() -> None:
            task = asyncio.create_task(event_bus.start_consuming())
            await asyncio.sleep(0.1)  # Let it start
            await event_bus.shutdown()  # Stop it
            await task

        await consume_briefly()

        # After shutdown, bus should be stopped
        health = await event_bus.health_check()
        assert health["started"] is False

    @pytest.mark.asyncio
    async def test_start_consuming_exits_on_shutdown(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that start_consuming exits when shutdown is called."""
        await event_bus.start()

        consuming_started = asyncio.Event()

        async def consume_with_signal() -> None:
            consuming_started.set()
            await event_bus.start_consuming()

        task = asyncio.create_task(consume_with_signal())

        # Wait for consuming to start
        await consuming_started.wait()
        await asyncio.sleep(0.15)

        # Shutdown should stop consuming
        await event_bus.shutdown()

        # Task should complete
        await asyncio.wait_for(task, timeout=1.0)


class TestInMemoryEventBusConcurrency:
    """Test suite for concurrent operations."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    @pytest.mark.asyncio
    async def test_concurrent_publish(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent publish operations."""
        await event_bus.start()

        async def publish_batch(start: int) -> None:
            for i in range(10):
                await event_bus.publish("test-topic", None, f"msg-{start + i}".encode())

        # Run multiple publishers concurrently
        await asyncio.gather(
            publish_batch(0),
            publish_batch(100),
            publish_batch(200),
        )

        history = await event_bus.get_event_history(limit=50)
        assert len(history) == 30

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent subscribe and unsubscribe operations."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        async def subscribe_unsubscribe(group_id: str) -> None:
            unsub = await event_bus.subscribe("test-topic", group_id, handler)
            await asyncio.sleep(0.01)
            await unsub()

        # Run multiple subscribe/unsubscribe cycles concurrently
        await asyncio.gather(*[subscribe_unsubscribe(f"group-{i}") for i in range(10)])

        # All should be unsubscribed
        count = await event_bus.get_subscriber_count()
        assert count == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_publish_subscribe(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent publish and subscribe operations."""
        await event_bus.start()

        received: list[ModelEventMessage] = []
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            async with lock:
                received.append(msg)

        await event_bus.subscribe("test-topic", "group1", handler)

        # Publish messages concurrently
        await asyncio.gather(
            *[
                event_bus.publish("test-topic", None, f"msg-{i}".encode())
                for i in range(20)
            ]
        )

        assert len(received) == 20

        await event_bus.close()


class TestInMemoryEventBusEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_value_publish(self) -> None:
        """Test publishing empty bytes."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        await event_bus.publish("test", None, b"")

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].value == b""

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_large_value_publish(self) -> None:
        """Test publishing large values."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        large_value = b"x" * 1_000_000  # 1MB
        await event_bus.publish("test", None, large_value)

        history = await event_bus.get_event_history()
        assert len(history) == 1
        assert history[0].value == large_value

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_topic_name_with_special_characters(self) -> None:
        """Test topics with special characters (hyphens)."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        await event_bus.subscribe("topic-with-special-chars", "group1", handler)
        await event_bus.publish("topic-with-special-chars", None, b"test")

        assert len(received) == 1
        assert received[0].topic == "topic-with-special-chars"

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_unicode_topic_name(self) -> None:
        """Test topics with unicode characters (Japanese, Chinese, emoji)."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        # Test with Japanese characters
        unicode_topic = "topic-日本語-テスト"
        await event_bus.subscribe(unicode_topic, "group1", handler)
        await event_bus.publish(unicode_topic, None, b"test")

        assert len(received) == 1
        assert received[0].topic == unicode_topic

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_special_characters_in_group_id(self) -> None:
        """Test group IDs with special characters."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        await event_bus.subscribe("test", "group-with.dots_and-dashes", handler)
        await event_bus.publish("test", None, b"test")

        assert len(received) == 1

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_get_topic_offset_nonexistent(self) -> None:
        """Test get_topic_offset for nonexistent topic."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        offset = await event_bus.get_topic_offset("nonexistent")
        assert offset == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_max_history_zero(self) -> None:
        """Test max_history of 0 (edge case)."""
        event_bus = InMemoryEventBus(max_history=0)
        await event_bus.start()

        await event_bus.publish("test", None, b"msg1")
        await event_bus.publish("test", None, b"msg2")

        # With max_history=0, history should be empty
        history = await event_bus.get_event_history()
        assert len(history) == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_close_clears_subscribers(self) -> None:
        """Test that close clears all subscribers."""
        event_bus = InMemoryEventBus()
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        await event_bus.subscribe("topic1", "g1", handler)
        await event_bus.subscribe("topic2", "g2", handler)

        assert await event_bus.get_subscriber_count() == 2

        await event_bus.close()

        # Subscribers should be cleared
        health = await event_bus.health_check()
        assert health["subscriber_count"] == 0

    @pytest.mark.asyncio
    async def test_publish_after_close_fails(self) -> None:
        """Test that publish after close fails."""
        event_bus = InMemoryEventBus()
        await event_bus.start()
        await event_bus.close()

        with pytest.raises(InfraUnavailableError, match="not started"):
            await event_bus.publish("test", None, b"test")


class TestInMemoryEventBusCircuitBreaker:
    """Test suite for circuit breaker functionality."""

    @pytest.fixture
    def event_bus(self) -> InMemoryEventBus:
        """Create event bus fixture."""
        return InMemoryEventBus(environment="test", group="test-group")

    def test_circuit_breaker_threshold_validation(self) -> None:
        """Test that invalid circuit_breaker_threshold raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="positive integer"):
            InMemoryEventBus(circuit_breaker_threshold=0)

        with pytest.raises(ProtocolConfigurationError, match="positive integer"):
            InMemoryEventBus(circuit_breaker_threshold=-1)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test circuit breaker opens after consecutive failures."""
        await event_bus.start()
        call_count = 0

        async def failing_handler(msg: ModelEventMessage) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Intentional failure")

        await event_bus.subscribe("test-topic", "fail-group", failing_handler)
        for _ in range(6):
            await event_bus.publish("test-topic", None, b"test")
        assert call_count == 5  # Circuit opens after 5 failures
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test circuit breaker resets after successful callback."""
        await event_bus.start()
        fail_count = 0
        should_fail = True

        async def flaky_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count, should_fail
            if should_fail:
                fail_count += 1
                raise ValueError("Intentional failure")

        await event_bus.subscribe("test-topic", "flaky-group", flaky_handler)
        for _ in range(3):
            await event_bus.publish("test-topic", None, b"test")
        assert fail_count == 3
        should_fail = False
        await event_bus.publish("test-topic", None, b"test")  # Success resets
        should_fail = True
        for _ in range(6):
            await event_bus.publish("test-topic", None, b"test")
        assert fail_count == 8  # 3 + 5 more after reset
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_reset_subscriber_circuit(self, event_bus: InMemoryEventBus) -> None:
        """Test manual circuit breaker reset."""
        await event_bus.start()
        call_count = 0

        async def failing_handler(msg: ModelEventMessage) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Intentional failure")

        await event_bus.subscribe("test-topic", "fail-group", failing_handler)
        for _ in range(6):
            await event_bus.publish("test-topic", None, b"test")
        assert call_count == 5
        reset = await event_bus.reset_subscriber_circuit("test-topic", "fail-group")
        assert reset is True
        await event_bus.publish("test-topic", None, b"test")
        assert call_count == 6
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_status(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test getting circuit breaker status."""
        await event_bus.start()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional failure")

        await event_bus.subscribe("test-topic", "fail-group", failing_handler)
        for _ in range(3):
            await event_bus.publish("test-topic", None, b"test")
        status = await event_bus.get_circuit_breaker_status()
        assert status["failure_counts"]["test-topic:fail-group"] == 3
        assert len(status["open_circuits"]) == 0
        for _ in range(3):
            await event_bus.publish("test-topic", None, b"test")
        status = await event_bus.get_circuit_breaker_status()
        assert len(status["open_circuits"]) == 1
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_reset_nonexistent_circuit(self, event_bus: InMemoryEventBus) -> None:
        """Test resetting a circuit that doesn't exist returns False."""
        await event_bus.start()
        reset = await event_bus.reset_subscriber_circuit("nonexistent", "group")
        assert reset is False
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_close_clears_circuit_breaker_state(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that closing the bus clears circuit breaker failure tracking."""
        await event_bus.start()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional failure")

        await event_bus.subscribe("test-topic", "fail-group", failing_handler)
        for _ in range(3):
            await event_bus.publish("test-topic", None, b"test")

        status = await event_bus.get_circuit_breaker_status()
        assert status["failure_counts"]["test-topic:fail-group"] == 3

        await event_bus.close()

        # After close and restart, circuit breaker state should be cleared
        await event_bus.start()
        status = await event_bus.get_circuit_breaker_status()
        assert len(status["failure_counts"]) == 0
        await event_bus.close()


class TestModelEventMessage:
    """Test suite for ModelEventMessage model."""

    @pytest.mark.asyncio
    async def test_message_ack(self) -> None:
        """Test message ack is a no-op for in-memory."""
        headers = ModelEventHeaders(source="test", event_type="test")
        message = ModelEventMessage(
            topic="test",
            key=b"key",
            value=b"value",
            headers=headers,
            offset="0",
            partition=0,
        )

        # Should not raise
        await message.ack()

    def test_message_fields(self) -> None:
        """Test message field access."""
        headers = ModelEventHeaders(source="test", event_type="test")
        message = ModelEventMessage(
            topic="my-topic",
            key=b"my-key",
            value=b"my-value",
            headers=headers,
            offset="42",
            partition=3,
        )

        assert message.topic == "my-topic"
        assert message.key == b"my-key"
        assert message.value == b"my-value"
        assert message.offset == "42"
        assert message.partition == 3
        assert message.headers.source == "test"


class TestModelEventHeaders:
    """Test suite for ModelEventHeaders model."""

    @pytest.mark.asyncio
    async def test_validate_headers_valid(self) -> None:
        """Test validate_headers returns True for valid headers."""
        headers = ModelEventHeaders(source="test", event_type="test")
        assert await headers.validate_headers() is True

    @pytest.mark.asyncio
    async def test_validate_headers_empty_event_type(self) -> None:
        """Test validate_headers returns False for empty event_type."""
        headers = ModelEventHeaders(source="test", event_type="")
        assert await headers.validate_headers() is False

    def test_headers_defaults(self) -> None:
        """Test header default values."""
        headers = ModelEventHeaders(source="test", event_type="test")

        assert headers.content_type == "application/json"
        assert headers.schema_version == "1.0.0"
        assert headers.priority == "normal"
        assert headers.retry_count == 0
        assert headers.max_retries == 3
        assert headers.correlation_id is not None
        assert headers.message_id is not None
        assert headers.timestamp is not None
