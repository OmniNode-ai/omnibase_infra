"""
Unit tests for IntentPublisherMixin.

Tests the intent publishing capability provided by the mixin,
ensuring proper intent construction, Kafka publishing, and error handling.
"""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import BaseModel

from omninode_bridge.events.models.intent_events import TOPIC_EVENT_PUBLISH_INTENT
from omninode_bridge.mixins import MixinIntentPublisher, ModelIntentPublishResult


class MockKafkaClient:
    """Mock Kafka client for testing."""

    def __init__(self):
        self.published_messages = []

    async def publish(self, topic: str, key: str, value: str):
        """Mock publish method."""
        self.published_messages.append({"topic": topic, "key": key, "value": value})


class MockContainer:
    """Mock ModelContainer for testing."""

    def __init__(self, kafka_client=None):
        self._kafka_client = kafka_client

    def get_service(self, name: str):
        """Mock get_service."""
        if name == "kafka_client":
            return self._kafka_client
        return None


class TestEvent(BaseModel):
    """Test event model."""

    message: str
    count: int


class TestNode(MixinIntentPublisher):
    """Test node using the mixin."""

    def __init__(self, container):
        self._init_intent_publisher(container)


# Tests
@pytest.mark.asyncio
async def test_publish_event_intent_success():
    """Test successful intent publishing."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    # Create test event
    event = TestEvent(message="test", count=42)

    # Publish intent
    result = await node.publish_event_intent(
        target_topic="test.topic.v1", target_key="test-key", event=event
    )

    # Verify result
    assert isinstance(result, ModelIntentPublishResult)
    assert isinstance(result.intent_id, UUID)
    assert isinstance(result.published_at, datetime)
    assert result.target_topic == "test.topic.v1"
    assert isinstance(result.correlation_id, UUID)

    # Verify Kafka message
    assert len(kafka_client.published_messages) == 1
    msg = kafka_client.published_messages[0]
    assert msg["topic"] == TOPIC_EVENT_PUBLISH_INTENT
    assert msg["key"] == str(result.intent_id)

    # Verify message contains event payload
    assert "test" in msg["value"]
    assert "42" in msg["value"]


@pytest.mark.asyncio
async def test_publish_event_intent_with_correlation_id():
    """Test intent publishing with explicit correlation ID."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    event = TestEvent(message="test", count=1)
    correlation_id = UUID("12345678-1234-1234-1234-123456789012")

    result = await node.publish_event_intent(
        target_topic="test.topic.v1",
        target_key="key",
        event=event,
        correlation_id=correlation_id,
    )

    assert result.correlation_id == correlation_id


@pytest.mark.asyncio
async def test_publish_event_intent_with_priority():
    """Test intent publishing with priority."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    event = TestEvent(message="urgent", count=1)

    result = await node.publish_event_intent(
        target_topic="test.topic.v1",
        target_key="key",
        event=event,
        priority=1,  # Highest priority
    )

    assert isinstance(result, ModelIntentPublishResult)

    # Verify priority in message
    msg = kafka_client.published_messages[0]
    assert '"priority":1' in msg["value"] or '"priority": 1' in msg["value"]


@pytest.mark.asyncio
async def test_publish_event_intent_invalid_priority():
    """Test intent publishing with invalid priority."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    event = TestEvent(message="test", count=1)

    with pytest.raises(ValueError, match="Priority must be 1-10"):
        await node.publish_event_intent(
            target_topic="test.topic.v1",
            target_key="key",
            event=event,
            priority=11,  # Invalid
        )


@pytest.mark.asyncio
async def test_publish_event_intent_non_pydantic_event():
    """Test intent publishing with non-Pydantic event."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    # Use dict instead of Pydantic model
    invalid_event = {"message": "test"}

    with pytest.raises(AttributeError, match="must be a Pydantic model"):
        await node.publish_event_intent(
            target_topic="test.topic.v1", target_key="key", event=invalid_event
        )


def test_init_intent_publisher_missing_kafka_client():
    """Test initialization without kafka_client service."""
    container = MockContainer(kafka_client=None)

    with pytest.raises(ValueError, match="requires 'kafka_client' service"):
        TestNode(container)


@pytest.mark.asyncio
async def test_multiple_intent_publications():
    """Test publishing multiple intents."""
    kafka_client = MockKafkaClient()
    container = MockContainer(kafka_client=kafka_client)
    node = TestNode(container)

    # Publish 3 intents
    for i in range(3):
        event = TestEvent(message=f"test-{i}", count=i)
        await node.publish_event_intent(
            target_topic=f"test.topic.{i}", target_key=f"key-{i}", event=event
        )

    # Verify all published
    assert len(kafka_client.published_messages) == 3

    # Verify unique intent IDs
    intent_ids = [msg["key"] for msg in kafka_client.published_messages]
    assert len(set(intent_ids)) == 3  # All unique
