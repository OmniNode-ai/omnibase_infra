#!/usr/bin/env python3
"""
Mock Kafka Producer/Consumer for integration testing.

Provides in-memory Kafka simulation for testing without real Kafka infrastructure.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional
from uuid import uuid4


class MockKafkaProducer:
    """
    Mock Kafka producer for testing.

    Stores published events in memory for verification.
    """

    def __init__(self):
        """Initialize mock producer."""
        self.published_events: list[dict[str, Any]] = []
        self.topics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.is_connected = True
        self.publish_delay_ms = 0

    async def publish(
        self, topic: str, value: dict[str, Any], key: Optional[str] = None
    ) -> None:
        """
        Publish event to mock topic.

        Args:
            topic: Kafka topic name
            value: Event payload
            key: Optional partition key
        """
        if self.publish_delay_ms > 0:
            await asyncio.sleep(self.publish_delay_ms / 1000)

        event = {
            "topic": topic,
            "key": key or str(uuid4()),
            "value": value,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self.published_events.append(event)
        self.topics[topic].append(event)

    def get_events_for_topic(self, topic: str) -> list[dict[str, Any]]:
        """Get all events published to a specific topic."""
        return self.topics[topic]

    def clear(self) -> None:
        """Clear all published events."""
        self.published_events.clear()
        self.topics.clear()

    async def close(self) -> None:
        """Close mock producer (no-op)."""
        self.is_connected = False


class MockKafkaConsumer:
    """
    Mock Kafka consumer for testing.

    Consumes events from mock producer.
    """

    def __init__(self, topics: list[str], producer: Optional[MockKafkaProducer] = None):
        """
        Initialize mock consumer.

        Args:
            topics: Topics to subscribe to
            producer: Optional producer to consume from
        """
        self.topics = topics
        self.producer = producer
        self.consumed_events: list[dict[str, Any]] = []
        self.handlers: dict[str, Callable] = {}
        self.is_running = False

    async def consume(self, timeout_ms: int = 1000) -> Optional[dict[str, Any]]:
        """
        Consume next event from subscribed topics.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Next event or None if timeout
        """
        if not self.producer:
            return None

        # Get events for subscribed topics
        for topic in self.topics:
            events = self.producer.get_events_for_topic(topic)
            unconsumed = [e for e in events if e not in self.consumed_events]

            if unconsumed:
                event = unconsumed[0]
                self.consumed_events.append(event)
                return event

        # Simulate timeout
        await asyncio.sleep(timeout_ms / 1000)
        return None

    async def start(self) -> None:
        """Start consuming events."""
        self.is_running = True

        while self.is_running:
            event = await self.consume(timeout_ms=100)

            if event:
                topic = event["topic"]
                if topic in self.handlers:
                    await self.handlers[topic](event)

    async def stop(self) -> None:
        """Stop consuming events."""
        self.is_running = False

    def register_handler(self, topic: str, handler: Callable) -> None:
        """
        Register event handler for topic.

        Args:
            topic: Topic name
            handler: Async handler function
        """
        self.handlers[topic] = handler

    async def close(self) -> None:
        """Close mock consumer."""
        await self.stop()


class MockKafkaClient:
    """
    Combined mock Kafka client with producer and consumer.

    Provides unified interface for testing both publishing and consuming.
    """

    def __init__(self):
        """Initialize mock client."""
        self.producer = MockKafkaProducer()
        self.consumers: list[MockKafkaConsumer] = []
        self.is_connected = True

    async def publish(
        self, topic: str, value: dict[str, Any], key: Optional[str] = None
    ) -> None:
        """Publish event (delegates to producer)."""
        await self.producer.publish(topic, value, key)

    def create_consumer(self, topics: list[str]) -> MockKafkaConsumer:
        """
        Create consumer for topics.

        Args:
            topics: Topics to subscribe to

        Returns:
            MockKafkaConsumer instance
        """
        consumer = MockKafkaConsumer(topics, self.producer)
        self.consumers.append(consumer)
        return consumer

    def get_all_events(self) -> list[dict[str, Any]]:
        """Get all published events."""
        return self.producer.published_events

    def get_events_by_topic(self, topic: str) -> list[dict[str, Any]]:
        """Get events for specific topic."""
        return self.producer.get_events_for_topic(topic)

    def clear(self) -> None:
        """Clear all events."""
        self.producer.clear()
        for consumer in self.consumers:
            consumer.consumed_events.clear()

    async def close(self) -> None:
        """Close client and all consumers."""
        await self.producer.close()
        for consumer in self.consumers:
            await consumer.close()
        self.is_connected = False
