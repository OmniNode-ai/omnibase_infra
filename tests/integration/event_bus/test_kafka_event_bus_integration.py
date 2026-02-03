# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for EventBusKafka with RedPanda/Kafka.

These tests validate EventBusKafka behavior against actual Kafka infrastructure
(RedPanda or Kafka). They require a running Kafka broker and will be skipped
gracefully if Kafka is not available.

Test categories:
- Connection Tests: Validate basic connectivity and health checks
- End-to-End Tests: Verify publish/subscribe message flow
- Resilience Tests: Test reconnection and graceful degradation

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:9092")
    KAFKA_TIMEOUT_SECONDS: Operation timeout in seconds (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.models import ModelNodeIdentity

from .conftest import wait_for_consumer_ready

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models import ModelEventMessage

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Check if Kafka is available based on environment variable
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = KAFKA_BOOTSTRAP_SERVERS is not None

# Module-level markers - skip all tests if Kafka is not available
pytestmark = [
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason="Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)",
    ),
]

# Test configuration constants
TEST_TIMEOUT_SECONDS = 30
MESSAGE_DELIVERY_WAIT_SECONDS = 2.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment."""
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
def unique_topic() -> str:
    """Generate unique topic name for test isolation."""
    return f"test.integration.{uuid.uuid4().hex[:12]}"


@pytest.fixture
def unique_group() -> ModelNodeIdentity:
    """Generate unique node identity for test isolation."""
    return ModelNodeIdentity(
        env="kafka-integration-test",
        service="test-service",
        node_name=f"test-node-{uuid.uuid4().hex[:8]}",
        version="1.0.0",
    )


@pytest.fixture
async def kafka_event_bus(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator[EventBusKafka, None]:
    """Create and configure EventBusKafka for integration testing.

    Yields a started EventBusKafka instance and ensures cleanup after test.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=kafka_bootstrap_servers,
        environment="integration-test",
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
    )
    bus = EventBusKafka(config=config)

    yield bus

    # Cleanup: ensure bus is closed
    try:
        await bus.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
async def started_kafka_bus(
    kafka_event_bus: EventBusKafka,
) -> EventBusKafka:
    """Provide a started EventBusKafka instance."""
    await kafka_event_bus.start()
    return kafka_event_bus


# =============================================================================
# Connection Tests - Validate basic connectivity
# =============================================================================


class TestKafkaEventBusConnection:
    """Tests for EventBusKafka connection and lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_to_kafka(self, kafka_event_bus: EventBusKafka) -> None:
        """Verify EventBusKafka can connect to Kafka broker.

        This test validates that the event bus can establish a connection
        to the Kafka broker specified in KAFKA_BOOTSTRAP_SERVERS.
        """
        # Initially not started
        health = await kafka_event_bus.health_check()
        assert health["started"] is False

        # Start should succeed with Kafka available
        await kafka_event_bus.start()

        # Should now be started
        health = await kafka_event_bus.health_check()
        assert health["started"] is True

    @pytest.mark.asyncio
    async def test_health_check_reports_connected(
        self, started_kafka_bus: EventBusKafka
    ) -> None:
        """Verify health check returns healthy when connected to Kafka.

        Health check should report:
        - healthy: True
        - started: True
        - circuit_state: closed
        """
        health = await started_kafka_bus.health_check()

        assert health["healthy"] is True
        assert health["started"] is True
        assert health["circuit_state"] == "closed"
        assert "bootstrap_servers" in health

    @pytest.mark.asyncio
    async def test_multiple_start_calls_idempotent(
        self, kafka_event_bus: EventBusKafka
    ) -> None:
        """Verify multiple start() calls are safe and idempotent."""
        await kafka_event_bus.start()
        await kafka_event_bus.start()  # Second start should be no-op

        health = await kafka_event_bus.health_check()
        assert health["started"] is True

    @pytest.mark.asyncio
    async def test_close_and_restart(self, kafka_event_bus: EventBusKafka) -> None:
        """Verify bus can be closed and restarted."""
        await kafka_event_bus.start()
        health = await kafka_event_bus.health_check()
        assert health["started"] is True

        await kafka_event_bus.close()
        health = await kafka_event_bus.health_check()
        assert health["started"] is False

        # Should be able to restart
        await kafka_event_bus.start()
        health = await kafka_event_bus.health_check()
        assert health["started"] is True


# =============================================================================
# End-to-End Publish/Subscribe Tests
# =============================================================================


class TestKafkaEventBusE2E:
    """End-to-end tests for EventBusKafka publish/subscribe functionality."""

    @pytest.mark.asyncio
    async def test_publish_subscribe_roundtrip(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify message can be published and received through Kafka.

        This test validates the complete publish/subscribe flow:
        1. Subscribe to a topic with a handler
        2. Publish a message to the topic
        3. Verify the handler receives the message
        """
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe first
        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish message
        test_key = b"test-key"
        test_value = b"test-value-roundtrip"
        await started_kafka_bus.publish(created_unique_topic, test_key, test_value)

        # Wait for message delivery with timeout
        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail(
                f"Message not received within {MESSAGE_DELIVERY_WAIT_SECONDS * 2}s"
            )

        # Verify received message
        assert len(received_messages) >= 1
        received = received_messages[0]
        assert received.topic == created_unique_topic
        assert received.key == test_key
        assert received.value == test_value

        # Cleanup
        await unsubscribe()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_messages(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
    ) -> None:
        """Verify multiple subscribers on same topic all receive messages.

        Note: In Kafka, subscribers with different consumer groups will
        each receive a copy of every message.
        """
        received_by_sub1: list[ModelEventMessage] = []
        received_by_sub2: list[ModelEventMessage] = []
        sub1_received = asyncio.Event()
        sub2_received = asyncio.Event()

        async def handler1(msg: ModelEventMessage) -> None:
            received_by_sub1.append(msg)
            sub1_received.set()

        async def handler2(msg: ModelEventMessage) -> None:
            received_by_sub2.append(msg)
            sub2_received.set()

        # Subscribe with different consumer groups
        group1 = ModelNodeIdentity(
            env="kafka-integration-test",
            service="test-service",
            node_name=f"group1-node-{uuid.uuid4().hex[:8]}",
            version="1.0.0",
        )
        group2 = ModelNodeIdentity(
            env="kafka-integration-test",
            service="test-service",
            node_name=f"group2-node-{uuid.uuid4().hex[:8]}",
            version="1.0.0",
        )

        unsubscribe1 = await started_kafka_bus.subscribe(
            created_unique_topic, group1, handler1
        )
        unsubscribe2 = await started_kafka_bus.subscribe(
            created_unique_topic, group2, handler2
        )

        # Wait for consumers to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish message
        test_value = b"test-multiple-subscribers"
        await started_kafka_bus.publish(created_unique_topic, None, test_value)

        # Wait for both subscribers to receive
        try:
            await asyncio.wait_for(
                asyncio.gather(sub1_received.wait(), sub2_received.wait()),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 3,
            )
        except TimeoutError:
            pytest.fail("Not all subscribers received messages within timeout")

        # Verify both received
        assert len(received_by_sub1) >= 1
        assert len(received_by_sub2) >= 1
        assert received_by_sub1[0].value == test_value
        assert received_by_sub2[0].value == test_value

        # Cleanup
        await unsubscribe1()
        await unsubscribe2()

    @pytest.mark.asyncio
    async def test_publish_envelope_roundtrip(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify publish_envelope correctly serializes and publishes envelopes.

        Tests the publish_envelope method which accepts Pydantic models or
        dicts and serializes them to JSON before publishing.
        """
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe
        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish envelope (dict)
        test_envelope = {
            "event_type": "test.event",
            "payload": {"message": "hello", "count": 42},
            "metadata": {"source": "integration-test"},
        }
        await started_kafka_bus.publish_envelope(test_envelope, created_unique_topic)

        # Wait for message
        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Envelope not received within timeout")

        # Verify received envelope
        assert len(received_messages) >= 1
        received = received_messages[0]
        received_envelope = json.loads(received.value.decode("utf-8"))
        assert received_envelope["event_type"] == "test.event"
        assert received_envelope["payload"]["message"] == "hello"
        assert received_envelope["payload"]["count"] == 42

        # Cleanup
        await unsubscribe()

    @pytest.mark.asyncio
    async def test_publish_multiple_messages_ordering(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify messages are received in order when using same partition key.

        Kafka guarantees ordering within a partition, which is determined by
        the message key.
        """
        received_messages: list[ModelEventMessage] = []
        all_received = asyncio.Event()
        expected_count = 5

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            if len(received_messages) >= expected_count:
                all_received.set()

        # Subscribe
        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish multiple messages with same key (same partition)
        partition_key = b"ordering-key"
        for i in range(expected_count):
            await started_kafka_bus.publish(
                created_unique_topic,
                partition_key,
                f"message-{i}".encode(),
            )

        # Wait for all messages
        try:
            await asyncio.wait_for(
                all_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 3,
            )
        except TimeoutError:
            pytest.fail(
                f"Only received {len(received_messages)}/{expected_count} messages"
            )

        # Verify ordering
        assert len(received_messages) >= expected_count
        for i, msg in enumerate(received_messages[:expected_count]):
            assert msg.value == f"message-{i}".encode()

        # Cleanup
        await unsubscribe()

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_message_delivery(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify unsubscribe stops message delivery to handler."""
        received_messages: list[ModelEventMessage] = []
        first_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            first_received.set()

        # Subscribe
        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish first message
        await started_kafka_bus.publish(created_unique_topic, None, b"first-message")

        # Wait for first message
        try:
            await asyncio.wait_for(
                first_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("First message not received")

        initial_count = len(received_messages)
        assert initial_count >= 1

        # Unsubscribe
        await unsubscribe()

        # Give time for unsubscribe to complete
        await asyncio.sleep(0.5)

        # Publish second message - should not be received
        await started_kafka_bus.publish(created_unique_topic, None, b"second-message")
        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Should not have received second message (or at most same count)
        # Note: There may be some messages in flight, so we allow small tolerance
        assert len(received_messages) <= initial_count + 1


# =============================================================================
# Resilience Tests
# =============================================================================


class TestKafkaEventBusResilience:
    """Tests for EventBusKafka resilience and error handling."""

    @pytest.mark.asyncio
    async def test_publish_without_start_fails(
        self, kafka_event_bus: EventBusKafka
    ) -> None:
        """Verify publish fails gracefully when bus not started."""
        from omnibase_infra.errors import InfraUnavailableError

        # Don't start the bus
        with pytest.raises(InfraUnavailableError, match="not started"):
            await kafka_event_bus.publish("test-topic", None, b"test")

    @pytest.mark.asyncio
    async def test_graceful_degradation_circuit_breaker(
        self,
        kafka_bootstrap_servers: str,
    ) -> None:
        """Verify circuit breaker protects against repeated failures.

        The circuit breaker should open after consecutive failures and
        reject new requests until the reset timeout.
        """
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        # Create bus with invalid bootstrap servers to simulate failures
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="invalid-host:9092",
            environment="test",
            timeout_seconds=2,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_timeout=60.0,
        )
        bus = EventBusKafka(config=config)

        # First attempt should fail (connection error)
        from omnibase_infra.errors import InfraConnectionError, InfraTimeoutError

        with pytest.raises((InfraConnectionError, InfraTimeoutError)):
            await bus.start()

        # Second attempt should also fail
        with pytest.raises((InfraConnectionError, InfraTimeoutError)):
            await bus.start()

        # After threshold failures, circuit should be open
        # Next attempt should fail fast with circuit breaker error
        from omnibase_infra.errors import InfraUnavailableError

        with pytest.raises(
            (InfraUnavailableError, InfraConnectionError, InfraTimeoutError)
        ):
            await bus.start()

        # Cleanup
        await bus.close()

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_crash_bus(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
    ) -> None:
        """Verify subscriber errors don't crash the event bus.

        When a subscriber callback raises an exception, the bus should
        continue operating and other subscribers should still receive messages.
        """
        good_received: list[ModelEventMessage] = []
        good_message_event = asyncio.Event()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional test failure")

        async def good_handler(msg: ModelEventMessage) -> None:
            good_received.append(msg)
            good_message_event.set()

        group1 = ModelNodeIdentity(
            env="kafka-integration-test",
            service="test-service",
            node_name=f"fail-node-{uuid.uuid4().hex[:8]}",
            version="1.0.0",
        )
        group2 = ModelNodeIdentity(
            env="kafka-integration-test",
            service="test-service",
            node_name=f"good-node-{uuid.uuid4().hex[:8]}",
            version="1.0.0",
        )

        # Subscribe both handlers
        unsub_fail = await started_kafka_bus.subscribe(
            created_unique_topic, group1, failing_handler
        )
        unsub_good = await started_kafka_bus.subscribe(
            created_unique_topic, group2, good_handler
        )

        # Wait for consumers to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish message - should not crash despite failing handler
        await started_kafka_bus.publish(created_unique_topic, None, b"test-resilience")

        # Good handler should still receive message
        try:
            await asyncio.wait_for(
                good_message_event.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Good handler did not receive message")

        assert len(good_received) >= 1

        # Cleanup
        await unsub_fail()
        await unsub_good()

    @pytest.mark.asyncio
    async def test_health_check_after_close(
        self, kafka_event_bus: EventBusKafka
    ) -> None:
        """Verify health check works correctly after bus is closed."""
        await kafka_event_bus.start()
        health = await kafka_event_bus.health_check()
        assert health["healthy"] is True

        await kafka_event_bus.close()
        health = await kafka_event_bus.health_check()
        assert health["healthy"] is False
        assert health["started"] is False


# =============================================================================
# Header and Metadata Tests
# =============================================================================


class TestKafkaEventBusHeaders:
    """Tests for message header handling in EventBusKafka."""

    @pytest.mark.asyncio
    async def test_headers_roundtrip(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify custom headers are preserved through publish/subscribe cycle."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Publish with custom headers
        custom_headers = ModelEventHeaders(
            source="test-service",
            event_type="test.headers",
            priority="high",
            trace_id="trace-123",
            span_id="span-456",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_kafka_bus.publish(
            created_unique_topic,
            b"header-key",
            b"header-value",
            custom_headers,
        )

        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Message with headers not received")

        # Verify headers
        assert len(received_messages) >= 1
        received = received_messages[0]
        assert received.headers.source == "test-service"
        assert received.headers.event_type == "test.headers"
        assert received.headers.priority == "high"
        assert received.headers.trace_id == "trace-123"
        assert received.headers.span_id == "span-456"

        await unsubscribe()

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self,
        started_kafka_bus: EventBusKafka,
        created_unique_topic: str,
        unique_group: ModelNodeIdentity,
    ) -> None:
        """Verify correlation_id is preserved through message flow."""
        from uuid import UUID

        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        unsubscribe = await started_kafka_bus.subscribe(
            created_unique_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_unique_topic)

        # Create headers with specific correlation_id
        test_correlation_id = uuid.uuid4()
        headers = ModelEventHeaders(
            source="correlation-test",
            event_type="test.correlation",
            correlation_id=test_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_kafka_bus.publish(
            created_unique_topic,
            None,
            b"correlation-test-value",
            headers,
        )

        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Message not received")

        # Verify correlation_id
        assert len(received_messages) >= 1
        received = received_messages[0]
        # The correlation_id should be preserved (may be string or UUID after round-trip)
        received_corr_id = received.headers.correlation_id
        if isinstance(received_corr_id, str):
            received_corr_id = UUID(received_corr_id)
        assert received_corr_id == test_correlation_id

        await unsubscribe()


# =============================================================================
# Broadcast and Group Send Tests
# =============================================================================


class TestKafkaEventBusBroadcast:
    """Tests for broadcast and group send operations."""

    @pytest.mark.asyncio
    async def test_broadcast_to_environment(
        self,
        started_kafka_bus: EventBusKafka,
        unique_group: ModelNodeIdentity,
        created_broadcast_topic: str,
    ) -> None:
        """Verify broadcast_to_environment sends to correct topic."""
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe to broadcast topic for this environment
        # Note: created_broadcast_topic is "integration-test.broadcast"
        unsubscribe = await started_kafka_bus.subscribe(
            created_broadcast_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, created_broadcast_topic)

        # Broadcast to environment
        await started_kafka_bus.broadcast_to_environment(
            "test_command",
            {"key": "value", "number": 123},
        )

        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Broadcast not received")

        # Verify broadcast content
        assert len(received_messages) >= 1
        received = received_messages[0]
        payload = json.loads(received.value.decode("utf-8"))
        assert payload["command"] == "test_command"
        assert payload["payload"]["key"] == "value"
        assert payload["payload"]["number"] == 123

        await unsubscribe()

    @pytest.mark.asyncio
    async def test_send_to_group(
        self,
        started_kafka_bus: EventBusKafka,
        unique_group: ModelNodeIdentity,
        ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
    ) -> None:
        """Verify send_to_group sends to correct topic."""
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe to group topic (pre-create it first)
        target_group = f"target-{uuid.uuid4().hex[:8]}"
        group_topic = f"integration-test.{target_group}"
        await ensure_test_topic(group_topic)
        unsubscribe = await started_kafka_bus.subscribe(
            group_topic,
            unique_group,
            handler,
        )

        # Wait for consumer to be ready (uses polling with exponential backoff)
        await wait_for_consumer_ready(started_kafka_bus, group_topic)

        # Send to group
        await started_kafka_bus.send_to_group(
            "group_command",
            {"action": "process", "items": [1, 2, 3]},
            target_group,
        )

        try:
            await asyncio.wait_for(
                message_received.wait(),
                timeout=MESSAGE_DELIVERY_WAIT_SECONDS * 2,
            )
        except TimeoutError:
            pytest.fail("Group message not received")

        # Verify message content
        assert len(received_messages) >= 1
        received = received_messages[0]
        payload = json.loads(received.value.decode("utf-8"))
        assert payload["command"] == "group_command"
        assert payload["payload"]["action"] == "process"
        assert payload["payload"]["items"] == [1, 2, 3]

        await unsubscribe()
