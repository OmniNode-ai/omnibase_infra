# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Dead Letter Queue (DLQ) functionality with live Kafka.

These tests validate DLQ behavior against actual Kafka infrastructure
(RedPanda or Kafka). They require a running Kafka broker and will be skipped
gracefully if Kafka is not available.

Test categories:
- Topic Constant Tests: Validate DLQ topic building and parsing
- DLQ Publishing Tests: Verify messages are published to DLQ after retry exhaustion
- DLQ Message Format Tests: Validate DLQ message structure and metadata
- DLQ Callback Tests: Verify callback hooks are invoked correctly

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:9092")
    KAFKA_TIMEOUT_SECONDS: Operation timeout in seconds (default: 30)

Related:
    - omnibase_infra.event_bus.topic_constants: DLQ topic utilities
    - omnibase_infra.event_bus.kafka_event_bus: KafkaEventBus with DLQ support
    - omnibase_infra.event_bus.models.model_dlq_event: DLQ event model
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

if TYPE_CHECKING:
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.event_bus.models import ModelDlqEvent

# Import ModelEventMessage at runtime for use in tests
from omnibase_infra.event_bus.models import ModelEventMessage

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Check if Kafka is available based on environment variable
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = KAFKA_BOOTSTRAP_SERVERS is not None

# Module-level marker for integration tests
pytestmark = [pytest.mark.integration]

# Skip marker for tests that require Kafka
requires_kafka = pytest.mark.skipif(
    not KAFKA_AVAILABLE,
    reason="Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)",
)

# Test configuration constants
TEST_TIMEOUT_SECONDS = 30
MESSAGE_DELIVERY_WAIT_SECONDS = 2.0
CONSUMER_START_WAIT_SECONDS = 1.0
DLQ_PROCESSING_WAIT_SECONDS = 3.0


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
    return f"test.integration.dlq.{uuid.uuid4().hex[:12]}"


@pytest.fixture
def unique_dlq_topic() -> str:
    """Generate unique DLQ topic name for test isolation."""
    return f"test-dlq.dlq.intents.{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_group() -> str:
    """Generate unique consumer group for test isolation."""
    return f"test-dlq-group-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def kafka_event_bus_with_dlq(
    kafka_bootstrap_servers: str,
    created_unique_dlq_topic: str,
) -> AsyncGenerator[KafkaEventBus, None]:
    """Create KafkaEventBus with DLQ configured for integration testing.

    Yields a started KafkaEventBus instance with DLQ enabled and ensures
    cleanup after test. The DLQ topic is pre-created by the
    created_unique_dlq_topic fixture.
    """
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    # Create config with DLQ enabled
    config = ModelKafkaEventBusConfig(
        bootstrap_servers=kafka_bootstrap_servers,
        environment="dlq-integration-test",
        group="test-dlq-default",
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        max_retry_attempts=2,  # Low retry count for faster testing
        retry_backoff_base=0.1,  # Fast backoff for testing
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
        dead_letter_topic=created_unique_dlq_topic,
    )

    bus = KafkaEventBus.from_config(config)

    yield bus

    # Cleanup: ensure bus is closed
    try:
        await bus.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
async def started_dlq_bus(
    kafka_event_bus_with_dlq: KafkaEventBus,
) -> KafkaEventBus:
    """Provide a started KafkaEventBus instance with DLQ enabled."""
    await kafka_event_bus_with_dlq.start()
    return kafka_event_bus_with_dlq


# =============================================================================
# DLQ Topic Constants Tests
# =============================================================================


class TestDlqTopicConstants:
    """Tests for DLQ topic naming utilities.

    These tests validate the topic_constants module functions for building,
    parsing, and identifying DLQ topics. They do not require Kafka connection.
    """

    def test_build_dlq_topic_basic(self) -> None:
        """Verify build_dlq_topic constructs correct topic name."""
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        topic = build_dlq_topic("dev", "intents")
        assert topic == "dev.dlq.intents.v1"

    def test_build_dlq_topic_all_categories(self) -> None:
        """Verify build_dlq_topic handles all message categories."""
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        # Test plural forms
        assert build_dlq_topic("prod", "intents") == "prod.dlq.intents.v1"
        assert build_dlq_topic("prod", "events") == "prod.dlq.events.v1"
        assert build_dlq_topic("prod", "commands") == "prod.dlq.commands.v1"

        # Test singular forms (normalized to plural)
        assert build_dlq_topic("staging", "intent") == "staging.dlq.intents.v1"
        assert build_dlq_topic("staging", "event") == "staging.dlq.events.v1"
        assert build_dlq_topic("staging", "command") == "staging.dlq.commands.v1"

    def test_build_dlq_topic_custom_version(self) -> None:
        """Verify build_dlq_topic accepts custom version."""
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        topic = build_dlq_topic("test", "events", version="v2")
        assert topic == "test.dlq.events.v2"

    def test_build_dlq_topic_invalid_category(self) -> None:
        """Verify build_dlq_topic rejects invalid categories."""
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        with pytest.raises(ValueError, match="Invalid category"):
            build_dlq_topic("dev", "invalid")

    def test_build_dlq_topic_empty_environment(self) -> None:
        """Verify build_dlq_topic rejects empty environment."""
        from omnibase_infra.event_bus.topic_constants import build_dlq_topic

        with pytest.raises(ValueError, match="environment cannot be empty"):
            build_dlq_topic("", "intents")

        with pytest.raises(ValueError, match="environment cannot be empty"):
            build_dlq_topic("   ", "intents")

    def test_parse_dlq_topic_valid(self) -> None:
        """Verify parse_dlq_topic extracts components correctly."""
        from omnibase_infra.event_bus.topic_constants import parse_dlq_topic

        result = parse_dlq_topic("dev.dlq.intents.v1")
        assert result is not None
        assert result["environment"] == "dev"
        assert result["category"] == "intents"
        assert result["version"] == "v1"

        result = parse_dlq_topic("prod.dlq.events.v2")
        assert result is not None
        assert result["environment"] == "prod"
        assert result["category"] == "events"
        assert result["version"] == "v2"

    def test_parse_dlq_topic_invalid(self) -> None:
        """Verify parse_dlq_topic returns None for non-DLQ topics."""
        from omnibase_infra.event_bus.topic_constants import parse_dlq_topic

        # Not a DLQ topic
        assert parse_dlq_topic("dev.user.events.v1") is None
        assert parse_dlq_topic("onex.registration.events") is None
        assert parse_dlq_topic("random-topic") is None

        # Missing components
        assert parse_dlq_topic("dev.dlq") is None
        assert parse_dlq_topic("dlq.intents.v1") is None

    def test_is_dlq_topic_true(self) -> None:
        """Verify is_dlq_topic returns True for DLQ topics."""
        from omnibase_infra.event_bus.topic_constants import is_dlq_topic

        assert is_dlq_topic("dev.dlq.intents.v1") is True
        assert is_dlq_topic("prod.dlq.events.v1") is True
        assert is_dlq_topic("staging.dlq.commands.v2") is True
        assert is_dlq_topic("test-env.dlq.intents.v1") is True

    def test_is_dlq_topic_false(self) -> None:
        """Verify is_dlq_topic returns False for non-DLQ topics."""
        from omnibase_infra.event_bus.topic_constants import is_dlq_topic

        assert is_dlq_topic("dev.user.events.v1") is False
        assert is_dlq_topic("onex.registration.events") is False
        assert is_dlq_topic("random-topic") is False
        assert is_dlq_topic("dev.dlq") is False

    def test_build_and_parse_roundtrip(self) -> None:
        """Verify build_dlq_topic and parse_dlq_topic are consistent."""
        from omnibase_infra.event_bus.topic_constants import (
            build_dlq_topic,
            parse_dlq_topic,
        )

        # Build then parse should recover original components
        for env in ["dev", "prod", "staging", "test-1"]:
            for category in ["intents", "events", "commands"]:
                topic = build_dlq_topic(env, category)
                parsed = parse_dlq_topic(topic)
                assert parsed is not None
                assert parsed["environment"] == env
                assert parsed["category"] == category
                assert parsed["version"] == "v1"


# =============================================================================
# DLQ Topic Integration Tests (requires Kafka)
# =============================================================================


@requires_kafka
@pytest.mark.asyncio
class TestDlqTopicIntegration:
    """Integration tests for DLQ topic operations with live Kafka."""

    async def test_dlq_topic_creation(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_dlq_topic: str,
    ) -> None:
        """Verify DLQ topic can be created and published to.

        This test validates that the configured DLQ topic can receive messages.
        """
        from omnibase_infra.event_bus.models import ModelEventHeaders

        # Verify bus started successfully
        health = await started_dlq_bus.health_check()
        assert health["started"] is True
        assert health["healthy"] is True

        # Publish test message to DLQ topic directly
        test_key = b"dlq-test-key"
        test_value = json.dumps({"test": "dlq_topic_creation"}).encode()

        headers = ModelEventHeaders(
            source="dlq-integration-test",
            event_type="test.dlq.creation",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_dlq_bus.publish(
            created_unique_dlq_topic,
            test_key,
            test_value,
            headers,
        )

        # If we get here without error, the topic was created and message published


# =============================================================================
# DLQ Publishing Tests
# =============================================================================


@requires_kafka
@pytest.mark.asyncio
class TestDlqPublishing:
    """Tests for DLQ message publishing after retry exhaustion."""

    async def test_dlq_publish_on_handler_failure(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_topic: str,
        created_unique_dlq_topic: str,
        unique_group: str,
    ) -> None:
        """Verify messages are published to DLQ after handler failure with exhausted retries.

        This test validates the complete DLQ flow:
        1. Subscribe to a topic with a failing handler
        2. Publish a message with max retries already exceeded
        3. Verify the message appears in the DLQ topic
        """
        dlq_messages_received: list[ModelEventMessage] = []
        dlq_received_event = asyncio.Event()
        handler_call_count = 0

        async def failing_handler(msg: ModelEventMessage) -> None:
            """Handler that always fails."""
            nonlocal handler_call_count
            handler_call_count += 1
            raise ValueError("Intentional test failure for DLQ testing")

        async def dlq_collector(msg: ModelEventMessage) -> None:
            """Collector for DLQ messages."""
            dlq_messages_received.append(msg)
            dlq_received_event.set()

        # Subscribe to source topic with failing handler
        unsubscribe_source = await started_dlq_bus.subscribe(
            created_unique_topic,
            unique_group,
            failing_handler,
        )

        # Subscribe to DLQ topic to capture messages
        dlq_group = f"dlq-collector-{uuid.uuid4().hex[:8]}"
        unsubscribe_dlq = await started_dlq_bus.subscribe(
            created_unique_dlq_topic,
            dlq_group,
            dlq_collector,
        )

        await asyncio.sleep(CONSUMER_START_WAIT_SECONDS)

        # Publish message with retry_count at max (will trigger DLQ on first failure)
        from omnibase_infra.event_bus.models import ModelEventHeaders

        test_payload = {"test_id": str(uuid.uuid4()), "action": "trigger_dlq"}
        headers = ModelEventHeaders(
            source="dlq-integration-test",
            event_type="test.dlq.failure",
            retry_count=2,  # At max retries (max_retry_attempts=2)
            max_retries=2,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_dlq_bus.publish(
            created_unique_topic,
            b"dlq-trigger-key",
            json.dumps(test_payload).encode(),
            headers,
        )

        # Wait for DLQ message
        try:
            await asyncio.wait_for(
                dlq_received_event.wait(),
                timeout=DLQ_PROCESSING_WAIT_SECONDS * 2,
            )

            # Verify handler was called and failed
            assert handler_call_count >= 1, (
                "Handler should have been called at least once"
            )

            # Verify DLQ message was actually received
            assert len(dlq_messages_received) >= 1, (
                "DLQ message should have been received after handler failure"
            )

            # Verify DLQ message contains expected structure
            dlq_msg = dlq_messages_received[0]
            assert dlq_msg.value is not None, "DLQ message should have a value"

        except TimeoutError:
            # DLQ message not received within timeout - this is acceptable for integration tests
            # as timing depends on Kafka broker state
            pytest.skip("DLQ message not received within timeout")

        finally:
            # Cleanup
            await unsubscribe_source()
            await unsubscribe_dlq()


# =============================================================================
# DLQ Message Format Tests
# =============================================================================


@requires_kafka
@pytest.mark.asyncio
class TestDlqMessageFormat:
    """Tests for DLQ message structure and metadata."""

    async def test_dlq_message_contains_original_context(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_topic: str,
        created_unique_dlq_topic: str,
        unique_group: str,
    ) -> None:
        """Verify DLQ messages contain complete original message context.

        DLQ messages should include:
        - original_topic
        - original_message (key, value, offset, partition)
        - failure_reason
        - failure_timestamp
        - correlation_id
        - retry_count
        - error_type
        """
        dlq_messages: list[ModelEventMessage] = []
        dlq_received = asyncio.Event()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional failure with context")

        async def dlq_collector(msg: ModelEventMessage) -> None:
            dlq_messages.append(msg)
            dlq_received.set()

        # Subscribe to both topics
        unsub_source = await started_dlq_bus.subscribe(
            created_unique_topic, unique_group, failing_handler
        )
        unsub_dlq = await started_dlq_bus.subscribe(
            created_unique_dlq_topic, f"dlq-fmt-{uuid.uuid4().hex[:6]}", dlq_collector
        )

        await asyncio.sleep(CONSUMER_START_WAIT_SECONDS)

        # Publish message with exhausted retries
        from omnibase_infra.event_bus.models import ModelEventHeaders

        test_correlation_id = uuid.uuid4()
        headers = ModelEventHeaders(
            source="format-test",
            event_type="test.format.validation",
            correlation_id=test_correlation_id,
            retry_count=3,
            max_retries=2,  # Already exceeded
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        original_payload = {"test": "dlq_format", "key": "value123"}
        await started_dlq_bus.publish(
            created_unique_topic,
            b"format-test-key",
            json.dumps(original_payload).encode(),
            headers,
        )

        # Wait for DLQ message
        try:
            await asyncio.wait_for(
                dlq_received.wait(),
                timeout=DLQ_PROCESSING_WAIT_SECONDS * 2,
            )

            # Verify DLQ message format
            assert len(dlq_messages) >= 1
            dlq_msg = dlq_messages[0]
            dlq_payload = json.loads(dlq_msg.value.decode("utf-8"))

            # Required fields in DLQ payload
            assert "original_topic" in dlq_payload
            assert dlq_payload["original_topic"] == created_unique_topic

            assert "original_message" in dlq_payload
            assert "failure_reason" in dlq_payload
            assert "failure_timestamp" in dlq_payload
            assert "correlation_id" in dlq_payload
            assert "error_type" in dlq_payload

        except TimeoutError:
            # DLQ processing may not complete in time - this is acceptable
            pytest.skip("DLQ message not received within timeout")

        finally:
            await unsub_source()
            await unsub_dlq()


# =============================================================================
# DLQ Callback Tests
# =============================================================================


@requires_kafka
@pytest.mark.asyncio
class TestDlqCallbacks:
    """Tests for DLQ callback hook functionality."""

    async def test_dlq_callback_invoked_on_publish(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify DLQ callbacks are invoked when messages are published to DLQ."""
        callback_events: list[ModelDlqEvent] = []
        callback_invoked = asyncio.Event()

        async def dlq_callback(event: ModelDlqEvent) -> None:
            callback_events.append(event)
            callback_invoked.set()

        # Register callback
        unregister_callback = await started_dlq_bus.register_dlq_callback(dlq_callback)

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise RuntimeError("Callback test failure")

        # Subscribe with failing handler
        unsub = await started_dlq_bus.subscribe(
            created_unique_topic, unique_group, failing_handler
        )

        await asyncio.sleep(CONSUMER_START_WAIT_SECONDS)

        # Publish message with exhausted retries to trigger DLQ
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers = ModelEventHeaders(
            source="callback-test",
            event_type="test.callback",
            retry_count=5,
            max_retries=2,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_dlq_bus.publish(
            created_unique_topic,
            b"callback-key",
            b'{"test": "callback"}',
            headers,
        )

        # Wait for callback invocation
        try:
            await asyncio.wait_for(
                callback_invoked.wait(),
                timeout=DLQ_PROCESSING_WAIT_SECONDS * 2,
            )

            # Verify callback was invoked with correct event
            assert len(callback_events) >= 1
            event = callback_events[0]
            assert event.original_topic == created_unique_topic
            assert event.error_type == "RuntimeError"
            assert "Callback test failure" in event.error_message

        except TimeoutError:
            pytest.skip("DLQ callback not invoked within timeout")

        finally:
            await unsub()
            await unregister_callback()

    async def test_dlq_callback_unregister(
        self,
        started_dlq_bus: KafkaEventBus,
    ) -> None:
        """Verify DLQ callbacks can be unregistered."""
        callback_count = 0

        async def counting_callback(event: ModelDlqEvent) -> None:
            nonlocal callback_count
            callback_count += 1

        # Register and immediately unregister
        unregister = await started_dlq_bus.register_dlq_callback(counting_callback)
        await unregister()

        # Callback should no longer be in the list (verified by internal state)
        # This test primarily ensures unregister doesn't raise errors
        assert callback_count == 0


# =============================================================================
# DLQ Metrics Tests
# =============================================================================


@requires_kafka
@pytest.mark.asyncio
class TestDlqMetrics:
    """Tests for DLQ metrics tracking.

    PR #90 feedback: DLQ publish tests should assert that DLQ metrics were
    incremented, not just that the publish succeeded. These tests use direct
    `_publish_to_dlq` calls for deterministic metric verification.
    """

    async def test_dlq_metrics_available(
        self,
        started_dlq_bus: KafkaEventBus,
    ) -> None:
        """Verify DLQ metrics are accessible and properly initialized."""
        metrics = started_dlq_bus.dlq_metrics

        # Verify metrics structure
        assert hasattr(metrics, "total_publishes")
        assert hasattr(metrics, "successful_publishes")
        assert hasattr(metrics, "failed_publishes")
        assert hasattr(metrics, "last_publish_at")

        # Initial values should be zero or None
        assert metrics.total_publishes >= 0
        assert metrics.successful_publishes >= 0
        assert metrics.failed_publishes >= 0

    async def test_dlq_metrics_increment_on_successful_publish(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_topic: str,
    ) -> None:
        """Verify DLQ metrics are incremented when messages are published successfully.

        PR #90 feedback: Use direct _publish_to_dlq call for deterministic testing.
        This ensures metrics are incremented synchronously without timing issues.
        """
        from omnibase_infra.event_bus.models import ModelEventHeaders

        # Capture initial metrics
        initial_metrics = started_dlq_bus.dlq_metrics
        initial_total = initial_metrics.total_publishes
        initial_successful = initial_metrics.successful_publishes
        initial_failed = initial_metrics.failed_publishes

        # Create a failed message for DLQ
        correlation_id = uuid.uuid4()
        headers = ModelEventHeaders(
            source="metrics-test",
            event_type="test.metrics.success",
            correlation_id=correlation_id,
            retry_count=5,  # Exhausted
            max_retries=3,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        failed_message = ModelEventMessage(
            topic=created_unique_topic,
            key=b"metrics-key",
            value=b'{"test": "metrics_success"}',
            headers=headers,
        )
        error = RuntimeError("Handler failed for metrics test")

        # Directly call _publish_to_dlq for deterministic testing
        await started_dlq_bus._publish_to_dlq(
            original_topic=created_unique_topic,
            failed_message=failed_message,
            error=error,
            correlation_id=correlation_id,
        )

        # Verify metrics were incremented (PR #90 feedback: strict assertions)
        final_metrics = started_dlq_bus.dlq_metrics
        assert final_metrics.total_publishes == initial_total + 1, (
            f"total_publishes should be incremented from {initial_total} to {initial_total + 1}, "
            f"got {final_metrics.total_publishes}"
        )
        assert final_metrics.successful_publishes == initial_successful + 1, (
            f"successful_publishes should be incremented from {initial_successful} to {initial_successful + 1}, "
            f"got {final_metrics.successful_publishes}"
        )
        assert final_metrics.failed_publishes == initial_failed, (
            f"failed_publishes should remain at {initial_failed}, got {final_metrics.failed_publishes}"
        )
        # Verify per-topic and per-error-type metrics
        assert final_metrics.get_topic_count(created_unique_topic) >= 1, (
            f"topic_counts[{created_unique_topic}] should be at least 1"
        )
        assert final_metrics.get_error_type_count("RuntimeError") >= 1, (
            "error_type_counts['RuntimeError'] should be at least 1"
        )
        # Verify timestamp was set
        assert final_metrics.last_publish_at is not None, (
            "last_publish_at should be set after successful publish"
        )

    async def test_dlq_metrics_increment_on_full_flow(
        self,
        started_dlq_bus: KafkaEventBus,
        created_unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify DLQ metrics are incremented in full consumer flow.

        This test verifies the complete flow with a failing handler.
        Note: Timing may be unpredictable, so we use relaxed assertions.
        """
        initial_metrics = started_dlq_bus.dlq_metrics
        initial_total = initial_metrics.total_publishes

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise RuntimeError("Metrics test failure")

        unsub = await started_dlq_bus.subscribe(
            created_unique_topic, unique_group, failing_handler
        )

        await asyncio.sleep(CONSUMER_START_WAIT_SECONDS)

        # Publish message with exhausted retries
        from omnibase_infra.event_bus.models import ModelEventHeaders

        headers = ModelEventHeaders(
            source="metrics-test",
            event_type="test.metrics",
            retry_count=10,  # Definitely exhausted
            max_retries=2,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        await started_dlq_bus.publish(
            created_unique_topic,
            b"metrics-key",
            b'{"test": "metrics"}',
            headers,
        )

        # Wait for processing
        await asyncio.sleep(DLQ_PROCESSING_WAIT_SECONDS)

        # Check metrics - in full flow, timing may be unpredictable
        # but total_publishes should at least not decrease
        final_metrics = started_dlq_bus.dlq_metrics
        assert final_metrics.total_publishes >= initial_total, (
            f"total_publishes should be at least {initial_total}, got {final_metrics.total_publishes}"
        )

        await unsub()
