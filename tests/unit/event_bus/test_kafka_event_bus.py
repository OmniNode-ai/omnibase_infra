# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for KafkaEventBus.

Comprehensive test suite covering all public methods, edge cases,
error handling, and circuit breaker functionality with mocked Kafka dependencies.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from aiokafka.errors import KafkaError

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class TestKafkaEventBusLifecycle:
    """Test suite for event bus lifecycle management."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer.send = AsyncMock()
        producer._closed = False
        return producer

    @pytest.fixture
    async def kafka_event_bus(self, mock_producer: AsyncMock) -> KafkaEventBus:
        """Create KafkaEventBus with mocked producer."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                group="test-group",
            )
            yield bus
            # Cleanup: Ensure resources are freed even if test fails
            try:
                await bus.close()
            except Exception:
                pass  # Best effort cleanup

    @pytest.mark.asyncio
    async def test_start_and_close(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test bus lifecycle - start and close operations."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            # Initially not started
            health = await kafka_event_bus.health_check()
            assert health["healthy"] is False
            assert health["started"] is False

            # Start the bus
            await kafka_event_bus.start()
            mock_producer.start.assert_called_once()
            health = await kafka_event_bus.health_check()
            assert health["started"] is True

            # Close the bus
            await kafka_event_bus.close()
            mock_producer.stop.assert_called_once()
            health = await kafka_event_bus.health_check()
            assert health["healthy"] is False
            assert health["started"] is False

    @pytest.mark.asyncio
    async def test_multiple_start_calls(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test that multiple start calls are safe (idempotent)."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()
            await kafka_event_bus.start()  # Second start should be idempotent

            # Producer.start should only be called once
            assert mock_producer.start.call_count == 1

            health = await kafka_event_bus.health_check()
            assert health["started"] is True

            await kafka_event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_close_calls(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test that multiple close calls are safe (idempotent)."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()
            await kafka_event_bus.close()
            await kafka_event_bus.close()  # Second close should be idempotent

            health = await kafka_event_bus.health_check()
            assert health["started"] is False

    @pytest.mark.asyncio
    async def test_shutdown_alias(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test shutdown() is an alias for close()."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()
            await kafka_event_bus.shutdown()

            health = await kafka_event_bus.health_check()
            assert health["started"] is False

    @pytest.mark.asyncio
    async def test_initialize_with_config(self, mock_producer: AsyncMock) -> None:
        """Test initialize() method with configuration override."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus()
            await event_bus.initialize(
                {
                    "environment": "production",
                    "group": "prod-group",
                    "bootstrap_servers": "kafka.prod:9092",
                    "timeout_seconds": 60,
                }
            )

            assert event_bus.environment == "production"
            assert event_bus.group == "prod-group"
            health = await event_bus.health_check()
            assert health["started"] is True

            await event_bus.close()


class TestKafkaEventBusProperties:
    """Test suite for event bus properties."""

    def test_default_properties(self) -> None:
        """Test default property values."""
        event_bus = KafkaEventBus()
        assert event_bus.environment == "local"
        assert event_bus.group == "default"
        assert event_bus.adapter is event_bus

    def test_custom_properties(self) -> None:
        """Test custom property values."""
        event_bus = KafkaEventBus(
            bootstrap_servers="kafka.staging:9092",
            environment="staging",
            group="worker-group",
            timeout_seconds=60,
            max_retry_attempts=5,
            retry_backoff_base=2.0,
        )
        assert event_bus.environment == "staging"
        assert event_bus.group == "worker-group"
        assert event_bus.adapter is event_bus

    def test_adapter_returns_self(self) -> None:
        """Test adapter property returns self."""
        event_bus = KafkaEventBus()
        assert event_bus.adapter is event_bus


class TestKafkaEventBusPublish:
    """Test suite for publish operations."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False

        # Mock the send method to return a future-like object
        mock_record_metadata = MagicMock()
        mock_record_metadata.partition = 0
        mock_record_metadata.offset = 42

        async def mock_send(*args, **kwargs):
            future = asyncio.get_running_loop().create_future()
            future.set_result(mock_record_metadata)
            return future

        producer.send = AsyncMock(side_effect=mock_send)
        return producer

    @pytest.fixture
    async def kafka_event_bus(self, mock_producer: AsyncMock) -> KafkaEventBus:
        """Create KafkaEventBus with mocked producer."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                group="test-group",
                max_retry_attempts=0,  # Disable retries for faster tests
            )
            yield bus
            # Cleanup: Ensure resources are freed even if test fails
            try:
                await bus.close()
            except Exception:
                pass  # Best effort cleanup

    @pytest.mark.asyncio
    async def test_publish_requires_start(self, kafka_event_bus: KafkaEventBus) -> None:
        """Test that publish fails if bus not started."""
        with pytest.raises(InfraUnavailableError, match="not started"):
            await kafka_event_bus.publish("test-topic", None, b"test")

    @pytest.mark.asyncio
    async def test_publish_basic(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test basic publish operation (mocked producer)."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()

            await kafka_event_bus.publish("test-topic", b"key1", b"value1")

            # Verify producer.send was called
            mock_producer.send.assert_called_once()
            call_args = mock_producer.send.call_args
            assert call_args[0][0] == "test-topic"  # topic
            assert call_args[1]["value"] == b"value1"  # value
            assert call_args[1]["key"] == b"key1"  # key

            await kafka_event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_with_none_key(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test publish with None key."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()

            await kafka_event_bus.publish("test-topic", None, b"value")

            # Verify producer.send was called with None key
            call_args = mock_producer.send.call_args
            assert call_args[1]["key"] is None

            await kafka_event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_with_custom_headers(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test publish with custom headers."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()

            headers = ModelEventHeaders(
                source="custom-source",
                event_type="custom-event",
                priority="high",
                timestamp=datetime.now(UTC),
            )
            await kafka_event_bus.publish("test-topic", None, b"value", headers)

            # Verify producer.send was called with headers
            call_args = mock_producer.send.call_args
            kafka_headers = call_args[1]["headers"]
            assert kafka_headers is not None
            # Find the source header
            source_header = next((h for h in kafka_headers if h[0] == "source"), None)
            assert source_header is not None
            assert source_header[1] == b"custom-source"

            await kafka_event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_circuit_breaker_open(self, mock_producer: AsyncMock) -> None:
        """Test error when circuit breaker is open."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                group="test-group",
                circuit_breaker_threshold=1,  # Open after 1 failure
            )
            await event_bus.start()

            # Record a failure to open the circuit
            async with event_bus._circuit_breaker_lock:
                await event_bus._record_circuit_failure(operation="test")

            # Verify circuit is open
            async with event_bus._circuit_breaker_lock:
                assert event_bus._circuit_breaker_open is True

            with pytest.raises(InfraUnavailableError, match="Circuit breaker is open"):
                await event_bus.publish("test-topic", None, b"test")

            await event_bus.close()


class TestKafkaEventBusSubscribe:
    """Test suite for subscribe operations."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.fixture
    async def kafka_event_bus(self, mock_producer: AsyncMock) -> KafkaEventBus:
        """Create KafkaEventBus with mocked producer."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                group="test-group",
            )
            yield bus
            # Cleanup: Ensure resources are freed even if test fails
            try:
                await bus.close()
            except Exception:
                pass  # Best effort cleanup

    @pytest.mark.asyncio
    async def test_subscribe_returns_unsubscribe_function(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test that subscribe returns an unsubscribe callable."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            # Don't start the bus - subscribe should still work for registration
            async def handler(msg: ModelEventMessage) -> None:
                pass

            unsubscribe = await kafka_event_bus.subscribe(
                "test-topic", "group1", handler
            )

            # Verify unsubscribe is a callable
            assert callable(unsubscribe)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test unsubscribe removes handler from registry."""

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsubscribe = await kafka_event_bus.subscribe("test-topic", "group1", handler)

        # Verify subscription exists
        assert len(kafka_event_bus._subscribers["test-topic"]) == 1

        await unsubscribe()

        # Verify subscription was removed
        assert len(kafka_event_bus._subscribers.get("test-topic", [])) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_topic(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test multiple subscribers on same topic."""

        async def handler1(msg: ModelEventMessage) -> None:
            pass

        async def handler2(msg: ModelEventMessage) -> None:
            pass

        await kafka_event_bus.subscribe("test-topic", "group1", handler1)
        await kafka_event_bus.subscribe("test-topic", "group2", handler2)

        # Verify both subscriptions exist
        assert len(kafka_event_bus._subscribers["test-topic"]) == 2

    @pytest.mark.asyncio
    async def test_double_unsubscribe_safe(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test that double unsubscribe is safe."""

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsubscribe = await kafka_event_bus.subscribe("test-topic", "group1", handler)
        await unsubscribe()
        await unsubscribe()  # Should not raise


class TestKafkaEventBusHealthCheck:
    """Test suite for health check operations."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.fixture
    async def kafka_event_bus(self, mock_producer: AsyncMock) -> KafkaEventBus:
        """Create KafkaEventBus with mocked producer."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                group="test-group",
            )
            yield bus
            # Cleanup: Ensure resources are freed even if test fails
            try:
                await bus.close()
            except Exception:
                pass  # Best effort cleanup

    @pytest.mark.asyncio
    async def test_health_check_not_started(
        self, kafka_event_bus: KafkaEventBus
    ) -> None:
        """Test health check when not started."""
        health = await kafka_event_bus.health_check()

        assert health["healthy"] is False
        assert health["started"] is False
        assert health["environment"] == "test"
        assert health["group"] == "test-group"
        assert health["bootstrap_servers"] == "localhost:9092"
        assert health["subscriber_count"] == 0
        assert health["topic_count"] == 0
        assert health["consumer_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_started(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test health check when started."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()
            health = await kafka_event_bus.health_check()

            assert health["started"] is True
            # healthy depends on producer not being closed
            assert health["healthy"] is True

            await kafka_event_bus.close()

    @pytest.mark.asyncio
    async def test_health_check_circuit_breaker_status(
        self, kafka_event_bus: KafkaEventBus, mock_producer: AsyncMock
    ) -> None:
        """Test health check includes circuit breaker status."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await kafka_event_bus.start()
            health = await kafka_event_bus.health_check()

            assert health["circuit_state"] == "closed"

            # Record failures to change circuit state
            async with kafka_event_bus._circuit_breaker_lock:
                kafka_event_bus._circuit_breaker_failures = 5
                kafka_event_bus._circuit_breaker_open = True

            health = await kafka_event_bus.health_check()
            assert health["circuit_state"] == "open"

            await kafka_event_bus.close()


class TestKafkaEventBusCircuitBreaker:
    """Test suite for circuit breaker functionality."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    def test_circuit_breaker_threshold_validation(self) -> None:
        """Test that invalid circuit_breaker_threshold raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="positive integer"):
            KafkaEventBus(circuit_breaker_threshold=0)

        with pytest.raises(ProtocolConfigurationError, match="positive integer"):
            KafkaEventBus(circuit_breaker_threshold=-1)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test circuit breaker opens after consecutive failures."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                circuit_breaker_threshold=3,
            )

            # Record failures
            async with event_bus._circuit_breaker_lock:
                await event_bus._record_circuit_failure(operation="test")
                assert event_bus._circuit_breaker_open is False
                assert event_bus._circuit_breaker_failures == 1

                await event_bus._record_circuit_failure(operation="test")
                assert event_bus._circuit_breaker_open is False
                assert event_bus._circuit_breaker_failures == 2

                await event_bus._record_circuit_failure(operation="test")
                # Should be open after 3 failures
                assert event_bus._circuit_breaker_open is True
                assert event_bus._circuit_breaker_failures == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test circuit breaker resets after successful operation."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                circuit_breaker_threshold=5,
            )

            # Record some failures
            async with event_bus._circuit_breaker_lock:
                await event_bus._record_circuit_failure(operation="test")
                await event_bus._record_circuit_failure(operation="test")
                assert event_bus._circuit_breaker_failures == 2

                # Reset on success
                await event_bus._reset_circuit_breaker()

                assert event_bus._circuit_breaker_open is False
                assert event_bus._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test circuit breaker transitions to half-open state."""

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                circuit_breaker_threshold=1,
                circuit_breaker_reset_timeout=0.1,  # Very short for testing
            )

            # Open the circuit
            async with event_bus._circuit_breaker_lock:
                await event_bus._record_circuit_failure(operation="test")
                assert event_bus._circuit_breaker_open is True

            # Wait for reset timeout
            await asyncio.sleep(0.15)

            # Check circuit breaker - should transition to half-open (circuit closes)
            async with event_bus._circuit_breaker_lock:
                await event_bus._check_circuit_breaker(operation="test")
                # After timeout, circuit transitions from OPEN to HALF_OPEN, which sets _circuit_breaker_open = False
                assert event_bus._circuit_breaker_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test circuit breaker blocks operations when open."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                circuit_breaker_threshold=1,
                circuit_breaker_reset_timeout=60,  # Long timeout
            )

            # Open the circuit
            async with event_bus._circuit_breaker_lock:
                await event_bus._record_circuit_failure(operation="test")
                assert event_bus._circuit_breaker_open is True

            # Should raise when checking circuit
            async with event_bus._circuit_breaker_lock:
                with pytest.raises(
                    InfraUnavailableError, match="Circuit breaker is open"
                ):
                    await event_bus._check_circuit_breaker(operation="test")


class TestKafkaEventBusErrors:
    """Test suite for error handling."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.mark.asyncio
    async def test_connection_error_type(self, mock_producer: AsyncMock) -> None:
        """Test that connection errors are properly typed."""
        mock_producer.start = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")

            with pytest.raises(InfraConnectionError) as exc_info:
                await event_bus.start()

            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unavailable_error_when_not_started(self) -> None:
        """Test InfraUnavailableError raised when bus not started."""
        event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")

        with pytest.raises(InfraUnavailableError) as exc_info:
            await event_bus.publish("test-topic", None, b"test")

        assert "not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, mock_producer: AsyncMock) -> None:
        """Test timeout error handling on start."""
        mock_producer.start = AsyncMock(side_effect=TimeoutError("Connection timeout"))

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                timeout_seconds=5,
            )

            with pytest.raises(InfraTimeoutError) as exc_info:
                await event_bus.start()

            assert "Timeout" in str(exc_info.value)


class TestKafkaEventBusPublishRetry:
    """Test suite for publish retry functionality."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer that can fail."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.mark.asyncio
    async def test_publish_retries_on_kafka_error(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test publish retries on KafkaError."""
        # Create a mock that fails twice then succeeds
        call_count = 0
        mock_record_metadata = MagicMock()
        mock_record_metadata.partition = 0
        mock_record_metadata.offset = 42

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise KafkaError("Temporary error")
            future = asyncio.get_running_loop().create_future()
            future.set_result(mock_record_metadata)
            return future

        mock_producer.send = AsyncMock(side_effect=mock_send)

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                max_retry_attempts=3,
                retry_backoff_base=0.01,  # Fast retries for testing
            )
            await event_bus.start()

            # This should succeed after retries
            await event_bus.publish("test-topic", None, b"test")

            # Verify send was called 3 times (2 failures + 1 success)
            assert call_count == 3

            await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_fails_after_all_retries(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test publish fails after exhausting all retries."""

        async def mock_send(*args, **kwargs):
            raise KafkaError("Persistent error")

        mock_producer.send = AsyncMock(side_effect=mock_send)

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                max_retry_attempts=2,
                retry_backoff_base=0.01,  # Fast retries for testing
            )
            await event_bus.start()

            with pytest.raises(InfraConnectionError) as exc_info:
                await event_bus.publish("test-topic", None, b"test")

            assert "after 3 attempts" in str(exc_info.value)  # initial + 2 retries

            await event_bus.close()


class TestKafkaEventBusPublishEnvelope:
    """Test suite for publish_envelope operation."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False

        mock_record_metadata = MagicMock()
        mock_record_metadata.partition = 0
        mock_record_metadata.offset = 42

        async def mock_send(*args, **kwargs):
            future = asyncio.get_running_loop().create_future()
            future.set_result(mock_record_metadata)
            return future

        producer.send = AsyncMock(side_effect=mock_send)
        return producer

    @pytest.mark.asyncio
    async def test_publish_envelope_with_pydantic_model(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test publish_envelope with a Pydantic model."""
        from pydantic import BaseModel

        class TestEnvelope(BaseModel):
            message: str
            count: int

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                max_retry_attempts=0,
            )
            await event_bus.start()

            envelope = TestEnvelope(message="hello", count=42)
            await event_bus.publish_envelope(envelope, "test-topic")

            # Verify the payload was serialized
            call_args = mock_producer.send.call_args
            value = call_args[1]["value"]
            payload = json.loads(value)
            assert payload["message"] == "hello"
            assert payload["count"] == 42

            await event_bus.close()

    @pytest.mark.asyncio
    async def test_publish_envelope_with_dict(self, mock_producer: AsyncMock) -> None:
        """Test publish_envelope with a plain dict."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                max_retry_attempts=0,
            )
            await event_bus.start()

            envelope = {"message": "hello", "count": 42}
            await event_bus.publish_envelope(envelope, "test-topic")

            # Verify the payload was serialized
            call_args = mock_producer.send.call_args
            value = call_args[1]["value"]
            payload = json.loads(value)
            assert payload["message"] == "hello"
            assert payload["count"] == 42

            await event_bus.close()


class TestKafkaEventBusBroadcast:
    """Test suite for broadcast and group send operations."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False

        mock_record_metadata = MagicMock()
        mock_record_metadata.partition = 0
        mock_record_metadata.offset = 42

        async def mock_send(*args, **kwargs):
            future = asyncio.get_running_loop().create_future()
            future.set_result(mock_record_metadata)
            return future

        producer.send = AsyncMock(side_effect=mock_send)
        return producer

    @pytest.mark.asyncio
    async def test_broadcast_to_environment(self, mock_producer: AsyncMock) -> None:
        """Test broadcast_to_environment publishes to correct topic."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                max_retry_attempts=0,
            )
            await event_bus.start()

            await event_bus.broadcast_to_environment("test_cmd", {"key": "value"})

            # Verify the topic is correct
            call_args = mock_producer.send.call_args
            assert call_args[0][0] == "test.broadcast"

            # Verify payload
            value = call_args[1]["value"]
            payload = json.loads(value)
            assert payload["command"] == "test_cmd"
            assert payload["payload"] == {"key": "value"}

            await event_bus.close()

    @pytest.mark.asyncio
    async def test_broadcast_to_specific_environment(
        self, mock_producer: AsyncMock
    ) -> None:
        """Test broadcast to a specific target environment."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                max_retry_attempts=0,
            )
            await event_bus.start()

            await event_bus.broadcast_to_environment(
                "deploy_cmd", {"version": "1.0"}, target_environment="production"
            )

            # Verify the topic is correct
            call_args = mock_producer.send.call_args
            assert call_args[0][0] == "production.broadcast"

            await event_bus.close()

    @pytest.mark.asyncio
    async def test_send_to_group(self, mock_producer: AsyncMock) -> None:
        """Test send_to_group publishes to correct topic."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(
                bootstrap_servers="localhost:9092",
                environment="test",
                max_retry_attempts=0,
            )
            await event_bus.start()

            await event_bus.send_to_group("test_cmd", {"key": "value"}, "target-group")

            # Verify the topic is correct
            call_args = mock_producer.send.call_args
            assert call_args[0][0] == "test.target-group"

            # Verify payload
            value = call_args[1]["value"]
            payload = json.loads(value)
            assert payload["command"] == "test_cmd"
            assert payload["payload"] == {"key": "value"}

            await event_bus.close()


class TestKafkaEventBusHeaderConversion:
    """Test suite for header conversion methods."""

    def test_model_headers_to_kafka(self) -> None:
        """Test conversion of ModelEventHeaders to Kafka format."""
        event_bus = KafkaEventBus()

        headers = ModelEventHeaders(
            source="test-source",
            event_type="test-event",
            priority="high",
            routing_key="test.route",
            timestamp=datetime.now(UTC),
        )

        kafka_headers = event_bus._model_headers_to_kafka(headers)

        # Verify it's a list of tuples
        assert isinstance(kafka_headers, list)
        assert all(isinstance(h, tuple) for h in kafka_headers)

        # Verify required headers exist
        header_dict = dict(kafka_headers)
        assert header_dict["source"] == b"test-source"
        assert header_dict["event_type"] == b"test-event"
        assert header_dict["priority"] == b"high"
        assert header_dict["routing_key"] == b"test.route"

    def test_kafka_headers_to_model(self) -> None:
        """Test conversion of Kafka headers to ModelEventHeaders."""
        event_bus = KafkaEventBus()

        kafka_headers = [
            ("content_type", b"application/json"),
            ("source", b"test-source"),
            ("event_type", b"test-event"),
            ("schema_version", b"2.0.0"),
        ]

        headers = event_bus._kafka_headers_to_model(kafka_headers)

        assert headers.content_type == "application/json"
        assert headers.source == "test-source"
        assert headers.event_type == "test-event"
        assert headers.schema_version == "2.0.0"

    def test_kafka_headers_to_model_empty(self) -> None:
        """Test conversion with empty headers."""
        event_bus = KafkaEventBus()

        headers = event_bus._kafka_headers_to_model(None)

        assert headers.source == "unknown"
        assert headers.event_type == "unknown"

    def test_kafka_headers_to_model_empty_list(self) -> None:
        """Test conversion with empty list."""
        event_bus = KafkaEventBus()

        headers = event_bus._kafka_headers_to_model([])

        assert headers.source == "unknown"
        assert headers.event_type == "unknown"

    def test_kafka_headers_to_model_invalid_uuid(self) -> None:
        """Test conversion with invalid UUID formats - should generate new UUIDs."""
        event_bus = KafkaEventBus()

        kafka_headers = [
            ("correlation_id", b"not-a-valid-uuid"),
            ("message_id", b"also-invalid"),
            ("source", b"test-source"),
            ("event_type", b"test-event"),
        ]

        headers = event_bus._kafka_headers_to_model(kafka_headers)

        # Should generate new UUIDs when invalid format detected
        assert headers.correlation_id is not None
        assert headers.message_id is not None
        assert str(headers.correlation_id) != "not-a-valid-uuid"
        assert str(headers.message_id) != "also-invalid"
        # Other fields should parse correctly
        assert headers.source == "test-source"
        assert headers.event_type == "test-event"


class TestKafkaEventBusMessageConversion:
    """Test suite for message conversion methods."""

    def test_kafka_msg_to_model(self) -> None:
        """Test conversion of Kafka message to ModelEventMessage."""
        event_bus = KafkaEventBus()

        # Create a mock Kafka message
        mock_msg = MagicMock()
        mock_msg.key = b"test-key"
        mock_msg.value = b"test-value"
        mock_msg.offset = 42
        mock_msg.partition = 0
        mock_msg.headers = [
            ("source", b"test-source"),
            ("event_type", b"test-event"),
        ]

        event_message = event_bus._kafka_msg_to_model(mock_msg, "test-topic")

        assert event_message.topic == "test-topic"
        assert event_message.key == b"test-key"
        assert event_message.value == b"test-value"
        assert event_message.offset == "42"
        assert event_message.partition == 0
        assert event_message.headers.source == "test-source"
        assert event_message.headers.event_type == "test-event"

    def test_kafka_msg_to_model_string_key(self) -> None:
        """Test conversion handles string key by encoding to bytes."""
        event_bus = KafkaEventBus()

        mock_msg = MagicMock()
        mock_msg.key = "string-key"  # String instead of bytes
        mock_msg.value = b"test-value"
        mock_msg.offset = 0
        mock_msg.partition = 0
        mock_msg.headers = None

        event_message = event_bus._kafka_msg_to_model(mock_msg, "test-topic")

        assert event_message.key == b"string-key"

    def test_kafka_msg_to_model_string_value(self) -> None:
        """Test conversion handles string value by encoding to bytes."""
        event_bus = KafkaEventBus()

        mock_msg = MagicMock()
        mock_msg.key = None
        mock_msg.value = "string-value"  # String instead of bytes
        mock_msg.offset = 0
        mock_msg.partition = 0
        mock_msg.headers = None

        event_message = event_bus._kafka_msg_to_model(mock_msg, "test-topic")

        assert event_message.value == b"string-value"

    def test_kafka_msg_to_model_none_key(self) -> None:
        """Test conversion handles None key."""
        event_bus = KafkaEventBus()

        mock_msg = MagicMock()
        mock_msg.key = None
        mock_msg.value = b"test-value"
        mock_msg.offset = 0
        mock_msg.partition = 0
        mock_msg.headers = None

        event_message = event_bus._kafka_msg_to_model(mock_msg, "test-topic")

        assert event_message.key is None


class TestKafkaEventBusConsumerManagement:
    """Test suite for consumer lifecycle management."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.fixture
    def mock_consumer(self) -> AsyncMock:
        """Create mock Kafka consumer."""
        consumer = AsyncMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()
        return consumer

    @pytest.mark.asyncio
    async def test_consumer_started_for_subscription(
        self, mock_producer: AsyncMock, mock_consumer: AsyncMock
    ) -> None:
        """Test consumer is started when subscribing to a topic."""
        with (
            patch(
                "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaConsumer",
                return_value=mock_consumer,
            ),
        ):
            event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")
            await event_bus.start()

            async def handler(msg: ModelEventMessage) -> None:
                pass

            await event_bus.subscribe("test-topic", "group1", handler)

            # Consumer should be started for the topic
            mock_consumer.start.assert_called_once()

            await event_bus.close()

    @pytest.mark.asyncio
    async def test_close_stops_all_consumers(
        self, mock_producer: AsyncMock, mock_consumer: AsyncMock
    ) -> None:
        """Test close stops all active consumers."""
        with (
            patch(
                "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaConsumer",
                return_value=mock_consumer,
            ),
        ):
            event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")
            await event_bus.start()

            async def handler(msg: ModelEventMessage) -> None:
                pass

            await event_bus.subscribe("test-topic", "group1", handler)
            await event_bus.close()

            # Consumer should be stopped
            mock_consumer.stop.assert_called_once()


class TestKafkaEventBusStartConsuming:
    """Test suite for start_consuming operation."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.mark.asyncio
    async def test_start_consuming_auto_starts(self, mock_producer: AsyncMock) -> None:
        """Test that start_consuming auto-starts the bus."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")

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
        self, mock_producer: AsyncMock
    ) -> None:
        """Test that start_consuming exits when shutdown is called."""
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            event_bus = KafkaEventBus(bootstrap_servers="localhost:9092")
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


class TestKafkaEventBusConfig:
    """Test suite for config-based KafkaEventBus construction."""

    def test_default_factory_creates_bus(self) -> None:
        """Test default() factory method creates a valid bus."""
        bus = KafkaEventBus.default()
        assert bus is not None
        assert bus.config is not None
        assert bus.environment == bus.config.environment
        assert bus.group == bus.config.group

    def test_from_config_creates_bus(self) -> None:
        """Test from_config() factory method."""
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="custom:9092",
            environment="staging",
            group="custom-group",
        )
        bus = KafkaEventBus.from_config(config)

        assert bus.config == config
        assert bus.environment == "staging"
        assert bus.group == "custom-group"

    def test_config_property_returns_model(self) -> None:
        """Test config property returns the config model."""
        config = ModelKafkaEventBusConfig.default()
        bus = KafkaEventBus(config=config)

        assert bus.config == config
        assert isinstance(bus.config, ModelKafkaEventBusConfig)

    def test_backwards_compatibility_with_direct_params(self) -> None:
        """Test that direct parameters still work for backwards compatibility."""
        bus = KafkaEventBus(
            bootstrap_servers="localhost:9092",
            environment="test",
            group="test-group",
        )

        assert bus.environment == "test"
        assert bus.group == "test-group"

    def test_parameter_overrides_config(self) -> None:
        """Test that explicit parameters override config values."""
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="config:9092",
            environment="config-env",
            group="config-group",
        )
        bus = KafkaEventBus(
            config=config,
            environment="override-env",  # This should override
        )

        assert bus.environment == "override-env"
        assert bus.group == "config-group"  # This should come from config

    def test_from_yaml_creates_bus(self, tmp_path: Path) -> None:
        """Test from_yaml() factory method with a temporary config file."""
        config_content = """bootstrap_servers: "yaml-server:9092"
environment: "yaml-env"
group: "yaml-group"
timeout_seconds: 45
max_retry_attempts: 5
retry_backoff_base: 2.0
circuit_breaker_threshold: 10
circuit_breaker_reset_timeout: 60.0
consumer_sleep_interval: 0.2
acks: "all"
enable_idempotence: true
auto_offset_reset: "earliest"
enable_auto_commit: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        bus = KafkaEventBus.from_yaml(config_file)

        assert bus.environment == "yaml-env"
        assert bus.group == "yaml-group"
        assert bus.config.timeout_seconds == 45
        assert bus.config.max_retry_attempts == 5
        assert bus.config.retry_backoff_base == 2.0
        assert bus.config.circuit_breaker_threshold == 10
        assert bus.config.auto_offset_reset == "earliest"
        assert bus.config.enable_auto_commit is False

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Test from_yaml() raises ProtocolConfigurationError for missing file."""
        from omnibase_infra.errors import ProtocolConfigurationError

        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(
            ProtocolConfigurationError, match="Configuration file not found"
        ):
            KafkaEventBus.from_yaml(missing_file)

    def test_from_yaml_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        """Test from_yaml() raises ProtocolConfigurationError for invalid YAML."""
        from omnibase_infra.errors import ProtocolConfigurationError

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: [")

        with pytest.raises(ProtocolConfigurationError, match="Failed to parse YAML"):
            KafkaEventBus.from_yaml(config_file)

    def test_from_yaml_non_dict_content(self, tmp_path: Path) -> None:
        """Test from_yaml() raises ProtocolConfigurationError for non-dict YAML."""
        from omnibase_infra.errors import ProtocolConfigurationError

        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")

        with pytest.raises(ProtocolConfigurationError, match="must be a dictionary"):
            KafkaEventBus.from_yaml(config_file)

    def test_config_defaults_match_property_defaults(self) -> None:
        """Test that config defaults match the documented property defaults."""
        bus = KafkaEventBus()  # No config, uses internal default

        assert bus.environment == "local"
        assert bus.group == "default"
        assert bus.config.timeout_seconds == 30
        assert bus.config.max_retry_attempts == 3
        assert bus.config.circuit_breaker_threshold == 5

    def test_config_with_all_parameters(self) -> None:
        """Test config with all parameters explicitly set."""
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="custom-broker:29092",
            environment="production",
            group="prod-workers",
            timeout_seconds=60,
            max_retry_attempts=5,
            retry_backoff_base=2.0,
            circuit_breaker_threshold=10,
            circuit_breaker_reset_timeout=60.0,
            consumer_sleep_interval=0.5,
            acks="1",
            enable_idempotence=False,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        bus = KafkaEventBus.from_config(config)

        assert bus.config == config
        assert bus.environment == "production"
        assert bus.group == "prod-workers"
        assert bus.config.acks == "1"
        assert bus.config.enable_idempotence is False


class TestKafkaEventBusTopicValidation:
    """Test suite for topic name validation.

    Tests the _validate_topic_name method which enforces Kafka topic naming rules:
    - Not empty
    - Max 255 characters
    - Only alphanumeric, period (.), underscore (_), hyphen (-)
    - Not reserved names ("." or "..")
    """

    @pytest.fixture
    def event_bus(self) -> KafkaEventBus:
        """Create KafkaEventBus for validation testing."""
        return KafkaEventBus(
            bootstrap_servers="localhost:9092",
            environment="test",
        )

    @pytest.fixture
    def correlation_id(self) -> UUID:
        """Create a correlation ID for tests."""
        return uuid4()

    def test_valid_topic_name_simple(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid simple topic name passes validation."""
        # Should not raise any exception
        event_bus._validate_topic_name("my-topic", correlation_id)

    def test_valid_topic_name_with_dots(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with dots passes validation."""
        event_bus._validate_topic_name("dev.events.user-created", correlation_id)

    def test_valid_topic_name_with_underscores(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with underscores passes validation."""
        event_bus._validate_topic_name("my_topic_name", correlation_id)

    def test_valid_topic_name_with_hyphens(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with hyphens passes validation."""
        event_bus._validate_topic_name("my-topic-name", correlation_id)

    def test_valid_topic_name_with_numbers(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with numbers passes validation."""
        event_bus._validate_topic_name("topic123", correlation_id)

    def test_valid_topic_name_mixed_characters(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with mixed valid characters."""
        event_bus._validate_topic_name("prod.user-events_v2.created", correlation_id)

    def test_valid_topic_name_uppercase(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name with uppercase letters."""
        event_bus._validate_topic_name("MyTopicName", correlation_id)

    def test_valid_topic_name_max_length(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test valid topic name at exactly 255 characters."""
        topic = "a" * 255
        # Should not raise
        event_bus._validate_topic_name(topic, correlation_id)

    def test_empty_topic_name_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test empty topic name raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="cannot be empty"):
            event_bus._validate_topic_name("", correlation_id)

    def test_topic_name_exceeds_max_length(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name exceeding 255 chars raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        topic = "a" * 256
        with pytest.raises(
            ProtocolConfigurationError, match="exceeds maximum length of 255"
        ):
            event_bus._validate_topic_name(topic, correlation_id)

    def test_reserved_topic_name_dot(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test reserved topic name '.' raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="reserved"):
            event_bus._validate_topic_name(".", correlation_id)

    def test_reserved_topic_name_double_dot(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test reserved topic name '..' raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="reserved"):
            event_bus._validate_topic_name("..", correlation_id)

    def test_topic_with_space_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with space raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            event_bus._validate_topic_name("my topic", correlation_id)

    def test_topic_with_at_symbol_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with @ symbol raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            event_bus._validate_topic_name("topic@name", correlation_id)

    def test_topic_with_special_chars_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with special characters raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        invalid_topics = [
            "topic#name",
            "topic$name",
            "topic%name",
            "topic&name",
            "topic*name",
            "topic!name",
            "topic/name",
            "topic\\name",
            "topic:name",
            "topic;name",
            "topic<name",
            "topic>name",
            "topic|name",
        ]
        for topic in invalid_topics:
            with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
                event_bus._validate_topic_name(topic, correlation_id)

    def test_topic_with_unicode_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with unicode characters raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            event_bus._validate_topic_name("topic\u00e9name", correlation_id)

    def test_topic_with_newline_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with newline raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            event_bus._validate_topic_name("topic\nname", correlation_id)

    def test_topic_with_tab_raises_error(
        self, event_bus: KafkaEventBus, correlation_id: UUID
    ) -> None:
        """Test topic name with tab raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            event_bus._validate_topic_name("topic\tname", correlation_id)
