"""
Comprehensive tests for Kafka Producer Pool.

Tests producer pooling, message sending, metrics collection,
and error handling.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from omnibase_infra.infrastructure.kafka.producer_pool import (
    KafkaProducerPool,
    ProducerInstance,
)
from omnibase_infra.models.kafka import (
    ModelKafkaProducerConfig,
    ModelKafkaProducerPoolStats,
)


@pytest.fixture
def mock_config():
    """Create a mock Kafka producer configuration."""
    return ModelKafkaProducerConfig(
        bootstrap_servers=["localhost:9092"],
        client_id_prefix="test",
        acks="all",
        retries=3,
        batch_size=16384,
        linger_ms=10,
        buffer_memory=33554432,
        compression_type="gzip",
        max_request_size=1048576,
        request_timeout_ms=30000,
        delivery_timeout_ms=120000,
        max_in_flight_requests_per_connection=5,
        enable_idempotence=True,
    )


@pytest.fixture
def pool(mock_config):
    """Create a KafkaProducerPool instance."""
    return KafkaProducerPool(
        config=mock_config,
        pool_name="test_pool",
        min_pool_size=2,
        max_pool_size=5,
    )


@pytest.fixture
def mock_producer():
    """Create a mock Kafka producer."""
    producer = AsyncMock(spec=AIOKafkaProducer)
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send = AsyncMock()
    return producer


class TestKafkaProducerPoolInit:
    """Test producer pool initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with provided configuration."""
        pool = KafkaProducerPool(
            config=mock_config,
            pool_name="my_pool",
            min_pool_size=3,
            max_pool_size=10,
        )

        assert pool.config == mock_config
        assert pool.pool_name == "my_pool"
        assert pool.min_pool_size == 3
        assert pool.max_pool_size == 10
        assert pool.is_initialized is False
        assert len(pool.producers) == 0
        assert len(pool.idle_producers) == 0
        assert len(pool.active_producers) == 0

    def test_init_default_pool_sizes(self, mock_config):
        """Test initialization with default pool sizes."""
        pool = KafkaProducerPool(config=mock_config)

        assert pool.min_pool_size == 2
        assert pool.max_pool_size == 10


class TestKafkaProducerPoolInitialization:
    """Test producer pool initialization."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self, pool, mock_producer):
        """Test successful pool initialization."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            assert pool.is_initialized is True
            assert len(pool.producers) == pool.min_pool_size
            assert len(pool.idle_producers) == pool.min_pool_size
            assert len(pool.active_producers) == 0

            # Verify all producers started
            assert mock_producer.start.call_count == pool.min_pool_size

    @pytest.mark.asyncio
    async def test_initialization_already_initialized(self, pool):
        """Test that re-initialization is skipped."""
        pool.is_initialized = True

        await pool.initialize()

        # Should not create new producers
        assert len(pool.producers) == 0

    @pytest.mark.asyncio
    async def test_initialization_failure(self, pool, mock_producer):
        """Test initialization failure on producer creation."""
        mock_producer.start.side_effect = Exception("Connection failed")

        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            with pytest.raises(OnexError) as exc_info:
                await pool.initialize()

            assert exc_info.value.code == CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE
            assert "Failed to initialize Kafka producer pool" in exc_info.value.message


class TestKafkaProducerPoolCreateProducer:
    """Test producer creation."""

    @pytest.mark.asyncio
    async def test_create_producer_success(self, pool, mock_producer):
        """Test successful producer creation."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            instance = await pool._create_producer("test_producer_1")

            assert isinstance(instance, ProducerInstance)
            assert instance.producer_id == "test_producer_1"
            assert instance.producer == mock_producer
            assert instance.is_active is True
            assert "test_producer_1" in pool.producers
            assert "test_producer_1" in pool.idle_producers
            mock_producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_producer_failure(self, pool, mock_producer):
        """Test producer creation failure."""
        mock_producer.start.side_effect = KafkaConnectionError("Broker unavailable")

        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            with pytest.raises(OnexError) as exc_info:
                await pool._create_producer("test_producer_1")

            assert exc_info.value.code == CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_create_producer_with_config(self, pool, mock_producer):
        """Test producer created with correct configuration."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer) as mock_class:
            await pool._create_producer("test_producer_1")

            # Verify configuration parameters
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["bootstrap_servers"] == pool.config.bootstrap_servers
            assert call_kwargs["client_id"] == "test_producer_1"
            assert call_kwargs["acks"] == pool.config.acks
            assert call_kwargs["retries"] == pool.config.retries
            assert call_kwargs["enable_idempotence"] == pool.config.enable_idempotence


class TestKafkaProducerPoolAcquireProducer:
    """Test producer acquisition from pool."""

    @pytest.mark.asyncio
    async def test_acquire_idle_producer(self, pool, mock_producer):
        """Test acquiring a producer from idle pool."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            async with pool.acquire_producer() as producer:
                assert producer == mock_producer
                assert len(pool.active_producers) == 1
                assert len(pool.idle_producers) == pool.min_pool_size - 1

            # After context exit, should return to idle
            assert len(pool.active_producers) == 0
            assert len(pool.idle_producers) == pool.min_pool_size

    @pytest.mark.asyncio
    async def test_acquire_create_new_producer(self, pool, mock_producer):
        """Test creating new producer when all are active."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            # Acquire all idle producers
            producers = []
            for _ in range(pool.min_pool_size):
                ctx = pool.acquire_producer()
                prod = await ctx.__aenter__()
                producers.append((ctx, prod))

            assert len(pool.active_producers) == pool.min_pool_size
            assert len(pool.idle_producers) == 0

            # Acquire one more - should create new
            async with pool.acquire_producer() as producer:
                assert len(pool.producers) == pool.min_pool_size + 1
                assert len(pool.active_producers) == pool.min_pool_size + 1

            # Clean up
            for ctx, _ in producers:
                await ctx.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_acquire_pool_at_capacity(self, pool, mock_producer):
        """Test error when pool is at maximum capacity."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            # Acquire maximum producers
            contexts = []
            for _ in range(pool.max_pool_size):
                ctx = pool.acquire_producer()
                await ctx.__aenter__()
                contexts.append(ctx)

            # Try to acquire one more
            with pytest.raises(OnexError) as exc_info:
                async with pool.acquire_producer():
                    pass

            assert exc_info.value.code == CoreErrorCode.RESOURCE_EXHAUSTED
            assert "at capacity" in exc_info.value.message

            # Clean up
            for ctx in contexts:
                await ctx.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_acquire_auto_initialize(self, pool, mock_producer):
        """Test auto-initialization when acquiring producer."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            async with pool.acquire_producer() as producer:
                assert pool.is_initialized is True
                assert producer == mock_producer

    @pytest.mark.asyncio
    async def test_acquire_release_on_error(self, pool, mock_producer):
        """Test producer is released even if error occurs during usage."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            with pytest.raises(ValueError):
                async with pool.acquire_producer():
                    raise ValueError("User error")

            # Producer should still be in idle pool
            assert len(pool.idle_producers) == pool.min_pool_size
            assert len(pool.active_producers) == 0

    @pytest.mark.asyncio
    async def test_acquire_kafka_connection_error(self, pool, mock_producer):
        """Test handling of Kafka connection errors."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            with pytest.raises(OnexError):
                async with pool.acquire_producer():
                    raise KafkaConnectionError("Broker unreachable")

            # Failed producer should be in failed_producers list
            assert len(pool.failed_producers) >= 1

    @pytest.mark.asyncio
    async def test_acquire_updates_activity_timestamp(self, pool, mock_producer):
        """Test that acquiring updates producer activity timestamp."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            producer_id = pool.idle_producers[0]
            instance = pool.producers[producer_id]
            original_activity = instance.last_activity

            await asyncio.sleep(0.01)  # Small delay

            async with pool.acquire_producer():
                pass

            # Activity timestamp should be updated
            assert instance.last_activity != original_activity


class TestKafkaProducerPoolSendMessage:
    """Test message sending through producer pool."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, pool, mock_producer):
        """Test successful message sending."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            result = await pool.send_message(
                topic="test_topic",
                value=b"test message",
            )

            assert result is True
            assert pool.total_messages_sent == 1
            assert pool.total_bytes_sent == len(b"test message")
            mock_producer.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_with_key(self, pool, mock_producer):
        """Test sending message with key."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            await pool.send_message(
                topic="test_topic",
                value=b"test message",
                key=b"test_key",
            )

            call_kwargs = mock_producer.send.call_args.kwargs
            assert call_kwargs["key"] == b"test_key"

    @pytest.mark.asyncio
    async def test_send_message_with_partition(self, pool, mock_producer):
        """Test sending message to specific partition."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            await pool.send_message(
                topic="test_topic",
                value=b"test message",
                partition=2,
            )

            call_kwargs = mock_producer.send.call_args.kwargs
            assert call_kwargs["partition"] == 2

    @pytest.mark.asyncio
    async def test_send_message_with_headers(self, pool, mock_producer):
        """Test sending message with headers."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            headers = {"trace_id": b"12345", "source": b"test"}
            await pool.send_message(
                topic="test_topic",
                value=b"test message",
                headers=headers,
            )

            call_kwargs = mock_producer.send.call_args.kwargs
            assert call_kwargs["headers"] == headers

    @pytest.mark.asyncio
    async def test_send_message_failure(self, pool, mock_producer):
        """Test message send failure."""
        mock_producer.send.side_effect = Exception("Send failed")

        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            with pytest.raises(OnexError) as exc_info:
                await pool.send_message(
                    topic="test_topic",
                    value=b"test message",
                )

            assert exc_info.value.code == CoreErrorCode.INTEGRATION_SERVICE_ERROR
            assert pool.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_send_message_updates_topic_stats(self, pool, mock_producer):
        """Test that sending updates topic statistics."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            await pool.send_message(
                topic="test_topic",
                value=b"message1",
            )
            await pool.send_message(
                topic="test_topic",
                value=b"message2",
            )

            assert "test_topic" in pool.topic_stats
            stats = pool.topic_stats["test_topic"]
            assert stats.messages_sent == 2
            assert stats.bytes_sent == len(b"message1") + len(b"message2")


class TestKafkaProducerPoolStatistics:
    """Test pool statistics collection."""

    @pytest.mark.asyncio
    async def test_get_pool_stats(self, pool, mock_producer):
        """Test retrieving pool statistics."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()
            await pool.send_message("topic1", b"message")

            stats = pool.get_pool_stats()

            assert isinstance(stats, ModelKafkaProducerPoolStats)
            assert stats.pool_name == "test_pool"
            assert stats.total_producers == pool.min_pool_size
            assert stats.idle_producers == pool.min_pool_size
            assert stats.active_producers == 0
            assert stats.total_messages_sent == 1
            assert stats.min_pool_size == pool.min_pool_size
            assert stats.max_pool_size == pool.max_pool_size

    @pytest.mark.asyncio
    async def test_pool_stats_health_determination(self, pool, mock_producer):
        """Test pool health status determination."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            stats = pool.get_pool_stats()

            assert stats.pool_health in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_pool_stats_with_topic_stats(self, pool, mock_producer):
        """Test pool stats include topic statistics."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()
            await pool.send_message("topic1", b"msg1")
            await pool.send_message("topic2", b"msg2")

            stats = pool.get_pool_stats()

            assert stats.topic_stats is not None
            assert len(stats.topic_stats) == 2


class TestKafkaProducerPoolClose:
    """Test pool closing and cleanup."""

    @pytest.mark.asyncio
    async def test_close_pool(self, pool, mock_producer):
        """Test closing the producer pool."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            await pool.close()

            assert pool.is_initialized is False
            assert len(pool.producers) == 0
            assert len(pool.idle_producers) == 0
            assert len(pool.active_producers) == 0
            assert mock_producer.stop.call_count == pool.min_pool_size

    @pytest.mark.asyncio
    async def test_close_with_error(self, pool, mock_producer):
        """Test closing with producer stop error."""
        mock_producer.stop.side_effect = Exception("Stop failed")

        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            # Should still complete cleanup despite errors
            await pool.close()

            assert len(pool.producers) == 0

    @pytest.mark.asyncio
    async def test_close_not_initialized(self, pool):
        """Test closing when pool not initialized."""
        await pool.close()

        # Should not raise errors
        assert len(pool.producers) == 0


class TestKafkaProducerPoolIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, pool, mock_producer):
        """Test complete lifecycle: init -> send -> close."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            # Initialize
            await pool.initialize()
            assert pool.is_initialized is True

            # Send messages
            await pool.send_message("topic1", b"message1")
            await pool.send_message("topic2", b"message2")

            # Get stats
            stats = pool.get_pool_stats()
            assert stats.total_messages_sent == 2

            # Close
            await pool.close()
            assert pool.is_initialized is False

    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self, pool, mock_producer):
        """Test concurrent message sending through pool."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            # Send multiple messages concurrently
            tasks = [
                pool.send_message(f"topic{i}", f"message{i}".encode())
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            assert all(results)
            assert pool.total_messages_sent == 10

    @pytest.mark.asyncio
    async def test_producer_reuse(self, pool, mock_producer):
        """Test that producers are properly reused."""
        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            await pool.initialize()

            initial_count = len(pool.producers)

            # Acquire and release multiple times
            for _ in range(5):
                async with pool.acquire_producer():
                    pass

            # Should not create additional producers
            assert len(pool.producers) == initial_count
