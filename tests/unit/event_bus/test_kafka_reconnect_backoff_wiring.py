# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests verifying reconnect_backoff kwargs are wired to AIOKafka constructors.

Tests that reconnect_backoff_ms and reconnect_backoff_max_ms from
ModelKafkaEventBusConfig are passed through to:
1. AIOKafkaProducer in start()
2. AIOKafkaProducer in _ensure_producer() (recreation after failure)
3. AIOKafkaConsumer in _start_consumer_for_topic()

OMN-2919
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_BOOTSTRAP_SERVERS: str = "localhost:9092"
TEST_ENVIRONMENT: str = "local"
TEST_RECONNECT_BACKOFF_MS: int = 4000
TEST_RECONNECT_BACKOFF_MAX_MS: int = 60000


@pytest.fixture
def config_with_backoff() -> ModelKafkaEventBusConfig:
    """Create config with non-default reconnect backoff values for assertion clarity."""
    return ModelKafkaEventBusConfig(
        bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
        environment=TEST_ENVIRONMENT,
        reconnect_backoff_ms=TEST_RECONNECT_BACKOFF_MS,
        reconnect_backoff_max_ms=TEST_RECONNECT_BACKOFF_MAX_MS,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_producer_receives_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """start() passes reconnect_backoff_ms and reconnect_backoff_max_ms to AIOKafkaProducer."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ) as mock_producer_cls:
        bus = EventBusKafka(config=config_with_backoff)
        await bus.start()

        # Verify AIOKafkaProducer was constructed with the backoff kwargs
        assert mock_producer_cls.call_count == 1
        _, kwargs = mock_producer_cls.call_args
        assert kwargs["reconnect_backoff_ms"] == TEST_RECONNECT_BACKOFF_MS
        assert kwargs["reconnect_backoff_max_ms"] == TEST_RECONNECT_BACKOFF_MAX_MS

        await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_producer_receives_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """_ensure_producer() passes reconnect_backoff_ms and reconnect_backoff_max_ms to AIOKafkaProducer.

    Simulates producer recreation after failure: bus is started, producer is set
    to None (as happens after a publish failure), then _ensure_producer is called
    under the producer lock to recreate it.
    """
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ) as mock_producer_cls:
        bus = EventBusKafka(config=config_with_backoff)
        # Start the bus so _started=True
        await bus.start()

        # Reset call count after initial start
        mock_producer_cls.reset_mock()

        # Simulate producer being lost (e.g., after a failed publish)
        bus._producer = None

        # Call _ensure_producer under the producer lock (as the real code does)
        async with bus._producer_lock:
            await bus._ensure_producer(uuid4())

        # Verify recreation call also passes backoff kwargs
        assert mock_producer_cls.call_count == 1
        _, kwargs = mock_producer_cls.call_args
        assert kwargs["reconnect_backoff_ms"] == TEST_RECONNECT_BACKOFF_MS
        assert kwargs["reconnect_backoff_max_ms"] == TEST_RECONNECT_BACKOFF_MAX_MS

        await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_consumer_for_topic_receives_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """_start_consumer_for_topic() passes reconnect_backoff_ms and reconnect_backoff_max_ms to AIOKafkaConsumer."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False

    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    mock_consumer.__aiter__ = MagicMock(return_value=iter([]))

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ),
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls,
    ):
        bus = EventBusKafka(config=config_with_backoff)
        await bus.start()

        # Directly call _start_consumer_for_topic with a valid group_id
        await bus._start_consumer_for_topic("test-topic", "test-group")

        # Verify AIOKafkaConsumer was constructed with the backoff kwargs
        assert mock_consumer_cls.call_count == 1
        _, kwargs = mock_consumer_cls.call_args
        assert kwargs["reconnect_backoff_ms"] == TEST_RECONNECT_BACKOFF_MS
        assert kwargs["reconnect_backoff_max_ms"] == TEST_RECONNECT_BACKOFF_MAX_MS

        await bus.close()


__all__: list[str] = []
