# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests verifying reconnect_backoff kwargs are NOT passed to AIOKafka constructors.

Tests that reconnect_backoff_ms and reconnect_backoff_max_ms are absent from
AIOKafkaProducer/Consumer init calls. These are kafka-python (sync) parameters
that are not supported by aiokafka 0.11.x — passing them causes TypeError on
every runtime start.

Config model fields are preserved (OMN-2916) but must not be forwarded to aiokafka.

Verifies the following constructors do NOT receive the invalid kwargs:
1. AIOKafkaProducer in start()
2. AIOKafkaProducer in _ensure_producer() (recreation after failure)
3. AIOKafkaConsumer in _start_consumer_for_topic()

OMN-3230
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
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
async def test_start_producer_does_not_receive_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """start() must NOT pass reconnect_backoff_ms or reconnect_backoff_max_ms to AIOKafkaProducer.

    These are kafka-python (sync) params — aiokafka 0.11.x raises TypeError if they are present.
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
        await bus.start()

        # Verify AIOKafkaProducer was NOT constructed with the invalid backoff kwargs
        assert mock_producer_cls.call_count == 1
        _, kwargs = mock_producer_cls.call_args
        assert "reconnect_backoff_ms" not in kwargs, (
            "reconnect_backoff_ms is not a valid aiokafka kwarg and must not be passed"
        )
        assert "reconnect_backoff_max_ms" not in kwargs, (
            "reconnect_backoff_max_ms is not a valid aiokafka kwarg and must not be passed"
        )

        await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_producer_does_not_receive_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """_ensure_producer() must NOT pass reconnect_backoff_ms or reconnect_backoff_max_ms to AIOKafkaProducer.

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

        # Verify recreation call also does NOT pass invalid backoff kwargs
        assert mock_producer_cls.call_count == 1
        _, kwargs = mock_producer_cls.call_args
        assert "reconnect_backoff_ms" not in kwargs, (
            "reconnect_backoff_ms is not a valid aiokafka kwarg and must not be passed"
        )
        assert "reconnect_backoff_max_ms" not in kwargs, (
            "reconnect_backoff_max_ms is not a valid aiokafka kwarg and must not be passed"
        )

        await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_consumer_for_topic_does_not_receive_reconnect_backoff_kwargs(
    config_with_backoff: ModelKafkaEventBusConfig,
) -> None:
    """_start_consumer_for_topic() must NOT pass reconnect_backoff_ms or reconnect_backoff_max_ms to AIOKafkaConsumer."""
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

        # Verify AIOKafkaConsumer was NOT constructed with the invalid backoff kwargs
        assert mock_consumer_cls.call_count == 1
        _, kwargs = mock_consumer_cls.call_args
        assert "reconnect_backoff_ms" not in kwargs, (
            "reconnect_backoff_ms is not a valid aiokafka kwarg and must not be passed"
        )
        assert "reconnect_backoff_max_ms" not in kwargs, (
            "reconnect_backoff_max_ms is not a valid aiokafka kwarg and must not be passed"
        )

        await bus.close()


__all__: list[str] = []
