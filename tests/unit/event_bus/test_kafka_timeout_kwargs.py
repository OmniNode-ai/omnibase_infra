# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for EventBusKafka consumer/producer session timeout and reconnect kwargs (OMN-5445).

Verifies that session_timeout_ms, heartbeat_interval_ms, and retry_backoff_ms are
correctly passed from ModelKafkaEventBusConfig through to the AIOKafkaConsumer and
AIOKafkaProducer constructors.

Also verifies that reconnect_backoff_ms and reconnect_backoff_max_ms (config model
fields) are NOT passed directly to aiokafka constructors, since aiokafka 0.11.0 uses
retry_backoff_ms instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.fixture
def kafka_config() -> ModelKafkaEventBusConfig:
    """Create a config with known session/heartbeat/reconnect values."""
    return ModelKafkaEventBusConfig(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        reconnect_backoff_ms=3000,
        reconnect_backoff_max_ms=60000,
    )


@pytest.fixture
def bus(kafka_config: ModelKafkaEventBusConfig) -> EventBusKafka:
    """Create EventBusKafka instance with test config."""
    return EventBusKafka(config=kafka_config)


class TestConsumerKwargs:
    """Test that AIOKafkaConsumer receives session/heartbeat/retry kwargs."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_session_timeout_ms(
        self, bus: EventBusKafka, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        """Consumer constructor must receive session_timeout_ms from config."""
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["session_timeout_ms"] == 45000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_heartbeat_interval_ms(
        self, bus: EventBusKafka, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        """Consumer constructor must receive heartbeat_interval_ms from config."""
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["heartbeat_interval_ms"] == 15000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_retry_backoff_ms(
        self, bus: EventBusKafka, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        """Consumer constructor must receive retry_backoff_ms mapped from config.reconnect_backoff_ms."""
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["retry_backoff_ms"] == 3000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_does_not_pass_reconnect_backoff_ms(
        self, bus: EventBusKafka, kafka_config: ModelKafkaEventBusConfig
    ) -> None:
        """Consumer constructor must NOT receive reconnect_backoff_ms (not a valid aiokafka kwarg)."""
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert "reconnect_backoff_ms" not in call_kwargs.kwargs
            assert "reconnect_backoff_max_ms" not in call_kwargs.kwargs


class TestProducerKwargs:
    """Test that AIOKafkaProducer receives retry backoff kwargs."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_producer_start_receives_retry_backoff_ms(
        self, bus: EventBusKafka
    ) -> None:
        """Producer constructor in start() must receive retry_backoff_ms mapped from config.reconnect_backoff_ms."""
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()
            call_kwargs = mock_producer_cls.call_args
            assert call_kwargs.kwargs["retry_backoff_ms"] == 3000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_producer_start_does_not_pass_reconnect_backoff_ms(
        self, bus: EventBusKafka
    ) -> None:
        """Producer constructor must NOT receive reconnect_backoff_ms or reconnect_backoff_max_ms."""
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()
            call_kwargs = mock_producer_cls.call_args
            assert "reconnect_backoff_ms" not in call_kwargs.kwargs
            assert "reconnect_backoff_max_ms" not in call_kwargs.kwargs


class TestDefaultValues:
    """Test that default config values are correctly wired through."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_session_timeout_wired_to_consumer(self) -> None:
        """Default 30000ms session timeout must reach consumer constructor."""
        config = ModelKafkaEventBusConfig()
        bus = EventBusKafka(config=config)

        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["session_timeout_ms"] == 30000
            assert call_kwargs.kwargs["heartbeat_interval_ms"] == 10000
