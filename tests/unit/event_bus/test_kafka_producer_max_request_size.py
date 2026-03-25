# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that max_request_size is passed to AIOKafkaProducer (OMN-6346)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.mark.unit
class TestProducerMaxRequestSize:
    """Verify AIOKafkaProducer receives max_request_size from config."""

    @pytest.mark.asyncio
    async def test_producer_receives_max_request_size_default(self) -> None:
        """AIOKafkaProducer gets default max_request_size (4MB) from config."""
        config = ModelKafkaEventBusConfig()
        bus = EventBusKafka(config=config)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

            call_kwargs = mock_producer_cls.call_args.kwargs
            assert call_kwargs["max_request_size"] == 4 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_producer_receives_custom_max_request_size(self) -> None:
        """AIOKafkaProducer gets custom max_request_size from config."""
        config = ModelKafkaEventBusConfig(max_request_size=5_000_000)
        bus = EventBusKafka(config=config)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer
            await bus.start()

            call_kwargs = mock_producer_cls.call_args.kwargs
            assert call_kwargs["max_request_size"] == 5_000_000
