# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that ProviderKafkaProducer passes max_request_size to AIOKafkaProducer (OMN-6320)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)


@pytest.mark.unit
class TestProviderKafkaProducerMaxRequestSize:
    """Verify AIOKafkaProducer receives max_request_size from config."""

    @pytest.mark.asyncio
    async def test_default_max_request_size_passed(self) -> None:
        """AIOKafkaProducer gets default 4 MB max_request_size."""
        config = ModelKafkaProducerConfig()
        provider = ProviderKafkaProducer(config)

        with patch("aiokafka.AIOKafkaProducer") as mock_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_cls.return_value = mock_producer

            await provider.create()

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["max_request_size"] == 4 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_custom_max_request_size_passed(self) -> None:
        """AIOKafkaProducer gets custom max_request_size from config."""
        config = ModelKafkaProducerConfig(max_request_size=8_000_000)
        provider = ProviderKafkaProducer(config)

        with patch("aiokafka.AIOKafkaProducer") as mock_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_cls.return_value = mock_producer

            await provider.create()

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["max_request_size"] == 8_000_000
