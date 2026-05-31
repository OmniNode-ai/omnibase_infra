# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for EventBusKafka aiokafka API version configuration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.mark.unit
class TestKafkaApiVersionConfig:
    """Verify explicit aiokafka API version config is typed and plumbed."""

    def test_api_version_defaults_to_auto_negotiation(self) -> None:
        config = ModelKafkaEventBusConfig()

        assert config.api_version is None

    def test_api_version_can_be_set_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_API_VERSION", "2.8.0")

        config = ModelKafkaEventBusConfig.default()

        assert config.api_version == "2.8.0"

    def test_blank_api_version_env_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_API_VERSION", "   ")

        config = ModelKafkaEventBusConfig.default()

        assert config.api_version is None

    def test_invalid_api_version_fails_closed(self) -> None:
        with pytest.raises(ProtocolConfigurationError, match="api_version"):
            ModelKafkaEventBusConfig(api_version="2.8")

    def test_api_version_kwargs_omitted_when_aiokafka_does_not_support_them(
        self,
    ) -> None:
        class ClientWithoutApiVersion:
            def __init__(self, *, bootstrap_servers: str) -> None:
                self.bootstrap_servers = bootstrap_servers

        config = ModelKafkaEventBusConfig(api_version="2.8.0")
        bus = EventBusKafka(config=config)

        assert bus._build_client_version_kwargs(ClientWithoutApiVersion) == {}

    def test_api_version_kwargs_present_when_aiokafka_supports_them(self) -> None:
        class ClientWithApiVersion:
            def __init__(
                self, *, bootstrap_servers: str, api_version: str = "auto"
            ) -> None:
                self.bootstrap_servers = bootstrap_servers
                self.api_version = api_version

        config = ModelKafkaEventBusConfig(api_version="2.8.0")
        bus = EventBusKafka(config=config)

        assert bus._build_client_version_kwargs(ClientWithApiVersion) == {
            "api_version": "2.8.0"
        }

    @pytest.mark.asyncio
    async def test_producer_omits_api_version_when_installed_aiokafka_lacks_support(
        self,
    ) -> None:
        config = ModelKafkaEventBusConfig(api_version="2.8.0")
        bus = EventBusKafka(config=config)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
        ) as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_cls.return_value = mock_producer

            await bus.start()

        assert "api_version" not in mock_producer_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_producer_omits_api_version_when_unset(self) -> None:
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

        assert "api_version" not in mock_producer_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_consumer_omits_api_version_when_installed_aiokafka_lacks_support(
        self,
    ) -> None:
        config = ModelKafkaEventBusConfig(api_version="2.8.0")
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

        assert "api_version" not in mock_consumer_cls.call_args.kwargs
