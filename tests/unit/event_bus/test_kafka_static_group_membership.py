# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for static group membership (OMN-7601).

Verifies that group_instance_id is correctly plumbed from ModelKafkaEventBusConfig
through to AIOKafkaConsumer, enabling static group membership to prevent rebalance
storms in multi-container deployments.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.fixture
def kafka_config_with_static_membership() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        session_timeout_ms=60000,
        heartbeat_interval_ms=20000,
        group_instance_id="omninode-runtime-1",
    )


@pytest.fixture
def kafka_config_no_static_membership() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        group_instance_id=None,
    )


async def _start_bus(bus: EventBusKafka) -> None:
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer"
    ) as mock_producer_cls:
        mock_producer = MagicMock()
        mock_producer.start = AsyncMock()
        mock_producer.stop = AsyncMock()
        mock_producer_cls.return_value = mock_producer
        await bus.start()


class TestGroupInstanceIdConfig:
    """Test group_instance_id field on ModelKafkaEventBusConfig."""

    @pytest.mark.unit
    def test_default_group_instance_id_is_none(self) -> None:
        config = ModelKafkaEventBusConfig()
        assert config.group_instance_id is None

    @pytest.mark.unit
    def test_group_instance_id_accepts_valid_value(self) -> None:
        config = ModelKafkaEventBusConfig(group_instance_id="omninode-runtime-1")
        assert config.group_instance_id == "omninode-runtime-1"

    @pytest.mark.unit
    def test_group_instance_id_accepts_hostname_style(self) -> None:
        config = ModelKafkaEventBusConfig(
            group_instance_id="runtime-worker-abc123.internal"
        )
        assert config.group_instance_id == "runtime-worker-abc123.internal"

    @pytest.mark.unit
    def test_group_instance_id_rejects_invalid_chars(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            ModelKafkaEventBusConfig(group_instance_id="runtime/worker 1")

    @pytest.mark.unit
    def test_group_instance_id_empty_string_becomes_none(self) -> None:
        config = ModelKafkaEventBusConfig(group_instance_id="   ")
        assert config.group_instance_id is None

    @pytest.mark.unit
    def test_group_instance_id_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_GROUP_INSTANCE_ID", "runtime-worker-2")
        config = ModelKafkaEventBusConfig.default()
        assert config.group_instance_id == "runtime-worker-2"

    @pytest.mark.unit
    def test_group_instance_id_env_var_not_set_gives_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAFKA_GROUP_INSTANCE_ID", raising=False)
        config = ModelKafkaEventBusConfig.default()
        assert config.group_instance_id is None


class TestGroupInstanceIdWiredToConsumer:
    """Test that group_instance_id reaches the AIOKafkaConsumer constructor."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_group_instance_id_when_set(
        self, kafka_config_with_static_membership: ModelKafkaEventBusConfig
    ) -> None:
        bus = EventBusKafka(config=kafka_config_with_static_membership)
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        await _start_bus(bus)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["group_instance_id"] == "omninode-runtime-1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_none_group_instance_id_when_not_set(
        self, kafka_config_no_static_membership: ModelKafkaEventBusConfig
    ) -> None:
        bus = EventBusKafka(config=kafka_config_no_static_membership)
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        await _start_bus(bus)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["group_instance_id"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_static_membership_combined_with_session_timeout(
        self, kafka_config_with_static_membership: ModelKafkaEventBusConfig
    ) -> None:
        """group_instance_id and session_timeout_ms must both reach the consumer."""
        bus = EventBusKafka(config=kafka_config_with_static_membership)
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        await _start_bus(bus)

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls:
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            assert call_kwargs.kwargs["group_instance_id"] == "omninode-runtime-1"
            assert call_kwargs.kwargs["session_timeout_ms"] == 60000
            assert call_kwargs.kwargs["heartbeat_interval_ms"] == 20000
