# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for static group membership (OMN-7601).

Verifies that group_instance_id is correctly plumbed from ModelKafkaEventBusConfig
through to AIOKafkaConsumer, enabling static group membership to prevent rebalance
storms in multi-container deployments.

group_instance_id is auto-derived from effective_group_id + hostname when not
explicitly set in config — no env var required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.fixture
def kafka_config_with_explicit_instance_id() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        session_timeout_ms=60000,
        heartbeat_interval_ms=20000,
        group_instance_id="omninode-runtime-1",
    )


@pytest.fixture
def kafka_config_default() -> ModelKafkaEventBusConfig:
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
    def test_group_instance_id_not_in_env_var_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_GROUP_INSTANCE_ID must NOT be an env var override — no env-var path."""
        monkeypatch.setenv("KAFKA_GROUP_INSTANCE_ID", "should-be-ignored")
        config = ModelKafkaEventBusConfig.default()
        assert config.group_instance_id is None


class TestGroupInstanceIdWiredToConsumer:
    """Test that group_instance_id reaches the AIOKafkaConsumer constructor."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consumer_receives_explicit_group_instance_id(
        self, kafka_config_with_explicit_instance_id: ModelKafkaEventBusConfig
    ) -> None:
        """Explicit config value is passed through unchanged."""
        bus = EventBusKafka(config=kafka_config_with_explicit_instance_id)
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
    async def test_consumer_receives_auto_derived_group_instance_id_when_none(
        self, kafka_config_default: ModelKafkaEventBusConfig
    ) -> None:
        """When group_instance_id is None, auto-derived value includes hostname."""
        bus = EventBusKafka(config=kafka_config_default)
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        await _start_bus(bus)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                return_value=mock_consumer,
            ) as mock_consumer_cls,
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.socket.gethostname",
                return_value="runtime-worker-1",
            ),
        ):
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            derived = call_kwargs.kwargs["group_instance_id"]
            assert derived is not None
            assert "runtime-worker-1" in derived

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_derived_id_sanitizes_hostname_special_chars(
        self, kafka_config_default: ModelKafkaEventBusConfig
    ) -> None:
        """Hostname chars outside [a-zA-Z0-9._-] are replaced with '-'."""
        bus = EventBusKafka(config=kafka_config_default)
        mock_consumer = MagicMock()
        mock_consumer.start = AsyncMock()

        await _start_bus(bus)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                return_value=mock_consumer,
            ) as mock_consumer_cls,
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.socket.gethostname",
                return_value="worker@node:1",
            ),
        ):
            await bus.subscribe(
                "test-topic", on_message=AsyncMock(), group_id="test-group"
            )
            call_kwargs = mock_consumer_cls.call_args
            derived = call_kwargs.kwargs["group_instance_id"]
            assert "@" not in derived
            assert ":" not in derived

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_static_membership_combined_with_session_timeout(
        self, kafka_config_with_explicit_instance_id: ModelKafkaEventBusConfig
    ) -> None:
        """group_instance_id and session_timeout_ms must both reach the consumer."""
        bus = EventBusKafka(config=kafka_config_with_explicit_instance_id)
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
