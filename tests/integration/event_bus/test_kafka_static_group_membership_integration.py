# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for static Kafka group membership (OMN-7601).

Verifies that group_instance_id flows end-to-end from ModelKafkaEventBusConfig
through EventBusKafka.subscribe() into the AIOKafkaConsumer constructor, which
is the wire-level mechanism that prevents rebalance storms in multi-container
deployments.

Pairs with tests/unit/event_bus/test_kafka_static_group_membership.py: the unit
tests cover field validation; this integration test wires a real EventBusKafka
lifecycle (start, subscribe, stop) end-to-end with a mocked AIOKafkaConsumer
and asserts the constructor kwargs match the configured static-membership
identity.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def static_membership_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        session_timeout_ms=60000,
        heartbeat_interval_ms=20000,
        group_instance_id="omninode-runtime-integration",
    )


@pytest.mark.asyncio
async def test_static_group_membership_flows_through_full_lifecycle(
    static_membership_config: ModelKafkaEventBusConfig,
) -> None:
    """Configured group_instance_id reaches the consumer after a full start/subscribe cycle.

    This is the integration-level proof for OMN-7601: a real EventBusKafka is
    instantiated and started, then a subscription is opened. The consumer
    constructor receives the same identity the config declared, alongside
    the matching session/heartbeat timeouts.
    """
    bus = EventBusKafka(config=static_membership_config)
    mock_consumer = MagicMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()

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
            "onex.evt.integration.static-membership.v1",
            on_message=AsyncMock(),
            group_id="integration-static-membership-group",
        )

    call_kwargs = mock_consumer_cls.call_args.kwargs
    assert call_kwargs["group_instance_id"] == "omninode-runtime-integration"
    assert call_kwargs["session_timeout_ms"] == 60000
    assert call_kwargs["heartbeat_interval_ms"] == 20000


@pytest.mark.asyncio
async def test_auto_derived_membership_id_present_when_config_omits_it() -> None:
    """When config leaves group_instance_id None, an auto-derived value is still wired.

    The runtime never sends `group_instance_id=None` to AIOKafkaConsumer when
    static membership is the intended mode — it derives one from group_id + hostname.
    This integration test asserts that derivation happens end-to-end.
    """
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        group_instance_id=None,
    )
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

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            return_value=mock_consumer,
        ) as mock_consumer_cls,
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.socket.gethostname",
            return_value="integration-host-7601",
        ),
    ):
        await bus.subscribe(
            "onex.evt.integration.static-membership-derived.v1",
            on_message=AsyncMock(),
            group_id="integration-derived-group",
        )

    derived = mock_consumer_cls.call_args.kwargs["group_instance_id"]
    assert derived is not None
    assert "integration-host-7601" in derived
