# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka cold-bootstrap retry coverage for EventBusKafka."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import KafkaConnectionError

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


def _retry_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        timeout_seconds=1,
        max_retry_attempts=1,
        retry_backoff_base=0.001,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_producer_start_retries_kafka_connection_error() -> None:
    bus = EventBusKafka(config=_retry_config())

    first_producer = MagicMock()
    first_producer.start = AsyncMock(side_effect=KafkaConnectionError())
    first_producer.stop = AsyncMock()

    second_producer = MagicMock()
    second_producer.start = AsyncMock()
    second_producer.stop = AsyncMock()

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            side_effect=[first_producer, second_producer],
        ) as producer_cls,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await bus.start()

    assert producer_cls.call_count == 2
    first_producer.stop.assert_awaited_once()
    second_producer.start.assert_awaited_once()
    assert bus._producer is second_producer

    await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consumer_start_retries_kafka_connection_error() -> None:
    bus = EventBusKafka(config=_retry_config())

    producer = MagicMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        await bus.start()

    first_consumer = MagicMock()
    first_consumer.start = AsyncMock(side_effect=KafkaConnectionError())
    first_consumer.stop = AsyncMock()

    second_consumer = MagicMock()
    second_consumer.start = AsyncMock()
    second_consumer.stop = AsyncMock()

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            side_effect=[first_consumer, second_consumer],
        ) as consumer_cls,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await bus.subscribe(
            "onex.evt.omnimarket.pattern-b-dispatch-completed.v1",
            on_message=AsyncMock(),
            group_id="codex-adapter-bootstrap-retry",
        )

    assert consumer_cls.call_count == 2
    first_consumer.stop.assert_awaited_once()
    second_consumer.start.assert_awaited_once()

    await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consumer_start_retries_timeout_error() -> None:
    bus = EventBusKafka(config=_retry_config())

    producer = MagicMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        await bus.start()

    first_consumer = MagicMock()
    first_consumer.start = AsyncMock(side_effect=TimeoutError())
    first_consumer.stop = AsyncMock()

    second_consumer = MagicMock()
    second_consumer.start = AsyncMock()
    second_consumer.stop = AsyncMock()

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
            side_effect=[first_consumer, second_consumer],
        ) as consumer_cls,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await bus.subscribe(
            "onex.evt.omnimarket.pattern-b-dispatch-completed.v1",
            on_message=AsyncMock(),
            group_id="codex-adapter-bootstrap-timeout-retry",
        )

    assert consumer_cls.call_count == 2
    first_consumer.stop.assert_awaited_once()
    second_consumer.start.assert_awaited_once()

    await bus.close()
