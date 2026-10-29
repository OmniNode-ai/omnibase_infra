# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A one-shot reply read joins no consumer group and leaves none behind (OMN-17427).

The in-memory and Kafka buses answer the same operation, ``subscribe_ungrouped``.
The Kafka leg is proved against a stubbed aiokafka consumer: what the operation
promises is a consumer built with ``group_id=None`` and no auto-commit, started
only once every assigned position is resolved, and stopped by the unsubscribe.
The broker-side count (the group list is unchanged after N calls) is the lab
step of the owning lane, not a unit test.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import InfraTimeoutError, InfraUnavailableError
from omnibase_infra.event_bus import event_bus_kafka as kafka_module
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TOPIC = "onex.evt.test.ungrouped-reply.v1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inmemory_ungrouped_sees_only_later_records_and_leaves_nothing() -> None:
    bus = EventBusInmemory(environment="test", group="g")
    await bus.start()
    await bus.publish(TOPIC, None, b"before")
    seen: list[bytes] = []

    async def on_message(message: ModelEventMessage) -> None:
        seen.append(message.value)

    unsubscribe = await bus.subscribe_ungrouped(TOPIC, on_message)
    await bus.publish(TOPIC, None, b"after")
    assert seen == [b"after"]
    await unsubscribe()
    await bus.publish(TOPIC, None, b"after-unsubscribe")
    assert seen == [b"after"]
    assert not bus._subscribers.get(TOPIC)
    await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inmemory_ungrouped_readers_do_not_share_a_subscription() -> None:
    bus = EventBusInmemory(environment="test", group="g")
    await bus.start()
    first: list[bytes] = []
    second: list[bytes] = []

    async def on_first(message: ModelEventMessage) -> None:
        first.append(message.value)

    async def on_second(message: ModelEventMessage) -> None:
        second.append(message.value)

    await bus.subscribe_ungrouped(TOPIC, on_first)
    await bus.subscribe_ungrouped(TOPIC, on_second)
    await bus.publish(TOPIC, None, b"reply")
    assert first == [b"reply"]
    assert second == [b"reply"]
    await bus.close()


def _consumer(partitions: set[TopicPartition]) -> MagicMock:
    consumer = MagicMock(spec=AIOKafkaConsumer)
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.assignment.return_value = partitions
    consumer.position = AsyncMock(return_value=0)
    consumer.getone = AsyncMock(side_effect=lambda: asyncio.Event().wait())
    return consumer


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_ungrouped_builds_a_group_less_consumer_that_commits_nothing() -> (
    None
):
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    consumer = _consumer({TopicPartition(TOPIC, 0), TopicPartition(TOPIC, 1)})
    with patch.object(kafka_module, "AIOKafkaConsumer", return_value=consumer) as built:
        unsubscribe = await bus.subscribe_ungrouped(TOPIC, AsyncMock())
    kwargs = built.call_args.kwargs
    assert kwargs["group_id"] is None
    assert kwargs["enable_auto_commit"] is False
    assert kwargs["auto_offset_reset"] == "latest"
    assert consumer.position.await_count == 2
    # No group was registered on the bus: nothing for a broker to remember.
    assert not bus._group_consumers
    assert not bus._subscribers.get(TOPIC)
    await unsubscribe()
    consumer.stop.assert_awaited_once()
    await unsubscribe()
    consumer.stop.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_ungrouped_delivers_the_record_as_a_bus_message() -> None:
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    consumer = _consumer({TopicPartition(TOPIC, 0)})
    record = MagicMock()
    record.key = b"k"
    record.value = b"reply"
    record.offset = 7
    record.partition = 0
    record.headers = []
    delivered = asyncio.Event()
    got: list[ModelEventMessage] = []
    waits = iter([record])

    async def getone() -> object:
        try:
            return next(waits)
        except StopIteration:
            await asyncio.Event().wait()
            raise

    consumer.getone = getone

    async def on_message(message: ModelEventMessage) -> None:
        got.append(message)
        delivered.set()

    with patch.object(kafka_module, "AIOKafkaConsumer", return_value=consumer):
        unsubscribe = await bus.subscribe_ungrouped(TOPIC, on_message)
    await asyncio.wait_for(delivered.wait(), timeout=1)
    assert got[0].value == b"reply"
    assert got[0].topic == TOPIC
    assert got[0].offset == "7"
    await unsubscribe()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_ungrouped_start_waits_for_every_position() -> None:
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    bus._timeout_seconds = 1
    consumer = _consumer({TopicPartition(TOPIC, 0)})

    async def never(partition: TopicPartition) -> int:
        await asyncio.Event().wait()
        return 0

    consumer.position = AsyncMock(side_effect=never)
    with patch.object(kafka_module, "AIOKafkaConsumer", return_value=consumer):
        with pytest.raises(InfraTimeoutError):
            await bus.subscribe_ungrouped(TOPIC, AsyncMock())
    consumer.stop.assert_awaited_once()
    assert not bus._ungrouped_readers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_ungrouped_requires_a_started_bus() -> None:
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    with pytest.raises(InfraUnavailableError):
        await bus.subscribe_ungrouped(TOPIC, AsyncMock())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kafka_close_stops_a_reader_nobody_unsubscribed() -> None:
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    consumer = _consumer({TopicPartition(TOPIC, 0)})
    with patch.object(kafka_module, "AIOKafkaConsumer", return_value=consumer):
        await bus.subscribe_ungrouped(TOPIC, AsyncMock())
    await bus.close()
    consumer.stop.assert_awaited_once()
    assert not bus._ungrouped_readers
