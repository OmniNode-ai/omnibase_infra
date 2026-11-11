# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21228: transactional publish on the in-memory bus.

The in-memory bus carries the same call as the Kafka bus: a producer is opened
under a caller-fixed ``transactional_id``, a batch published through it is
visible whole or not at all, and a newer producer under the same id fences the
older one so that the older one's next publish fails and leaves nothing behind.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from omnibase_infra.errors import InfraUnavailableError, ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.handlers.transactional_producer_fencing import (
    InfraProducerFencedError,
)
from omnibase_infra.models.event_bus.model_transactional_record import (
    ModelTransactionalRecord,
)
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.unit

TOPIC = "test.transactional.topic"


def _record(value: bytes, topic: str = TOPIC) -> ModelTransactionalRecord:
    return ModelTransactionalRecord(topic=topic, key=b"k", value=value)


@pytest.fixture
async def bus() -> AsyncIterator[EventBusInmemory]:
    event_bus = EventBusInmemory(environment="test", group="txn")
    await event_bus.start()
    yield event_bus
    await event_bus.close()


async def test_batch_is_published_whole_with_receipts(bus: EventBusInmemory) -> None:
    producer = await bus.transactional_producer("txn-id-1")

    receipts = await producer.publish([_record(b"a"), _record(b"b")])

    history = await bus.get_event_history()
    assert [m.value for m in history] == [b"a", b"b"]
    assert [r.offset for r in receipts] == [0, 1]
    assert all(r.topic == TOPIC for r in receipts)


async def test_newer_producer_with_same_id_fences_the_older_one(
    bus: EventBusInmemory,
) -> None:
    first = await bus.transactional_producer("txn-id-2")
    await first.publish([_record(b"before-fence")])

    second = await bus.transactional_producer("txn-id-2")

    with pytest.raises(InfraProducerFencedError):
        await first.publish([_record(b"from-fenced-producer")])
    await second.publish([_record(b"from-newer-producer")])

    values = [m.value for m in await bus.get_event_history()]
    assert values == [b"before-fence", b"from-newer-producer"]


async def test_fenced_producer_stays_fenced(bus: EventBusInmemory) -> None:
    first = await bus.transactional_producer("txn-id-3")
    await bus.transactional_producer("txn-id-3")

    for _ in range(2):
        with pytest.raises(InfraProducerFencedError):
            await first.publish([_record(b"x")])

    assert await bus.get_event_history() == []


async def test_distinct_ids_do_not_fence_each_other(bus: EventBusInmemory) -> None:
    one = await bus.transactional_producer("txn-id-4a")
    two = await bus.transactional_producer("txn-id-4b")

    await one.publish([_record(b"one")])
    await two.publish([_record(b"two")])

    values = [m.value for m in await bus.get_event_history()]
    assert values == [b"one", b"two"]


async def test_subscriber_receives_committed_records_only(
    bus: EventBusInmemory,
) -> None:
    received: list[bytes] = []

    async def on_message(message: ModelEventMessage) -> None:
        received.append(message.value)

    await bus.subscribe(TOPIC, make_test_node_identity(), on_message)
    first = await bus.transactional_producer("txn-id-5")
    await bus.transactional_producer("txn-id-5")

    with pytest.raises(InfraProducerFencedError):
        await first.publish([_record(b"never-delivered")])

    assert received == []


async def test_closed_producer_refuses_publish(bus: EventBusInmemory) -> None:
    producer = await bus.transactional_producer("txn-id-6")
    await producer.close()

    with pytest.raises(InfraUnavailableError):
        await producer.publish([_record(b"x")])


async def test_empty_batch_is_refused(bus: EventBusInmemory) -> None:
    producer = await bus.transactional_producer("txn-id-7")

    with pytest.raises(ProtocolConfigurationError):
        await producer.publish([])


async def test_unstarted_bus_refuses_a_producer() -> None:
    event_bus = EventBusInmemory(environment="test", group="txn")

    with pytest.raises(InfraUnavailableError):
        await event_bus.transactional_producer("txn-id-8")
