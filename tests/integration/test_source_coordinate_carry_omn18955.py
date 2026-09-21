# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18955 — the source coordinates survive the trip from bus to dispatch.

RE-EXPRESSED UNDER OMN-18918. This module was written against a task-local
context channel (``bind_source_coordinate`` / ``current_source_coordinate``).
That channel is gone: the operator ruled on 2026-09-20 for an explicit typed
parameter on ``ProtocolDispatchEngine.dispatch`` over ambient state, on the
ground that partition and offset are facts about a DELIVERY rather than about
the event, and the protocol should say what it carries. The assertions below
are the same four properties, re-pointed at the surviving mechanism. They are
kept rather than deleted because the properties are the transport's, not the
mechanism's, and they would have to be re-invented otherwise.

The premise the whole fix rests on is unchanged: a projection writer publishes
its snapshot deltas at partition 0 / offset 0 unless the record it is reacting
to carried real ones. Measured on the .201 dev lane before either fix landed:
6,210,195 lifetime drops on the consumer-flow exposure, and a readiness
endpoint answering 503 because it correctly refused to call that healthy.

The seam, in the order the runtime uses it:

    publish -> the bus builds a ModelEventMessage carrying (partition, offset)
            -> the consume callback builds a ModelMessageDeliveryContext from
               it, in the frame that still holds the record
            -> the context travels to the projection site as an argument

Both directions are asserted, because only one of them is the defect. A record
with coordinates must produce them; a record without must produce NOTHING,
since a defaulted zero is precisely the bug being removed -- the snapshot cache
drops a delta whose source offset does not exceed the cached one for the same
source topic and partition, so a constant zero means every delta after the
first for a key is discarded as an idempotent replay and the exposure freezes
at lag zero.

The in-memory bus is the transport here rather than Kafka. It is the same
``ModelEventMessage``, populated by the same contract (monotonic per-topic
offset, partition 0), and it makes the assertion run in CI without a broker.
The Kafka path populates the identical fields straight off the aiokafka record.
"""

from __future__ import annotations

import asyncio
from typing import Final

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.delivery_context import (
    delivery_context_from_message,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TOPIC: Final[str] = "onex.evt.omnibase-infra.omn18955-coordinate-carry.v1"


async def _publish_and_capture(
    bus: EventBusInmemory, payloads: list[bytes]
) -> list[ModelEventMessage]:
    """Publish each payload and return the records the subscriber actually saw."""
    received: list[ModelEventMessage] = []
    delivered = asyncio.Event()

    async def _on_message(message: ModelEventMessage) -> None:
        received.append(message)
        if len(received) == len(payloads):
            delivered.set()

    await bus.subscribe(_TOPIC, on_message=_on_message, group_id="omn18955-carry")
    for payload in payloads:
        await bus.publish(_TOPIC, key=None, value=payload)
    await asyncio.wait_for(delivered.wait(), timeout=10)
    return received


class TestTheTransportSuppliesCoordinates:
    """The premise. Everything downstream is vacuous if this is false."""

    async def test_every_delivered_record_carries_a_partition_and_an_offset(
        self,
    ) -> None:
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b", b"c"])
        finally:
            await bus.shutdown()

        assert len(received) == 3
        for message in received:
            assert message.partition is not None
            assert message.offset is not None

    async def test_the_offsets_advance_across_records(self) -> None:
        # The defect this fix removes is a CONSTANT coordinate. A transport that
        # reported the same offset for every record would reproduce it exactly,
        # so the advance is asserted rather than assumed.
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b", b"c"])
        finally:
            await bus.shutdown()

        offsets = [int(str(message.offset)) for message in received]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == 3, (
            f"offsets did not advance across three records: {offsets} -- a constant "
            "coordinate is the defect itself, not a fix for it"
        )


class TestTheCoordinateReachesTheProjectionSite:
    """The carry. Bus record -> typed delivery context -> projection argument."""

    async def test_the_context_carries_the_records_own_coordinates(self) -> None:
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b"])
        finally:
            await bus.shutdown()

        # What the consume callback does with each record, in the same order:
        # build in the frame holding the record, hand it to dispatch as an
        # argument.
        contexts = [
            delivery_context_from_message(message, _TOPIC) for message in received
        ]

        assert all(context is not None for context in contexts)
        assert [
            (context.partition, context.offset)
            for context in contexts
            if context is not None
        ] == [
            (int(message.partition or 0), int(str(message.offset)))
            for message in received
        ]
        # And distinct per record, which is the property the snapshot cache's
        # replay guard actually consumes.
        assert contexts[0] != contexts[1]

    async def test_a_record_without_coordinates_resolves_to_nothing(self) -> None:
        # The negative direction, and the one that proves the fix did not simply
        # default to zero. A transport that cannot report coordinates must leave
        # the projection site with no coordinate at all.
        #
        # Derived from a REAL delivered record with the coordinates stripped,
        # rather than hand-built: a synthetic message could diverge from the
        # shape the bus actually emits and then assert nothing about it.
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a"])
        finally:
            await bus.shutdown()

        assert delivery_context_from_message(received[0], _TOPIC) is not None

        bare = received[0].model_copy(update={"partition": None, "offset": None})
        assert isinstance(bare, ModelEventMessage)
        assert delivery_context_from_message(bare, _TOPIC) is None

    async def test_one_records_coordinates_cannot_reach_another(self) -> None:
        # The leak property, and the reason it is now cheap to hold. Under the
        # retired context channel this needed a reset on every exit path,
        # because a missed reset would attribute one record's offset to the
        # next and read downstream exactly like the constant-coordinate defect.
        # A value passed as an argument cannot leak: there is no ambient slot
        # for it to persist in. The assertion stays because the PROPERTY still
        # has to hold, not because the mechanism is still at risk of breaking
        # it -- if a future change reintroduces ambient state here, this is the
        # test that notices.
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b"])
        finally:
            await bus.shutdown()

        first = delivery_context_from_message(received[0], _TOPIC)
        second = delivery_context_from_message(received[1], _TOPIC)

        assert first is not None and second is not None
        assert first != second
        assert first.offset != second.offset
        # Rebuilding the first record's context after the second one exists
        # returns the first record's own coordinates, unchanged.
        assert delivery_context_from_message(received[0], _TOPIC) == first
