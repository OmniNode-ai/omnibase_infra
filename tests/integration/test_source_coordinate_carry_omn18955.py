# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18955 — the source coordinates survive the trip from bus to dispatch.

The unit suite (``tests/unit/runtime/test_source_coordinate_injection_omn18955.py``)
pins the channel and the injection site in isolation. Neither of those can fail if
the TRANSPORT stops supplying the coordinates in the first place, and that is the
premise the whole fix rests on: a projection writer publishes its snapshot deltas
at partition 0 / offset 0 unless the record it is reacting to carried real ones.

So this module exercises the seam through a real event bus, end to end and in the
order the runtime uses it:

    publish -> the bus builds a ModelEventMessage carrying (partition, offset)
            -> the consume callback binds it, in the frame that still holds it
            -> a projection-shaped reader resolves the pair from the channel

Both directions are asserted, because only one of them is the defect. A record
with coordinates must produce them; a record without must produce NOTHING, since
a defaulted zero is precisely the bug being removed -- the snapshot cache drops a
delta whose source offset does not exceed the cached one for the same source
topic and partition, so a constant zero means every delta after the first for a
key is discarded as an idempotent replay and the exposure freezes at lag zero.

The in-memory bus is the transport here rather than Kafka. It is the same
``ModelEventMessage``, populated by the same contract (monotonic per-topic offset,
partition 0), and it makes the assertion run in CI without a broker. The Kafka
path populates the identical fields straight off the aiokafka record.
"""

from __future__ import annotations

import asyncio
from typing import Final

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_source_coordinate,
    current_source_coordinate,
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


class TestTheCoordinateReachesTheProjectionReader:
    """The carry. Bus record -> consume-boundary bind -> projection-shaped read."""

    async def test_the_reader_resolves_the_records_own_coordinates(self) -> None:
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b"])
        finally:
            await bus.shutdown()

        # What the projection dispatch site does with the channel, in the same
        # order: bind in the frame holding the record, resolve where the payload
        # is assembled.
        resolved: list[tuple[int, str] | None] = []
        for message in received:
            with bind_source_coordinate(message):
                resolved.append(current_source_coordinate())

        assert all(pair is not None for pair in resolved)
        assert resolved == [
            (int(message.partition or 0), str(message.offset)) for message in received
        ]
        # And distinct per record, which is the property the snapshot cache's
        # replay guard actually consumes.
        assert resolved[0] != resolved[1]

    async def test_a_record_without_coordinates_resolves_to_nothing(self) -> None:
        # The negative direction, and the one that proves the fix did not simply
        # default to zero. A transport that cannot report coordinates must leave
        # the reader with no key at all.
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

        with bind_source_coordinate(received[0]):
            assert current_source_coordinate() is not None  # positive control

        bare = received[0].model_copy(update={"partition": None, "offset": None})
        assert isinstance(bare, ModelEventMessage)
        with bind_source_coordinate(bare):
            assert current_source_coordinate() is None

    async def test_the_channel_does_not_leak_between_records(self) -> None:
        # Consecutive deliveries on one task must not see each other's
        # coordinates; a leak would attribute one record's offset to the next and
        # is indistinguishable from the constant-coordinate defect downstream.
        bus = EventBusInmemory()
        await bus.start()
        try:
            received = await _publish_and_capture(bus, [b"a", b"b"])
        finally:
            await bus.shutdown()

        with bind_source_coordinate(received[0]):
            first = current_source_coordinate()
        assert current_source_coordinate() is None
        with bind_source_coordinate(received[1]):
            second = current_source_coordinate()
        assert current_source_coordinate() is None
        assert first != second
