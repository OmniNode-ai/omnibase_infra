# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end recovery from a coordinator-loss wedge (OMN-18640).

The unit tests cover the detector and the supervisor in isolation. This one
drives the surface a runtime actually uses -- ``EventBusKafka.subscribe`` --
through the whole sequence measured on the .201 dev lane at
2026-09-18T23:16:46Z: the client keeps its group membership, the partition
leaders keep answering, the fetch position stays pinned at 7473 while the log
end offset walks to 7489, and no record reaches the handler.

What it adds over the unit coverage is the wiring: the real consumer
construction path, the real rejoin rebuild, the real dispatch to a subscriber
callback, and the real bus-level rejoin record. Each of those is a place the
recovery could be correct in isolation and unreachable in the runtime, which
is exactly what happened to the detector that did not exist: the loop could
not have used one, because it was waiting on an iterator that never returned.

The Kafka client is substituted at the ``AIOKafkaConsumer`` seam rather than
run against a live broker, because the fault being reproduced is a CLIENT
state -- a coordinator marked dead while every leader is reachable -- and no
broker-side action produces it on demand. Provoking it for real means
restarting a shared lane broker, which is exactly the mutation this ticket
forbids.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.integration

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
PARTITION = TopicPartition(TOPIC, 0)

# Read from the .201 dev lane at 2026-09-19T00:07:29Z, 50 minutes into the
# second wedge: the committed offset had not moved since 23:52Z while the
# topic end kept advancing.
WEDGED_POSITION = 7473
LOG_END_OFFSET = 7489
BACKLOG = LOG_END_OFFSET - WEDGED_POSITION


def _record(offset: int) -> Any:
    """A record shaped like the ones the OCC autobind command carries."""

    class _Record:
        topic = TOPIC
        partition = 0
        timestamp = 0
        headers: tuple[tuple[str, bytes], ...] = (
            ("event_type", b"occ_autobind_requested"),
            ("source", b"omn18640-integration"),
        )

        def __init__(self, at: int) -> None:
            self.offset = at
            self.key = f"key-{at}".encode()
            self.value = json.dumps({"offset": at}).encode()

    return _Record(offset)


class _FakeKafkaConsumer:
    """Stands in for ``AIOKafkaConsumer`` and can be told to wedge.

    ``wedged`` is the measured shape and not an approximation of it: the
    assignment is retained, an end-offset probe against the leaders succeeds
    and reports a growing log end, the fetch position never moves, and
    ``getmany`` returns nothing however long it is called.
    """

    #: Every instance built during a test, in construction order. The first is
    #: the wedged one; a second existing at all is the recovery.
    instances: list[_FakeKafkaConsumer] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.topic = args[0] if args else kwargs.get("topic", TOPIC)
        self.group_id = kwargs.get("group_id", "")
        self.group_instance_id = kwargs.get("group_instance_id", "")
        self.member_id = f"member-{len(_FakeKafkaConsumer.instances)}"
        self.started = False
        self.stopped = False
        self.wedged = len(_FakeKafkaConsumer.instances) == 0
        self._position = WEDGED_POSITION
        self._pending: list[Any] = (
            []
            if self.wedged
            else [_record(WEDGED_POSITION + n) for n in range(BACKLOG)]
        )
        _FakeKafkaConsumer.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> Mapping[TopicPartition, Sequence[Any]]:
        if self.wedged:
            # A wedged client returns from getmany at its deadline having
            # fetched nothing, forever. Honour the deadline so the loop is
            # paced the way a real client paces it.
            await asyncio.sleep(max(timeout_ms, 1) / 1000.0)
            return {}
        if not self._pending:
            await asyncio.sleep(max(timeout_ms, 1) / 1000.0)
            return {}
        drained = self._pending
        self._pending = []
        self._position += len(drained)
        return {PARTITION: drained}

    def assignment(self) -> set[TopicPartition]:
        return {PARTITION}

    async def position(self, partition: TopicPartition) -> int:
        return self._position

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {PARTITION: LOG_END_OFFSET}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._position = offset


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _FakeKafkaConsumer.instances = []


@pytest.mark.asyncio
async def test_a_subscribed_consumer_recovers_from_a_coordinator_wedge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subscribe -> wedge -> self-rejoin -> the handler receives the backlog."""
    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _FakeKafkaConsumer,
    )

    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
            # Compressed so the test asserts the mechanism rather than the
            # clock; the shipped defaults are asserted separately in
            # tests/unit/event_bus/test_consumer_rejoin_on_coordinator_loss.py.
            consumer_poll_timeout_ms=100,
            consumer_stall_seconds=0.3,
            consumer_stall_required_confirmations=2,
            consumer_rejoin_cooldown_seconds=0.0,
        )
    )
    bus._started = True

    received: list[ModelEventMessage] = []
    delivered = asyncio.Event()

    async def handler(message: ModelEventMessage) -> None:
        received.append(message)
        if len(received) >= BACKLOG:
            delivered.set()

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn18640"), handler
    )

    try:
        await asyncio.wait_for(delivered.wait(), timeout=15.0)
    finally:
        await unsubscribe()
        bus._shutdown = True

    # The wedged client was closed and exactly one replacement was built.
    assert len(_FakeKafkaConsumer.instances) == 2, (
        "expected one wedged consumer and one replacement, got "
        f"{len(_FakeKafkaConsumer.instances)}"
    )
    wedged, replacement = _FakeKafkaConsumer.instances
    assert wedged.wedged and wedged.stopped
    assert replacement.started
    assert replacement.member_id != wedged.member_id

    # The replacement rejoined the SAME group -- this is a recovery, not a
    # fresh subscription that would read from a different offset.
    assert replacement.group_id == wedged.group_id
    assert replacement.group_instance_id == wedged.group_instance_id

    # Offsets advanced to the log end, and every withheld record arrived.
    assert await replacement.position(PARTITION) == LOG_END_OFFSET
    assert len(received) == BACKLOG

    # The typed record is on the bus, readable with no feature flag.
    events = bus.consumer_rejoin_events()
    assert len(events) == 1
    assert events[0].topic == TOPIC
    assert events[0].reason is EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING
    assert events[0].rejoin_succeeded is True
    assert events[0].backlog_records == BACKLOG


@pytest.mark.asyncio
async def test_a_healthy_subscription_is_never_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control on the same wiring: no wedge, no rejoin, ever.

    Without this, the test above passes equally well against a bus that
    recreates its consumer on every quiet poll -- which would be a worse
    outage than the one being fixed.
    """

    class _HealthyConsumer(_FakeKafkaConsumer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.wedged = False
            self._pending = []
            # Caught up: the fetch position is already at the log end, so
            # silence is idleness.
            self._position = LOG_END_OFFSET

        async def end_offsets(
            self, partitions: Collection[TopicPartition]
        ) -> Mapping[TopicPartition, int]:
            return {PARTITION: LOG_END_OFFSET}

    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _HealthyConsumer,
    )

    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
            consumer_poll_timeout_ms=100,
            consumer_stall_seconds=0.3,
            consumer_stall_required_confirmations=2,
            consumer_rejoin_cooldown_seconds=0.0,
        )
    )
    bus._started = True

    async def handler(_message: ModelEventMessage) -> None:
        return None

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn18640-control"), handler
    )
    try:
        # Many multiples of the stall window, all of them quiet.
        await asyncio.sleep(3.0)
    finally:
        await unsubscribe()
        bus._shutdown = True

    assert len(_FakeKafkaConsumer.instances) == 1, (
        "an idle consumer must never be recreated, however long it stays quiet"
    )
    assert bus.consumer_rejoin_events() == ()
