# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A subscription keeps a polling member through a failed poll and a dead task (OMN-17427).

The unit tests in ``tests/unit/event_bus/test_consume_loop_survival_omn17427.py``
drive ``_consume_loop`` and the watchdog directly. This one drives the surface a
runtime uses -- ``EventBusKafka.subscribe`` -- through the two ways a group was
lost on the dev lane on 2026-10-22: the poll raising (the consume task used to
end on the first exception, with its consumer still started), and the consume
task ending for any other reason (nothing restarted it). In both, aiokafka kept
the membership through heartbeats for ``max_poll_interval_ms`` and then left the
group for good, because it rejoins only inside the next poll.

What it adds over the unit coverage is the wiring: the real consumer
construction path, the real replacement rebuild in the same group, the real
dispatch to a subscriber callback and the real watchdog started by
``subscribe``. The Kafka client is substituted at the ``AIOKafkaConsumer`` seam,
because the faults are client states that no broker action produces on demand,
and provoking them for real means restarting a shared lane broker.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from aiokafka.errors import KafkaConnectionError
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.integration

TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: replay of a recorded incident
PARTITION = TopicPartition(TOPIC, 0)
RECORDS = 4


def _record(offset: int) -> Any:
    """A record shaped like a node heartbeat."""

    class _Record:
        topic = TOPIC
        partition = 0
        timestamp = 0
        headers: tuple[tuple[str, bytes], ...] = (
            ("event_type", b"node_heartbeat"),
            ("source", b"omn17427-integration"),
        )

        def __init__(self, at: int) -> None:
            self.offset = at
            self.key = f"key-{at}".encode()
            self.value = json.dumps({"offset": at}).encode()

    return _Record(offset)


class _FakeKafkaConsumer:
    """Stands in for ``AIOKafkaConsumer``; the first one built can be told to fail.

    ``fail_first_poll`` makes the first instance's first ``getmany`` raise, the
    way a poll raises a coordinator error pushed to the user after a broker
    restart. Every instance delivers ``RECORDS`` records once and is then idle
    and caught up, so a replacement is the only way the records arrive after a
    failure.
    """

    instances: list[_FakeKafkaConsumer] = []
    fail_first_poll = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.topic = args[0] if args else kwargs.get("topic", TOPIC)
        self.group_id = kwargs.get("group_id", "")
        self.group_instance_id = kwargs.get("group_instance_id", "")
        self.started = False
        self.stopped = False
        self.polls = 0
        self.failing = (
            _FakeKafkaConsumer.fail_first_poll and not _FakeKafkaConsumer.instances
        )
        self._position = 0
        self._pending: list[Any] = [_record(n) for n in range(RECORDS)]
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
        self.polls += 1
        if self.failing:
            raise KafkaConnectionError("Connection at broker:9092 closed")
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
        return {PARTITION: self._position}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._position = offset


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _FakeKafkaConsumer.instances = []
    _FakeKafkaConsumer.fail_first_poll = False


def _bus() -> EventBusKafka:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
            # Compressed so the test asserts the mechanism rather than the
            # clock; the shipped defaults are asserted in the unit tests.
            consumer_poll_timeout_ms=100,
            consumer_rejoin_cooldown_seconds=0.0,
            consume_loop_failure_backoff_initial_seconds=0.05,
            consume_loop_failure_backoff_max_seconds=0.1,
            consume_task_watchdog_interval_seconds=0.05,
        )
    )
    bus._started = True
    return bus


@pytest.mark.asyncio
async def test_a_subscription_survives_a_failed_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subscribe -> the poll raises -> the consumer is replaced in the same group -> records arrive."""
    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _FakeKafkaConsumer,
    )
    _FakeKafkaConsumer.fail_first_poll = True
    bus = _bus()

    received: list[ModelEventMessage] = []
    delivered = asyncio.Event()

    async def handler(message: ModelEventMessage) -> None:
        received.append(message)
        if len(received) >= RECORDS:
            delivered.set()

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn17427-poll"), handler
    )
    try:
        await asyncio.wait_for(delivered.wait(), timeout=15.0)
        tasks = list(bus._group_consumer_tasks.values())
        assert tasks, "the subscription has a consume task"
        assert all(not task.done() for task in tasks), (
            "the consume task is still running after the failed poll"
        )
    finally:
        await unsubscribe()
        bus._shutdown = True

    failed, replacement = _FakeKafkaConsumer.instances[:2]
    assert failed.failing and failed.stopped
    assert replacement.started and not replacement.failing
    # The replacement joined the SAME group: a recovery, not a fresh
    # subscription reading from a different committed offset.
    assert replacement.group_id == failed.group_id
    assert replacement.group_instance_id == failed.group_instance_id
    assert len(received) == RECORDS


@pytest.mark.asyncio
async def test_the_watchdog_restarts_a_consume_task_that_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subscribe -> the consume task ends -> the watchdog rebuilds the consumer and a new loop delivers."""
    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _FakeKafkaConsumer,
    )
    bus = _bus()

    received: list[ModelEventMessage] = []
    first_batch = asyncio.Event()
    second_batch = asyncio.Event()

    async def handler(message: ModelEventMessage) -> None:
        received.append(message)
        if len(received) >= RECORDS:
            first_batch.set()
        if len(received) >= 2 * RECORDS:
            second_batch.set()

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn17427-watchdog"), handler
    )
    try:
        await asyncio.wait_for(first_batch.wait(), timeout=15.0)
        (consumer_key, task), *_ = bus._group_consumer_tasks.items()
        # End the consume task the way an escaped error or a stray cancel
        # did: the subscription stays registered and the consumer started.
        task.cancel()
        await asyncio.wait({task}, timeout=5.0)
        assert task.done()

        await asyncio.wait_for(second_batch.wait(), timeout=15.0)
        restarted = bus._group_consumer_tasks[consumer_key]
        assert restarted is not task
        assert not restarted.done()
    finally:
        await unsubscribe()
        bus._shutdown = True

    original, replacement = _FakeKafkaConsumer.instances[:2]
    assert original.stopped
    assert replacement.started
    assert replacement.group_id == original.group_id
    assert replacement.group_instance_id == original.group_instance_id

    reasons = [event.reason for event in bus.consumer_rejoin_events()]
    assert EnumConsumerStallReason.CONSUME_TASK_DEAD in reasons
