# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay coordinator loss after broker restart (OMN-21337)."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Collection, Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aiokafka.consumer.group_coordinator import GroupCoordinator
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.consumer_rejoin_policy import (
    ModelConsumerRejoinPolicy,
)
from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
    ConsumerRejoinSupervisor,
)

if TYPE_CHECKING:
    from omnibase_infra.models.health.model_consumer_rejoin_alarm import (
        ModelConsumerRejoinAlarm,
    )

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
GROUP = (
    "local.omnimarket.pr_lifecycle_fix_effect.consume.1.0.0"
    ".__i.runtime-effects.__t.onex.cmd.omnimarket.occ-autobind.v1"
)
PARTITION = TopicPartition(TOPIC, 0)

POLICY = ModelConsumerRejoinPolicy(
    stall_seconds=120.0,
    required_consecutive_stalls=3,
    rejoin_cooldown_seconds=300.0,
    sync_unready_seconds=600.0,
)


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakeConsumer:
    """A typed ``ProtocolRejoinableConsumer`` that can be told to wedge.

    ``wedged`` reproduces the exact incident shape: ``getmany`` returns nothing
    however long it is called, the assignment is retained, the leaders answer
    an end-offset probe, and the fetch position never moves.
    """

    def __init__(
        self,
        *,
        member_id: str,
        position: int,
        end_offsets: Sequence[int],
        wedged: bool,
        leaders_reachable: bool = True,
        assigned: bool = True,
        pending_records: Sequence[Any] = (),
    ) -> None:
        self._coordinator = SimpleNamespace(coordinator_id=None)
        self.hang_position = False
        self.hang_end_offsets = False
        self.member_id = member_id
        self.stopped = False
        self.getmany_calls = 0
        self.end_offset_probes = 0
        self._position = position
        self._end_offsets = list(end_offsets)
        self._wedged = wedged
        self._leaders_reachable = leaders_reachable
        self._assigned = assigned
        self._pending: list[Any] = list(pending_records)

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> Mapping[TopicPartition, Sequence[Any]]:
        self.getmany_calls += 1
        if self._wedged or not self._pending:
            return {}
        drained = self._pending
        self._pending = []
        self._position += len(drained)
        return {PARTITION: drained}

    def assignment(self) -> set[TopicPartition]:
        return {PARTITION} if self._assigned else set()

    async def position(self, partition: TopicPartition) -> int:
        if self.hang_position:
            await asyncio.Event().wait()
        return self._position

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        if self.hang_end_offsets:
            await asyncio.Event().wait()
        if not self._leaders_reachable:
            raise ConnectionError("leaders unreachable")
        self.end_offset_probes += 1
        index = min(self.end_offset_probes - 1, len(self._end_offsets) - 1)
        return {PARTITION: self._end_offsets[index]}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._position = offset

    async def stop(self) -> None:
        self.stopped = True


class Harness:
    def __init__(self, *, fail: bool = False, **kwargs: Any) -> None:
        self.clock = FakeClock()
        self.consumer = FakeConsumer(
            member_id="original", position=10, end_offsets=(10,), wedged=True
        )
        self.replacement = FakeConsumer(
            member_id="replacement", position=10, end_offsets=(10,), wedged=True
        )
        self.replacement._coordinator.coordinator_id = 0
        self.recreates = 0
        self.fail = fail

        async def recreate() -> FakeConsumer:
            self.recreates += 1
            if self.fail:
                raise ConnectionError("replacement cannot start")
            return self.replacement

        self.supervisor = ConsumerRejoinSupervisor(
            topic=TOPIC,
            group_id=GROUP,
            policy=POLICY,
            poll_timeout_ms=1,
            recreate_consumer=recreate,
            clock=self.clock,
            **kwargs,
        )

    async def poll(self, seconds: float = 5) -> Any:
        self.clock.advance(seconds)
        batch = await asyncio.wait_for(
            self.supervisor.next_batch(self.consumer), timeout=5
        )
        self.consumer = batch.consumer
        return batch


@pytest.mark.asyncio
async def test_consumer_rejoins_after_broker_restart_when_idle_and_coordinator_lost() -> (
    None
):
    h = Harness()  # Exercise constructor defaults.
    await h.poll(0)
    for _ in range(11):
        await h.poll()
    assert h.recreates == 0
    assert h.consumer.end_offset_probes == 0
    batch = await h.poll()
    assert h.recreates == 1
    assert batch.rejoined and batch.consumer is h.replacement
    assert batch.verdict.should_rejoin
    assert (
        h.supervisor.rejoin_events[0].reason.value == "stalled_coordinator_unavailable"
    )
    await h.poll()
    assert h.recreates == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_known", [False, True])
async def test_consumer_rejoins_after_broker_restart_when_position_waits_on_coordinator(
    reported_known: bool,
) -> None:
    h = Harness(probe_timeout_seconds=0.05)
    h.consumer._coordinator.coordinator_id = 0 if reported_known else None
    h.consumer._end_offsets = [100]
    h.consumer.hang_position = True
    h.clock.advance(125)
    await h.poll(0)
    for _ in range(13):
        batch = await h.poll()
        if batch.rejoined:
            break
    assert h.recreates == 1
    assert batch.consumer is h.replacement


@pytest.mark.asyncio
async def test_consumer_rejoins_after_broker_restart_not_while_broker_unreachable() -> (
    None
):
    h = Harness()
    h.consumer._leaders_reachable = False
    for _ in range(360):
        await h.poll()
    assert h.recreates == 0
    h.consumer._leaders_reachable = True
    await h.poll()
    for _ in range(11):
        await h.poll()
    assert h.recreates == 0
    await h.poll()
    assert h.recreates == 1


@pytest.mark.asyncio
async def test_consumer_rejoins_after_broker_restart_leaves_a_healthy_idle_consumer_alone() -> (
    None
):
    h = Harness()
    h.consumer._coordinator.coordinator_id = 0
    for _ in range(360):
        await h.poll()
    assert h.recreates == 0


@pytest.mark.asyncio
async def test_consumer_rejoin_failure_alarms_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
        COORDINATOR_ERROR_LOG_GATE,
    )
    from omnibase_infra.models.health.model_consumer_rejoin_alarm import (
        ModelConsumerRejoinAlarm,
    )

    emitted: list[ModelConsumerRejoinAlarm] = []

    async def emit(alarm: ModelConsumerRejoinAlarm) -> None:
        emitted.append(alarm)

    h = Harness(
        fail=True,
        broker="broker:9092",
        kafka_group_id="effective-kafka-group",
        emit_alarm=emit,
    )
    with caplog.at_level(logging.INFO):
        for _ in range(360):
            await h.poll()
    assert h.recreates > 10
    assert len(h.supervisor.alarms) == 1
    alarm = h.supervisor.alarms[0]
    assert isinstance(alarm, ModelConsumerRejoinAlarm)
    assert emitted == [alarm]
    records = [
        r
        for r in caplog.records
        if r.getMessage().startswith("consumer_group_rejoin_alarm ")
    ]
    assert len(records) == 1
    assert "effective-kafka-group" in records[0].getMessage()
    assert "broker:9092" in records[0].getMessage()
    assert h.supervisor.alarm_active
    coordinator_logger = logging.getLogger("aiokafka.consumer.group_coordinator")
    message = (
        "Group Coordinator Request failed: [Error 15] GroupCoordinatorNotAvailableError"
    )
    record = coordinator_logger.makeRecord(
        coordinator_logger.name, logging.ERROR, "", 0, message, (), None
    )
    assert not COORDINATOR_ERROR_LOG_GATE.filter(record)
    assert COORDINATOR_ERROR_LOG_GATE.filter(
        logging.makeLogRecord({"name": "aiokafka.other", "msg": "unrelated"})
    )
    before = len(caplog.records)
    coordinator_logger.error(message)
    assert len(caplog.records) == before
    h.consumer._wedged = False
    h.consumer._pending = [object()]
    await h.poll()


@pytest.mark.asyncio
async def test_consumer_rejoin_failure_alarms_once_per_episode_and_rearms_after_recovery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = Harness(fail=True)
    with caplog.at_level(logging.INFO):
        for _ in range(15):
            await h.poll()
        assert len(h.supervisor.alarms) == 1
        h.consumer._wedged = False
        h.consumer._pending = [object()]
        await h.poll()
        assert not h.supervisor.alarm_active
        assert any(
            r.getMessage().startswith("consumer_group_rejoin_recovered ")
            for r in caplog.records
        )
        for _ in range(30):
            await h.poll()
        assert len(h.supervisor.alarms) == 2
        h.consumer._pending = [object()]
        await h.poll()


def test_read_coordinator_known_pins_aiokafka_group_coordinator() -> None:
    from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
        read_coordinator_known,
    )

    assert "self.coordinator_id = None" in inspect.getsource(
        GroupCoordinator.coordinator_dead
    )
    assert read_coordinator_known(object()) is None
    consumer = SimpleNamespace(_coordinator=SimpleNamespace(coordinator_id=0))
    assert read_coordinator_known(consumer) is True
    consumer._coordinator.coordinator_id = None
    assert read_coordinator_known(consumer) is False
    consumer._coordinator = None
    assert read_coordinator_known(consumer) is None
    consumer._coordinator = object()
    assert read_coordinator_known(consumer) is None


def test_coordinator_log_gate_reference_counts_and_preserves_other_errors() -> None:
    from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
        CoordinatorErrorLogGate,
    )

    gate = CoordinatorErrorLogGate()
    record = logging.makeLogRecord(
        {
            "name": "aiokafka.consumer.group_coordinator",
            "msg": "GroupCoordinatorNotAvailable",
        }
    )
    gate.activate("first")
    gate.activate("first")
    gate.activate("second")
    gate.deactivate("first")
    gate.deactivate("second")
    assert not gate.filter(record)
    gate.deactivate("first")
    assert gate.filter(record)
    gate.activate("first")
    for message in (
        "Group Coordinator Request failed",
        "GroupCoordinatorNotAvailable",
        "is either not started or not valid",
    ):
        assert not gate.filter(
            logging.makeLogRecord({"name": "aiokafka.client", "msg": message})
        )
        assert gate.filter(
            logging.makeLogRecord({"name": "application", "msg": message})
        )
    assert gate.filter(
        logging.makeLogRecord(
            {"name": "aiokafka.client", "msg": "authorization failed"}
        )
    )


@pytest.mark.asyncio
async def test_end_offset_timeout_does_not_trigger_rejoin() -> None:
    h = Harness(probe_timeout_seconds=0.01)
    h.consumer.hang_end_offsets = True
    for _ in range(20):
        await h.poll(60)
    assert h.recreates == 0


@pytest.mark.asyncio
async def test_started_replacement_without_coordinator_counts_as_failed_rejoin() -> (
    None
):
    h = Harness()
    h.replacement._coordinator.coordinator_id = None
    for _ in range(25):
        await h.poll()
    assert h.recreates == 2
    assert len(h.supervisor.alarms) == 1
    assert h.supervisor.alarms[0].failed_rejoins == 1
    h.consumer._coordinator.coordinator_id = 0
    await h.poll()  # A known coordinator and progressing/idle evaluation recover.
    assert not h.supervisor.alarm_active


@pytest.mark.asyncio
async def test_backlog_rejoin_failure_alarms_without_claiming_coordinator_loss() -> (
    None
):
    h = Harness(fail=True)
    h.consumer._coordinator.coordinator_id = 0
    h.consumer._end_offsets = [100]
    for _ in range(4):
        await h.poll(125)
    assert h.recreates >= 1
    assert len(h.supervisor.alarms) == 1
    assert h.supervisor.alarms[0].reason == "stalled_backlog_not_draining"
    assert "coordinator is unavailable" not in h.supervisor.alarms[0].message
    h.consumer._end_offsets = [10]
    await h.poll()
    assert not h.supervisor.alarm_active


@pytest.mark.asyncio
async def test_alarm_emitter_failure_never_breaks_recovery() -> None:
    async def emit(alarm: ModelConsumerRejoinAlarm) -> None:
        raise ConnectionError("health pipeline unavailable")

    h = Harness(fail=True, emit_alarm=emit)
    for _ in range(15):
        await h.poll()
    assert len(h.supervisor.alarms) == 1
    h.consumer._coordinator.coordinator_id = 0
    h.consumer._end_offsets = [10]
    await h.poll(125)
    assert not h.supervisor.alarm_active


@pytest.mark.asyncio
async def test_bus_wires_bounds_identity_and_retains_bounded_alarms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
        REJOIN_HISTORY_CAPACITY,
    )
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
        ModelKafkaEventBusConfig,
    )
    from omnibase_infra.models.health.enum_consumer_health_event_type import (
        EnumConsumerHealthEventType,
    )
    from omnibase_infra.models.health.enum_consumer_health_severity import (
        EnumConsumerHealthSeverity,
    )

    config = ModelKafkaEventBusConfig(
        bootstrap_servers="broker:9092",
        environment="test",
        consumer_coordinator_unavailable_seconds=20,
        consumer_coordinator_rejoin_cooldown_seconds=30,
        consumer_stall_probe_timeout_seconds=0.02,
    )
    bus = EventBusKafka(config=config)
    clock = FakeClock()
    monkeypatch.setattr(
        bus,
        "_rebuild_consumer_for_rejoin",
        AsyncMock(side_effect=ConnectionError("rebuild failed")),
    )
    supervisor = bus._rejoin_supervisor_for(TOPIC, GROUP)
    supervisor._clock = clock
    consumer = FakeConsumer(
        member_id="bus-original", position=10, end_offsets=(10,), wedged=True
    )
    for _ in range(5):
        clock.advance(5)
        await asyncio.wait_for(supervisor.next_batch(consumer), timeout=5)
    assert len(bus.consumer_rejoin_alarms()) == 1
    alarm = bus.consumer_rejoin_alarms()[0]
    assert alarm.consumer_group == bus._resolve_effective_group_id(
        GROUP, TOPIC, uuid4(), (TOPIC, GROUP)
    )
    assert alarm.broker == "broker:9092"
    assert alarm.coordinator_unavailable_seconds == 20
    assert supervisor._probe_timeout_seconds == 0.02
    assert supervisor._coordinator_rejoin_cooldown_seconds == 30
    assert supervisor.alarms == bus.consumer_rejoin_alarms()
    consumer._wedged = False
    consumer._pending = [object()]
    await asyncio.wait_for(supervisor.next_batch(consumer), timeout=5)
    emitter = AsyncMock()
    bus._health_emitter = emitter
    await bus._record_consumer_rejoin_alarm(alarm)
    assert (
        emitter.emit_event.await_args.kwargs["event_type"]
        is EnumConsumerHealthEventType.CONSUMER_GROUP_STALLED
    )
    assert (
        emitter.emit_event.await_args.kwargs["severity"]
        is EnumConsumerHealthSeverity.CRITICAL
    )
    emitter.emit_event.side_effect = ConnectionError("mirror failed")
    for _ in range(REJOIN_HISTORY_CAPACITY + 1):
        await bus._record_consumer_rejoin_alarm(alarm)
    assert len(bus.consumer_rejoin_alarms()) == REJOIN_HISTORY_CAPACITY


@pytest.mark.asyncio
async def test_consumer_rejoins_after_broker_restart_within_five_minutes_at_shipped_bounds() -> (
    None
):
    """AC2's budget at the shipped config: one failed rebuild still rejoins inside 300 s."""
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(bootstrap_servers="broker:9092")
    h = Harness(
        fail=True,
        coordinator_unavailable_seconds=config.consumer_coordinator_unavailable_seconds,
        coordinator_rejoin_cooldown_seconds=(
            config.consumer_coordinator_rejoin_cooldown_seconds
        ),
        probe_timeout_seconds=config.consumer_stall_probe_timeout_seconds,
    )
    await h.poll(0)  # the broker answers again; the coordinator is still lost
    elapsed = 0.0
    poll_seconds = config.consumer_poll_timeout_ms / 1000
    while h.recreates < 1:
        await h.poll(poll_seconds)
        elapsed += poll_seconds
    h.fail = False  # the first rebuild failed; the broker's coordinator is back
    while h.consumer is not h.replacement:
        await h.poll(poll_seconds)
        elapsed += poll_seconds
    assert h.recreates == 2
    assert elapsed < 300.0
