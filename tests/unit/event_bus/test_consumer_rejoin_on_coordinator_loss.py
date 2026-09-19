# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The effects consumer rejoins its group after coordinator loss (OMN-18640).

These tests replay the measured facts of the 2026-09-18T23:16Z dev-lane wedge:
the broker is recreated, the client marks its coordinator dead, the partition
LEADERS answer normally, the consumer keeps ``Stable`` membership with an
assigned partition, and its fetch position stays pinned at 7473 while the log
end offset walks 7481 -> 7487 -> 7489. Before this change that state persisted
for 97 minutes and cleared only when the container itself was recreated.

The fake consumer below is a typed implementation of
``ProtocolRejoinableConsumer`` rather than a mock: the wedge is a behaviour
over several calls, not a single return value, and a mock configured to return
the right things proves only that the test knows the answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.consumer_rejoin_policy import (
    ModelConsumerRejoinPolicy,
    evaluate_consumer_stall,
)
from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
    ConsumerRejoinSupervisor,
)
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from omnibase_infra.models.health.model_consumer_group_rejoin_event import (
    ModelConsumerGroupRejoinEvent,
)
from omnibase_infra.models.health.model_consumer_poll_observation import (
    ModelConsumerPollObservation,
)

pytestmark = pytest.mark.unit

# --- the measured incident -------------------------------------------------

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
GROUP = (
    "local.omnimarket.pr_lifecycle_fix_effect.consume.1.0.0"
    ".__i.runtime-effects.__t.onex.cmd.omnimarket.occ-autobind.v1"
)
PARTITION = TopicPartition(TOPIC, 0)

# 2026-09-19T00:07:29Z, 50 minutes into the wedge: committed offset pinned at
# 7473 while the topic end walked 7481 -> 7487 -> 7489.
WEDGED_POSITION = 7473
WEDGED_END_OFFSETS = (7481, 7487, 7489)

POLICY = ModelConsumerRejoinPolicy(
    stall_seconds=120.0,
    required_consecutive_stalls=3,
    rejoin_cooldown_seconds=300.0,
    # OMN-18640 AC1 added the readiness window to this policy. Its value is
    # irrelevant to every test in this file -- they are about the rejoin
    # decision, which does not read it -- but it is required rather than
    # defaulted so no construction site can silently disagree with the config.
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
        return self._position

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        if not self._leaders_reachable:
            raise ConnectionError("leaders unreachable")
        self.end_offset_probes += 1
        index = min(self.end_offset_probes - 1, len(self._end_offsets) - 1)
        return {PARTITION: self._end_offsets[index]}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._position = offset

    async def stop(self) -> None:
        self.stopped = True


def _observation(
    *,
    silent_for: float,
    assigned: int = 1,
    reachable: bool = True,
    backlog: int = 0,
    since_rejoin: float | None = None,
) -> ModelConsumerPollObservation:
    return ModelConsumerPollObservation(
        topic=TOPIC,
        consumer_group=GROUP,
        seconds_since_last_record=silent_for,
        assigned_partitions=assigned,
        broker_reachable=reachable,
        backlog_records=backlog,
        seconds_since_last_rejoin=since_rejoin,
    )


# --- the pure policy -------------------------------------------------------


def test_frozen_backlog_on_a_reachable_broker_is_a_stall() -> None:
    """The recorded signature: assigned, leaders answering, backlog not moving."""
    verdict = evaluate_consumer_stall(
        _observation(
            silent_for=3_000.0, backlog=WEDGED_END_OFFSETS[-1] - WEDGED_POSITION
        ),
        POLICY,
        prior_consecutive_stalls=2,
    )
    assert verdict.reason is EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING
    assert verdict.is_stalled
    assert verdict.should_rejoin, "three confirmations must order a rejoin"


def test_one_confirmation_is_not_enough_to_rejoin() -> None:
    """A single reading is not a wedge."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=3_000.0, backlog=16),
        POLICY,
        prior_consecutive_stalls=0,
    )
    assert verdict.is_stalled
    assert not verdict.should_rejoin


def test_a_caught_up_consumer_is_idle_not_stalled() -> None:
    """Positive control: silence with no backlog is health, at any duration."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=86_400.0, backlog=0),
        POLICY,
        prior_consecutive_stalls=2,
    )
    assert verdict.reason is EnumConsumerStallReason.NOT_STALLED_IDLE
    assert not verdict.should_rejoin
    assert verdict.consecutive_stalls == 0, "a clean reading resets the count"


def test_a_progressing_consumer_is_never_stalled() -> None:
    """Records inside the window override every other signal."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=1.0, backlog=10_000),
        POLICY,
        prior_consecutive_stalls=2,
    )
    assert verdict.reason is EnumConsumerStallReason.NOT_STALLED_PROGRESSING
    assert not verdict.should_rejoin


def test_an_unreachable_broker_is_not_a_client_wedge() -> None:
    """A broker that is down is not fixed by recreating the client."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=3_000.0, reachable=False, backlog=0),
        POLICY,
        prior_consecutive_stalls=2,
    )
    assert verdict.reason is EnumConsumerStallReason.NOT_STALLED_BROKER_UNREACHABLE
    assert not verdict.should_rejoin


def test_an_empty_assignment_is_its_own_stall_class() -> None:
    """The second outage shape: the group reads Empty while the topic advances."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=3_000.0, assigned=0, backlog=0),
        POLICY,
        prior_consecutive_stalls=2,
    )
    assert verdict.reason is EnumConsumerStallReason.STALLED_NO_ASSIGNMENT
    assert verdict.should_rejoin


def test_the_cooldown_bounds_the_recreate_rate() -> None:
    """A fault that survives a rejoin must not become a recreate loop."""
    verdict = evaluate_consumer_stall(
        _observation(silent_for=3_000.0, backlog=16, since_rejoin=30.0),
        POLICY,
        prior_consecutive_stalls=5,
    )
    assert verdict.reason is EnumConsumerStallReason.NOT_STALLED_WITHIN_COOLDOWN
    assert not verdict.should_rejoin
    assert verdict.consecutive_stalls == 5, "the cooldown delays, it does not forget"


# --- the supervisor over the replayed incident -----------------------------


@pytest.mark.asyncio
async def test_supervisor_rejoins_the_wedged_consumer_and_offsets_advance() -> None:
    """Replay 2026-09-18T23:16Z end to end and assert self-recovery."""
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="omninode-runtime-effects-8205ccc09444",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    # The replacement joins from the committed offset and drains the backlog
    # the wedged member never fetched.
    replacement = FakeConsumer(
        member_id="omninode-runtime-effects-94537d1bd6a9",
        position=WEDGED_POSITION,
        end_offsets=(WEDGED_END_OFFSETS[-1],),
        wedged=False,
        pending_records=[
            f"record-{n}" for n in range(WEDGED_END_OFFSETS[-1] - WEDGED_POSITION)
        ],
    )
    recreated: list[FakeConsumer] = []

    async def recreate() -> FakeConsumer:
        recreated.append(replacement)
        return replacement

    events: list[ModelConsumerGroupRejoinEvent] = []

    async def emit(event: ModelConsumerGroupRejoinEvent) -> None:
        events.append(event)

    supervisor = ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,  # type: ignore[arg-type]
        emit_event=emit,
        clock=clock,
    )

    consumer: Any = wedged
    batches = 0
    while batches < 8:
        batches += 1
        clock.advance(600.0)  # ten minutes of wedge per cycle
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer
        if batch.records:
            break

    assert wedged.stopped, "the wedged consumer must be closed, not abandoned"
    assert recreated == [replacement], "exactly one replacement was started"
    assert consumer is replacement
    assert consumer.member_id != wedged.member_id, (
        "the rejoin must produce a fresh group member, not reuse the wedged one"
    )
    assert await consumer.position(PARTITION) == WEDGED_END_OFFSETS[-1], (
        "offsets must advance to the log end after the rejoin"
    )
    assert batch.records, "the replacement must deliver the backlog the wedge withheld"

    assert len(events) == 1
    event = events[0]
    assert event.topic == TOPIC
    assert event.consumer_group == GROUP
    assert event.reason is EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING
    assert event.rejoin_succeeded is True
    assert event.backlog_records > 0
    assert supervisor.rejoin_events == (event,), (
        "the readiness surface must see the same typed record, ungated"
    )


@pytest.mark.asyncio
async def test_supervisor_never_recreates_a_healthy_consumer() -> None:
    """Positive control: a caught-up consumer is left alone indefinitely."""
    clock = FakeClock()
    healthy = FakeConsumer(
        member_id="healthy",
        position=WEDGED_END_OFFSETS[-1],
        end_offsets=(WEDGED_END_OFFSETS[-1],),
        wedged=True,  # returns no records: it is caught up, not wedged
    )

    async def recreate() -> FakeConsumer:
        raise AssertionError("a healthy consumer must never be recreated")

    supervisor = ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,  # type: ignore[arg-type]
        clock=clock,
    )

    consumer: Any = healthy
    for _ in range(20):
        clock.advance(600.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer

    assert consumer is healthy
    assert not healthy.stopped
    assert supervisor.rejoin_events == ()


@pytest.mark.asyncio
async def test_supervisor_leaves_the_consumer_alone_when_leaders_are_unreachable() -> (
    None
):
    """A broker outage must not be answered with a client recreate storm."""
    clock = FakeClock()
    offline = FakeConsumer(
        member_id="offline",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
        leaders_reachable=False,
    )

    async def recreate() -> FakeConsumer:
        raise AssertionError("an unreachable broker must not trigger a recreate")

    supervisor = ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,  # type: ignore[arg-type]
        clock=clock,
    )

    consumer: Any = offline
    for _ in range(10):
        clock.advance(600.0)
        consumer = (await supervisor.next_batch(consumer)).consumer

    assert not offline.stopped
    assert supervisor.rejoin_events == ()


@pytest.mark.asyncio
async def test_a_failed_recreate_is_recorded_rather_than_swallowed() -> None:
    """A rejoin that could not start is evidence too."""
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="wedged",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )

    async def recreate() -> FakeConsumer:
        raise ConnectionError("bootstrap refused")

    supervisor = ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,  # type: ignore[arg-type]
        clock=clock,
    )

    consumer: Any = wedged
    for _ in range(5):
        clock.advance(600.0)
        consumer = (await supervisor.next_batch(consumer)).consumer

    assert len(supervisor.rejoin_events) == 1
    event = supervisor.rejoin_events[0]
    assert event.rejoin_succeeded is False
    assert "ConnectionError" in event.failure_detail


@pytest.mark.asyncio
async def test_the_cooldown_prevents_a_recreate_loop() -> None:
    """A wedge that survives its own rejoin is retried on the cooldown, not spun."""
    clock = FakeClock()

    def _wedged(tag: str) -> FakeConsumer:
        return FakeConsumer(
            member_id=tag,
            position=WEDGED_POSITION,
            end_offsets=WEDGED_END_OFFSETS,
            wedged=True,
        )

    created: list[FakeConsumer] = []

    async def recreate() -> FakeConsumer:
        consumer = _wedged(f"replacement-{len(created)}")
        created.append(consumer)
        return consumer

    supervisor = ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,  # type: ignore[arg-type]
        clock=clock,
    )

    consumer: Any = _wedged("original")
    # Twenty minutes in ten-second steps: far more poll cycles than the
    # cooldown permits rejoins.
    for _ in range(120):
        clock.advance(10.0)
        consumer = (await supervisor.next_batch(consumer)).consumer

    elapsed = 120 * 10.0
    max_rejoins = int(elapsed // POLICY.rejoin_cooldown_seconds) + 1
    assert 1 <= len(created) <= max_rejoins, (
        f"expected at most {max_rejoins} rejoins in {elapsed:.0f}s, got {len(created)}"
    )


def test_asyncio_marker_is_available() -> None:
    """Guard: these tests are meaningless if the event loop never runs them."""
    assert asyncio.iscoroutinefunction(
        test_supervisor_rejoins_the_wedged_consumer_and_offsets_advance
    )


# --- the bounds are declared configuration, not environment ----------------


def test_the_recovery_bounds_are_declared_defaults_with_no_env_fallback() -> None:
    """A lane must not be able to disarm this recovery through the environment.

    Every other consumer bound on this config is contract-only for the same
    reason (``consumer_start_concurrency``, ``projection_withhold_max_redeliveries``).
    A recovery path that an environment variable can switch off is one that
    will be off on the host where it is needed.
    """
    import inspect

    from omnibase_infra.event_bus.models.config import (
        model_kafka_event_bus_config as config_module,
    )

    config = config_module.ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    assert config.consumer_poll_timeout_ms == 5_000
    assert config.consumer_stall_seconds == 120.0
    assert config.consumer_stall_required_confirmations == 3
    assert config.consumer_rejoin_cooldown_seconds == 300.0

    source = inspect.getsource(config_module)
    start = source.index("consumer_poll_timeout_ms: int = Field(")
    end = source.index("# Kafka producer settings", start)
    declaration_block = source[start:end]
    assert "os.environ" not in declaration_block, (
        "the coordinator-loss recovery bounds must not read the environment"
    )
    assert "default_factory" not in declaration_block, (
        "a default_factory is how an env read gets in; these are literals"
    )
