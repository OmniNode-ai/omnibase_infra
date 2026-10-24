# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A consumer survives a broker that is down while it rejoins (OMN-19513).

Measured on the dev lane: the broker ``omnibase-infra-redpanda`` segfaulted at
2026-10-23T05:27:35Z and restarted 8 times, then 11 by 07:47Z. The
``claude_hook_events`` projection consumer never delivered another record, and
its group held 0 members.

The path that leaves a group at 0 members: the rejoin supervisor (OMN-18640)
closes a wedged consumer FIRST and then builds a replacement. When the broker is
down at that moment the replacement cannot start, and ``_rejoin`` handed the
already-stopped handle back to the loop. aiokafka's ``getmany`` raises
``ConsumerStoppedError`` on a stopped consumer; ``EventBusKafka._consume_loop``
logs any exception and lets the task end. Nothing restarted the task, so the
consumer never rejoined and only recreating the container cleared it.

The fake consumer below behaves like aiokafka in the one respect that matters:
``getmany`` on a stopped consumer raises. The older fake in
``test_consumer_rejoin_on_coordinator_loss.py`` keeps polling quietly after
``stop()``, which is why a failed recreate looked harmless there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from aiokafka.errors import ConsumerStoppedError, KafkaConnectionError
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.consumer_rejoin_policy import ModelConsumerRejoinPolicy
from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
    ConsumerRejoinSupervisor,
)
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from omnibase_infra.models.health.model_consumer_stall_verdict import (
    ModelConsumerStallVerdict,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omniclaude.hook-event.v1"  # onex-topic-allow: replay of a recorded incident
GROUP = (
    "local.omnimarket.projection_claude_hook_events.consume.1.0.0"
    ".__i.runtime-main.__t.onex.evt.omniclaude.hook-event.v1"
)
PARTITION = TopicPartition(TOPIC, 0)

POLICY = ModelConsumerRejoinPolicy(
    stall_seconds=120.0,
    required_consecutive_stalls=3,
    rejoin_cooldown_seconds=300.0,
    sync_unready_seconds=600.0,
)


class FakeClock:
    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class AiokafkaLikeConsumer:
    """Typed ``ProtocolRejoinableConsumer`` whose stopped state is real.

    ``wedged`` keeps the assignment and a growing backlog, as the dev-lane
    consumer did when the broker went away. After ``stop()`` every ``getmany``
    raises ``ConsumerStoppedError``, as aiokafka's does.
    """

    def __init__(
        self,
        *,
        wedged: bool,
        pending: Sequence[Any] = (),
        poll_error: Exception | None = None,
    ) -> None:
        self.stopped = False
        self._wedged = wedged
        self._pending = list(pending)
        self._poll_error = poll_error
        self._position = 0

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> Mapping[TopicPartition, Sequence[Any]]:
        if self.stopped:
            raise ConsumerStoppedError
        if self._poll_error is not None:
            raise self._poll_error
        if self._wedged or not self._pending:
            return {}
        drained, self._pending = self._pending, []
        self._position += len(drained)
        return {PARTITION: drained}

    def assignment(self) -> set[TopicPartition]:
        return {PARTITION}

    async def position(self, partition: TopicPartition) -> int:
        return self._position

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {PARTITION: self._position + 40}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._position = offset

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """A virtual clock that also drives ``asyncio.sleep``.

    The supervisor paces lost-consumer retries with ``asyncio.sleep``. The
    fake advances the clock instead of spending real time, and still yields
    to the event loop once through the real ``sleep(0)``. Only tests that ask
    for ``clock`` get the patch.
    """
    fake_clock = FakeClock()
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float, result: Any = None) -> Any:
        fake_clock.advance(seconds)
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return fake_clock


def _supervisor(clock: FakeClock, recreate: Any) -> ConsumerRejoinSupervisor:
    return ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=recreate,
        clock=clock,
    )


@pytest.mark.asyncio
async def test_consumer_rejoins_after_the_broker_was_down_during_the_rejoin(
    clock: FakeClock,
) -> None:
    """Replay: wedged at the segfault, replacement refused twice, then served.

    Fails on today's code: the first failed recreate returns the stopped
    handle and the next poll raises ``ConsumerStoppedError`` out of
    ``next_batch``, which ends ``_consume_loop`` for good.
    """
    wedged = AiokafkaLikeConsumer(wedged=True)
    replacement = AiokafkaLikeConsumer(
        wedged=False, pending=[f"hook-event-{n}" for n in range(40)]
    )
    attempts: list[int] = []

    async def recreate() -> AiokafkaLikeConsumer:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 2:  # the broker is still restarting
            raise KafkaConnectionError("redpanda:9092 refused the connection")
        return replacement

    supervisor = _supervisor(clock, recreate)

    consumer: Any = wedged
    delivered = False
    for _ in range(40):
        clock.advance(60.0)
        batch = await supervisor.next_batch(consumer)  # must never raise
        consumer = batch.consumer
        if batch.records:
            delivered = True
            break

    assert delivered, "the consumer never rejoined; the feed stays down"
    assert consumer is replacement
    assert wedged.stopped
    assert attempts == [1, 2, 3], "recreate is retried until the broker is back"
    assert [e.rejoin_succeeded for e in supervisor.rejoin_events] == [
        False,
        False,
        True,
    ]
    assert supervisor.sync_status().ready, "flow resumed, so readiness recovers"


@pytest.mark.asyncio
async def test_a_consumer_that_raises_on_poll_is_replaced_not_abandoned(
    clock: FakeClock,
) -> None:
    """A poll that raises (a stopped or broken client) triggers a rejoin."""
    broken = AiokafkaLikeConsumer(wedged=False, poll_error=ConsumerStoppedError())
    replacement = AiokafkaLikeConsumer(wedged=False, pending=["hook-event-0"])

    async def recreate() -> AiokafkaLikeConsumer:
        return replacement

    supervisor = _supervisor(clock, recreate)

    batch = await supervisor.next_batch(broken)  # must never raise
    consumer: Any = batch.consumer
    for _ in range(5):
        if batch.records:
            break
        clock.advance(60.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer

    assert consumer is replacement
    assert batch.records
    assert supervisor.rejoin_events[0].rejoin_succeeded is True


@pytest.mark.asyncio
async def test_readiness_fails_while_no_replacement_can_be_started(
    clock: FakeClock,
) -> None:
    """The health surface goes unready, not green, while the consumer is lost."""
    wedged = AiokafkaLikeConsumer(wedged=True)

    async def recreate() -> AiokafkaLikeConsumer:
        raise KafkaConnectionError("redpanda:9092 refused the connection")

    supervisor = _supervisor(clock, recreate)

    consumer: Any = wedged
    for _ in range(12):
        clock.advance(60.0)
        consumer = (await supervisor.next_batch(consumer)).consumer

    status = supervisor.sync_status()
    assert status.last_rejoin_failed
    assert not status.ready
    # The fourth poll confirms the stall; each of the 8 later polls (60 s
    # apart, past LOST_CONSUMER_RETRY_SECONDS) retries the replacement once.
    events = supervisor.rejoin_events
    assert len(events) == 9
    assert all(event.rejoin_succeeded is False for event in events)
    assert all("KafkaConnectionError" in event.failure_detail for event in events)
    assert wedged.stopped


def test_a_failed_consumer_is_a_stall_signature() -> None:
    """``STALLED_CONSUMER_FAILED`` counts as stalled, so readiness can fail on it."""
    verdict = ModelConsumerStallVerdict(
        reason=EnumConsumerStallReason.STALLED_CONSUMER_FAILED,
        consecutive_stalls=POLICY.required_consecutive_stalls,
        required_consecutive_stalls=POLICY.required_consecutive_stalls,
    )
    assert verdict.is_stalled


@pytest.mark.asyncio
async def test_a_lost_consumer_rejoin_event_names_the_failed_reason(
    clock: FakeClock,
) -> None:
    """A lost consumer's rejoin event carries its own reason and both errors."""
    broken = AiokafkaLikeConsumer(wedged=False, poll_error=ConsumerStoppedError())

    async def recreate() -> AiokafkaLikeConsumer:
        raise KafkaConnectionError("redpanda:9092 refused the connection")

    supervisor = _supervisor(clock, recreate)
    await supervisor.next_batch(broken)

    (event,) = supervisor.rejoin_events
    assert event.rejoin_succeeded is False
    assert event.reason is EnumConsumerStallReason.STALLED_CONSUMER_FAILED
    assert "ConsumerStoppedError" in event.failure_detail
    assert "KafkaConnectionError" in event.failure_detail
