# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's readiness reports a stalled consumer group (OMN-18640 AC1).

**The blind spot these tests close.** At 2026-09-19T04:51:48Z the `.201` dev
lane broker was recreated, the effects consumer wedged with its offsets frozen
at 7540, and it stayed wedged for about fifty minutes until a full-lane runtime
recreate at 05:41:22Z. The deploy agent's force-recreate backstop (AC7,
`omnibase_infra#3802`) never fired, because the one thing it probes --
``omninode-runtime-effects``'s own container health -- was green for the whole
window. ``omnibase_infra#3803`` put a consumer-group-sync probe on the BROKER's
healthcheck, which is a different container; ``omnibase_infra#3807`` makes the
consumer rebuild itself, which covers most cases once deployed but is itself a
thing that can fail. None of the three makes the RUNTIME say it is unwell.

So these tests are about one predicate: the supervisor that already measures
the wedge must also be able to answer "is this group in sync?", using the same
evidence -- fetch position against the partition leaders' end offsets, not
advancing -- and the answer must reach the readiness dimension the container
healthcheck already reads.

The fakes are the ones ``omnibase_infra#3807`` added, reused deliberately: a
readiness surface that agrees with the recovery path only because it was
written against a different fake would be two opinions, not one fact.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.consumer_rejoin_policy import ModelConsumerRejoinPolicy
from omnibase_infra.event_bus.consumer_rejoin_supervisor import (
    ConsumerRejoinSupervisor,
)
from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from tests.unit.event_bus.test_consumer_rejoin_on_coordinator_loss import (
    GROUP,
    TOPIC,
    WEDGED_END_OFFSETS,
    WEDGED_POSITION,
    FakeClock,
    FakeConsumer,
)

pytestmark = pytest.mark.unit

# The readiness window is deliberately LONGER than the recovery bounds it sits
# behind: stall (120s) + one cooldown (300s) is one failed self-heal and its
# retry. Readiness that flipped sooner would race the consumer's own rejoin and
# the deploy agent's force-recreate against each other over a wedge that the
# cheapest remedy was still in the middle of fixing.
SYNC_UNREADY_SECONDS = 600.0

POLICY = ModelConsumerRejoinPolicy(
    stall_seconds=120.0,
    required_consecutive_stalls=3,
    rejoin_cooldown_seconds=300.0,
    sync_unready_seconds=SYNC_UNREADY_SECONDS,
)


def _supervisor(
    clock: FakeClock,
    *,
    replacement: FakeConsumer | None = None,
    recreate_fails: bool = False,
) -> ConsumerRejoinSupervisor:
    async def _recreate() -> FakeConsumer:
        if recreate_fails:
            raise ConnectionError("replacement consumer could not start")
        return replacement or FakeConsumer(
            member_id="replacement",
            position=WEDGED_POSITION,
            end_offsets=WEDGED_END_OFFSETS,
            wedged=False,
        )

    return ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=GROUP,
        policy=POLICY,
        poll_timeout_ms=5_000,
        # Why: the fake implements the protocol structurally; mypy sees the
        # concrete return type and the protocol is runtime-structural.
        recreate_consumer=_recreate,  # type: ignore[arg-type]
        clock=clock,
    )


# --- the three the brief names ---------------------------------------------


@pytest.mark.asyncio
async def test_a_frozen_consumer_is_reported_out_of_sync() -> None:
    """The recorded wedge: position pinned, end offsets walking, group NOT READY.

    This is the whole point of the ticket. Every surface that existed during
    the outage said this consumer was fine.
    """
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="wedged-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    # The replacement is wedged too: this is the case the readiness dimension
    # exists for. The self-heal runs, and the group is still not consuming.
    replacement = FakeConsumer(
        member_id="replacement-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    supervisor = _supervisor(clock, replacement=replacement)

    consumer: FakeConsumer = wedged
    # Fifty minutes of the measured wedge, polled at the declared cadence.
    for _ in range(60):
        clock.advance(50.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]

    status = supervisor.sync_status()
    assert not status.ready, (
        "a consumer whose fetch position has been behind the leaders and not "
        "advancing for longer than the declared window must report NOT READY"
    )
    assert status.stalled_seconds >= SYNC_UNREADY_SECONDS
    assert status.backlog_records > 0, (
        "the measured lag is the dimension's evidence; a NOT READY with no "
        "number is an assertion, not a measurement"
    )
    assert status.topic == TOPIC
    assert status.consumer_group == GROUP

    # The load-bearing reason the stall clock is its own field. A successful
    # rejoin resets _last_record_at, so seconds-since-last-record restarts
    # from zero on a recovery that changed nothing. A readiness surface built
    # on that number would go green here.
    assert wedged.stopped, "the wedge must have ordered at least one rejoin"
    assert status.seconds_since_last_record < status.stalled_seconds, (
        "the rejoin reset the last-record clock while the stall clock "
        "survived -- which is the whole reason the two are separate"
    )


@pytest.mark.asyncio
async def test_a_draining_consumer_is_reported_in_sync() -> None:
    """A consumer that is behind but advancing is READY: it is working."""
    clock = FakeClock()
    draining = FakeConsumer(
        member_id="draining-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=False,
        pending_records=["record"],
    )
    supervisor = _supervisor(clock)

    consumer: FakeConsumer = draining
    for _ in range(60):
        clock.advance(50.0)
        # Refill so every poll delivers: this is a consumer keeping up.
        consumer._pending = ["record"]
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]

    status = supervisor.sync_status()
    assert status.ready
    assert status.stalled_seconds == 0.0
    assert status.reason is EnumConsumerStallReason.NOT_STALLED_PROGRESSING
    assert not consumer.stopped, "a draining consumer must never be recreated"


@pytest.mark.asyncio
async def test_a_fresh_group_with_no_traffic_is_ready_not_stalled() -> None:
    """Positive control. An idle topic is the common case and must stay green.

    Three hours of silence on a caught-up consumer. If this one ever goes red
    the dimension is a pager that fires on a quiet Sunday, and it will be
    turned off -- which is how the surface that should have caught the outage
    comes not to exist.
    """
    clock = FakeClock()
    idle = FakeConsumer(
        member_id="idle-1",
        position=7_489,
        end_offsets=(7_489,),
        wedged=True,  # no records, but caught up
    )
    supervisor = _supervisor(clock)

    consumer: FakeConsumer = idle
    for _ in range(120):
        clock.advance(90.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]

    status = supervisor.sync_status()
    assert status.ready, "an idle caught-up consumer is health, not a wedge"
    assert status.reason is EnumConsumerStallReason.NOT_STALLED_IDLE
    assert status.backlog_records == 0
    assert status.stalled_seconds == 0.0
    assert not consumer.stopped


def test_a_group_that_has_never_been_evaluated_is_ready() -> None:
    """Boot. No poll has completed, so there is no measurement to be red about.

    Reported as ``evaluated=False`` rather than as a green measurement, so a
    reader can tell "proven in sync" from "nothing known yet".
    """
    supervisor = _supervisor(FakeClock())
    status = supervisor.sync_status()
    assert status.ready
    assert not status.evaluated
    assert status.stalled_seconds == 0.0


# --- the two the cooldown and the failed rejoin make necessary --------------


@pytest.mark.asyncio
async def test_a_stalled_group_inside_the_rejoin_cooldown_is_still_out_of_sync() -> (
    None
):
    """The cooldown bounds the REMEDY. It does not make the group healthy.

    ``evaluate_consumer_stall`` reports ``NOT_STALLED_WITHIN_COOLDOWN`` for a
    group that still carries the full stall signature, because from the
    rejoin path's point of view the only question is "may I act yet?".
    Readiness asks a different question and must not inherit that answer, or a
    consumer that wedges, rejoins, and wedges again reads green for the whole
    of every cooldown.
    """
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="wedged-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    replacement = FakeConsumer(
        member_id="replacement-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,  # the rejoin did not fix it
    )
    supervisor = _supervisor(clock, replacement=replacement)

    consumer: FakeConsumer = wedged
    inside_cooldown: list[float] = []
    for _ in range(60):
        clock.advance(50.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]
        status = supervisor.sync_status()
        if status.reason is EnumConsumerStallReason.NOT_STALLED_WITHIN_COOLDOWN:
            inside_cooldown.append(status.stalled_seconds)

    assert wedged.stopped, "the recorded wedge must still order its rejoin"
    assert inside_cooldown, (
        "the replacement wedged too, so the run must have passed through the "
        "cooldown at least once -- otherwise this test proves nothing"
    )
    assert all(stalled > 0.0 for stalled in inside_cooldown), (
        "a group inside its rejoin cooldown is still behind and still not "
        "advancing; the cooldown is a bound on the remedy, not a verdict, and "
        "the stall clock must not be reset by passing through it"
    )
    assert not supervisor.sync_status().ready


@pytest.mark.asyncio
async def test_a_failed_rejoin_leaves_the_group_out_of_sync() -> None:
    """A recovery that could not run is the case AC7 exists to back up.

    Recorded distinctly from the stall itself: a rejoin that raised is a fact
    about the remedy, and a readiness surface that only reported the stall
    would go green the moment the stall window was reset by a rejoin that did
    not actually happen.
    """
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="wedged-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    supervisor = _supervisor(clock, recreate_fails=True)

    consumer: FakeConsumer = wedged
    for _ in range(12):
        clock.advance(50.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]

    status = supervisor.sync_status()
    assert status.last_rejoin_failed, (
        "the supervisor attempted a rejoin and it raised; that must be visible"
    )
    assert not status.ready
    assert status.rejoin_count >= 1


@pytest.mark.asyncio
async def test_flow_resuming_clears_a_failed_rejoin() -> None:
    """Resolved means records moved again. Nothing else counts as resolved.

    Without this the dimension latches: one failed rejoin would hold the
    runtime unhealthy forever even after the fault cleared, which is the same
    class of lie as reporting green through a wedge, pointed the other way.
    """
    clock = FakeClock()
    wedged = FakeConsumer(
        member_id="wedged-1",
        position=WEDGED_POSITION,
        end_offsets=WEDGED_END_OFFSETS,
        wedged=True,
    )
    supervisor = _supervisor(clock, recreate_fails=True)

    consumer: FakeConsumer = wedged
    for _ in range(12):
        clock.advance(50.0)
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer  # type: ignore[assignment]
    assert not supervisor.sync_status().ready

    # The broker comes back and the same client starts fetching again.
    consumer._wedged = False
    consumer._pending = ["record"]
    clock.advance(1.0)
    await supervisor.next_batch(consumer)

    status = supervisor.sync_status()
    assert status.ready
    assert not status.last_rejoin_failed
    assert status.stalled_seconds == 0.0


# --- the declared configuration --------------------------------------------


def test_the_readiness_window_is_a_declared_default_with_no_env_fallback() -> None:
    """Rule 8. A readiness bound an environment variable can move is one that
    will be moved on the host where the truth is inconvenient."""
    import inspect

    from omnibase_infra.event_bus.models.config import (
        model_kafka_event_bus_config as config_module,
    )

    config = config_module.ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    assert config.consumer_sync_unready_seconds == 600.0
    assert config.consumer_sync_unready_seconds > (
        config.consumer_stall_seconds + config.consumer_rejoin_cooldown_seconds
    ), (
        "the readiness window must outlast one full self-heal attempt and its "
        "cooldown, or readiness and the rejoin path race over the same wedge"
    )

    source = inspect.getsource(config_module)
    start = source.index("consumer_poll_timeout_ms: int = Field(")
    end = source.index("# Kafka producer settings", start)
    declaration_block = source[start:end]
    assert "consumer_sync_unready_seconds" in declaration_block, (
        "the readiness bound must sit inside the block the no-env-fallback "
        "assertions below are taken over"
    )
    assert "os.environ" not in declaration_block
    assert "default_factory" not in declaration_block
