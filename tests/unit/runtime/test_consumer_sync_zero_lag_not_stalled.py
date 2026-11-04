# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A quiet local stack is not a wedge (OMN-22291).

**The incident.** On 2026-11-03T04:52Z, on a fresh macOS account running the
pinned participant onboarding, every component of the local stack reported
healthy and the runtime container still failed its healthcheck 217 times in a
row. The whole of the verdict was one dimension:

    3 of 488 consumer group(s) out of sync:
    local.runtime_config.node_savings_estimation_compute.consume.v1
    lag=0 stalled=8021s last_advance=239s reason=not_stalled_within_cooldown; ...

So `make up-local` failed on the compose dependency, phase 4 of the onboarding
failed with it, and OMN-21261 AC1 could not be met by anyone who answered yes
to the Docker question.

**What the measured cause turned out to be, which is not what the detail says.**
``lag=0`` there is not "caught up". The three groups held NO partition
assignment -- the same payload's log lines show a clean join at generation 11
followed by ``Setting newly assigned partitions set()`` -- and the
no-assignment branch of the observation reported a placeholder zero backlog,
because a member with no assignment has no fetch position to measure against.
The classifier then correctly refused to call that idle, the stall clock ran
from the first such evaluation and never cleared, and after the readiness
window the dimension went CRITICAL and stayed there. On a laptop this is the
normal case rather than a fault: the local stack's partition cap leaves one
partition per topic, so every group with more members than partitions has
standby members by construction, assigned exactly as Kafka intends.

**What tells a standby from the outage this class exists for.** OMN-18640's
``Empty``-group wedge has records waiting with nobody taking them. A standby
has nothing waiting. The difference is the GROUP's backlog over the whole
topic -- committed offsets from the coordinator against the leaders' end
offsets -- which a member holding no assignment can still ask for, and which
the fix measures before it classifies. A backlog that could not be measured
stays unknown and still falls toward the remedy, so the zero this incident
turned on can never again read as health.

The fakes are the ones OMN-18640 built, reused deliberately: a readiness
surface that agrees with the recovery path only because it was written against
a different fake is two opinions, not one fact.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from typing import Any

import pytest

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
from omnibase_infra.models.health.model_consumer_poll_observation import (
    ModelConsumerPollObservation,
)
from omnibase_infra.models.health.model_consumer_sync_status import (
    ModelConsumerSyncStatus,
)
from omnibase_infra.protocols.protocol_rejoinable_consumer import (
    ProtocolRejoinableConsumer,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    describe_consumer_sync,
    evaluate_consumer_sync,
)
from tests.unit.event_bus.test_consumer_rejoin_on_coordinator_loss import (
    FakeClock,
    FakeConsumer,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    pathlib.Path(__file__).parent / "fixtures" / "omn22291_consumer_sync_critical.json"
)

# The recorded payload names the groups, not the topics they subscribe to, so
# the replay declares one of its own. No verdict in this dimension reads it.
TOPIC = "onex.evt.runtime-config.updated.v1"  # onex-topic-allow: OMN-22291 replay

SYNC_UNREADY_SECONDS = 600.0
POLICY = ModelConsumerRejoinPolicy(
    stall_seconds=120.0,
    required_consecutive_stalls=3,
    rejoin_cooldown_seconds=300.0,
    sync_unready_seconds=SYNC_UNREADY_SECONDS,
)

#: Where the three groups' offsets sat. Any value does: what matters is that
#: the group's committed offset equals the leaders' end offset, which is what
#: "nothing is waiting on this topic" means.
QUIET_OFFSET = 4_210


class _SyncSource:
    """The one method the health dimension reads off the event bus."""

    def __init__(self, statuses: tuple[ModelConsumerSyncStatus, ...]) -> None:
        self._statuses = statuses

    def consumer_sync_statuses(self) -> tuple[ModelConsumerSyncStatus, ...]:
        return self._statuses


def _standby_consumer(group: str) -> FakeConsumer:
    """A member of ``group`` that holds no assignment and has nothing waiting.

    ``wedged=True`` only means "delivers no records", which is true of every
    standby: the records it is not delivering are records nobody published.
    """
    return FakeConsumer(
        member_id=f"standby::{group}",
        position=QUIET_OFFSET,
        end_offsets=(QUIET_OFFSET,),
        wedged=True,
        assigned=False,
        topic_partitions=(0,),
        group_committed=QUIET_OFFSET,
    )


def _supervisor(group: str, clock: FakeClock) -> ConsumerRejoinSupervisor:
    # Typed against the protocol, not the fake: the fake satisfies it
    # structurally, which is the whole reason OMN-18640 declared a protocol
    # instead of mocking a consumer.
    async def _recreate() -> ProtocolRejoinableConsumer:
        return _standby_consumer(group)

    return ConsumerRejoinSupervisor(
        topic=TOPIC,
        group_id=group,
        policy=POLICY,
        poll_timeout_ms=5_000,
        recreate_consumer=_recreate,
        clock=clock,
    )


async def _replay(group: str, *, quiet_seconds: float) -> ModelConsumerSyncStatus:
    """Run one standby group through ``quiet_seconds`` of the incident."""
    clock = FakeClock()
    supervisor = _supervisor(group, clock)
    first = _standby_consumer(group)
    consumer: ProtocolRejoinableConsumer = first
    elapsed = 0.0
    while elapsed < quiet_seconds:
        clock.advance(50.0)
        elapsed += 50.0
        batch = await supervisor.next_batch(consumer)
        consumer = batch.consumer
    assert not first.stopped, (
        "a standby member holding no assignment with nothing waiting must "
        "never be recreated: the replacement is handed the same nothing"
    )
    return supervisor.sync_status()


def _recorded_status(group: Mapping[str, Any]) -> ModelConsumerSyncStatus:
    """The incident's own numbers, as the payload stated them.

    Deliberately built from the recorded cells rather than from a replay: this
    is the shape the container healthcheck read, and it must still grade
    CRITICAL, or the fixture is no longer reproducing the incident.
    """
    return ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group=str(group["consumer_group"]),
        evaluated=True,
        reason=EnumConsumerStallReason(str(group["reason"])),
        backlog_records=int(group["lag"]),
        seconds_since_last_record=float(group["last_advance_seconds"]),
        stalled_seconds=float(group["stalled_seconds"]),
        assigned_partitions=0,
        broker_reachable=True,
        rejoin_count=0,
        last_rejoin_failed=False,
        unready_after_seconds=SYNC_UNREADY_SECONDS,
    )


# --- AC1 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_lag_recent_advance_is_never_counted_out_of_sync() -> None:
    """AC1. Advance inside the window is in sync, and a standby stays in sync.

    Two halves, because the detail's ``lag=0`` had two different meanings and
    only one of them was ever health. A consumer that delivered a record
    inside the stall window is working, which no reading may override. And the
    incident's own shape -- no assignment, nothing waiting -- must stay in
    sync however long it is quiet, which is what failed.
    """
    inside_the_window = evaluate_consumer_stall(
        ModelConsumerPollObservation(
            topic=TOPIC,
            consumer_group="local.runtime_config.quiet.consume.v1",
            seconds_since_last_record=POLICY.stall_seconds - 1.0,
            assigned_partitions=1,
            broker_reachable=True,
            backlog_records=0,
        ),
        POLICY,
        prior_consecutive_stalls=0,
    )
    assert inside_the_window.reason is (EnumConsumerStallReason.NOT_STALLED_PROGRESSING)
    assert inside_the_window.consecutive_stalls == 0

    # The incident ran 8021 seconds, so replay past it.
    status = await _replay(
        "local.runtime_config.node_savings_estimation_compute.consume.v1",
        quiet_seconds=9_000.0,
    )
    assert status.ready, (
        "a standby member with no assignment and nothing waiting on its topic "
        "is health; it is what every group beyond the partition count looks "
        "like on a laptop"
    )
    assert status.stalled_seconds == 0.0, (
        "the stall clock must not run for a standby: it never cleared in the "
        "incident, which is how 8021s accumulated on a quiet stack"
    )
    # Pinned to NOT_STALLED_IDLE rather than merely away from
    # NOT_STALLED_WITHIN_COOLDOWN: the cooldown reason the incident reported is
    # now reachable only from a backlog that is not draining, so naming the one
    # right answer is the stronger statement.
    assert status.reason is EnumConsumerStallReason.NOT_STALLED_IDLE
    assert status.backlog_records == 0


# --- AC2 -------------------------------------------------------------------


def test_the_detail_names_exclusions() -> None:
    """AC2. A group held inside the window is named, with the window.

    Before this, a group carrying a stall that the window was still covering
    was simply absent from the detail, which reads identically to a group
    nobody measured. That is the exact confusion this dimension was built to
    end, and it is what let a CRITICAL verdict rest on three groups whose own
    reason string cleared them.
    """
    held = ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group="local.runtime_config.held.consume.v1",
        evaluated=True,
        reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
        backlog_records=12,
        seconds_since_last_record=200.0,
        stalled_seconds=180.0,
        assigned_partitions=1,
        broker_reachable=True,
        rejoin_count=1,
        last_rejoin_failed=False,
        unready_after_seconds=SYNC_UNREADY_SECONDS,
    )
    quiet = ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group="local.runtime_config.quiet.consume.v1",
        evaluated=True,
        reason=EnumConsumerStallReason.NOT_STALLED_IDLE,
        backlog_records=0,
        seconds_since_last_record=240.0,
        stalled_seconds=0.0,
        assigned_partitions=1,
        broker_reachable=True,
        rejoin_count=0,
        last_rejoin_failed=False,
        unready_after_seconds=SYNC_UNREADY_SECONDS,
    )

    detail = describe_consumer_sync((held, quiet))
    assert "2 consumer group(s) in sync" in detail
    assert "local.runtime_config.held.consume.v1" in detail, (
        "a group carrying a stall the window is covering must be named, or an "
        "absent group and a measured-green group read the same"
    )
    assert "stalled=180s of 600s window" in detail, (
        "the exclusion states the number it was judged against, so the "
        "verdict is self-describing"
    )
    assert "reason=stalled_backlog_not_draining" in detail
    assert "local.runtime_config.quiet.consume.v1" not in detail, (
        "a group carrying no stall at all was not excluded from anything"
    )

    # The same sentence rides on a failing verdict, where it is what says
    # which groups the CRITICAL does NOT rest on.
    wedged = held.model_copy(update={"stalled_seconds": 900.0})
    failing = describe_consumer_sync((wedged, held, quiet))
    assert "1 of 3 consumer group(s) out of sync" in failing
    assert "carrying a stall held inside the window" in failing


# --- AC3 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiet_local_stack_not_critical_on_the_healthcheck_payload() -> None:
    """AC3. The three incident groups, replayed, grade the dimension non-CRITICAL.

    Graded through ``evaluate_consumer_sync``, which is the function the
    runtime_health verdict and therefore the container healthcheck read, so
    this is the same path that failed 217 times.
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    statuses = tuple(
        [
            await _replay(str(group["consumer_group"]), quiet_seconds=9_000.0)
            for group in payload["groups"]
        ]
    )
    status, detail = evaluate_consumer_sync(_SyncSource(statuses))
    assert status != "CRITICAL", (
        "a local stack with no traffic on these topics must not fail the "
        "container healthcheck: compose makes every dependent wait on it"
    )
    assert status == "HEALTHY"
    assert "3 consumer group(s) in sync" in detail
    assert "out of sync" not in detail


# --- AC4 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_incident_payload_fixture_grades_critical_only_as_recorded() -> None:
    """AC4. The recorded payload still grades CRITICAL; the replay does not.

    The first half is the control: it holds the incident's own numbers, so if
    it ever stops grading CRITICAL the fixture has stopped reproducing the
    incident and the second half proves nothing. The second half runs the same
    three groups through the code under test from the conditions the payload
    recorded, and is what the fix changes.
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["consumer_sync"]["status"] == "CRITICAL"
    assert payload["failing_streak"] == 217
    assert payload["observed"]["assigned_partitions"] == 0
    assert payload["observed"]["group_backlog_records"] == 0

    as_recorded = tuple(_recorded_status(group) for group in payload["groups"])
    recorded_status, recorded_detail = evaluate_consumer_sync(_SyncSource(as_recorded))
    assert recorded_status == "CRITICAL"
    assert "3 of 3 consumer group(s) out of sync" in recorded_detail
    for group in payload["groups"]:
        assert str(group["consumer_group"]) in recorded_detail

    replayed = tuple(
        [
            await _replay(str(group["consumer_group"]), quiet_seconds=9_000.0)
            for group in payload["groups"]
        ]
    )
    replayed_status, _ = evaluate_consumer_sync(_SyncSource(replayed))
    assert replayed_status != "CRITICAL", (
        "replayed from the conditions the payload recorded -- no assignment, "
        "nothing waiting on the topic -- the same three groups must not fail "
        "the healthcheck"
    )
    assert all(status.ready for status in replayed)
    assert all(status.stalled_seconds == 0.0 for status in replayed)
