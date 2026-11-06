# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One-runtime plan T8 (OMN-21288): the ``last_ran`` handover rule for effects.

The writers' rule (``copy_if_uncommitted``) keeps a runtime group's own commit,
which is safe for a projection writer because it UPSERTs. An effects or worker
handler cannot take a repeat, and the TLA+ model in
``formal/consumer_group_handover`` (verdict SAFE) gives the rule it needs: every
cutover fences both sides and copies the commit of the side that LAST RAN onto
the destination, over whatever the destination holds; a rollback is the same
cutover back, so it carries the reverse handover.

``plan_offset_handover(handover_rule="last_ran")`` reads the side that last ran
as the later of the two commits. The simulation below drives the model's path
(fold, rollback, second fold, each with a crashed and retried procedure, events
published throughout) through the real planner from whichever side is starting,
and counts how many times each event's effect ran. The two controls are the
model's mutants: the writers' rule on the same path (mut_stale_trust) and a
rollback that copies nothing back (mut_no_reverse); each runs an effect twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pytest

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    hand_over_committed_offsets,
    plan_offset_handover,
)
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)
from tests.unit.handlers.test_consumer_group_handover_kafka_path_omn21753 import (
    RETIRED_GROUP,
    SUCCESSOR_GROUP,
    TOPIC,
    _broker,
    _kafka_bus,
    _new_bus,
    _tp,
)

pytestmark = pytest.mark.unit

Rule = Literal["copy_if_uncommitted", "last_ran"]
Side = Literal["old", "new"]


def _plan_one(
    *,
    successor: int | None,
    retired: int | None,
    rule: Rule,
    first: int = 0,
    end: int = 100,
    policy: Literal["earliest", "latest"] = "earliest",
) -> dict[int, int]:
    return plan_offset_handover(
        partitions=(0,),
        successor_committed={0: successor},
        retired_committed={0: retired},
        first_offsets={0: first},
        end_offsets={0: end},
        retired_auto_offset_reset=policy,
        handover_rule=rule,
    )


def test_last_ran_copies_a_later_retired_commit_over_the_successors() -> None:
    """A second fold: the runtime group's commit is from before the rollback."""
    assert _plan_one(successor=10, retired=25, rule="last_ran") == {0: 25}
    assert _plan_one(successor=10, retired=25, rule="copy_if_uncommitted") == {}


def test_last_ran_keeps_the_successors_commit_when_it_is_the_later() -> None:
    """A restart of the side that last ran changes nothing."""
    assert _plan_one(successor=30, retired=25, rule="last_ran") == {}
    assert _plan_one(successor=25, retired=25, rule="last_ran") == {}


@pytest.mark.parametrize(
    ("successor", "retired", "policy", "expected"),
    [
        (None, None, "earliest", {0: 5}),
        (None, None, "latest", {0: 100}),
        (3, None, "earliest", {0: 5}),  # successor below retention, no retired
        (3, 8, "earliest", {0: 8}),  # successor below retention, retired inside
        (None, 2, "latest", {0: 100}),  # retired below retention
        (None, 40, "latest", {0: 40}),  # first fold: copied as the writers do
    ],
)
def test_last_ran_reads_only_commits_inside_retention(
    successor: int | None,
    retired: int | None,
    policy: Literal["earliest", "latest"],
    expected: dict[int, int],
) -> None:
    assert (
        _plan_one(
            successor=successor,
            retired=retired,
            rule="last_ran",
            first=5,
            policy=policy,
        )
        == expected
    )


def test_the_declaration_defaults_to_the_writers_rule_and_refuses_others() -> None:
    base = {
        "from_service": "omnimarket-projections",
        "from_node": "live-events-writer",
        "from_version": "v1",
        "from_auto_offset_reset": "earliest",
    }
    assert ModelConsumerGroupHandover.model_validate(base).handover_rule == (
        "copy_if_uncommitted"
    )
    declared = ModelConsumerGroupHandover.model_validate(
        {**base, "handover_rule": "last_ran"}
    )
    assert declared.handover_rule == "last_ran"
    with pytest.raises(ValueError, match="handover_rule"):
        ModelConsumerGroupHandover.model_validate({**base, "handover_rule": "max"})


@dataclass
class _Lane:
    """One partition, two consumer groups and a count of effects run per event.

    Offsets are Kafka's: a group's commit is the next offset it will process.
    """

    rule: Rule
    reverse: bool = True
    log_end: int = 0
    committed: dict[Side, int | None] = field(
        default_factory=lambda: {"old": None, "new": None}
    )
    running: Side | None = "old"
    runs: dict[int, int] = field(default_factory=dict)

    def publish(self, count: int) -> None:
        self.log_end += count

    def consume(self, count: int) -> None:
        """The running side processes and commits ``count`` events, one at a time."""
        assert self.running is not None, "a fenced lane consumes nothing"
        side = self.running
        start = self.committed[side]
        position = self.log_end if start is None else start
        for _ in range(count):
            if position >= self.log_end:
                break
            self.runs[position] = self.runs.get(position, 0) + 1
            position += 1
        self.committed[side] = position

    def cutover(self, destination: Side, *, crash_after: int | None = None) -> None:
        """Fence both sides, plan from the destination's view, start it.

        ``crash_after`` lets the destination run that many events and then
        loses the procedure before it is recorded; the retry starts again at
        the fence, as the model's procedure crash does.
        """
        peer: Side = "new" if destination == "old" else "old"
        self.running = None
        if destination == "new" or self.reverse:
            plan = plan_offset_handover(
                partitions=(0,),
                successor_committed={0: self.committed[destination]},
                retired_committed={0: self.committed[peer]},
                first_offsets={0: 0},
                end_offsets={0: self.log_end},
                retired_auto_offset_reset="earliest",
                handover_rule=self.rule,
            )
            if 0 in plan:
                self.committed[destination] = plan[0]
        self.running = destination
        if crash_after is not None:
            self.consume(crash_after)
            self.publish(2)
            self.cutover(destination)


def _the_models_path(lane: _Lane) -> None:
    lane.committed["old"] = 0
    lane.publish(5)
    lane.consume(3)
    lane.publish(4)  # published during the fold window
    lane.cutover("new", crash_after=2)  # fold, crashed and retried
    lane.consume(4)
    lane.publish(3)
    lane.cutover("old", crash_after=1)  # rollback, crashed and retried
    lane.consume(3)
    lane.publish(3)
    lane.cutover("new", crash_after=1)  # second fold, crashed and retried
    lane.publish(2)
    lane.consume(50)


def test_last_ran_runs_each_event_once_through_the_models_path() -> None:
    """Fold, rollback and second fold, each crashed once and retried."""
    lane = _Lane(rule="last_ran")
    _the_models_path(lane)
    assert lane.log_end == 23
    assert lane.runs == dict.fromkeys(range(lane.log_end), 1), (
        "NoEventLost and NoDoubleEffect"
    )


def test_control_the_writers_rule_runs_effects_twice_on_the_second_fold() -> None:
    """The model's mut_stale_trust: a committed runtime group keeps its old commit."""
    lane = _Lane(rule="copy_if_uncommitted")
    _the_models_path(lane)
    assert max(lane.runs.values()) > 1


def test_control_a_rollback_with_no_reverse_handover_runs_effects_twice() -> None:
    """The model's mut_no_reverse: the retired side resumes from its own commit."""
    lane = _Lane(rule="last_ran", reverse=False)
    _the_models_path(lane)
    assert max(lane.runs.values()) > 1


@pytest.mark.asyncio
async def test_the_kafka_path_overwrites_a_stale_successor_commit() -> None:
    """On EventBusKafka the rule reaches the commit, per partition."""
    physical = _new_bus().resolve_bus_binding(TOPIC).physical_topic
    broker = _broker(physical, {0, 1, 2})
    with _kafka_bus(broker) as bus:
        first = await hand_over_committed_offsets(
            bus,
            topic=TOPIC,
            from_group=RETIRED_GROUP,
            to_group=SUCCESSOR_GROUP,
            from_auto_offset_reset="earliest",
        )
        successor = first.to_group
        broker.committed[successor] = {_tp(bus, 0): 20, _tp(bus, 1): 30}
        broker.committed[RETIRED_GROUP] = {_tp(bus, 0): 35, _tp(bus, 1): 12}
        broker.commit_calls.clear()

        result = await hand_over_committed_offsets(
            bus,
            topic=TOPIC,
            from_group=RETIRED_GROUP,
            to_group=SUCCESSOR_GROUP,
            from_auto_offset_reset="earliest",
            handover_rule="last_ran",
        )

    # partition 0: the retired side ran later; 1: the successor did, kept;
    # 2: neither committed, so the retired side's reset policy.
    assert result.committed == {0: 35, 2: 0}
    assert result.kept == (1,)
    ((group, offsets),) = broker.commit_calls
    assert group == successor
    assert {tp.partition: o for tp, o in offsets.items()} == {0: 35, 2: 0}
