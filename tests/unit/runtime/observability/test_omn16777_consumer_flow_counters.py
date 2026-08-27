# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16777 — the throughput accumulator that rides the existing heartbeat.

These tests pin the four properties that make the difference between a signal
and a lie:

1. A registered-but-silent subscription emits a ZERO row, not no row. "Alive and
   took nothing" is a fact; it is how ``IDLE`` gets proven downstream.
2. A DROPPED window is a sequence gap, never a zero row. Materializing a missed
   heartbeat as "no traffic" would reintroduce the exact defect this ticket
   exists to close (AC5).
3. Windows abut exactly (``window_start[n+1] == window_end[n]``) so no interval
   of platform time is unaccounted for.
4. The accumulator owns no clock. Every timestamp is injected; nothing here can
   drift, and replay of the same inputs reproduces byte-identical rows (AC6).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.models.observability import ModelConsumerFlowDelta
from omnibase_infra.runtime.observability import ConsumerFlowCounters

_T0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
_GROUP = "onex-dev.omnimarket.gateway-link-health-projection-compute.consume"
_TOPIC = (
    "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: real topic under test
)


def _delta_for(
    window: object, *, group: str = _GROUP, topic: str = _TOPIC
) -> ModelConsumerFlowDelta:
    """Pull one (group, topic) row out of a drained window."""
    deltas = getattr(window, "consumer_deltas", ())
    matches = [d for d in deltas if d.consumer_group == group and d.topic == topic]
    assert matches, f"no row for ({group}, {topic}) in window {window!r}"
    return matches[0]


@pytest.mark.unit
def test_priming_drain_reports_no_window_rather_than_a_fabricated_one() -> None:
    """The first drain has no observed window start, so it reports nothing.

    Emitting a window whose start we never observed would put a fabricated
    interval on the wire. Returning ``None`` says "no window", which the
    projection reads as absence — the honest answer.
    """
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.record_in(_GROUP, _TOPIC, 5)

    assert counters.drain(node_id=uuid4(), now=_T0) is None


@pytest.mark.unit
def test_registered_but_silent_consumer_emits_a_zero_row_not_no_row() -> None:
    """AC4's foundation: observed-idle must be distinguishable from unobserved.

    A subscription that takes nothing still emits a row of zeros. Without this
    the projection could not tell "this consumer is alive and the topic is
    quiet" from "this consumer was never instrumented", and every quiet topic
    would either vanish or light up.
    """
    node_id = uuid4()
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.drain(node_id=node_id, now=_T0)  # priming

    window = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))

    assert window is not None
    row = _delta_for(window)
    assert (
        row.messages_in,
        row.messages_out,
        row.messages_dlq,
        row.handler_errors,
    ) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.unit
def test_counters_are_per_group_topic_and_reset_every_window() -> None:
    """Counts are deltas, not running totals — a window reports its own traffic."""
    node_id = uuid4()
    other_topic = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: real topic under test
    counters = ConsumerFlowCounters()
    counters.drain(node_id=node_id, now=_T0)  # priming

    counters.record_in(_GROUP, _TOPIC, 3)
    counters.record_out(_GROUP, _TOPIC, 2)
    counters.record_dlq(_GROUP, _TOPIC, 1)
    counters.record_error(_GROUP, _TOPIC, 1)
    counters.record_in("other-group", other_topic, 7)

    first = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))
    assert first is not None
    row = _delta_for(first)
    assert (
        row.messages_in,
        row.messages_out,
        row.messages_dlq,
        row.handler_errors,
    ) == (
        3,
        2,
        1,
        1,
    )
    assert _delta_for(first, group="other-group", topic=other_topic).messages_in == 7

    second = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=120))
    assert second is not None
    assert _delta_for(second).messages_in == 0, (
        "counters leaked across windows — a delta that carries the previous "
        "window's traffic makes a stalled consumer look like it is still moving"
    )


@pytest.mark.unit
def test_windows_abut_exactly_so_no_platform_time_is_unaccounted_for() -> None:
    node_id = uuid4()
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.drain(node_id=node_id, now=_T0)

    first = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))
    second = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=120))

    assert first is not None and second is not None
    assert first.window_start == _T0
    assert first.window_end == second.window_start
    assert second.window_sequence == first.window_sequence + 1


@pytest.mark.unit
def test_a_dropped_window_is_a_sequence_gap_and_never_a_zero_row() -> None:
    """AC5: ``UNKNOWN != 0 messages``.

    Simulated by dropping a drained window on the floor, exactly as a heartbeat
    lost in transit would be. What the downstream projection then sees is a
    sequence that jumps — 1 then 3 — with NO row claiming zero traffic for
    window 2. A dropped window that arrived as a zero row would say "nothing
    happened", which is precisely the false-green this ticket exists to kill.
    """
    node_id = uuid4()
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.drain(node_id=node_id, now=_T0)

    delivered_first = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))
    counters.record_in(_GROUP, _TOPIC, 12)
    _lost = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=120))
    delivered_next = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=180))

    assert delivered_first is not None and delivered_next is not None
    assert delivered_first.window_sequence == 1
    assert delivered_next.window_sequence == 3, (
        "the sequence did not advance across the lost window, so a dropped "
        "heartbeat is indistinguishable from a quiet one"
    )
    assert _delta_for(delivered_next).messages_in == 0
    assert _lost is not None and _delta_for(_lost).messages_in == 12, (
        "the traffic that vanished with the lost window must not silently "
        "reappear in the next one"
    )


@pytest.mark.unit
def test_only_one_node_carries_the_window_per_process() -> None:
    """A second heartbeating node in the same process must not steal a partial
    window. Rows are keyed by (consumer_group, topic), so carriage is a
    labelling choice — but a split window would under-report both halves."""
    carrier, other = uuid4(), uuid4()
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.drain(node_id=carrier, now=_T0)

    counters.record_in(_GROUP, _TOPIC, 4)
    assert counters.drain(node_id=other, now=_T0 + timedelta(seconds=30)) is None

    window = counters.drain(node_id=carrier, now=_T0 + timedelta(seconds=60))
    assert window is not None
    assert _delta_for(window).messages_in == 4


@pytest.mark.unit
def test_produced_topic_tally_is_the_upstream_evidence_for_starved() -> None:
    """STARVED needs proof that something upstream was actually producing.

    That proof comes from the runtime's own publish seam, not a broker query —
    a broker query on a timer is the poller this ticket forbids.
    """
    node_id = uuid4()
    dest = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: real topic under test
    counters = ConsumerFlowCounters()
    counters.drain(node_id=node_id, now=_T0)

    counters.record_produced(dest, 5)
    window = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))

    assert window is not None
    produced = {p.topic: p.messages_produced for p in window.produce_deltas}
    assert produced == {dest: 5}


@pytest.mark.unit
def test_heartbeat_carries_the_window_and_absence_is_not_zero() -> None:
    """The carrier is the heartbeat the runtime already emits — no new transport.

    ``flow_window=None`` is the ABSENCE of a report (priming tick, or another
    node in the process carries it). It must never be read as "zero traffic";
    that conflation is the defect this whole ticket exists to close.
    """
    from omnibase_core.enums import EnumNodeKind
    from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

    node_id = uuid4()
    counters = ConsumerFlowCounters()
    counters.register(_GROUP, _TOPIC)
    counters.drain(node_id=node_id, now=_T0)
    counters.record_in(_GROUP, _TOPIC, 15750)
    window = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))

    carried = ModelNodeHeartbeatEvent(
        node_id=node_id,
        node_type=EnumNodeKind.COMPUTE,
        uptime_seconds=1.0,
        flow_window=window,
        timestamp=_T0 + timedelta(seconds=60),
    )
    silent = ModelNodeHeartbeatEvent(
        node_id=node_id,
        node_type=EnumNodeKind.COMPUTE,
        uptime_seconds=1.0,
        timestamp=_T0 + timedelta(seconds=60),
    )

    assert silent.flow_window is None
    assert carried.flow_window is not None
    # Survives the wire round-trip the projection will actually decode.
    round_tripped = ModelNodeHeartbeatEvent.model_validate_json(
        carried.model_dump_json()
    )
    assert round_tripped.flow_window is not None
    assert _delta_for(round_tripped.flow_window).messages_in == 15750


@pytest.mark.unit
def test_flow_delta_wire_field_set_is_pinned() -> None:
    """Drift guard for the projection's decode.

    ``node_projection_consumer_flow`` (omnimarket) decodes these rows off the
    wire. If a field is added, renamed or dropped here without the projection
    following, the projection silently stops seeing it — the failure mode is a
    quietly wrong row, not a crash, which is the worst kind. Break here first.
    """
    assert set(ModelConsumerFlowDelta.model_fields) == {
        "consumer_group",
        "topic",
        "node_id",
        "window_start",
        "window_end",
        "window_sequence",
        "messages_in",
        "messages_out",
        "messages_dlq",
        "handler_errors",
    }


@pytest.mark.unit
def test_replaying_the_same_inputs_reproduces_byte_identical_rows() -> None:
    """AC6 (determinism half): the accumulator reads no clock and no ambient
    state, so identical inputs serialize identically."""
    node_id = uuid4()

    def _run() -> str:
        counters = ConsumerFlowCounters()
        counters.register(_GROUP, _TOPIC)
        counters.drain(node_id=node_id, now=_T0)
        counters.record_in(_GROUP, _TOPIC, 15750)
        window = counters.drain(node_id=node_id, now=_T0 + timedelta(seconds=60))
        assert window is not None
        return window.model_dump_json()

    assert _run() == _run()
