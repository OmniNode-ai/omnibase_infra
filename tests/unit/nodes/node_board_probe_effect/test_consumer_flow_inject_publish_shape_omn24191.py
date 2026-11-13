# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 probe's sole lane write is one marked malformed trigger (OMN-24191)."""

from __future__ import annotations

import json

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
    inject,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    APPLIED_TOPIC,
    PROBE_CID_PREFIX,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
)

pytestmark = pytest.mark.unit

#: The newest real envelope on APPLIED_TOPIC, as rpk's JSON consume line shapes it.
_COPIED_ENVELOPE: dict[str, object] = {
    "envelope_id": "env-00000041",
    "correlation_id": "orig-correlation-id",
    "envelope_timestamp": "2026-11-13T00:00:00.000000Z",
    "payload": {"a": 1},
    "tenant_id": "t-1",
}
_NEWEST_MESSAGE: str = json.dumps(
    {
        "offset": 41,
        "headers": [{"key": "producer", "value": "x"}],
        "value": json.dumps(_COPIED_ENVELOPE),
    }
)
_PRODUCE_ACK: str = "Produced to partition 0 at offset 42 with timestamp 1."
_READ_PREFIX: tuple[str, ...] = (
    "topic",
    "consume",
    APPLIED_TOPIC,
    "-o",
    "-1",
    "-n",
    "1",
)


class _FakeLane:
    """Stands in for the lane: answers every rpk call from canned broker output."""

    def __init__(self, newest: str = _NEWEST_MESSAGE) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self._newest = newest

    def rpk(self, *args: str, stdin: str | None = None, timeout: float = 60.0) -> str:
        self.calls.append((args, stdin))
        if args[: len(_READ_PREFIX)] == _READ_PREFIX:
            return self._newest
        if args[:2] == ("topic", "produce"):
            return _PRODUCE_ACK
        raise AssertionError(f"unexpected rpk call: {args}")


def test_inject_publishes_exactly_one_record_to_the_applied_topic_only() -> None:
    """One run is a single read plus one produce to APPLIED_TOPIC — nothing else."""
    lane = _FakeLane()
    inject(lane)
    produces = [c for c in lane.calls if c[0][:2] == ("topic", "produce")]
    assert len(produces) == 1
    assert produces[0][0][:3] == ("topic", "produce", APPLIED_TOPIC)
    assert len(lane.calls) == 2  # exactly one read and one write
    consume = lane.calls[0]
    assert consume[0][: len(_READ_PREFIX)] == _READ_PREFIX
    assert consume[1] is None  # a read ships no stdin


def test_the_published_record_is_the_marked_malformed_trigger() -> None:
    """The record is the copied envelope with its payload replaced by the marker."""
    lane = _FakeLane()
    result = inject(lane)
    stdin = next(c[1] for c in lane.calls if c[0][:2] == ("topic", "produce"))
    assert stdin is not None
    published = json.loads(stdin)
    assert published["payload"] == {"c28_probe_marker": result["marker"]}
    assert published["correlation_id"] == result["correlation_id"]
    assert published["correlation_id"].startswith(PROBE_CID_PREFIX)
    assert published["tenant_id"] == "t-1"  # the copied key survives the rewrite
    assert result["offset"] == 42
    assert result["envelope_copied_from_offset"] == 41


def test_an_empty_topic_publishes_nothing() -> None:
    """A newest message that is not an envelope aborts before any write."""
    lane = _FakeLane(newest="not json")
    with pytest.raises(ConsumerFlowInputError):
        inject(lane)
    assert not [c for c in lane.calls if c[0][:2] == ("topic", "produce")]
