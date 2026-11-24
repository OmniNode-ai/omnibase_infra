# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The savings signal topics join together, and one failed join stops no other (OMN-26418).

Counted, never timed: the fake bus records how many ``subscribe`` calls are
inside the join at once. A serial loop peaks at one; the topics joining
together peak at the number of topics.

Ticket: OMN-26418
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, cast

import pytest

from omnibase_infra.handlers.boot_deferred_subscribe import (
    subscribe_topics_concurrently,
)

if TYPE_CHECKING:
    from omnibase_infra.models import ModelNodeIdentity

pytestmark = [pytest.mark.unit]

_TOPICS = ("topic.a.v1", "topic.b.v1", "topic.c.v1")
_IDENTITY = cast("ModelNodeIdentity", object())


async def _noop(message: object) -> None:
    return None


class _GatedBus:
    """A bus whose subscribe returns only once every topic is inside it."""

    def __init__(self, expected: int, failing: frozenset[str] = frozenset()) -> None:
        self._expected = expected
        self._failing = failing
        self._inside = 0
        self.peak_inside = 0
        self.joined: list[str] = []
        self._all_inside = asyncio.Event()

    async def subscribe(
        self, topic: str, node_identity: object = None, on_message: object = None
    ) -> None:
        self._inside += 1
        self.peak_inside = max(self.peak_inside, self._inside)
        if self._inside >= self._expected:
            self._all_inside.set()
        # A serial caller never gets past this wait; the guard turns that hang
        # into a counted failure instead of a stuck test.
        await asyncio.wait_for(self._all_inside.wait(), timeout=5)
        self._inside -= 1
        if topic in self._failing:
            raise RuntimeError(f"broker refused {topic}")
        self.joined.append(topic)


@pytest.mark.asyncio
async def test_every_topic_is_inside_the_join_at_once() -> None:
    bus = _GatedBus(expected=len(_TOPICS))
    joined = await subscribe_topics_concurrently(
        bus.subscribe,
        [(t, _noop) for t in _TOPICS],
        node_identity=_IDENTITY,
        purpose="savings correlation",
    )
    assert bus.peak_inside == len(_TOPICS)
    assert joined == len(_TOPICS)
    assert sorted(bus.joined) == sorted(_TOPICS)


@pytest.mark.asyncio
async def test_a_failed_join_is_logged_and_stops_no_other(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = _GatedBus(expected=len(_TOPICS), failing=frozenset({_TOPICS[1]}))
    with caplog.at_level(logging.ERROR):
        joined = await subscribe_topics_concurrently(
            bus.subscribe,
            [(t, _noop) for t in _TOPICS],
            node_identity=_IDENTITY,
            purpose="savings correlation",
        )
    assert joined == len(_TOPICS) - 1
    assert sorted(bus.joined) == sorted((_TOPICS[0], _TOPICS[2]))
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert warned == [f"Could not subscribe to {_TOPICS[1]} for savings correlation"]


@pytest.mark.asyncio
async def test_cancellation_reaches_every_join_in_flight() -> None:
    bus = _GatedBus(
        expected=len(_TOPICS) + 1
    )  # never fills: every join stays in flight
    task = asyncio.create_task(
        subscribe_topics_concurrently(
            bus.subscribe,
            [(t, _noop) for t in _TOPICS],
            node_identity=_IDENTITY,
            purpose="savings correlation",
        )
    )
    while bus.peak_inside < len(_TOPICS):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
