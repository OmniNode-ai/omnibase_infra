# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-25106 -- a cancelled ``DLQConsumer`` start or stop left a GHOST group member.

Measured read-only on the h201 dev runtime, 2026-11-16, from the runtime log
(``docker logs omninode-runtime``, 23:14Z-23:40Z) and the replay group itself::

    group onex-dlq-replay joins in the window ......... 409
    'Heartbeat failed ... because it is rebalancing' .. 308
    members joined in ONE generation (772314, 772315) . 8 and 9
        (the node holds three consumers, one per declared topic, and the
         handler runs one consumer at a time)
    members alive across 37-38 generations ............ 3  (minutes, not seconds)
    DlqDependencyLifecycleTimeoutError 'start() did
      not complete within 15.00s' ..................... 11
    'Unclosed AIOKafkaConsumer' (asyncio) ............. 11
    'teardown ... was ABANDONED' ...................... 2

The handler bounds ``DLQConsumer.start()`` and ``stop()`` with
``asyncio.wait_for``. A timeout CANCELS the call half way through. Before this
change ``DLQConsumer.start()`` had already assigned ``self._consumer`` but not
yet set ``_started``, and ``stop()`` only acts when ``_started`` is True, so a
cancelled start left an ``AIOKafkaConsumer`` that had joined the group (its
coordinator task and heartbeat keep running) with nothing holding a path to
stop it; the next ``start()`` then overwrote the reference. A cancelled
``stop()`` dropped the reference in its ``finally`` the same way. Each ghost
keeps its seat and its partition assignment, so the live consumer of the next
run is assigned nothing, reads 0 records, every other join rebalances the
whole group, and the replay lag never falls (commands 3,331, events 260 at
23:43Z with the group otherwise Empty).

These tests drive the real ``DLQConsumer`` over a fake ``AIOKafkaConsumer`` and
assert the invariant: every ``AIOKafkaConsumer`` the engine constructs is
stopped, whether ``start()`` was cancelled, failed, or ``stop()`` was cancelled.

Ticket: OMN-25106
"""

from __future__ import annotations

import asyncio

import pytest
from aiokafka.errors import KafkaConnectionError

from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    ModelDlqReplayEngineConfig,
)

pytestmark = pytest.mark.unit

_TOPIC = "onex.dlq.omnibase-infra.events.v1"


class _FakeAIOKafkaConsumer:
    """Stands in for ``AIOKafkaConsumer``; records whether it was stopped."""

    instances: list[_FakeAIOKafkaConsumer] = []
    start_mode: str = "ok"  # ok | hang | refuse
    stop_seconds: float = 0.0

    def __init__(self, *topics: str, **kwargs: object) -> None:
        self.topics = topics
        self.joined = False
        self.stopped = False
        _FakeAIOKafkaConsumer.instances.append(self)

    async def start(self) -> None:
        # A real consumer joins the group inside start(); from then on its
        # coordinator task heartbeats until stop() is awaited to completion.
        self.joined = True
        if _FakeAIOKafkaConsumer.start_mode == "hang":
            await asyncio.Event().wait()
        if _FakeAIOKafkaConsumer.start_mode == "refuse":
            raise KafkaConnectionError("broker refused the join")

    async def stop(self) -> None:
        await asyncio.sleep(_FakeAIOKafkaConsumer.stop_seconds)
        self.stopped = True


@pytest.fixture(autouse=True)
def _fake_kafka_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAIOKafkaConsumer.instances = []
    _FakeAIOKafkaConsumer.start_mode = "ok"
    _FakeAIOKafkaConsumer.stop_seconds = 0.0
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", _FakeAIOKafkaConsumer)


def _consumer() -> DLQConsumer:
    return DLQConsumer(
        ModelDlqReplayEngineConfig(bootstrap_servers="localhost:9092", dlq_topic=_TOPIC)
    )


async def _settle() -> None:
    """Let a detached close run to completion."""
    for _ in range(20):
        await asyncio.sleep(0.01)


async def test_start_then_stop_stops_the_one_consumer_it_made() -> None:
    """Positive control: the normal path stops exactly what it started."""
    consumer = _consumer()
    await consumer.start()
    assert [c.stopped for c in _FakeAIOKafkaConsumer.instances] == [False]
    await consumer.stop()
    assert [c.stopped for c in _FakeAIOKafkaConsumer.instances] == [True]


async def test_start_cancelled_mid_join_does_not_leave_a_joined_ghost() -> None:
    """``wait_for`` cancelling ``start()`` must still end the member's seat."""
    _FakeAIOKafkaConsumer.start_mode = "hang"
    consumer = _consumer()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consumer.start(), timeout=0.05)
    await _settle()

    ghosts = [c for c in _FakeAIOKafkaConsumer.instances if c.joined and not c.stopped]
    assert ghosts == []


async def test_start_that_raises_does_not_leave_a_joined_ghost() -> None:
    _FakeAIOKafkaConsumer.start_mode = "refuse"
    consumer = _consumer()

    with pytest.raises(KafkaConnectionError):
        await consumer.start()
    await _settle()

    assert [c.stopped for c in _FakeAIOKafkaConsumer.instances] == [True]


async def test_a_start_after_a_cancelled_start_does_not_orphan_the_first() -> None:
    """The next run's ``start()`` overwrote the reference to the cancelled one."""
    _FakeAIOKafkaConsumer.start_mode = "hang"
    consumer = _consumer()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consumer.start(), timeout=0.05)

    _FakeAIOKafkaConsumer.start_mode = "ok"
    await consumer.start()
    await consumer.stop()
    await _settle()

    assert len(_FakeAIOKafkaConsumer.instances) == 2
    assert all(c.stopped for c in _FakeAIOKafkaConsumer.instances)


async def test_stop_cancelled_mid_close_still_finishes_the_close() -> None:
    """``wait_for`` cancelling ``stop()`` must not abandon a half-closed member."""
    consumer = _consumer()
    await consumer.start()
    _FakeAIOKafkaConsumer.stop_seconds = 0.1

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consumer.stop(), timeout=0.02)
    await asyncio.sleep(0.3)

    assert [c.stopped for c in _FakeAIOKafkaConsumer.instances] == [True]


async def test_stop_returns_when_no_consumer_was_ever_started() -> None:
    await _consumer().stop()
    assert _FakeAIOKafkaConsumer.instances == []
