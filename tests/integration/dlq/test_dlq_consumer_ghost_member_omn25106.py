# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-25106 -- DLQConsumer start/stop cancelled by ``wait_for`` leaves no ghost member.

Drives the real ``DLQConsumer`` through the same ``asyncio.wait_for`` bound the
dlq replay handler applies, over a stand-in ``AIOKafkaConsumer`` that records
whether each constructed member was stopped.

Ticket: OMN-25106
"""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    ModelDlqReplayEngineConfig,
)

pytestmark = pytest.mark.integration

_TOPIC = "onex.dlq.omnibase-infra.events.v1"


class _Member:
    instances: list[_Member] = []
    hang_on_start = False
    stop_seconds = 0.0

    def __init__(self, *topics: str, **kwargs: object) -> None:
        self.joined = False
        self.stopped = False
        _Member.instances.append(self)

    async def start(self) -> None:
        self.joined = True
        if _Member.hang_on_start:
            await asyncio.Event().wait()

    async def stop(self) -> None:
        await asyncio.sleep(_Member.stop_seconds)
        self.stopped = True


@pytest.fixture(autouse=True)
def _fake_member(monkeypatch: pytest.MonkeyPatch) -> None:
    _Member.instances = []
    _Member.hang_on_start = False
    _Member.stop_seconds = 0.0
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", _Member)
    engine_dlq_replay._MEMBER_CLOSES.clear()


def _consumer() -> DLQConsumer:
    return DLQConsumer(
        ModelDlqReplayEngineConfig(bootstrap_servers="localhost:9092", dlq_topic=_TOPIC)
    )


async def _settle() -> None:
    await asyncio.gather(*engine_dlq_replay._MEMBER_CLOSES, return_exceptions=True)


async def test_wait_for_cancelled_start_then_restart_leaves_no_ghost() -> None:
    _Member.hang_on_start = True
    consumer = _consumer()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consumer.start(), timeout=0.05)
    _Member.hang_on_start = False
    await consumer.start()
    await consumer.stop()
    await _settle()
    assert len(_Member.instances) == 2
    assert all(m.stopped for m in _Member.instances)


async def test_wait_for_cancelled_stop_still_closes_member() -> None:
    consumer = _consumer()
    await consumer.start()
    _Member.stop_seconds = 0.1
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consumer.stop(), timeout=0.02)
    await _settle()
    assert [m.stopped for m in _Member.instances] == [True]
