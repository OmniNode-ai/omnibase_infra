# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S6 phantom-subscription alarm (OMN-14758, §d)."""

from __future__ import annotations

import pytest

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.runtime.core_runtime.phantom_alarm import PhantomAlarmMonitor

TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"


class _FakeConsumer:
    """Serves one batch of ``queued`` messages, then empty."""

    def __init__(self, queued: list[ModelTransportMessage]) -> None:
        self._queued = queued
        self.committed: list[ModelTransportMessage] = []

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def poll(self, *, max_messages: int, timeout_ms: int):
        batch, self._queued = self._queued[:max_messages], self._queued[max_messages:]
        return batch

    async def commit(self, message) -> None:
        self.committed.append(message)

    async def nack(self, message) -> None: ...


def _msg(offset: int) -> ModelTransportMessage:
    return ModelTransportMessage(
        topic=TOPIC,
        partition=0,
        offset=offset,
        key=None,
        value=b"{}",
        headers={},
        ack_token=(TOPIC, 0, offset),
    )


@pytest.mark.asyncio
async def test_phantom_flagged_when_polled_but_never_dispatched() -> None:
    inner = _FakeConsumer([_msg(0)])
    monitor = PhantomAlarmMonitor(inner, core_runtime_topics=frozenset({TOPIC}))
    # Poll the message (HWM advances) but NEVER commit ⇒ dispatched_count stays 0.
    await monitor.consumer.poll(max_messages=64, timeout_ms=0)
    phantom = monitor.evaluate(lag_zero_topics=frozenset({TOPIC}))
    assert phantom == frozenset({TOPIC})
    assert not monitor.is_healthy(lag_zero_topics=frozenset({TOPIC}))


@pytest.mark.asyncio
async def test_not_phantom_when_dispatched() -> None:
    inner = _FakeConsumer([_msg(0)])
    monitor = PhantomAlarmMonitor(inner, core_runtime_topics=frozenset({TOPIC}))
    batch = await monitor.consumer.poll(max_messages=64, timeout_ms=0)
    await monitor.consumer.commit(batch[0])  # dispatched
    assert monitor.evaluate(lag_zero_topics=frozenset({TOPIC})) == frozenset()
    assert monitor.is_healthy(lag_zero_topics=frozenset({TOPIC}))


@pytest.mark.asyncio
async def test_not_phantom_when_lag_nonzero() -> None:
    inner = _FakeConsumer([_msg(0)])
    monitor = PhantomAlarmMonitor(inner, core_runtime_topics=frozenset({TOPIC}))
    await monitor.consumer.poll(max_messages=64, timeout_ms=0)
    # LAG not yet zero (still catching up) ⇒ not a settled phantom.
    assert monitor.evaluate(lag_zero_topics=frozenset()) == frozenset()
