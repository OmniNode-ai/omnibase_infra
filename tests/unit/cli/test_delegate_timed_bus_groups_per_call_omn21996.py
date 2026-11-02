# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21996: an ``onex delegate`` run through the timing bus joins no consumer group.

``RuntimeLocal`` reads its terminals with no group only when the bus it holds offers
``subscribe_ungrouped``. ``DelegatePhaseTimedBus`` wraps every delegate bus and hid that
read, so each delegation left ``runtime_local_terminal_t1_run_<id>`` groups Empty on the
dev broker (1,425 of them at 2026-11-02T10:50Z).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    ProtocolLocalRuntimeUngroupedBus,
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.cli.receipt_mode import DelegatePhaseStopwatch
from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase
from omnibase_infra.models.delegation.model_delegate_phase_timed_bus import (
    wrap_delegate_bus,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omnimarket.delegate-skill-failed.v1"
RUNS = 100

Callback = Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]]


class _GroupedBus:
    """The four-method bus a broker-backed run holds; records every consumer group."""

    def __init__(self) -> None:
        self.groups: list[str] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def publish(self, topic: str, key: object, value: bytes) -> object:
        return None

    async def subscribe(
        self, topic: str, *, on_message: Callback, group_id: str
    ) -> UnsubscribeCallback:
        self.groups.append(group_id)

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


class _UngroupedBus(_GroupedBus):
    """Adds the group-less read the Kafka bus offers."""

    def __init__(self, stopwatch: DelegatePhaseStopwatch | None = None) -> None:
        super().__init__()
        self.ungrouped_topics: list[str] = []
        self.reply_subscribe_running: list[bool] = []
        self._stopwatch = stopwatch

    async def subscribe_ungrouped(
        self, topic: str, on_message: Callback
    ) -> UnsubscribeCallback:
        self.ungrouped_topics.append(topic)
        if self._stopwatch is not None:
            self.reply_subscribe_running.append(
                self._stopwatch.is_running(EnumDelegatePhase.REPLY_SUBSCRIBE)
            )

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


async def _noop(_: ProtocolLocalRuntimeMessage) -> None:
    return None


def _armed_runtime(tmp_path: Path) -> RuntimeLocal:
    runtime = RuntimeLocal(
        workflow_path=tmp_path / "contract.yaml", state_root=tmp_path / "state"
    )
    runtime._expected_correlation_id = uuid.uuid4()
    return runtime


@pytest.mark.asyncio
async def test_groups_per_call_a_hundred_timed_runs_leave_no_group(
    tmp_path: Path,
) -> None:
    inner = _UngroupedBus()
    for _ in range(RUNS):
        bus = wrap_delegate_bus(inner, DelegatePhaseStopwatch())
        runtime = _armed_runtime(tmp_path)
        await runtime._subscribe_run_scoped(
            bus, TOPIC, _noop, "runtime_local_terminal_t1"
        )

    assert inner.groups == []
    assert inner.ungrouped_topics == [TOPIC] * RUNS


def test_groups_per_call_the_timing_bus_is_an_ungrouped_bus_when_its_inner_is() -> None:
    bus = wrap_delegate_bus(_UngroupedBus(), DelegatePhaseStopwatch())

    assert isinstance(bus, ProtocolLocalRuntimeUngroupedBus)


def test_groups_per_call_the_timing_bus_offers_no_ungrouped_read_when_its_inner_has_none() -> (
    None
):
    bus = wrap_delegate_bus(_GroupedBus(), DelegatePhaseStopwatch())

    assert not isinstance(bus, ProtocolLocalRuntimeUngroupedBus)


@pytest.mark.asyncio
async def test_groups_per_call_the_ungrouped_read_is_timed_as_reply_subscribe() -> None:
    stopwatch = DelegatePhaseStopwatch()
    inner = _UngroupedBus(stopwatch)
    bus = wrap_delegate_bus(inner, stopwatch)

    assert isinstance(bus, ProtocolLocalRuntimeUngroupedBus)
    unsubscribe = await bus.subscribe_ungrouped(TOPIC, _noop)
    await unsubscribe()

    assert inner.reply_subscribe_running == [True]
    assert stopwatch.durations().reply_subscribe_seconds is not None
