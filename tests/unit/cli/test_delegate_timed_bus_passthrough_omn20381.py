# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected timed buses preserve in-process handler traffic (OMN-20381)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    UnsubscribeCallback,
)
from omnibase_infra.cli.receipt_mode import (
    DelegatePhaseStopwatch,
    DelegatePhaseTimedBus,
    DelegatePhaseTimedRuntime,
)
from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase

pytestmark = pytest.mark.unit


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def _unsubscribe() -> None:
    return None


async def _on_message(_: object) -> None:
    return None


class _RecordingBus:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.publishes: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.subscriptions: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.publish_result = object()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def publish(self, topic: str, *args: object, **kwargs: object) -> object:
        self.publishes.append((topic, args, kwargs))
        self.clock.now += 0.25
        return self.publish_result

    async def subscribe(
        self, topic: str, *args: object, **kwargs: object
    ) -> UnsubscribeCallback:
        self.subscriptions.append((topic, args, kwargs))
        self.clock.now += 0.125
        return _unsubscribe


class _InlineBus(_RecordingBus):
    wrapper: DelegatePhaseTimedBus
    stopwatch: DelegatePhaseStopwatch

    async def publish(self, topic: str, *args: object, **kwargs: object) -> object:
        result = await super().publish(topic, *args, **kwargs)
        if topic == "command":
            nested_result = await self.wrapper.publish("output", None, b"x")
            assert nested_result is self.publish_result
            assert self.stopwatch.is_running(EnumDelegatePhase.PUBLISH)
            assert self.stopwatch.durations().publish_seconds is None
            assert not self.stopwatch.is_running(EnumDelegatePhase.TERMINAL_WAIT)
        return result


@pytest.mark.asyncio
async def test_subscribe_forwards_positional_arguments_and_callback() -> None:
    clock = _FakeClock()
    stopwatch = DelegatePhaseStopwatch(clock=clock)
    inner = _RecordingBus(clock)
    bus = DelegatePhaseTimedBus(inner, stopwatch)

    result = await bus.subscribe("reply", None, _on_message, group_id="g")

    assert result is _unsubscribe
    assert inner.subscriptions == [("reply", (None, _on_message), {"group_id": "g"})]
    assert stopwatch.durations().reply_subscribe_seconds == pytest.approx(0.125)


@pytest.mark.asyncio
async def test_inline_publish_preserves_outer_timing() -> None:
    clock = _FakeClock()
    stopwatch = DelegatePhaseStopwatch(clock=clock)
    inner = _InlineBus(clock)
    bus = DelegatePhaseTimedBus(inner, stopwatch)
    inner.wrapper = bus
    inner.stopwatch = stopwatch

    result = await bus.publish("command", None, b"command")

    assert result is inner.publish_result
    assert inner.publishes == [
        ("command", (None, b"command"), {}),
        ("output", (None, b"x"), {}),
    ]
    assert stopwatch.durations().publish_seconds == pytest.approx(0.5)
    assert not stopwatch.is_running(EnumDelegatePhase.PUBLISH)
    assert stopwatch.is_running(EnumDelegatePhase.TERMINAL_WAIT)


@pytest.mark.asyncio
async def test_sequential_publishes_keep_the_original_terminal_wait() -> None:
    clock = _FakeClock()
    stopwatch = DelegatePhaseStopwatch(clock=clock)
    inner = _RecordingBus(clock)
    bus = DelegatePhaseTimedBus(inner, stopwatch)

    first = await bus.publish("first", None, b"a")
    clock.now += 1.0
    second = await bus.publish("second", key=None, value=b"b")

    assert first is second is inner.publish_result
    assert inner.publishes == [
        ("first", (None, b"a"), {}),
        ("second", (), {"key": None, "value": b"b"}),
    ]
    assert stopwatch.durations().publish_seconds == pytest.approx(0.5)
    assert stopwatch.is_running(EnumDelegatePhase.TERMINAL_WAIT)
    await bus.close()
    assert stopwatch.durations().terminal_wait_seconds == pytest.approx(1.25)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", [EnumDelegatePhase.PUBLISH, EnumDelegatePhase.REPLY_SUBSCRIBE]
)
async def test_subscribe_during_client_traffic_is_forwarded_untimed(
    phase: EnumDelegatePhase,
) -> None:
    clock = _FakeClock()
    stopwatch = DelegatePhaseStopwatch(clock=clock)
    inner = _RecordingBus(clock)
    bus = DelegatePhaseTimedBus(inner, stopwatch)
    stopwatch.begin(phase)

    result = await bus.subscribe("handler", None, _on_message, group_id="g")

    assert result is _unsubscribe
    assert inner.subscriptions == [("handler", (None, _on_message), {"group_id": "g"})]
    assert stopwatch.is_running(phase)
    assert stopwatch.durations().reply_subscribe_seconds is None
    assert stopwatch.durations().publish_seconds is None
    stopwatch.end(phase)


class _BusProbeHandler:
    """Stands in for a handler that picks its dispatch port by the bus type."""

    def __init__(self, event_bus: object) -> None:
        self.event_bus = event_bus


def test_handlers_receive_the_untimed_runtime_bus(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: probe\nterminal_event: onex.evt.proof.probe-completed.v1\n",
        encoding="utf-8",
    )
    stopwatch = DelegatePhaseStopwatch()
    runtime = DelegatePhaseTimedRuntime(
        workflow_path=contract,
        state_root=tmp_path / "state",
        phase_stopwatch=stopwatch,
    )
    inner = EventBusInmemory()
    timed = DelegatePhaseTimedBus(inner, stopwatch)

    handler = runtime._instantiate_handler(__name__, "_BusProbeHandler", bus=timed)

    assert isinstance(handler, _BusProbeHandler)
    assert handler.event_bus is inner
