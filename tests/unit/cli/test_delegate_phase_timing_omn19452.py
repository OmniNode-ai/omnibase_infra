# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19452 AC2: the client phases of a delegation are timed individually.

A delegation that takes ninety seconds is one number until each phase has its
own: the CLI's own startup, the locus probe, the bus connect, the reply
subscription, the publish and the wait for the terminal. These tests pin the
recorder, the strict model the receipt carries, and the bus wrapper that
observes the three phases that happen below the CLI's frame (inside the
runtime, on the bus), against a fake bus on a fake clock.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from pydantic import ValidationError

from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_infra.cli.receipt_mode import (
    DelegatePhaseStopwatch,
    DelegatePhaseTimedBus,
)
from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase
from omnibase_infra.models.delegation.model_delegate_phase_durations import (
    ModelDelegatePhaseDurations,
)

pytestmark = pytest.mark.unit


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeBus:
    """The four-method bus shape the runtime uses, spending fake time per call."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.calls: list[str] = []
        self.fail_publish = False

    async def start(self) -> None:
        self.calls.append("start")
        self._clock.advance(0.5)

    async def close(self) -> None:
        self.calls.append("close")
        self._clock.advance(0.01)

    async def publish(self, topic: str, key: object, value: bytes) -> object:
        self.calls.append(f"publish:{topic}")
        self._clock.advance(0.25)
        if self.fail_publish:
            raise ConnectionError("synthetic publish failure")
        return "published"

    async def subscribe(
        self,
        topic: str,
        *,
        on_message: Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]],
        group_id: str,
    ) -> UnsubscribeCallback:
        self.calls.append(f"subscribe:{topic}:{group_id}")
        self._clock.advance(0.125)

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


async def _noop_on_message(_: object) -> None:
    return None


def _assign(model: object, name: str, value: object) -> None:
    setattr(model, name, value)


class TestTheDurationsModelIsStrict:
    def test_every_phase_defaults_to_not_reached(self) -> None:
        durations = ModelDelegatePhaseDurations()

        assert durations.model_dump() == {
            "startup_seconds": None,
            "locus_probe_seconds": None,
            "bus_connect_seconds": None,
            "reply_subscribe_seconds": None,
            "publish_seconds": None,
            "terminal_wait_seconds": None,
        }

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegatePhaseDurations.model_validate({"publish_ms": 3})

    def test_a_negative_duration_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegatePhaseDurations(publish_seconds=-0.1)

    def test_the_model_is_frozen(self) -> None:
        durations = ModelDelegatePhaseDurations(publish_seconds=0.1)

        with pytest.raises(ValidationError):
            _assign(durations, "publish_seconds", 0.2)


class TestTheStopwatch:
    def test_a_phase_that_never_ended_is_not_reached(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)

        stopwatch.begin(EnumDelegatePhase.STARTUP)
        clock.advance(2.0)

        assert stopwatch.durations().startup_seconds is None

    def test_a_phase_is_the_time_between_begin_and_end(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)

        stopwatch.begin(EnumDelegatePhase.LOCUS_PROBE)
        clock.advance(1.5)
        stopwatch.end(EnumDelegatePhase.LOCUS_PROBE)

        assert stopwatch.durations().locus_probe_seconds == pytest.approx(1.5)

    def test_repeated_spans_of_one_phase_accumulate(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)

        for seconds in (0.25, 0.5):
            with stopwatch.phase(EnumDelegatePhase.REPLY_SUBSCRIBE):
                clock.advance(seconds)

        assert stopwatch.durations().reply_subscribe_seconds == pytest.approx(0.75)

    def test_a_span_that_raises_is_still_closed(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)

        with pytest.raises(RuntimeError):
            with stopwatch.phase(EnumDelegatePhase.PUBLISH):
                clock.advance(0.5)
                raise RuntimeError("synthetic")

        assert stopwatch.durations().publish_seconds == pytest.approx(0.5)
        assert not stopwatch.is_running(EnumDelegatePhase.PUBLISH)

    def test_ending_a_phase_that_never_began_is_refused(self) -> None:
        stopwatch = DelegatePhaseStopwatch(clock=_FakeClock())

        with pytest.raises(ValueError, match="terminal_wait"):
            stopwatch.end(EnumDelegatePhase.TERMINAL_WAIT)

    def test_beginning_a_running_phase_is_refused(self) -> None:
        stopwatch = DelegatePhaseStopwatch(clock=_FakeClock())
        stopwatch.begin(EnumDelegatePhase.STARTUP)

        with pytest.raises(ValueError, match="startup"):
            stopwatch.begin(EnumDelegatePhase.STARTUP)


class TestTheTimedBus:
    @pytest.mark.asyncio
    async def test_each_bus_phase_is_timed_in_the_order_the_runtime_runs_them(
        self,
    ) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)
        inner = _FakeBus(clock)
        bus = DelegatePhaseTimedBus(inner, stopwatch)

        await bus.start()
        await bus.subscribe(
            "terminal.one", on_message=_noop_on_message, group_id="run-a"
        )
        await bus.subscribe(
            "terminal.two", on_message=_noop_on_message, group_id="run-b"
        )
        published = await bus.publish("command", None, b"{}")
        clock.advance(4.0)  # the wait for the terminal
        await bus.close()

        durations = stopwatch.durations()
        assert published == "published"
        assert durations.bus_connect_seconds == pytest.approx(0.5)
        assert durations.reply_subscribe_seconds == pytest.approx(0.25)
        assert durations.publish_seconds == pytest.approx(0.25)
        assert durations.terminal_wait_seconds == pytest.approx(4.0)
        assert inner.calls == [
            "start",
            "subscribe:terminal.one:run-a",
            "subscribe:terminal.two:run-b",
            "publish:command",
            "close",
        ]

    @pytest.mark.asyncio
    async def test_the_wrapper_returns_the_unsubscribe_callback_it_was_given(
        self,
    ) -> None:
        clock = _FakeClock()
        bus = DelegatePhaseTimedBus(
            _FakeBus(clock), DelegatePhaseStopwatch(clock=clock)
        )

        unsubscribe = await bus.subscribe(
            "terminal", on_message=_noop_on_message, group_id="run"
        )

        assert await unsubscribe() is None

    @pytest.mark.asyncio
    async def test_a_run_that_never_published_never_waited(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)
        bus = DelegatePhaseTimedBus(_FakeBus(clock), stopwatch)

        await bus.start()
        await bus.close()

        durations = stopwatch.durations()
        assert durations.publish_seconds is None
        assert durations.terminal_wait_seconds is None

    @pytest.mark.asyncio
    async def test_a_failed_publish_is_timed_and_starts_no_wait(self) -> None:
        clock = _FakeClock()
        stopwatch = DelegatePhaseStopwatch(clock=clock)
        inner = _FakeBus(clock)
        inner.fail_publish = True
        bus = DelegatePhaseTimedBus(inner, stopwatch)

        with pytest.raises(ConnectionError):
            await bus.publish("command", None, b"{}")
        await bus.close()

        durations = stopwatch.durations()
        assert durations.publish_seconds == pytest.approx(0.25)
        assert durations.terminal_wait_seconds is None
