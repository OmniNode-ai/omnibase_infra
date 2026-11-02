# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The bus wrapper that times the client phases of an ``onex delegate`` run (OMN-19452).

OMN-21996: ``RuntimeLocal`` reads its command and terminals with no consumer group only
when the bus it holds offers ``subscribe_ungrouped``. This wrapper hid that read, so
every delegation left ``runtime_local_terminal_t1_run_<id>`` groups Empty on the broker
(1,425 on the dev broker at 2026-11-02T10:50Z). :func:`wrap_delegate_bus` returns the
wrapper that offers the read exactly when the wrapped bus has it: a protocol
``isinstance`` looks at the class, so a bus with no such read must not be given the
method.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import cast

from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    ProtocolLocalRuntimeBus,
    ProtocolLocalRuntimeUngroupedBus,
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase
from omnibase_infra.models.delegation.model_delegate_phase_durations import (
    ModelDelegatePhaseDurations,
)


class DelegatePhaseStopwatch:
    """Accumulate completed spans using an injectable monotonic clock."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started: dict[EnumDelegatePhase, float] = {}
        self._totals: dict[EnumDelegatePhase, float] = {}

    def begin(self, phase: EnumDelegatePhase) -> None:
        """Start a span, refusing to replace an already running span."""
        if self.is_running(phase):
            raise ValueError(f"Phase {phase.value} is already running")
        self._started[phase] = self._clock()

    def end(self, phase: EnumDelegatePhase) -> None:
        """End a running span and add its elapsed seconds to the phase total."""
        if not self.is_running(phase):
            raise ValueError(f"Phase {phase.value} is not running")
        elapsed = self._clock() - self._started.pop(phase)
        self._totals[phase] = self._totals.get(phase, 0.0) + elapsed

    @contextmanager
    def phase(self, phase: EnumDelegatePhase) -> Iterator[None]:
        """Record a span, including elapsed time when its body raises."""
        self.begin(phase)
        try:
            yield
        finally:
            self.end(phase)

    def is_running(self, phase: EnumDelegatePhase) -> bool:
        """Whether a phase has a span that has begun and has not ended."""
        return phase in self._started

    def durations(self) -> ModelDelegatePhaseDurations:
        """Snapshot completed totals, leaving phases with no ended span absent."""
        return ModelDelegatePhaseDurations.model_validate(
            {
                f"{phase.value}_seconds": self._totals.get(phase)
                for phase in EnumDelegatePhase
            }
        )


class DelegatePhaseTimedBus:
    """Time the client's bus calls, forwarding their arguments unchanged."""

    def __init__(
        self, inner: ProtocolLocalRuntimeBus, stopwatch: DelegatePhaseStopwatch
    ) -> None:
        self._inner = inner
        self._stopwatch = stopwatch

    @property
    def inner(self) -> ProtocolLocalRuntimeBus:
        """The runtime's own bus, which in-process handlers receive (OMN-20381)."""
        return self._inner

    async def start(self) -> None:
        """Record connection time, including a failed connection attempt."""
        with self._stopwatch.phase(EnumDelegatePhase.BUS_CONNECT):
            await self._inner.start()

    async def close(self) -> None:
        """Finish terminal waiting before the bus begins closing."""
        if self._stopwatch.is_running(EnumDelegatePhase.TERMINAL_WAIT):
            self._stopwatch.end(EnumDelegatePhase.TERMINAL_WAIT)
        await self._inner.close()

    async def publish(self, topic: str, *args: object, **kwargs: object) -> object:
        """Skip inline handler timing and avoid restarting terminal wait (OMN-20381)."""
        publish = cast("Callable[..., Awaitable[object]]", self._inner.publish)
        if self._stopwatch.is_running(EnumDelegatePhase.PUBLISH):
            return await publish(topic, *args, **kwargs)
        with self._stopwatch.phase(EnumDelegatePhase.PUBLISH):
            result = await publish(topic, *args, **kwargs)
        if not self._stopwatch.is_running(EnumDelegatePhase.TERMINAL_WAIT):
            self._stopwatch.begin(EnumDelegatePhase.TERMINAL_WAIT)
        return result

    async def subscribe(
        self, topic: str, *args: object, **kwargs: object
    ) -> UnsubscribeCallback:
        """Preserve handler subscription arguments without nested timing (OMN-20381)."""
        subscribe = cast(
            "Callable[..., Awaitable[UnsubscribeCallback]]", self._inner.subscribe
        )
        if self._stopwatch.is_running(
            EnumDelegatePhase.PUBLISH
        ) or self._stopwatch.is_running(EnumDelegatePhase.REPLY_SUBSCRIBE):
            return await subscribe(topic, *args, **kwargs)
        with self._stopwatch.phase(EnumDelegatePhase.REPLY_SUBSCRIBE):
            return await subscribe(topic, *args, **kwargs)


class DelegatePhaseTimedUngroupedBus(DelegatePhaseTimedBus):
    """The timing bus over a bus that can read a topic with no consumer group (OMN-21996)."""

    async def subscribe_ungrouped(
        self,
        topic: str,
        on_message: Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]],
    ) -> UnsubscribeCallback:
        """Time the reply read like ``subscribe``, without nesting inside an open span."""
        ungrouped = cast("ProtocolLocalRuntimeUngroupedBus", self._inner)
        if self._stopwatch.is_running(
            EnumDelegatePhase.PUBLISH
        ) or self._stopwatch.is_running(EnumDelegatePhase.REPLY_SUBSCRIBE):
            return await ungrouped.subscribe_ungrouped(topic, on_message)
        with self._stopwatch.phase(EnumDelegatePhase.REPLY_SUBSCRIBE):
            return await ungrouped.subscribe_ungrouped(topic, on_message)


def wrap_delegate_bus(
    inner: ProtocolLocalRuntimeBus, stopwatch: DelegatePhaseStopwatch
) -> DelegatePhaseTimedBus:
    """Wrap ``inner`` so a delegation's terminals are read with no group when it can."""
    if isinstance(inner, ProtocolLocalRuntimeUngroupedBus):
        reader = inner.subscribe_ungrouped
        if inspect.iscoroutinefunction(reader):
            try:
                # Check the positional call the wrapper makes without invoking
                # the reader, which could open a broker connection.
                inspect.signature(reader).bind("topic", object())
            except (TypeError, ValueError):
                pass
            else:
                return DelegatePhaseTimedUngroupedBus(inner, stopwatch)
    return DelegatePhaseTimedBus(inner, stopwatch)


__all__ = [
    "DelegatePhaseTimedBus",
    "DelegatePhaseTimedUngroupedBus",
    "wrap_delegate_bus",
]
