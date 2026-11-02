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
from collections.abc import Awaitable, Callable
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
from omnibase_infra.protocols.protocol_delegate_phase_stopwatch import (
    ProtocolDelegatePhaseStopwatch,
)


class DelegatePhaseTimedBus:
    """Time the client's bus calls, forwarding their arguments unchanged."""

    def __init__(
        self, inner: ProtocolLocalRuntimeBus, stopwatch: ProtocolDelegatePhaseStopwatch
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
    inner: ProtocolLocalRuntimeBus, stopwatch: ProtocolDelegatePhaseStopwatch
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
