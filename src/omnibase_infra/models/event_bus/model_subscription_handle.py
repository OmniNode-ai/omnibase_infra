# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The handle a bus returns from ``subscribe`` (OMN-21229).

Before this handle ``subscribe`` returned one bare ``unsubscribe`` coroutine
function, so the only thing a caller could do with a subscription was end it,
and ending it cancelled the consume task, which could cut an in-flight
callback. Contract hot loading needs the three steps apart and in order:

``pause``
    take no new record. The consumer stops fetching and a batch already
    fetched is released from its first unstarted record.
``drain``
    wait for the in-flight callback, and for the fetched batch to be finished
    or released. Only completed records are committed.
``stop``
    remove the callback and stop the consumer. It pauses and drains first, so
    stopping alone is the old unsubscribe with the in-flight callback allowed
    to finish.

The handle is awaitable-callable: ``await handle()`` is ``await handle.stop()``,
which is what every existing ``unsubscribe`` caller already does.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from omnibase_infra.enums.enum_subscription_phase import EnumSubscriptionPhase

__all__ = ["ModelSubscriptionHandle"]

logger = logging.getLogger(__name__)

_Step = Callable[[], Awaitable[None]]


class ModelSubscriptionHandle(BaseModel):
    """One subscription's pause, drain and stop, which only move forward."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    consumer_group: str
    subscription_key: str

    _pause: _Step = PrivateAttr()
    _drain: _Step = PrivateAttr()
    _stop: _Step = PrivateAttr()
    _phase: EnumSubscriptionPhase = PrivateAttr(default=EnumSubscriptionPhase.ACTIVE)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @classmethod
    def bind(
        cls,
        *,
        topic: str,
        consumer_group: str,
        subscription_key: str,
        pause: _Step,
        drain: _Step,
        stop: _Step,
    ) -> ModelSubscriptionHandle:
        """Build a handle over the owning bus's three steps."""
        handle = cls(
            topic=topic,
            consumer_group=consumer_group,
            subscription_key=subscription_key,
        )
        handle._pause = pause
        handle._drain = drain
        handle._stop = stop
        return handle

    @property
    def phase(self) -> EnumSubscriptionPhase:
        """The last phase this handle completed."""
        return self._phase

    async def pause(self) -> None:
        """Take no new record. A no-op once paused."""
        async with self._lock:
            await self._advance_to(EnumSubscriptionPhase.PAUSED)

    async def drain(self) -> None:
        """Pause if needed, then wait out the in-flight callback."""
        async with self._lock:
            await self._advance_to(EnumSubscriptionPhase.DRAINED)

    async def stop(self) -> None:
        """Pause, drain, then remove the callback. A no-op once stopped.

        The removal runs even when pause or drain raises: a subscription that
        could not be quiesced must still not keep its callback registered.
        """
        async with self._lock:
            try:
                await self._advance_to(EnumSubscriptionPhase.DRAINED)
            finally:
                if self._phase is not EnumSubscriptionPhase.STOPPED:
                    await self._stop()
                    self._phase = EnumSubscriptionPhase.STOPPED

    async def __call__(self) -> None:
        await self.stop()

    async def _advance_to(self, target: EnumSubscriptionPhase) -> None:
        order = list(EnumSubscriptionPhase)
        steps = {
            EnumSubscriptionPhase.PAUSED: self._pause,
            EnumSubscriptionPhase.DRAINED: self._drain,
            EnumSubscriptionPhase.STOPPED: self._stop,
        }
        for phase in order[order.index(self._phase) + 1 : order.index(target) + 1]:
            await steps[phase]()
            self._phase = phase
