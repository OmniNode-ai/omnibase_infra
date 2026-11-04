# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-flight and pause bookkeeping for one in-memory subscription (OMN-21229).

The in-memory bus delivers inline from ``publish``, so a subscription has no
consumer to pause. The gate is what pause and drain mean there: a paused
subscription is handed no new record, and drain waits for the callback a
publisher is already inside. The Kafka bus keeps the same ordering on its
consumers and its dispatch registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

__all__ = ["ModelSubscriptionGate"]

_Callback = Callable[[ModelEventMessage], Awaitable[None]]


class ModelSubscriptionGate(BaseModel):
    """Counts a subscription's running callbacks and holds back new ones once paused."""

    model_config = ConfigDict(extra="forbid")

    _paused: bool = PrivateAttr(default=False)
    _running: int = PrivateAttr(default=0)
    _idle: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    def model_post_init(self, __context: object) -> None:
        self._idle.set()

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Hand the callback no further record."""
        self._paused = True

    async def drain(self) -> None:
        """Wait for the callbacks already running to return."""
        await self._idle.wait()

    def wrap(self, callback: _Callback) -> _Callback:
        """The callback the bus registers in place of ``callback``."""

        async def gated(message: ModelEventMessage) -> None:
            if self._paused:
                return
            self._running += 1
            self._idle.clear()
            try:
                await callback(message)
            finally:
                self._running -= 1
                if self._running == 0:
                    self._idle.set()

        return gated
