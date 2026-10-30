# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two bus operations a lease caller needs: publish a command, read replies (OMN-21424)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage


class ProtocolLeaseBus(Protocol):
    """Satisfied by ``EventBusKafka`` and ``EventBusInmemory``."""

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> object:
        """Publish one record."""
        ...

    async def subscribe_ungrouped(
        self,
        topic: str,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        """Read ``topic`` from now on; the returned callable stops the read."""
        ...


__all__ = ["ProtocolLeaseBus"]
