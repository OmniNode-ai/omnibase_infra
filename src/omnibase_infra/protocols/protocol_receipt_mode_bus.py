# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The bus operations the receipt-mode client needs for one command (OMN-21341)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage


class ProtocolReceiptModeBus(Protocol):
    """Bus lifecycle and callback operations needed for a single command."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> object: ...

    async def subscribe(
        self,
        topic: str,
        *,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
        group_id: str,
    ) -> Callable[[], Awaitable[None]]: ...


__all__ = ["ProtocolReceiptModeBus"]
