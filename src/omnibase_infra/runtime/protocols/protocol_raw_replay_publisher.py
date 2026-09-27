# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Broker-acknowledged raw replay publisher shape."""

from __future__ import annotations

from typing import Protocol


class ProtocolRawReplayPublisher(Protocol):
    """Publish original bytes and headers; return only after broker confirmation."""

    async def publish(
        self,
        topic: str,
        *,
        key: bytes | None,
        value: bytes,
        headers: list[tuple[str, bytes | None]],
        timestamp_ms: int,
    ) -> None: ...


__all__ = ["ProtocolRawReplayPublisher"]
