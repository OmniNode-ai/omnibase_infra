# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Producer seam for RuntimeLogEventBridge (OMN-19992)."""

from __future__ import annotations

from typing import Protocol


class ProtocolRuntimeLogProducer(Protocol):
    """The two producer calls RuntimeLogEventBridge makes.

    ``AIOKafkaProducer`` satisfies this for the Kafka transport; the in-memory
    transport is served by ``PublisherEventBusRuntimeLog``. The bridge neither
    knows nor cares which bus the bytes land on.
    """

    async def send(self, topic: str, *, value: bytes) -> object:
        """Publish ``value`` to ``topic``."""
        ...

    async def stop(self) -> None:
        """Release the producer."""
        ...


__all__ = ["ProtocolRuntimeLogProducer"]
