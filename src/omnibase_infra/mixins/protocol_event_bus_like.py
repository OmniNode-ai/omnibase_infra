# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Protocol for Introspection.

This module provides the minimal protocol interface for event bus compatibility
with the MixinNodeIntrospection mixin.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolEventBusLike(Protocol):
    """Protocol for event bus compatibility.

    This protocol defines the minimal interface required for an event bus
    to be used with introspection. Any object implementing either
    ``publish_envelope`` or ``publish`` method is compatible.

    The mixin prefers ``publish_envelope`` when available, falling back
    to ``publish`` for raw bytes publishing.
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an event envelope to a topic.

        Args:
            envelope: The event envelope/model to publish.
            topic: The topic to publish to.
        """
        ...

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Publish raw bytes to a topic (fallback method).

        Args:
            topic: The topic to publish to.
            key: Optional message key as bytes.
            value: The message value as bytes.
        """
        ...


__all__: list[str] = ["ProtocolEventBusLike"]
