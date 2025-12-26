# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Protocol for Introspection.

This module provides the minimal protocol interface for event bus compatibility
with the MixinNodeIntrospection mixin.

Concurrency Safety:
    Implementations of ProtocolEventBusLike MUST be safe for concurrent
    async access. Multiple coroutines may invoke publish methods simultaneously.

Related:
    - KafkaEventBus: Production implementation with circuit breaker integration
    - InMemoryEventBus: Simple implementation for testing
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolEventBusLike(Protocol):
    """Protocol for event bus compatibility.

    This protocol defines the minimal interface required for an event bus
    to be used with introspection and timeout emission.

    Concurrency Safety:
        Implementations MUST be safe for concurrent async access.
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an event envelope to a topic."""
        ...

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Publish raw bytes to a topic (fallback method)."""
        ...


__all__: list[str] = ["ProtocolEventBusLike"]
