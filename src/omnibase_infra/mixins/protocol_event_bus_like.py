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
    to be used with introspection. Any object with a ``publish_envelope``
    method is compatible.
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an event envelope to a topic."""
        ...


__all__: list[str] = ["ProtocolEventBusLike"]
