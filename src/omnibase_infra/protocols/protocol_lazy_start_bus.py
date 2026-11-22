# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime bus surface a lazy start attaches through (OMN-21592)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__: list[str] = ["ProtocolLazyStartBus"]


@runtime_checkable
class ProtocolLazyStartBus(Protocol):
    """The runtime bus surface a start attaches through and its node publishes on."""

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        """Attach one subscription; returns its stop handle."""
        ...

    async def publish_envelope(self, *args: object, **kwargs: object) -> object:
        """Publish one envelope."""
        ...

    async def publish(self, *args: object, **kwargs: object) -> object:
        """Publish raw bytes."""
        ...

    def get_consumer_groups(self) -> object:
        """The bus's consumer groups by topic and node."""
        ...
