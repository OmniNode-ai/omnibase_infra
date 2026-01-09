# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event projector protocol for event-to-state projection.

Provides the protocol definition for event projectors that transform
events into persistent state projections.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_core.models.events import ModelEventEnvelope
    from omnibase_core.models.projectors import ModelProjectionResult


@runtime_checkable
class ProtocolEventProjector(Protocol):
    """Protocol for event-to-state projection.

    Defines the interface that event projectors must implement to transform
    events into persistent state projections.

    Note:
        This protocol is defined locally until omnibase_spi provides an
        official definition. Once available, this should be imported from
        omnibase_spi instead.
    """

    @property
    def projector_id(self) -> str:
        """Unique identifier for this projector."""
        ...

    @property
    def aggregate_type(self) -> str:
        """The aggregate type this projector handles."""
        ...

    @property
    def consumed_events(self) -> list[str]:
        """Event types this projector consumes."""
        ...

    async def project(
        self,
        event: ModelEventEnvelope,
    ) -> ModelProjectionResult:
        """Project event to persistence store."""
        ...

    async def get_state(
        self,
        aggregate_id: UUID,
    ) -> object | None:
        """Get current projected state for an aggregate."""
        ...


__all__ = [
    "ProtocolEventProjector",
]
