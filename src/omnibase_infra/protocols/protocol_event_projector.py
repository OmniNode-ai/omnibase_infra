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
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
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

    @property
    def is_placeholder(self) -> bool:
        """Whether this is a placeholder implementation.

        Returns:
            True if this is a placeholder that will raise NotImplementedError
            on projection methods, False for full implementations.
        """
        ...

    async def project(
        self,
        event: ModelEventEnvelope,
        correlation_id: UUID,
    ) -> ModelProjectionResult:
        """Project event to persistence store.

        Args:
            event: The event envelope to project.
            correlation_id: Correlation ID for distributed tracing. Required
                to ensure proper observability across service boundaries.

        Returns:
            Result of the projection operation.
        """
        ...

    async def get_state(
        self,
        aggregate_id: UUID,
        correlation_id: UUID,
    ) -> object | None:
        """Get current projected state for an aggregate.

        Args:
            aggregate_id: The unique identifier of the aggregate.
            correlation_id: Correlation ID for distributed tracing. Required
                to ensure proper observability across service boundaries.

        Returns:
            The current projected state, or None if no state exists.
        """
        ...


__all__ = [
    "ProtocolEventProjector",
]
