# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Introspection Protocol.

This module provides the protocol interface for node introspection,
enabling typed dependency injection for auto-introspection on node startup.

The protocol is designed to work with existing infrastructure:
    - MixinNodeIntrospection provides the concrete implementation
    - RuntimeHostProcess accepts this protocol for startup introspection
    - EnumIntrospectionReason categorizes introspection events

Thread Safety:
    Implementations of ProtocolNodeIntrospection MUST be thread-safe for
    concurrent async calls. Multiple coroutines may invoke publish methods
    simultaneously during runtime lifecycle events.

Related Tickets:
    - OMN-1930: Phase 1 - Fix Auto-Introspection (P0)

See Also:
    - MixinNodeIntrospection: Primary implementation via adapter
    - ModelRuntimeIntrospectionConfig: Configuration for jitter/throttle
    - EnumIntrospectionReason: Reason codes for introspection events

.. versionadded:: 0.4.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumIntrospectionReason


@runtime_checkable
class ProtocolNodeIntrospection(Protocol):
    """Protocol for node introspection services.

    This protocol defines the interface required for auto-introspection
    functionality in RuntimeHostProcess. Implementations enable nodes to
    announce their presence on startup with configurable jitter and throttling.

    Thread Safety:
        Implementations MUST be thread-safe for concurrent async calls.

        **Guarantees implementers MUST provide:**
            - Concurrent calls to publish methods are safe
            - Heartbeat task management is thread-safe
            - Internal state is protected by appropriate locks

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    Example:
        >>> class MyIntrospectionService:
        ...     async def publish_introspection(
        ...         self,
        ...         reason: EnumIntrospectionReason,
        ...         correlation_id: UUID | None = None,
        ...     ) -> None:
        ...         # Publish introspection event to Kafka
        ...         ...
        ...
        ...     async def start_heartbeat_task(self) -> None:
        ...         # Start periodic heartbeat publishing
        ...         ...
        ...
        ...     async def stop_heartbeat_task(self) -> None:
        ...         # Stop heartbeat task gracefully
        ...         ...

    See Also:
        - MixinNodeIntrospection: Provides publish_introspection implementation
        - RuntimeHostProcess: Consumer of this protocol for startup introspection
        - ModelRuntimeIntrospectionConfig: Configuration for timing parameters
    """

    async def publish_introspection(
        self,
        reason: EnumIntrospectionReason,
        correlation_id: UUID | None = None,
    ) -> None:
        """Publish an introspection event for this node.

        Thread Safety:
            This method MUST be safe for concurrent calls from multiple
            coroutines. Implementations should not rely on call ordering.

        Args:
            reason: The reason for this introspection event (STARTUP, HEARTBEAT, etc.)
            correlation_id: Optional correlation ID for distributed tracing.
                If None, implementations should generate a new UUID.

        Raises:
            InfraUnavailableError: If the event bus is unavailable (circuit open).
            InfraConnectionError: If publishing fails due to transport error.
        """
        ...

    async def start_heartbeat_task(self) -> None:
        """Start the periodic heartbeat publishing task.

        This method starts a background task that publishes heartbeat events
        at the configured interval. Should be called after initial introspection
        on startup.

        Thread Safety:
            This method MUST be idempotent - calling it multiple times should
            not create multiple heartbeat tasks.

        Raises:
            RuntimeError: If heartbeat task is already running.
        """
        ...

    async def stop_heartbeat_task(self) -> None:
        """Stop the periodic heartbeat publishing task.

        This method gracefully stops the heartbeat background task. Should be
        called during node shutdown.

        Thread Safety:
            This method MUST be safe to call even if no task is running.
        """
        ...


__all__: list[str] = ["ProtocolNodeIntrospection"]
