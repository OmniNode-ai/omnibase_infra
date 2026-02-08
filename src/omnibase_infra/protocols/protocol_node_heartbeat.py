# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Heartbeat Handler Protocol.

This module provides the protocol interface for node heartbeat handling,
enabling protocol-based dependency injection for HandlerNodeHeartbeat in
the DI container.

The protocol captures the minimal interface required for DI resolution:
    - handle(): Process a heartbeat event envelope
    - handler_id: Unique identifier for the handler
    - category: Message category (EVENT)
    - message_types: Set of handled message type names
    - node_kind: Node archetype (ORCHESTRATOR)

Architecture:
    This protocol enables structural typing for the heartbeat handler,
    allowing the DI container to register and resolve the handler by
    protocol rather than concrete class. This decouples consumers from
    the implementation.

Thread Safety:
    Implementations of ProtocolNodeHeartbeat MUST be coroutine-safe.
    The handler is stateless and uses connection pools for database access.

Related Tickets:
    - OMN-1990: Finish infra integration, wire remaining registry workflow
    - OMN-1006: Add last_heartbeat_at for liveness expired event reporting

See Also:
    - HandlerNodeHeartbeat: Primary concrete implementation
    - DispatcherNodeHeartbeat: Dispatcher adapter that wraps this protocol
    - wiring.py: Registration and resolution via DI container

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
    from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.models.registration import ModelNodeHeartbeatEvent


@runtime_checkable
class ProtocolNodeHeartbeat(Protocol):
    """Protocol for node heartbeat handler services.

    This protocol defines the interface required for heartbeat handling
    in the registration orchestrator. Implementations process
    ModelNodeHeartbeatEvent envelopes and update the registration
    projection with heartbeat timestamps and liveness deadlines.

    Thread Safety:
        Implementations MUST be coroutine-safe for concurrent async calls.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    Example:
        >>> class MyHeartbeatHandler:
        ...     @property
        ...     def handler_id(self) -> str:
        ...         return "handler-node-heartbeat"
        ...
        ...     @property
        ...     def category(self) -> EnumMessageCategory:
        ...         return EnumMessageCategory.EVENT
        ...
        ...     @property
        ...     def message_types(self) -> set[str]:
        ...         return {"ModelNodeHeartbeatEvent"}
        ...
        ...     @property
        ...     def node_kind(self) -> EnumNodeKind:
        ...         return EnumNodeKind.ORCHESTRATOR
        ...
        ...     async def handle(
        ...         self,
        ...         envelope: ModelEventEnvelope[ModelNodeHeartbeatEvent],
        ...     ) -> ModelHandlerOutput[object]:
        ...         ...

    See Also:
        - HandlerNodeHeartbeat: Concrete implementation
        - wiring.py: DI registration using this protocol
    """

    @property
    def handler_id(self) -> str:
        """Return unique identifier for this handler."""
        ...

    @property
    def category(self) -> EnumMessageCategory:
        """Return the message category this handler processes."""
        ...

    @property
    def message_types(self) -> set[str]:
        """Return the set of message types this handler processes."""
        ...

    @property
    def node_kind(self) -> EnumNodeKind:
        """Return the node kind this handler belongs to."""
        ...

    async def handle(
        self,
        envelope: ModelEventEnvelope[ModelNodeHeartbeatEvent],
    ) -> ModelHandlerOutput[object]:
        """Process a node heartbeat event.

        Args:
            envelope: Event envelope containing the heartbeat event payload.

        Returns:
            ModelHandlerOutput with processing results.
        """
        ...


__all__: list[str] = ["ProtocolNodeHeartbeat"]
