# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeRegistrationOrchestrator handler wiring.

This registry provides a static factory method for creating handler instances
used by the NodeRegistrationOrchestrator. It follows the ONEX registry pattern
and provides a declarative mapping between event models and their handlers.

Handler Wiring (from contract.yaml):
    - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
    - ModelRuntimeTick -> HandlerRuntimeTick
    - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked
    - ModelNodeHeartbeatEvent -> HandlerNodeHeartbeat

Handler Implementation:
    All handlers implement ProtocolMessageHandler directly with:
    - handler_id, category, message_types, node_kind properties
    - handle(envelope) -> ModelHandlerOutput signature

    Handlers are registered directly with ServiceHandlerRegistry without
    adapter classes.

Handler Dependencies:
    All handlers require ProjectionReaderRegistration for state queries.
    Some handlers optionally accept:
    - ProjectorRegistration: For projection persistence
    - HandlerConsul: For Consul service registration (dual registration)

Usage:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )

    registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
        projection_reader=reader,
        projector=projector,
        consul_handler=consul_handler,
    )
    # registry is frozen and thread-safe

    # Get handler by ID
    handler = registry.get_handler_by_id("handler-node-introspected")
    result = await handler.handle(envelope)
    ```

Related Tickets:
    - OMN-1102: Make NodeRegistrationOrchestrator fully declarative
    - OMN-888 (C1): Registration Orchestrator
    - OMN-1006: Node Heartbeat for Liveness Tracking
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


def _validate_handler_protocol(handler: object) -> tuple[bool, list[str]]:
    """Validate handler implements ProtocolMessageHandler via duck typing.

    Uses duck typing to verify the handler has the required properties and
    methods for ProtocolMessageHandler compliance. Per ONEX conventions,
    protocol compliance is verified via structural typing rather than
    isinstance checks.

    Protocol Requirements (ProtocolMessageHandler):
        - handler_id (property): Unique identifier string
        - category (property): EnumMessageCategory value
        - message_types (property): set[str] of message type names
        - node_kind (property): EnumNodeKind value
        - handle (method): async def handle(envelope) -> ModelHandlerOutput

    Args:
        handler: The handler instance to validate.

    Returns:
        A tuple of (is_valid, missing_members) where:
        - is_valid: True if handler implements all required members
        - missing_members: List of member names that are missing.
          Empty list if all members are present.
    """
    missing_members: list[str] = []

    # Required properties
    if not hasattr(handler, "handler_id"):
        missing_members.append("handler_id")
    if not hasattr(handler, "category"):
        missing_members.append("category")
    if not hasattr(handler, "message_types"):
        missing_members.append("message_types")
    if not hasattr(handler, "node_kind"):
        missing_members.append("node_kind")

    # Required method - handle()
    if not callable(getattr(handler, "handle", None)):
        missing_members.append("handle")

    return (len(missing_members) == 0, missing_members)


if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )


# TODO(OMN-1316): Tech debt - subcontract should be loaded from contract.yaml
# instead of being constructed programmatically. The _create_handler_routing_subcontract()
# function in node.py already loads handler routing from contract.yaml. Consider:
# 1. Using that function here instead of duplicating handler registration
# 2. Or making the registry reference the subcontract for consistency
# See: _create_handler_routing_subcontract() in node.py for contract-driven approach.


class RegistryInfraNodeRegistrationOrchestrator:
    """Handler registry for NodeRegistrationOrchestrator.

    This registry provides a static factory method for creating handler registries
    used by the NodeRegistrationOrchestrator. It follows the ONEX registry pattern
    with the naming convention ``RegistryInfra<NodeName>``.

    Why a class instead of a function?
        ONEX registry pattern (CLAUDE.md) requires registry classes. This enables:

        - **Centralized wiring**: All handler creation logic in one place
        - **Contract alignment**: Maps event models to handlers per contract.yaml
        - **Testability**: Mock dependencies for unit testing
        - **Extensibility**: Subclassing for specialized registries

    Usage:
        ```python
        registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
            projection_reader=reader,
            projector=projector,
            consul_handler=consul_handler,
        )
        handler = registry.get_handler_by_id("handler-node-introspected")
        result = await handler.handle(envelope)
        ```
    """

    @staticmethod
    def create_registry(
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration | None = None,
        consul_handler: HandlerConsul | None = None,
        *,
        require_heartbeat_handler: bool = True,
    ) -> ServiceHandlerRegistry:
        """Create a frozen ServiceHandlerRegistry with all handlers wired.

        This is the preferred method for creating handler registries. It returns
        a thread-safe, frozen registry that can be used by the orchestrator.

        Handler Registration:
            The contract.yaml defines 4 handlers:
            - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected (always registered)
            - ModelRuntimeTick -> HandlerRuntimeTick (always registered)
            - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked (always registered)
            - ModelNodeHeartbeatEvent -> HandlerNodeHeartbeat (requires projector)

        Fail-Fast Behavior:
            By default (require_heartbeat_handler=True), this method raises
            ProtocolConfigurationError if projector is None, because the contract
            defines heartbeat routing which requires a projector for persistence.

            This fail-fast approach prevents silent failures where heartbeat events
            would be silently dropped at runtime due to missing handler registration.

        Args:
            projection_reader: Projection reader for state queries.
            projector: Projector for state persistence. Required for
                HandlerNodeHeartbeat to persist heartbeat timestamps.
            consul_handler: Optional Consul handler for service registration.
            require_heartbeat_handler: If True (default), raises ProtocolConfigurationError
                when projector is None. Set to False only for testing scenarios where
                heartbeat functionality is intentionally disabled. This creates a
                contract.yaml mismatch (4 handlers defined, only 3 registered).

        Returns:
            Frozen ServiceHandlerRegistry with handlers registered:
            - 4 handlers when projector is provided
            - 3 handlers when projector is None and require_heartbeat_handler=False

        Raises:
            ProtocolConfigurationError: If projector is None and
                require_heartbeat_handler is True (default).

        Example:
            ```python
            # Production usage - projector required
            registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=reader,
                projector=projector,
                consul_handler=consul_handler,
            )

            # Testing without heartbeat support (explicit opt-in)
            registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=reader,
                projector=None,
                require_heartbeat_handler=False,  # Explicitly disable
            )

            # Get handler by ID
            handler = registry.get_handler_by_id("handler-node-introspected")

            # Or iterate all handlers
            for handler in registry.get_handlers():
                print(f"{handler.handler_id}: {handler.message_types}")
            ```
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeHeartbeat,
            HandlerNodeIntrospected,
            HandlerNodeRegistrationAcked,
            HandlerRuntimeTick,
        )

        # Fail-fast: contract.yaml defines heartbeat routing which requires projector
        if projector is None and require_heartbeat_handler:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="create_registry",
                target_name="RegistryInfraNodeRegistrationOrchestrator",
            )
            raise ProtocolConfigurationError(
                "Heartbeat handler requires projector but none was provided. "
                "The contract.yaml defines ModelNodeHeartbeatEvent routing which "
                "requires a ProjectorRegistration instance to persist heartbeat updates. "
                "Either provide a projector or set require_heartbeat_handler=False "
                "to explicitly disable heartbeat support (testing only).",
                context=ctx,
            )

        registry = ServiceHandlerRegistry()

        # Create handlers with dependencies
        handler_introspected = HandlerNodeIntrospected(
            projection_reader=projection_reader,
            projector=projector,
            consul_handler=consul_handler,
        )
        handler_runtime_tick = HandlerRuntimeTick(
            projection_reader=projection_reader,
        )
        handler_registration_acked = HandlerNodeRegistrationAcked(
            projection_reader=projection_reader,
        )

        # Validate and register handlers (duck typing verification for ProtocolMessageHandler)
        # Handlers must implement: handler_id, category, message_types, node_kind, handle()
        handlers_to_register = [
            handler_introspected,
            handler_runtime_tick,
            handler_registration_acked,
        ]

        for handler in handlers_to_register:
            is_valid, missing = _validate_handler_protocol(handler)
            if not is_valid:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name="RegistryInfraNodeRegistrationOrchestrator",
                )
                handler_name = type(handler).__name__
                raise ProtocolConfigurationError(
                    f"Handler {handler_name} does not implement ProtocolMessageHandler. "
                    f"Missing required members: {', '.join(missing)}. "
                    f"Handlers must have: handler_id, category, message_types, node_kind properties "
                    f"and handle(envelope) method.",
                    context=ctx,
                )
            registry.register_handler(handler)

        # Heartbeat handler requires projector for persistence.
        # At this point, if projector is None, require_heartbeat_handler must be False
        # (otherwise we would have raised ProtocolConfigurationError above).
        if projector is not None:
            handler_heartbeat = HandlerNodeHeartbeat(
                projection_reader=projection_reader,
                projector=projector,
            )
            # Validate heartbeat handler before registration
            is_valid, missing = _validate_handler_protocol(handler_heartbeat)
            if not is_valid:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name="RegistryInfraNodeRegistrationOrchestrator",
                )
                raise ProtocolConfigurationError(
                    f"Handler HandlerNodeHeartbeat does not implement ProtocolMessageHandler. "
                    f"Missing required members: {', '.join(missing)}. "
                    f"Handlers must have: handler_id, category, message_types, node_kind properties "
                    f"and handle(envelope) method.",
                    context=ctx,
                )
            registry.register_handler(handler_heartbeat)
        else:
            # This branch only executes when require_heartbeat_handler=False
            # (explicit opt-in for testing without heartbeat support)
            logger.warning(
                "HandlerNodeHeartbeat NOT registered: require_heartbeat_handler=False. "
                "This creates a contract.yaml mismatch (4 handlers defined, only 3 registered). "
                "Heartbeat events (ModelNodeHeartbeatEvent) will NOT be handled. "
                "This configuration is intended for testing only."
            )

        # Freeze registry to make it thread-safe
        registry.freeze()

        return registry


__all__ = ["RegistryInfraNodeRegistrationOrchestrator"]
