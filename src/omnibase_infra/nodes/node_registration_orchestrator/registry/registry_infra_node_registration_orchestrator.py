# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeRegistrationOrchestrator handler wiring.

This registry provides dependency injection and factory methods for creating
handler instances used by the NodeRegistrationOrchestrator. It follows the ONEX
container-based DI pattern and provides a declarative mapping between event
models and their corresponding handlers.

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

    # Create registry with ServiceHandlerRegistry (preferred)
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
from typing import TYPE_CHECKING, cast

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
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )


class RegistryInfraNodeRegistrationOrchestrator:
    """Handler registry for NodeRegistrationOrchestrator.

    Why a class instead of a function?
        ONEX registry pattern (CLAUDE.md) requires registry classes with
        the naming convention ``RegistryInfra<NodeName>``. This enables:

        - **Handler factory methods**: Create handlers with proper DI
        - **Centralized wiring**: All handler creation logic in one place
        - **Contract alignment**: Maps event models to handlers per contract.yaml
        - **Testability**: Mock dependencies for unit testing

    The registry resolves dependencies from the ONEX container and creates
    handler instances with proper dependency injection.

    Preferred Usage (static factory):
        ```python
        registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
            projection_reader=reader,
            projector=projector,
            consul_handler=consul_handler,
        )
        handler = registry.get_handler_by_id("handler-node-introspected")
        result = await handler.handle(envelope)
        ```

    Legacy Usage (instance-based):
        ```python
        container = ModelONEXContainer()
        registry = RegistryInfraNodeRegistrationOrchestrator(container)
        handler_map = registry.get_handler_map()
        ```

    Attributes:
        _container: ONEX dependency injection container.
        _projection_reader: Cached projection reader instance (shared by handlers).
        _projector: Cached projector instance (optional, for persistence).
        _consul_handler: Cached Consul handler (optional, for dual registration).
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration | None = None,
        projector: ProjectorRegistration | None = None,
        consul_handler: HandlerConsul | None = None,
    ) -> None:
        """Initialize the registry with ONEX container and optional dependencies.

        Args:
            container: ONEX dependency injection container.
            projection_reader: Optional pre-configured projection reader.
                If None, will be resolved from container when needed.
            projector: Optional pre-configured projector for persistence.
                If None, handlers will operate in read-only mode.
            consul_handler: Optional HandlerConsul for Consul service registration.
                If None, Consul registration is skipped.
        """
        self._container = container
        self._projection_reader = projection_reader
        self._projector = projector
        self._consul_handler = consul_handler

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

    def _get_projection_reader(self) -> ProjectionReaderRegistration:
        """Get ProjectionReaderRegistration from constructor or container.

        Resolution order:
            1. Return instance provided via constructor
            2. Resolve from container.service_registry
            3. Raise ProtocolConfigurationError if unavailable

        Returns:
            ProjectionReaderRegistration instance.

        Raises:
            ProtocolConfigurationError: If no ProjectionReaderRegistration is
                available from constructor or service_registry. Also raised if
                the resolved object lacks the required ``get_entity_state`` method.

        Note:
            This method does NOT create a new ProjectionReaderRegistration because
            that requires an asyncpg.Pool which cannot be obtained from the
            ModelONEXContainer. Callers must either:
            - Provide a pre-configured ProjectionReaderRegistration via constructor
            - Register one in the container's service_registry before calling
        """
        if self._projection_reader is not None:
            return self._projection_reader

        # Try to resolve from container service registry
        if self._container.service_registry is not None:
            from omnibase_infra.projectors import ProjectionReaderRegistration

            reader = self._container.service_registry.get(ProjectionReaderRegistration)
            if reader is not None:
                # Duck typing: verify required projection reader capabilities
                if not hasattr(reader, "get_entity_state"):
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="get_projection_reader",
                        target_name="RegistryInfraNodeRegistrationOrchestrator",
                    )
                    raise ProtocolConfigurationError(
                        f"Expected object with get_entity_state method, got {type(reader).__name__}",
                        context=ctx,
                    )
                # Cast verified by duck typing check above - reader has get_entity_state
                return cast(ProjectionReaderRegistration, reader)

        # Fallback: Cannot create without a pool - raise configuration error
        # The container does not directly provide an asyncpg.Pool; callers must
        # either provide a ProjectionReaderRegistration via constructor or register
        # one in the container's service_registry.
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_projection_reader",
            target_name="RegistryInfraNodeRegistrationOrchestrator",
        )
        raise ProtocolConfigurationError(
            "No ProjectionReaderRegistration available. Either provide one via "
            "constructor or register in container.service_registry. "
            "ProjectionReaderRegistration requires an asyncpg.Pool which cannot "
            "be obtained from ModelONEXContainer directly.",
            context=ctx,
        )

    def create_handler_node_introspected(self) -> HandlerNodeIntrospected:
        """Create HandlerNodeIntrospected with dependencies.

        Creates the handler for processing ModelNodeIntrospectionEvent payloads.
        This is the canonical registration trigger handler.

        Returns:
            HandlerNodeIntrospected instance with:
                - projection_reader: For state queries
                - projector: For persistence (optional)
                - consul_handler: For Consul registration (optional)
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )

        return HandlerNodeIntrospected(
            projection_reader=self._get_projection_reader(),
            projector=self._projector,
            consul_handler=self._consul_handler,
        )

    def create_handler_runtime_tick(self) -> HandlerRuntimeTick:
        """Create HandlerRuntimeTick with dependencies.

        Creates the handler for processing ModelRuntimeTick payloads.
        This handler detects timeout conditions for ack and liveness deadlines.

        Returns:
            HandlerRuntimeTick instance with projection_reader for state queries.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerRuntimeTick,
        )

        return HandlerRuntimeTick(projection_reader=self._get_projection_reader())

    def create_handler_node_registration_acked(self) -> HandlerNodeRegistrationAcked:
        """Create HandlerNodeRegistrationAcked with dependencies.

        Creates the handler for processing ModelNodeRegistrationAcked commands.
        This handler processes acknowledgment commands from nodes.

        Returns:
            HandlerNodeRegistrationAcked instance with:
                - projection_reader: For state queries
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeRegistrationAcked,
        )

        return HandlerNodeRegistrationAcked(
            projection_reader=self._get_projection_reader(),
        )

    def create_handler_node_heartbeat(self) -> HandlerNodeHeartbeat:
        """Create HandlerNodeHeartbeat with dependencies.

        Creates the handler for processing ModelNodeHeartbeatEvent payloads.
        This handler updates liveness tracking for active nodes.

        Returns:
            HandlerNodeHeartbeat instance with:
                - projection_reader: For state queries
                - projector: For persistence (required for heartbeat updates)

        Raises:
            ProtocolConfigurationError: If no projector is configured. HandlerNodeHeartbeat
                requires a projector to persist heartbeat timestamp updates.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeHeartbeat,
        )

        if self._projector is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="create_handler_node_heartbeat",
                target_name="RegistryInfraNodeRegistrationOrchestrator",
            )
            raise ProtocolConfigurationError(
                "HandlerNodeHeartbeat requires a projector for heartbeat updates. "
                "Configure the registry with a ProjectorRegistration instance.",
                context=ctx,
            )

        return HandlerNodeHeartbeat(
            projection_reader=self._get_projection_reader(),
            projector=self._projector,
        )

    def get_handler_map(
        self,
    ) -> dict[
        str,
        HandlerNodeIntrospected
        | HandlerRuntimeTick
        | HandlerNodeRegistrationAcked
        | HandlerNodeHeartbeat,
    ]:
        """Get mapping of event model names to handler instances.

        Creates all handlers and returns a dictionary mapping event model
        class names to their corresponding handler instances. This map
        aligns with the handler_routing section in contract.yaml.

        Returns:
            Dictionary mapping event model names to handler instances:
                - "ModelNodeIntrospectionEvent" -> HandlerNodeIntrospected
                - "ModelRuntimeTick" -> HandlerRuntimeTick
                - "ModelNodeRegistrationAcked" -> HandlerNodeRegistrationAcked
                - "ModelNodeHeartbeatEvent" -> HandlerNodeHeartbeat (if projector configured)

        Note:
            Handler instances are created fresh on each call. This is intentional:

            1. **Stateless handlers**: These handlers are designed to be stateless;
               creating fresh instances ensures no accumulated state between calls.
            2. **Fresh dependencies**: Dependencies (projection_reader, projector) are
               resolved at creation time, ensuring current configuration is used.
            3. **No stale caching**: Avoids potential issues with cached handlers
               holding references to closed connections or outdated state.

            For production use, prefer the static ``create_registry()`` method which
            returns a frozen ``ServiceHandlerRegistry`` with cached handler instances.

            HandlerNodeHeartbeat is only included if a projector is configured,
            as it requires a projector to persist heartbeat timestamp updates.

        Example:
            ```python
            registry = RegistryInfraNodeRegistrationOrchestrator(container)
            handler_map = registry.get_handler_map()

            # Route event to handler
            event_type = type(envelope.payload).__name__
            handler = handler_map.get(event_type)
            if handler:
                result = await handler.handle(envelope)
            ```
        """
        handler_map: dict[
            str,
            HandlerNodeIntrospected
            | HandlerRuntimeTick
            | HandlerNodeRegistrationAcked
            | HandlerNodeHeartbeat,
        ] = {
            "ModelNodeIntrospectionEvent": self.create_handler_node_introspected(),
            "ModelRuntimeTick": self.create_handler_runtime_tick(),
            "ModelNodeRegistrationAcked": self.create_handler_node_registration_acked(),
        }

        # HandlerNodeHeartbeat requires projector - only add if configured
        if self._projector is not None:
            handler_map["ModelNodeHeartbeatEvent"] = (
                self.create_handler_node_heartbeat()
            )

        return handler_map


__all__ = ["RegistryInfraNodeRegistrationOrchestrator"]
