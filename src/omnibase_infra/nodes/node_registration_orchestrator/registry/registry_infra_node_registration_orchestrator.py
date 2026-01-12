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

from typing import TYPE_CHECKING

from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry

from omnibase_infra.errors import ProtocolConfigurationError

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
        # TODO(tech-debt): projector and consul_handler are optional for testing
        # flexibility, but production deployments should provide these via DI.
        # Future: resolve from container if not explicitly provided (OMN-1102).
        projector: ProjectorRegistration | None = None,
        consul_handler: HandlerConsul | None = None,
    ) -> ServiceHandlerRegistry:
        """Create a frozen ServiceHandlerRegistry with all handlers wired.

        This is the preferred method for creating handler registries. It returns
        a thread-safe, frozen registry that can be used by the orchestrator.

        Args:
            projection_reader: Projection reader for state queries.
            projector: Optional projector for state persistence.
            consul_handler: Optional Consul handler for service registration.

        Returns:
            Frozen ServiceHandlerRegistry with all handlers registered.

        Example:
            ```python
            registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=reader,
                projector=projector,
                consul_handler=consul_handler,
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

        # Register handlers directly (no adapters needed - handlers implement
        # ProtocolMessageHandler with handle(envelope) -> ModelHandlerOutput)
        registry.register_handler(handler_introspected)
        registry.register_handler(handler_runtime_tick)
        registry.register_handler(handler_registration_acked)

        # Heartbeat handler requires projector for persistence
        if projector is not None:
            handler_heartbeat = HandlerNodeHeartbeat(
                projection_reader=projection_reader,
                projector=projector,
            )
            registry.register_handler(handler_heartbeat)

        # Freeze registry to make it thread-safe
        registry.freeze()

        return registry

    def _get_projection_reader(self) -> ProjectionReaderRegistration:
        """Get or create ProjectionReaderRegistration.

        Returns:
            ProjectionReaderRegistration instance.

        Note:
            If no projection_reader was provided to constructor and container
            cannot resolve one, this will raise an error at handler creation time.
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
                    raise ProtocolConfigurationError(
                        f"Expected object with get_entity_state method, got {type(reader).__name__}"
                    )
                return reader  # type: ignore[no-any-return]

        # Fallback: Create new instance (requires container to have DB config)
        from omnibase_infra.projectors import ProjectionReaderRegistration

        return ProjectionReaderRegistration(self._container)

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
            raise ProtocolConfigurationError(
                "HandlerNodeHeartbeat requires a projector for heartbeat updates. "
                "Configure the registry with a ProjectorRegistration instance."
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
