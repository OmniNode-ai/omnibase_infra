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

Handler Adapters:
    The registry uses adapter classes to bridge existing handlers to the
    ProtocolMessageHandler interface required by ServiceHandlerRegistry.
    Each adapter:
    - Implements handler_id, category, message_types, node_kind properties
    - Extracts payload, timestamp, and correlation_id from ModelEventEnvelope
    - Delegates to the inner handler's handle() method
    - Wraps results in ModelHandlerOutput

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

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from pydantic import BaseModel

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


class AdapterNodeIntrospected:
    """Adapter for HandlerNodeIntrospected to ProtocolMessageHandler interface.

    Bridges the existing handler signature:
        handle(event, now, correlation_id) -> list[BaseModel]

    To the ProtocolMessageHandler signature:
        handle(envelope) -> ModelHandlerOutput
    """

    def __init__(self, inner: HandlerNodeIntrospected) -> None:
        """Initialize adapter with inner handler.

        Args:
            inner: The HandlerNodeIntrospected instance to wrap.
        """
        self._inner = inner

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-node-introspected"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelNodeIntrospectionEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[object]:
        """Process event envelope and return handler output.

        Args:
            envelope: Event envelope containing ModelNodeIntrospectionEvent payload.

        Returns:
            ModelHandlerOutput containing emitted events.
        """
        import time

        start_time = time.perf_counter()

        # Extract from envelope
        payload = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Delegate to inner handler
        events: list[BaseModel] = await self._inner.handle(
            event=payload,  # type: ignore[arg-type]
            now=now,
            correlation_id=correlation_id,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=tuple(events),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


class AdapterRuntimeTick:
    """Adapter for HandlerRuntimeTick to ProtocolMessageHandler interface.

    Bridges the existing handler signature:
        handle(tick, now, correlation_id) -> list[BaseModel]

    To the ProtocolMessageHandler signature:
        handle(envelope) -> ModelHandlerOutput
    """

    def __init__(self, inner: HandlerRuntimeTick) -> None:
        """Initialize adapter with inner handler.

        Args:
            inner: The HandlerRuntimeTick instance to wrap.
        """
        self._inner = inner

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-runtime-tick"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelRuntimeTick"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[object]:
        """Process event envelope and return handler output.

        Args:
            envelope: Event envelope containing ModelRuntimeTick payload.

        Returns:
            ModelHandlerOutput containing emitted events.
        """
        import time

        start_time = time.perf_counter()

        # Extract from envelope
        payload = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Delegate to inner handler
        events: list[BaseModel] = await self._inner.handle(
            tick=payload,  # type: ignore[arg-type]
            now=now,
            correlation_id=correlation_id,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=tuple(events),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


class AdapterNodeRegistrationAcked:
    """Adapter for HandlerNodeRegistrationAcked to ProtocolMessageHandler interface.

    Bridges the existing handler signature:
        handle(command, now, correlation_id) -> list[BaseModel]

    To the ProtocolMessageHandler signature:
        handle(envelope) -> ModelHandlerOutput
    """

    def __init__(self, inner: HandlerNodeRegistrationAcked) -> None:
        """Initialize adapter with inner handler.

        Args:
            inner: The HandlerNodeRegistrationAcked instance to wrap.
        """
        self._inner = inner

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-node-registration-acked"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.COMMAND

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelNodeRegistrationAcked"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[object]:
        """Process event envelope and return handler output.

        Args:
            envelope: Event envelope containing ModelNodeRegistrationAcked payload.

        Returns:
            ModelHandlerOutput containing emitted events.
        """
        import time

        start_time = time.perf_counter()

        # Extract from envelope
        payload = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Delegate to inner handler
        events: list[BaseModel] = await self._inner.handle(
            command=payload,  # type: ignore[arg-type]
            now=now,
            correlation_id=correlation_id,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=tuple(events),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=processing_time_ms,
            timestamp=now,
        )


class AdapterNodeHeartbeat:
    """Adapter for HandlerNodeHeartbeat to ProtocolMessageHandler interface.

    Bridges the existing handler signature:
        handle(event, domain) -> ModelHeartbeatHandlerResult

    To the ProtocolMessageHandler signature:
        handle(envelope) -> ModelHandlerOutput

    Note:
        HandlerNodeHeartbeat has a different return type (ModelHeartbeatHandlerResult)
        than other handlers. The adapter wraps this in ModelHandlerOutput.result.
    """

    def __init__(self, inner: HandlerNodeHeartbeat) -> None:
        """Initialize adapter with inner handler.

        Args:
            inner: The HandlerNodeHeartbeat instance to wrap.
        """
        self._inner = inner

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-node-heartbeat"

    @property
    def category(self) -> EnumMessageCategory:
        """Message category this handler processes."""
        return EnumMessageCategory.EVENT

    @property
    def message_types(self) -> set[str]:
        """Set of message type names this handler can process."""
        return {"ModelNodeHeartbeatEvent"}

    @property
    def node_kind(self) -> EnumNodeKind:
        """Node kind this handler belongs to."""
        return EnumNodeKind.ORCHESTRATOR

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[object]:
        """Process event envelope and return handler output.

        Args:
            envelope: Event envelope containing ModelNodeHeartbeatEvent payload.

        Returns:
            ModelHandlerOutput with result containing ModelHeartbeatHandlerResult.
        """
        import time

        start_time = time.perf_counter()

        # Extract from envelope
        payload = envelope.payload
        now: datetime = envelope.envelope_timestamp
        correlation_id: UUID = envelope.correlation_id or uuid4()

        # Delegate to inner handler
        # Note: HandlerNodeHeartbeat takes (event, domain) not (event, now, correlation_id)
        result = await self._inner.handle(
            event=payload,  # type: ignore[arg-type]
            domain="registration",
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ModelHandlerOutput(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=self.handler_id,
            node_kind=self.node_kind,
            events=(),
            intents=(),
            projections=(),
            result=result,
            processing_time_ms=processing_time_ms,
            timestamp=now,
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

        # Create registry
        registry = ServiceHandlerRegistry()

        # Create inner handlers with dependencies
        inner_introspected = HandlerNodeIntrospected(
            projection_reader=projection_reader,
            projector=projector,
            consul_handler=consul_handler,
        )

        inner_runtime_tick = HandlerRuntimeTick(
            projection_reader=projection_reader,
        )

        inner_registration_acked = HandlerNodeRegistrationAcked(
            projection_reader=projection_reader,
        )

        # HandlerNodeHeartbeat requires projector (not optional for heartbeat updates)
        # If no projector provided, create handler anyway (it will log warnings)
        if projector is not None:
            inner_node_heartbeat = HandlerNodeHeartbeat(
                projection_reader=projection_reader,
                projector=projector,
            )
            # Wrap and register heartbeat handler
            registry.register_handler(AdapterNodeHeartbeat(inner_node_heartbeat))

        # Wrap inner handlers in adapters
        adapter_introspected = AdapterNodeIntrospected(inner_introspected)
        adapter_runtime_tick = AdapterRuntimeTick(inner_runtime_tick)
        adapter_registration_acked = AdapterNodeRegistrationAcked(
            inner_registration_acked
        )

        # Register handlers
        registry.register_handler(adapter_introspected)
        registry.register_handler(adapter_runtime_tick)
        registry.register_handler(adapter_registration_acked)

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
                # Type assertion: service registry returns the registered type
                assert isinstance(reader, ProjectionReaderRegistration)
                return reader

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
            ValueError: If no projector is configured. HandlerNodeHeartbeat
                requires a projector to persist heartbeat timestamp updates.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeHeartbeat,
        )

        if self._projector is None:
            raise ValueError(
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
            event_type = type(event).__name__
            handler = handler_map.get(event_type)
            if handler:
                result = await handler.handle(event, now, correlation_id)
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
