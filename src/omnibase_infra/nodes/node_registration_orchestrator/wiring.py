# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration domain wiring for MessageDispatchEngine integration.

This module provides domain-specific wiring functions for the Registration
orchestrator, enabling dispatchers to be registered with MessageDispatchEngine.

The wiring follows the domain-driven design principle where Registration-specific
code (dispatchers, route IDs, handlers) lives in the Registration domain rather
than the generic runtime layer.

Design Pattern:
    The container_wiring.py module in runtime/ delegates to this domain wiring
    module for Registration-specific wiring. This keeps the generic runtime
    layer clean while allowing domain-specific customization.

    ```python
    # In container_wiring.py (generic runtime)
    from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
        wire_registration_dispatchers,
    )

    # Delegation pattern - no Registration-specific logic in runtime
    result = await wire_registration_dispatchers(container, engine)
    ```

Route ID Constants:
    This module defines Registration-specific route IDs used for topic-based
    routing in the MessageDispatchEngine:
    - ROUTE_ID_NODE_INTROSPECTION: route.registration.node-introspection
    - ROUTE_ID_RUNTIME_TICK: route.registration.runtime-tick
    - ROUTE_ID_NODE_REGISTRATION_ACKED: route.registration.node-registration-acked

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
    - OMN-934: Message Dispatch Engine
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict, cast
from uuid import UUID

from omnibase_core.enums import EnumInjectionScope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ContainerValidationError,
    ContainerWiringError,
    ServiceResolutionError,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class WiringResult(TypedDict):
    """Result of wire_registration_handlers operation.

    This TypedDict provides precise typing for the return value,
    eliminating the need for type narrowing in callers.
    """

    services: list[str]
    status: str


if TYPE_CHECKING:
    import asyncpg

    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_consul import HandlerConsul
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import MessageDispatchEngine, ProjectorShell

logger = logging.getLogger(__name__)

# =============================================================================
# Registration Domain Route IDs
# =============================================================================
# These route IDs are Registration-specific and belong in this domain module
# rather than the generic runtime layer.

ROUTE_ID_NODE_INTROSPECTION = "route.registration.node-introspection"
"""Route ID for node introspection events.

Topic pattern: *.node.introspection.events.*
Message type: ModelNodeIntrospectionEvent
Category: EVENT
"""

ROUTE_ID_RUNTIME_TICK = "route.registration.runtime-tick"
"""Route ID for runtime tick events.

Topic pattern: *.runtime.tick.events.*
Message type: ModelRuntimeTick
Category: EVENT
"""

ROUTE_ID_NODE_REGISTRATION_ACKED = "route.registration.node-registration-acked"
"""Route ID for node registration ack commands.

Topic pattern: *.node.registration.commands.*
Message type: ModelNodeRegistrationAcked
Category: COMMAND
"""


def _validate_service_registry(
    container: ModelONEXContainer,
    operation: str,
) -> None:
    """Validate that container.service_registry is not None.

    This validation should be called before any operation that uses
    container.service_registry to provide clear error messages when
    the service registry is unavailable.

    Note:
        This is a local copy of the validation function to avoid circular
        imports between this module and container_wiring.py.

    Args:
        container: The ONEX container to validate.
        operation: Description of the operation being attempted.

    Raises:
        ServiceRegistryUnavailableError: If service_registry is None.
    """
    # Import here to avoid circular import at module level
    from omnibase_infra.errors import ServiceRegistryUnavailableError

    if not hasattr(container, "service_registry"):
        raise ServiceRegistryUnavailableError(
            "Container missing 'service_registry' attribute",
            operation=operation,
            hint=(
                "Expected ModelONEXContainer from omnibase_core. "
                "Check that omnibase_core is properly installed."
            ),
        )

    if container.service_registry is None:
        raise ServiceRegistryUnavailableError(
            "Container service_registry is None",
            operation=operation,
            hint=(
                "ModelONEXContainer.service_registry returns None when:\n"
                "  1. enable_service_registry=False was passed to constructor\n"
                "  2. ServiceRegistry module is not available/installed\n"
                "  3. Container initialization encountered an import error\n"
                "Check container logs for 'ServiceRegistry not available' warnings."
            ),
        )


async def wire_registration_dispatchers(
    container: ModelONEXContainer,
    engine: MessageDispatchEngine,
    correlation_id: UUID | None = None,
) -> dict[str, list[str] | str]:
    """Wire registration dispatchers into MessageDispatchEngine.

    Creates dispatcher adapters for the registration handlers and registers
    them with the MessageDispatchEngine. This enables the engine to route
    introspection events to the appropriate handlers.

    Prerequisites:
        - wire_registration_handlers() must be called first to register
          the underlying handlers in the container.
        - MessageDispatchEngine must not be frozen yet. If the engine is already
          frozen, dispatcher registration will fail with a RuntimeError from the
          engine's register_dispatcher() method.

    Args:
        container: ONEX container with registered handlers.
        engine: MessageDispatchEngine instance to register dispatchers with.
        correlation_id: Optional correlation ID for error tracking. If not provided,
            one will be auto-generated when errors are raised.

    Returns:
        Summary dict with diagnostic information:
            - dispatchers: List of registered dispatcher IDs (e.g.,
              ['dispatcher.node-introspected', 'dispatcher.runtime-tick',
               'dispatcher.node-registration-acked'])
            - routes: List of registered route IDs (e.g.,
              ['route.registration.node-introspection', 'route.registration.runtime-tick',
               'route.registration.node-registration-acked'])
            - status: Always "success" (errors raise exceptions)

        This diagnostic output can be logged or used to verify correct wiring.

    Raises:
        ServiceRegistryUnavailableError: If service_registry is missing or None.
        ContainerWiringError: If required handlers are not registered in the container,
            or if the engine is already frozen (cannot register new dispatchers).

    Engine Frozen Behavior:
        If engine.freeze() has been called before this function, the engine
        will reject new dispatcher registrations. Ensure this function is called
        during the wiring phase before engine.freeze() is invoked.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> from omnibase_infra.runtime import MessageDispatchEngine
        >>> import asyncpg
        >>>
        >>> container = ModelONEXContainer()
        >>> pool = await asyncpg.create_pool(dsn)
        >>> await wire_registration_handlers(container, pool)
        >>>
        >>> engine = MessageDispatchEngine()
        >>> summary = await wire_registration_dispatchers(container, engine)
        >>> print(summary)
        {'dispatchers': [...], 'routes': [...]}
        >>> engine.freeze()  # Must freeze after wiring
    """
    # Validate service_registry is available and has required methods.
    # NOTE: Validation is done BEFORE imports for fail-fast behavior - no point loading
    # heavy infrastructure modules if service_registry is unavailable.
    _validate_service_registry(container, "wire_registration_dispatchers")

    # Deferred imports: These imports are placed inside the function to avoid circular
    # import issues and to delay loading dispatcher infrastructure until this function
    # is actually called.
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
        DispatcherNodeIntrospected,
        DispatcherNodeRegistrationAcked,
        DispatcherRuntimeTick,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )

    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []

    try:
        # 1. Resolve handlers from container
        handler_introspected: HandlerNodeIntrospected = (
            await container.service_registry.resolve_service(HandlerNodeIntrospected)
        )
        handler_runtime_tick: HandlerRuntimeTick = (
            await container.service_registry.resolve_service(HandlerRuntimeTick)
        )
        handler_acked: HandlerNodeRegistrationAcked = (
            await container.service_registry.resolve_service(
                HandlerNodeRegistrationAcked
            )
        )

        # 2. Create dispatcher adapters
        dispatcher_introspected = DispatcherNodeIntrospected(handler_introspected)
        dispatcher_runtime_tick = DispatcherRuntimeTick(handler_runtime_tick)
        dispatcher_acked = DispatcherNodeRegistrationAcked(handler_acked)

        # 3. Register dispatchers with engine
        # Note: Using the function-based API rather than protocol-based API
        # because MessageDispatchEngine.register_dispatcher() takes a callable

        # 3a. Register DispatcherNodeIntrospected
        # Note: node_kind is NOT passed to register_dispatcher because the dispatcher's
        # handle() method doesn't accept ModelDispatchContext - it handles time injection
        # internally. The node_kind property is informational only.
        engine.register_dispatcher(
            dispatcher_id=dispatcher_introspected.dispatcher_id,
            dispatcher=dispatcher_introspected.handle,
            category=dispatcher_introspected.category,
            message_types=dispatcher_introspected.message_types,
        )
        dispatchers_registered.append(dispatcher_introspected.dispatcher_id)

        # 3b. Register DispatcherRuntimeTick
        engine.register_dispatcher(
            dispatcher_id=dispatcher_runtime_tick.dispatcher_id,
            dispatcher=dispatcher_runtime_tick.handle,
            category=dispatcher_runtime_tick.category,
            message_types=dispatcher_runtime_tick.message_types,
        )
        dispatchers_registered.append(dispatcher_runtime_tick.dispatcher_id)

        # 3c. Register DispatcherNodeRegistrationAcked
        engine.register_dispatcher(
            dispatcher_id=dispatcher_acked.dispatcher_id,
            dispatcher=dispatcher_acked.handle,
            category=dispatcher_acked.category,
            message_types=dispatcher_acked.message_types,
        )
        dispatchers_registered.append(dispatcher_acked.dispatcher_id)

        # 4. Register routes for topic-based routing
        # 4a. Route for introspection events
        route_introspection = ModelDispatchRoute(
            route_id=ROUTE_ID_NODE_INTROSPECTION,
            topic_pattern="*.node.introspection.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_introspected.dispatcher_id,
            message_type="ModelNodeIntrospectionEvent",
        )
        engine.register_route(route_introspection)
        routes_registered.append(route_introspection.route_id)

        # 4b. Route for runtime tick events
        route_runtime_tick = ModelDispatchRoute(
            route_id=ROUTE_ID_RUNTIME_TICK,
            topic_pattern="*.runtime.tick.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_runtime_tick.dispatcher_id,
            message_type="ModelRuntimeTick",
        )
        engine.register_route(route_runtime_tick)
        routes_registered.append(route_runtime_tick.route_id)

        # 4c. Route for registration ack commands
        route_acked = ModelDispatchRoute(
            route_id=ROUTE_ID_NODE_REGISTRATION_ACKED,
            topic_pattern="*.node.registration.commands.*",
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id=dispatcher_acked.dispatcher_id,
            message_type="ModelNodeRegistrationAcked",
        )
        engine.register_route(route_acked)
        routes_registered.append(route_acked.route_id)

        logger.info(
            "Registration dispatchers wired successfully",
            extra={
                "dispatcher_count": len(dispatchers_registered),
                "dispatchers": dispatchers_registered,
                "route_count": len(routes_registered),
                "routes": routes_registered,
            },
        )

    except Exception as e:
        logger.exception(
            "Failed to wire registration dispatchers",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="wire_registration_dispatchers",
        )
        raise ContainerWiringError(
            f"Failed to wire registration dispatchers: {e}\n"
            f"Fix: Ensure wire_registration_handlers(container, pool) "
            f"was called first.",
            context=context,
        ) from e

    return {
        "dispatchers": dispatchers_registered,
        "routes": routes_registered,
        "status": "success",
    }


# =============================================================================
# Handler Wiring (OMN-1346)
# =============================================================================


async def wire_registration_handlers(
    container: ModelONEXContainer,
    pool: asyncpg.Pool,
    liveness_interval_seconds: int | None = None,
    projector: ProjectorShell | None = None,
    consul_handler: HandlerConsul | None = None,
    correlation_id: UUID | None = None,
) -> WiringResult:
    """Register registration orchestrator handlers with the container.

    Registers ProjectionReaderRegistration and the three registration handlers:
    - HandlerNodeIntrospected
    - HandlerRuntimeTick
    - HandlerNodeRegistrationAcked

    All handlers depend on ProjectionReaderRegistration, which is registered first.

    Args:
        container: ONEX container instance to register services in.
        pool: asyncpg connection pool for database access.
        liveness_interval_seconds: Liveness deadline interval for ack handler.
            If None, uses ONEX_LIVENESS_INTERVAL_SECONDS env var or default (60s).
        projector: Optional ProjectorShell for persisting state transitions.
        consul_handler: Optional HandlerConsul for dual registration with Consul.
        correlation_id: Optional correlation ID for error tracking. If not provided,
            one will be auto-generated when errors are raised.

    Returns:
        WiringResult TypedDict with:
            - services: List of registered service names
            - status: Always "success" (errors raise exceptions)

    Raises:
        ServiceRegistryUnavailableError: If service_registry is missing or None.
        ContainerValidationError: If container missing required service_registry API.
        ContainerWiringError: If service registration fails.

    Note:
        Services are registered with scope=EnumInjectionScope.GLOBAL and may conflict if multiple
        plugins register the same interface type. This is acceptable for the
        Registration domain as these handlers are singletons by design. If you
        need to register multiple implementations of the same interface, use
        domain-specific interface types or scoped registrations to ensure isolation.
    """
    _validate_service_registry(container, "wire_registration_handlers")

    from omnibase_core.models.primitives import ModelSemVer
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
        get_liveness_interval_seconds,
    )
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime.projector_shell import ProjectorShell

    semver_default = ModelSemVer.parse("1.0.0")
    resolved_liveness_interval = get_liveness_interval_seconds(
        liveness_interval_seconds
    )
    services_registered: list[str] = []

    try:
        projection_reader = ProjectionReaderRegistration(pool)
        await container.service_registry.register_instance(
            interface=ProjectionReaderRegistration,
            instance=projection_reader,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Registration projection reader",
                "version": str(semver_default),
            },
        )
        services_registered.append("ProjectionReaderRegistration")
        logger.debug("Registered ProjectionReaderRegistration in container")

        if projector is not None:
            await container.service_registry.register_instance(
                interface=ProjectorShell,
                instance=projector,
                scope=EnumInjectionScope.GLOBAL,
                metadata={
                    "description": "Registration projector",
                    "version": str(semver_default),
                },
            )
            services_registered.append("ProjectorShell")
            logger.debug("Registered ProjectorShell in container")

        handler_introspected = HandlerNodeIntrospected(
            projection_reader,
            projector=projector,
            consul_handler=consul_handler,
        )
        await container.service_registry.register_instance(
            interface=HandlerNodeIntrospected,
            instance=handler_introspected,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Handler for NodeIntrospectionEvent",
                "version": str(semver_default),
                "has_projector": projector is not None,
                "has_consul_handler": consul_handler is not None,
            },
        )
        services_registered.append("HandlerNodeIntrospected")
        logger.debug("Registered HandlerNodeIntrospected in container")

        handler_runtime_tick = HandlerRuntimeTick(projection_reader)
        await container.service_registry.register_instance(
            interface=HandlerRuntimeTick,
            instance=handler_runtime_tick,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Handler for RuntimeTick",
                "version": str(semver_default),
            },
        )
        services_registered.append("HandlerRuntimeTick")
        logger.debug("Registered HandlerRuntimeTick in container")

        handler_acked = HandlerNodeRegistrationAcked(
            projection_reader,
            liveness_interval_seconds=resolved_liveness_interval,
        )
        await container.service_registry.register_instance(
            interface=HandlerNodeRegistrationAcked,
            instance=handler_acked,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": "Handler for NodeRegistrationAcked",
                "version": str(semver_default),
                "liveness_interval_seconds": resolved_liveness_interval,
            },
        )
        services_registered.append("HandlerNodeRegistrationAcked")
        logger.debug("Registered HandlerNodeRegistrationAcked in container")

    except AttributeError as e:
        error_str = str(e)
        hint = (
            "Container.service_registry missing 'register_instance' method."
            if "register_instance" in error_str
            else f"Missing attribute: {e}"
        )
        logger.exception("Failed to register handlers", extra={"hint": hint})
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="wire_registration_handlers",
        )
        raise ContainerValidationError(
            f"Handler wiring failed - {hint}\nOriginal: {e}",
            context=context,
            missing_attribute="register_instance"
            if "register_instance" in error_str
            else str(e),
        ) from e
    except Exception as e:
        logger.exception("Failed to register handlers", extra={"error": str(e)})
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="wire_registration_handlers",
        )
        raise ContainerWiringError(
            f"Failed to wire registration handlers: {e}",
            context=context,
        ) from e

    logger.info(
        "Registration handlers wired successfully",
        extra={
            "service_count": len(services_registered),
            "services": services_registered,
        },
    )
    return {"services": services_registered, "status": "success"}


# =============================================================================
# Handler Getters (OMN-1346)
# =============================================================================


async def get_projection_reader_from_container(
    container: ModelONEXContainer,
    correlation_id: UUID | None = None,
) -> ProjectionReaderRegistration:
    """Get ProjectionReaderRegistration from container.

    Args:
        container: ONEX container with registered services.
        correlation_id: Optional correlation ID for error tracking.

    Returns:
        ProjectionReaderRegistration instance from container.

    Raises:
        ServiceResolutionError: If service is not registered.
    """
    from omnibase_infra.projectors import ProjectionReaderRegistration

    _validate_service_registry(container, "resolve ProjectionReaderRegistration")
    try:
        return cast(
            "ProjectionReaderRegistration",
            await container.service_registry.resolve_service(
                ProjectionReaderRegistration
            ),
        )
    except Exception as e:
        logger.exception("Failed to resolve ProjectionReaderRegistration")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_ProjectionReaderRegistration",
        )
        raise ServiceResolutionError(
            f"ProjectionReaderRegistration not registered. "
            f"Call wire_registration_handlers first. Error: {e}",
            service_name="ProjectionReaderRegistration",
            context=context,
        ) from e


async def get_handler_node_introspected_from_container(
    container: ModelONEXContainer,
    correlation_id: UUID | None = None,
) -> HandlerNodeIntrospected:
    """Get HandlerNodeIntrospected from container.

    Args:
        container: ONEX container with registered services.
        correlation_id: Optional correlation ID for error tracking.

    Returns:
        HandlerNodeIntrospected instance from container.

    Raises:
        ServiceResolutionError: If service is not registered.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
    )

    _validate_service_registry(container, "resolve HandlerNodeIntrospected")
    try:
        return cast(
            "HandlerNodeIntrospected",
            await container.service_registry.resolve_service(HandlerNodeIntrospected),
        )
    except Exception as e:
        logger.exception("Failed to resolve HandlerNodeIntrospected")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_HandlerNodeIntrospected",
        )
        raise ServiceResolutionError(
            f"HandlerNodeIntrospected not registered. "
            f"Call wire_registration_handlers first. Error: {e}",
            service_name="HandlerNodeIntrospected",
            context=context,
        ) from e


async def get_handler_runtime_tick_from_container(
    container: ModelONEXContainer,
    correlation_id: UUID | None = None,
) -> HandlerRuntimeTick:
    """Get HandlerRuntimeTick from container.

    Args:
        container: ONEX container with registered services.
        correlation_id: Optional correlation ID for error tracking.

    Returns:
        HandlerRuntimeTick instance from container.

    Raises:
        ServiceResolutionError: If service is not registered.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerRuntimeTick,
    )

    _validate_service_registry(container, "resolve HandlerRuntimeTick")
    try:
        return cast(
            "HandlerRuntimeTick",
            await container.service_registry.resolve_service(HandlerRuntimeTick),
        )
    except Exception as e:
        logger.exception("Failed to resolve HandlerRuntimeTick")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_HandlerRuntimeTick",
        )
        raise ServiceResolutionError(
            f"HandlerRuntimeTick not registered. "
            f"Call wire_registration_handlers first. Error: {e}",
            service_name="HandlerRuntimeTick",
            context=context,
        ) from e


async def get_handler_node_registration_acked_from_container(
    container: ModelONEXContainer,
    correlation_id: UUID | None = None,
) -> HandlerNodeRegistrationAcked:
    """Get HandlerNodeRegistrationAcked from container.

    Args:
        container: ONEX container with registered services.
        correlation_id: Optional correlation ID for error tracking.

    Returns:
        HandlerNodeRegistrationAcked instance from container.

    Raises:
        ServiceResolutionError: If service is not registered.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeRegistrationAcked,
    )

    _validate_service_registry(container, "resolve HandlerNodeRegistrationAcked")
    try:
        return cast(
            "HandlerNodeRegistrationAcked",
            await container.service_registry.resolve_service(
                HandlerNodeRegistrationAcked
            ),
        )
    except Exception as e:
        logger.exception("Failed to resolve HandlerNodeRegistrationAcked")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_HandlerNodeRegistrationAcked",
        )
        raise ServiceResolutionError(
            f"HandlerNodeRegistrationAcked not registered. "
            f"Call wire_registration_handlers first. Error: {e}",
            service_name="HandlerNodeRegistrationAcked",
            context=context,
        ) from e


__all__: list[str] = [
    # Route ID constants
    "ROUTE_ID_NODE_INTROSPECTION",
    "ROUTE_ID_NODE_REGISTRATION_ACKED",
    "ROUTE_ID_RUNTIME_TICK",
    # Dispatcher wiring
    "wire_registration_dispatchers",
    # Handler wiring (OMN-1346)
    "wire_registration_handlers",
    "WiringResult",
    # Handler getters (OMN-1346)
    "get_projection_reader_from_container",
    "get_handler_node_introspected_from_container",
    "get_handler_runtime_tick_from_container",
    "get_handler_node_registration_acked_from_container",
]
