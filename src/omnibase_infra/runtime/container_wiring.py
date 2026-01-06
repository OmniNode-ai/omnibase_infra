# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Container wiring for omnibase_infra services.

This module provides functions to register infrastructure services
with ModelONEXContainer from omnibase_core. It establishes container-based
dependency injection for PolicyRegistry and other infrastructure components.

Design Principles:
- Explicit registration: All services registered explicitly
- Singleton per container: Each container gets its own service instances
- Type-safe resolution: Services registered with proper type interfaces
- Testability: Easy to mock services via container

Service Keys:
- PolicyRegistry: Registered as interface=PolicyRegistry
- ProtocolBindingRegistry: Registered as interface=ProtocolBindingRegistry
- RegistryCompute: Registered as interface=RegistryCompute

Example Usage:
    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
    from omnibase_infra.runtime.policy_registry import PolicyRegistry

    # Bootstrap container
    container = ModelONEXContainer()

    # Wire infrastructure services
    summary = await wire_infrastructure_services(container)
    print(f"Registered {len(summary['services'])} services")

    # Resolve services using type interface
    policy_registry = await container.service_registry.resolve_service(PolicyRegistry)

    # Use the registry
    policy_registry.register_policy(
        policy_id="exponential_backoff",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
    )
    ```

Integration Notes:
- Uses ModelONEXContainer.service_registry for registration
- Services registered as global scope (singleton per container)
- Type-safe resolution via interface types
- Compatible with omnibase_core v0.5.6 and later (async service registry)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.registry_compute import RegistryCompute

# Default semantic version for infrastructure components (from omnibase_core)
SEMVER_DEFAULT = ModelSemVer.parse("1.0.0")

if TYPE_CHECKING:
    import asyncpg
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.handlers import ConsulHandler
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

from omnibase_infra.errors import ServiceRegistryUnavailableError

logger = logging.getLogger(__name__)


def _validate_service_registry(
    container: ModelONEXContainer,
    operation: str,
) -> None:
    """Validate that container.service_registry is not None.

    This validation should be called before any operation that uses
    container.service_registry to provide clear error messages when
    the service registry is unavailable.

    Args:
        container: The ONEX container to validate.
        operation: Description of the operation being attempted.

    Raises:
        ServiceRegistryUnavailableError: If service_registry is None.

    Example:
        >>> _validate_service_registry(container, "register PolicyRegistry")
        >>> # Proceed with registration...
    """
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


def _analyze_attribute_error(error_str: str) -> tuple[str, str]:
    """Analyze AttributeError and return (missing_attribute, hint).

    Extracts the missing attribute name from the error string and provides
    a user-friendly hint for common container API issues.

    Note: service_registry missing/None cases are handled by _validate_service_registry()
    which is called before operations. This function handles other AttributeErrors
    (e.g., missing register_instance method).

    Args:
        error_str: The string representation of the AttributeError.

    Returns:
        Tuple of (missing_attribute, hint) for error context.
    """
    missing_attr = error_str.split("'")[-2] if "'" in error_str else "unknown"

    if "register_instance" in error_str:
        hint = (
            "Container.service_registry missing 'register_instance' method. "
            "Check omnibase_core version compatibility (requires v0.5.6 or later)."
        )
    else:
        hint = f"Missing attribute: '{missing_attr}'"

    return missing_attr, hint


def _analyze_type_error(error_str: str) -> tuple[str, str]:
    """Analyze TypeError and return (invalid_argument, hint).

    Extracts which argument caused the type error and provides
    a user-friendly hint for fixing registration issues.

    Args:
        error_str: The string representation of the TypeError.

    Returns:
        Tuple of (invalid_argument, hint) for error context.
    """
    if "interface" in error_str:
        return "interface", (
            "Invalid 'interface' argument. "
            "Expected a type class (e.g., PolicyRegistry), not an instance."
        )
    if "instance" in error_str:
        return "instance", (
            "Invalid 'instance' argument. Expected an instance of the interface type."
        )
    if "scope" in error_str:
        return "scope", (
            "Invalid 'scope' argument. Expected 'global', 'request', or 'transient'."
        )
    if "metadata" in error_str:
        return "metadata", "Invalid 'metadata' argument. Expected dict[str, object]."
    if "positional" in error_str or "argument" in error_str:
        return "signature", (
            "Argument count mismatch. "
            "Check register_instance() signature compatibility with omnibase_core version."
        )
    return "unknown", "Check register_instance() signature compatibility."


async def wire_infrastructure_services(
    container: ModelONEXContainer,
) -> dict[str, list[str] | str]:
    """Register infrastructure services with the container.

    Registers PolicyRegistry, ProtocolBindingRegistry, and RegistryCompute as global
    singleton services in the container. Uses ModelONEXContainer.service_registry.register_instance()
    with the respective class as the interface type.

    Note: This function is async because ModelONEXContainer.service_registry.register_instance()
    is async in omnibase_core v0.5.6 and later (see omnibase_core.container.ModelONEXContainer).

    If the container's service_registry is None (e.g., in omnibase_core 0.6.x when not
    configured), this function will log a warning and return early with status="skipped".

    Args:
        container: ONEX container instance to register services in.

    Returns:
        Summary dict with:
            - services: List of registered service class names
            - status: "skipped" if service_registry was unavailable

    Raises:
        RuntimeError: If service registration fails

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> summary = await wire_infrastructure_services(container)
        >>> print(summary)
        {'services': ['PolicyRegistry', 'ProtocolBindingRegistry', 'RegistryCompute']}
        >>> policy_reg = await container.service_registry.resolve_service(PolicyRegistry)
        >>> handler_reg = await container.service_registry.resolve_service(ProtocolBindingRegistry)
        >>> compute_reg = await container.service_registry.resolve_service(RegistryCompute)
        >>> # Verify via duck typing (per ONEX conventions)
        >>> hasattr(policy_reg, 'register_policy') and callable(policy_reg.register_policy)
        True
        >>> hasattr(handler_reg, 'register') and callable(handler_reg.register)
        True
        >>> hasattr(compute_reg, 'register_plugin') and callable(compute_reg.register_plugin)
        True
    """
    # Check if service_registry is available (may be None in omnibase_core 0.6.x)
    if container.service_registry is None:
        logger.warning(
            "wire_infrastructure_services: service_registry is None, "
            "skipping service registration"
        )
        return {"services": [], "status": "skipped"}

    # Validate service_registry has required methods
    _validate_service_registry(container, "wire_infrastructure_services")

    services_registered: list[str] = []

    try:
        # Create PolicyRegistry instance
        policy_registry = PolicyRegistry()

        # Register with container using type interface (global scope = singleton)
        await container.service_registry.register_instance(
            interface=PolicyRegistry,
            instance=policy_registry,
            scope="global",
            metadata={
                "description": "ONEX policy plugin registry",
                "version": str(SEMVER_DEFAULT),
            },
        )

        services_registered.append("PolicyRegistry")
        logger.debug("Registered PolicyRegistry in container (global scope)")

        # Create ProtocolBindingRegistry instance
        handler_registry = ProtocolBindingRegistry()

        # Register with container using type interface (global scope = singleton)
        await container.service_registry.register_instance(
            interface=ProtocolBindingRegistry,
            instance=handler_registry,
            scope="global",
            metadata={
                "description": "ONEX protocol handler binding registry",
                "version": str(SEMVER_DEFAULT),
            },
        )

        services_registered.append("ProtocolBindingRegistry")
        logger.debug("Registered ProtocolBindingRegistry in container (global scope)")

        # Create RegistryCompute instance
        compute_registry = RegistryCompute()

        # Register with container using type interface (global scope = singleton)
        await container.service_registry.register_instance(
            interface=RegistryCompute,
            instance=compute_registry,
            scope="global",
            metadata={
                "description": "ONEX compute plugin registry",
                "version": str(SEMVER_DEFAULT),
            },
        )

        services_registered.append("RegistryCompute")
        logger.debug("Registered RegistryCompute in container (global scope)")

    except AttributeError as e:
        # Container missing service_registry or registration method
        error_str = str(e)
        missing_attr, hint = _analyze_attribute_error(error_str)

        logger.exception(
            "Container missing required service_registry API",
            extra={
                "error": error_str,
                "error_type": "AttributeError",
                "missing_attribute": missing_attr,
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Container wiring failed - {hint}\n"
            f"Required API: container.service_registry.register_instance("
            f"interface, instance, scope, metadata)\n"
            f"Original error: {e}"
        ) from e
    except TypeError as e:
        # Invalid arguments to register_instance
        error_str = str(e)
        invalid_arg, hint = _analyze_type_error(error_str)

        logger.exception(
            "Invalid arguments during service registration",
            extra={
                "error": error_str,
                "error_type": "TypeError",
                "invalid_argument": invalid_arg,
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Container wiring failed - {hint}\n"
            f"Expected signature: register_instance(interface=Type, instance=obj, "
            f"scope='global'|'request'|'transient', metadata=dict)\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        # Generic fallback for unexpected errors
        logger.exception(
            "Failed to register infrastructure services",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise RuntimeError(f"Failed to wire infrastructure services: {e}") from e

    logger.info(
        "Infrastructure services wired successfully",
        extra={
            "service_count": len(services_registered),
            "services": services_registered,
        },
    )

    return {"services": services_registered}


async def get_policy_registry_from_container(
    container: ModelONEXContainer,
) -> PolicyRegistry:
    """Get PolicyRegistry from container.

    Resolves PolicyRegistry using ModelONEXContainer.service_registry.resolve_service().
    This is the preferred method for accessing PolicyRegistry in container-based code.

    Note: This function is async because ModelONEXContainer.service_registry.resolve_service()
    is async in omnibase_core v0.5.6 and later (see omnibase_core.container.ModelONEXContainer).

    Args:
        container: ONEX container instance with registered PolicyRegistry.

    Returns:
        PolicyRegistry instance from container.

    Raises:
        RuntimeError: If PolicyRegistry not registered in container.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> await wire_infrastructure_services(container)
        >>> registry = await get_policy_registry_from_container(container)
        >>> isinstance(registry, PolicyRegistry)
        True

    Note:
        This function assumes PolicyRegistry was registered via
        wire_infrastructure_services(). If not, it will raise RuntimeError.
        For auto-registration, use get_or_create_policy_registry() instead.
    """
    # Validate service_registry is available
    _validate_service_registry(container, "resolve PolicyRegistry")

    try:
        registry: PolicyRegistry = await container.service_registry.resolve_service(
            PolicyRegistry
        )
        return registry
    except AttributeError as e:
        # Note: service_registry case is now handled by _validate_service_registry
        # This block handles other AttributeErrors like missing resolve_service
        error_str = str(e)
        if "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires v0.5.6 or later)."
            )
        else:
            hint = f"Missing attribute in resolution chain: {e}"

        logger.exception(
            "Failed to resolve PolicyRegistry from container",
            extra={
                "error": error_str,
                "error_type": "AttributeError",
                "service_type": "PolicyRegistry",
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Failed to resolve PolicyRegistry - {hint}\n"
            f"Required API: container.service_registry.resolve_service(PolicyRegistry)\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to resolve PolicyRegistry from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "PolicyRegistry",
            },
        )
        raise RuntimeError(
            f"PolicyRegistry not registered in container.\n"
            f"Service type requested: PolicyRegistry\n"
            f"Resolution method: container.service_registry.resolve_service(PolicyRegistry)\n"
            f"Fix: Call wire_infrastructure_services(container) first.\n"
            f"Original error: {e}"
        ) from e


async def get_or_create_policy_registry(
    container: ModelONEXContainer,
) -> PolicyRegistry:
    """Get PolicyRegistry from container, creating if not registered.

    Convenience function that provides lazy initialization semantics.
    Attempts to resolve PolicyRegistry from container, and if not found,
    creates and registers a new instance.

    This function is useful when code paths may not have called
    wire_infrastructure_services() yet or when lazy initialization is desired.

    Note: This function is async because ModelONEXContainer.service_registry methods
    (resolve_service and register_instance) are async in omnibase_core v0.5.6 and later.

    Args:
        container: ONEX container instance.

    Returns:
        PolicyRegistry instance from container (existing or newly created).

    Example:
        >>> container = ModelONEXContainer()
        >>> # No wiring yet, but this still works
        >>> registry = await get_or_create_policy_registry(container)
        >>> isinstance(registry, PolicyRegistry)
        True
        >>> # Second call returns same instance
        >>> registry2 = await get_or_create_policy_registry(container)
        >>> registry is registry2
        True

    Note:
        While this function provides convenience, prefer explicit wiring via
        wire_infrastructure_services() for production code to ensure proper
        initialization order and error handling.
    """
    # Validate service_registry is available
    _validate_service_registry(container, "get_or_create PolicyRegistry")

    try:
        # Try to resolve existing PolicyRegistry
        registry: PolicyRegistry = await container.service_registry.resolve_service(
            PolicyRegistry
        )
        return registry
    except Exception:
        # PolicyRegistry not registered, create and register it
        logger.debug("PolicyRegistry not found in container, auto-registering")

        try:
            policy_registry = PolicyRegistry()
            await container.service_registry.register_instance(
                interface=PolicyRegistry,
                instance=policy_registry,
                scope="global",
                metadata={
                    "description": "ONEX policy plugin registry (auto-registered)",
                    "version": str(SEMVER_DEFAULT),
                    "auto_registered": True,
                },
            )
            logger.debug("Auto-registered PolicyRegistry in container (lazy init)")
            return policy_registry

        except Exception as e:
            logger.exception(
                "Failed to auto-register PolicyRegistry",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise RuntimeError(
                f"Failed to create and register PolicyRegistry: {e}"
            ) from e


async def get_handler_registry_from_container(
    container: ModelONEXContainer,
) -> ProtocolBindingRegistry:
    """Get ProtocolBindingRegistry from container.

    Resolves ProtocolBindingRegistry using ModelONEXContainer.service_registry.resolve_service().
    This is the preferred method for accessing ProtocolBindingRegistry in container-based code.

    Note: This function is async because ModelONEXContainer.service_registry.resolve_service()
    is async in omnibase_core v0.5.6 and later (see omnibase_core.container.ModelONEXContainer).

    Args:
        container: ONEX container instance with registered ProtocolBindingRegistry.

    Returns:
        ProtocolBindingRegistry instance from container.

    Raises:
        RuntimeError: If ProtocolBindingRegistry not registered in container.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> await wire_infrastructure_services(container)
        >>> registry = await get_handler_registry_from_container(container)
        >>> # Verify via duck typing (per ONEX conventions)
        >>> hasattr(registry, 'register') and callable(registry.register)
        True

    Note:
        This function assumes ProtocolBindingRegistry was registered via
        wire_infrastructure_services(). If not, it will raise RuntimeError.
    """
    # Validate service_registry is available
    _validate_service_registry(container, "resolve ProtocolBindingRegistry")

    try:
        registry: ProtocolBindingRegistry = (
            await container.service_registry.resolve_service(ProtocolBindingRegistry)
        )
        return registry
    except AttributeError as e:
        # Note: service_registry case is now handled by _validate_service_registry
        # This block handles other AttributeErrors like missing resolve_service
        error_str = str(e)
        if "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires v0.5.6 or later)."
            )
        else:
            hint = f"Missing attribute in resolution chain: {e}"

        logger.exception(
            "Failed to resolve ProtocolBindingRegistry from container",
            extra={
                "error": error_str,
                "error_type": "AttributeError",
                "service_type": "ProtocolBindingRegistry",
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Failed to resolve ProtocolBindingRegistry - {hint}\n"
            f"Required API: container.service_registry.resolve_service(ProtocolBindingRegistry)\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to resolve ProtocolBindingRegistry from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "ProtocolBindingRegistry",
            },
        )
        raise RuntimeError(
            f"ProtocolBindingRegistry not registered in container.\n"
            f"Service type requested: ProtocolBindingRegistry\n"
            f"Resolution method: container.service_registry.resolve_service(ProtocolBindingRegistry)\n"
            f"Fix: Call wire_infrastructure_services(container) first.\n"
            f"Original error: {e}"
        ) from e


async def get_compute_registry_from_container(
    container: ModelONEXContainer,
) -> RegistryCompute:
    """Get RegistryCompute from container.

    Resolves RegistryCompute using ModelONEXContainer.service_registry.resolve_service().
    This is the preferred method for accessing RegistryCompute in container-based code.

    Note: This function is async because ModelONEXContainer.service_registry.resolve_service()
    is async in omnibase_core v0.5.6 and later (see omnibase_core.container.ModelONEXContainer).

    Args:
        container: ONEX container instance with registered RegistryCompute.

    Returns:
        RegistryCompute instance from container.

    Raises:
        RuntimeError: If RegistryCompute not registered in container.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> await wire_infrastructure_services(container)
        >>> registry = await get_compute_registry_from_container(container)
        >>> isinstance(registry, RegistryCompute)
        True

    Note:
        This function assumes RegistryCompute was registered via
        wire_infrastructure_services(). If not, it will raise RuntimeError.
        For auto-registration, use get_or_create_compute_registry() instead.
    """
    # Validate service_registry is available
    _validate_service_registry(container, "resolve RegistryCompute")

    try:
        registry: RegistryCompute = await container.service_registry.resolve_service(
            RegistryCompute
        )
        return registry
    except AttributeError as e:
        # Note: service_registry case is now handled by _validate_service_registry
        # This block handles other AttributeErrors like missing resolve_service
        error_str = str(e)
        if "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires v0.5.6 or later)."
            )
        else:
            hint = f"Missing attribute in resolution chain: {e}"

        logger.exception(
            "Failed to resolve RegistryCompute from container",
            extra={
                "error": error_str,
                "error_type": "AttributeError",
                "service_type": "RegistryCompute",
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Failed to resolve RegistryCompute - {hint}\n"
            f"Required API: container.service_registry.resolve_service(RegistryCompute)\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to resolve RegistryCompute from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "RegistryCompute",
            },
        )
        raise RuntimeError(
            f"RegistryCompute not registered in container.\n"
            f"Service type requested: RegistryCompute\n"
            f"Resolution method: container.service_registry.resolve_service(RegistryCompute)\n"
            f"Fix: Call wire_infrastructure_services(container) first.\n"
            f"Original error: {e}"
        ) from e


async def get_or_create_compute_registry(
    container: ModelONEXContainer,
) -> RegistryCompute:
    """Get RegistryCompute from container, creating if not registered.

    Convenience function that provides lazy initialization semantics.
    Attempts to resolve RegistryCompute from container, and if not found,
    creates and registers a new instance.

    This function is useful when code paths may not have called
    wire_infrastructure_services() yet or when lazy initialization is desired.

    Note: This function is async because ModelONEXContainer.service_registry methods
    (resolve_service and register_instance) are async in omnibase_core v0.5.6 and later.

    Args:
        container: ONEX container instance.

    Returns:
        RegistryCompute instance from container (existing or newly created).

    Example:
        >>> container = ModelONEXContainer()
        >>> # No wiring yet, but this still works
        >>> registry = await get_or_create_compute_registry(container)
        >>> isinstance(registry, RegistryCompute)
        True
        >>> # Second call returns same instance
        >>> registry2 = await get_or_create_compute_registry(container)
        >>> registry is registry2
        True

    Note:
        While this function provides convenience, prefer explicit wiring via
        wire_infrastructure_services() for production code to ensure proper
        initialization order and error handling.
    """
    # Validate service_registry is available
    _validate_service_registry(container, "get_or_create RegistryCompute")

    try:
        # Try to resolve existing RegistryCompute
        registry: RegistryCompute = await container.service_registry.resolve_service(
            RegistryCompute
        )
        return registry
    except Exception:
        # RegistryCompute not registered, create and register it
        logger.debug("RegistryCompute not found in container, auto-registering")

        try:
            compute_registry = RegistryCompute()
            await container.service_registry.register_instance(
                interface=RegistryCompute,
                instance=compute_registry,
                scope="global",
                metadata={
                    "description": "ONEX compute plugin registry (auto-registered)",
                    "version": str(SEMVER_DEFAULT),
                    "auto_registered": True,
                },
            )
            logger.debug("Auto-registered RegistryCompute in container (lazy init)")
            return compute_registry

        except Exception as e:
            logger.exception(
                "Failed to auto-register RegistryCompute",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise RuntimeError(
                f"Failed to create and register RegistryCompute: {e}"
            ) from e


async def wire_registration_handlers(
    container: ModelONEXContainer,
    pool: asyncpg.Pool,
    liveness_interval_seconds: int | None = None,
    projector: ProjectorRegistration | None = None,
    consul_handler: ConsulHandler | None = None,
) -> dict[str, list[str] | str]:
    """Register registration orchestrator handlers with the container.

    Registers ProjectionReaderRegistration and the three registration handlers:
    - HandlerNodeIntrospected
    - HandlerRuntimeTick
    - HandlerNodeRegistrationAcked

    All handlers depend on ProjectionReaderRegistration, which is registered first.
    This enables declarative dependency resolution when constructing the
    NodeRegistrationOrchestrator.

    If the container's service_registry is None (e.g., in omnibase_core 0.6.x when not
    configured), this function will log a warning and return early with status="skipped".

    Args:
        container: ONEX container instance to register services in.
        pool: asyncpg connection pool for database access.
        liveness_interval_seconds: Liveness deadline interval for ack handler.
            If None, uses ONEX_LIVENESS_INTERVAL_SECONDS env var or default (60s).
        projector: Optional ProjectorRegistration for persisting state transitions.
            If provided, HandlerNodeIntrospected will persist projections to the
            database. If None, the handler operates in read-only mode (useful for
            testing or when projection persistence is handled elsewhere).
        consul_handler: Optional ConsulHandler for dual registration with Consul.
            If provided, HandlerNodeIntrospected will register nodes with Consul
            for service discovery. If None, only PostgreSQL registration occurs.

    Returns:
        Summary dict with:
            - services: List of registered service class names
            - status: "skipped" if service_registry was unavailable

    Raises:
        RuntimeError: If service registration fails

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> import asyncpg
        >>> container = ModelONEXContainer()
        >>> pool = await asyncpg.create_pool(dsn)
        >>> projector = ProjectorRegistration(pool)
        >>> await projector.initialize_schema()
        >>> summary = await wire_registration_handlers(container, pool, projector=projector)
        >>> print(summary)
        {'services': ['ProjectionReaderRegistration', 'HandlerNodeIntrospected', ...]}
        >>> # Resolve handlers from container
        >>> handler = await container.service_registry.resolve_service(HandlerNodeIntrospected)
    """
    # Check if service_registry is available (may be None in omnibase_core 0.6.x)
    if container.service_registry is None:
        logger.warning(
            "wire_registration_handlers: service_registry is None, "
            "skipping handler registration"
        )
        return {"services": [], "status": "skipped"}

    # Validate service_registry has required methods.
    # NOTE: Validation is done BEFORE imports for fail-fast behavior - no point loading
    # heavy infrastructure modules if service_registry is unavailable.
    _validate_service_registry(container, "wire_registration_handlers")

    # Deferred imports: These imports are placed inside the function to avoid circular
    # import issues and to delay loading registration infrastructure until this function
    # is actually called (which requires a PostgreSQL pool). This follows the pattern
    # of lazy-loading optional dependencies to reduce import-time overhead.
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
        get_liveness_interval_seconds,
    )
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )

    # Resolve the actual liveness interval (from param, env var, or default)
    resolved_liveness_interval = get_liveness_interval_seconds(
        liveness_interval_seconds
    )

    services_registered: list[str] = []

    try:
        # 1. Register ProjectionReaderRegistration (dependency for all handlers)
        projection_reader = ProjectionReaderRegistration(pool)

        await container.service_registry.register_instance(
            interface=ProjectionReaderRegistration,
            instance=projection_reader,
            scope="global",
            metadata={
                "description": "Registration projection reader for orchestrator state queries",
                "version": str(SEMVER_DEFAULT),
            },
        )

        services_registered.append("ProjectionReaderRegistration")
        logger.debug(
            "Registered ProjectionReaderRegistration in container (global scope)"
        )

        # 1.5. Register ProjectorRegistration if provided
        if projector is not None:
            await container.service_registry.register_instance(
                interface=ProjectorRegistration,
                instance=projector,
                scope="global",
                metadata={
                    "description": "Registration projector for persisting state transitions",
                    "version": str(SEMVER_DEFAULT),
                },
            )
            services_registered.append("ProjectorRegistration")
            logger.debug("Registered ProjectorRegistration in container (global scope)")

        # 2. Register HandlerNodeIntrospected (with projector and consul_handler if available)
        handler_introspected = HandlerNodeIntrospected(
            projection_reader,
            projector=projector,
            consul_handler=consul_handler,
        )

        await container.service_registry.register_instance(
            interface=HandlerNodeIntrospected,
            instance=handler_introspected,
            scope="global",
            metadata={
                "description": "Handler for NodeIntrospectionEvent - registration trigger",
                "version": str(SEMVER_DEFAULT),
                "has_projector": projector is not None,
                "has_consul_handler": consul_handler is not None,
            },
        )

        services_registered.append("HandlerNodeIntrospected")
        logger.debug("Registered HandlerNodeIntrospected in container (global scope)")

        # 3. Register HandlerRuntimeTick
        handler_runtime_tick = HandlerRuntimeTick(projection_reader)

        await container.service_registry.register_instance(
            interface=HandlerRuntimeTick,
            instance=handler_runtime_tick,
            scope="global",
            metadata={
                "description": "Handler for RuntimeTick - timeout detection",
                "version": str(SEMVER_DEFAULT),
            },
        )

        services_registered.append("HandlerRuntimeTick")
        logger.debug("Registered HandlerRuntimeTick in container (global scope)")

        # 4. Register HandlerNodeRegistrationAcked
        handler_acked = HandlerNodeRegistrationAcked(
            projection_reader,
            liveness_interval_seconds=resolved_liveness_interval,
        )

        await container.service_registry.register_instance(
            interface=HandlerNodeRegistrationAcked,
            instance=handler_acked,
            scope="global",
            metadata={
                "description": "Handler for NodeRegistrationAcked command - ack processing",
                "version": str(SEMVER_DEFAULT),
                "liveness_interval_seconds": resolved_liveness_interval,
            },
        )

        services_registered.append("HandlerNodeRegistrationAcked")
        logger.debug(
            "Registered HandlerNodeRegistrationAcked in container (global scope)"
        )

    except AttributeError as e:
        # Note: service_registry case is now handled by _validate_service_registry
        # This block handles other AttributeErrors like missing register_instance
        error_str = str(e)
        if "register_instance" in error_str:
            hint = (
                "Container.service_registry missing 'register_instance' method. "
                "Check omnibase_core version compatibility (requires v0.5.6 or later)."
            )
        else:
            hint = f"Missing attribute in registration chain: {e}"

        logger.exception(
            "Failed to register registration handlers",
            extra={
                "error": error_str,
                "error_type": "AttributeError",
                "hint": hint,
            },
        )
        raise RuntimeError(
            f"Registration handler wiring failed - {hint}\nOriginal error: {e}"
        ) from e

    except Exception as e:
        logger.exception(
            "Failed to register registration handlers",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise RuntimeError(f"Failed to wire registration handlers: {e}") from e

    logger.info(
        "Registration handlers wired successfully",
        extra={
            "service_count": len(services_registered),
            "services": services_registered,
        },
    )

    return {"services": services_registered}


async def get_projection_reader_from_container(
    container: ModelONEXContainer,
) -> ProjectionReaderRegistration:
    """Get ProjectionReaderRegistration from container.

    Resolves ProjectionReaderRegistration using ModelONEXContainer.service_registry.
    This is the preferred method for accessing the projection reader in container-based code.

    Args:
        container: ONEX container instance with registered ProjectionReaderRegistration.

    Returns:
        ProjectionReaderRegistration instance from container.

    Raises:
        RuntimeError: If ProjectionReaderRegistration not registered in container.

    Example:
        >>> pool = await asyncpg.create_pool(dsn)
        >>> await wire_registration_handlers(container, pool)
        >>> reader = await get_projection_reader_from_container(container)
    """
    from omnibase_infra.projectors import ProjectionReaderRegistration

    # Validate service_registry is available
    _validate_service_registry(container, "resolve ProjectionReaderRegistration")

    try:
        reader: ProjectionReaderRegistration = (
            await container.service_registry.resolve_service(
                ProjectionReaderRegistration
            )
        )
        return reader
    except Exception as e:
        logger.exception(
            "Failed to resolve ProjectionReaderRegistration from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "ProjectionReaderRegistration",
            },
        )
        raise RuntimeError(
            f"ProjectionReaderRegistration not registered in container.\n"
            f"Fix: Call wire_registration_handlers(container, pool) first.\n"
            f"Original error: {e}"
        ) from e


async def get_handler_node_introspected_from_container(
    container: ModelONEXContainer,
) -> HandlerNodeIntrospected:
    """Get HandlerNodeIntrospected from container.

    Args:
        container: ONEX container instance with registered handlers.

    Returns:
        HandlerNodeIntrospected instance from container.

    Raises:
        RuntimeError: If handler not registered in container.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
    )

    # Validate service_registry is available
    _validate_service_registry(container, "resolve HandlerNodeIntrospected")

    try:
        handler: HandlerNodeIntrospected = (
            await container.service_registry.resolve_service(HandlerNodeIntrospected)
        )
        return handler
    except Exception as e:
        logger.exception(
            "Failed to resolve HandlerNodeIntrospected from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "HandlerNodeIntrospected",
            },
        )
        raise RuntimeError(
            f"HandlerNodeIntrospected not registered in container.\n"
            f"Fix: Call wire_registration_handlers(container, pool) first.\n"
            f"Original error: {e}"
        ) from e


async def get_handler_runtime_tick_from_container(
    container: ModelONEXContainer,
) -> HandlerRuntimeTick:
    """Get HandlerRuntimeTick from container.

    Args:
        container: ONEX container instance with registered handlers.

    Returns:
        HandlerRuntimeTick instance from container.

    Raises:
        RuntimeError: If handler not registered in container.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerRuntimeTick,
    )

    # Validate service_registry is available
    _validate_service_registry(container, "resolve HandlerRuntimeTick")

    try:
        handler: HandlerRuntimeTick = await container.service_registry.resolve_service(
            HandlerRuntimeTick
        )
        return handler
    except Exception as e:
        logger.exception(
            "Failed to resolve HandlerRuntimeTick from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "HandlerRuntimeTick",
            },
        )
        raise RuntimeError(
            f"HandlerRuntimeTick not registered in container.\n"
            f"Fix: Call wire_registration_handlers(container, pool) first.\n"
            f"Original error: {e}"
        ) from e


async def get_handler_node_registration_acked_from_container(
    container: ModelONEXContainer,
) -> HandlerNodeRegistrationAcked:
    """Get HandlerNodeRegistrationAcked from container.

    Args:
        container: ONEX container instance with registered handlers.

    Returns:
        HandlerNodeRegistrationAcked instance from container.

    Raises:
        RuntimeError: If handler not registered in container.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeRegistrationAcked,
    )

    # Validate service_registry is available
    _validate_service_registry(container, "resolve HandlerNodeRegistrationAcked")

    try:
        handler: HandlerNodeRegistrationAcked = (
            await container.service_registry.resolve_service(
                HandlerNodeRegistrationAcked
            )
        )
        return handler
    except Exception as e:
        logger.exception(
            "Failed to resolve HandlerNodeRegistrationAcked from container",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "service_type": "HandlerNodeRegistrationAcked",
            },
        )
        raise RuntimeError(
            f"HandlerNodeRegistrationAcked not registered in container.\n"
            f"Fix: Call wire_registration_handlers(container, pool) first.\n"
            f"Original error: {e}"
        ) from e


async def wire_registration_dispatchers(
    container: ModelONEXContainer,
    engine: MessageDispatchEngine,
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

    If the container's service_registry is None (e.g., in omnibase_core 0.6.x when not
    configured), this function will log a warning and return early with status="skipped".

    Args:
        container: ONEX container with registered handlers.
        engine: MessageDispatchEngine instance to register dispatchers with.

    Returns:
        Summary dict with diagnostic information:
            - dispatchers: List of registered dispatcher IDs (e.g.,
              ['dispatcher.node-introspected', 'dispatcher.runtime-tick',
               'dispatcher.node-registration-acked'])
            - routes: List of registered route IDs (e.g.,
              ['route.registration.node-introspection', 'route.registration.runtime-tick',
               'route.registration.node-registration-acked'])
            - status: "skipped" if service_registry was unavailable

        This diagnostic output can be logged or used to verify correct wiring.

    Raises:
        RuntimeError: If required handlers are not registered in the container,
            or if the engine is already frozen (cannot register new dispatchers).

    Engine Frozen Behavior:
        If engine.freeze() has been called before this function, the engine
        will reject new dispatcher registrations. Ensure this function is called
        during the wiring phase before engine.freeze() is invoked.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
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
    # Check if service_registry is available (may be None in omnibase_core 0.6.x)
    if container.service_registry is None:
        logger.warning(
            "wire_registration_dispatchers: service_registry is None, "
            "skipping dispatcher registration"
        )
        return {"dispatchers": [], "routes": [], "status": "skipped"}

    # Validate service_registry has required methods.
    # NOTE: Validation is done BEFORE imports for fail-fast behavior - no point loading
    # heavy infrastructure modules if service_registry is unavailable.
    _validate_service_registry(container, "wire_registration_dispatchers")

    # Deferred imports: These imports are placed inside the function to avoid circular
    # import issues and to delay loading dispatcher infrastructure until this function
    # is actually called.
    from omnibase_infra.enums.enum_message_category import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeIntrospected,
        HandlerNodeRegistrationAcked,
        HandlerRuntimeTick,
    )
    from omnibase_infra.runtime.dispatchers import (
        DispatcherNodeIntrospected,
        DispatcherNodeRegistrationAcked,
        DispatcherRuntimeTick,
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
        engine.register_dispatcher(
            dispatcher_id=dispatcher_introspected.dispatcher_id,
            dispatcher=dispatcher_introspected.handle,
            category=dispatcher_introspected.category,
            message_types=dispatcher_introspected.message_types,
            node_kind=dispatcher_introspected.node_kind,
        )
        dispatchers_registered.append(dispatcher_introspected.dispatcher_id)

        # 3b. Register DispatcherRuntimeTick
        engine.register_dispatcher(
            dispatcher_id=dispatcher_runtime_tick.dispatcher_id,
            dispatcher=dispatcher_runtime_tick.handle,
            category=dispatcher_runtime_tick.category,
            message_types=dispatcher_runtime_tick.message_types,
            node_kind=dispatcher_runtime_tick.node_kind,
        )
        dispatchers_registered.append(dispatcher_runtime_tick.dispatcher_id)

        # 3c. Register DispatcherNodeRegistrationAcked
        engine.register_dispatcher(
            dispatcher_id=dispatcher_acked.dispatcher_id,
            dispatcher=dispatcher_acked.handle,
            category=dispatcher_acked.category,
            message_types=dispatcher_acked.message_types,
            node_kind=dispatcher_acked.node_kind,
        )
        dispatchers_registered.append(dispatcher_acked.dispatcher_id)

        # 4. Register routes for topic-based routing
        # 4a. Route for introspection events
        route_introspection = ModelDispatchRoute(
            route_id="route.registration.node-introspection",
            topic_pattern="*.node.introspection.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_introspected.dispatcher_id,
            message_type="ModelNodeIntrospectionEvent",
        )
        engine.register_route(route_introspection)
        routes_registered.append(route_introspection.route_id)

        # 4b. Route for runtime tick events
        route_runtime_tick = ModelDispatchRoute(
            route_id="route.registration.runtime-tick",
            topic_pattern="*.runtime.tick.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher_runtime_tick.dispatcher_id,
            message_type="ModelRuntimeTick",
        )
        engine.register_route(route_runtime_tick)
        routes_registered.append(route_runtime_tick.route_id)

        # 4c. Route for registration ack commands
        route_acked = ModelDispatchRoute(
            route_id="route.registration.node-registration-acked",
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
        raise RuntimeError(
            f"Failed to wire registration dispatchers: {e}\n"
            f"Fix: Ensure wire_registration_handlers(container, pool) "
            f"was called first."
        ) from e

    return {
        "dispatchers": dispatchers_registered,
        "routes": routes_registered,
    }


__all__: list[str] = [
    # Container wiring functions
    "get_compute_registry_from_container",
    "get_handler_node_introspected_from_container",
    "get_handler_node_registration_acked_from_container",
    "get_handler_registry_from_container",
    "get_handler_runtime_tick_from_container",
    "get_or_create_compute_registry",
    "get_or_create_policy_registry",
    "get_policy_registry_from_container",
    "get_projection_reader_from_container",
    "wire_infrastructure_services",
    # Registration handlers (OMN-888)
    "wire_registration_handlers",
    # Registration dispatchers (OMN-892)
    "wire_registration_dispatchers",
]
