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
- Compatible with omnibase_core 0.4.x API
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.models.model_semver import SEMVER_DEFAULT
from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.registry_compute import RegistryCompute

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


async def wire_infrastructure_services(
    container: ModelONEXContainer,
) -> dict[str, list[str]]:
    """Register infrastructure services with the container.

    Registers PolicyRegistry, ProtocolBindingRegistry, and RegistryCompute as global
    singleton services in the container. Uses ModelONEXContainer.service_registry.register_instance()
    with the respective class as the interface type.

    Note: This function is async because ModelONEXContainer.service_registry.register_instance()
    is async in omnibase_core 0.4.x+.

    Args:
        container: ONEX container instance to register services in.

    Returns:
        Summary dict with:
            - services: List of registered service class names

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
        missing_attr = error_str.split("'")[-2] if "'" in error_str else "unknown"

        if "service_registry" in error_str:
            hint = (
                "Container missing 'service_registry' attribute. "
                "Expected ModelONEXContainer from omnibase_core."
            )
        elif "register_instance" in error_str:
            hint = (
                "Container.service_registry missing 'register_instance' method. "
                "Check omnibase_core version compatibility (requires 0.4.x+)."
            )
        else:
            hint = f"Missing attribute: '{missing_attr}'"

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

        # Identify which argument caused the issue
        if "interface" in error_str:
            invalid_arg = "interface"
            hint = (
                "Invalid 'interface' argument. "
                "Expected a type class (e.g., PolicyRegistry), not an instance."
            )
        elif "instance" in error_str:
            invalid_arg = "instance"
            hint = (
                "Invalid 'instance' argument. "
                "Expected an instance of the interface type."
            )
        elif "scope" in error_str:
            invalid_arg = "scope"
            hint = (
                "Invalid 'scope' argument. "
                "Expected 'global', 'request', or 'transient'."
            )
        elif "metadata" in error_str:
            invalid_arg = "metadata"
            hint = "Invalid 'metadata' argument. Expected dict[str, object]."
        elif "positional" in error_str or "argument" in error_str:
            invalid_arg = "signature"
            hint = (
                "Argument count mismatch. "
                "Check register_instance() signature compatibility with omnibase_core version."
            )
        else:
            invalid_arg = "unknown"
            hint = "Check register_instance() signature compatibility."

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
    is async in omnibase_core 0.4.x+.

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
    try:
        registry: PolicyRegistry = await container.service_registry.resolve_service(
            PolicyRegistry
        )
        return registry
    except AttributeError as e:
        error_str = str(e)
        if "service_registry" in error_str:
            hint = (
                "Container missing 'service_registry' attribute. "
                "Expected ModelONEXContainer from omnibase_core."
            )
        elif "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires 0.4.x+)."
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
    (resolve_service and register_instance) are async in omnibase_core 0.4.x+.

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
    is async in omnibase_core 0.4.x+.

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
    try:
        registry: ProtocolBindingRegistry = (
            await container.service_registry.resolve_service(ProtocolBindingRegistry)
        )
        return registry
    except AttributeError as e:
        error_str = str(e)
        if "service_registry" in error_str:
            hint = (
                "Container missing 'service_registry' attribute. "
                "Expected ModelONEXContainer from omnibase_core."
            )
        elif "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires 0.4.x+)."
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
    is async in omnibase_core 0.4.x+.

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
    try:
        registry: RegistryCompute = await container.service_registry.resolve_service(
            RegistryCompute
        )
        return registry
    except AttributeError as e:
        error_str = str(e)
        if "service_registry" in error_str:
            hint = (
                "Container missing 'service_registry' attribute. "
                "Expected ModelONEXContainer from omnibase_core."
            )
        elif "resolve_service" in error_str:
            hint = (
                "Container.service_registry missing 'resolve_service' method. "
                "Check omnibase_core version compatibility (requires 0.4.x+)."
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
    (resolve_service and register_instance) are async in omnibase_core 0.4.x+.

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


__all__: list[str] = [
    "wire_infrastructure_services",
    "get_policy_registry_from_container",
    "get_handler_registry_from_container",
    "get_or_create_policy_registry",
    "get_compute_registry_from_container",
    "get_or_create_compute_registry",
]
