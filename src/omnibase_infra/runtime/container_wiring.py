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

Example Usage:
    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
    from omnibase_infra.runtime.policy_registry import PolicyRegistry

    # Bootstrap container
    container = ModelONEXContainer()

    # Wire infrastructure services
    summary = wire_infrastructure_services(container)
    print(f"Registered {len(summary['services'])} services")

    # Resolve services using type interface
    policy_registry = container.service_registry.resolve_service(PolicyRegistry)

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

from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.policy_registry import PolicyRegistry

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


async def wire_infrastructure_services(
    container: ModelONEXContainer,
) -> dict[str, list[str]]:
    """Register infrastructure services with the container.

    Registers PolicyRegistry and ProtocolBindingRegistry as global singleton
    services in the container. Uses ModelONEXContainer.service_registry.register_instance()
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
        {'services': ['PolicyRegistry', 'ProtocolBindingRegistry']}
        >>> policy_reg = await container.service_registry.resolve_service(PolicyRegistry)
        >>> handler_reg = await container.service_registry.resolve_service(ProtocolBindingRegistry)
        >>> isinstance(policy_reg, PolicyRegistry)
        True
        >>> isinstance(handler_reg, ProtocolBindingRegistry)
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
                "version": "1.0.0",
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
                "version": "1.0.0",
            },
        )

        services_registered.append("ProtocolBindingRegistry")
        logger.debug("Registered ProtocolBindingRegistry in container (global scope)")

    except Exception as e:
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


def get_policy_registry_from_container(container: ModelONEXContainer) -> PolicyRegistry:
    """Get PolicyRegistry from container.

    Resolves PolicyRegistry using ModelONEXContainer.service_registry.resolve_service().
    This is the preferred method for accessing PolicyRegistry in container-based code.

    Args:
        container: ONEX container instance with registered PolicyRegistry.

    Returns:
        PolicyRegistry instance from container.

    Raises:
        RuntimeError: If PolicyRegistry not registered in container.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> wire_infrastructure_services(container)
        >>> registry = get_policy_registry_from_container(container)
        >>> isinstance(registry, PolicyRegistry)
        True

    Note:
        This function assumes PolicyRegistry was registered via
        wire_infrastructure_services(). If not, it will raise RuntimeError.
        For auto-registration, use get_or_create_policy_registry() instead.
    """
    try:
        registry: PolicyRegistry = container.service_registry.resolve_service(
            PolicyRegistry
        )
        return registry
    except Exception as e:
        logger.exception(
            "Failed to resolve PolicyRegistry from container",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise RuntimeError(
            "PolicyRegistry not registered in container. "
            "Call wire_infrastructure_services(container) first."
        ) from e


def get_or_create_policy_registry(container: ModelONEXContainer) -> PolicyRegistry:
    """Get PolicyRegistry from container, creating if not registered.

    Helper function for backwards compatibility during migration.
    Attempts to resolve PolicyRegistry from container, and if not found,
    creates and registers a new instance.

    This function is useful during incremental migration when some code paths
    may not have called wire_infrastructure_services() yet.

    Args:
        container: ONEX container instance.

    Returns:
        PolicyRegistry instance from container (existing or newly created).

    Example:
        >>> container = ModelONEXContainer()
        >>> # No wiring yet, but this still works
        >>> registry = get_or_create_policy_registry(container)
        >>> isinstance(registry, PolicyRegistry)
        True
        >>> # Second call returns same instance
        >>> registry2 = get_or_create_policy_registry(container)
        >>> registry is registry2
        True

    Note:
        While this function provides convenience, prefer explicit wiring via
        wire_infrastructure_services() for production code to ensure proper
        initialization order and error handling.
    """
    try:
        # Try to resolve existing PolicyRegistry
        registry: PolicyRegistry = container.service_registry.resolve_service(
            PolicyRegistry
        )
        return registry
    except Exception:
        # PolicyRegistry not registered, create and register it
        logger.debug("PolicyRegistry not found in container, auto-registering")

        try:
            policy_registry = PolicyRegistry()
            container.service_registry.register_instance(
                interface=PolicyRegistry,
                instance=policy_registry,
                scope="global",
                metadata={
                    "description": "ONEX policy plugin registry (auto-registered)",
                    "version": "1.0.0",
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


def get_handler_registry_from_container(
    container: ModelONEXContainer,
) -> ProtocolBindingRegistry:
    """Get ProtocolBindingRegistry from container.

    Resolves ProtocolBindingRegistry using ModelONEXContainer.service_registry.resolve_service().
    This is the preferred method for accessing ProtocolBindingRegistry in container-based code.

    Args:
        container: ONEX container instance with registered ProtocolBindingRegistry.

    Returns:
        ProtocolBindingRegistry instance from container.

    Raises:
        RuntimeError: If ProtocolBindingRegistry not registered in container.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> wire_infrastructure_services(container)
        >>> registry = get_handler_registry_from_container(container)
        >>> isinstance(registry, ProtocolBindingRegistry)
        True

    Note:
        This function assumes ProtocolBindingRegistry was registered via
        wire_infrastructure_services(). If not, it will raise RuntimeError.
    """
    try:
        registry: ProtocolBindingRegistry = container.service_registry.resolve_service(
            ProtocolBindingRegistry
        )
        return registry
    except Exception as e:
        logger.exception(
            "Failed to resolve ProtocolBindingRegistry from container",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise RuntimeError(
            "ProtocolBindingRegistry not registered in container. "
            "Call wire_infrastructure_services(container) first."
        ) from e


__all__: list[str] = [
    "wire_infrastructure_services",
    "get_policy_registry_from_container",
    "get_handler_registry_from_container",
    "get_or_create_policy_registry",
]
