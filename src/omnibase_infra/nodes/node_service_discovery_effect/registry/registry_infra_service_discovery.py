# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for Service Discovery Node Dependencies.

This module provides RegistryInfraServiceDiscovery for registering
service discovery node dependencies with the ONEX container.

Architecture:
    RegistryInfraServiceDiscovery handles dependency injection setup
    for the NodeServiceDiscoveryEffect node:
    - Registers protocol implementations (Consul, K8s, Etcd handlers)
    - Configures handler resolution based on environment/configuration
    - Provides factory methods for handler instantiation

Usage:
    The registry is typically called during application bootstrap:

    .. code-block:: python

        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_service_discovery_effect.registry import (
            RegistryInfraServiceDiscovery,
        )

        container = ModelONEXContainer()
        RegistryInfraServiceDiscovery.register(container)

        # Or with specific handler configuration
        RegistryInfraServiceDiscovery.register_with_handler(
            container,
            handler=consul_handler_instance,
        )

Related:
    - NodeServiceDiscoveryEffect: Node that consumes registered dependencies
    - ProtocolDiscoveryOperations: Protocol for handlers
    - ModelONEXContainer: DI container for dependency resolution
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.errors import ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
        ProtocolDiscoveryOperations,
    )

logger = logging.getLogger(__name__)


class RegistryInfraServiceDiscovery:
    """Registry for service discovery node dependencies.

    Provides static methods for registering service discovery
    dependencies with the ONEX container. Supports both default
    registration and explicit handler configuration.

    API Pattern Note:
        This registry uses ``container.service_registry.register_instance()``
        for protocol-based type-safe DI resolution. This differs from
        RegistryInfraRegistrationStorage which uses a module-level dict for
        multi-handler routing by type string (e.g., "postgresql", "mock"),
        as ServiceRegistry does not support dict-style indexed access.

        The different patterns serve different purposes:
        - Protocol-based registration: Single handler per protocol type
        - Dict-based registration: Multiple handlers with type routing

    Class Methods:
        register: Register with default/environment-based configuration.
        register_with_handler: Register with explicit handler instance.

    Example:
        .. code-block:: python

            from omnibase_core.models.container import ModelONEXContainer

            # Default registration (uses environment configuration)
            container = ModelONEXContainer()
            RegistryInfraServiceDiscovery.register(container)

            # Explicit handler registration
            consul_handler = HandlerServiceDiscoveryConsul(
                container=container,
                consul_host="localhost",
            )
            RegistryInfraServiceDiscovery.register_with_handler(
                container,
                handler=consul_handler,
            )
    """

    @staticmethod
    def register(container: ModelONEXContainer) -> None:
        """Register service discovery dependencies with default configuration.

        Registers service discovery dependencies using environment-based
        or default configuration. The specific handler implementation
        is determined by configuration at runtime.

        Args:
            container: ONEX dependency injection container.

        Note:
            This method registers a factory function that creates the
            appropriate handler based on configuration. The actual
            handler is not instantiated until resolution time.

        Example:
            >>> container = ModelONEXContainer()
            >>> RegistryInfraServiceDiscovery.register(container)
        """
        if container.service_registry is None:
            return

        # NOTE: Factory registration (register_factory) is not implemented in
        # omnibase_core v1.0. This method provides no-op registration for forward
        # compatibility. Use register_with_handler() to explicitly provide a
        # pre-configured handler instance.
        logger.debug(
            "Service discovery factory registration skipped - "
            "factory registration not implemented in v1.0. "
            "Use register_with_handler() to register an explicit handler instance."
        )

    @staticmethod
    async def register_with_handler(
        container: ModelONEXContainer,
        handler: ProtocolDiscoveryOperations,
    ) -> None:
        """Register service discovery dependencies with explicit handler.

        Registers the provided handler implementation directly.
        Use this method when you have a pre-configured handler instance.

        Args:
            container: ONEX dependency injection container.
            handler: Pre-configured handler implementation.

        Raises:
            TypeError: If handler does not implement ProtocolDiscoveryOperations.

        Example:
            >>> container = ModelONEXContainer()
            >>> handler = HandlerServiceDiscoveryConsul(config)
            >>> await RegistryInfraServiceDiscovery.register_with_handler(
            ...     container,
            ...     handler=handler,
            ... )
        """
        # Import at runtime for isinstance check (protocol is @runtime_checkable)
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolDiscoveryOperations,
        )

        # NOTE: isinstance() is intentionally used here instead of duck typing for:
        # 1. Fail-fast validation: Immediately reject invalid handlers at registration
        #    time rather than discovering missing methods at runtime during operations
        # 2. Type safety: The @runtime_checkable decorator enables structural subtyping
        #    checks that verify all required Protocol methods exist
        # 3. Clear error messages: TypeError with specific protocol name aids debugging
        # Duck typing (hasattr checks) would defer validation to method call time,
        # making it harder to diagnose misconfigured handlers.
        # See: conftest.py "Protocol Compliance Strategy" for when to use each approach.
        if not isinstance(handler, ProtocolDiscoveryOperations):
            raise TypeError(
                f"Handler must implement ProtocolDiscoveryOperations, "
                f"got {type(handler).__name__}"
            )

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=ProtocolDiscoveryOperations,  # type: ignore[type-abstract]
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )

    @staticmethod
    def _create_handler_from_config(
        _container: ModelONEXContainer,
    ) -> ProtocolDiscoveryOperations:
        """Create handler based on configuration.

        Factory function that creates the appropriate handler
        implementation based on environment/configuration.

        Args:
            container: ONEX container (for accessing configuration).

        Returns:
            Configured handler implementation.

        Raises:
            ProtocolConfigurationError: If no handler can be configured.

        Note:
            This is a placeholder implementation. Actual handler
            creation depends on available backend implementations
            (Consul, Kubernetes, Etcd adapters).
        """
        # Placeholder implementation
        # In production, this would:
        # 1. Read configuration (CONSUL_AGENT_URL, K8S_NAMESPACE, etc.)
        # 2. Instantiate appropriate handler (ConsulHandler, K8sHandler, etc.)
        # 3. Return the configured handler
        raise ProtocolConfigurationError(
            "No service discovery handler configured. "
            "Use register_with_handler() to provide a handler, "
            "or implement handler auto-configuration."
        )


__all__ = ["RegistryInfraServiceDiscovery"]
