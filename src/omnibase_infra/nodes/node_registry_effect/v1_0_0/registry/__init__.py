# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry module for Registry Effect Node v1.0.0.

This module provides dependency injection configuration and container registration
for the Registry Effect Node.

Usage:
    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registry_effect.v1_0_0.registry import (
        RegistryInfraRegistryEffect,
    )

    # Bootstrap container (handlers must be registered first)
    container = ModelONEXContainer()

    # Option 1: Register factory for lazy creation
    await RegistryInfraRegistryEffect.register(container)

    # Option 2: Directly resolve with optional config
    node = await RegistryInfraRegistryEffect.resolve(container, config)
    ```

Prerequisites:
    Before using this registry, the following services must be registered
    in the container's service_registry:
    - ProtocolEnvelopeExecutor with name="consul" (required)
    - ProtocolEnvelopeExecutor with name="postgres" (required)
    - ProtocolEventBus (optional, for request_introspection operation)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    ModelNodeRegistryEffectConfig,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.node import NodeRegistryEffect

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)


class RegistryInfraRegistryEffect:
    """Registry configuration for NodeRegistryEffect DI container integration.

    This class provides static methods for registering and resolving
    NodeRegistryEffect instances via ModelONEXContainer.

    The registry follows ONEX naming conventions:
    - File: registry_infra_<node_name>.py (in registry/ subdirectory)
    - Class: RegistryInfra<NodeName>

    Registration Pattern:
        The registry supports two patterns:

        1. **Factory Registration**: Register a factory function that creates
           NodeRegistryEffect on demand. Use when you want container to manage
           lifecycle.

        2. **Direct Resolution**: Create and return a NodeRegistryEffect instance
           immediately. Use when you need immediate access to the node.

    Prerequisites:
        Both patterns require handler services to be registered first:
        - ProtocolEnvelopeExecutor (name="consul")
        - ProtocolEnvelopeExecutor (name="postgres")
        - ProtocolEventBus (optional)

    Example:
        ```python
        # Register factory for lazy instantiation
        await RegistryInfraRegistryEffect.register(container)

        # Later, resolve when needed
        node = await container.service_registry.resolve_service(NodeRegistryEffect)

        # Or resolve directly with custom config
        config = ModelNodeRegistryEffectConfig(circuit_breaker_threshold=10)
        node = await RegistryInfraRegistryEffect.resolve(container, config)
        ```
    """

    @staticmethod
    async def register(container: ModelONEXContainer) -> None:
        """Register NodeRegistryEffect factory with container.

        Registers a factory function that creates NodeRegistryEffect instances
        using `create_from_container()`. The factory is registered with global
        scope (singleton per container).

        Note: This registers a factory, not an instance. The NodeRegistryEffect
        will be created when first resolved from the container.

        Args:
            container: ONEX container to register the factory in.

        Raises:
            RuntimeError: If factory registration fails.

        Example:
            ```python
            container = ModelONEXContainer()
            # Register handlers first...
            await RegistryInfraRegistryEffect.register(container)

            # Later, resolve the node
            node = await container.service_registry.resolve_service(NodeRegistryEffect)
            ```
        """
        try:
            await container.service_registry.register_factory(
                interface=NodeRegistryEffect,
                factory=NodeRegistryEffect.create_from_container,
                scope="global",
                metadata={
                    "description": "Registry Effect Node for dual registration (Consul + PostgreSQL)",
                    "version": "1.0.0",
                    "node_type": "EFFECT",
                },
            )
            logger.debug(
                "Registered NodeRegistryEffect factory in container (global scope)"
            )
        except AttributeError as e:
            error_str = str(e)
            if "register_factory" in error_str:
                hint = (
                    "Container.service_registry missing 'register_factory' method. "
                    "Check omnibase_core version compatibility."
                )
            else:
                hint = f"Missing attribute: {e}"

            logger.exception(
                "Failed to register NodeRegistryEffect factory",
                extra={
                    "error": error_str,
                    "error_type": "AttributeError",
                    "hint": hint,
                },
            )
            raise RuntimeError(
                f"Failed to register NodeRegistryEffect factory - {hint}\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            logger.exception(
                "Failed to register NodeRegistryEffect factory",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise RuntimeError(
                f"Failed to register NodeRegistryEffect factory: {e}"
            ) from e

    @staticmethod
    async def resolve(
        container: ModelONEXContainer,
        config: ModelNodeRegistryEffectConfig | None = None,
    ) -> NodeRegistryEffect:
        """Resolve NodeRegistryEffect from container with optional config.

        Creates a new NodeRegistryEffect instance using dependencies from
        the container. This method calls `NodeRegistryEffect.create_from_container()`
        directly, bypassing any registered factory.

        Use this method when you need to provide custom configuration, or when
        you want to create multiple independent instances.

        Args:
            container: ONEX container with registered handler services.
            config: Optional configuration override. If not provided, uses
                ModelNodeRegistryEffectConfig with default values.

        Returns:
            Configured NodeRegistryEffect instance ready for use.
            Note: You must call `await node.initialize()` before use.

        Raises:
            RuntimeError: If required handler services are not registered.

        Example:
            ```python
            # With default config
            node = await RegistryInfraRegistryEffect.resolve(container)

            # With custom config
            config = ModelNodeRegistryEffectConfig(
                circuit_breaker_threshold=10,
                circuit_breaker_reset_timeout=120.0,
            )
            node = await RegistryInfraRegistryEffect.resolve(container, config)

            # Don't forget to initialize
            await node.initialize()
            ```
        """
        return await NodeRegistryEffect.create_from_container(container, config)


__all__ = ["RegistryInfraRegistryEffect"]
