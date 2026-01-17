# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeRegistryEffect infrastructure dependencies.

This registry provides factory methods for creating NodeRegistryEffect
instances with their required dependencies resolved from the container.

Following ONEX naming conventions:
    - File: registry_infra_<node_name>.py
    - Class: RegistryInfra<NodeName>

The registry serves as the entry point for creating properly configured
effect node instances, documenting required protocols, and providing
node metadata for introspection.

Related:
    - contract.yaml: Node contract defining operations and dependencies
    - node.py: Declarative node implementation
    - handlers/: Backend-specific operation handlers

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_registry_effect.node import NodeRegistryEffect


class RegistryInfraRegistryEffect:
    """Infrastructure registry for NodeRegistryEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeRegistryEffect instances.

    This registry follows the ONEX infrastructure registry pattern:
        - Factory method for node creation with container injection
        - Protocol requirements documentation for container validation
        - Node type classification for routing decisions
        - Capability listing for service discovery

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_registry_effect.registry import (
        ...     RegistryInfraRegistryEffect,
        ... )
        >>>
        >>> # Create container with required protocols registered
        >>> container = ModelONEXContainer()
        >>> # ... register protocols ...
        >>>
        >>> # Create node instance via registry
        >>> effect = RegistryInfraRegistryEffect.create(container)

    .. versionadded:: 0.5.0
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeRegistryEffect:
        """Create a NodeRegistryEffect instance with resolved dependencies.

        Factory method that creates a fully configured NodeRegistryEffect
        using the provided ONEX container for dependency injection.

        Args:
            container: ONEX dependency injection container. Must have the
                following protocols registered:
                - ProtocolConsulClient: Consul service discovery operations
                - ProtocolPostgresAdapter: PostgreSQL registration persistence
                - ProtocolEffectIdempotencyStore: Idempotency tracking
                - ProtocolCircuitBreaker: Backend circuit breaker protection

        Returns:
            Configured NodeRegistryEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.

        Example:
            >>> container = ModelONEXContainer()
            >>> container.register(ProtocolConsulClient, consul_client)
            >>> container.register(ProtocolPostgresAdapter, postgres_adapter)
            >>> effect = RegistryInfraRegistryEffect.create(container)

        .. versionadded:: 0.5.0
        """
        from omnibase_infra.nodes.node_registry_effect.node import NodeRegistryEffect

        return NodeRegistryEffect(container)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        Returns the protocol class names that must be registered in the
        container before creating a NodeRegistryEffect instance.

        Returns:
            List of protocol class names required for node operation.

        Example:
            >>> protocols = RegistryInfraRegistryEffect.get_required_protocols()
            >>> for proto in protocols:
            ...     if not container.has(proto):
            ...         raise ConfigurationError(f"Missing: {proto}")

        .. versionadded:: 0.5.0
        """
        return [
            "ProtocolConsulClient",
            "ProtocolPostgresAdapter",
            "ProtocolEffectIdempotencyStore",
            "ProtocolCircuitBreaker",
        ]

    @staticmethod
    def get_node_type() -> str:
        """Get the node type classification.

        Returns the ONEX node archetype for this node, used for
        routing decisions and execution context selection.

        Returns:
            Node type string ("EFFECT").

        Note:
            EFFECT nodes perform external I/O operations and should
            be treated as side-effecting by the runtime.

        .. versionadded:: 0.5.0
        """
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name.

        Returns:
            The node name as defined in contract.yaml.

        .. versionadded:: 0.5.0
        """
        return "node_registry_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node.

        Returns capability identifiers that can be used for service
        discovery and feature detection.

        Returns:
            List of capability identifiers.

        .. versionadded:: 0.5.0
        """
        return [
            "dual_backend_registration",
            "partial_failure_handling",
            "targeted_retry",
            "idempotent_operations",
            "circuit_breaker_protection",
            "parallel_execution",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers as defined in contract.yaml.

        .. versionadded:: 0.5.0
        """
        return [
            "register_node",
            "deregister_node",
            "retry_partial_failure",
        ]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend identifiers.

        .. versionadded:: 0.5.0
        """
        return ["consul", "postgres"]


__all__ = ["RegistryInfraRegistryEffect"]
