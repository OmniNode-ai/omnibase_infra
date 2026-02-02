# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeContractPersistenceEffect infrastructure dependencies.

This registry provides factory methods for creating NodeContractPersistenceEffect
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
    - handlers/: PostgreSQL operation handlers
    - OMN-1845: Implementation ticket

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_contract_persistence_effect.node import (
        NodeContractPersistenceEffect,
    )


class RegistryInfraContractPersistenceEffect:
    """Infrastructure registry for NodeContractPersistenceEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeContractPersistenceEffect instances.

    This registry follows the ONEX infrastructure registry pattern:
        - Factory method for node creation with container injection
        - Protocol requirements documentation for container validation
        - Node type classification for routing decisions
        - Capability listing for service discovery

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_contract_persistence_effect.registry import (
        ...     RegistryInfraContractPersistenceEffect,
        ... )
        >>>
        >>> # Create container with required protocols registered
        >>> container = ModelONEXContainer()
        >>> # ... register protocols ...
        >>>
        >>> # Create node instance via registry
        >>> effect = RegistryInfraContractPersistenceEffect.create(container)

    .. versionadded:: 0.5.0
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeContractPersistenceEffect:
        """Create a NodeContractPersistenceEffect instance with resolved dependencies.

        Factory method that creates a fully configured NodeContractPersistenceEffect
        using the provided ONEX container for dependency injection.

        Args:
            container: ONEX dependency injection container. Must have the
                following protocols registered:
                - ProtocolPostgresAdapter: PostgreSQL database operations
                - ProtocolCircuitBreakerAware: Backend circuit breaker protection

        Returns:
            Configured NodeContractPersistenceEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.

        Example:
            >>> container = ModelONEXContainer()
            >>> container.register(ProtocolPostgresAdapter, postgres_adapter)
            >>> effect = RegistryInfraContractPersistenceEffect.create(container)

        .. versionadded:: 0.5.0
        """
        from omnibase_infra.nodes.node_contract_persistence_effect.node import (
            NodeContractPersistenceEffect,
        )

        return NodeContractPersistenceEffect(container)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        Returns the protocol class names that must be registered in the
        container before creating a NodeContractPersistenceEffect instance.

        Returns:
            List of protocol class names required for node operation.

        Example:
            >>> protocols = RegistryInfraContractPersistenceEffect.get_required_protocols()
            >>> for proto in protocols:
            ...     if not container.has(proto):
            ...         raise ConfigurationError(f"Missing: {proto}")

        .. versionadded:: 0.5.0
        """
        return [
            "ProtocolPostgresAdapter",
            "ProtocolCircuitBreakerAware",
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
        return "node_contract_persistence_effect"

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
            "contract_persistence",
            "topic_routing",
            "staleness_detection",
            "heartbeat_tracking",
            "soft_delete",
            "circuit_breaker_protection",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers as defined in contract.yaml.

        .. versionadded:: 0.5.0
        """
        return [
            "upsert_contract",
            "update_topic",
            "mark_stale",
            "update_heartbeat",
            "deactivate_contract",
            "cleanup_topic_references",
        ]

    @staticmethod
    def get_supported_intent_types() -> list[str]:
        """Get list of intent types routed by this node.

        Returns the payload.intent_type values that this effect node
        can handle, matching ContractRegistryReducer output.

        Returns:
            List of intent type strings.

        .. versionadded:: 0.5.0
        """
        return [
            "postgres.upsert_contract",
            "postgres.update_topic",
            "postgres.mark_stale",
            "postgres.update_heartbeat",
            "postgres.deactivate_contract",
            "postgres.cleanup_topic_references",
        ]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend identifiers.

        .. versionadded:: 0.5.0
        """
        return ["postgres"]


__all__ = ["RegistryInfraContractPersistenceEffect"]
