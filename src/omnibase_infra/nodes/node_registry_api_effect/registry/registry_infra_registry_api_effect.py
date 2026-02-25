# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Infrastructure registry for NodeRegistryApiEffect.

Provides factory methods for creating ``NodeRegistryApiEffect`` instances
and exposes node metadata for introspection and service discovery.

Following ONEX naming conventions:
    - File: ``registry_infra_<node_name>.py``
    - Class: ``RegistryInfra<NodeName>``

Ticket: OMN-1441
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_registry_api_effect.node import NodeRegistryApiEffect


class RegistryInfraRegistryApiEffect:
    """Infrastructure registry for NodeRegistryApiEffect.

    Provides dependency resolution and factory methods for creating
    properly configured ``NodeRegistryApiEffect`` instances.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_registry_api_effect.registry import (
        ...     RegistryInfraRegistryApiEffect,
        ... )
        >>> container = ModelONEXContainer()
        >>> effect = RegistryInfraRegistryApiEffect.create(container)
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeRegistryApiEffect:
        """Create a ``NodeRegistryApiEffect`` instance via container injection.

        Args:
            container: ONEX dependency injection container.  Optional
                protocols (``ProjectionReaderRegistration``,
                ``HandlerServiceDiscoveryConsul``, ``ProjectionReaderContract``)
                may be registered to enable full backend connectivity.

        Returns:
            Configured ``NodeRegistryApiEffect`` instance.
        """
        from omnibase_infra.nodes.node_registry_api_effect.node import (
            NodeRegistryApiEffect,
        )

        return NodeRegistryApiEffect(container)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Return protocol class names required by this node.

        All dependencies for this node are optional (partial-success semantics),
        so the required list is empty.  Callers should still register the
        optional protocols to enable full functionality.

        Returns:
            Empty list (all dependencies are optional).
        """
        return []

    @staticmethod
    def get_optional_protocols() -> list[str]:
        """Return optional protocol class names that enhance this node.

        Returns:
            List of optional protocol class names.
        """
        return [
            "ProjectionReaderRegistration",
            "HandlerServiceDiscoveryConsul",
            "ProjectionReaderContract",
        ]

    @staticmethod
    def get_node_type() -> str:
        """Return the ONEX node archetype for routing decisions.

        Returns:
            ``"EFFECT"``
        """
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Return the canonical node name as defined in ``contract.yaml``.

        Returns:
            ``"node_registry_api_effect"``
        """
        return "node_registry_api_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Return capability identifiers provided by this node.

        Returns:
            List of capability name strings.
        """
        return [
            "registry.discovery",
            "registry.nodes",
            "registry.instances",
            "registry.contracts",
            "registry.topics",
            "registry.widget_mapping",
            "registry.health",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Return operation identifiers supported by this node.

        Returns:
            List of operation name strings as defined in ``contract.yaml``.
        """
        return [
            "list_nodes",
            "get_node",
            "list_instances",
            "get_widget_mapping",
            "get_discovery",
            "get_health",
            "list_contracts",
            "get_contract",
            "list_topics",
            "get_topic",
        ]


__all__ = ["RegistryInfraRegistryApiEffect"]
