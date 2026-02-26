# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Infrastructure registry for NodeContractResolverBridge.

Provides factory methods for creating ``NodeContractResolverBridge`` instances
and exposes node metadata for introspection and service discovery.

Following ONEX naming conventions:
    - File: ``registry_infra_<node_name>.py``
    - Class: ``RegistryInfra<NodeName>``

Ticket: OMN-2756
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_contract_resolver_bridge.node import (
        NodeContractResolverBridge,
    )


class RegistryInfraContractResolverBridge:
    """Infrastructure registry for NodeContractResolverBridge.

    Provides dependency resolution and factory methods for creating
    properly configured ``NodeContractResolverBridge`` instances.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_contract_resolver_bridge.registry import (
        ...     RegistryInfraContractResolverBridge,
        ... )
        >>> container = ModelONEXContainer()
        >>> bridge = RegistryInfraContractResolverBridge.create(container)
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeContractResolverBridge:
        """Create a ``NodeContractResolverBridge`` instance via container injection.

        Args:
            container: ONEX dependency injection container. The optional
                ``ProtocolEventBus`` may be registered to enable fire-and-forget
                Kafka event emission.

        Returns:
            Configured ``NodeContractResolverBridge`` instance.
        """
        from omnibase_infra.nodes.node_contract_resolver_bridge.node import (
            NodeContractResolverBridge,
        )

        return NodeContractResolverBridge(container)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Return protocol class names required by this node.

        The event bus is optional (fire-and-forget Kafka emission), so the
        required list is empty. The service degrades gracefully without it.

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
        return ["ProtocolEventBus"]

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
            ``"node_contract_resolver_bridge"``
        """
        return "node_contract_resolver_bridge"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Return capability identifiers provided by this node.

        Returns:
            List of capability name strings.
        """
        return [
            "contract.resolve",
            "contract.health",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Return operation identifiers supported by this node.

        Returns:
            List of operation name strings as defined in ``contract.yaml``.
        """
        return [
            "contract_resolve",
            "health_check",
        ]


__all__ = ["RegistryInfraContractResolverBridge"]
