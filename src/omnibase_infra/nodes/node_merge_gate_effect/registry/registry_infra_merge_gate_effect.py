# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry for NodeMergeGateEffect infrastructure dependencies.

Provides factory methods for creating NodeMergeGateEffect instances
with dependencies resolved from the container.

Following ONEX naming conventions:
    - File: registry_infra_<node_name>.py
    - Class: RegistryInfra<NodeName>

Related:
    - contract.yaml: Node contract defining operations and dependencies
    - node.py: Declarative node implementation
    - handlers/: Merge gate upsert + Linear quarantine handler
    - OMN-3140: NodeMergeGateEffect implementation

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_merge_gate_effect.node import (
        NodeMergeGateEffect,
    )


class RegistryInfraMergeGateEffect:
    """Infrastructure registry for NodeMergeGateEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeMergeGateEffect instances.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_merge_gate_effect.registry import (
        ...     RegistryInfraMergeGateEffect,
        ... )
        >>>
        >>> container = ModelONEXContainer()
        >>> effect = RegistryInfraMergeGateEffect.create(container)

    .. versionadded:: 0.8.0
    """

    @staticmethod
    def create(container: ModelONEXContainer) -> NodeMergeGateEffect:
        """Create a NodeMergeGateEffect instance with resolved dependencies.

        Args:
            container: ONEX dependency injection container. Must have the
                following protocols registered:
                - ProtocolPostgresAdapter: PostgreSQL database operations

        Returns:
            Configured NodeMergeGateEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.

        .. versionadded:: 0.8.0
        """
        from omnibase_infra.nodes.node_merge_gate_effect.node import (
            NodeMergeGateEffect,
        )

        return NodeMergeGateEffect(container)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        .. deprecated:: 0.8.0
            Use contract.yaml dependencies field instead.

        Returns:
            List of protocol class names required for node operation.

        .. versionadded:: 0.8.0
        """
        warnings.warn(
            "get_required_protocols() is deprecated. Use contract.yaml dependencies "
            "field instead. The contract is the single source of truth for protocol "
            "requirements (OMN-1732).",
            DeprecationWarning,
            stacklevel=2,
        )
        return ["ProtocolPostgresAdapter"]

    @staticmethod
    def get_node_type() -> str:
        """Get the node type classification.

        Returns:
            Node type string ("EFFECT").

        .. versionadded:: 0.8.0
        """
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name.

        Returns:
            The node name as defined in contract.yaml.

        .. versionadded:: 0.8.0
        """
        return "node_merge_gate_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node.

        Returns:
            List of capability identifiers.

        .. versionadded:: 0.8.0
        """
        return [
            "merge_gate_persistence",
            "idempotent_upsert",
            "quarantine_ticket_creation",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers as defined in contract.yaml.

        .. versionadded:: 0.8.0
        """
        return ["upsert_merge_gate"]

    @staticmethod
    def get_supported_intent_types() -> list[str]:
        """Get list of intent types routed by this node.

        Returns:
            List of intent type strings.

        .. versionadded:: 0.8.0
        """
        return ["merge_gate.upsert"]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend identifiers.

        .. versionadded:: 0.8.0
        """
        return ["postgres", "linear"]


__all__ = ["RegistryInfraMergeGateEffect"]
