# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeDecisionStoreEffect infrastructure dependencies.

Provides factory methods for creating NodeDecisionStoreEffect instances
with dependencies resolved from the container.

Following ONEX naming conventions:
    - File: registry_infra_<node_name>.py
    - Class: RegistryInfra<NodeName>

Related:
    - contract.yaml: Node contract defining operations and dependencies
    - node.py: Declarative node implementation
    - handlers/: PostgreSQL operation handlers
    - OMN-2765: NodeDecisionStoreEffect implementation

.. versionadded:: 0.7.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.models.runtime.model_resolved_dependencies import (
        ModelResolvedDependencies,
    )
    from omnibase_infra.nodes.node_decision_store_effect.node import (
        NodeDecisionStoreEffect,
    )


class RegistryInfraDecisionStoreEffect:
    """Infrastructure registry for NodeDecisionStoreEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeDecisionStoreEffect instances.

    Example:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> from omnibase_infra.nodes.node_decision_store_effect.registry import (
        ...     RegistryInfraDecisionStoreEffect,
        ... )
        >>>
        >>> container = ModelONEXContainer()
        >>> effect = RegistryInfraDecisionStoreEffect.create(container)

    .. versionadded:: 0.7.0
    """

    @staticmethod
    def create(
        container: ModelONEXContainer,
        dependencies: ModelResolvedDependencies | None = None,
    ) -> NodeDecisionStoreEffect:
        """Create a NodeDecisionStoreEffect instance with resolved dependencies.

        Args:
            container: ONEX dependency injection container. Must have the
                following protocols registered:
                - ProtocolPostgresAdapter: PostgreSQL database operations
                - ProtocolCircuitBreakerAware: Backend circuit breaker protection
            dependencies: Optional pre-resolved protocol dependencies from
                ContractDependencyResolver. Part of OMN-1732 runtime DI.

        Returns:
            Configured NodeDecisionStoreEffect instance ready for operation.

        Raises:
            OnexError: If required protocols are not registered in container.

        .. versionadded:: 0.7.0
        """
        from omnibase_infra.nodes.node_decision_store_effect.node import (
            NodeDecisionStoreEffect,
        )

        return NodeDecisionStoreEffect(container, dependencies=dependencies)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        .. deprecated:: 0.7.0
            Use contract.yaml dependencies field instead.

        Returns:
            List of protocol class names required for node operation.

        .. versionadded:: 0.7.0
        """
        warnings.warn(
            "get_required_protocols() is deprecated. Use contract.yaml dependencies "
            "field instead. The contract is the single source of truth for protocol "
            "requirements (OMN-1732).",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            "ProtocolPostgresAdapter",
            "ProtocolCircuitBreakerAware",
        ]

    @staticmethod
    def get_node_type() -> str:
        """Get the node type classification.

        Returns:
            Node type string ("EFFECT").

        .. versionadded:: 0.7.0
        """
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name.

        Returns:
            The node name as defined in contract.yaml.

        .. versionadded:: 0.7.0
        """
        return "node_decision_store_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node.

        Returns:
            List of capability identifiers.

        .. versionadded:: 0.7.0
        """
        return [
            "decision_persistence",
            "structural_conflict_detection",
            "two_stage_write",
            "idempotent_conflict_insert",
            "active_invariant_enforcement",
            "circuit_breaker_protection",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers as defined in contract.yaml.

        .. versionadded:: 0.7.0
        """
        return [
            "write_decision",
            "write_conflict",
        ]

    @staticmethod
    def get_supported_intent_types() -> list[str]:
        """Get list of intent types routed by this node.

        Returns:
            List of intent type strings.

        .. versionadded:: 0.7.0
        """
        return [
            "decision_store.write_decision",
            "decision_store.write_conflict",
        ]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend identifiers.

        .. versionadded:: 0.7.0
        """
        return ["postgres"]


__all__ = ["RegistryInfraDecisionStoreEffect"]
