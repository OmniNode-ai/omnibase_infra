# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeDeltaMetricsEffect infrastructure dependencies.

Provides factory methods for creating NodeDeltaMetricsEffect instances
with dependencies resolved from the container.

Related:
    - contract.yaml: Node contract defining operations and dependencies
    - node.py: Declarative node implementation
    - handlers/: PostgreSQL operation handler
    - OMN-3142: NodeDeltaMetricsEffect implementation

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.models.runtime.model_resolved_dependencies import (
        ModelResolvedDependencies,
    )
    from omnibase_infra.nodes.node_delta_metrics_effect.node import (
        NodeDeltaMetricsEffect,
    )


class RegistryInfraDeltaMetricsEffect:
    """Infrastructure registry for NodeDeltaMetricsEffect.

    Provides dependency resolution and factory methods for creating
    properly configured NodeDeltaMetricsEffect instances.

    .. versionadded:: 0.8.0
    """

    @staticmethod
    def create(
        container: ModelONEXContainer,
        dependencies: ModelResolvedDependencies | None = None,
    ) -> NodeDeltaMetricsEffect:
        """Create a NodeDeltaMetricsEffect instance with resolved dependencies.

        Args:
            container: ONEX dependency injection container.
            dependencies: Optional pre-resolved protocol dependencies from
                ContractDependencyResolver. Part of OMN-1732 runtime DI.

        Returns:
            Configured NodeDeltaMetricsEffect instance ready for operation.

        .. versionadded:: 0.8.0
        """
        from omnibase_infra.nodes.node_delta_metrics_effect.node import (
            NodeDeltaMetricsEffect,
        )

        return NodeDeltaMetricsEffect(container, dependencies=dependencies)

    @staticmethod
    def get_required_protocols() -> list[str]:
        """Get list of protocols required by this node.

        .. deprecated:: 0.8.0
            Use contract.yaml dependencies field instead.

        .. versionadded:: 0.8.0
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
        """Get the node type classification."""
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name."""
        return "node_delta_metrics_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node."""
        return [
            "metrics_rollup",
            "idempotent_upsert",
            "period_based_rollups",
            "circuit_breaker_protection",
        ]

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node."""
        return ["upsert_metrics"]

    @staticmethod
    def get_supported_intent_types() -> list[str]:
        """Get list of intent types routed by this node."""
        return ["delta_metrics.upsert_metrics"]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with."""
        return ["postgres"]


__all__ = ["RegistryInfraDeltaMetricsEffect"]
