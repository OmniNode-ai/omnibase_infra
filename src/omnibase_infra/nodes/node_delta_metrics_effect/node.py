# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeDeltaMetricsEffect -- declarative effect node for metrics rollup.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic -- all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Related:
    - contract.yaml: Handler routing and I/O model definitions
    - handlers/: PostgreSQL operation handler
    - OMN-3142: NodeDeltaMetricsEffect implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.models.runtime.model_resolved_dependencies import (
        ModelResolvedDependencies,
    )


# ONEX_EXCLUDE: declarative_node -- OMN-1732 DEC-003 requires constructor injection
class NodeDeltaMetricsEffect(NodeEffect):
    """Declarative effect node for delta metrics rollup.

    Routes bundle completion signals through the metrics upsert handler
    which aggregates counters by (coding_model, subsystem, period).

    NO custom routing logic -- all behavior is
    defined in contract.yaml.

    Args:
        container: ONEX dependency injection container.
        dependencies: Optional pre-resolved protocol dependencies.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        dependencies: ModelResolvedDependencies | None = None,
    ) -> None:
        """Initialise effect node with container dependency injection.

        Args:
            container: ONEX dependency injection container.
            dependencies: Optional pre-resolved protocol dependencies from
                ContractDependencyResolver. Part of OMN-1732.
        """
        super().__init__(container)
        self._resolved_dependencies = dependencies


__all__ = ["NodeDeltaMetricsEffect"]
