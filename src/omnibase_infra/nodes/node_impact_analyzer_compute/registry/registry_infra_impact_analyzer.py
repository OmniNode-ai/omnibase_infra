# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeImpactAnalyzerCompute dependencies.

Provides dependency injection configuration for the impact analyzer
compute node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_impact_analyzer_compute.node import (
        NodeImpactAnalyzerCompute,
    )


class RegistryInfraImpactAnalyzer:
    """Registry for NodeImpactAnalyzerCompute dependency injection.

    Provides factory methods for creating NodeImpactAnalyzerCompute
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraImpactAnalyzer(container)
        >>> compute = registry.create_compute()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_compute(self) -> NodeImpactAnalyzerCompute:
        """Create a NodeImpactAnalyzerCompute instance.

        Returns:
            Configured NodeImpactAnalyzerCompute instance.
        """
        from omnibase_infra.nodes.node_impact_analyzer_compute.node import (
            NodeImpactAnalyzerCompute,
        )

        return NodeImpactAnalyzerCompute(self._container)


__all__: list[str] = ["RegistryInfraImpactAnalyzer"]
