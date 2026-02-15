# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeBaselineComparisonCompute dependencies.

Provides dependency injection configuration for the baseline comparison
compute node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_baseline_comparison_compute.node import (
        NodeBaselineComparisonCompute,
    )


class RegistryInfraBaselineComparison:
    """Registry for NodeBaselineComparisonCompute dependency injection.

    Provides factory methods for creating NodeBaselineComparisonCompute
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraBaselineComparison(container)
        >>> compute = registry.create_compute()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_compute(self) -> NodeBaselineComparisonCompute:
        """Create a NodeBaselineComparisonCompute instance.

        Returns:
            Configured NodeBaselineComparisonCompute instance.
        """
        from omnibase_infra.nodes.node_baseline_comparison_compute.node import (
            NodeBaselineComparisonCompute,
        )

        return NodeBaselineComparisonCompute(self._container)


__all__: list[str] = ["RegistryInfraBaselineComparison"]
