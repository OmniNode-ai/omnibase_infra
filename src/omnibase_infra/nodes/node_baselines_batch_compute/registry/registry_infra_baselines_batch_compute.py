# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeBaselinesBatchCompute dependencies.

Provides dependency injection configuration for the baselines batch compute
effect node, following the ONEX container-based DI pattern.

Mirrors node_baseline_comparison_compute/registry/registry_infra_baseline_comparison.py.

Ticket: OMN-3045
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_baselines_batch_compute.node import (
        NodeBaselinesBatchCompute,
    )


class RegistryInfraBaselinesBatchCompute:
    """Registry for NodeBaselinesBatchCompute dependency injection.

    Provides factory methods for creating NodeBaselinesBatchCompute
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraBaselinesBatchCompute(container)
        >>> effect = registry.create_effect()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_effect(self) -> NodeBaselinesBatchCompute:
        """Create a NodeBaselinesBatchCompute instance.

        Returns:
            Configured NodeBaselinesBatchCompute instance.
        """
        from omnibase_infra.nodes.node_baselines_batch_compute.node import (
            NodeBaselinesBatchCompute,
        )

        return NodeBaselinesBatchCompute(self._container)


__all__: list[str] = ["RegistryInfraBaselinesBatchCompute"]
