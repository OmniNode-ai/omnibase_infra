# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeSavingsEstimationCompute dependencies.

Provides dependency injection configuration for the savings estimation
compute node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_savings_estimation_compute.node import (
        NodeSavingsEstimationCompute,
    )


class RegistryInfraSavingsEstimation:
    """Registry for NodeSavingsEstimationCompute dependency injection.

    Provides factory methods for creating NodeSavingsEstimationCompute
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraSavingsEstimation(container)
        >>> compute = registry.create_compute()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_compute(self) -> NodeSavingsEstimationCompute:
        """Create a NodeSavingsEstimationCompute instance.

        Returns:
            Configured NodeSavingsEstimationCompute instance.
        """
        from omnibase_infra.nodes.node_savings_estimation_compute.node import (
            NodeSavingsEstimationCompute,
        )

        return NodeSavingsEstimationCompute(self._container)


__all__: list[str] = ["RegistryInfraSavingsEstimation"]
