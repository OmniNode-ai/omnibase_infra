# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeSavingsEstimationCompute dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_savings_estimation_compute.node import (
        NodeSavingsEstimationCompute,
    )


class RegistryInfraSavingsEstimation:
    """Registry for NodeSavingsEstimationCompute dependency injection."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    def create_compute(self) -> NodeSavingsEstimationCompute:
        from omnibase_infra.nodes.node_savings_estimation_compute.node import (
            NodeSavingsEstimationCompute,
        )

        return NodeSavingsEstimationCompute(self._container)


__all__: list[str] = ["RegistryInfraSavingsEstimation"]
