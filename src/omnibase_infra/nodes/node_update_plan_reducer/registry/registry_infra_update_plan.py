# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeUpdatePlanReducer dependencies.

Provides dependency injection configuration for the update plan reducer
node, following the ONEX container-based DI pattern.

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_update_plan_reducer.node import (
        NodeUpdatePlanReducer,
    )


class RegistryInfraUpdatePlan:
    """Registry for NodeUpdatePlanReducer dependency injection.

    Provides factory methods for creating NodeUpdatePlanReducer instances
    with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraUpdatePlan(container)
        >>> reducer = registry.create_reducer()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_reducer(self) -> NodeUpdatePlanReducer:
        """Create a NodeUpdatePlanReducer instance.

        Returns:
            Configured NodeUpdatePlanReducer instance.
        """
        from omnibase_infra.nodes.node_update_plan_reducer.node import (
            NodeUpdatePlanReducer,
        )

        return NodeUpdatePlanReducer(self._container)


__all__: list[str] = ["RegistryInfraUpdatePlan"]
