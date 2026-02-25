# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeDecisionStoreQueryCompute dependency injection.

Provides factory methods for creating NodeDecisionStoreQueryCompute instances
with properly configured dependencies from the ONEX container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_decision_store_query_compute.node import (
        NodeDecisionStoreQueryCompute,
    )


class RegistryInfraDecisionStoreQuery:
    """Registry for NodeDecisionStoreQueryCompute dependency injection.

    Provides factory methods for creating NodeDecisionStoreQueryCompute
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraDecisionStoreQuery(container)
        >>> compute = registry.create_compute()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with the ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_compute(self) -> NodeDecisionStoreQueryCompute:
        """Create a NodeDecisionStoreQueryCompute instance.

        Returns:
            Configured NodeDecisionStoreQueryCompute instance.
        """
        from omnibase_infra.nodes.node_decision_store_query_compute.node import (
            NodeDecisionStoreQueryCompute,
        )

        return NodeDecisionStoreQueryCompute(self._container)


__all__: list[str] = ["RegistryInfraDecisionStoreQuery"]
