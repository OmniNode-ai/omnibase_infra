# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeValidationOrchestrator dependencies.

Provides dependency injection configuration for the validation orchestrator
node, following the ONEX container-based DI pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_validation_orchestrator.node import (
        NodeValidationOrchestrator,
    )


class RegistryInfraValidationOrchestrator:
    """Registry for NodeValidationOrchestrator dependency injection.

    Provides factory methods for creating NodeValidationOrchestrator
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraValidationOrchestrator(container)
        >>> orchestrator = registry.create_orchestrator()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_orchestrator(self) -> NodeValidationOrchestrator:
        """Create a NodeValidationOrchestrator instance.

        Returns:
            Configured NodeValidationOrchestrator instance.
        """
        from omnibase_infra.nodes.node_validation_orchestrator.node import (
            NodeValidationOrchestrator,
        )

        return NodeValidationOrchestrator(self._container)


__all__: list[str] = ["RegistryInfraValidationOrchestrator"]
