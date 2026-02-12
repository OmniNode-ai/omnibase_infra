# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeValidationExecutor dependencies.

Provides dependency injection configuration for the validation executor
effect node, following the ONEX container-based DI pattern.

Ticket: OMN-2147
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_validation_executor.node import (
        NodeValidationExecutor,
    )


class RegistryInfraValidationExecutor:
    """Registry for NodeValidationExecutor dependency injection.

    Provides factory methods for creating NodeValidationExecutor instances
    with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraValidationExecutor(container)
        >>> node = registry.create_executor()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_executor(self) -> NodeValidationExecutor:
        """Create a NodeValidationExecutor instance.

        Returns:
            Configured NodeValidationExecutor instance.
        """
        from omnibase_infra.nodes.node_validation_executor.node import (
            NodeValidationExecutor,
        )

        return NodeValidationExecutor(self._container)


__all__: list[str] = ["RegistryInfraValidationExecutor"]
