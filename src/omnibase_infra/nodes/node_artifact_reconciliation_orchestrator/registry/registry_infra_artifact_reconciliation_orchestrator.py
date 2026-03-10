# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeArtifactReconciliationOrchestrator dependencies.

Provides dependency injection configuration for the artifact reconciliation
orchestrator node, following the ONEX container-based DI pattern.

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.node import (
        NodeArtifactReconciliationOrchestrator,
    )


class RegistryInfraArtifactReconciliationOrchestrator:
    """Registry for NodeArtifactReconciliationOrchestrator dependency injection.

    Provides factory methods for creating NodeArtifactReconciliationOrchestrator
    instances with properly configured dependencies from the ONEX container.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> registry = RegistryInfraArtifactReconciliationOrchestrator(container)
        >>> orchestrator = registry.create_orchestrator()
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the registry with ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container

    def create_orchestrator(self) -> NodeArtifactReconciliationOrchestrator:
        """Create a NodeArtifactReconciliationOrchestrator instance.

        Returns:
            Configured NodeArtifactReconciliationOrchestrator instance.
        """
        from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.node import (
            NodeArtifactReconciliationOrchestrator,
        )

        return NodeArtifactReconciliationOrchestrator(self._container)


__all__: list[str] = ["RegistryInfraArtifactReconciliationOrchestrator"]
