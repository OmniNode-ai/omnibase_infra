# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope workflow orchestrator - declarative workflow coordinator.

Coordinates the scope-check workflow:
    1. Receive scope-check command
    2. Dispatch file read effect
    3. Dispatch scope extraction compute
    4. Dispatch manifest write effect
    5. Emit completion event

All workflow logic is 100% driven by contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeScopeWorkflowOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for scope-check workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeScopeWorkflowOrchestrator"]
