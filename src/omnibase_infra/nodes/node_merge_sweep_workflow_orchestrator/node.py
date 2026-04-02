# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merge-sweep workflow orchestrator - declarative workflow coordinator.

Coordinates the merge-sweep workflow:
    1. Receive merge-sweep command
    2. Dispatch PR list scan across repos (EFFECT)
    3. Dispatch PR classification (COMPUTE)
    4. Dispatch auto-merge on Track A PRs (EFFECT)
    5. Emit completion event with summary

All workflow logic is 100% driven by contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeMergeSweepWorkflowOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for merge-sweep workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeMergeSweepWorkflowOrchestrator"]
