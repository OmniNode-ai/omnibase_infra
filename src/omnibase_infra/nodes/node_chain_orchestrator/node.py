# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain learning orchestrator -- declarative workflow coordinator.

Coordinates the prompt-chain learning workflow:
    1. Receive chain learn command
    2. Dispatch prompt embedding + Qdrant retrieval
    3. Evaluate hit/miss, dispatch replay or explore
    4. Verify chain against contract
    5. Store verified chain to Qdrant
    6. Emit completion event

All workflow logic is 100% driven by contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeChainOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for chain learning workflow.

    All behavior is defined in contract.yaml -- no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeChainOrchestrator"]
