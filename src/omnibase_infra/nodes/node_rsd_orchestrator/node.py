# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD orchestrator - declarative workflow coordinator.

Coordinates the RSD priority scoring workflow:
    1. Receive scoring command
    2. Dispatch data fetch effect
    3. Dispatch score compute
    4. Dispatch state store via reducer
    5. Emit completion event

All workflow logic is 100% driven by contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRsdOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for RSD priority scoring workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeRsdOrchestrator"]
