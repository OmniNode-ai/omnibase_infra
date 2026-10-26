# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic lease orchestrator (OMN-21424).

All behavior is defined in contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeLeaseOrchestrator(NodeOrchestrator):
    """Declarative node; all behavior is defined in contract.yaml."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeLeaseOrchestrator"]
