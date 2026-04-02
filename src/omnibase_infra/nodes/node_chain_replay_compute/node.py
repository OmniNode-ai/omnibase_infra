# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain replay compute -- adapts cached chain to new context.

All behavior is defined in contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeChainReplayCompute(NodeCompute):
    """Declarative compute node for chain replay.

    All behavior is defined in contract.yaml -- no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeChainReplayCompute"]
