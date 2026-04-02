# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain verify reducer -- FSM state tracking for chain learning.

All behavior is defined in contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeChainVerifyReducer(NodeReducer):
    """Declarative reducer for chain verification FSM.

    All behavior is defined in contract.yaml -- no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeChainVerifyReducer"]
