# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative reducer shell for cost-by-repo snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeProjectionCostByRepo(NodeReducer):
    """Reducer node for ``cost.by_repo.v1`` snapshot emission."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeProjectionCostByRepo"]
