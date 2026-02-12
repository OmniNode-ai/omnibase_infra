# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative EFFECT node for RRH result storage.

Writes RRH result artifacts as JSON and manages symlinks
(latest_by_ticket/, latest_by_repo/).

All behavior is defined in contract.yaml and delegated to
``HandlerRRHStorageWrite``.  This node contains no custom logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRRHStorageEffect(NodeEffect):
    """Declarative effect node for RRH artifact storage.

    Writes JSON result files and manages convenience symlinks for
    quick access by ticket and repository.

    All behavior is defined in contract.yaml — no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
