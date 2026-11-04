# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative census oneshot effect node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeCensusOneshotEffect(NodeEffect):
    """Run the contract-bound completion sampling handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
