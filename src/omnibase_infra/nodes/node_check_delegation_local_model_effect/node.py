# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for the delegation local-model doctor diagnosis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeCheckDelegationLocalModelEffect(NodeEffect):
    """Bind the no-local-model diagnosis to the canonical infra handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeCheckDelegationLocalModelEffect"]
