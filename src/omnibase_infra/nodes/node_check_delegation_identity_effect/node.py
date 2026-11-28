# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for delegation identity doctor diagnosis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeCheckDelegationIdentityEffect(NodeEffect):
    """Bind the identity diagnosis to the canonical infra handler."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeCheckDelegationIdentityEffect"]
