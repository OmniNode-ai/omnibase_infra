# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodePrStateWriteEffect - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_pr_state_write_effect.node import (
    NodePrStateWriteEffect,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraPrStateWrite:
    """DI registry for the pr_state write effect node."""

    @staticmethod
    def get_node_class() -> type[NodePrStateWriteEffect]:
        return NodePrStateWriteEffect

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodePrStateWriteEffect:
        return NodePrStateWriteEffect(container)


__all__ = [
    "NodePrStateWriteEffect",
    "RegistryInfraPrStateWrite",
]
