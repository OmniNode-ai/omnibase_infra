# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodePrStateProjectionCompute - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_pr_state_projection_compute.node import (
    NodePrStateProjectionCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraPrStateProjection:
    """DI registry for the pr_state projection compute node."""

    @staticmethod
    def get_node_class() -> type[NodePrStateProjectionCompute]:
        return NodePrStateProjectionCompute

    @staticmethod
    def create_node(
        container: ModelONEXContainer,
    ) -> NodePrStateProjectionCompute:
        return NodePrStateProjectionCompute(container)


__all__ = [
    "NodePrStateProjectionCompute",
    "RegistryInfraPrStateProjection",
]
