# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeBuildLoopProjectionCompute - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_build_loop_projection_compute.node import (
    NodeBuildLoopProjectionCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraBuildLoopProjection:
    """DI registry for the build_loop projection compute node."""

    @staticmethod
    def get_node_class() -> type[NodeBuildLoopProjectionCompute]:
        return NodeBuildLoopProjectionCompute

    @staticmethod
    def create_node(
        container: ModelONEXContainer,
    ) -> NodeBuildLoopProjectionCompute:
        return NodeBuildLoopProjectionCompute(container)


__all__ = [
    "NodeBuildLoopProjectionCompute",
    "RegistryInfraBuildLoopProjection",
]
