# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeBuildLoopWriteEffect - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_build_loop_write_effect.node import (
    NodeBuildLoopWriteEffect,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraBuildLoopWrite:
    """DI registry for the build_loop write effect node."""

    @staticmethod
    def get_node_class() -> type[NodeBuildLoopWriteEffect]:
        return NodeBuildLoopWriteEffect

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeBuildLoopWriteEffect:
        return NodeBuildLoopWriteEffect(container)


__all__ = [
    "NodeBuildLoopWriteEffect",
    "RegistryInfraBuildLoopWrite",
]
