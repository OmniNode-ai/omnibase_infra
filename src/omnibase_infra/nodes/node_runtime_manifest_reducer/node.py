# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runtime manifest reducer node (OMN-11197)."""

from __future__ import annotations

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.nodes import NodeReducer


class NodeRuntimeManifestReducer(NodeReducer):
    """Append-only reducer — persists runtime manifests to PostgreSQL."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
