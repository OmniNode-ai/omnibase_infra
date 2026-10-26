# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic lease projection writer (OMN-21424).

All behavior is defined in contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeProjectionLease(NodeEffect):
    """Declarative node; all behavior is defined in contract.yaml."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeProjectionLease"]
