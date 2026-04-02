# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event forward effect -- forwards platform events to HTTP backends.

All behavior is defined in contract.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeEventForwardEffect(NodeEffect):
    """Declarative effect node for event forwarding.

    All behavior is defined in contract.yaml -- no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeEventForwardEffect"]
