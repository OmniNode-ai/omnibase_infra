# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bus forwarder effect node.

The long-running process owns publishing on both bus legs. Handlers under this
node only validate and return transformed envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeBusForwarderEffect(NodeEffect):
    """Contract-driven effect node for tenant gateway bus forwarding."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeBusForwarderEffect"]
