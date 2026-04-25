# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative remote-agent invocation effect node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRemoteAgentInvokeEffect(NodeEffect):
    """Declarative effect node for remote-agent invocation.

    All behavior is defined in contract.yaml. Protocol-specific handlers are
    registered by later implementation tickets.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeRemoteAgentInvokeEffect"]
