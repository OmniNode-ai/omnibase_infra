# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative EFFECT node for a dispatched delegation run's terminal.

All behavior is defined in contract.yaml and delegated to
HandlerDelegationDispatchTerminal.

Ticket: OMN-19972
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeDelegationDispatchTerminalEffect(NodeEffect):
    """Declarative effect node for a dispatched delegation run's terminal.

    Handler:
        - ``HandlerDelegationDispatchTerminal`` (``omnibase_infra.handlers``): the
          delegate-skill result and, for a waited run, its step envelopes.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the delegation dispatch terminal effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeDelegationDispatchTerminalEffect"]
