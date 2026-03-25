# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative REDUCER node for runtime node graph lifecycle [OMN-6349].

Pure-function reducer wrapping RuntimeHostProcess lifecycle management.
FSM: initializing -> wiring -> running -> draining -> stopped.
Wildcard fatal_error from any state -> stopped.

All state transition logic is driven by contract.yaml, not Python code.
The reducer is stateless -- state is passed in and returned.

Ticket: OMN-6349
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeNodeGraphReducer(NodeReducer):  # type: ignore[type-arg]
    """Declarative reducer node for runtime node graph lifecycle.

    FSM States: initializing -> wiring -> running -> draining -> stopped
    Wildcard: fatal_error transitions to stopped from any state.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the node graph reducer."""
        super().__init__(container)
