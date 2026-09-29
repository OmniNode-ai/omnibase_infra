# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeBoardProbeEffect — declarative effect node.

Runs board checks against a lab or CI surface and grades each one PASS, FAIL
or INDETERMINATE. It reads the surface and never mutates it.

Handlers:
    - ``HandlerForwarderRefusedTopic``: forwarder request -> board probe result.

All behavior is declared in ``contract.yaml``. No custom logic here.

Ticket: OMN-19930
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBoardProbeEffect(NodeEffect):
    """Effect node that runs board checks.

    Capability: board_probe.forwarder_refused_topic
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the board probe effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeBoardProbeEffect"]
