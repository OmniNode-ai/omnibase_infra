# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative EFFECT node that records each runtime-image container's command.

All behavior is defined in contract.yaml and delegated to
HandlerLaneEntryPointCensus.

Ticket: OMN-21288
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLaneEntryPointCensusEffect(NodeEffect):
    """Declarative effect node for the lane entry-point census.

    Handler:
        - ``HandlerLaneEntryPointCensus`` (``omnibase_infra.handlers``): reads container
          inspection and the declared policy, returns one census record per lane.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the lane entry-point census effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeLaneEntryPointCensusEffect"]
