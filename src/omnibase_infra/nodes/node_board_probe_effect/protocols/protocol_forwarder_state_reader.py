# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reader the ``forwarder_refused_topic`` check reads through.

Bound per surface: the lab binds a docker reader on the lab host that runs the
forwarder. Plan task A1 moves board-probe target protocols to omnibase_spi.

Ticket: OMN-19930
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.nodes.node_board_probe_effect.models.model_forwarder_refused_topic_request import (
    ModelForwarderRefusedTopicRequest,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_forwarder_state_observation import (
    ModelForwarderStateObservation,
)


@runtime_checkable
class ProtocolForwarderStateReader(Protocol):
    """Reads one forwarder's state. A failed read returns ``read_ok=False``."""

    async def observe(
        self, request: ModelForwarderRefusedTopicRequest
    ) -> ModelForwarderStateObservation:
        """Observe the forwarder named by ``request``."""
        ...


__all__ = ["ProtocolForwarderStateReader"]
