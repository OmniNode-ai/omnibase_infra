# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer-flow target boundary.

Plan task A1 moves the target protocols to omnibase_spi and models to
omnibase_core; they are node-local until those packages release.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
    ModelConsumerFlowObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)


@runtime_checkable
class ProtocolConsumerFlowTarget(Protocol):
    """A failed read returns read_ok=False with its error text, never raises."""

    async def observe(
        self, request: ModelConsumerFlowRequest
    ) -> ModelConsumerFlowObservation:
        """Collect the four C28 clauses against the requested lane."""
        ...
