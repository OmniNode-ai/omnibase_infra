# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Savings Estimation Compute -- token savings calculation.

This compute node takes injection effectiveness data and computes
token and cost savings using tiered model pricing. The result is
a ModelSavingsEstimate suitable for Kafka emission to
onex.evt.omnibase-infra.savings-estimated.v1.

Follows the ONEX declarative pattern:
    - DECLARATIVE compute driven by contract.yaml
    - Zero custom logic -- all behavior from handlers
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerSavingsEstimation: Compute savings from effectiveness data

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Savings estimation models
    - handlers/: Savings computation handler

Tracking:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSavingsEstimationCompute(NodeCompute):
    """Compute node for token savings estimation.

    Capability: savings.estimate

    Takes injection effectiveness data and computes dollar savings
    using tiered model pricing. All behavior is defined in
    contract.yaml and implemented through handlers. No custom logic
    exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the savings estimation compute node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeSavingsEstimationCompute"]
