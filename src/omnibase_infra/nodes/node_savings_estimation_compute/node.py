# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Savings Estimation Compute -- token savings correlation + calculation.

This EFFECT node ingests raw savings signals (injection effectiveness,
validator catches) into Postgres, periodically correlates them per session
against llm_call_metrics / session_outcomes, and computes token and cost
savings using tiered model pricing. The result is a ModelSavingsEstimate
published to onex.evt.omnibase-infra.savings-estimated.v1.

Follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom logic -- all behavior from handlers
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerSavingsCorrelation: Ingest signals + periodic correlation batch
    - HandlerSavingsEstimation: Pure savings computation from effectiveness
      data, invoked in-process by HandlerSavingsCorrelation

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Savings estimation models
    - handlers/: Savings correlation + computation handlers

Tracking:
    - OMN-6964: Token savings emitter
    - OMN-16293: Wire real correlation; COMPUTE_GENERIC -> EFFECT_GENERIC
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSavingsEstimationCompute(NodeEffect):
    """EFFECT node for token savings correlation and estimation.

    Capabilities: savings.estimate, savings.correlation_batch_compute

    Ingests raw savings signals, correlates them per session, and computes
    dollar savings using tiered model pricing. All behavior is defined in
    contract.yaml and implemented through handlers. No custom logic
    exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the savings estimation effect node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeSavingsEstimationCompute"]
