# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Baseline Comparison Compute -- A/B delta computation.

This compute node takes paired baseline and candidate run results and
computes cost and outcome deltas to produce a ModelAttributionRecord
that proves (or disproves) pattern ROI for promotion decisions.

Follows the ONEX declarative pattern:
    - DECLARATIVE compute driven by contract.yaml
    - Zero custom logic -- all behavior from handlers
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerBaselineComparison: Compute deltas and ROI

Design Decisions:
    - Pure compute: no external I/O, deterministic from inputs
    - Cost deltas computed as baseline - candidate (positive = savings)
    - ROI = non-negative token delta + candidate passed + no quality regression

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Comparison input model
    - handlers/: Baseline comparison handler

Tracking:
    - OMN-2155: Baselines + ROI -- A/B Run Infrastructure
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBaselineComparisonCompute(NodeCompute):
    """Compute node for A/B baseline comparison.

    Capability: baseline.compare

    Takes paired baseline and candidate run results and computes
    cost/outcome deltas plus ROI determination.  All behavior is
    defined in contract.yaml and implemented through handlers.
    No custom logic exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the baseline comparison compute node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeBaselineComparisonCompute"]
