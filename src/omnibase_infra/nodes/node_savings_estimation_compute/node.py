# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Savings Estimation Compute -- tiered attribution.

Declarative compute node for token savings estimation. All behavior
is defined in contract.yaml and implemented through handlers.

Tracking:
    - OMN-5547: Create HandlerSavingsEstimator compute handler
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSavingsEstimationCompute(NodeCompute):
    """Compute node for tiered token savings attribution.

    Capability: savings.estimate

    Takes session LLM call records, injection signals, validator catches,
    and baseline config, then computes savings across 5 categories with
    two tiers (direct/heuristic).
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeSavingsEstimationCompute"]
