# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeBaselinesBatchCompute — EFFECT node for 3-phase baselines batch computation.

Follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom logic — all behavior from HandlerBaselinesBatchCompute
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerBaselinesBatchCompute: Run 3-phase baselines batch computation

Design Decisions:
    - EFFECT node: performs external I/O (PostgreSQL reads/writes, Kafka publish)
    - correlation_id required on all commands (D1)
    - Emit snapshot only on total_rows > 0 (D5)

Related:
    - contract.yaml: Capability definitions and subscribed/published topics
    - handlers/: HandlerBaselinesBatchCompute (lifted from ServiceBatchComputeBaselines)
    - models/: Command and output models

Tracking:
    - OMN-3039: NodeBaselinesBatchCompute EFFECT node + validation topic fix
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBaselinesBatchCompute(NodeEffect):
    """EFFECT node for 3-phase baselines batch computation.

    Runs comparisons, trend, and breakdown phases against the observability
    database and emits a baselines-computed snapshot event for omnidash.

    All behavior is defined in contract.yaml and implemented through
    HandlerBaselinesBatchCompute. No custom logic exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the baselines batch compute effect node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeBaselinesBatchCompute"]
