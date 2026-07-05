# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet-maintain workflow orchestrator - declarative workflow coordinator (OMN-13942).

Coordinates the Increment 1 (read-only detection) workflow:
    1. Receive maintain-start command
    2. Dispatch snapshot gather (EFFECT)
    3. Dispatch health classification (COMPUTE)
    4. Emit completion event with the health verdict -- NO mutation

All workflow logic is 100% driven by contract.yaml. No I/O happens in this
orchestrator or any of its handlers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRunnerFleetMaintainOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for the runner-fleet-maintain workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeRunnerFleetMaintainOrchestrator"]
