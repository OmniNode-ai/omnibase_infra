# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative COMPUTE node that plans a developer machine's broker principal.

All behavior is defined in contract.yaml and delegated to
HandlerDeveloperPrincipalPlan.

Ticket: OMN-20821
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeDeveloperPrincipalPlanCompute(NodeCompute):
    """Declarative compute node for developer principal planning.

    Handler:
        - ``HandlerDeveloperPrincipalPlan`` (``omnibase_infra.handlers``): the principal's name, create or
          rotate, its exact grants and the declaration digest, without I/O.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the developer principal plan compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeDeveloperPrincipalPlanCompute"]
