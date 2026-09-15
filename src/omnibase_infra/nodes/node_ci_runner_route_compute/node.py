# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeCIRunnerRouteCompute — declarative compute node.

Chooses where one CI run's jobs execute: the lab runner fleet when it has
headroom, GitHub-hosted when it does not, and never GitHub-hosted for a private
repository.

Handlers:
    - ``HandlerCIRunnerRoute``: the ordered elimination and its two guards.

All behavior is declared in ``contract.yaml``, including the thresholds, the
label sets and the private-repository refusal. No custom logic here.

Ticket: OMN-18412
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeCIRunnerRouteCompute(NodeCompute):
    """Compute node for per-run CI runner placement.

    Capability: ci.runner.route
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the CI runner routing compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeCIRunnerRouteCompute"]
