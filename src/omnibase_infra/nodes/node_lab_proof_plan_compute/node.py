# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeLabProofPlanCompute — declarative compute node.

Renders a lab proof run from a registry profile row (``lab_proof.plan``) and
decides a run's outcome from its observations (``lab_proof.verdict``). Both
are pure; the host work between them is ``node_lab_proof_run_effect``.

Handlers:
    - ``HandlerLabProofPlan``: profile row + subject -> ordered argv steps.
    - ``HandlerLabProofVerdict``: plan + run report -> outcome and checks.

All behavior is declared in ``contract.yaml``. No custom logic here.

Ticket: OMN-19572
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLabProofPlanCompute(NodeCompute):
    """Compute node for lab proof planning and verdicts.

    Capabilities: lab_proof.plan, lab_proof.verdict
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the lab proof plan compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeLabProofPlanCompute"]
