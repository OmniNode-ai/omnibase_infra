# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeLabProofRunEffect — declarative effect node.

Executes one rendered lab proof plan on the host it runs on: clones at exact
commits, builds, boots an isolated compose project, reads back, tears down and
proves zero residue. Every step is an argv from the plan; nothing is decided
here.

Handlers:
    - ``HandlerLabProofRun``: plan -> run report.

All behavior is declared in ``contract.yaml``. No custom logic here.

Ticket: OMN-19572
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLabProofRunEffect(NodeEffect):
    """Effect node that runs a lab proof plan on a lab host.

    Capability: lab_proof.run
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the lab proof run effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeLabProofRunEffect"]
