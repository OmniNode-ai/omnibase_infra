# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Declarative COMPUTE node for checkpoint validation.

Pure validation — verifies checkpoint data consistency without I/O.
All behavior is defined in contract.yaml and delegated to
HandlerCheckpointValidate.

Ticket: OMN-2143
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeCheckpointValidateCompute(NodeCompute):
    """Declarative compute node for checkpoint structural validation.

    Handler:
        - ``HandlerCheckpointValidate``: Pure validation of checkpoint data.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the checkpoint validate compute node."""
        super().__init__(container)
