# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative COMPUTE node for the fault-injection fixture.

All behavior is defined in contract.yaml and delegated to
HandlerFaultInjectFixture.

Ticket: OMN-16265
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeFaultInjectFixtureCompute(NodeCompute):
    """Declarative compute node for the permanent DLQ fault-injection fixture.

    Handler:
        - ``HandlerFaultInjectFixture``: deterministic, size-controlled
          result for driving primary-publish + DLQ-leg failures on demand.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the fault-injection fixture compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeFaultInjectFixtureCompute"]
