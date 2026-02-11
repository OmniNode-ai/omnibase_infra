# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeAuthGateCompute - Declarative COMPUTE node for authorization decisions.

Pure COMPUTE node that evaluates a 10-step authorization cascade.
Receives auth state + tool request, returns allow/deny/soft_deny.
No I/O — all business logic is delegated to HandlerAuthGate per ONEX
declarative node pattern.

Design Rationale:
    ONEX nodes are declarative shells driven by contract.yaml. The node class
    extends the appropriate archetype base class and contains no custom logic.
    All compute behavior is defined in handlers configured via handler_routing
    in the contract.

Ticket: OMN-2125
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeAuthGateCompute(NodeCompute):
    """Declarative COMPUTE node for authorization decisions.

    All behavior is defined in contract.yaml and delegated to
    HandlerAuthGate. This node contains no custom logic.

    See Also:
        - handlers/handler_auth_gate.py: Contains the 10-step cascade logic
        - contract.yaml: Node I/O and handler routing configuration
    """

    # Declarative node - all behavior defined in contract.yaml


__all__: list[str] = ["NodeAuthGateCompute"]
