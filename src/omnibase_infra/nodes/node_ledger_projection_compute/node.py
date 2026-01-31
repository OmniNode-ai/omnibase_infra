# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeLedgerProjectionCompute - Declarative COMPUTE node for ledger projection.

This node extracts metadata from platform events for ledger persistence.
All business logic is delegated to HandlerLedgerProjection per ONEX
declarative node pattern.

Design Rationale:
    ONEX nodes are declarative shells driven by contract.yaml. The node class
    extends the appropriate archetype base class and contains no custom logic.
    All compute behavior is defined in handlers configured via handler_routing
    in the contract.

Subscribed Topics (via contract.yaml):
    - onex.evt.platform.node-registration.v1
    - onex.evt.platform.node-introspection.v1
    - onex.evt.platform.node-heartbeat.v1
    - onex.cmd.platform.request-introspection.v1
    - onex.evt.platform.fsm-state-transitions.v1
    - onex.intent.platform.runtime-tick.v1
    - onex.snapshot.platform.registration-snapshots.v1

Ticket: OMN-1648, OMN-1726
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


# ONEX_EXCLUDE: declarative_node - legacy compute node with projection logic (OMN-1725)
class NodeLedgerProjectionCompute(NodeCompute):
    """Declarative COMPUTE node for ledger projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerLedgerProjection. This node contains no custom logic.

    See Also:
        - handlers/handler_ledger_projection.py: Contains all compute logic
        - handlers/contract.yaml: Handler routing configuration
        - contract.yaml: Node subscription and I/O configuration
    """

    # Declarative node - all behavior defined in contract.yaml


__all__ = ["NodeLedgerProjectionCompute"]
