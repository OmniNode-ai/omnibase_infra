# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""NodeReleaseIdentityCompute - Declarative COMPUTE node for the fitness gate.

Pure COMPUTE node that evaluates the release-identity invariant: packaged source
may not merge onto an already-published version without a bump. Receives
pre-collected inputs (pyproject version, published tags, changed files) and returns
an exit-code/message decision. No I/O — all logic is delegated to
HandlerReleaseIdentity per the ONEX declarative node pattern.

Design Rationale:
    ONEX nodes are declarative shells driven by contract.yaml. The node class
    extends the archetype base class and contains no custom logic; all compute
    behavior is defined in handlers configured via handler_routing in the contract.

Ticket: OMN-14471
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeReleaseIdentityCompute(NodeCompute):
    """Declarative COMPUTE node for the release-identity fitness decision.

    All behavior is defined in contract.yaml and delegated to
    HandlerReleaseIdentity. This node contains no custom logic.

    See Also:
        - handlers/handler_release_identity.py: the pure version-ahead gate logic
        - contract.yaml: node I/O and handler routing configuration
    """

    # Declarative node - all behavior defined in contract.yaml


__all__: list[str] = ["NodeReleaseIdentityCompute"]
