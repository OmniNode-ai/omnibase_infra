# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Entry method placement — declarative compute node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeEntryMethodPlacementCompute(NodeCompute):
    """Declarative compute node for the pure entry-method placement decision.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeEntryMethodPlacementCompute"]
