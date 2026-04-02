# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD score compute - pure transformation node for priority scoring."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeRsdScoreCompute(NodeCompute):
    """Declarative compute node for RSD 5-factor priority scoring.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRsdScoreCompute"]
