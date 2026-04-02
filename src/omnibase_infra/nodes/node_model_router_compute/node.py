# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model router compute - pure scoring of candidate models."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeModelRouterCompute(NodeCompute):
    """Declarative compute node for model routing scoring.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeModelRouterCompute"]
